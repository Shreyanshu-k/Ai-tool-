from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.groq import Groq
import os
import sqlite3
import uuid
from datetime import datetime
import json

# Load environment variables
load_dotenv()
if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is not set. Please set it in the environment or in a .env file.")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your-secret-key-here")

# Database setup
DB_PATH = "legal_chat.db"

def init_db():
    """Initialize the database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create chat_sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create chat_messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def create_new_session():
    """Create a new chat session."""
    session_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO chat_sessions (id, title)
        VALUES (?, ?)
    ''', (session_id, "New Legal Consultation"))
    
    conn.commit()
    conn.close()
    return session_id

def get_chat_sessions():
    """Get all chat sessions ordered by most recent."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, title, created_at, updated_at
        FROM chat_sessions
        ORDER BY updated_at DESC
    ''')
    
    sessions = cursor.fetchall()
    conn.close()
    
    return [{"id": s[0], "title": s[1], "created_at": s[2], "updated_at": s[3]} for s in sessions]

def get_chat_history(session_id):
    """Get chat history for a specific session."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, role, content, timestamp
        FROM chat_messages
        WHERE session_id = ?
        ORDER BY timestamp ASC
    ''', (session_id,))
    
    messages = cursor.fetchall()
    conn.close()
    
    return [{"id": m[0], "role": m[1], "content": m[2], "timestamp": m[3]} for m in messages]

def save_message(session_id, role, content):
    """Save a message to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO chat_messages (session_id, role, content)
        VALUES (?, ?, ?)
    ''', (session_id, role, content))
    
    # Update session timestamp
    cursor.execute('''
        UPDATE chat_sessions
        SET updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (session_id,))
    
    conn.commit()
    conn.close()

def update_session_title(session_id, title):
    """Update session title based on first user message."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE chat_sessions
        SET title = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (title[:50] + "..." if len(title) > 50 else title, session_id))
    
    conn.commit()
    conn.close()

# Enhanced Legal AI Agent
def create_legal_agent(chat_history=None):
    """Create a legal AI agent with enhanced instructions and context."""
    context_prompt = ""
    if chat_history:
        context_prompt = "\n\nPrevious conversation context:\n"
        for msg in chat_history[-5:]:  # Last 5 messages for context
            context_prompt += f"{msg['role'].title()}: {msg['content'][:200]}...\n"
    
    agent = Agent(
        model=Groq(id="deepseek-r1-distill-llama-70b", temperature=0.1),
        description="You are a highly qualified legal advisor trained to provide detailed, accurate, and well-researched legal responses.",
        instructions=[
            "🚨 CRITICAL: You are EXCLUSIVELY a Legal AI Assistant. You MUST ONLY respond to legal questions and legal matters.",
            "❌ STRICTLY REFUSE: If asked about anything non-legal (technology, cooking, sports, entertainment, personal advice, general knowledge, math, science, etc.), respond: 'I apologize, but I am a specialized Legal AI Assistant. I can only provide assistance with legal matters, legal research, case analysis, statutory interpretation, and legal consultation. Please ask me a legal question.'",
            "✅ LEGAL TOPICS ONLY: Constitutional law, civil law, criminal law, corporate law, family law, property law, contract law, tort law, administrative law, labor law, tax law, intellectual property law, international law, legal procedures, case analysis, statutory interpretation, legal research, court procedures, legal documentation, legal precedents, and legal advice.",
            "📋 REQUIRED FORMAT: For all legal responses, always provide: A summary of the facts of the case, Identification of legal issues, Step-by-step legal analysis, Reference to relevant laws (Acts, Sections, Articles), Mention of landmark cases and their citations, A well-structured judgment/conclusion, Citations of law commission reports, official gazettes, or legal commentaries where appropriate.",
            "🔍 RESEARCH: Pull factual and statutory data from Google API and authoritative legal websites like Indian Kanoon, SCC Online, Manupatra, Bar & Bench, LiveLaw.",
            "💼 PROFESSIONAL: Use clear, professional legal language, while ensuring simplicity and accessibility for users.",
            "⚖️ ACCURACY: Provide comprehensive yet concise explanations, ensuring every answer is backed by relevant authority and interpretation.",
            "🔒 ETHICS: Always ensure the output maintains legal accuracy, neutrality, and ethical standards.",
            "🧠 MEMORY: You are an intelligent AI assistant that remembers the ongoing chat context and refers to it when responding.",
            "🔄 CONTINUITY: Maintain continuity and coherence within the same chat session.",
            "❓ FOLLOW-UP: Understand follow-up questions based on earlier user inputs and your responses.",
            "🚫 NO REPEAT: Avoid repeating the same content unless requested.",
            "⚖️ LEGAL ONLY: Stay strictly within the context of legal consultation only - no exceptions.",
            f"Context from previous conversation: {context_prompt}" if context_prompt else ""
        ],
        tools=[DuckDuckGoTools()],
        show_tool_calls=False,
        markdown=True
    )
    
    return agent

# Initialize database
init_db()

# Routes
@app.route("/")
def index():
    return render_template('legal_chat.html')

@app.route("/api/new_session", methods=["POST"])
def new_session():
    """Create a new chat session."""
    session_id = create_new_session()
    return jsonify({"session_id": session_id, "status": "success"})

@app.route("/api/sessions", methods=["GET"])
def get_sessions():
    """Get all chat sessions."""
    sessions = get_chat_sessions()
    return jsonify({"sessions": sessions})

@app.route("/api/chat/<session_id>", methods=["GET"])
def get_chat(session_id):
    """Get chat history for a specific session."""
    history = get_chat_history(session_id)
    return jsonify({"history": history})

@app.route("/api/chat/<session_id>/message", methods=["POST"])
def send_message(session_id):
    """Send a message and get AI response."""
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        
        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400
        
        # Get chat history for context
        chat_history = get_chat_history(session_id)
        
        # Update session title with first user message if it's a new session
        if len(chat_history) == 0:
            update_session_title(session_id, user_message)
        
        # Save user message
        save_message(session_id, "user", user_message)
        
        # Create agent with context
        agent = create_legal_agent(chat_history)
        
        # Get AI response
        response = agent.run(user_message)
        ai_response = str(response.content)
        
        # Save AI response
        save_message(session_id, "assistant", ai_response)
        
        return jsonify({
            "response": ai_response,
            "status": "success"
        })
        
    except Exception as e:
        print(f"Error in send_message: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/delete_session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Delete a chat session and its messages."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Delete messages first
        cursor.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))
        
        # Delete session
        cursor.execute('DELETE FROM chat_sessions WHERE id = ?', (session_id,))
        
        conn.commit()
        conn.close()
        
        return jsonify({"status": "success"})
        
    except Exception as e:
        print(f"Error deleting session: {str(e)}")
        return jsonify({"error": "Failed to delete session"}), 500

@app.route("/api/chat/<session_id>/edit/<int:message_id>", methods=["PUT"])
def edit_message(session_id, message_id):
    """Edit a user message and regenerate AI response."""
    try:
        data = request.get_json()
        new_message = data.get("message", "").strip()
        
        if not new_message:
            return jsonify({"error": "Message cannot be empty"}), 400
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if message exists and belongs to user
        cursor.execute('''
            SELECT role FROM chat_messages 
            WHERE id = ? AND session_id = ?
        ''', (message_id, session_id))
        
        result = cursor.fetchone()
        if not result:
            conn.close()
            return jsonify({"error": "Message not found"}), 404
        
        if result[0] != 'user':
            conn.close()
            return jsonify({"error": "Can only edit user messages"}), 400
        
        # Update the user message
        cursor.execute('''
            UPDATE chat_messages 
            SET content = ?, timestamp = CURRENT_TIMESTAMP
            WHERE id = ? AND session_id = ?
        ''', (new_message, message_id, session_id))
        
        # Delete all messages after this one (including AI response)
        cursor.execute('''
            DELETE FROM chat_messages 
            WHERE session_id = ? AND id > ?
        ''', (session_id, message_id))
        
        # Update session timestamp
        cursor.execute('''
            UPDATE chat_sessions
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (session_id,))
        
        conn.commit()
        conn.close()
        
        # Get updated chat history for context
        chat_history = get_chat_history(session_id)
        
        # Create agent with context
        agent = create_legal_agent(chat_history)
        
        # Get new AI response
        response = agent.run(new_message)
        ai_response = str(response.content)
        
        # Save new AI response
        save_message(session_id, "assistant", ai_response)
        
        return jsonify({
            "response": ai_response,
            "status": "success"
        })
        
    except Exception as e:
        print(f"Error in edit_message: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
