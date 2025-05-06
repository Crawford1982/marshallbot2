import os
import logging
from flask import Flask, render_template, redirect, url_for, request, flash, send_from_directory, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API Key from environment variable is now used directly in the client initialization

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Ensure upload and embeddings folders exist
UPLOAD_FOLDER = 'uploads'
EMBEDDINGS_FOLDER = 'embeddings'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)

# Paths for embeddings and texts
EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_FOLDER, 'embeddings.pkl')
TEXTS_PATH = os.path.join(EMBEDDINGS_FOLDER, 'texts.pkl')

# Load embeddings and texts if they exist
document_embeddings = {}
document_texts = {}
if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(TEXTS_PATH):
    try:
        with open(EMBEDDINGS_PATH, 'rb') as f:
            document_embeddings = pickle.load(f)
        with open(TEXTS_PATH, 'rb') as f:
            document_texts = pickle.load(f)
        logger.info("Embeddings and texts loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading embeddings or texts: {e}", exc_info=True)

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id_, username, password_hash):
        self.id = id_
        self.username = username
        self.password_hash = password_hash

# User loader callback for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user_record = cursor.fetchone()
    conn.close()
    if user_record:
        return User(*user_record)
    return None

# Initialize the SQLite database
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Routes
@app.route('/')
def index():
    return redirect(url_for('login'))

# Secure Sign-In Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user_record = cursor.fetchone()
        conn.close()

        if user_record and check_password_hash(user_record[2], password):
            user = User(*user_record)
            login_user(user)
            return redirect(url_for('chatbot'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('login'))

    return render_template('login.html')

# Registration Page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        password_hash = generate_password_hash(password)
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
            conn.commit()
            conn.close()
            flash('Registration successful')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists')
            return redirect(url_for('register'))

    return render_template('register.html')

# Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Chatbot Interface Page
@app.route('/chatbot')
@login_required
def chatbot():
    return render_template('chatbot.html')

# Handle Chatbot Responses
@app.route('/get_response', methods=['POST'])
@login_required
def get_response():
    user_input = request.json['message']

    # Generate embedding for the user query
    query_embedding = generate_embedding(user_input)

    if query_embedding is None:
        return jsonify({'reply': "I'm sorry, I couldn't process your request."})

    # Find the most similar chunks from all documents
    similarities = []
    for filename, embeddings in document_embeddings.items():
        for idx, embedding in enumerate(embeddings):
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities.append((similarity, filename, idx))

    # Sort by similarity
    similarities.sort(reverse=True)

    # Select top chunks
    top_chunks = similarities[:3]  # Adjust the number as needed

    # Combine contexts from top chunks
    context = ""
    for _, filename, idx in top_chunks:
        context += document_texts[filename][idx] + "\n\n"

    # Construct the prompt
    if context.strip() != "":
        prompt = f"""
You are an assistant using the following context to answer the question, and you must use UK English in your responses.

Context:
{context}

Question:
{user_input}
"""
    else:
        prompt = user_input

    # OpenAI API call
    try:
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who uses the provided context to answer questions."},
                {"role": "user", "content": prompt}
            ]
        )
        assistant_reply = response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {e}", exc_info=True)
        assistant_reply = "I'm sorry, there was an issue processing your request."

    return jsonify({'reply': assistant_reply})

# Document Upload Page
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        uploaded_file = request.files['document']
        if uploaded_file.filename != '':
            filename = uploaded_file.filename
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            uploaded_file.save(upload_path)
            flash('File uploaded successfully')

            # Extract text based on file type
            if filename.lower().endswith('.pdf'):
                text = extract_text_from_pdf(upload_path)
            elif filename.lower().endswith('.docx'):
                text = extract_text_from_docx(upload_path)
            else:
                with open(upload_path, 'r', encoding='utf-8') as f:
                    text = f.read()

            # Split text into chunks
            text_chunks = split_text(text)

            # Generate embeddings for each chunk
            embeddings = []
            for chunk in text_chunks:
                embedding = generate_embedding(chunk)
                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    logger.warning(f"Failed to generate embedding for chunk: {chunk[:100]}")  # Log the first 100 characters of the chunk

            document_embeddings[filename] = embeddings
            document_texts[filename] = text_chunks

            # Save embeddings and texts
            try:
                with open(EMBEDDINGS_PATH, 'wb') as f:
                    pickle.dump(document_embeddings, f)
                with open(TEXTS_PATH, 'wb') as f:
                    pickle.dump(document_texts, f)
                logger.info(f"Embeddings and texts saved successfully for file: {filename}")
            except Exception as e:
                logger.error(f"Error saving embeddings or texts: {e}", exc_info=True)
        else:
            flash('No file selected')
        return redirect(url_for('upload'))

    return render_template('upload.html')

# Utility Functions
def generate_embedding(text):
    try:
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.embeddings.create(
            input=[text],
            model='text-embedding-ada-002'
        )
        embedding = response.data[0].embedding
        logger.info("Embedding generated successfully.")
        return np.array(embedding)
    except Exception as e:
        logger.error(f"Error generating embedding: {e}", exc_info=True)
        return None

def extract_text_from_pdf(file_path):
    text = ''
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        logger.info(f"Extracted text from PDF: {text[:200]}...")  # Log first 200 characters
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}", exc_info=True)
    return text

def extract_text_from_docx(file_path):
    text = ''
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + '\n'
        logger.info(f"Extracted text from DOCX: {text[:200]}...")  # Log first 200 characters
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}", exc_info=True)
    return text

def split_text(text, max_tokens=500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_chunk.append(word)
        current_length += 1
        if current_length >= max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

if __name__ == '__main__':
    app.run(debug=True, port=5001)
