# Marshall Bot

A chatbot application for answering questions about cement factory operations, designed to help employees access information from company documents.

## Features

- Document upload and processing
- AI-powered question answering based on document content
- User authentication system
- Chat history tracking
- Responsive web interface

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up your OpenAI API key:
   - Create a .env file in the root directory with:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```
   - Or set it as an environment variable

3. Run the application:
   - On Windows: Run `start_app.bat` or `start_app.ps1`

## Technologies

- Python with Flask
- HTML/CSS/JavaScript
- LLM for natural language processing
- SQLite for database 