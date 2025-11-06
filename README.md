# Backend API

FastAPI backend for the Insomniac Hedge Fund Chatbot.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file with your API keys (see `.env.example`)

3. Run the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Environment Variables

Required:
- `OPENAI_API_KEY`: OpenAI API key
- `PINECONE_API_KEY`: Pinecone API key
- `PINECONE_ENVIRONMENT`: Pinecone environment (optional for newer versions)
- `SUPABASE_URL`: Supabase project URL
- `SUPABASE_KEY`: Supabase anon key
- `SUPABASE_SERVICE_KEY`: Supabase service role key

Optional:
- `EMAIL_SERVICE_API_KEY`: Email service API key
- `EMAIL_SERVICE_URL`: Email service endpoint
- `EMAIL_FROM`: Sender email
- `EMAIL_TO`: Recipient email
