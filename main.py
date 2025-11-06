from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import os
import time
from dotenv import load_dotenv
import uvicorn

from app.routers import chat, documents, data
from app.core.config import settings
from app.core.logging_config import setup_logging, get_logger

load_dotenv()

# Setup logging
setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)

app = FastAPI(
    title="Insomniac Hedge Fund Chatbot API",
    description="AI-powered chatbot with RAG integration",
    version="1.0.0"
)

# Request/Response logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests and responses"""
    start_time = time.time()
    
    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path} - "
        f"Client: {request.client.host if request.client else 'unknown'}"
    )
    
    # Log query parameters if present
    if request.query_params:
        logger.debug(f"Query params: {dict(request.query_params)}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Error: {request.method} {request.url.path} - "
            f"Error: {str(e)} - "
            f"Time: {process_time:.3f}s",
            exc_info=True
        )
        raise

# CORS middleware - Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Cannot use credentials with wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(data.router, prefix="/api/data", tags=["data"])

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Insomniac Hedge Fund Chatbot API", "status": "running"}

@app.get("/health")
async def health():
    logger.debug("Health check endpoint accessed")
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
