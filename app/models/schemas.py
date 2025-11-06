"""Pydantic schemas for request/response models"""
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Any, Optional


class ChatMessage(BaseModel):
    """Chat message request model"""
    message: str
    conversation_history: List[Dict[str, str]] = []
    user_data: Optional[Dict[str, Any]] = {}


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    user_data: Dict[str, Any]
    rag_used: bool


class UserData(BaseModel):
    """User data model"""
    name: Optional[str] = None
    email: Optional[str] = None
    income: Optional[str] = None


class DocumentStats(BaseModel):
    """Document statistics model"""
    total_vectors: int = 0
    dimension: int = 0
    index_fullness: float = 0.0
    error: Optional[str] = None


class ProcessingResult(BaseModel):
    """Document processing result model"""
    success: bool
    filename: Optional[str] = None
    chunks_processed: Optional[int] = None
    total_chars: Optional[int] = None
    error: Optional[str] = None

