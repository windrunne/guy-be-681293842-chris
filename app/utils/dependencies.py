"""Dependency injection utilities for external services"""
from typing import Optional
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from app.core.config import settings
from langchain_pinecone import PineconeVectorStore

# Global clients (singleton pattern)
_openai_client: Optional[OpenAI] = None
_pinecone_client: Optional[Pinecone] = None
_embeddings: Optional[OpenAIEmbeddings] = None
_vector_store: Optional[PineconeVectorStore] = None


def get_openai_client() -> OpenAI:
    """Get or create OpenAI client (singleton)"""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


def get_pinecone_client() -> Optional[Pinecone]:
    """Get or create Pinecone client (singleton)"""
    global _pinecone_client
    if _pinecone_client is None:
        try:
            _pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
        except Exception as e:
            _pinecone_client = None
    return _pinecone_client


def get_embeddings() -> OpenAIEmbeddings:
    """Get or create embeddings model (singleton)"""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_EMBEDDING_MODEL
        )
    return _embeddings


def get_vector_store() -> Optional[PineconeVectorStore]:
    """Get or create vector store (singleton)"""
    global _vector_store
    if _vector_store is None:        
        pinecone_client = get_pinecone_client()
        if not pinecone_client:
            return None
        
        try:
            embeddings = get_embeddings()
            _vector_store = PineconeVectorStore(
                index_name=settings.PINECONE_INDEX_NAME,
                embedding=embeddings,
                pinecone_api_key=settings.PINECONE_API_KEY
            )
        except Exception as e:
            _vector_store = None
    return _vector_store

