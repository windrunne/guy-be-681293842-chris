"""Utility functions and helpers"""
from .dependencies import (
    get_openai_client,
    get_pinecone_client,
    get_vector_store,
    get_embeddings
)
from .validators import (
    validate_name,
    validate_email,
    validate_income
)

__all__ = [
    "get_openai_client",
    "get_pinecone_client", 
    "get_vector_store",
    "get_embeddings",
    "validate_name",
    "validate_email",
    "validate_income"
]

