from pydantic_settings import BaseSettings
from typing import List, Optional
from pydantic import field_validator

class Settings(BaseSettings):
    # OpenAI Configuration
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: int = 1500  # Increased for complete responses
    OPENAI_EXTRACTION_MODEL: str = "gpt-4-turbo-preview"
    OPENAI_EXTRACTION_TEMPERATURE: float = 0.1
    OPENAI_EXTRACTION_MAX_TOKENS: int = 200
    OPENAI_QUERY_GEN_MODEL: str = "gpt-4-turbo-preview"
    OPENAI_QUERY_GEN_TEMPERATURE: float = 0.3
    OPENAI_QUERY_GEN_MAX_TOKENS: int = 200
    OPENAI_VISION_MODEL: str = "gpt-4-vision-preview"
    OPENAI_VISION_MAX_TOKENS: int = 500
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Pinecone Configuration
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str = "us-east-1"
    PINECONE_INDEX_NAME: str = "rag-chatbot-index"
    PINECONE_BATCH_SIZE: int = 100
    PINECONE_RAG_K: int = 30  # Increased for better context retrieval
    PINECONE_RAG_SIMILARITY_THRESHOLD: float = 0.3  # Lower threshold for better recall (0-1)
    
    # Supabase Configuration
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_SERVICE_KEY: str
    SUPABASE_TABLE_NAME: str = "user_data"
    
    # Email Configuration
    EMAIL_FROM: str
    EMAIL_FROM_NAME: str
    EMAIL_RECIPIENT: str
    
    # SendGrid Configuration
    SENDGRID_API_KEY: str
    SENDGRID_ENABLED: bool = True
    
    # Document Processing Configuration
    DOCUMENT_CHUNK_SIZE: int = 1500
    DOCUMENT_CHUNK_OVERLAP: int = 150
    DOCUMENT_EMBEDDING_BATCH_SIZE: int = 100
    DOCUMENT_EMBEDDING_PARALLEL_WORKERS: int = 5  # Number of parallel workers for embedding generation
    DOCUMENT_PINECONE_UPSERT_PARALLEL_WORKERS: int = 10  # Number of parallel workers for Pinecone upsert
    DOCUMENT_IMAGE_EXTRACTION_PARALLEL_WORKERS: int = 5  # Number of parallel workers for image extraction/description
    
    # Chat Configuration
    CHAT_HISTORY_LIMIT: int = 5
    CHAT_RAG_ENABLED: bool = True
    CHAT_QUERY_GEN_ENABLED: bool = True  # Enable/disable query generation for speed
    CHAT_PARALLEL_SEARCH: bool = True  # Enable parallel vector search
    
    # Data Validation Configuration
    VALIDATION_NAME_MIN_LENGTH: int = 2
    VALIDATION_NAME_INVALID_WORDS: str = "interested,looking,trading,stock,market,hi,hello,hey,i,am,in,yes,no,ok,okay"
    VALIDATION_EMAIL_MIN_LENGTH: int = 5
    VALIDATION_INCOME_MAX_LENGTH: int = 50
    VALIDATION_INCOME_MIN_LENGTH: int = 1
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    
    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            # Split comma-separated string and strip whitespace
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        elif isinstance(v, list):
            return v
        return v
    
    @property
    def invalid_name_words_list(self) -> List[str]:
        """Parse invalid name words from comma-separated string"""
        return [word.strip().lower() for word in self.VALIDATION_NAME_INVALID_WORDS.split(',') if word.strip()]
    
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
