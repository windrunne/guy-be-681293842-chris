"""Main chatbot service - orchestrates conversation flow"""
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, Optional
import json
import time
from app.core.config import settings
from app.core.logging_config import get_logger
from app.utils.dependencies import get_openai_client
from app.services.rag_service import RAGService
from app.services.data_extraction_service import DataExtractionService
from app.services.response_builder import ResponseBuilder
from app.services.data_service import DataService

logger = get_logger(__name__)


class ChatbotService:
    """Main chatbot service orchestrating conversation flow"""
    
    def __init__(self):
        self.client = get_openai_client()
        self.llm = ChatOpenAI(
            model_name=settings.OPENAI_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.rag_service = RAGService()
        self.data_extraction_service = DataExtractionService()
        self.response_builder = ResponseBuilder()
        self.data_service = DataService()
    
    def get_chat_response_stream(
        self,
        message: str,
        conversation_history: List[Dict[str, str]],
        user_data: Optional[Dict[str, Any]] = None
    ):
        """Generate chatbot response with streaming support"""
        try:
            yield {"type": "start", "status": "processing"}
            
            user_data = user_data or {}
            original_user_data = user_data.copy()
            
            # Check if user data is complete
            has_all_user_data = self.data_extraction_service.is_data_complete(user_data)
            had_all_data_before = has_all_user_data
            
            # Extract user data if incomplete
            extracted_data = None
            if not has_all_user_data:
                extracted_data = self.data_extraction_service.extract_user_data(message, user_data)
                user_data = extracted_data
                has_all_user_data = self.data_extraction_service.is_data_complete(user_data)
            
            # Track if data was just completed in this request
            data_just_completed = not had_all_data_before and has_all_user_data
            
            # Check if new data was actually extracted
            new_data_extracted = False
            if extracted_data:
                for field in ['name', 'email', 'income']:
                    original_value = original_user_data.get(field, '')
                    new_value = extracted_data.get(field, '')
                    if new_value and new_value != original_value:
                        new_data_extracted = True
                        break
            
            # Retrieve RAG context
            rag_context = None
            rag_used = False
            if has_all_user_data and settings.CHAT_RAG_ENABLED and not data_just_completed:
                yield {"type": "progress", "status": "rag_search", "message": "Searching knowledge base..."}
                rag_context = self.rag_service.retrieve_context(message)
                rag_used = bool(rag_context)
            
            # Build messages
            messages = self.response_builder.build_messages(
                message=message,
                conversation_history=conversation_history,
                user_data=user_data,
                rag_context=rag_context
            )
            
            # Stream response
            yield {"type": "progress", "status": "generating", "message": "Generating response..."}
            
            stream = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=messages,
                temperature=settings.OPENAI_TEMPERATURE,
                max_tokens=settings.OPENAI_MAX_TOKENS,
                stream=True
            )
            
            response_text = ""
            for chunk in stream:
                try:
                    if (chunk.choices and 
                        len(chunk.choices) > 0 and 
                        hasattr(chunk.choices[0], 'delta') and
                        chunk.choices[0].delta and
                        hasattr(chunk.choices[0].delta, 'content') and
                        chunk.choices[0].delta.content is not None):
                        
                        content = chunk.choices[0].delta.content
                        response_text += content
                        yield {"type": "chunk", "data": content}
                except Exception as e:
                    logger.warning(f"Error processing chunk: {e}")
                    continue
            
            # Save data only if data was just completed AND new data was extracted
            if data_just_completed and new_data_extracted:
                try:
                    result = self.data_service.save_user_data(user_data)
                    if result.get('success'):
                        if result.get('already_exists'):
                            logger.info(f"User data already exists - ID: {result.get('id')}")
                        else:
                            logger.info(f"User data saved - ID: {result.get('id')}")
                    else:
                        logger.error(f"Failed to save user data: {result.get('error')}")
                except Exception as e:
                    logger.error(f"Data save error: {e}", exc_info=True)
            
            yield {
                "type": "complete",
                "response": response_text,
                "user_data": user_data,
                "rag_used": rag_used
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "error": "Sorry, I'm having a moment. Let me get back to you on that market question."
            }
