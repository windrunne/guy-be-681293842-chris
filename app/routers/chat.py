from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any
import json
from app.models.schemas import ChatMessage, ChatResponse
from app.services.chat_service import ChatbotService
from app.core.logging_config import get_logger

router = APIRouter()
chatbot_service = ChatbotService()
logger = get_logger(__name__)

@router.post("/stream")
async def chat_stream(message_data: ChatMessage):
    """Handle chat messages with streaming response"""    
    def generate():
        """Generator function that yields SSE events"""
        try:
            # Call the streaming method (which is a regular generator)
            for event in chatbot_service.get_chat_response_stream(
                message=message_data.message,
                conversation_history=message_data.conversation_history,
                user_data=message_data.user_data
            ):
                # Format as Server-Sent Events (SSE)
                # Ensure proper JSON serialization
                try:
                    event_json = json.dumps(event, ensure_ascii=False)
                    yield f"data: {event_json}\n\n"
                except Exception as e:
                    # Send error event instead
                    error_event = {
                        "type": "error",
                        "error": "Error formatting response"
                    }
                    yield f"data: {json.dumps(error_event)}\n\n"
        except Exception as e:
            error_event = {
                "type": "error",
                "error": "Sorry, I'm having trouble right now. Please try again."
            }
            yield f"data: {json.dumps(error_event)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no", 
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        }
    )
