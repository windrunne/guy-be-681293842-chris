"""Service for building conversation context and messages"""
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.utils.prompts import SYSTEM_PROMPT
from app.utils.validators import validate_name, validate_email, validate_income

class ResponseBuilder:
    """Service for building conversation context and messages"""
    
    def build_messages(
        self,
        message: str,
        conversation_history: List[Dict[str, str]],
        user_data: Optional[Dict[str, Any]] = None,
        rag_context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Build complete message list for OpenAI API"""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add user data context
        user_context = self._build_user_context(user_data)
        if user_context:
            messages.append({"role": "system", "content": user_context})
        
        # Add RAG context if available
        if rag_context:
            messages.insert(-1, {"role": "system", "content": rag_context})
        
        # Add conversation history
        for hist in conversation_history[-settings.CHAT_HISTORY_LIMIT:]:
            messages.append({"role": hist["role"], "content": hist["content"]})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        return messages
    
    def _build_user_context(self, user_data: Optional[Dict[str, Any]]) -> str:
        """Build user context message for system prompt"""
        if not user_data:
            return "\nIMPORTANT: You need to ask for: name, email, income. Ask naturally, one at a time, within the conversation."
        
        user_context_parts = []
        validated_fields = []
        invalid_fields = []
        
        # Validate each field
        if user_data.get('name'):
            name = str(user_data.get('name', '')).strip()
            if validate_name(name, "", check_user_feedback=False):
                validated_fields.append('name')
                user_context_parts.append(f"User's name: {name}")
            else:
                invalid_fields.append('name')
        
        if user_data.get('email'):
            email = str(user_data.get('email', '')).strip()
            if validate_email(email, "", check_user_feedback=False):
                validated_fields.append('email')
                user_context_parts.append(f"User's email: {email}")
            else:
                invalid_fields.append('email')
        
        if user_data.get('income'):
            income = str(user_data.get('income', '')).strip()
            if validate_income(income, "", check_user_feedback=False):
                validated_fields.append('income')
                user_context_parts.append(f"User's income: {income}")
            else:
                invalid_fields.append('income')
        
        # Determine what to ask for
        fields_to_ask = []
        for field in ['name', 'email', 'income']:
            if field not in validated_fields:
                fields_to_ask.append(field)
        
        for field in invalid_fields:
            if field not in fields_to_ask:
                fields_to_ask.append(field)
        
        # Build instruction message
        if fields_to_ask:
            # Remove invalid data
            for field in invalid_fields:
                if user_data and field in user_data:
                    user_data.pop(field, None)
            
            if len(fields_to_ask) == 1:
                field_name = fields_to_ask[0]
                if field_name in invalid_fields:
                    user_context_parts.append(f"\nIMPORTANT: The user provided {field_name} but it seems invalid or unclear. Ask them again to provide their {field_name} clearly.")
                else:
                    user_context_parts.append(f"\nIMPORTANT: You need to ask for the user's {field_name}. Ask naturally within the conversation.")
            else:
                field_list = ', '.join(fields_to_ask)
                invalid_list = [f for f in fields_to_ask if f in invalid_fields]
                if invalid_list:
                    user_context_parts.append(f"\nIMPORTANT: The user provided {', '.join(invalid_list)} but it seems invalid or unclear. Ask them again to provide their {', '.join(invalid_list)} clearly. Also, you still need to ask for: {', '.join([f for f in fields_to_ask if f not in invalid_list])}.")
                else:
                    user_context_parts.append(f"\nIMPORTANT: You need to ask for: {field_list}. Ask naturally, one at a time, within the conversation.")
        
        return "\n".join(user_context_parts) if user_context_parts else ""

