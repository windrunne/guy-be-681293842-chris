"""Service for extracting user data from conversation"""
from typing import Dict, Any, Optional
import json
import re
from app.core.config import settings
from app.core.logging_config import get_logger
from app.utils.dependencies import get_openai_client
from app.utils.validators import validate_name, validate_email, validate_income
from app.utils.prompts import get_data_extraction_prompt

logger = get_logger(__name__)


class DataExtractionService:
    """Service for extracting structured user data from natural language"""
    
    def __init__(self):
        self.client = get_openai_client()
    
    def extract_user_data(
        self,
        message: str,
        existing_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract user data from conversation using AI analysis"""
        data = existing_data.copy() if existing_data else {}
        
        try:
            extraction_prompt = get_data_extraction_prompt(message, existing_data or {})
            
            extraction_response = self.client.chat.completions.create(
                model=settings.OPENAI_EXTRACTION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data extraction assistant. Extract user information from messages and return ONLY valid JSON."
                    },
                    {
                        "role": "user",
                        "content": extraction_prompt
                    }
                ],
                temperature=settings.OPENAI_EXTRACTION_TEMPERATURE,
                max_tokens=settings.OPENAI_EXTRACTION_MAX_TOKENS
            )
            
            extraction_text = extraction_response.choices[0].message.content.strip()
            
            # Clean up response
            if extraction_text.startswith("```json"):
                extraction_text = extraction_text[7:]
            if extraction_text.startswith("```"):
                extraction_text = extraction_text[3:]
            if extraction_text.endswith("```"):
                extraction_text = extraction_text[:-3]
            extraction_text = extraction_text.strip()
            
            extracted_data = json.loads(extraction_text)
            
            # Validate and update data
            self._update_data_with_validation(data, extracted_data)
            
        except json.JSONDecodeError as e:
            self._fallback_email_extraction(message, data)
        except Exception as e:
            self._fallback_email_extraction(message, data)
        
        return data
    
    def _update_data_with_validation(self, data: Dict[str, Any], extracted_data: Dict[str, Any]):
        """Update data dictionary with validated extracted values"""
        # Name validation and extraction
        if extracted_data.get("name"):
            name_value = extracted_data["name"]
            name_str = str(name_value).strip() if name_value else ""
            
            if (name_str and 
                name_str.lower() not in ["null", "none", ""] and
                validate_name(name_str)):
                if not data.get("name") or name_str.lower() != data.get("name", "").lower():
                    data["name"] = name_str
        
        # Email validation and extraction
        if extracted_data.get("email"):
            email_value = extracted_data["email"]
            email_str = str(email_value).strip() if email_value else ""
            
            if (email_str and 
                email_str.lower() not in ["null", "none", ""] and
                validate_email(email_str)):
                if not data.get("email") or email_str.lower() != data.get("email", "").lower():
                    data["email"] = email_str
        
        # Income validation and extraction
        if extracted_data.get("income"):
            income_value = extracted_data["income"]
            income_str = str(income_value).strip() if income_value else ""
            
            if (income_str and 
                income_str.lower() not in ["null", "none", ""] and
                validate_income(income_str)):
                if not data.get("income") or income_str != data.get("income", ""):
                    data["income"] = income_str
    
    def _fallback_email_extraction(self, message: str, data: Dict[str, Any]):
        """Fallback to regex-based email extraction"""
        if "@" in message and not data.get("email"):
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            match = re.search(email_pattern, message)
            if match:
                data["email"] = match.group(0)
    
    def validate_user_data(self, user_data: Dict[str, Any]) -> Dict[str, bool]:
        """Validate all user data fields"""
        return {
            "name": validate_name(user_data.get("name", "")) if user_data.get("name") else False,
            "email": validate_email(user_data.get("email", "")) if user_data.get("email") else False,
            "income": validate_income(user_data.get("income", "")) if user_data.get("income") else False
        }
    
    def is_data_complete(self, user_data: Dict[str, Any]) -> bool:
        """Check if all required user data is collected and valid"""
        validation = self.validate_user_data(user_data)
        return all(validation.values())

