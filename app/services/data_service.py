from supabase import create_client, Client
from typing import Dict, Any, List, Optional
from app.core.config import settings
from app.services.email_service import EmailService
from app.core.logging_config import get_logger

logger = get_logger(__name__)

class DataService:
    def __init__(self):
        self.supabase: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_KEY
        )
        self.email_service = EmailService()
    
    def save_user_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save user data to Supabase (prevents duplicates by email)"""
        try:
            # Prepare data
            data = {
                "name": user_data.get("name"),
                "email": user_data.get("email"),
                "income": user_data.get("income"),
            }
            
            # Check if user with this email already exists
            email = data.get("email")
            if email:
                existing = self.supabase.table(settings.SUPABASE_TABLE_NAME).select("*").eq("email", email).execute()
                
                if existing.data and len(existing.data) > 0:
                    existing_record = existing.data[0]
                    return {
                        "success": True,
                        "id": existing_record.get("id"),
                        "data": existing_record,
                        "already_exists": True,
                        "message": "User data already exists in database"
                    }
            
            # No existing record found, proceed with insert
            data["created_at"] = "now()"
            
            # Insert into Supabase
            result = self.supabase.table(settings.SUPABASE_TABLE_NAME).insert(data).execute()
            
            if result.data:
                
                # Send structured output only for new records
                try:
                    self._send_structured_output(data)
                except Exception as email_error:
                    logger.warning(f"Failed to send email notification: {email_error}")
                
                return {
                    "success": True,
                    "id": result.data[0].get("id"),
                    "data": data,
                    "already_exists": False
                }
            else:
                return {"success": False, "error": "No data returned"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_user_data(self, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve user data from Supabase"""
        try:
            if user_id:
                result = self.supabase.table(settings.SUPABASE_TABLE_NAME).select("*").eq("id", user_id).execute()
            else:
                result = self.supabase.table(settings.SUPABASE_TABLE_NAME).select("*").order("created_at", desc=True).execute()
            
            data_count = len(result.data) if result.data else 0
            return result.data or []
        except Exception as e:
            return []
    
    def _send_structured_output(self, data: Dict[str, Any]):
        """Send structured data to external destination (email)"""
        try:
            self.email_service.send_user_data(data)
        except Exception as e:
            logger.error(f"Error sending structured output (email): {str(e)}", exc_info=True)
