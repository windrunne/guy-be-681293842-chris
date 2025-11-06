from fastapi import APIRouter, HTTPException
from typing import Optional
from app.models.schemas import UserData
from app.services.data_service import DataService
from app.core.logging_config import get_logger

router = APIRouter()
data_service = DataService()
logger = get_logger(__name__)

@router.post("/save")
async def save_user_data(user_data: UserData):
    """Save user data to database"""
    try:
        data_dict = user_data.dict(exclude_none=True)
        result = data_service.save_user_data(data_dict)
        if result.get("success"):
            return {
                "success": True,
                "id": result.get("id"),
                "message": "User data saved and sent successfully"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Failed to save user data")
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def get_user_data(user_id: Optional[int] = None):
    """Retrieve user data"""
    try:
        data = data_service.get_user_data(user_id)
        return {"success": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
