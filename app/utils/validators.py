"""Validation utilities for user data"""
from app.core.config import settings

def validate_name(name: str, message: str = "", check_user_feedback: bool = True) -> bool:
    """Validate name field"""
    if not name:
        return False
    
    name_str = str(name).strip()
    
    if len(name_str) < settings.VALIDATION_NAME_MIN_LENGTH:
        return False
    
    if name_str.lower() in settings.invalid_name_words_list:
        return False
    
    return True


def validate_email(email: str, message: str = "", check_user_feedback: bool = True) -> bool:
    """Validate email field"""
    if not email:
        return False
    
    email_str = str(email).strip()
    
    if len(email_str) < settings.VALIDATION_EMAIL_MIN_LENGTH:
        return False
    
    if '@' not in email_str:
        return False
    
    if '@' in email_str:
        domain = email_str.split('@')[-1]
        if '.' not in domain:
            return False
    
    return True


def validate_income(income: str, message: str = "", check_user_feedback: bool = True) -> bool:
    """Validate income field"""
    if not income:
        return False
    
    income_str = str(income).strip()
    
    if len(income_str) < settings.VALIDATION_INCOME_MIN_LENGTH:
        return False
    
    if len(income_str) > settings.VALIDATION_INCOME_MAX_LENGTH:
        return False
    
    return True

