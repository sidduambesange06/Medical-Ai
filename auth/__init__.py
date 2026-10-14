"""
🔐 Smart Authentication Module
Provides unified authentication system for MedAI
"""

# Import main components for easier access
from .decorators import (
    smart_auth_required, 
    admin_required, 
    auth_required, 
    guest_allowed,
    validate_session_endpoint, 
    rate_limit_by_user,
    get_current_user, 
    is_current_user_admin, 
    is_current_user_guest,
    init_auth_decorators
)

__all__ = [
    'smart_auth_required',
    'admin_required', 
    'auth_required',
    'guest_allowed',
    'validate_session_endpoint',
    'rate_limit_by_user', 
    'get_current_user',
    'is_current_user_admin',
    'is_current_user_guest',
    'init_auth_decorators'
]