"""
Authentication decorators for FastAPI routes.
Unified decorator system that works with SmartAuthenticationManager.

Provides:
- Single decorator system for authentication handling
- Automatic conflict resolution
- Graceful degradation
- Security logging
- Support for guest users
"""
import logging
import json
import hashlib
from datetime import datetime
from functools import wraps
from fastapi import Request, HTTPException, Depends
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN, HTTP_429_TOO_MANY_REQUESTS
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)

# Will be set by main app during initialization
smart_auth_manager = None
logging_system = None

def init_auth_decorators(auth_manager, logger_system=None):
    """Initialize decorators with auth manager and logging system"""
    global smart_auth_manager, logging_system
    smart_auth_manager = auth_manager
    logging_system = logger_system
    logger.info("🔐 Smart authentication decorators initialized")

def smart_auth_required(allow_guest: bool = False, require_admin: bool = False):
    """
    🧠 SMART AUTHENTICATION DECORATOR
    
    Unified decorator that replaces all old auth decorators:
    - @auth_required
    - @require_auth  
    - @require_admin
    
    Args:
        allow_guest: Allow guest users to access this endpoint
        require_admin: Require admin privileges
        
    Usage:
        @smart_auth_required()  # Standard auth
        @smart_auth_required(allow_guest=True)  # Allow guests
        @smart_auth_required(require_admin=True)  # Admin only
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs) -> Any:
            try:
                # Check if auth manager is available
                if not smart_auth_manager:
                    logger.error("❌ Smart auth manager not initialized")
                    return _handle_auth_error("Authentication system unavailable", 503)
                
                # Get current authentication context
                auth_context = smart_auth_manager.get_current_auth()
                
                if not auth_context:
                    # No authentication found
                    logger.warning(f"🔐 Unauthenticated access attempt to {request.endpoint}")
                    _log_security_event('UNAUTHENTICATED_ACCESS', {
                        'endpoint': request.endpoint,
                        'ip': request.remote_addr,
                        'user_agent': request.user_agent.string,
                        'allow_guest': allow_guest
                    })
                    return _handle_auth_redirect('No authentication found')
                
                # Check if user is blocked
                is_blocked, block_reason = smart_auth_manager.is_user_blocked(auth_context.user_email)
                if is_blocked:
                    logger.warning(f"🚫 Blocked user access attempt: {auth_context.user_email}")
                    _log_security_event('BLOCKED_USER_ACCESS', {
                        'email': auth_context.user_email,
                        'reason': block_reason,
                        'endpoint': request.endpoint
                    })
                    return _handle_auth_error(f"Access blocked: {block_reason}", 403)
                
                # Admin requirement check
                if require_admin and not auth_context.is_admin:
                    logger.warning(f"🔒 Non-admin access attempt: {auth_context.user_email} to {request.endpoint}")
                    _log_security_event('ADMIN_REQUIRED_ACCESS', {
                        'email': auth_context.user_email,
                        'endpoint': request.endpoint,
                        'is_admin': auth_context.is_admin
                    })
                    return _handle_auth_redirect('Admin access required', redirect_to='admin_login')
                
                # Guest user check
                if auth_context.is_guest and not allow_guest:
                    logger.warning(f"👤 Guest access denied: {request.endpoint}")
                    _log_security_event('GUEST_ACCESS_DENIED', {
                        'endpoint': request.endpoint,
                        'allow_guest': allow_guest
                    })
                    return _handle_auth_redirect('Guest access not allowed')
                
                # Authentication successful - add context to request
                request.auth_context = auth_context
                request.current_user_email = auth_context.user_email
                request.is_admin = auth_context.is_admin
                request.is_guest = auth_context.is_guest
                
                logger.debug(f"✅ Authentication successful: {auth_context.user_email} accessing {request.endpoint}")
                return f(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"❌ Authentication decorator error: {e}")
                _log_security_event('AUTH_DECORATOR_ERROR', {
                    'error': str(e),
                    'endpoint': request.endpoint
                })
                return _handle_auth_error("Authentication system error", 500)
        
        return decorated_function
    return decorator

def admin_required(f: Callable) -> Callable:
    """
    🔐 ADMIN REQUIRED DECORATOR - Convenience wrapper
    
    Equivalent to @smart_auth_required(require_admin=True)
    """
    return smart_auth_required(require_admin=True)(f)

def auth_required(f: Callable) -> Callable:
    """
    🔐 STANDARD AUTH REQUIRED - Convenience wrapper
    
    Equivalent to @smart_auth_required()
    For backward compatibility with existing code
    """
    return smart_auth_required()(f)

def guest_allowed(f: Callable) -> Callable:
    """
    🔐 GUEST ALLOWED DECORATOR - Convenience wrapper
    
    Equivalent to @smart_auth_required(allow_guest=True)
    """
    return smart_auth_required(allow_guest=True)(f)

def validate_session_endpoint(f: Callable) -> Callable:
    """
    🔍 SESSION VALIDATION DECORATOR
    
    For endpoints that need to validate session without requiring authentication
    Used for health checks and session validation endpoints
    """
    @wraps(f)
    def decorated_function(*args, **kwargs) -> Any:
        try:
            if smart_auth_manager:
                auth_context = smart_auth_manager.get_current_auth()
                request.auth_context = auth_context
                request.current_user_email = auth_context.user_email if auth_context else None
                request.is_authenticated = bool(auth_context)
            else:
                request.auth_context = None
                request.current_user_email = None
                request.is_authenticated = False
                
            return f(*args, **kwargs)
            
        except Exception as e:
            logger.error(f"❌ Session validation decorator error: {e}")
            request.auth_context = None
            request.current_user_email = None  
            request.is_authenticated = False
            return f(*args, **kwargs)
    
    return decorated_function

def require_specific_user(user_email: str):
    """
    🔐 SPECIFIC USER REQUIRED DECORATOR
    
    Requires authentication as a specific user (useful for admin endpoints)
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs) -> Any:
            try:
                if not smart_auth_manager:
                    return _handle_auth_error("Authentication system unavailable", 503)
                
                auth_context = smart_auth_manager.get_current_auth()
                
                if not auth_context or auth_context.user_email != user_email:
                    logger.warning(f"🔒 Specific user access denied: required={user_email}, actual={auth_context.user_email if auth_context else 'none'}")
                    _log_security_event('SPECIFIC_USER_ACCESS_DENIED', {
                        'required_user': user_email,
                        'actual_user': auth_context.user_email if auth_context else 'none',
                        'endpoint': request.endpoint
                    })
                    return _handle_auth_error("Unauthorized access", 403)
                
                request.auth_context = auth_context
                return f(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"❌ Specific user decorator error: {e}")
                return _handle_auth_error("Authentication error", 500)
        
        return decorated_function
    return decorator

def rate_limit_by_user(max_requests: int = 100, time_window_minutes: int = 60):
    """
    🚦 USER-BASED RATE LIMITING DECORATOR
    
    Rate limits requests per authenticated user
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs) -> Any:
            try:
                if not smart_auth_manager:
                    return f(*args, **kwargs)  # No rate limiting if auth manager unavailable
                
                auth_context = smart_auth_manager.get_current_auth()
                if not auth_context:
                    return f(*args, **kwargs)  # No rate limiting for unauthenticated users
                
                # Simple rate limiting using Redis if available
                if hasattr(smart_auth_manager, 'redis_client') and smart_auth_manager.redis_client:
                    try:
                        user_id = auth_context.user_id
                        rate_key = f"rate_limit:{user_id}:{request.endpoint}"
                        
                        current_requests = smart_auth_manager.redis_client.get(rate_key)
                        if current_requests and int(current_requests) >= max_requests:
                            logger.warning(f"🚦 Rate limit exceeded: {auth_context.user_email} on {request.endpoint}")
                            _log_security_event('RATE_LIMIT_EXCEEDED', {
                                'email': auth_context.user_email,
                                'endpoint': request.endpoint,
                                'requests': int(current_requests)
                            })
                            return _handle_auth_error("Rate limit exceeded", 429)
                        
                        # Increment counter
                        pipe = smart_auth_manager.redis_client.pipeline()
                        pipe.incr(rate_key)
                        pipe.expire(rate_key, time_window_minutes * 60)
                        pipe.execute()
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Rate limiting failed: {e}")
                
                return f(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"❌ Rate limit decorator error: {e}")
                return f(*args, **kwargs)  # Continue without rate limiting on error
        
        return decorated_function
    return decorator

# ============ HELPER FUNCTIONS ============

def _handle_auth_redirect(reason: str, redirect_to: str = 'auth_page', request: Request = None) -> Any:
    """Handle authentication redirect"""
    try:
        # For FastAPI, always return JSON response for API endpoints
        if request and request.url.path.startswith('/api/'):
            return JSONResponse({
                "error": "Authentication required",
                "reason": reason,
                "redirect": f"/{redirect_to}" if redirect_to != 'auth_page' else '/auth'
            }, status_code=HTTP_401_UNAUTHORIZED)
        else:
            # Determine redirect URL based on request
            if redirect_to == 'admin_login':
                return RedirectResponse('/admin/login', status_code=302)
            else:
                return RedirectResponse('/auth', status_code=302)
    except Exception as e:
        logger.error(f"❌ Auth redirect error: {e}")
        return _handle_auth_error("Authentication required", HTTP_401_UNAUTHORIZED, request)

def _handle_auth_error(message: str, status_code: int, request: Request = None) -> Any:
    """Handle authentication error"""
    if request and request.url.path.startswith('/api/'):
        return JSONResponse({"error": message}, status_code=status_code)
    else:
        # For non-JSON requests, redirect to appropriate auth page
        if status_code == HTTP_403_FORBIDDEN:
            return RedirectResponse('/auth?error=access_denied', status_code=302)
        elif status_code == HTTP_429_TOO_MANY_REQUESTS:
            return RedirectResponse('/auth?error=rate_limited', status_code=302)
        else:
            return RedirectResponse('/auth?error=auth_error', status_code=302)

def _log_security_event(event_type: str, event_data: dict, level: str = 'WARNING'):
    """Log security events"""
    try:
        if logging_system:
            logging_system.log_security_event(event_type, event_data, level)
        else:
            # Fallback to standard logging
            logger.warning(f"🔒 SECURITY EVENT [{event_type}]: {event_data}")
    except Exception as e:
        logger.error(f"❌ Security event logging failed: {e}")

def get_current_user(request: Request = None) -> Optional[str]:
    """
    🔍 GET CURRENT USER - Convenience function

    Returns current user email if authenticated, None otherwise
    """
    try:
        if request and hasattr(request.state, 'current_user_email'):
            return request.state.current_user_email
        elif smart_auth_manager:
            auth_context = smart_auth_manager.get_current_auth()
            return auth_context.user_email if auth_context else None
        else:
            return None
    except Exception as e:
        logger.error(f"❌ Get current user failed: {e}")
        return None

def is_current_user_admin(request: Request = None) -> bool:
    """
    🔍 IS CURRENT USER ADMIN - Convenience function
    """
    try:
        if request and hasattr(request.state, 'is_admin'):
            return request.state.is_admin
        elif smart_auth_manager:
            auth_context = smart_auth_manager.get_current_auth()
            return auth_context.is_admin if auth_context else False
        else:
            return False
    except Exception as e:
        logger.error(f"❌ Admin check failed: {e}")
        return False

def is_current_user_guest(request: Request = None) -> bool:
    """
    🔍 IS CURRENT USER GUEST - Convenience function
    """
    try:
        if request and hasattr(request.state, 'is_guest'):
            return request.state.is_guest
        elif smart_auth_manager:
            auth_context = smart_auth_manager.get_current_auth()
            return auth_context.is_guest if auth_context else False
        else:
            return False
    except Exception as e:
        logger.error(f"❌ Guest check failed: {e}")
        return False

# ============ BACKWARD COMPATIBILITY DECORATORS ============

def require_auth(f: Callable) -> Callable:
    """Backward compatibility for old @require_auth decorator"""
    return smart_auth_required()(f)

def require_admin(f: Callable) -> Callable:
    """Backward compatibility for old @require_admin decorator"""  
    return smart_auth_required(require_admin=True)(f)