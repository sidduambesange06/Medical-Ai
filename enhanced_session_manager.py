"""
Session management system.
Handles login state detection and persistence, session validation and cleanup,
admin removal detection, and cross-request session management.
"""

import os
import json
import time
import logging
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager

try:
    from fastapi import Request
    from supabase import create_client, Client
    import redis
    import jwt
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Run: pip install fastapi supabase redis PyJWT")

logger = logging.getLogger(__name__)

@dataclass
class SessionData:
    """Enhanced session data structure"""
    user_id: str
    email: str
    name: str
    auth_provider: str
    session_id: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    is_verified: bool = True
    permissions: List[str] = None
    metadata: Dict[str, Any] = None
    is_admin: bool = False
    login_count: int = 1
    last_ip: str = ""
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = ['chat', 'profile']
        if self.metadata is None:
            self.metadata = {}

class EnhancedSessionManager:
    """Production-grade session manager with comprehensive features"""
    
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY') 
        self.service_role_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        self.jwt_secret = os.getenv('JWT_SECRET', secrets.token_urlsafe(32))
        
        if not all([self.supabase_url, self.supabase_key]):
            raise ValueError("Missing Supabase credentials")
            
        self.client = create_client(self.supabase_url, self.supabase_key)
        self.admin_client = create_client(self.supabase_url, self.service_role_key) if self.service_role_key else self.client
        
        # Initialize Redis for session storage
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_available = True
            # Test connection
            self.redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Using memory storage.")
            self.redis_available = False
            self.memory_sessions = {}
        
        self.session_timeout = int(os.getenv('SESSION_TIMEOUT_HOURS', 24)) * 3600
        logger.info("Enhanced Session Manager initialized")
    
    def create_session(self, user_data: Dict, remember_me: bool = False) -> SessionData:
        """Create new user session with persistence"""
        try:
            session_id = f"sess_{secrets.token_urlsafe(32)}"
            current_time = datetime.now(timezone.utc)
            
            # Extended session for remember me
            timeout = (7 * 24 * 3600) if remember_me else self.session_timeout
            expires_at = current_time + timedelta(seconds=timeout)
            
            session_data = SessionData(
                user_id=user_data['user_id'],
                email=user_data['email'],
                name=user_data.get('name', user_data['email'].split('@')[0]),
                auth_provider=user_data.get('auth_provider', 'email'),
                session_id=session_id,
                created_at=current_time,
                expires_at=expires_at,
                last_activity=current_time,
                is_admin=self._check_admin_status(user_data['email']),
                last_ip=request.remote_addr if request else ""
            )
            
            # Store session
            self._store_session(session_data)
            
            # Update Flask session
            session['session_id'] = session_id
            session['user_id'] = user_data['user_id']
            session['email'] = user_data['email']
            session['is_admin'] = session_data.is_admin
            session.permanent = remember_me
            
            # Log session creation
            self._log_session_event('created', session_data)
            
            logger.info(f"Session created for {user_data['email']}: {session_id}")
            return session_data
            
        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            raise
    
    def get_session(self, session_id: str = None) -> Optional[SessionData]:
        """Get session data with validation"""
        try:
            if not session_id:
                session_id = session.get('session_id')
            
            if not session_id:
                return None
            
            # Get from storage
            session_data = self._get_session(session_id)
            if not session_data:
                return None
            
            # Check if session expired
            if datetime.now(timezone.utc) > session_data.expires_at:
                self.destroy_session(session_id)
                return None
            
            # Validate user still exists and is active
            if not self._validate_user_exists(session_data.user_id, session_data.email):
                logger.warning(f"User {session_data.email} no longer exists - destroying session")
                self.destroy_session(session_id)
                return None
            
            # Update last activity
            session_data.last_activity = datetime.now(timezone.utc)
            session_data.last_ip = request.remote_addr if request else session_data.last_ip
            self._store_session(session_data)
            
            return session_data
            
        except Exception as e:
            logger.error(f"Session retrieval failed: {e}")
            return None
    
    def _validate_user_exists(self, user_id: str, email: str) -> bool:
        """Validate user still exists in database"""
        try:
            result = self.client.table('users').select('id, is_active').eq('email', email).execute()
            if not result.data:
                return False
            
            user = result.data[0]
            return user.get('is_active', False)
            
        except Exception as e:
            logger.error(f"User validation failed: {e}")
            return False
    
    def _check_admin_status(self, email: str) -> bool:
        """Check if user has admin privileges"""
        try:
            admin_emails = os.getenv('ADMIN_EMAILS', '').split(',')
            admin_emails = [e.strip().lower() for e in admin_emails if e.strip()]
            
            if email.lower() in admin_emails:
                return True
            
            # Check admin table if exists
            try:
                result = self.admin_client.table('admin_users').select('*').eq('email', email).execute()
                return bool(result.data)
            except:
                pass
                
            return False
            
        except Exception as e:
            logger.error(f"Admin check failed: {e}")
            return False
    
    def update_session_activity(self, session_id: str = None):
        """Update session last activity"""
        try:
            if not session_id:
                session_id = session.get('session_id')
            
            if session_id:
                session_data = self._get_session(session_id)
                if session_data:
                    session_data.last_activity = datetime.now(timezone.utc)
                    session_data.last_ip = request.remote_addr if request else session_data.last_ip
                    self._store_session(session_data)
                    
        except Exception as e:
            logger.error(f"Activity update failed: {e}")
    
    def destroy_session(self, session_id: str = None):
        """Destroy user session"""
        try:
            if not session_id:
                session_id = session.get('session_id')
            
            if session_id:
                # Get session for logging
                session_data = self._get_session(session_id)
                
                # Remove from storage
                self._remove_session(session_id)
                
                # Clear Flask session
                session.clear()
                
                # Log session destruction
                if session_data:
                    self._log_session_event('destroyed', session_data)
                
                logger.info(f"Session destroyed: {session_id}")
                
        except Exception as e:
            logger.error(f"Session destruction failed: {e}")
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        try:
            if self.redis_available:
                # Redis TTL handles this automatically
                pass
            else:
                # Manual cleanup for memory storage
                current_time = datetime.now(timezone.utc)
                expired_sessions = []
                
                for session_id, data in self.memory_sessions.items():
                    try:
                        expires_at = datetime.fromisoformat(data['expires_at'].replace('Z', '+00:00'))
                        if current_time > expires_at:
                            expired_sessions.append(session_id)
                    except:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.memory_sessions[session_id]
                
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")
    
    def get_all_user_sessions(self, user_id: str) -> List[SessionData]:
        """Get all sessions for a user"""
        try:
            sessions = []
            
            if self.redis_available:
                pattern = f"session:*"
                for key in self.redis_client.scan_iter(match=pattern):
                    try:
                        data = json.loads(self.redis_client.get(key))
                        if data.get('user_id') == user_id:
                            session_data = self._dict_to_session(data)
                            sessions.append(session_data)
                    except:
                        continue
            else:
                for session_id, data in self.memory_sessions.items():
                    if data.get('user_id') == user_id:
                        session_data = self._dict_to_session(data)
                        sessions.append(session_data)
            
            return sorted(sessions, key=lambda x: x.last_activity, reverse=True)
            
        except Exception as e:
            logger.error(f"Get user sessions failed: {e}")
            return []
    
    def revoke_all_user_sessions(self, user_id: str):
        """Revoke all sessions for a user"""
        try:
            user_sessions = self.get_all_user_sessions(user_id)
            for session_data in user_sessions:
                self._remove_session(session_data.session_id)
            
            logger.info(f"Revoked {len(user_sessions)} sessions for user {user_id}")
            
        except Exception as e:
            logger.error(f"Session revocation failed: {e}")
    
    def _store_session(self, session_data: SessionData):
        """Store session data"""
        try:
            session_dict = asdict(session_data)
            # Convert datetime to ISO strings
            session_dict['created_at'] = session_data.created_at.isoformat()
            session_dict['expires_at'] = session_data.expires_at.isoformat()
            session_dict['last_activity'] = session_data.last_activity.isoformat()
            
            if self.redis_available:
                key = f"session:{session_data.session_id}"
                ttl = int((session_data.expires_at - datetime.now(timezone.utc)).total_seconds())
                self.redis_client.setex(key, ttl, json.dumps(session_dict))
            else:
                self.memory_sessions[session_data.session_id] = session_dict
                
        except Exception as e:
            logger.error(f"Session storage failed: {e}")
            raise
    
    def _get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data from storage"""
        try:
            if self.redis_available:
                key = f"session:{session_id}"
                data = self.redis_client.get(key)
                if not data:
                    return None
                session_dict = json.loads(data)
            else:
                session_dict = self.memory_sessions.get(session_id)
                if not session_dict:
                    return None
            
            return self._dict_to_session(session_dict)
            
        except Exception as e:
            logger.error(f"Session retrieval failed: {e}")
            return None
    
    def _remove_session(self, session_id: str):
        """Remove session from storage"""
        try:
            if self.redis_available:
                key = f"session:{session_id}"
                self.redis_client.delete(key)
            else:
                self.memory_sessions.pop(session_id, None)
                
        except Exception as e:
            logger.error(f"Session removal failed: {e}")
    
    def _dict_to_session(self, session_dict: Dict) -> SessionData:
        """Convert dict to SessionData object"""
        # Convert ISO strings back to datetime
        session_dict = session_dict.copy()
        session_dict['created_at'] = datetime.fromisoformat(session_dict['created_at'].replace('Z', '+00:00'))
        session_dict['expires_at'] = datetime.fromisoformat(session_dict['expires_at'].replace('Z', '+00:00'))
        session_dict['last_activity'] = datetime.fromisoformat(session_dict['last_activity'].replace('Z', '+00:00'))
        
        return SessionData(**session_dict)
    
    def _log_session_event(self, event: str, session_data: SessionData):
        """Log session events for audit"""
        try:
            log_data = {
                'event': event,
                'session_id': session_data.session_id,
                'user_id': session_data.user_id,
                'email': session_data.email,
                'ip': session_data.last_ip,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Store in database if available
            try:
                self.client.table('session_logs').insert(log_data).execute()
            except:
                pass
                
            logger.info(f"Session {event}: {session_data.email}")
            
        except Exception as e:
            logger.warning(f"Session logging failed: {e}")
    
    def is_logged_in(self) -> bool:
        """Check if current request has valid session"""
        return self.get_current_user() is not None
    
    def get_current_user(self) -> Optional[SessionData]:
        """Get current logged-in user"""
        return self.get_session()
    
    def require_login(self):
        """Decorator/function to require login"""
        if not self.is_logged_in():
            raise Exception("Login required")
    
    def require_admin(self):
        """Decorator/function to require admin privileges"""
        user = self.get_current_user()
        if not user or not user.is_admin:
            raise Exception("Admin privileges required")
    
    def get_session_stats(self) -> Dict:
        """Get session statistics"""
        try:
            stats = {
                'total_sessions': 0,
                'active_sessions': 0,
                'expired_sessions': 0,
                'admin_sessions': 0,
                'last_cleanup': datetime.now(timezone.utc).isoformat()
            }
            
            current_time = datetime.now(timezone.utc)
            
            if self.redis_available:
                pattern = f"session:*"
                for key in self.redis_client.scan_iter(match=pattern):
                    stats['total_sessions'] += 1
                    try:
                        data = json.loads(self.redis_client.get(key))
                        expires_at = datetime.fromisoformat(data['expires_at'].replace('Z', '+00:00'))
                        if current_time <= expires_at:
                            stats['active_sessions'] += 1
                        else:
                            stats['expired_sessions'] += 1
                        
                        if data.get('is_admin'):
                            stats['admin_sessions'] += 1
                    except:
                        continue
            else:
                for session_id, data in self.memory_sessions.items():
                    stats['total_sessions'] += 1
                    try:
                        expires_at = datetime.fromisoformat(data['expires_at'].replace('Z', '+00:00'))
                        if current_time <= expires_at:
                            stats['active_sessions'] += 1
                        else:
                            stats['expired_sessions'] += 1
                            
                        if data.get('is_admin'):
                            stats['admin_sessions'] += 1
                    except:
                        continue
            
            return stats
            
        except Exception as e:
            logger.error(f"Session stats failed: {e}")
            return {}

# Global instance
session_manager = None

def get_session_manager() -> EnhancedSessionManager:
    """Get global session manager instance"""
    global session_manager
    if session_manager is None:
        session_manager = EnhancedSessionManager()
    return session_manager

# Flask decorators
def require_login(f):
    """Decorator to require login"""
    def wrapper(*args, **kwargs):
        sm = get_session_manager()
        if not sm.is_logged_in():
            return {"error": "Login required"}, 401
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

def require_admin(f):
    """Decorator to require admin privileges"""
    def wrapper(*args, **kwargs):
        sm = get_session_manager()
        user = sm.get_current_user()
        if not user or not user.is_admin:
            return {"error": "Admin privileges required"}, 403
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

if __name__ == "__main__":
    # Test the session manager
    sm = EnhancedSessionManager()
    stats = sm.get_session_stats()
    print(f"Session stats: {json.dumps(stats, indent=2)}")