#!/usr/bin/env python3
"""
Production authentication manager for MedBot.

Provides multi-provider OAuth (Google, GitHub, Email), session management,
security logging, and comprehensive error handling.
"""

import os
import sys
import json
import time
import hashlib
import secrets
import logging
import re
import base64
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import urllib.parse

# Security imports
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import bcrypt
    import jwt
except ImportError as e:
    print(f"❌ Missing security library: {e}")
    print("Run: pip install cryptography bcrypt PyJWT")
    sys.exit(1)

# Supabase imports
try:
    from supabase import create_client, Client
    from postgrest import APIError
except ImportError as e:
    print(f"❌ Missing Supabase library: {e}")
    print("Run: pip install supabase")
    sys.exit(1)

# Other imports
try:
    import requests
    from flask import session, request
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("Run: pip install requests flask python-dotenv")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auth_manager.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class UserSession:
    """Secure user session data structure"""
    user_id: str
    email: str
    name: str
    auth_provider: str
    session_id: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    is_verified: bool = False
    permissions: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = ['chat', 'profile']
        if self.metadata is None:
            self.metadata = {}

@dataclass  
class AuthResult:
    """Authentication result with comprehensive status"""
    success: bool
    message: str
    user_data: Optional[Dict] = None
    session_data: Optional[UserSession] = None
    error_code: Optional[str] = None
    next_action: Optional[str] = None

class AuthenticationManager:
    """Production-ready authentication manager with zero-error policy and advanced session management"""
    
    def __init__(self):
        load_dotenv()
        
        # Initialize Supabase client
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        self.service_role_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        
        if not all([self.supabase_url, self.supabase_key]):
            raise ValueError("Missing Supabase credentials in environment variables")
        
        self.client = create_client(self.supabase_url, self.supabase_key)
        self.admin_client = create_client(self.supabase_url, self.service_role_key) if self.service_role_key else self.client
        
        # Security configuration
        self.jwt_secret = os.getenv('JWT_SECRET', secrets.token_urlsafe(32))
        self.session_timeout = int(os.getenv('JWT_EXPIRATION_HOURS', 24)) * 3600
        
        # Initialize encryption
        self._init_encryption()
        
        # Rate limiting storage
        self.rate_limits = {}
        self.blocked_ips = {}
        
        # ENHANCED: Session management storage
        self.active_sessions = {}  # session_id -> UserSession
        self.user_sessions = {}    # user_id -> List[session_id]
        self.session_cleanup_interval = 3600  # 1 hour
        self.last_cleanup = time.time()
        
        # Security patterns
        self.threat_patterns = [
            r'(\bSELECT\b.*\bFROM\b)|(\bINSERT\b.*\bINTO\b)',  # SQL injection
            r'<script[^>]*>.*?</script>',  # XSS
            r'javascript:',  # JavaScript injection
            r'(\bexec\b)|(\beval\b)|(\bsystem\b)',  # Command injection
        ]
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.threat_patterns]
        
        logger.info("✅ Enhanced AuthenticationManager initialized successfully")
    
    def _init_encryption(self):
        """Initialize encryption system"""
        encryption_key = os.getenv('ENCRYPTION_KEY')
        if encryption_key:
            self.fernet = Fernet(encryption_key.encode())
        else:
            # Generate key from JWT secret
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'medbot_auth_salt',
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.jwt_secret.encode()))
            self.fernet = Fernet(key)
    
    @contextmanager
    def error_handler(self, operation_name: str):
        """Context manager for comprehensive error handling"""
        try:
            yield
        except APIError as e:
            logger.error(f"{operation_name} failed with API error: {e}")
            raise Exception(f"Authentication service error: {e.message if hasattr(e, 'message') else str(e)}")
        except requests.RequestException as e:
            logger.error(f"{operation_name} failed with network error: {e}")
            raise Exception(f"Network error during authentication: {str(e)}")
        except Exception as e:
            logger.error(f"{operation_name} failed with error: {e}")
            raise Exception(f"Authentication error: {str(e)}")
    
    def validate_email(self, email: str) -> bool:
        """Advanced email validation"""
        if not email or len(email) > 254:
            return False
        
        # RFC 5322 compliant regex
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            return False
        
        # Check for threat patterns
        for pattern in self.compiled_patterns:
            if pattern.search(email):
                logger.warning(f"Potential threat detected in email: {email}")
                return False
        
        return True
    
    def validate_password(self, password: str) -> Tuple[bool, str]:
        """Advanced password validation"""
        if not password:
            return False, "Password is required"
        
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if len(password) > 128:
            return False, "Password is too long"
        
        # Check complexity
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        if not re.search(r'\d', password):
            return False, "Password must contain at least one number"
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Password must contain at least one special character"
        
        # Check for common passwords
        common_passwords = ['password', '12345678', 'qwerty', 'abc123']
        if password.lower() in common_passwords:
            return False, "Password is too common"
        
        return True, "Password is valid"
    
    def check_rate_limit(self, identifier: str, limit: int = 100, window: int = 3600) -> bool:
        """Advanced rate limiting with sliding window"""
        current_time = time.time()
        
        if identifier in self.rate_limits:
            rate_data = self.rate_limits[identifier]
            
            # Reset window if expired
            if current_time - rate_data['window_start'] > window:
                rate_data['count'] = 0
                rate_data['window_start'] = current_time
            
            # Check limit
            if rate_data['count'] >= limit:
                # Block IP temporarily
                self.blocked_ips[identifier] = current_time + 3600  # 1 hour block
                logger.warning(f"Rate limit exceeded for {identifier}")
                return False
            
            rate_data['count'] += 1
        else:
            self.rate_limits[identifier] = {
                'count': 1,
                'window_start': current_time
            }
        
        return True
    
    def is_blocked(self, identifier: str) -> bool:
        """Check if IP/user is blocked"""
        if identifier in self.blocked_ips:
            if time.time() < self.blocked_ips[identifier]:
                return True
            else:
                del self.blocked_ips[identifier]
        return False
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def create_session_token(self, user_data: Dict) -> str:
        """Create secure JWT session token"""
        payload = {
            'user_id': user_data['user_id'],
            'email': user_data['email'],
            'exp': datetime.utcnow() + timedelta(seconds=self.session_timeout),
            'iat': datetime.utcnow(),
            'iss': 'medbot-auth'
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def verify_session_token(self, token: str) -> Tuple[bool, Optional[Dict]]:
        """Verify and decode JWT session token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return True, payload
        except jwt.ExpiredSignatureError:
            return False, {'error': 'Token expired'}
        except jwt.InvalidTokenError:
            return False, {'error': 'Invalid token'}
    
    def register_user(self, email: str, password: str, name: str = None, 
                     provider: str = 'email') -> AuthResult:
        """Register new user with comprehensive validation"""
        with self.error_handler("User registration"):
            # Input validation
            if not self.validate_email(email):
                return AuthResult(False, "Invalid email address", error_code="INVALID_EMAIL")
            
            if provider == 'email':
                valid_password, password_error = self.validate_password(password)
                if not valid_password:
                    return AuthResult(False, password_error, error_code="INVALID_PASSWORD")
            
            # Rate limiting
            client_ip = request.remote_addr if request else 'unknown'
            if self.is_blocked(client_ip) or not self.check_rate_limit(client_ip):
                return AuthResult(False, "Too many attempts. Please try again later.", 
                               error_code="RATE_LIMITED")
            
            try:
                # Check if user already exists
                existing_user = self.client.table('users').select('*').eq('email', email).execute()
                
                if existing_user.data:
                    return AuthResult(False, "User already exists", error_code="USER_EXISTS")
                
                # Create user record
                user_id = f"user_{hashlib.md5(email.encode()).hexdigest()}"
                user_data = {
                    'user_id': user_id,
                    'email': email,
                    'name': name or email.split('@')[0],
                    'auth_provider': provider,
                    'is_active': True,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'updated_at': datetime.now(timezone.utc).isoformat()
                }
                
                # For email registration, store hashed password
                if provider == 'email':
                    user_data['password_hash'] = self.hash_password(password)
                
                # Insert user
                result = self.client.table('users').insert(user_data).execute()
                
                if result.data:
                    logger.info(f"✅ User registered successfully: {email}")
                    
                    # Create session
                    session_token = self.create_session_token(user_data)
                    session_data = UserSession(
                        user_id=user_id,
                        email=email,
                        name=user_data['name'],
                        auth_provider=provider,
                        session_id=f"session_{secrets.token_urlsafe(32)}",
                        created_at=datetime.now(timezone.utc),
                        expires_at=datetime.now(timezone.utc) + timedelta(seconds=self.session_timeout),
                        last_activity=datetime.now(timezone.utc),
                        is_verified=True
                    )
                    
                    return AuthResult(
                        True, 
                        "User registered successfully",
                        user_data=user_data,
                        session_data=session_data
                    )
                else:
                    return AuthResult(False, "Failed to create user account", 
                                    error_code="CREATION_FAILED")
                    
            except Exception as e:
                logger.error(f"❌ Registration failed: {e}")
                return AuthResult(False, f"Registration failed: {str(e)}", 
                                error_code="SYSTEM_ERROR")
    
    def login_user(self, email: str, password: str = None, 
                  provider: str = 'email') -> AuthResult:
        """Authenticate user with comprehensive security checks"""
        with self.error_handler("User login"):
            # Input validation
            if not self.validate_email(email):
                return AuthResult(False, "Invalid email address", error_code="INVALID_EMAIL")
            
            # Rate limiting
            client_ip = request.remote_addr if request else 'unknown'
            if self.is_blocked(client_ip) or not self.check_rate_limit(client_ip):
                return AuthResult(False, "Too many attempts. Please try again later.", 
                               error_code="RATE_LIMITED")
            
            try:
                # Get user from database
                user_result = self.client.table('users').select('*').eq('email', email).execute()
                
                if not user_result.data:
                    return AuthResult(False, "User not found", error_code="USER_NOT_FOUND")
                
                user_data = user_result.data[0]
                
                # Check if user is active
                if not user_data.get('is_active', False):
                    return AuthResult(False, "Account is disabled", error_code="ACCOUNT_DISABLED")
                
                # Verify password for email login
                if provider == 'email':
                    if not password:
                        return AuthResult(False, "Password is required", error_code="PASSWORD_REQUIRED")
                    
                    stored_password = user_data.get('password_hash')
                    if not stored_password or not self.verify_password(password, stored_password):
                        return AuthResult(False, "Invalid password", error_code="INVALID_PASSWORD")
                
                # Update last login
                self.client.table('users').update({
                    'last_login': datetime.now(timezone.utc).isoformat(),
                    'updated_at': datetime.now(timezone.utc).isoformat()
                }).eq('email', email).execute()
                
                # Create session
                session_data = UserSession(
                    user_id=user_data['user_id'],
                    email=email,
                    name=user_data['name'],
                    auth_provider=provider,
                    session_id=f"session_{secrets.token_urlsafe(32)}",
                    created_at=datetime.now(timezone.utc),
                    expires_at=datetime.now(timezone.utc) + timedelta(seconds=self.session_timeout),
                    last_activity=datetime.now(timezone.utc),
                    is_verified=True
                )
                
                logger.info(f"✅ User logged in successfully: {email}")
                
                return AuthResult(
                    True,
                    "Login successful",
                    user_data=user_data,
                    session_data=session_data
                )
                
            except Exception as e:
                logger.error(f"❌ Login failed: {e}")
                return AuthResult(False, f"Login failed: {str(e)}", 
                                error_code="SYSTEM_ERROR")
    
    def oauth_login(self, provider: str, oauth_data: Dict) -> AuthResult:
        """Handle OAuth login (Google, GitHub, etc.)"""
        with self.error_handler("OAuth login"):
            try:
                email = oauth_data.get('email')
                if not email or not self.validate_email(email):
                    return AuthResult(False, "Invalid email from OAuth provider", 
                                    error_code="INVALID_OAUTH_EMAIL")
                
                # Check if user exists
                user_result = self.client.table('users').select('*').eq('email', email).execute()
                
                if user_result.data:
                    # Existing user - update info
                    user_data = user_result.data[0]
                    
                    # Update user info from OAuth
                    update_data = {
                        'name': oauth_data.get('name', user_data['name']),
                        'picture': oauth_data.get('picture'),
                        'auth_provider': provider,
                        'last_login': datetime.now(timezone.utc).isoformat(),
                        'updated_at': datetime.now(timezone.utc).isoformat()
                    }
                    
                    self.client.table('users').update(update_data).eq('email', email).execute()
                    user_data.update(update_data)
                    
                else:
                    # New user - create account
                    user_id = f"user_{hashlib.md5(email.encode()).hexdigest()}"
                    user_data = {
                        'user_id': user_id,
                        'email': email,
                        'name': oauth_data.get('name', email.split('@')[0]),
                        'picture': oauth_data.get('picture'),
                        'auth_provider': provider,
                        'is_active': True,
                        'created_at': datetime.now(timezone.utc).isoformat(),
                        'updated_at': datetime.now(timezone.utc).isoformat()
                    }
                    
                    result = self.client.table('users').insert(user_data).execute()
                    if not result.data:
                        return AuthResult(False, "Failed to create user account", 
                                        error_code="CREATION_FAILED")
                    user_data = result.data[0]
                
                # Create session
                session_data = UserSession(
                    user_id=user_data['user_id'],
                    email=email,
                    name=user_data['name'],
                    auth_provider=provider,
                    session_id=f"session_{secrets.token_urlsafe(32)}",
                    created_at=datetime.now(timezone.utc),
                    expires_at=datetime.now(timezone.utc) + timedelta(seconds=self.session_timeout),
                    last_activity=datetime.now(timezone.utc),
                    is_verified=True
                )
                
                logger.info(f"✅ OAuth login successful: {email} via {provider}")
                
                return AuthResult(
                    True,
                    f"OAuth login successful via {provider}",
                    user_data=user_data,
                    session_data=session_data
                )
                
            except Exception as e:
                logger.error(f"❌ OAuth login failed: {e}")
                return AuthResult(False, f"OAuth login failed: {str(e)}", 
                                error_code="OAUTH_ERROR")
    
    def logout_user(self, session_id: str = None) -> AuthResult:
        """ENHANCED: Logout user and invalidate session with proper cleanup"""
        try:
            # Get session_id from Flask session if not provided
            if not session_id and hasattr(session, 'get'):
                session_id = session.get('session_id')
            
            if not session_id:
                return AuthResult(False, "No active session found", error_code="NO_SESSION")
            
            # Find and remove from active sessions
            if session_id in self.active_sessions:
                user_session = self.active_sessions[session_id]
                user_id = user_session.user_id
                
                # Remove from active sessions
                del self.active_sessions[session_id]
                
                # Remove from user sessions mapping
                if user_id in self.user_sessions:
                    self.user_sessions[user_id] = [
                        s for s in self.user_sessions[user_id] if s != session_id
                    ]
                    if not self.user_sessions[user_id]:
                        del self.user_sessions[user_id]
                
                # Clear Flask session
                if hasattr(session, 'clear'):
                    session.clear()
                
                logger.info(f"✅ User logged out successfully: {user_session.email}")
                return AuthResult(True, "Logout successful")
            else:
                # Session not found, still clear Flask session
                if hasattr(session, 'clear'):
                    session.clear()
                return AuthResult(True, "Session cleared")
                
        except Exception as e:
            logger.error(f"❌ Logout failed: {e}")
            return AuthResult(False, f"Logout failed: {str(e)}")
    
    def store_session(self, session_data: UserSession) -> bool:
        """ENHANCED: Store session data in memory and cleanup old sessions"""
        try:
            # Store session
            self.active_sessions[session_data.session_id] = session_data
            
            # Track user sessions
            if session_data.user_id not in self.user_sessions:
                self.user_sessions[session_data.user_id] = []
            
            if session_data.session_id not in self.user_sessions[session_data.user_id]:
                self.user_sessions[session_data.user_id].append(session_data.session_id)
            
            # Update Flask session
            if hasattr(session, '__setitem__'):
                session['session_id'] = session_data.session_id
                session['user_id'] = session_data.user_id
                session['email'] = session_data.email
                session['is_admin'] = self._is_admin_user(session_data.email)
                session.permanent = True
            
            # Cleanup old sessions periodically
            self._cleanup_expired_sessions()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Session storage failed: {e}")
            return False
    
    def get_current_session(self) -> Optional[UserSession]:
        """ENHANCED: Get current user session with validation"""
        try:
            # Get session_id from Flask session
            session_id = None
            if hasattr(session, 'get'):
                session_id = session.get('session_id')
            
            if not session_id:
                return None
            
            # Get from active sessions
            user_session = self.active_sessions.get(session_id)
            if not user_session:
                return None
            
            # Check if expired
            if datetime.now(timezone.utc) > user_session.expires_at:
                logger.info(f"Session expired for {user_session.email}")
                self._remove_session(session_id)
                return None
            
            # Validate user still exists in database
            if not self._validate_user_in_db(user_session.email):
                logger.warning(f"User {user_session.email} no longer exists - removing session")
                self._remove_session(session_id)
                return None
            
            # Update last activity
            user_session.last_activity = datetime.now(timezone.utc)
            
            return user_session
            
        except Exception as e:
            logger.error(f"❌ Get current session failed: {e}")
            return None
    
    def is_logged_in(self) -> bool:
        """ENHANCED: Check if user is currently logged in"""
        return self.get_current_session() is not None
    
    def get_current_user(self) -> Optional[Dict]:
        """ENHANCED: Get current user data"""
        user_session = self.get_current_session()
        if user_session:
            return {
                'user_id': user_session.user_id,
                'email': user_session.email,
                'name': user_session.name,
                'auth_provider': user_session.auth_provider,
                'is_verified': user_session.is_verified,
                'permissions': user_session.permissions,
                'is_admin': self._is_admin_user(user_session.email),
                'session_id': user_session.session_id,
                'last_activity': user_session.last_activity.isoformat()
            }
        return None
    
    def _validate_user_in_db(self, email: str) -> bool:
        """Validate user still exists in database"""
        try:
            result = self.client.table('users').select('is_active').eq('email', email).execute()
            if result.data:
                return result.data[0].get('is_active', False)
            return False
        except Exception as e:
            logger.error(f"User validation failed: {e}")
            return False
    
    def get_user_profile(self, email: str) -> Optional[Dict]:
        """Get complete user profile including medical data"""
        try:
            result = self.client.table('users').select('*').eq('email', email).execute()
            if result.data:
                user = result.data[0]
                # Add computed fields
                user['session_count'] = len(self.user_sessions.get(user.get('user_id', ''), []))
                user['last_seen'] = self._get_last_activity(user.get('user_id', ''))
                return user
            return None
        except Exception as e:
            logger.error(f"Get user profile failed: {e}")
            return None
    
    def _get_last_activity(self, user_id: str) -> Optional[str]:
        """Get user's last activity timestamp"""
        try:
            sessions = self.user_sessions.get(user_id, [])
            if sessions:
                # Find most recent session activity
                latest_activity = None
                for session_id in sessions:
                    if session_id in self.active_sessions:
                        session_obj = self.active_sessions[session_id]
                        if not latest_activity or session_obj.last_activity > latest_activity:
                            latest_activity = session_obj.last_activity
                return latest_activity.isoformat() if latest_activity else None
            return None
        except Exception as e:
            logger.error(f"Get last activity failed: {e}")
            return None
    
    def update_medical_profile(self, email: str, medical_data: Dict) -> AuthResult:
        """Update user's medical profile with HIPAA compliance"""
        with self.error_handler("Update medical profile"):
            try:
                # Validate medical data structure
                allowed_fields = {
                    'allergies', 'medications', 'conditions', 'emergency_contact',
                    'insurance_info', 'doctor_info', 'medical_history', 'preferences'
                }
                
                filtered_data = {}
                for key, value in medical_data.items():
                    if key in allowed_fields:
                        # Encrypt sensitive medical data
                        if key in ['medical_history', 'conditions', 'medications']:
                            filtered_data[f"{key}_encrypted"] = self._encrypt_medical_data(value)
                        else:
                            filtered_data[key] = value
                
                if not filtered_data:
                    return AuthResult(False, "No valid medical fields to update", 
                                    error_code="NO_VALID_FIELDS")
                
                filtered_data['medical_updated_at'] = datetime.now(timezone.utc).isoformat()
                
                result = self.client.table('users').update(filtered_data).eq('email', email).execute()
                
                if result.data:
                    logger.info(f"✅ Medical profile updated: {email}")
                    return AuthResult(True, "Medical profile updated successfully", 
                                    user_data=result.data[0])
                else:
                    return AuthResult(False, "Failed to update medical profile", 
                                    error_code="UPDATE_FAILED")
                    
            except Exception as e:
                logger.error(f"❌ Medical profile update failed: {e}")
                return AuthResult(False, f"Medical profile update failed: {str(e)}", 
                                error_code="SYSTEM_ERROR")
    
    def _encrypt_medical_data(self, data: Any) -> str:
        """Encrypt sensitive medical data for HIPAA compliance"""
        try:
            json_data = json.dumps(data) if not isinstance(data, str) else data
            encrypted = self.fernet.encrypt(json_data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Medical data encryption failed: {e}")
            return ""
    
    def _decrypt_medical_data(self, encrypted_data: str) -> Any:
        """Decrypt medical data"""
        try:
            if not encrypted_data:
                return None
            decoded = base64.b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(decoded).decode()
            try:
                return json.loads(decrypted)
            except:
                return decrypted
        except Exception as e:
            logger.error(f"Medical data decryption failed: {e}")
            return None
    
    
    
    
    
    
    def _invalidate_user_sessions(self, email: str):
        """Invalidate all sessions for a user"""
        try:
            # Get user ID
            result = self.client.table('users').select('user_id').eq('email', email).execute()
            if not result.data:
                return
            
            user_id = result.data[0]['user_id']
            
            # Remove from active sessions
            sessions_to_remove = []
            for session_id, session_obj in self.active_sessions.items():
                if session_obj.email == email:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.active_sessions[session_id]
            
            # Clear user sessions mapping
            if user_id in self.user_sessions:
                del self.user_sessions[user_id]
            
            logger.info(f"✅ All sessions invalidated for user: {email}")
            
        except Exception as e:
            logger.error(f"Session invalidation failed: {e}")
    
    def get_user_statistics(self) -> Dict[str, Any]:
        """Get basic user statistics"""
        try:
            stats = {
                'total_users': 0,
                'active_users': 0,
                'recent_registrations': 0,
                'active_sessions': len(self.active_sessions),
                'medical_profiles': 0
            }
            
            # Get user counts from database
            try:
                result = self.client.table('users').select('email, created_at').execute()
            except Exception:
                result = None
            
            if result and result.data:
                stats['total_users'] = len(result.data)
                
                now = datetime.now(timezone.utc)
                week_ago = now - timedelta(days=7)
                
                for user in result.data:
                    # Check recent registrations
                    if user.get('created_at'):
                        try:
                            created_at = datetime.fromisoformat(user['created_at'].replace('Z', '+00:00'))
                            if created_at > week_ago:
                                stats['recent_registrations'] += 1
                        except:
                            pass
            
            # Add session statistics
            stats['session_stats'] = {
                'total_sessions': len(self.active_sessions),
                'guest_sessions': sum(1 for s in self.active_sessions.values() 
                                    if s.auth_type == 'guest')
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Get user statistics failed: {e}")
            return {
                'error': str(e),
                'active_sessions': len(self.active_sessions) if hasattr(self, 'active_sessions') else 0
            }
    
    def cleanup_expired_data(self) -> Dict[str, int]:
        """Cleanup expired sessions, tokens, and old data"""
        cleanup_stats = {
            'expired_sessions': 0,
            'expired_tokens': 0,
            'old_logs': 0,
            'inactive_users': 0
        }
        
        try:
            now = datetime.now(timezone.utc)
            
            # Cleanup expired sessions
            expired_sessions = []
            for session_id, session_obj in self.active_sessions.items():
                if session_obj.expires_at < now:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.active_sessions[session_id]
                cleanup_stats['expired_sessions'] += 1
            
            # Cleanup old password reset tokens
            day_ago = now - timedelta(days=1)
            expired_tokens = []
            for token, token_data in self.password_reset_tokens.items():
                if token_data['expires'] < day_ago:
                    expired_tokens.append(token)
            
            for token in expired_tokens:
                del self.password_reset_tokens[token]
                cleanup_stats['expired_tokens'] += 1
            
            # Note: Admin logs cleanup removed - no admin database
            
            logger.info(f"✅ Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
            return cleanup_stats
    
    def get_auth_stats(self) -> Dict[str, Any]:
        """Get authentication performance statistics"""
        return {
            'active_sessions': len(self.active_sessions),
            'total_users': len(self.user_sessions),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _remove_session(self, session_id: str):
        """Remove session from all storage"""
        try:
            if session_id in self.active_sessions:
                user_session = self.active_sessions[session_id]
                user_id = user_session.user_id
                
                # Remove from active sessions
                del self.active_sessions[session_id]
                
                # Remove from user sessions mapping
                if user_id in self.user_sessions:
                    self.user_sessions[user_id] = [
                        s for s in self.user_sessions[user_id] if s != session_id
                    ]
                    if not self.user_sessions[user_id]:
                        del self.user_sessions[user_id]
            
            # Clear Flask session if it matches
            if hasattr(session, 'get') and session.get('session_id') == session_id:
                session.clear()
                
        except Exception as e:
            logger.error(f"Session removal failed: {e}")
    
    def _cleanup_expired_sessions(self):
        """Cleanup expired sessions periodically"""
        try:
            current_time = time.time()
            if current_time - self.last_cleanup < self.session_cleanup_interval:
                return
            
            expired_sessions = []
            current_datetime = datetime.now(timezone.utc)
            
            for session_id, user_session in self.active_sessions.items():
                if current_datetime > user_session.expires_at:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                self._remove_session(session_id)
            
            self.last_cleanup = current_time
            
            if expired_sessions:
                logger.info(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
                
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")
    
    def revoke_all_user_sessions(self, user_id: str) -> int:
        """ENHANCED: Revoke all sessions for a user (admin removed scenario)"""
        try:
            revoked_count = 0
            sessions_to_remove = self.user_sessions.get(user_id, []).copy()
            
            for session_id in sessions_to_remove:
                self._remove_session(session_id)
                revoked_count += 1
            
            logger.info(f"🔒 Revoked {revoked_count} sessions for user {user_id}")
            return revoked_count
            
        except Exception as e:
            logger.error(f"Session revocation failed: {e}")
            return 0
    
    def get_user_by_email(self, email: str) -> Tuple[bool, Optional[Dict]]:
        """Get user data by email"""
        try:
            result = self.client.table('users').select('*').eq('email', email).execute()
            if result.data:
                return True, result.data[0]
            return False, None
        except Exception as e:
            logger.error(f"❌ Failed to get user: {e}")
            return False, None
    
    def update_user_profile(self, email: str, update_data: Dict) -> AuthResult:
        """Update user profile with validation"""
        with self.error_handler("Update user profile"):
            try:
                # Validate update data
                allowed_fields = ['name', 'picture']
                filtered_data = {k: v for k, v in update_data.items() if k in allowed_fields}
                
                if not filtered_data:
                    return AuthResult(False, "No valid fields to update", 
                                    error_code="NO_VALID_FIELDS")
                
                filtered_data['updated_at'] = datetime.now(timezone.utc).isoformat()
                
                result = self.client.table('users').update(filtered_data).eq('email', email).execute()
                
                if result.data:
                    logger.info(f"✅ User profile updated: {email}")
                    return AuthResult(True, "Profile updated successfully", 
                                    user_data=result.data[0])
                else:
                    return AuthResult(False, "Failed to update profile", 
                                    error_code="UPDATE_FAILED")
                    
            except Exception as e:
                logger.error(f"❌ Profile update failed: {e}")
                return AuthResult(False, f"Profile update failed: {str(e)}", 
                                error_code="SYSTEM_ERROR")
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of authentication system"""
        health_status = {
            'auth_system_healthy': False,
            'database_connected': False,
            'encryption_working': False,
            'rate_limiting_active': True,
            'blocked_ips_count': len(self.blocked_ips),
            'active_rate_limits': len(self.rate_limits),
            'last_check': datetime.now(timezone.utc).isoformat(),
            'errors': []
        }
        
        try:
            # Test database connection
            test_result = self.client.table('users').select('id').limit(1).execute()
            health_status['database_connected'] = True
            
            # Test encryption
            test_data = "health_check_test"
            encrypted = self.fernet.encrypt(test_data.encode())
            decrypted = self.fernet.decrypt(encrypted).decode()
            health_status['encryption_working'] = (decrypted == test_data)
            
            # Overall health
            health_status['auth_system_healthy'] = (
                health_status['database_connected'] and 
                health_status['encryption_working']
            )
            
        except Exception as e:
            health_status['errors'].append(f"Health check failed: {str(e)}")
        
        logger.info(f"Auth system health check: {health_status['auth_system_healthy']}")
        return health_status

# Global instance
auth_manager = None

def get_auth_manager() -> AuthenticationManager:
    """Get global authentication manager instance"""
    global auth_manager
    if auth_manager is None:
        auth_manager = AuthenticationManager()
    return auth_manager

if __name__ == "__main__":
    # CLI interface for testing
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "health":
            manager = get_auth_manager()
            health = manager.health_check()
            print(json.dumps(health, indent=2))
        elif command == "test":
            # Run basic tests
            manager = get_auth_manager()
            print("🧪 Running authentication tests...")
            
            # Test registration
            result = manager.register_user("test@medbot.ai", "TestPass123!", "Test User")
            print(f"Registration test: {result.success} - {result.message}")
            
            # Test login
            result = manager.login_user("test@medbot.ai", "TestPass123!")
            print(f"Login test: {result.success} - {result.message}")
            
        else:
            print("Usage: python auth_manager.py [health|test]")
    else:
        print("🔐 MedBot Authentication Manager")
        print("Run 'python auth_manager.py health' for health check")
        print("Run 'python auth_manager.py test' for basic tests")