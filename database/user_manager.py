"""
Smart Authentication Manager.
Replaces the old UnifiedUserManager with intelligent conflict resolution.

Manages authentication across Flask Sessions, Redis Cache, and Supabase Database.
Provides automatic Redis/Supabase conflict resolution, session synchronization,
graceful degradation, and TrOCR authentication error isolation.
"""
import json
import time
import hashlib
import logging
import uuid
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
from dataclasses import dataclass
from fastapi import Request
from .sync_manager import SmartCacheSyncManager

logger = logging.getLogger(__name__)

class AuthState(Enum):
    """Authentication states"""
    NOT_AUTHENTICATED = "not_authenticated"
    AUTHENTICATED = "authenticated"
    ADMIN = "admin"
    GUEST = "guest"
    BLOCKED = "blocked"
    SESSION_EXPIRED = "session_expired"

class AuthSource(Enum):
    """Authentication data sources"""
    FLASK_SESSION = "flask_session"
    REDIS_CACHE = "redis_cache"
    SUPABASE_DB = "supabase_db"
    CONFLICT_DETECTED = "conflict_detected"

@dataclass
class AuthContext:
    """Authentication context container"""
    user_id: str
    user_email: str
    user_name: str = ""
    auth_provider: str = "unknown"
    auth_state: AuthState = AuthState.NOT_AUTHENTICATED
    auth_source: AuthSource = AuthSource.FLASK_SESSION
    last_validated: datetime = None
    session_data: Dict = None
    is_guest: bool = False
    is_admin: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'user_id': self.user_id,
            'user_email': self.user_email,
            'user_name': self.user_name,
            'auth_provider': self.auth_provider,
            'auth_state': self.auth_state.value,
            'auth_source': self.auth_source.value,
            'last_validated': self.last_validated.isoformat() if self.last_validated else None,
            'is_guest': self.is_guest,
            'is_admin': self.is_admin,
            'session_data': self.session_data or {}
        }

class SmartAuthenticationManager:
    """
    Smart Authentication Manager — replaces UnifiedUserManager.

    Manages authentication across Flask Sessions (primary), Redis Cache (fast lookups),
    and Supabase Database (persistent storage). Resolves conflicts automatically,
    provides a single authentication flow, and isolates TrOCR errors from auth.
    """
    
    def __init__(self, supabase_url=None, supabase_key=None, redis_client=None, 
                 session_timeout_minutes=60, validation_interval_minutes=30):
        self.redis_client = redis_client
        self.supabase_client = None
        self.sync_manager = None
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.validation_interval = timedelta(minutes=validation_interval_minutes)
        
        # Initialize Supabase client if credentials available
        if supabase_url and supabase_key:
            try:
                from supabase import create_client
                self.supabase_client = create_client(supabase_url, supabase_key)
                logger.info("✅ Supabase client initialized for Smart Authentication")
                
                # Initialize enhanced sync manager 
                self.sync_manager = SmartCacheSyncManager(
                    redis_client=redis_client,
                    supabase_client=self.supabase_client,
                    ttl_seconds=86400  # 24 hours
                )
                logger.info("🔄 Enhanced sync manager initialized")
                
            except Exception as e:
                logger.warning(f"⚠️ Supabase initialization failed: {e}")
        
        # Thread-safe locks for concurrent operations
        self._cache_lock = threading.RLock()
        self._session_lock = threading.RLock()
        
        # Performance and error tracking
        self.metrics = {
            'auth_checks': 0,
            'successful_auths': 0,
            'failed_auths': 0,
            'conflicts_resolved': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_calls': 0,
            'trocr_errors_isolated': 0
        }
        
        logger.info("🧠 Smart Authentication Manager fully initialized")
    
    # ============ MAIN AUTHENTICATION METHODS ============
    
    def authenticate_user(self, email: str, user_data: Dict, auth_provider: str = 'unknown') -> Optional[Dict]:
        """
        🔐 SMART AUTHENTICATION - Replaces old authenticate_user method
        
        Main authentication entry point with comprehensive conflict resolution
        """
        try:
            self.metrics['auth_checks'] += 1
            
            with self._session_lock:
                logger.info(f"🔐 Smart authentication started: {email}")
                
                # Generate consistent user ID
                user_id = self.generate_user_id(email)
                
                # Create authentication context
                auth_context = AuthContext(
                    user_id=user_id,
                    user_email=email,
                    user_name=user_data.get('name', ''),
                    auth_provider=auth_provider,
                    auth_state=AuthState.AUTHENTICATED,
                    last_validated=datetime.now(),
                    session_data=user_data,
                    is_guest=auth_provider == 'guest',
                    is_admin=self._is_admin_email(email)
                )
                
                # Step 1: Set Flask session (primary authentication)
                self._set_flask_session(auth_context)
                
                # Step 2: Cache in Redis for performance (non-blocking)
                self._cache_auth_context(auth_context)
                
                # Step 3: Sync to Supabase using conflict-safe sync manager
                if self.sync_manager and not auth_context.is_guest:
                    try:
                        sync_success = self.sync_manager.save_user_safe(email, user_data, auth_provider)
                        if sync_success:
                            logger.info(f"🔄 User synced via enhanced sync manager: {email}")
                    except Exception as sync_error:
                        # Sync errors don't block authentication
                        logger.warning(f"⚠️ Sync failed but authentication continues: {sync_error}")
                
                self.metrics['successful_auths'] += 1
                logger.info(f"✅ Smart authentication successful: {email}")
                
                # Return data in old format for compatibility
                return {
                    'user_id': user_id,
                    'email': email,
                    'session_data': auth_context.to_dict(),
                    'authenticated': True
                }
                
        except Exception as e:
            self.metrics['failed_auths'] += 1
            logger.error(f"❌ Smart authentication failed for {email}: {e}")
            return None
    
    def get_current_auth(self) -> Optional[AuthContext]:
        """
        🎯 GET CURRENT AUTHENTICATION - Smart conflict resolution
        
        Intelligent authentication check with automatic conflict resolution
        """
        try:
            self.metrics['auth_checks'] += 1
            
            with self._session_lock:
                # Step 1: Check Flask session first (fastest)
                flask_auth = self._get_flask_auth()
                if not flask_auth:
                    logger.debug("🔍 No Flask session found")
                    return None
                
                # Step 2: Check if session needs validation
                if self._needs_validation(flask_auth):
                    logger.info(f"🔍 Session needs validation: {flask_auth.user_email}")
                    return self._validate_and_resolve(flask_auth)
                
                # Step 3: Session is valid
                self.metrics['cache_hits'] += 1
                logger.debug(f"✅ Valid session found: {flask_auth.user_email}")
                return flask_auth
                
        except Exception as e:
            logger.error(f"❌ Get current auth failed: {e}")
            return None
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """
        👤 GET USER BY EMAIL - Enhanced with conflict resolution
        
        Replaces old get_user_by_email with smart conflict handling
        """
        try:
            # Use enhanced sync manager if available
            if self.sync_manager:
                user_data = self.sync_manager.get_user_safe(email)
                if user_data:
                    logger.info(f"👤 User found via sync manager: {email}")
                    return user_data
            
            # Fallback to direct lookup
            user_id = self.generate_user_id(email)
            
            # Try Redis first
            if self.redis_client:
                redis_key = f"medai:user:{user_id}"
                try:
                    cached_data = self.redis_client.get(redis_key)
                    if cached_data:
                        user_data = json.loads(cached_data)
                        logger.info(f"👤 User found in Redis: {email}")
                        return user_data
                except Exception as e:
                    logger.warning(f"Redis lookup failed: {e}")
            
            # Try Supabase
            if self.supabase_client:
                try:
                    result = self.supabase_client.table('users').select('*').eq('email', email).execute()
                    if result.data:
                        user_data = result.data[0]
                        logger.info(f"👤 User found in Supabase: {email}")
                        return user_data
                except Exception as e:
                    logger.warning(f"Supabase lookup failed: {e}")
            
            logger.info(f"🔍 User not found: {email}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Get user by email failed: {e}")
            return None
    
    def validate_session(self, user_email: str) -> Tuple[bool, str]:
        """
        🔍 VALIDATE SESSION - Comprehensive validation with conflict resolution
        """
        try:
            self.metrics['validation_calls'] += 1
            
            # Check if user is blocked first
            is_blocked, block_reason = self.is_user_blocked(user_email)
            if is_blocked:
                return False, f"User blocked: {block_reason}"
            
            # Get current authentication context
            current_auth = self.get_current_auth()
            if not current_auth or current_auth.user_email != user_email:
                return False, "Session not found or email mismatch"
            
            # Check session expiry
            if self._is_session_expired(current_auth):
                return False, "Session expired"
            
            # Validate against Supabase (source of truth) if available
            if self.supabase_client:
                try:
                    result = self.supabase_client.table('users').select('*').eq('email', user_email).execute()
                    if not result.data:
                        logger.warning(f"⚠️ User not found in Supabase: {user_email}")
                        return False, "User account not found in database"
                except Exception as e:
                    # Database errors don't invalidate session in degraded mode
                    logger.warning(f"⚠️ Supabase validation failed, continuing in degraded mode: {e}")
            
            return True, "Session valid"
            
        except Exception as e:
            logger.error(f"❌ Session validation failed: {e}")
            return False, f"Validation error: {str(e)}"
    
    def logout_user(self, user_email: str = None) -> bool:
        """
        🚪 SMART LOGOUT - Comprehensive session cleanup
        """
        return self.logout(user_email)
    
    def logout(self, user_email: str = None) -> bool:
        """
        🚪 SMART LOGOUT - Comprehensive session cleanup
        """
        try:
            with self._session_lock:
                current_email = user_email or session.get('user_email')
                if not current_email:
                    logger.warning("⚠️ No user to logout")
                    return False
                
                user_id = self.generate_user_id(current_email)
                
                # Step 1: Clear Flask session
                session.clear()
                logger.info(f"🧹 Flask session cleared: {current_email}")
                
                # Step 2: Clear Redis cache (non-blocking)
                if self.redis_client:
                    try:
                        session_key = f"medai:auth:session:{user_id}"
                        user_key = f"medai:user:{user_id}"
                        self.redis_client.delete(session_key, user_key)
                        logger.info(f"🧹 Redis cache cleared: {current_email}")
                    except Exception as e:
                        logger.warning(f"⚠️ Redis clear failed (non-critical): {e}")
                
                # Step 3: Update Supabase activity (non-blocking)
                if self.supabase_client:
                    try:
                        self.supabase_client.table('users').update({
                            'last_activity': datetime.now().isoformat(),
                            'is_active': False
                        }).eq('email', current_email).execute()
                        logger.info(f"📊 Supabase activity updated: {current_email}")
                    except Exception as e:
                        logger.warning(f"⚠️ Supabase update failed (non-critical): {e}")
                
                logger.info(f"✅ Smart logout completed: {current_email}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Smart logout failed: {e}")
            return False
    
    # ============ USER MANAGEMENT METHODS (COMPATIBILITY) ============
    
    def generate_user_id(self, email: str) -> str:
        """Generate consistent user ID from email"""
        return hashlib.md5(email.encode()).hexdigest()
    
    def get_user_patient_profile(self, email: str) -> Optional[Dict]:
        """Get user's patient profile from Supabase"""
        try:
            if not self.supabase_client:
                return None
                
            result = self.supabase_client.table('patient_profiles').select('*').eq('email', email).execute()
            if result.data:
                profile = result.data[0]
                logger.info(f"🏥 Patient profile found: {email}")
                return profile
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Patient profile retrieval failed: {e}")
            return None
    
    def save_patient_profile(self, user_email: str, patient_data: Dict) -> bool:
        """Save patient profile to Supabase"""
        try:
            if not self.supabase_client:
                logger.warning("⚠️ No Supabase client available for patient profile storage")
                return False
            
            if not user_email or not patient_data:
                logger.error("❌ Missing user email or patient data")
                return False
            
            # Prepare patient data
            profile_data = {
                'email': user_email,
                'name': patient_data.get('name', ''),
                'age': patient_data.get('age'),
                'gender': patient_data.get('gender', ''),
                'medical_conditions': patient_data.get('medical_conditions', []),
                'medications': patient_data.get('medications', []),
                'allergies': patient_data.get('allergies', []),
                'emergency_contact': patient_data.get('emergency_contact', {}),
                'insurance_info': patient_data.get('insurance_info', {}),
                'updated_at': datetime.now().isoformat(),
                'profile_completeness': self._calculate_profile_completeness(patient_data)
            }
            
            result = self.supabase_client.table('patient_profiles').upsert(
                profile_data, 
                on_conflict='email'
            ).execute()
            
            logger.info(f"🏥 Patient profile saved: {user_email}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Patient profile save failed: {e}")
            return False
    
    def get_user_chat_history(self, email: str, limit: int = 20) -> List[Dict]:
        """Get user's chat history from Supabase"""
        try:
            if not self.supabase_client:
                return []
            
            # Get patient profile first
            profile = self.get_user_patient_profile(email)
            if not profile:
                return []
            
            patient_id = profile['id']
            
            result = self.supabase_client.table('chat_history')\
                .select('*')\
                .eq('patient_id', patient_id)\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()
            
            if result.data:
                logger.info(f"💬 Chat history found: {len(result.data)} messages for {email}")
                return result.data
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Chat history retrieval failed: {e}")
            return []
    
    def save_chat_message(self, email: str, user_message: str, ai_response: str, session_id: str = None) -> bool:
        """Save chat message to Supabase"""
        try:
            if not self.supabase_client:
                return False
            
            # Get patient profile
            profile = self.get_user_patient_profile(email)
            if not profile:
                logger.warning(f"⚠️ No patient profile found for chat storage: {email}")
                return False
            
            patient_id = profile['id']
            
            chat_data = {
                'patient_id': patient_id,
                'user_message': user_message,
                'ai_response': ai_response,
                'session_id': session_id or str(uuid.uuid4()),
                'created_at': datetime.now().isoformat(),
                'message_metadata': {
                    'user_email': email,
                    'response_length': len(ai_response),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            result = self.supabase_client.table('chat_history').insert(chat_data).execute()
            logger.info(f"💾 Chat message saved: {email}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Chat message save failed: {e}")
            return False
    
    def delete_user_safely(self, email: str) -> bool:
        """
        🗑️ SAFE USER DELETION - Enhanced with conflict resolution
        """
        try:
            user_id = self.generate_user_id(email)
            
            # Step 1: Delete from Supabase (source of truth)
            delete_success = False
            if self.supabase_client:
                try:
                    # Delete user and profile
                    user_result = self.supabase_client.table('users').delete().eq('email', email).execute()
                    profile_result = self.supabase_client.table('patient_profiles').delete().eq('email', email).execute()
                    
                    delete_success = True
                    logger.info(f"🗑️ User deleted from Supabase: {email}")
                    
                except Exception as e:
                    logger.error(f"❌ Supabase user deletion failed: {e}")
                    return False
            
            # Step 2: Clear all caches and sessions
            if delete_success:
                # Clear Flask session if it's the active user
                if session.get('user_email') == email:
                    session.clear()
                    logger.info(f"🧹 Active Flask session cleared: {email}")
                
                # Clear Redis cache
                if self.redis_client:
                    try:
                        keys_to_delete = [
                            f"medai:auth:session:{user_id}",
                            f"medai:user:{user_id}",
                            f"medai:blocked:{user_id}"
                        ]
                        self.redis_client.delete(*keys_to_delete)
                        logger.info(f"🧹 Redis cache cleared for deleted user: {email}")
                    except Exception as e:
                        logger.warning(f"⚠️ Redis cache clear failed: {e}")
                
                # Invalidate sync manager cache if available
                if self.sync_manager:
                    try:
                        self.sync_manager.invalidate_user_cache(email)
                        logger.info(f"🔄 Sync manager cache invalidated: {email}")
                    except Exception as e:
                        logger.warning(f"⚠️ Sync cache invalidation failed: {e}")
            
            logger.info(f"✅ User safely deleted: {email}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Safe user deletion failed: {e}")
            return False
    
    # ============ SECURITY AND BLOCKING METHODS ============
    
    def block_user(self, user_email: str, reason: str = "Security block") -> bool:
        """🚫 BLOCK USER - Enhanced security function"""
        try:
            user_id = self.generate_user_id(user_email)
            
            # Set block in Flask session if it's the current user
            if session.get('user_email') == user_email:
                session['auth_blocked'] = True
                session['block_reason'] = reason
                session['blocked_at'] = datetime.now().isoformat()
            
            # Cache block in Redis
            if self.redis_client:
                try:
                    block_key = f"medai:blocked:{user_id}"
                    self.redis_client.setex(block_key, 3600, json.dumps({
                        'blocked_at': datetime.now().isoformat(),
                        'reason': reason,
                        'user_email': user_email
                    }))
                except Exception as e:
                    logger.warning(f"⚠️ Redis block failed: {e}")
            
            logger.warning(f"🚫 User blocked: {user_email} - {reason}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Block user failed: {e}")
            return False
    
    def is_user_blocked(self, user_email: str) -> Tuple[bool, str]:
        """Check if user is blocked"""
        try:
            user_id = self.generate_user_id(user_email)
            
            # Check Flask session
            if session.get('user_email') == user_email and session.get('auth_blocked'):
                reason = session.get('block_reason', 'Security block')
                return True, reason
            
            # Check Redis cache
            if self.redis_client:
                try:
                    block_key = f"medai:blocked:{user_id}"
                    block_data = self.redis_client.get(block_key)
                    if block_data:
                        block_info = json.loads(block_data)
                        return True, block_info.get('reason', 'Security block')
                except Exception as e:
                    logger.warning(f"⚠️ Redis block check failed: {e}")
            
            return False, ""
            
        except Exception as e:
            logger.error(f"❌ Block check failed: {e}")
            return False, ""
    
    # ============ HEALTH AND MONITORING ============
    
    def get_sync_health(self) -> Dict:
        """Get authentication system health status"""
        health = {
            'flask_session_active': bool(session.get('authenticated')),
            'redis_healthy': False,
            'supabase_healthy': False,
            'sync_manager_healthy': False,
            'current_user': session.get('user_email', 'none'),
            'metrics': self.metrics.copy(),
            'overall_status': 'unknown'
        }
        
        # Test Redis
        if self.redis_client:
            try:
                self.redis_client.ping()
                health['redis_healthy'] = True
            except Exception as e:
                logger.warning(f"⚠️ Redis health check failed: {e}")
        
        # Test Supabase
        if self.supabase_client:
            try:
                result = self.supabase_client.table('users').select('email').limit(1).execute()
                health['supabase_healthy'] = True
            except Exception as e:
                logger.warning(f"⚠️ Supabase health check failed: {e}")
        
        # Test Sync Manager
        if self.sync_manager:
            try:
                sync_health = self.sync_manager.health_check()
                health['sync_manager_healthy'] = sync_health.get('sync_status') in ['fully_operational', 'degraded_no_cache']
            except Exception as e:
                logger.warning(f"⚠️ Sync manager health check failed: {e}")
        
        # Overall status
        if all([health['flask_session_active'], health['redis_healthy'], health['supabase_healthy']]):
            health['overall_status'] = 'fully_operational'
        elif health['flask_session_active'] and health['supabase_healthy']:
            health['overall_status'] = 'operational_no_cache'
        elif health['flask_session_active']:
            health['overall_status'] = 'degraded_session_only'
        else:
            health['overall_status'] = 'not_authenticated'
        
        return health
    
    # ============ COMPATIBILITY METHODS ============
    
    def get_user_settings(self, email: str) -> Optional[Dict]:
        """Get user settings - maintains compatibility"""
        try:
            if not email:
                return None
                
            user_id = self.generate_user_id(email)
            settings_key = f"user_settings:{user_id}"
            
            # Try Redis first
            if self.redis_client:
                try:
                    cached_settings = self.redis_client.get(settings_key)
                    if cached_settings:
                        return json.loads(cached_settings)
                except Exception as e:
                    logger.warning(f"⚠️ Redis get_user_settings failed: {e}")
            
            # Try Supabase
            if self.supabase_client:
                try:
                    response = self.supabase_client.table('user_settings').select('*').eq('user_email', email).execute()
                    if response.data:
                        settings = response.data[0].get('settings', {})
                        
                        # Cache in Redis
                        if self.redis_client:
                            try:
                                self.redis_client.setex(settings_key, 3600, json.dumps(settings))
                            except Exception:
                                pass
                        
                        return settings
                except Exception as e:
                    logger.warning(f"⚠️ Supabase get_user_settings failed: {e}")
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ get_user_settings failed: {e}")
            return None
    
    def update_user_settings(self, email: str, new_settings: Dict) -> bool:
        """Update user settings - maintains compatibility"""
        try:
            if not email or not new_settings:
                return False
                
            user_id = self.generate_user_id(email)
            settings_key = f"user_settings:{user_id}"
            
            # Update Supabase first
            if self.supabase_client:
                try:
                    self.supabase_client.table('user_settings').upsert({
                        'user_email': email,
                        'user_id': user_id,
                        'settings': new_settings,
                        'updated_at': datetime.now().isoformat()
                    }).execute()
                except Exception as e:
                    logger.warning(f"⚠️ Supabase update_user_settings failed: {e}")
            
            # Update Redis cache
            if self.redis_client:
                try:
                    self.redis_client.setex(settings_key, 3600, json.dumps(new_settings))
                except Exception as e:
                    logger.warning(f"⚠️ Redis update_user_settings failed: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ update_user_settings failed: {e}")
            return False
    
    def get_recent_users(self, limit: int = 50) -> List[Dict]:
        """Get recent users - maintains compatibility"""
        try:
            if self.supabase_client:
                try:
                    response = self.supabase_client.table('users').select('*').order('created_at', desc=True).limit(limit).execute()
                    return response.data if response.data else []
                except Exception as e:
                    logger.warning(f"⚠️ Supabase get_recent_users failed: {e}")
            
            return []
            
        except Exception as e:
            logger.error(f"❌ get_recent_users failed: {e}")
            return []
    
    def save_medical_image_analysis(self, user_email: str, medical_record: Dict) -> bool:
        """Save medical image analysis record - maintains compatibility"""
        try:
            if not user_email or not medical_record:
                return False
                
            user_id = self.generate_user_id(user_email)
            record_id = str(uuid.uuid4())
            
            medical_record.update({
                'record_id': record_id,
                'user_email': user_email,
                'user_id': user_id,
                'created_at': datetime.now().isoformat()
            })
            
            # Save to Supabase
            if self.supabase_client:
                try:
                    self.supabase_client.table('medical_records').insert(medical_record).execute()
                except Exception as e:
                    logger.warning(f"⚠️ Supabase save_medical_record failed: {e}")
            
            # Cache recent record in Redis
            if self.redis_client:
                try:
                    cache_key = f"medical_record:{user_id}:latest"
                    self.redis_client.setex(cache_key, 7200, json.dumps(medical_record, default=str))
                except Exception as e:
                    logger.warning(f"⚠️ Redis cache_medical_record failed: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ save_medical_image_analysis failed: {e}")
            return False
    
    # ============ PRIVATE HELPER METHODS ============
    
    def _is_admin_email(self, email: str) -> bool:
        """Check if email is admin"""
        admin_emails = ['admin@medai.com', 'siddhu@medai.com']  # Configure as needed
        return email.lower() in [e.lower() for e in admin_emails]
    
    def _set_flask_session(self, auth_context: AuthContext):
        """Set Flask session with auth context"""
        session.permanent = True
        session['authenticated'] = auth_context.auth_state != AuthState.NOT_AUTHENTICATED
        session['user_id'] = auth_context.user_id
        session['user_email'] = auth_context.user_email
        session['user_name'] = auth_context.user_name
        session['auth_provider'] = auth_context.auth_provider
        session['login_time'] = datetime.now().isoformat()
        session['last_validated'] = auth_context.last_validated.isoformat()
        session['is_guest'] = auth_context.is_guest
        session['is_admin'] = auth_context.is_admin
        
        logger.debug(f"🎯 Flask session set: {auth_context.user_email}")
    
    def _get_flask_auth(self) -> Optional[AuthContext]:
        """Get auth context from Flask session"""
        if not session.get('authenticated'):
            return None
        
        try:
            last_validated_str = session.get('last_validated')
            last_validated = datetime.fromisoformat(last_validated_str) if last_validated_str else None
            
            return AuthContext(
                user_id=session.get('user_id', ''),
                user_email=session.get('user_email', ''),
                user_name=session.get('user_name', ''),
                auth_provider=session.get('auth_provider', 'unknown'),
                auth_state=AuthState.ADMIN if session.get('is_admin') else 
                          AuthState.GUEST if session.get('is_guest') else AuthState.AUTHENTICATED,
                auth_source=AuthSource.FLASK_SESSION,
                last_validated=last_validated,
                is_guest=session.get('is_guest', False),
                is_admin=session.get('is_admin', False)
            )
        except Exception as e:
            logger.error(f"❌ Flask auth parsing failed: {e}")
            return None
    
    def _cache_auth_context(self, auth_context: AuthContext):
        """Cache auth context in Redis"""
        if not self.redis_client:
            return False
        
        try:
            with self._cache_lock:
                session_key = f"medai:auth:session:{auth_context.user_id}"
                cache_data = {
                    **auth_context.to_dict(),
                    'cached_at': datetime.now().isoformat()
                }
                
                self.redis_client.setex(
                    session_key,
                    int(self.session_timeout.total_seconds()),
                    json.dumps(cache_data, default=str)
                )
                
                logger.debug(f"💾 Auth context cached: {auth_context.user_email}")
                return True
                
        except Exception as e:
            # Cache failures don't block authentication
            logger.warning(f"⚠️ Cache auth context failed: {e}")
            return False
    
    def _needs_validation(self, auth_context: AuthContext) -> bool:
        """Check if auth context needs validation"""
        if not auth_context.last_validated:
            return True
        
        time_since_validation = datetime.now() - auth_context.last_validated
        return time_since_validation > self.validation_interval
    
    def _is_session_expired(self, auth_context: AuthContext) -> bool:
        """Check if session is expired"""
        if not auth_context.last_validated:
            return True
        
        time_since_validation = datetime.now() - auth_context.last_validated
        return time_since_validation > self.session_timeout
    
    def _validate_and_resolve(self, flask_auth: AuthContext) -> Optional[AuthContext]:
        """🔍 VALIDATE AND RESOLVE CONFLICTS - Critical method"""
        try:
            self.metrics['conflicts_resolved'] += 1
            
            user_email = flask_auth.user_email
            logger.info(f"🔍 Validating and resolving conflicts for: {user_email}")
            
            # Check if user is blocked
            is_blocked, block_reason = self.is_user_blocked(user_email)
            if is_blocked:
                logger.warning(f"🚫 User blocked during validation: {user_email} - {block_reason}")
                self.logout(user_email)
                return None
            
            # Validate session
            is_valid, message = self.validate_session(user_email)
            if not is_valid:
                logger.warning(f"❌ Session validation failed: {user_email} - {message}")
                self.logout(user_email)
                return None
            
            # Update validation timestamp
            flask_auth.last_validated = datetime.now()
            session['last_validated'] = flask_auth.last_validated.isoformat()
            
            # Re-cache with updated validation
            self._cache_auth_context(flask_auth)
            
            logger.info(f"✅ Session validated and conflicts resolved: {user_email}")
            return flask_auth
            
        except Exception as e:
            logger.error(f"❌ Validation and resolution failed: {e}")
            return None
    
    def _calculate_profile_completeness(self, patient_data: Dict) -> int:
        """Calculate profile completeness percentage"""
        required_fields = ['name', 'age', 'gender']
        optional_fields = ['medical_conditions', 'medications', 'allergies', 'emergency_contact']
        
        completed_required = sum(1 for field in required_fields if patient_data.get(field))
        completed_optional = sum(1 for field in optional_fields if patient_data.get(field))
        
        total_possible = len(required_fields) + len(optional_fields)
        completed_total = completed_required + completed_optional
        
        return int((completed_total / total_possible) * 100)

# BACKWARD COMPATIBILITY - Alias for old UnifiedUserManager
UnifiedUserManager = SmartAuthenticationManager

# Module-level convenience functions for easy access
def create_smart_auth_manager(supabase_url=None, supabase_key=None, redis_client=None):
    """Factory function to create SmartAuthenticationManager"""
    return SmartAuthenticationManager(
        supabase_url=supabase_url,
        supabase_key=supabase_key, 
        redis_client=redis_client
    )