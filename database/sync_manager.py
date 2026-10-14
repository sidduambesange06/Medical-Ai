"""
Smart Cache Synchronization Manager - Solves Redis/Supabase conflicts
Professional solution for cache invalidation and data consistency
"""
import json
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Cache strategies for different scenarios"""
    WRITE_THROUGH = "write_through"  # Write to both Redis and Supabase
    WRITE_BEHIND = "write_behind"    # Write to Redis first, sync to Supabase async
    CACHE_ASIDE = "cache_aside"      # Read from cache, fallback to DB
    REFRESH_AHEAD = "refresh_ahead"  # Proactively refresh cache before expiry

class DataSource(Enum):
    """Track data source for consistency"""
    REDIS_ONLY = "redis"
    SUPABASE_ONLY = "supabase" 
    BOTH_SYNCED = "synced"
    CONFLICT_DETECTED = "conflict"

class SmartCacheSyncManager:
    """
    Professional cache synchronization manager that prevents Redis/Supabase conflicts
    
    Key Features:
    - Cache invalidation on Supabase changes
    - Conflict detection and resolution
    - Data version tracking
    - Automatic cache refresh
    - Graceful degradation
    """
    
    def __init__(self, redis_client, supabase_client, ttl_seconds=3600):
        self.redis = redis_client
        self.supabase = supabase_client
        self.ttl = ttl_seconds
        self.version_prefix = "v2:"  # Version for cache invalidation
        
    def _generate_cache_key(self, key_type: str, identifier: str) -> str:
        """Generate versioned cache key"""
        return f"medai:{self.version_prefix}{key_type}:{identifier}"
    
    def _generate_hash_key(self, data: Dict) -> str:
        """Generate hash for data consistency checking"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def _add_cache_metadata(self, data: Dict) -> Dict:
        """Add metadata for cache management"""
        if not isinstance(data, dict):
            data = {"value": data}
            
        return {
            **data,
            "_cache_meta": {
                "cached_at": datetime.now().isoformat(),
                "data_hash": self._generate_hash_key(data),
                "source": DataSource.BOTH_SYNCED.value,
                "version": self.version_prefix.rstrip(':')
            }
        }
    
    def get_user_safe(self, email: str) -> Optional[Dict]:
        """
        PROFESSIONAL SOLUTION: Get user with conflict detection and resolution
        """
        try:
            user_id = hashlib.md5(email.encode()).hexdigest()
            cache_key = self._generate_cache_key("user", user_id)
            
            # Step 1: Try Redis cache first
            cached_data = None
            if self.redis:
                try:
                    cached_raw = self.redis.get(cache_key)
                    if cached_raw:
                        cached_data = json.loads(cached_raw)
                        logger.info(f"🎯 Cache HIT for user: {email}")
                except Exception as e:
                    logger.warning(f"Redis read error: {e}")
            
            # Step 2: Get from Supabase (source of truth)
            supabase_data = None
            if self.supabase:
                try:
                    result = self.supabase.table('users').select('*').eq('email', email).execute()
                    if result.data and len(result.data) > 0:
                        supabase_data = result.data[0]
                        logger.info(f"📊 Supabase data found for: {email}")
                except Exception as e:
                    logger.warning(f"Supabase read error: {e}")
            
            # Step 3: CONFLICT DETECTION & RESOLUTION
            if cached_data and supabase_data:
                return self._resolve_data_conflict(email, cached_data, supabase_data, cache_key)
            
            # Step 4: Cache miss - use available data
            if supabase_data:
                # Cache the fresh data
                self._cache_user_data(cache_key, supabase_data)
                return supabase_data
            
            if cached_data:
                # Only cache data available (degraded mode)
                logger.warning(f"⚠️  DEGRADED MODE: Using cache-only data for {email}")
                return cached_data
            
            # Step 5: No data found
            logger.info(f"🔍 User not found: {email}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Smart cache get failed: {e}")
            return None
    
    def _resolve_data_conflict(self, email: str, cached_data: Dict, supabase_data: Dict, cache_key: str) -> Dict:
        """
        INTELLIGENT CONFLICT RESOLUTION
        """
        try:
            # Extract metadata
            cache_meta = cached_data.get('_cache_meta', {})
            cached_hash = cache_meta.get('data_hash', '')
            cached_time = cache_meta.get('cached_at', '')
            
            # Generate hash for current Supabase data
            supabase_hash = self._generate_hash_key(supabase_data)
            
            # Check for conflicts
            if cached_hash == supabase_hash:
                # ✅ NO CONFLICT: Data is identical
                logger.info(f"✅ Data consistent for {email}")
                return cached_data
            
            # 🚨 CONFLICT DETECTED: Data differs
            logger.warning(f"🚨 CONFLICT DETECTED for {email}")
            logger.warning(f"   Cache hash: {cached_hash}")
            logger.warning(f"   DB hash: {supabase_hash}")
            
            # RESOLUTION STRATEGY: Supabase wins (source of truth)
            logger.info(f"🔄 RESOLVING: Using Supabase data (source of truth)")
            
            # Update cache with fresh data
            self._cache_user_data(cache_key, supabase_data)
            
            # Log conflict resolution
            self._log_conflict_resolution(email, cached_data, supabase_data)
            
            return supabase_data
            
        except Exception as e:
            logger.error(f"❌ Conflict resolution failed: {e}")
            # Fallback to Supabase data
            return supabase_data
    
    def _cache_user_data(self, cache_key: str, user_data: Dict):
        """Cache user data with metadata"""
        if not self.redis:
            return False
            
        try:
            # Add cache metadata
            enriched_data = self._add_cache_metadata(user_data)
            
            # Cache with TTL
            self.redis.setex(
                cache_key, 
                self.ttl, 
                json.dumps(enriched_data, default=str)
            )
            
            logger.info(f"💾 User data cached successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache storage failed: {e}")
            return False
    
    def save_user_safe(self, email: str, user_data: Dict, auth_provider: str = 'google') -> bool:
        """
        PROFESSIONAL SOLUTION: Save user with cache invalidation
        """
        try:
            user_id = hashlib.md5(email.encode()).hexdigest()
            cache_key = self._generate_cache_key("user", user_id)
            
            # Prepare user data
            save_data = {
                'email': email,
                'name': user_data.get('name', ''),
                'picture': user_data.get('picture', ''),
                'auth_provider': auth_provider,
                'last_login': datetime.now().isoformat(),
                'is_active': True,
                'user_id': user_id,
                'updated_at': datetime.now().isoformat()
            }
            
            # STRATEGY: Write-Through (write to both immediately)
            success_supabase = False
            success_redis = False
            
            # 1. Write to Supabase (source of truth) FIRST
            if self.supabase:
                try:
                    result = self.supabase.table('users').upsert(
                        save_data, 
                        on_conflict='email'
                    ).execute()
                    success_supabase = True
                    logger.info(f"✅ User saved to Supabase: {email}")
                except Exception as e:
                    logger.error(f"❌ Supabase save failed: {e}")
            
            # 2. Update cache ONLY if Supabase succeeded
            if success_supabase and self.redis:
                try:
                    success_redis = self._cache_user_data(cache_key, save_data)
                except Exception as e:
                    logger.error(f"❌ Cache update failed: {e}")
            
            # 3. Return overall success
            if success_supabase:
                logger.info(f"🎉 User save completed: {email} (DB: ✅, Cache: {'✅' if success_redis else '❌'})")
                return True
            else:
                logger.error(f"❌ User save failed: {email}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Smart cache save failed: {e}")
            return False
    
    def invalidate_user_cache(self, email: str) -> bool:
        """
        CACHE INVALIDATION: Remove user from cache when deleted from Supabase
        """
        try:
            user_id = hashlib.md5(email.encode()).hexdigest()
            cache_key = self._generate_cache_key("user", user_id)
            
            if self.redis:
                deleted = self.redis.delete(cache_key)
                if deleted:
                    logger.info(f"🗑️  Cache invalidated for user: {email}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Cache invalidation failed: {e}")
            return False
    
    def _log_conflict_resolution(self, email: str, cached_data: Dict, supabase_data: Dict):
        """Log conflict resolution for monitoring"""
        try:
            conflict_log = {
                "timestamp": datetime.now().isoformat(),
                "email": email,
                "conflict_type": "data_mismatch",
                "resolution": "supabase_wins",
                "cached_fields": list(cached_data.keys()) if cached_data else [],
                "supabase_fields": list(supabase_data.keys()) if supabase_data else []
            }
            
            logger.warning(f"📋 CONFLICT RESOLVED: {json.dumps(conflict_log)}")
            
        except Exception as e:
            logger.error(f"Conflict logging failed: {e}")
    
    def health_check(self) -> Dict:
        """Check cache and database health"""
        status = {
            "redis_healthy": False,
            "supabase_healthy": False,
            "sync_status": "unknown"
        }
        
        # Test Redis
        if self.redis:
            try:
                self.redis.ping()
                status["redis_healthy"] = True
            except:
                pass
        
        # Test Supabase
        if self.supabase:
            try:
                result = self.supabase.table('users').select('email').limit(1).execute()
                status["supabase_healthy"] = True
            except:
                pass
        
        # Overall sync status
        if status["redis_healthy"] and status["supabase_healthy"]:
            status["sync_status"] = "fully_operational"
        elif status["supabase_healthy"]:
            status["sync_status"] = "degraded_no_cache"
        elif status["redis_healthy"]:
            status["sync_status"] = "cache_only"
        else:
            status["sync_status"] = "offline"
        
        return status

# Global sync manager will be initialized in main app
sync_manager = None