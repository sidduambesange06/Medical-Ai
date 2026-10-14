"""
Redis performance manager.
Handles AI response caching, WebSocket connection management, real-time metrics,
file processing queues, and rate limiting. Authentication is handled by Supabase.
"""

import json
import time
import redis
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)

# @dataclass
# class HealthMetrics:
#     """Smartwatch health metrics data structure - DISABLED FOR NOW"""
#     # SMARTWATCH INTEGRATION COMMENTED OUT - NOT USING FOR NOW
#     pass

@dataclass
class AIResponseCache:
    """Cached AI response structure"""
    query_hash: str
    response: str
    timestamp: str
    user_id: str
    confidence_score: float = 0.0

class RedisPerformanceManager:
    """
    HIGH-PERFORMANCE Redis Manager
    Focused on speed, caching, and real-time data
    NO AUTHENTICATION LOGIC - Pure performance
    """
    
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=1):
        """Initialize Redis for performance tasks only"""
        try:
            self.redis_client = redis.Redis(
                host=redis_host, 
                port=redis_port, 
                db=redis_db,  # Different DB from auth
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("🚀 Redis Performance Manager initialized successfully")
            
            # Initialize performance counters
            self.init_performance_counters()
            
        except Exception as e:
            logger.error(f"❌ Redis Performance Manager failed to initialize: {e}")
            self.redis_client = None
    
    def init_performance_counters(self):
        """Initialize performance monitoring counters"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            counters = {
                'ai_cache_hits': 0,
                'ai_cache_misses': 0,
                'websocket_connections': 0,
                'file_processing_jobs': 0,
                'rate_limit_blocks': 0
            }
            
            for counter, value in counters.items():
                key = f"perf_counter:{today}:{counter}"
                if not self.redis_client.exists(key):
                    self.redis_client.setex(key, 86400, value)  # 24 hour expiry
                    
        except Exception as e:
            logger.warning(f"Failed to initialize performance counters: {e}")
    
    def is_available(self) -> bool:
        """Check if Redis is available"""
        if not self.redis_client:
            return False
        try:
            self.redis_client.ping()
            return True
        except:
            return False
    
    # ================== HEALTH DATA CACHING (DISABLED) ==================
    # SMARTWATCH/HEALTH DATA INTEGRATION COMMENTED OUT - NOT USING FOR NOW
    
    # def cache_realtime_health_data(self, user_id: str, health_data, ttl: int = 300) -> bool:
    #     """Cache real-time health data - DISABLED"""
    #     return False
    
    # def get_realtime_health_data(self, user_id: str) -> Optional[Dict]:
    #     """Get health data - DISABLED"""
    #     return None
    
    # def cache_health_history(self, user_id: str, date: str, daily_summary: Dict, ttl: int = 86400) -> bool:
    #     """Cache health history - DISABLED"""
    #     return False
    
    # ================== AI RESPONSE CACHING ==================
    
    def cache_ai_response(self, query: str, response: str, user_id: str, 
                         confidence_score: float = 0.0, ttl: int = 3600) -> bool:
        """
        Cache AI medical responses for faster replies
        TTL: 1 hour default
        """
        if not self.is_available():
            return False
            
        try:
            # Create hash of query for consistent caching
            query_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]
            key = f"ai_cache:{query_hash}"
            
            cache_data = {
                'query_hash': query_hash,
                'original_query': query,
                'response': response,
                'user_id': user_id,
                'confidence_score': confidence_score,
                'cached_at': datetime.now().isoformat()
            }
            
            self.redis_client.setex(key, ttl, json.dumps(cache_data))
            logger.debug(f"✅ Cached AI response for query hash {query_hash}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to cache AI response: {e}")
            return False
    
    def get_cached_ai_response(self, query: str) -> Optional[Dict]:
        """Get cached AI response if available"""
        if not self.is_available():
            return None
            
        try:
            query_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]
            key = f"ai_cache:{query_hash}"
            data = self.redis_client.get(key)
            
            if data:
                # Update cache hit counter
                self.redis_client.incr(f"perf_counter:{datetime.now().strftime('%Y-%m-%d')}:ai_cache_hits")
                return json.loads(data)
            else:
                # Update cache miss counter
                self.redis_client.incr(f"perf_counter:{datetime.now().strftime('%Y-%m-%d')}:ai_cache_misses")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get cached AI response: {e}")
            return None
    
    # ================== WEBSOCKET CONNECTION MANAGEMENT ==================
    
    def register_websocket_connection(self, user_id: str, connection_id: str, 
                                    connection_type: str = 'chat') -> bool:
        """Register active WebSocket connection"""
        if not self.is_available():
            return False
            
        try:
            key = f"websocket:active:{user_id}"
            connection_data = {
                'connection_id': connection_id,
                'connection_type': connection_type,
                'connected_at': datetime.now().isoformat(),
                'last_activity': datetime.now().isoformat()
            }
            
            # Store connection (expires in 1 hour if not updated)
            self.redis_client.setex(key, 3600, json.dumps(connection_data))
            
            # Update connection counter
            self.redis_client.incr(f"perf_counter:{datetime.now().strftime('%Y-%m-%d')}:websocket_connections")
            
            logger.debug(f"✅ Registered WebSocket connection for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register WebSocket connection: {e}")
            return False
    
    def update_websocket_activity(self, user_id: str) -> bool:
        """Update last activity for WebSocket connection"""
        if not self.is_available():
            return False
            
        try:
            key = f"websocket:active:{user_id}"
            data = self.redis_client.get(key)
            
            if data:
                connection_data = json.loads(data)
                connection_data['last_activity'] = datetime.now().isoformat()
                self.redis_client.setex(key, 3600, json.dumps(connection_data))
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to update WebSocket activity: {e}")
            return False
    
    def get_active_websocket_connections(self) -> List[Dict]:
        """Get all active WebSocket connections"""
        if not self.is_available():
            return []
            
        try:
            pattern = "websocket:active:*"
            keys = self.redis_client.keys(pattern)
            connections = []
            
            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    connection_info = json.loads(data)
                    connection_info['user_id'] = key.replace('websocket:active:', '')
                    connections.append(connection_info)
                    
            return connections
            
        except Exception as e:
            logger.error(f"❌ Failed to get active WebSocket connections: {e}")
            return []
    
    # ================== FILE PROCESSING QUEUES ==================
    
    def add_file_processing_job(self, job_id: str, user_id: str, file_type: str, 
                               file_path: str, processing_type: str = 'medical_analysis') -> bool:
        """Add file processing job to queue"""
        if not self.is_available():
            return False
            
        try:
            job_data = {
                'job_id': job_id,
                'user_id': user_id,
                'file_type': file_type,
                'file_path': file_path,
                'processing_type': processing_type,
                'status': 'queued',
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # Add to processing queue
            queue_key = f"file_processing_queue"
            job_key = f"file_processing_job:{job_id}"
            
            # Store job details
            self.redis_client.setex(job_key, 3600, json.dumps(job_data))
            
            # Add to queue
            self.redis_client.lpush(queue_key, job_id)
            
            # Update counter
            self.redis_client.incr(f"perf_counter:{datetime.now().strftime('%Y-%m-%d')}:file_processing_jobs")
            
            logger.debug(f"✅ Added file processing job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add file processing job: {e}")
            return False
    
    def get_next_file_processing_job(self) -> Optional[Dict]:
        """Get next file processing job from queue"""
        if not self.is_available():
            return None
            
        try:
            queue_key = "file_processing_queue"
            job_id = self.redis_client.rpop(queue_key)
            
            if job_id:
                job_key = f"file_processing_job:{job_id}"
                job_data = self.redis_client.get(job_key)
                
                if job_data:
                    return json.loads(job_data)
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get next file processing job: {e}")
            return None
    
    def update_file_processing_status(self, job_id: str, status: str, 
                                    result: Optional[Dict] = None) -> bool:
        """Update file processing job status"""
        if not self.is_available():
            return False
            
        try:
            job_key = f"file_processing_job:{job_id}"
            job_data = self.redis_client.get(job_key)
            
            if job_data:
                job_info = json.loads(job_data)
                job_info['status'] = status
                job_info['updated_at'] = datetime.now().isoformat()
                
                if result:
                    job_info['result'] = result
                
                self.redis_client.setex(job_key, 3600, json.dumps(job_info))
                logger.debug(f"✅ Updated job {job_id} status to {status}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to update file processing status: {e}")
            return False
    
    # ================== RATE LIMITING ==================
    
    def check_rate_limit(self, identifier: str, limit: int, window: int = 60) -> Tuple[bool, int]:
        """
        Check rate limit for API calls/requests
        Returns: (allowed, remaining_requests)
        """
        if not self.is_available():
            return True, limit  # Allow if Redis unavailable
            
        try:
            key = f"rate_limit:{identifier}"
            current = self.redis_client.get(key)
            
            if current is None:
                # First request in window
                self.redis_client.setex(key, window, 1)
                return True, limit - 1
            else:
                current = int(current)
                if current >= limit:
                    return False, 0
                else:
                    self.redis_client.incr(key)
                    return True, limit - current - 1
                    
        except Exception as e:
            logger.error(f"❌ Failed to check rate limit: {e}")
            return True, limit  # Allow on error
    
    # ================== PERFORMANCE MONITORING ==================
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        if not self.is_available():
            return {'redis_available': False}
            
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            stats = {
                'redis_available': True,
                'date': today,
                'ai_cache_hits': int(self.redis_client.get(f"perf_counter:{today}:ai_cache_hits") or 0),
                'ai_cache_misses': int(self.redis_client.get(f"perf_counter:{today}:ai_cache_misses") or 0),
                'websocket_connections': int(self.redis_client.get(f"perf_counter:{today}:websocket_connections") or 0),
                'file_processing_jobs': int(self.redis_client.get(f"perf_counter:{today}:file_processing_jobs") or 0),
                'rate_limit_blocks': int(self.redis_client.get(f"perf_counter:{today}:rate_limit_blocks") or 0),
                'active_websockets': len(self.get_active_websocket_connections()),
                'pending_file_jobs': self.redis_client.llen('file_processing_queue'),
                'memory_usage': self.redis_client.info('memory').get('used_memory_human', 'Unknown'),
                'connected_clients': self.redis_client.info('clients').get('connected_clients', 0)
            }
            
            # Calculate cache hit rate
            total_ai_requests = stats['ai_cache_hits'] + stats['ai_cache_misses']
            if total_ai_requests > 0:
                stats['ai_cache_hit_rate'] = round((stats['ai_cache_hits'] / total_ai_requests) * 100, 2)
            else:
                stats['ai_cache_hit_rate'] = 0
                
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get performance stats: {e}")
            return {'redis_available': False, 'error': str(e)}
    
    def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up expired data and return cleanup stats"""
        if not self.is_available():
            return {'cleaned': 0}
            
        try:
            cleanup_stats = {
                'health_data_cleaned': 0,
                'ai_cache_cleaned': 0,
                'websocket_cleaned': 0,
                'job_data_cleaned': 0
            }
            
            # Clean expired AI cache data (older than 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            ai_cache_keys = self.redis_client.keys("ai_cache:*")
            
            for key in ai_cache_keys:
                data = self.redis_client.get(key)
                if data:
                    try:
                        cached_data = json.loads(data)
                        cached_time = datetime.fromisoformat(cached_data.get('cached_at', ''))
                        if cached_time < cutoff_time:
                            self.redis_client.delete(key)
                            cleanup_stats['ai_cache_cleaned'] += 1
                    except:
                        # Delete malformed data
                        self.redis_client.delete(key)
                        cleanup_stats['ai_cache_cleaned'] += 1
            
            logger.info(f"✅ Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup expired data: {e}")
            return {'error': str(e)}

# Global instance for easy import
redis_performance = RedisPerformanceManager()

# Utility functions for easy access - CORE FEATURES ONLY
# def cache_health_data(user_id: str, health_data: Dict, ttl: int = 300) -> bool:
#     """Quick function to cache health data - DISABLED FOR NOW"""
#     # SMARTWATCH INTEGRATION COMMENTED OUT - NOT USING FOR NOW
#     return False

def get_cached_ai_response(query: str) -> Optional[str]:
    """Quick function to get cached AI response"""
    cached = redis_performance.get_cached_ai_response(query)
    return cached['response'] if cached else None

def cache_ai_response(query: str, response: str, user_id: str = 'system') -> bool:
    """Quick function to cache AI response"""
    return redis_performance.cache_ai_response(query, response, user_id)

if __name__ == "__main__":
    # Test the Redis Performance Manager
    print("🚀 Testing Redis Performance Manager...")
    
    # Test AI response caching (core feature)
    print("Testing AI response caching...")
    
    # Test AI response caching
    ai_success = cache_ai_response("What is diabetes?", "Diabetes is a condition where blood sugar levels are too high.", "test_user")
    print(f"AI response cached: {ai_success}")
    
    cached_response = get_cached_ai_response("What is diabetes?")
    print(f"AI response retrieved: {cached_response is not None}")
    
    # Get performance stats
    stats = redis_performance.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    print("✅ Redis Performance Manager test completed!")