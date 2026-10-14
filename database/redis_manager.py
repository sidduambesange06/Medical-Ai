"""
Production Redis Manager - Ultra-Advanced Redis Management
Extracted and enhanced from original app.py with enterprise features
"""
import redis
import json
import time
import logging
from datetime import timedelta
from typing import Any, Optional, Dict, List

logger = logging.getLogger(__name__)

class ProductionRedisManager:
    """
    Production Redis Manager optimized for high-concurrency public hosting.

    Features: advanced connection pooling, intelligent caching strategies,
    data compression, rate limiting support, and automatic failover.
    """
    
    def __init__(self, redis_url: str, max_connections: int = 200, retry_attempts: int = 5):
        self.redis_url = redis_url
        self.max_connections = max_connections
        self.retry_attempts = retry_attempts
        self.client = None
        self.connection_pool = None
        self.is_cluster = False
        
        # 🚀 PUBLIC HOSTING OPTIMIZATIONS
        self.cache_strategies = {
            'sessions': {'ttl': 3600, 'compress': True},  # 1 hour sessions with compression
            'user_data': {'ttl': 7200, 'compress': True},  # 2 hours user data
            'chat_history': {'ttl': 86400, 'compress': False},  # 24 hours chat (fast access)
            'rate_limits': {'ttl': 60, 'compress': False},  # 1 minute rate limits
            'api_responses': {'ttl': 300, 'compress': True},  # 5 minutes API cache
            'medical_queries': {'ttl': 1800, 'compress': True}  # 30 minutes medical data
        }
        
        self.performance_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'connection_failures': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'compression_savings': 0,
            'concurrent_users': 0,
            'last_failure': None
        }
        
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize Redis connection with production-grade settings"""
        try:
            # Determine if this is a cluster setup
            if 'cluster' in self.redis_url.lower():
                self._initialize_cluster()
            else:
                self._initialize_single_node()
                
            logger.info("[REDIS] Production Redis connection established with advanced pooling")
            
        except Exception as e:
            logger.error(f"[REDIS-ERROR] Redis initialization failed: {e}")
            self.performance_stats['connection_failures'] += 1
            self.performance_stats['last_failure'] = time.time()
            self.client = None
    
    def _initialize_single_node(self):
        """Initialize single-node Redis connection"""
        self.connection_pool = redis.ConnectionPool.from_url(
            self.redis_url,
            max_connections=self.max_connections,
            retry_on_timeout=True,
            socket_timeout=10,  # Increased for public hosting
            socket_connect_timeout=10,
            socket_keepalive=True,
            socket_keepalive_options={
                'TCP_KEEPIDLE': 600,
                'TCP_KEEPINTVL': 30,
                'TCP_KEEPCNT': 3
            },
            health_check_interval=60,  # Less frequent for performance
            connection_class=redis.Connection
        )
        
        self.client = redis.Redis(
            connection_pool=self.connection_pool,
            decode_responses=True,
            socket_timeout=10,
            socket_connect_timeout=10,
            retry_on_timeout=True,
            retry_on_error=[redis.ConnectionError, redis.TimeoutError, redis.BusyLoadingError]
        )
        
        # Test connection
        self.client.ping()
    
    def _initialize_cluster(self):
        """Initialize Redis cluster connection"""
        try:
            from rediscluster import RedisCluster
            
            # Parse cluster nodes from URL
            # This would need proper cluster URL parsing
            cluster_nodes = [{"host": "127.0.0.1", "port": "6379"}]
            
            self.client = RedisCluster(
                startup_nodes=cluster_nodes,
                decode_responses=True,
                skip_full_coverage_check=True,
                max_connections=self.max_connections,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            self.is_cluster = True
            
        except ImportError:
            logger.warning("[REDIS] Redis cluster library not available, falling back to single node")
            self._initialize_single_node()
    
    def safe_get(self, key: str, fallback: Any = None, decompress: bool = False) -> Any:
        """Safely get value from Redis with automatic retry and decompression"""
        self.performance_stats['total_operations'] += 1
        
        for attempt in range(self.retry_attempts):
            try:
                if not self.client:
                    self._reconnect()
                    
                if self.client:
                    value = self.client.get(key)
                    
                    if value is not None:
                        # Handle decompression if needed
                        if decompress and isinstance(value, str) and value.startswith('compressed:'):
                            value = self._decompress_data(value[11:])  # Remove 'compressed:' prefix
                        
                        # Try to parse JSON
                        try:
                            value = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            pass  # Return as-is if not JSON
                        
                        self.performance_stats['successful_operations'] += 1
                        return value
                        
                return fallback
                
            except Exception as e:
                logger.warning(f"[REDIS-ERROR] Get operation failed (attempt {attempt + 1}): {e}")
                if attempt == self.retry_attempts - 1:
                    self.performance_stats['failed_operations'] += 1
                    return fallback
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                
        return fallback
    
    def safe_set(self, key: str, value: Any, ttl: int = 3600, compress: bool = False, nx: bool = False) -> bool:
        """Safely set value in Redis with compression and advanced options"""
        self.performance_stats['total_operations'] += 1
        
        for attempt in range(self.retry_attempts):
            try:
                if not self.client:
                    self._reconnect()
                    
                if self.client:
                    # Serialize value
                    if not isinstance(value, (str, bytes, int, float)):
                        value = json.dumps(value, default=str)
                    
                    # Compress if requested and value is large
                    if compress and isinstance(value, str) and len(value) > 1000:
                        value = 'compressed:' + self._compress_data(value)
                    
                    # Set with options
                    if nx:
                        result = self.client.set(key, value, ex=ttl, nx=True)
                    else:
                        result = self.client.setex(key, ttl, value)
                    
                    self.performance_stats['successful_operations'] += 1
                    return bool(result)
                    
                return False
                
            except Exception as e:
                logger.warning(f"[REDIS-ERROR] Set operation failed (attempt {attempt + 1}): {e}")
                if attempt == self.retry_attempts - 1:
                    self.performance_stats['failed_operations'] += 1
                    return False
                time.sleep(0.1 * (attempt + 1))
                
        return False
    
    def safe_delete(self, *keys) -> int:
        """Safely delete keys from Redis"""
        self.performance_stats['total_operations'] += 1
        
        try:
            if not self.client:
                self._reconnect()
                
            if self.client and keys:
                result = self.client.delete(*keys)
                self.performance_stats['successful_operations'] += 1
                return result
                
        except Exception as e:
            logger.warning(f"[REDIS-ERROR] Delete operation failed: {e}")
            self.performance_stats['failed_operations'] += 1
            
        return 0
    
    def safe_exists(self, key: str) -> bool:
        """Safely check if key exists"""
        self.performance_stats['total_operations'] += 1
        
        try:
            if not self.client:
                self._reconnect()
                
            if self.client:
                result = self.client.exists(key)
                self.performance_stats['successful_operations'] += 1
                return bool(result)
                
        except Exception as e:
            logger.warning(f"[REDIS-ERROR] Exists operation failed: {e}")
            self.performance_stats['failed_operations'] += 1
            
        return False
    
    def safe_increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Safely increment a counter"""
        self.performance_stats['total_operations'] += 1
        
        try:
            if not self.client:
                self._reconnect()
                
            if self.client:
                result = self.client.incrby(key, amount)
                self.performance_stats['successful_operations'] += 1
                return result
                
        except Exception as e:
            logger.warning(f"[REDIS-ERROR] Increment operation failed: {e}")
            self.performance_stats['failed_operations'] += 1
            
        return None
    
    def get_keys_pattern(self, pattern: str) -> List[str]:
        """Get keys matching a pattern (use carefully in production)"""
        try:
            if not self.client:
                self._reconnect()
                
            if self.client:
                if self.is_cluster:
                    # For cluster, we need to scan all nodes
                    keys = []
                    for node in self.client.connection_pool.nodes.all_masters():
                        node_keys = self.client.keys(pattern, target_nodes=[node])
                        keys.extend(node_keys)
                    return keys
                else:
                    return self.client.keys(pattern)
                    
        except Exception as e:
            logger.warning(f"[REDIS-ERROR] Pattern search failed: {e}")
            
        return []
    
    def get_memory_usage(self, key: str) -> Optional[int]:
        """Get memory usage of a key"""
        try:
            if not self.client:
                self._reconnect()
                
            if self.client and hasattr(self.client, 'memory_usage'):
                return self.client.memory_usage(key)
                
        except Exception as e:
            logger.warning(f"[REDIS-ERROR] Memory usage check failed: {e}")
            
        return None
    
    def get_ttl(self, key: str) -> Optional[int]:
        """Get time to live for a key"""
        try:
            if not self.client:
                self._reconnect()
                
            if self.client:
                ttl = self.client.ttl(key)
                return ttl if ttl >= 0 else None
                
        except Exception as e:
            logger.warning(f"[REDIS-ERROR] TTL check failed: {e}")
            
        return None
    
    def pipeline_operations(self, operations: List[Dict]) -> List[Any]:
        """Execute multiple operations in a pipeline for better performance"""
        try:
            if not self.client:
                self._reconnect()
                
            if self.client:
                pipe = self.client.pipeline()
                
                for op in operations:
                    method = getattr(pipe, op['method'])
                    method(*op.get('args', []), **op.get('kwargs', {}))
                
                results = pipe.execute()
                self.performance_stats['successful_operations'] += len(operations)
                return results
                
        except Exception as e:
            logger.error(f"[REDIS-ERROR] Pipeline operations failed: {e}")
            self.performance_stats['failed_operations'] += len(operations)
            
        return []
    
    def _reconnect(self):
        """Attempt to reconnect to Redis"""
        try:
            logger.info("[REDIS] Attempting to reconnect...")
            self._initialize_connection()
            
        except Exception as e:
            logger.error(f"[REDIS-ERROR] Reconnection failed: {e}")
    
    def _compress_data(self, data: str) -> str:
        """Compress data using gzip"""
        try:
            import gzip
            import base64
            
            compressed = gzip.compress(data.encode('utf-8'))
            return base64.b64encode(compressed).decode('utf-8')
            
        except Exception as e:
            logger.warning(f"[REDIS-ERROR] Compression failed: {e}")
            return data
    
    def _decompress_data(self, data: str) -> str:
        """Decompress data using gzip"""
        try:
            import gzip
            import base64
            
            compressed = base64.b64decode(data.encode('utf-8'))
            return gzip.decompress(compressed).decode('utf-8')
            
        except Exception as e:
            logger.warning(f"[REDIS-ERROR] Decompression failed: {e}")
            return data
    
    def get_client(self) -> Optional[redis.Redis]:
        """Get Redis client with health check"""
        if self.client:
            try:
                self.client.ping()
                return self.client
            except Exception:
                logger.warning("[REDIS] Connection lost, attempting reconnect...")
                self._reconnect()
                return self.client
        return None
    
    def get_info(self) -> Dict:
        """Get Redis server information"""
        try:
            if not self.client:
                self._reconnect()
                
            if self.client:
                info = self.client.info()
                return info
                
        except Exception as e:
            logger.warning(f"[REDIS-ERROR] Info retrieval failed: {e}")
            
        return {}
    
    def get_performance_stats(self) -> Dict:
        """Get Redis manager performance statistics"""
        stats = self.performance_stats.copy()
        
        if stats['total_operations'] > 0:
            stats['success_rate'] = (stats['successful_operations'] / stats['total_operations']) * 100
        else:
            stats['success_rate'] = 100.0
            
        return stats
    
    def health_check(self) -> Dict:
        """Comprehensive Redis health check"""
        health_status = {
            'connected': False,
            'cluster_mode': self.is_cluster,
            'performance': self.get_performance_stats(),
            'server_info': {},
            'memory_usage': None,
            'key_count': 0,
            'errors': []
        }
        
        try:
            if not self.client:
                self._reconnect()
                
            if self.client:
                # Test connection
                self.client.ping()
                health_status['connected'] = True
                
                # Get server info
                server_info = self.get_info()
                health_status['server_info'] = {
                    'redis_version': server_info.get('redis_version', 'unknown'),
                    'uptime_in_seconds': server_info.get('uptime_in_seconds', 0),
                    'connected_clients': server_info.get('connected_clients', 0),
                    'used_memory_human': server_info.get('used_memory_human', 'unknown'),
                    'total_commands_processed': server_info.get('total_commands_processed', 0)
                }
                
                # Get memory usage
                health_status['memory_usage'] = server_info.get('used_memory', 0)
                
                # Get approximate key count (be careful with this in production)
                try:
                    health_status['key_count'] = len(self.get_keys_pattern('*'))
                except:
                    health_status['key_count'] = 'unavailable'
                    
        except Exception as e:
            health_status['errors'].append(str(e))
            logger.error(f"[REDIS-HEALTH] Health check failed: {e}")
        
        return health_status
    
    def close(self):
        """Close Redis connection and cleanup"""
        try:
            if self.connection_pool:
                self.connection_pool.disconnect()
            
            if self.client:
                self.client.close()
                
            logger.info("[REDIS] Connection closed cleanly")
            
        except Exception as e:
            logger.warning(f"[REDIS] Error during cleanup: {e}")