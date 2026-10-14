"""
ADMIN PANEL INTEGRATION v3.0
Real-time book management, upload monitoring, and system control
Compatible with ultra-fast indexing system
"""

import os
import json
import time
import asyncio
import threading
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import psutil
import redis
from collections import defaultdict, deque
import uuid

from store_index import UltimateMedicalIndexer, UltimateConfig, ProcessingMode
from src.helper import BookRegistry

logger = logging.getLogger(__name__)

# ==================== AI ADMIN MANAGER ====================
class SimpleAdminMonitor:
    """Simple admin monitoring - no database dependencies"""
    
    def __init__(self):
        self.request_counts = defaultdict(int)
        self.response_times = []
        self.user_sessions = {}
        self.start_time = datetime.now()
        
        logger.info("✅ Simple Admin Monitor initialized")
    
    def get_basic_metrics(self, admin_email: str = None) -> Dict[str, Any]:
        """Get basic system metrics without database dependencies"""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'admin_requesting': admin_email,
                'uptime_seconds': uptime,
                'system_status': 'running',
                'quick_stats': {
                    'active_sessions': len(self.user_sessions),
                    'requests_handled': sum(self.request_counts.values()),
                    'average_response_time': sum(self.response_times) / len(self.response_times) if self.response_times else 0
                }
            }
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Create alias for compatibility
EnhancedMedicalAdminDashboard = SimpleAdminMonitor

class AIAdminManager:
    """AI-powered admin manager for intelligent system monitoring and optimization"""
    
    def __init__(self):
        self.alerts = deque(maxlen=100)  # Store last 100 alerts
        self.recommendations = []
        self.auto_actions = []
        self.monitoring_data = defaultdict(list)
        
    def analyze_system_health(self, metrics: Dict) -> Dict:
        """AI-powered system health analysis"""
        health_score = 100
        issues = []
        recommendations = []
        
        try:
            # CPU Analysis
            cpu_usage = metrics.get('cpu_usage', 0)
            if cpu_usage > 90:
                health_score -= 20
                issues.append("High CPU usage detected")
                recommendations.append("Consider scaling resources or optimizing queries")
            elif cpu_usage > 70:
                health_score -= 10
                issues.append("Moderate CPU usage")
                recommendations.append("Monitor CPU usage trends")
            
            # Memory Analysis
            memory_usage = metrics.get('memory_usage', 0)
            if memory_usage > 85:
                health_score -= 15
                issues.append("High memory usage detected")
                recommendations.append("Clear cache or restart services")
            
            # Error Rate Analysis
            error_rate = metrics.get('error_rate', 0)
            if error_rate > 5:
                health_score -= 25
                issues.append("High error rate detected")
                recommendations.append("Check logs and fix critical issues")
            
            # Response Time Analysis
            avg_response = metrics.get('avg_response_time', 0)
            if avg_response > 5000:  # 5 seconds
                health_score -= 15
                issues.append("Slow response times detected")
                recommendations.append("Optimize database queries and caching")
            
            # Active Users Analysis
            active_users = metrics.get('active_users', 0)
            if active_users > 1000:
                recommendations.append("High traffic detected - ensure auto-scaling is enabled")
            
            # Generate AI insights
            ai_insights = self._generate_ai_insights(metrics, issues)
            
            return {
                'health_score': max(0, health_score),
                'status': 'critical' if health_score < 50 else 'warning' if health_score < 80 else 'healthy',
                'issues': issues,
                'recommendations': recommendations,
                'ai_insights': ai_insights,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"AI health analysis error: {e}")
            return {
                'health_score': 0,
                'status': 'unknown',
                'issues': ['Analysis system error'],
                'recommendations': ['Check AI admin manager'],
                'ai_insights': [],
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_ai_insights(self, metrics: Dict, issues: List[str]) -> List[str]:
        """Generate AI-powered insights and predictions"""
        insights = []
        
        # Traffic pattern analysis
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 17:
            insights.append("Peak business hours detected - increased resource allocation recommended")
        elif 22 <= current_hour or current_hour <= 6:
            insights.append("Low traffic period - consider maintenance window")
        
        # Predictive insights based on metrics
        if metrics.get('requests_per_minute', 0) > 100:
            insights.append("High request volume may require load balancing")
        
        if len(issues) == 0:
            insights.append("System operating optimally - all metrics within normal ranges")
        elif len(issues) >= 3:
            insights.append("Multiple issues detected - immediate attention required")
        
        return insights
    
    def create_alert(self, level: str, message: str, source: str = "system"):
        """Create system alert"""
        alert = {
            'id': str(uuid.uuid4())[:8],
            'level': level,  # critical, warning, info
            'message': message,
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'resolved': False
        }
        self.alerts.appendleft(alert)
        logger.info(f"[ALERT-{level.upper()}] {message}")
        return alert
    
    def get_recent_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent alerts"""
        return list(self.alerts)[:limit]

# ==================== TRAFFIC ANALYTICS SYSTEM ====================
class TrafficAnalytics:
    """Real-time traffic monitoring and analytics"""
    
    def __init__(self):
        self.request_log = deque(maxlen=10000)  # Last 10k requests
        self.user_sessions = {}
        self.endpoint_stats = defaultdict(lambda: {'count': 0, 'avg_time': 0, 'errors': 0})
        self.hourly_stats = defaultdict(int)
        
    def log_request(self, endpoint: str, method: str, response_time: float, 
                   status_code: int, user_id: str = None, ip: str = None):
        """Log request for analytics"""
        request_data = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'method': method,
            'response_time': response_time,
            'status_code': status_code,
            'user_id': user_id,
            'ip': ip,
            'hour': datetime.now().hour
        }
        
        self.request_log.appendleft(request_data)
        
        # Update endpoint stats
        stats = self.endpoint_stats[endpoint]
        stats['count'] += 1
        stats['avg_time'] = (stats['avg_time'] + response_time) / 2
        if status_code >= 400:
            stats['errors'] += 1
        
        # Update hourly stats
        self.hourly_stats[datetime.now().hour] += 1
        
        # Track user session
        if user_id:
            self.user_sessions[user_id] = datetime.now()
    
    def get_admin_dashboard_metrics(self, admin_email: str = None) -> Dict[str, Any]:
        """Get comprehensive admin dashboard metrics with auth integration"""
        try:
            # Get auth manager statistics
            auth_stats = {}
            try:
                from auth_manager import get_auth_manager
                auth_manager = get_auth_manager()
                if auth_manager:
                    auth_stats = auth_manager.get_user_statistics()
                    auth_stats['recent_admin_actions'] = auth_manager.get_admin_logs(limit=10)
                    auth_stats['auth_performance'] = auth_manager.get_auth_stats()
            except Exception as e:
                logger.warning(f"Could not get auth statistics: {e}")
                auth_stats = {'error': 'Auth statistics unavailable'}
            
            # Get conversation engine stats
            conversation_stats = {}
            try:
                from enhanced_conversational_engine import get_conversational_engine
                engine = get_conversational_engine()
                if engine:
                    conversation_stats = engine.get_conversation_analytics()
            except Exception as e:
                logger.warning(f"Could not get conversation statistics: {e}")
                conversation_stats = {'error': 'Conversation statistics unavailable'}
            
            # Get book processing stats
            book_stats = {}
            try:
                from advanced_book_processor import UltimateMedicalBookProcessor
                # This would require a global instance or registry
                book_stats = {'processed_books': 0, 'total_chunks': 0}
            except Exception as e:
                logger.warning(f"Could not get book processing statistics: {e}")
                book_stats = {'error': 'Book processing statistics unavailable'}
            
            # Combine all metrics
            dashboard_metrics = {
                'timestamp': datetime.now().isoformat(),
                'admin_requesting': admin_email,
                'auth_metrics': auth_stats,
                'conversation_metrics': conversation_stats,
                'book_processing_metrics': book_stats,
                'system_health': self.get_realtime_metrics(),
                'quick_stats': {
                    'total_users': auth_stats.get('total_users', 0),
                    'active_sessions': auth_stats.get('active_sessions', 0),
                    'admin_users': auth_stats.get('admin_users', 0),
                    'recent_registrations': auth_stats.get('recent_registrations', 0),
                    'total_conversations': conversation_stats.get('total_active_conversations', 0),
                    'processed_books': book_stats.get('processed_books', 0)
                }
            }
            
            # Log admin access
            if admin_email:
                logger.info(f"📊 Admin dashboard accessed by: {admin_email}")
            
            return dashboard_metrics
            
        except Exception as e:
            logger.error(f"Get admin dashboard metrics failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'admin_requesting': admin_email
            }

    def get_realtime_metrics(self) -> Dict:
        """Get real-time traffic metrics"""
        now = datetime.now()
        last_minute = now - timedelta(minutes=1)
        last_hour = now - timedelta(hours=1)
        
        # Count requests in last minute and hour
        recent_requests = [r for r in self.request_log 
                         if datetime.fromisoformat(r['timestamp']) > last_minute]
        hourly_requests = [r for r in self.request_log 
                         if datetime.fromisoformat(r['timestamp']) > last_hour]
        
        # Active users (active in last 30 minutes)
        active_cutoff = now - timedelta(minutes=30)
        active_users = len([uid for uid, last_seen in self.user_sessions.items()
                          if last_seen > active_cutoff])
        
        # Error rate calculation
        total_recent = len(recent_requests)
        error_recent = len([r for r in recent_requests if r['status_code'] >= 400])
        error_rate = (error_recent / total_recent * 100) if total_recent > 0 else 0
        
        # Average response time
        if recent_requests:
            avg_response = sum(r['response_time'] for r in recent_requests) / len(recent_requests)
        else:
            avg_response = 0
        
        return {
            'requests_per_minute': len(recent_requests),
            'requests_per_hour': len(hourly_requests),
            'active_users': active_users,
            'total_users': len(self.user_sessions),
            'error_rate': error_rate,
            'avg_response_time': avg_response,
            'top_endpoints': self._get_top_endpoints(),
            'traffic_pattern': self._get_traffic_pattern(),
            'current_load': self._calculate_load_level(),
            'timestamp': now.isoformat()
        }
    
    def _get_top_endpoints(self, limit: int = 5) -> List[Dict]:
        """Get most accessed endpoints"""
        sorted_endpoints = sorted(self.endpoint_stats.items(), 
                                key=lambda x: x[1]['count'], reverse=True)
        return [{'endpoint': ep, **stats} for ep, stats in sorted_endpoints[:limit]]
    
    def _get_traffic_pattern(self) -> List[Dict]:
        """Get 24-hour traffic pattern"""
        return [{'hour': hour, 'requests': count} 
               for hour, count in sorted(self.hourly_stats.items())]
    
    def _calculate_load_level(self) -> str:
        """Calculate current system load level"""
        rpm = len([r for r in self.request_log 
                  if datetime.fromisoformat(r['timestamp']) > datetime.now() - timedelta(minutes=1)])
        
        if rpm > 200:
            return 'high'
        elif rpm > 50:
            return 'medium'
        else:
            return 'low'

# ==================== API KEY MANAGEMENT SYSTEM ====================
class APIKeyManager:
    """Dynamic API key management with auto-switching"""
    
    def __init__(self):
        self.api_keys = {}  # {service: [keys]}
        self.key_usage = {}  # {key: usage_count}
        self.key_limits = {}  # {key: limit}
        self.current_keys = {}  # {service: current_active_key}
        self.failed_keys = set()  # Keys that have failed
        
    def add_api_key(self, service: str, key: str, limit: int = 1000000) -> bool:
        """Add new API key for service"""
        try:
            if service not in self.api_keys:
                self.api_keys[service] = []
            
            self.api_keys[service].append(key)
            self.key_usage[key] = 0
            self.key_limits[key] = limit
            
            # Set as current if first key for service
            if service not in self.current_keys:
                self.current_keys[service] = key
            
            logger.info(f"Added API key for {service}")
            return True
        except Exception as e:
            logger.error(f"Failed to add API key: {e}")
            return False
    
    def get_active_key(self, service: str) -> Optional[str]:
        """Get currently active API key for service"""
        return self.current_keys.get(service)
    
    def switch_api_key(self, service: str) -> Optional[str]:
        """Switch to next available API key"""
        try:
            available_keys = [key for key in self.api_keys.get(service, []) 
                            if key not in self.failed_keys]
            
            if not available_keys:
                logger.error(f"No available keys for {service}")
                return None
            
            # Find key with lowest usage
            best_key = min(available_keys, key=lambda k: self.key_usage.get(k, 0))
            
            self.current_keys[service] = best_key
            logger.info(f"Switched to new API key for {service}")
            return best_key
            
        except Exception as e:
            logger.error(f"Failed to switch API key: {e}")
            return None
    
    def mark_key_failed(self, service: str, key: str):
        """Mark API key as failed"""
        self.failed_keys.add(key)
        logger.warning(f"Marked API key as failed for {service}")
        
        # Auto-switch to next available key
        if self.current_keys.get(service) == key:
            new_key = self.switch_api_key(service)
            if new_key:
                return new_key
        return None
    
    def increment_usage(self, key: str):
        """Increment usage counter for key"""
        self.key_usage[key] = self.key_usage.get(key, 0) + 1
        
        # Check if key is approaching limit
        limit = self.key_limits.get(key, float('inf'))
        if self.key_usage[key] > limit * 0.9:  # 90% of limit
            logger.warning(f"API key approaching limit: {self.key_usage[key]}/{limit}")
    
    def get_key_stats(self) -> Dict:
        """Get API key statistics"""
        stats = {}
        for service, keys in self.api_keys.items():
            service_stats = {
                'total_keys': len(keys),
                'active_key': self.current_keys.get(service, 'None'),
                'failed_keys': len([k for k in keys if k in self.failed_keys]),
                'key_details': []
            }
            
            for key in keys:
                key_masked = key[:8] + '...' + key[-4:] if len(key) > 12 else key
                service_stats['key_details'].append({
                    'key': key_masked,
                    'usage': self.key_usage.get(key, 0),
                    'limit': self.key_limits.get(key, 0),
                    'status': 'failed' if key in self.failed_keys else 'active',
                    'usage_percent': (self.key_usage.get(key, 0) / self.key_limits.get(key, 1)) * 100
                })
            
            stats[service] = service_stats
        
        return stats

@dataclass
class BookStatus:
    """Book status for admin panel"""
    filename: str
    size_mb: float
    status: str  # 'indexed', 'processing', 'pending', 'error'
    last_modified: str
    book_type: str
    chunks_count: int = 0
    upload_progress: float = 0
    error_message: str = ""
    indexed_at: str = ""

@dataclass
class SystemStatus:
    """Complete system status"""
    is_indexing: bool = False
    current_operation: str = ""
    progress_percent: float = 0
    books_processing: List[str] = None
    books_completed: List[str] = None
    upload_rate: float = 0
    chunks_processed: int = 0
    chunks_uploaded: int = 0
    estimated_remaining: float = 0
    system_resources: Dict = None
    cache_stats: Dict = None
    
    def __post_init__(self):
        if self.books_processing is None:
            self.books_processing = []
        if self.books_completed is None:
            self.books_completed = []
        if self.system_resources is None:
            self.system_resources = {}
        if self.cache_stats is None:
            self.cache_stats = {}

class RealTimeBookManager:
    """Real-time book management system"""
    
    def __init__(self):
        self.registry = BookRegistry()
        self.data_path = Path("data")
        self.status_file = Path("admin_status.json")
        self.upload_dir = Path("uploads")
        self.indexer = None
        self.indexing_thread = None
        self.is_indexing = False
        
        # Create directories
        self.data_path.mkdir(exist_ok=True)
        self.upload_dir.mkdir(exist_ok=True)
        
        # System monitoring
        self.system_monitor = SystemMonitor()
        
    def get_all_books_status(self) -> List[BookStatus]:
        """Get status of all books in data directory"""
        books_status = []
        
        for pdf_file in self.data_path.glob("*.pdf"):
            try:
                size_mb = pdf_file.stat().st_size / (1024 * 1024)
                last_modified = datetime.fromtimestamp(pdf_file.stat().st_mtime).isoformat()
                
                # Check if book is indexed
                if pdf_file.name in self.registry.registry:
                    registry_info = self.registry.registry[pdf_file.name]
                    status = "indexed"
                    indexed_at = registry_info.get('uploaded_at', '')
                    chunks_count = registry_info.get('chunks_processed', 0)
                else:
                    status = "pending"
                    indexed_at = ""
                    chunks_count = 0
                
                # Detect book type (simplified)
                book_type = self._detect_book_type(pdf_file.name)
                
                book_status = BookStatus(
                    filename=pdf_file.name,
                    size_mb=size_mb,
                    status=status,
                    last_modified=last_modified,
                    book_type=book_type,
                    chunks_count=chunks_count,
                    indexed_at=indexed_at
                )
                
                books_status.append(book_status)
                
            except Exception as e:
                logger.error(f"Error getting status for {pdf_file.name}: {e}")
                
        return sorted(books_status, key=lambda x: x.last_modified, reverse=True)
    
    def _detect_book_type(self, filename: str) -> str:
        """Simple book type detection"""
        filename_lower = filename.lower()
        
        if any(keyword in filename_lower for keyword in ['anatomy', 'anatomical']):
            return 'anatomy'
        elif any(keyword in filename_lower for keyword in ['physiology', 'physiological']):
            return 'physiology'
        elif any(keyword in filename_lower for keyword in ['pathology', 'disease']):
            return 'pathology'
        elif any(keyword in filename_lower for keyword in ['pharmacology', 'drug']):
            return 'pharmacology'
        elif any(keyword in filename_lower for keyword in ['surgery', 'surgical']):
            return 'surgery'
        elif any(keyword in filename_lower for keyword in ['harrison', 'internal', 'medicine']):
            return 'internal_medicine'
        elif any(keyword in filename_lower for keyword in ['emergency', 'tintinalli']):
            return 'emergency'
        else:
            return 'general'
    
    def upload_book(self, file_data, filename: str) -> Dict:
        """Upload new book file"""
        try:
            # Secure filename
            filename = secure_filename(filename)
            if not filename.endswith('.pdf'):
                return {"success": False, "error": "Only PDF files are allowed"}
            
            # Save to data directory
            file_path = self.data_path / filename
            
            # Check if file already exists
            if file_path.exists():
                return {"success": False, "error": "File already exists"}
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            logger.info(f"Uploaded book: {filename}")
            return {
                "success": True, 
                "message": f"Successfully uploaded {filename}",
                "filename": filename,
                "size_mb": file_path.stat().st_size / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_book(self, filename: str) -> Dict:
        """Delete book file and remove from registry"""
        try:
            file_path = self.data_path / filename
            
            if not file_path.exists():
                return {"success": False, "error": "File not found"}
            
            # Remove from registry
            if filename in self.registry.registry:
                self.registry.remove_book(filename)
            
            # Delete file
            file_path.unlink()
            
            logger.info(f"Deleted book: {filename}")
            return {"success": True, "message": f"Successfully deleted {filename}"}
            
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return {"success": False, "error": str(e)}
    
    def start_indexing(self, selected_books: List[str] = None, mode: str = "ultra") -> Dict:
        """Start indexing process"""
        if self.is_indexing:
            return {"success": False, "error": "Indexing already in progress"}
        
        try:
            # Get books to process
            if selected_books:
                pdf_files = [self.data_path / book for book in selected_books if (self.data_path / book).exists()]
            else:
                pdf_files = list(self.data_path.glob("*.pdf"))
            
            if not pdf_files:
                return {"success": False, "error": "No books to index"}
            
            # Start indexing in background thread
            processing_mode = ProcessingMode(mode) if mode in [m.value for m in ProcessingMode] else ProcessingMode.ULTRA
            
            self.indexing_thread = threading.Thread(
                target=self._run_indexing,
                args=(pdf_files, processing_mode),
                daemon=True
            )
            
            self.is_indexing = True
            self.indexing_thread.start()
            
            logger.info(f"Started indexing {len(pdf_files)} books in {mode} mode")
            return {
                "success": True,
                "message": f"Started indexing {len(pdf_files)} books",
                "mode": mode,
                "books_count": len(pdf_files)
            }
            
        except Exception as e:
            logger.error(f"Indexing start error: {e}")
            self.is_indexing = False
            return {"success": False, "error": str(e)}
    
    def _run_indexing(self, pdf_files: List[Path], mode: ProcessingMode):
        """Run indexing in background thread"""
        try:
            self.indexer = UltimateMedicalIndexer(mode)
            
            # Setup progress callback
            def progress_callback(data):
                self._save_indexing_status(data)
            
            self.indexer.monitor.add_callback(progress_callback)
            
            # Run async indexing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            success = loop.run_until_complete(
                self.indexer.create_ultra_index(pdf_files)
            )
            
            # Update final status
            if success:
                logger.info("Indexing completed successfully")
                for pdf in pdf_files:
                    self.registry.mark_uploaded(pdf)
            else:
                logger.error("Indexing completed with errors")
                
        except Exception as e:
            logger.error(f"Indexing error: {e}")
            
        finally:
            self.is_indexing = False
            loop.close()
    
    def _save_indexing_status(self, data: Dict):
        """Save indexing status for real-time monitoring"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'is_indexing': self.is_indexing,
                'indexing_data': data,
                'system_resources': self.system_monitor.get_current_resources()
            }
            
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Status save error: {e}")
    
    def stop_indexing(self) -> Dict:
        """Stop indexing process"""
        try:
            if not self.is_indexing:
                return {"success": False, "error": "No indexing in progress"}
            
            if self.indexer:
                self.indexer.shutdown_requested = True
            
            self.is_indexing = False
            
            return {"success": True, "message": "Indexing stop requested"}
            
        except Exception as e:
            logger.error(f"Stop indexing error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_system_status(self) -> SystemStatus:
        """Get complete system status"""
        try:
            # Read status file
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    status_data = json.load(f)
                
                indexing_data = status_data.get('indexing_data', {})
                
                return SystemStatus(
                    is_indexing=self.is_indexing,
                    current_operation=indexing_data.get('message', ''),
                    progress_percent=indexing_data.get('progress', 0),
                    upload_rate=indexing_data.get('upload_rate', 0),
                    estimated_remaining=indexing_data.get('estimated_remaining', 0),
                    system_resources=status_data.get('system_resources', {}),
                    cache_stats=indexing_data.get('cache_stats', {})
                )
            else:
                return SystemStatus(
                    is_indexing=self.is_indexing,
                    system_resources=self.system_monitor.get_current_resources()
                )
                
        except Exception as e:
            logger.warning(f"Status read error: {e}")
            return SystemStatus(is_indexing=self.is_indexing)

class SystemMonitor:
    """System resource monitoring"""
    
    def get_current_resources(self) -> Dict:
        """Get current system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
                'disk_percent': (disk.used / disk.total) * 100
            }
        except Exception as e:
            logger.warning(f"Resource monitoring error: {e}")
            return {}

# Flask Blueprint for Admin Panel
def create_admin_blueprint():
    """Create Flask blueprint for admin panel"""
    from flask import Blueprint, render_template, request, jsonify
    from dataclasses import asdict
    import logging
    
    admin_bp = Blueprint('admin', __name__, url_prefix='/admin')
    book_manager = RealTimeBookManager()
    logger = logging.getLogger(__name__)
    
    @admin_bp.route('/dashboard')
    def dashboard():
        """Admin dashboard"""
        try:
            books_status = book_manager.get_all_books_status()
            system_status = book_manager.get_system_status()
            
            return render_template('admin_dashboard.html', 
                                 books=books_status,
                                 system_status=asdict(system_status))
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @admin_bp.route('/api/books')
    def get_books():
        """API: Get all books status"""
        try:
            books_status = book_manager.get_all_books_status()
            return jsonify([asdict(book) for book in books_status])
        except Exception as e:
            logger.error(f"Get books error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @admin_bp.route('/api/upload', methods=['POST'])
    def upload_book():
        """API: Upload new book"""
        try:
            if 'file' not in request.files:
                return jsonify({"success": False, "error": "No file provided"}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({"success": False, "error": "No file selected"}), 400
            
            result = book_manager.upload_book(file.read(), file.filename)
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Upload API error: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    @admin_bp.route('/api/delete/<filename>', methods=['DELETE'])
    def delete_book(filename):
        """API: Delete book"""
        try:
            result = book_manager.delete_book(filename)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Delete API error: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    @admin_bp.route('/api/index', methods=['POST'])
    def start_indexing():
        """API: Start indexing"""
        try:
            data = request.get_json() or {}
            selected_books = data.get('books', [])
            mode = data.get('mode', 'ultra')
            
            result = book_manager.start_indexing(selected_books, mode)
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Start indexing API error: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    @admin_bp.route('/api/index/stop', methods=['POST'])
    def stop_indexing():
        """API: Stop indexing"""
        try:
            result = book_manager.stop_indexing()
            return jsonify(result)
        except Exception as e:
            logger.error(f"Stop indexing API error: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    @admin_bp.route('/api/status')
    def get_status():
        """API: Get system status"""
        try:
            status = book_manager.get_system_status()
            return jsonify(asdict(status))
        except Exception as e:
            logger.error(f"Status API error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @admin_bp.route('/api/system/resources')
    def get_system_resources():
        """API: Get system resources"""
        try:
            resources = book_manager.system_monitor.get_current_resources()
            return jsonify(resources)
        except Exception as e:
            logger.error(f"Resources API error: {e}")
            return jsonify({"error": str(e)}), 500
    
    return admin_bp

# Enhanced Admin Dashboard Template (to be saved as admin_dashboard.html)
ADMIN_DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MedAI Admin Dashboard</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --success-color: #28a745;
            --warning-color: #ffc107;
            --danger-color: #dc3545;
            --dark-bg: #0a0e1a;
            --card-bg: rgba(255, 255, 255, 0.05);
            --text-primary: #ffffff;
            --text-secondary: #a0aec0;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--dark-bg);
            color: var(--text-primary);
            line-height: 1.6;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 250px 1fr;
            min-height: 100vh;
        }
        
        .sidebar {
            background: var(--card-bg);
            padding: 20px;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .sidebar h2 {
            margin-bottom: 30px;
            color: var(--primary-color);
        }
        
        .nav-item {
            display: block;
            padding: 12px 16px;
            margin-bottom: 8px;
            text-decoration: none;
            color: var(--text-secondary);
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        
        .nav-item:hover, .nav-item.active {
            background: var(--primary-color);
            color: white;
        }
        
        .main-content {
            padding: 30px;
            overflow-y: auto;
        }
        
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: var(--card-bg);
            padding: 25px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .stat-label {
            color: var(--text-secondary);
        }
        
        .books-section {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 30px;
        }
        
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-primary {
            background: var(--primary-color);
            color: white;
        }
        
        .btn-success {
            background: var(--success-color);
            color: white;
        }
        
        .btn-danger {
            background: var(--danger-color);
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        
        .books-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .books-table th,
        .books-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .books-table th {
            background: rgba(255, 255, 255, 0.05);
            font-weight: 600;
        }
        
        .status-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 500;
        }
        
        .status-indexed {
            background: var(--success-color);
            color: white;
        }
        
        .status-pending {
            background: var(--warning-color);
            color: black;
        }
        
        .status-processing {
            background: var(--primary-color);
            color: white;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            margin: 20px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            transition: width 0.3s ease;
        }
        
        .system-status {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 25px;
        }
        
        .resource-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .resource-bar {
            flex: 1;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            margin: 0 15px;
            overflow: hidden;
        }
        
        .resource-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s ease;
        }
        
        .upload-area {
            border: 2px dashed rgba(255, 255, 255, 0.3);
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .upload-area:hover {
            border-color: var(--primary-color);
            background: rgba(255, 255, 255, 0.02);
        }
        
        .upload-area.dragover {
            border-color: var(--success-color);
            background: rgba(40, 167, 69, 0.1);
        }
        
        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
            
            .sidebar {
                display: none;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <nav class="sidebar">
            <h2><i class="fas fa-hospital"></i> MedAI Admin</h2>
            <a href="#dashboard" class="nav-item active">
                <i class="fas fa-tachometer-alt"></i> Dashboard
            </a>
            <a href="#books" class="nav-item">
                <i class="fas fa-book-medical"></i> Books
            </a>
            <a href="#indexing" class="nav-item">
                <i class="fas fa-cogs"></i> Indexing
            </a>
            <a href="#system" class="nav-item">
                <i class="fas fa-server"></i> System
            </a>
        </nav>
        
        <main class="main-content">
            <div class="header">
                <h1>Medical AI Dashboard</h1>
                <div>
                    <button class="btn btn-success" onclick="refreshData()">
                        <i class="fas fa-sync-alt"></i> Refresh
                    </button>
                </div>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="total-books">{{ books|length }}</div>
                    <div class="stat-label">Total Books</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="indexed-books">
                        {{ books|selectattr("status", "equalto", "indexed")|list|length }}
                    </div>
                    <div class="stat-label">Indexed Books</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="total-chunks">
                        {{ books|sum(attribute="chunks_count") }}
                    </div>
                    <div class="stat-label">Total Chunks</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="system-status">
                        {% if system_status.is_indexing %}Processing{% else %}Ready{% endif %}
                    </div>
                    <div class="stat-label">System Status</div>
                </div>
            </div>
            
            {% if system_status.is_indexing %}
            <div class="books-section">
                <h3>Indexing Progress</h3>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {{ system_status.progress_percent }}%"></div>
                </div>
                <p>{{ system_status.current_operation }}</p>
                <p>{{ "%.1f"|format(system_status.estimated_remaining) }} seconds remaining</p>
            </div>
            {% endif %}
            
            <div class="books-section">
                <div class="section-header">
                    <h3>Medical Books</h3>
                    <div>
                        <button class="btn btn-primary" onclick="showUploadModal()">
                            <i class="fas fa-upload"></i> Upload Book
                        </button>
                        <button class="btn btn-success" onclick="startIndexing()">
                            <i class="fas fa-play"></i> Start Indexing
                        </button>
                    </div>
                </div>
                
                <table class="books-table">
                    <thead>
                        <tr>
                            <th>Book Name</th>
                            <th>Type</th>
                            <th>Size</th>
                            <th>Status</th>
                            <th>Chunks</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for book in books %}
                        <tr>
                            <td>{{ book.filename }}</td>
                            <td>{{ book.book_type.title() }}</td>
                            <td>{{ "%.1f"|format(book.size_mb) }} MB</td>
                            <td>
                                <span class="status-badge status-{{ book.status }}">
                                    {{ book.status.title() }}
                                </span>
                            </td>
                            <td>{{ book.chunks_count }}</td>
                            <td>
                                <button class="btn btn-danger" onclick="deleteBook('{{ book.filename }}')">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            
            <div class="system-status">
                <h3>System Resources</h3>
                <div class="resource-item">
                    <span>CPU Usage</span>
                    <div class="resource-bar">
                        <div class="resource-fill" style="width: {{ system_status.system_resources.cpu_percent or 0 }}%; background: #ffc107;"></div>
                    </div>
                    <span>{{ "%.1f"|format(system_status.system_resources.cpu_percent or 0) }}%</span>
                </div>
                <div class="resource-item">
                    <span>Memory Usage</span>
                    <div class="resource-bar">
                        <div class="resource-fill" style="width: {{ system_status.system_resources.memory_percent or 0 }}%; background: #28a745;"></div>
                    </div>
                    <span>{{ "%.1f"|format(system_status.system_resources.memory_percent or 0) }}%</span>
                </div>
                <div class="resource-item">
                    <span>Disk Usage</span>
                    <div class="resource-bar">
                        <div class="resource-fill" style="width: {{ system_status.system_resources.disk_percent or 0 }}%; background: #dc3545;"></div>
                    </div>
                    <span>{{ "%.1f"|format(system_status.system_resources.disk_percent or 0) }}%</span>
                </div>
            </div>
        </main>
    </div>
    
    <script>
        // Real-time dashboard updates
        let updateInterval;
        
        function startRealTimeUpdates() {
            updateInterval = setInterval(refreshData, 2000); // Update every 2 seconds
        }
        
        function stopRealTimeUpdates() {
            if (updateInterval) {
                clearInterval(updateInterval);
            }
        }
        
        async function refreshData() {
            try {
                const [booksResponse, statusResponse] = await Promise.all([
                    fetch('/admin/api/books'),
                    fetch('/admin/api/status')
                ]);
                
                const books = await booksResponse.json();
                const status = await statusResponse.json();
                
                updateDashboard(books, status);
            } catch (error) {
                console.error('Refresh error:', error);
            }
        }
        
        function updateDashboard(books, status) {
            // Update stats
            document.getElementById('total-books').textContent = books.length;
            document.getElementById('indexed-books').textContent = books.filter(b => b.status === 'indexed').length;
            document.getElementById('total-chunks').textContent = books.reduce((sum, b) => sum + (b.chunks_count || 0), 0);
            document.getElementById('system-status').textContent = status.is_indexing ? 'Processing' : 'Ready';
            
            // Update progress bar if indexing
            const progressBar = document.querySelector('.progress-fill');
            if (progressBar && status.is_indexing) {
                progressBar.style.width = status.progress_percent + '%';
            }
        }
        
        async function startIndexing() {
            try {
                const response = await fetch('/admin/api/index', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        mode: 'ultra'
                    })
                });
                
                const result = await response.json();
                if (result.success) {
                    alert('Indexing started successfully!');
                    refreshData();
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Network error: ' + error.message);
            }
        }
        
        async function deleteBook(filename) {
            if (!confirm(`Are you sure you want to delete "${filename}"?`)) {
                return;
            }
            
            try {
                const response = await fetch(`/admin/api/delete/${encodeURIComponent(filename)}`, {
                    method: 'DELETE'
                });
                
                const result = await response.json();
                if (result.success) {
                    alert('Book deleted successfully!');
                    refreshData();
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Network error: ' + error.message);
            }
        }
        
        function showUploadModal() {
            // Simple file input for now
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.pdf';
            input.onchange = async (e) => {
                const file = e.target.files[0];
                if (file) {
                    await uploadBook(file);
                }
            };
            input.click();
        }
        
        async function uploadBook(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/admin/api/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                if (result.success) {
                    alert('Book uploaded successfully!');
                    refreshData();
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Network error: ' + error.message);
            }
        }
        
        // Start real-time updates when page loads
        document.addEventListener('DOMContentLoaded', () => {
            startRealTimeUpdates();
        });
        
        // Stop updates when page unloads
        window.addEventListener('beforeunload', () => {
            stopRealTimeUpdates();
        });
    </script>
</body>
</html>
'''

if __name__ == "__main__":
    # Test the admin panel system
    manager = RealTimeBookManager()
    books = manager.get_all_books_status()
    status = manager.get_system_status()
    
    print(f"Found {len(books)} books")
    print(f"System status: {asdict(status)}")