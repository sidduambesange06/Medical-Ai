
# Enhanced Admin Dashboard Endpoints Fix
from flask import Blueprint, request, jsonify, session, redirect, url_for, render_template_string
import psutil
import json
import os
from datetime import datetime

admin_bp = Blueprint('admin_enhanced', __name__, url_prefix='/admin')

# Enhanced metrics endpoint
@admin_bp.route('/api/metrics/enhanced', methods=['GET'])
def enhanced_metrics():
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # File system metrics
        data_dir = os.path.join(os.getcwd(), 'data')
        file_count = len(os.listdir(data_dir)) if os.path.exists(data_dir) else 0
        
        metrics = {
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            },
            "file_manager": {
                "data_directory": data_dir,
                "file_count": file_count,
                "directory_exists": os.path.exists(data_dir)
            },
            "cache": {
                "redis_status": "connected" if test_redis() else "disconnected",
                "cache_enabled": True
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def test_redis():
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        return True
    except:
        return False

# File manager endpoint
@admin_bp.route('/api/files', methods=['GET'])
def list_files():
    try:
        data_dir = os.path.join(os.getcwd(), 'data')
        if not os.path.exists(data_dir):
            return jsonify({"files": [], "message": "Data directory not found"})
        
        files = []
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath):
                stat = os.stat(filepath)
                files.append({
                    "name": filename,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "type": os.path.splitext(filename)[1]
                })
        
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
