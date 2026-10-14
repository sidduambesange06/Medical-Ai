"""
Enhanced Admin Dashboard Endpoints
===================================
FastAPI APIRouter providing real-time system metrics and file listing for the admin panel.
Replaces the previous Flask Blueprint implementation.
"""

import os
from datetime import datetime

import psutil
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse

router = APIRouter(prefix="/admin", tags=["admin"])


def _test_redis() -> bool:
    """Return True if a local Redis instance is reachable."""
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=1)
        r.ping()
        return True
    except Exception:
        return False


@router.get("/api/metrics/enhanced")
async def enhanced_metrics(request: Request):
    """Return real-time system and cache metrics."""
    # Guard: only admins may call this endpoint
    if request.cookies.get("is_admin") != "true":
        return JSONResponse({"error": "Admin access required"}, status_code=403)

    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        data_dir = os.path.join(os.getcwd(), "data")
        file_count = len(os.listdir(data_dir)) if os.path.exists(data_dir) else 0

        metrics = {
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024 ** 3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024 ** 3), 2),
            },
            "file_manager": {
                "data_directory": data_dir,
                "file_count": file_count,
                "directory_exists": os.path.exists(data_dir),
            },
            "cache": {
                "redis_status": "connected" if _test_redis() else "disconnected",
                "cache_enabled": True,
            },
            "timestamp": datetime.now().isoformat(),
        }
        return JSONResponse(metrics)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/api/files")
async def list_files(request: Request):
    """List files in the data directory (admin only)."""
    if request.cookies.get("is_admin") != "true":
        return JSONResponse({"error": "Admin access required"}, status_code=403)

    try:
        data_dir = os.path.join(os.getcwd(), "data")
        if not os.path.exists(data_dir):
            return JSONResponse({"files": [], "message": "Data directory not found"})

        files = []
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath):
                stat = os.stat(filepath)
                files.append({
                    "name": filename,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "type": os.path.splitext(filename)[1],
                })

        return JSONResponse({"files": files})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
