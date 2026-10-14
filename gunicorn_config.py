#!/usr/bin/env python3
"""
PRODUCTION GUNICORN CONFIGURATION
Optimized for millions of concurrent users
"""
import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '8080')}"
backlog = 2048

# Worker processes
workers = int(os.getenv('WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = "gevent"
worker_connections = 1000
max_requests = 10000
max_requests_jitter = 1000

# Timeouts
timeout = 30
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'medbot'

# Server mechanics
daemon = False
pidfile = '/tmp/medbot.pid'
user = None
group = None
tmp_upload_dir = None

# SSL (for production)
keyfile = os.getenv('SSL_KEYFILE')
certfile = os.getenv('SSL_CERTFILE')

# Preload app for better memory usage
preload_app = True

# Memory management
max_requests_jitter = 1000
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

def when_ready(server):
    server.log.info("MedBot server is ready. Accepting connections.")

def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def pre_exec(server):
    server.log.info("Forked child, re-executing.")

def worker_exit(server, worker):
    server.log.info("Worker exited (pid: %s)", worker.pid)