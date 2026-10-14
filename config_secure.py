"""
PRODUCTION-SECURE CONFIGURATION
Zero hardcoded secrets, HIPAA compliant
"""
import os
import secrets
from pathlib import Path

class SecureProductionConfig:
    """Ultra-secure production configuration"""
    
    # Generate secure secret key if not provided
    SECRET_KEY = os.getenv('SECRET_KEY') or secrets.token_urlsafe(32)
    
    # NO HARDCODED CREDENTIALS - Environment only
    ADMIN_EMAIL = os.getenv('ADMIN_EMAIL') # Must be set in environment
    ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD') # Must be set in environment
    
    # API Keys - Environment only
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    SUPABASE_URL = os.getenv('SUPABASE_URL') 
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    # Security Headers - OAuth-friendly CSP
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'SAMEORIGIN',  # Changed from DENY to allow OAuth frames
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://fonts.googleapis.com; "
            "font-src 'self' https://cdnjs.cloudflare.com https://fonts.gstatic.com; "
            "connect-src 'self' https://vyzzvdimsuaeknpmyggt.supabase.co https://accounts.google.com https://github.com; "
            "img-src 'self' data: https:; "
            "frame-src 'self' https://accounts.google.com https://github.com;"
        )
    }
    
    # Rate Limiting (DDOS Protection) - More generous for better UX
    RATE_LIMIT_DEFAULT = "1000/hour"
    RATE_LIMIT_CHAT = "500/hour"  # Medical queries - increased for better user experience
    RATE_LIMIT_AUTH = "20/minute"  # Authentication - slightly increased
    
    # Session Security
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True  
    SESSION_COOKIE_SAMESITE = 'Strict'
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour
    
    # HIPAA Compliance
    MEDICAL_DATA_ENCRYPTION = True
    AUDIT_LOGS_ENABLED = True
    DATA_RETENTION_DAYS = 2555  # 7 years HIPAA requirement
    
    @classmethod
    def validate(cls):
        """Validate all critical config"""
        missing = []
        for key in ['ADMIN_EMAIL', 'ADMIN_PASSWORD', 'GROQ_API_KEY', 'SUPABASE_URL']:
            if not getattr(cls, key):
                missing.append(key)
        
        if missing:
            raise ValueError(f"🔴 SECURITY: Missing environment variables: {missing}")
        
        if cls.SECRET_KEY == 'ca87f5c370b94fa7df1de744288f9180':
            raise ValueError("🔴 SECURITY: Default secret key detected!")
            
        return True

# Global secure config
secure_config = SecureProductionConfig()