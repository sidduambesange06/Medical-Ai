"""
Production-grade configuration management
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class ProductionConfig:
    """Production configuration with intelligent defaults and validation"""
    
    # Core Application Settings
    FLASK_ENV = os.getenv('FLASK_ENV', 'production')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    SECRET_KEY = os.getenv('SECRET_KEY')  # Required: min 32-char random string
    
    # Supabase Configuration
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    SUPABASE_SERVICE_ROLE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    # AI/ML Configuration
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Vector Database
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1-gcp')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'medical-chatbot-v2')
    
    # Redis Configuration
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    # Rate Limiting
    RATE_LIMIT_CHAT = int(os.getenv('RATE_LIMIT_CHAT', '50'))
    RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', '3600'))
    
    # Medical Knowledge Base
    MEDICAL_DOCS_PATH = os.getenv('MEDICAL_DOCS_PATH', './data/')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))
    
    # Admin Credentials
    ADMIN_EMAIL = os.getenv('ADMIN_EMAIL')  # Required: admin account email
    ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')  # Required: admin account password
    
    # OAuth Settings
    GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
    GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
    
    @classmethod
    def validate_config(cls):
        """Validate critical configuration"""
        missing = []
        if not cls.SUPABASE_URL:
            missing.append('SUPABASE_URL')
        if not cls.SUPABASE_KEY:
            missing.append('SUPABASE_KEY')
        if not cls.GROQ_API_KEY:
            missing.append('GROQ_API_KEY')
        
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
        
        return True

# Global config instance
config = ProductionConfig()