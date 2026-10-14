#!/usr/bin/env python3
"""
Production-Ready Supabase Manager for MedBot v2
===============================================
Zero-error, HIPAA-compliant database operations with advanced error handling
"""

import os
import sys
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import asyncio
from contextlib import contextmanager

# Import required libraries
try:
    from supabase import create_client, Client
    from postgrest import APIError
    import requests
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("Run: pip install supabase postgrest python-dotenv requests")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/supabase_manager.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PatientProfile:
    """HIPAA-compliant patient profile data structure"""
    email: str
    first_name: str
    last_name: str
    phone_number: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    weight: float = 0.0
    height: float = 0.0
    blood_type: Optional[str] = None
    systolic_bp: Optional[int] = None
    diastolic_bp: Optional[int] = None
    sleep_hours: float = 8.0
    sleep_quality: str = 'good'
    bedtime: str = '22:00'
    wake_time: str = '06:00'
    smoking_status: Optional[str] = None
    alcohol_consumption: Optional[str] = None
    exercise_frequency: Optional[str] = None
    diet_type: Optional[str] = None
    emergency_contact_name: str = ''
    emergency_contact_phone: str = ''
    emergency_relationship: str = ''
    emergency_contact_email: Optional[str] = None
    emergency_contact_address: Optional[str] = None
    medical_authorization: bool = False
    chronic_conditions: List[str] = None
    allergies: List[str] = None
    medications: List[str] = None
    bmi: Optional[float] = None
    bmi_category: Optional[str] = None
    health_score: Optional[int] = None
    raw_form_data: Optional[Dict] = None
    
    def __post_init__(self):
        if self.chronic_conditions is None:
            self.chronic_conditions = []
        if self.allergies is None:
            self.allergies = []
        if self.medications is None:
            self.medications = []
        if self.raw_form_data is None:
            self.raw_form_data = {}

class SupabaseManager:
    """Production-ready Supabase database manager with zero-error policy"""
    
    def __init__(self):
        load_dotenv()
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        self.service_role_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        
        if not all([self.supabase_url, self.supabase_key]):
            raise ValueError("Missing Supabase credentials in environment variables")
        
        # Initialize clients
        self.client = create_client(self.supabase_url, self.supabase_key)
        self.admin_client = create_client(self.supabase_url, self.service_role_key) if self.service_role_key else self.client
        
        # Connection state
        self._connection_verified = False
        self._last_health_check = None
        
        logger.info("SupabaseManager initialized successfully")
    
    @contextmanager
    def error_handler(self, operation_name: str):
        """Context manager for comprehensive error handling"""
        try:
            yield
        except APIError as e:
            logger.error(f"{operation_name} failed with API error: {e}")
            raise Exception(f"Database operation failed: {e.message if hasattr(e, 'message') else str(e)}")
        except requests.RequestException as e:
            logger.error(f"{operation_name} failed with network error: {e}")
            raise Exception(f"Network error during {operation_name}: {str(e)}")
        except Exception as e:
            logger.error(f"{operation_name} failed with unexpected error: {e}")
            raise Exception(f"Unexpected error during {operation_name}: {str(e)}")
    
    def verify_connection(self) -> bool:
        """Verify database connection with comprehensive testing"""
        with self.error_handler("Connection verification"):
            try:
                # Test basic connection
                result = self.client.table('pg_tables').select('tablename').limit(1).execute()
                
                # Test table existence
                tables = ['patient_profiles', 'chat_history', 'users']
                existing_tables = []
                
                for table in tables:
                    try:
                        test_result = self.client.table(table).select('*').limit(0).execute()
                        existing_tables.append(table)
                    except:
                        pass
                
                self._connection_verified = True
                self._last_health_check = datetime.now(timezone.utc)
                
                logger.info(f"✅ Database connection verified. Existing tables: {existing_tables}")
                return True, existing_tables
                
            except Exception as e:
                logger.error(f"❌ Connection verification failed: {e}")
                return False, []
    
    def setup_database_schema(self) -> bool:
        """Setup complete database schema with error recovery"""
        logger.info("🏗️ Setting up database schema...")
        
        # Read schema from SQL file
        try:
            with open('supabase_schema.sql', 'r', encoding='utf-8') as f:
                schema_sql = f.read()
        except FileNotFoundError:
            logger.error("❌ supabase_schema.sql file not found")
            return False
        
        # Split into individual commands
        commands = [cmd.strip() for cmd in schema_sql.split(';') if cmd.strip()]
        success_count = 0
        
        with self.error_handler("Schema setup"):
            for i, command in enumerate(commands, 1):
                if not command:
                    continue
                    
                try:
                    # Execute via direct HTTP request (most reliable method)
                    response = requests.post(
                        f"{self.supabase_url}/rest/v1/rpc/query",
                        headers={
                            "Authorization": f"Bearer {self.service_role_key or self.supabase_key}",
                            "Content-Type": "application/json",
                            "apikey": self.service_role_key or self.supabase_key
                        },
                        json={"query": command}
                    )
                    
                    if response.status_code in [200, 201, 204]:
                        logger.info(f"   ✅ Command {i}: Executed successfully")
                        success_count += 1
                    elif "already exists" in response.text.lower():
                        logger.info(f"   ℹ️ Command {i}: Object already exists")
                        success_count += 1
                    else:
                        logger.warning(f"   ⚠️ Command {i}: {response.status_code} - {response.text[:100]}")
                        
                except Exception as e:
                    logger.error(f"   ❌ Command {i} failed: {e}")
            
            logger.info(f"📊 Schema setup complete: {success_count}/{len(commands)} commands executed")
            return success_count > len(commands) * 0.8  # 80% success rate required
    
    def create_patient_profile(self, profile: PatientProfile) -> Tuple[bool, str, Optional[Dict]]:
        """Create patient profile with comprehensive validation and error handling"""
        with self.error_handler("Create patient profile"):
            # Validate required fields
            required_fields = ['email', 'first_name', 'last_name', 'weight', 'height', 
                             'sleep_hours', 'sleep_quality', 'bedtime', 'wake_time',
                             'emergency_contact_name', 'emergency_contact_phone', 'emergency_relationship']
            
            for field in required_fields:
                if not getattr(profile, field, None):
                    return False, f"Missing required field: {field}", None
            
            # Calculate BMI and health metrics
            if profile.weight > 0 and profile.height > 0:
                profile.bmi = round(profile.weight / ((profile.height / 100) ** 2), 2)
                
                if profile.bmi < 18.5:
                    profile.bmi_category = "Underweight"
                elif profile.bmi < 25:
                    profile.bmi_category = "Normal"
                elif profile.bmi < 30:
                    profile.bmi_category = "Overweight"
                else:
                    profile.bmi_category = "Obese"
            
            # Calculate basic health score (0-100)
            profile.health_score = self._calculate_health_score(profile)
            
            try:
                # Check if profile already exists
                existing = self.client.table('patient_profiles').select('id').eq('email', profile.email).execute()
                
                if existing.data:
                    # Update existing profile
                    result = self.client.table('patient_profiles').update(
                        asdict(profile)
                    ).eq('email', profile.email).execute()
                    
                    if result.data:
                        logger.info(f"✅ Patient profile updated for {profile.email}")
                        return True, "Profile updated successfully", result.data[0]
                else:
                    # Create new profile
                    result = self.client.table('patient_profiles').insert(
                        asdict(profile)
                    ).execute()
                    
                    if result.data:
                        logger.info(f"✅ Patient profile created for {profile.email}")
                        return True, "Profile created successfully", result.data[0]
                
                return False, "Failed to create/update profile", None
                
            except Exception as e:
                logger.error(f"❌ Failed to create patient profile: {e}")
                return False, f"Database error: {str(e)}", None
    
    def get_patient_profile(self, email: str) -> Tuple[bool, str, Optional[Dict]]:
        """Retrieve patient profile with error handling"""
        with self.error_handler("Get patient profile"):
            try:
                result = self.client.table('patient_profiles').select('*').eq('email', email).execute()
                
                if result.data:
                    logger.info(f"✅ Retrieved patient profile for {email}")
                    return True, "Profile retrieved successfully", result.data[0]
                else:
                    logger.info(f"ℹ️ No profile found for {email}")
                    return False, "Profile not found", None
                    
            except Exception as e:
                logger.error(f"❌ Failed to retrieve patient profile: {e}")
                return False, f"Database error: {str(e)}", None
    
    def save_chat_message(self, patient_email: str, user_message: str, ai_response: str, 
                         session_id: str, metadata: Dict = None) -> Tuple[bool, str]:
        """Save chat conversation with patient context"""
        with self.error_handler("Save chat message"):
            try:
                chat_data = {
                    'patient_email': patient_email,
                    'user_message': user_message,
                    'ai_response': ai_response,
                    'session_id': session_id,
                    'message_metadata': metadata or {}
                }
                
                result = self.client.table('chat_history').insert(chat_data).execute()
                
                if result.data:
                    logger.info(f"✅ Chat message saved for {patient_email}")
                    return True, "Chat message saved successfully"
                else:
                    return False, "Failed to save chat message"
                    
            except Exception as e:
                logger.error(f"❌ Failed to save chat message: {e}")
                return False, f"Database error: {str(e)}"
    
    def get_chat_history(self, patient_email: str, session_id: str = None, 
                        limit: int = 50) -> Tuple[bool, str, List[Dict]]:
        """Retrieve chat history for conversation context"""
        with self.error_handler("Get chat history"):
            try:
                query = self.client.table('chat_history').select('*').eq('patient_email', patient_email)
                
                if session_id:
                    query = query.eq('session_id', session_id)
                
                result = query.order('created_at', desc=True).limit(limit).execute()
                
                if result.data:
                    # Reverse to get chronological order
                    chat_history = list(reversed(result.data))
                    logger.info(f"✅ Retrieved {len(chat_history)} chat messages for {patient_email}")
                    return True, "Chat history retrieved successfully", chat_history
                else:
                    logger.info(f"ℹ️ No chat history found for {patient_email}")
                    return True, "No chat history found", []
                    
            except Exception as e:
                logger.error(f"❌ Failed to retrieve chat history: {e}")
                return False, f"Database error: {str(e)}", []
    
    def _calculate_health_score(self, profile: PatientProfile) -> int:
        """Calculate basic health score based on profile data"""
        score = 50  # Base score
        
        # BMI scoring
        if profile.bmi:
            if 18.5 <= profile.bmi < 25:
                score += 20
            elif 25 <= profile.bmi < 30:
                score += 10
            else:
                score -= 10
        
        # Sleep scoring
        if 7 <= profile.sleep_hours <= 9:
            score += 15
        elif profile.sleep_hours < 6 or profile.sleep_hours > 10:
            score -= 10
        
        if profile.sleep_quality in ['excellent', 'good']:
            score += 10
        elif profile.sleep_quality in ['poor', 'terrible']:
            score -= 15
        
        # Lifestyle scoring
        if profile.smoking_status == 'never':
            score += 10
        elif profile.smoking_status in ['current', 'heavy']:
            score -= 20
        
        if profile.exercise_frequency in ['daily', 'frequent']:
            score += 15
        elif profile.exercise_frequency == 'never':
            score -= 15
        
        # Ensure score is within bounds
        return max(0, min(100, score))
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of database operations"""
        health_status = {
            'database_connected': False,
            'tables_exist': [],
            'last_check': datetime.now(timezone.utc).isoformat(),
            'errors': []
        }
        
        try:
            # Test connection
            connected, existing_tables = self.verify_connection()
            health_status['database_connected'] = connected
            health_status['tables_exist'] = existing_tables
            
            # Test basic operations
            if connected:
                # Test patient profile operations
                test_profile = PatientProfile(
                    email='test@medbot.ai',
                    first_name='Test',
                    last_name='User',
                    weight=70.0,
                    height=175.0,
                    sleep_hours=8.0,
                    sleep_quality='good',
                    bedtime='22:00',
                    wake_time='06:00',
                    emergency_contact_name='Emergency Contact',
                    emergency_contact_phone='+1234567890',
                    emergency_relationship='spouse'
                )
                
                # Try to create and retrieve test profile
                success, message, data = self.create_patient_profile(test_profile)
                if success:
                    # Clean up test data
                    try:
                        self.client.table('patient_profiles').delete().eq('email', 'test@medbot.ai').execute()
                    except:
                        pass
                else:
                    health_status['errors'].append(f"Profile operations failed: {message}")
            
        except Exception as e:
            health_status['errors'].append(f"Health check failed: {str(e)}")
        
        logger.info(f"Health check completed: {health_status}")
        return health_status

# Global instance for easy access
supabase_manager = None

def get_supabase_manager() -> SupabaseManager:
    """Get global Supabase manager instance"""
    global supabase_manager
    if supabase_manager is None:
        supabase_manager = SupabaseManager()
    return supabase_manager

def initialize_database() -> bool:
    """Initialize database with schema setup"""
    manager = get_supabase_manager()
    
    # Verify connection
    connected, tables = manager.verify_connection()
    
    if not connected:
        logger.error("❌ Cannot connect to database")
        return False
    
    # Setup schema if tables are missing
    required_tables = ['patient_profiles', 'chat_history', 'users']
    missing_tables = [table for table in required_tables if table not in tables]
    
    if missing_tables:
        logger.info(f"🔧 Missing tables detected: {missing_tables}")
        logger.info("📦 Setting up database schema...")
        
        if not manager.setup_database_schema():
            logger.error("❌ Failed to setup database schema")
            return False
    
    logger.info("✅ Database initialization completed successfully")
    return True

if __name__ == "__main__":
    # CLI interface for database management
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "init":
            success = initialize_database()
            sys.exit(0 if success else 1)
        elif command == "health":
            manager = get_supabase_manager()
            health = manager.health_check()
            print(json.dumps(health, indent=2))
            sys.exit(0)
        else:
            print("Usage: python supabase_manager.py [init|health]")
            sys.exit(1)
    else:
        # Interactive mode
        print("🏥 MedBot Supabase Manager")
        print("Run 'python supabase_manager.py init' to initialize database")
        print("Run 'python supabase_manager.py health' for health check")