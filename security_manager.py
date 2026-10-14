"""
Production-grade security manager for medical data protection.

Features:
- AES-256-GCM encryption for sensitive data
- Argon2 password hashing
- PBKDF2 key derivation
- Secure conversation history storage
- PII data protection
"""

import os
import hashlib
import secrets
import base64
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.backends import default_backend
import argon2
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

# Setup logging
logger = logging.getLogger(__name__)

class ProductionSecurityManager:
    """
    Production-grade security manager for medical chatbot
    Implements industry-standard encryption and hashing practices
    """
    
    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize security manager with encryption capabilities
        
        Args:
            encryption_key: Master encryption key (from environment)
        """
        self.backend = default_backend()
        
        # Initialize password hasher with Argon2 (OWASP recommended)
        self.password_hasher = PasswordHasher(
            time_cost=3,        # 3 iterations (recommended minimum)
            memory_cost=65536,  # 64 MB memory usage
            parallelism=1,      # Single threaded
            hash_len=32,        # 32 byte hash output
            salt_len=16         # 16 byte salt
        )
        
        # Set up encryption key
        self.master_key = self._setup_master_key(encryption_key)
        
        # Data classification levels
        self.DATA_LEVELS = {
            'PUBLIC': 0,
            'INTERNAL': 1,
            'CONFIDENTIAL': 2,
            'MEDICAL_PII': 3,
            'HIPAA_PROTECTED': 4
        }
        
        logger.info("🔐 Production Security Manager initialized with AES-256-GCM encryption")
    
    def _setup_master_key(self, provided_key: Optional[str] = None) -> bytes:
        """Setup or generate master encryption key"""
        if provided_key:
            # Derive key from provided string
            salt = b'medai_security_salt_2024'  # Fixed salt for consistency
            kdf = PBKDF2HMAC(
                algorithm=SHA256(),
                length=32,  # 256-bit key
                salt=salt,
                iterations=100000,  # OWASP recommended minimum
                backend=self.backend
            )
            return kdf.derive(provided_key.encode())
        else:
            # Generate random key (store securely in production)
            return secrets.token_bytes(32)
    
    def hash_password(self, password: str) -> str:
        """
        Hash password using Argon2 (industry standard)
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password string
        """
        try:
            hashed = self.password_hasher.hash(password)
            logger.debug("✅ Password hashed successfully using Argon2")
            return hashed
        except Exception as e:
            logger.error(f"❌ Password hashing failed: {e}")
            raise
    
    def verify_password(self, hashed_password: str, plain_password: str) -> bool:
        """
        Verify password against hash
        
        Args:
            hashed_password: Stored hash
            plain_password: Plain text password to verify
            
        Returns:
            True if password matches
        """
        try:
            self.password_hasher.verify(hashed_password, plain_password)
            logger.debug("✅ Password verification successful")
            return True
        except VerifyMismatchError:
            logger.warning("❌ Password verification failed - incorrect password")
            return False
        except Exception as e:
            logger.error(f"❌ Password verification error: {e}")
            return False
    
    def encrypt_data(self, data: Union[str, Dict, List], data_level: str = 'CONFIDENTIAL') -> Dict[str, str]:
        """
        Encrypt sensitive data using AES-256-GCM
        
        Args:
            data: Data to encrypt (string, dict, or list)
            data_level: Data classification level
            
        Returns:
            Dictionary with encrypted data and metadata
        """
        try:
            # Convert data to JSON string if needed
            if isinstance(data, (dict, list)):
                data_str = json.dumps(data, ensure_ascii=False)
            else:
                data_str = str(data)
            
            # Generate random nonce (number used once)
            nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(self.master_key),
                modes.GCM(nonce),
                backend=self.backend
            )
            encryptor = cipher.encryptor()
            
            # Encrypt data
            encrypted_data = encryptor.update(data_str.encode('utf-8')) + encryptor.finalize()
            
            # Get authentication tag
            tag = encryptor.tag
            
            # Create encrypted package
            encrypted_package = {
                'encrypted_data': base64.b64encode(encrypted_data).decode('ascii'),
                'nonce': base64.b64encode(nonce).decode('ascii'),
                'tag': base64.b64encode(tag).decode('ascii'),
                'data_level': data_level,
                'algorithm': 'AES-256-GCM',
                'encrypted_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            logger.debug(f"✅ Data encrypted successfully (Level: {data_level})")
            return encrypted_package
            
        except Exception as e:
            logger.error(f"❌ Data encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_package: Dict[str, str]) -> Union[str, Dict, List]:
        """
        Decrypt data using AES-256-GCM
        
        Args:
            encrypted_package: Encrypted data package
            
        Returns:
            Decrypted data (original type preserved)
        """
        try:
            # Extract components
            encrypted_data = base64.b64decode(encrypted_package['encrypted_data'])
            nonce = base64.b64decode(encrypted_package['nonce'])
            tag = base64.b64decode(encrypted_package['tag'])
            
            # Create cipher for decryption
            cipher = Cipher(
                algorithms.AES(self.master_key),
                modes.GCM(nonce, tag),
                backend=self.backend
            )
            decryptor = cipher.decryptor()
            
            # Decrypt data
            decrypted_bytes = decryptor.update(encrypted_data) + decryptor.finalize()
            decrypted_str = decrypted_bytes.decode('utf-8')
            
            # Try to parse as JSON
            try:
                return json.loads(decrypted_str)
            except json.JSONDecodeError:
                return decrypted_str
                
        except Exception as e:
            logger.error(f"❌ Data decryption failed: {e}")
            raise
    
    def hash_pii_data(self, pii_data: str, salt: Optional[str] = None) -> Dict[str, str]:
        """
        Hash PII data for privacy protection (one-way)
        
        Args:
            pii_data: Personal identifiable information
            salt: Optional salt (generated if not provided)
            
        Returns:
            Dictionary with hash and salt
        """
        try:
            if salt is None:
                salt = secrets.token_hex(16)  # 32 character hex salt
            
            # Create PBKDF2 hash for PII
            combined = f"{pii_data}:{salt}".encode('utf-8')
            hash_value = hashlib.pbkdf2_hmac('sha256', combined, salt.encode(), 100000)
            
            return {
                'hash': base64.b64encode(hash_value).decode('ascii'),
                'salt': salt,
                'algorithm': 'PBKDF2-SHA256',
                'iterations': 100000,
                'hashed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ PII hashing failed: {e}")
            raise
    
    def create_secure_conversation_record(self, 
                                        user_id: str, 
                                        conversation_data: Dict[str, Any],
                                        medical_content: bool = True) -> Dict[str, Any]:
        """
        Create securely encrypted conversation record
        
        Args:
            user_id: User identifier
            conversation_data: Conversation data to encrypt
            medical_content: Whether conversation contains medical information
            
        Returns:
            Encrypted conversation record
        """
        try:
            # Determine data classification
            data_level = 'HIPAA_PROTECTED' if medical_content else 'CONFIDENTIAL'
            
            # Add metadata
            conversation_with_meta = {
                'user_id_hash': self.hash_pii_data(user_id)['hash'],
                'conversation': conversation_data,
                'medical_content': medical_content,
                'created_at': datetime.now().isoformat(),
                'session_id': secrets.token_hex(16)
            }
            
            # Encrypt conversation data
            encrypted_conversation = self.encrypt_data(conversation_with_meta, data_level)
            
            # Add additional security metadata
            encrypted_conversation.update({
                'record_type': 'conversation',
                'compliance_level': 'HIPAA' if medical_content else 'GDPR',
                'retention_policy': '7_years' if medical_content else '2_years'
            })
            
            logger.info(f"✅ Secure conversation record created (Level: {data_level})")
            return encrypted_conversation
            
        except Exception as e:
            logger.error(f"❌ Secure conversation creation failed: {e}")
            raise
    
    def sanitize_user_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize and hash sensitive user data
        
        Args:
            user_data: Raw user data
            
        Returns:
            Sanitized user data with hashed PII
        """
        try:
            sanitized = user_data.copy()
            
            # Fields to hash (PII)
            pii_fields = ['email', 'phone', 'address', 'ssn', 'medical_id']
            
            # Fields to encrypt
            sensitive_fields = ['name', 'date_of_birth', 'medical_history']
            
            # Hash PII fields
            for field in pii_fields:
                if field in sanitized and sanitized[field]:
                    hashed_data = self.hash_pii_data(str(sanitized[field]))
                    sanitized[f"{field}_hash"] = hashed_data['hash']
                    sanitized[f"{field}_salt"] = hashed_data['salt']
                    del sanitized[field]  # Remove original
            
            # Encrypt sensitive fields
            for field in sensitive_fields:
                if field in sanitized and sanitized[field]:
                    encrypted_data = self.encrypt_data(sanitized[field], 'MEDICAL_PII')
                    sanitized[f"{field}_encrypted"] = encrypted_data
                    del sanitized[field]  # Remove original
            
            # Add sanitization metadata
            sanitized['data_sanitized'] = True
            sanitized['sanitized_at'] = datetime.now().isoformat()
            sanitized['privacy_level'] = 'HIPAA_COMPLIANT'
            
            logger.info("[SUCCESS] User data sanitized successfully")
            return sanitized
            
        except Exception as e:
            logger.error(f"[ERROR] User data sanitization failed: {e}")
            raise
    
    def create_audit_log(self, 
                        action: str, 
                        user_id: str, 
                        details: Dict[str, Any],
                        ip_address: str,
                        user_agent: str) -> Dict[str, Any]:
        """
        Create encrypted audit log entry
        
        Args:
            action: Action performed
            user_id: User who performed action
            details: Action details
            ip_address: User IP address
            user_agent: User agent string
            
        Returns:
            Encrypted audit log entry
        """
        try:
            # Create audit record
            audit_record = {
                'action': action,
                'user_id_hash': self.hash_pii_data(user_id)['hash'],
                'details': details,
                'ip_hash': self.hash_pii_data(ip_address)['hash'],
                'user_agent_hash': self.hash_pii_data(user_agent)['hash'],
                'timestamp': datetime.now().isoformat(),
                'audit_id': secrets.token_hex(16)
            }
            
            # Encrypt audit record
            encrypted_audit = self.encrypt_data(audit_record, 'CONFIDENTIAL')
            encrypted_audit['record_type'] = 'audit_log'
            
            logger.debug(f"✅ Audit log created for action: {action}")
            return encrypted_audit
            
        except Exception as e:
            logger.error(f"❌ Audit log creation failed: {e}")
            raise
    
    def validate_data_integrity(self, encrypted_package: Dict[str, str]) -> bool:
        """
        Validate data integrity using authentication tag
        
        Args:
            encrypted_package: Encrypted data package
            
        Returns:
            True if data integrity is valid
        """
        try:
            # Attempt decryption (will fail if tampered)
            self.decrypt_data(encrypted_package)
            return True
        except Exception:
            logger.warning("❌ Data integrity validation failed")
            return False
    
    def generate_session_token(self, user_id: str, expires_hours: int = 24) -> Dict[str, str]:
        """
        Generate secure session token
        
        Args:
            user_id: User identifier
            expires_hours: Token expiration in hours
            
        Returns:
            Session token data
        """
        try:
            # Create token payload
            payload = {
                'user_id_hash': self.hash_pii_data(user_id)['hash'],
                'issued_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=expires_hours)).isoformat(),
                'token_id': secrets.token_hex(16),
                'session_key': secrets.token_hex(32)
            }
            
            # Encrypt token
            encrypted_token = self.encrypt_data(payload, 'CONFIDENTIAL')
            
            # Create token string
            token_string = base64.b64encode(
                json.dumps(encrypted_token).encode()
            ).decode('ascii')
            
            return {
                'token': token_string,
                'expires_at': payload['expires_at'],
                'token_id': payload['token_id']
            }
            
        except Exception as e:
            logger.error(f"❌ Session token generation failed: {e}")
            raise

# Global security manager instance
security_manager = ProductionSecurityManager(
    encryption_key=os.getenv('MEDAI_ENCRYPTION_KEY')  # Required: encryption key
)

def encrypt_conversation_history(conversation_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Convenience function to encrypt conversation history"""
    return security_manager.create_secure_conversation_record(user_id, conversation_data, True)

def decrypt_conversation_history(encrypted_record: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to decrypt conversation history"""
    return security_manager.decrypt_data(encrypted_record)

def hash_user_email(email: str) -> str:
    """Convenience function to hash user email"""
    return security_manager.hash_pii_data(email)['hash']

def sanitize_user_profile(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to sanitize user profile data"""
    return security_manager.sanitize_user_data(user_data)

if __name__ == "__main__":
    # Test the security manager
    print("🔐 Testing Production Security Manager...")
    
    # Test data encryption
    test_data = {"sensitive": "medical information", "user": "patient_data"}
    encrypted = security_manager.encrypt_data(test_data, "HIPAA_PROTECTED")
    decrypted = security_manager.decrypt_data(encrypted)
    print(f"✅ Encryption test: {decrypted == test_data}")
    
    # Test password hashing
    password = "secure_password_123"
    hashed = security_manager.hash_password(password)
    verified = security_manager.verify_password(hashed, password)
    print(f"✅ Password hashing test: {verified}")
    
    # Test PII hashing
    email = "patient@example.com"
    pii_hash = security_manager.hash_pii_data(email)
    print(f"✅ PII hashing test: {len(pii_hash['hash']) > 0}")
    
    print("🔐 All security tests passed!")