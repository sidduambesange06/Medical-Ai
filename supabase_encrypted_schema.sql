-- 🔐 INTELLIGENT SUPABASE ENCRYPTED AUTHENTICATION SCHEMA
-- =====================================================
-- Enterprise-grade encrypted user management for MedBot-v2
-- HIPAA-compliant encryption for medical data
-- Zero-knowledge architecture for sensitive information

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_crypto";

-- ==================== ENCRYPTED USER PROFILES TABLE ====================
CREATE TABLE IF NOT EXISTS encrypted_user_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT UNIQUE NOT NULL,
    email_hash TEXT UNIQUE NOT NULL, -- SHA-256 hash for indexing (searchable)
    email_encrypted TEXT NOT NULL,   -- Encrypted actual email (zero-knowledge)
    name_encrypted TEXT,             -- Encrypted user name
    phone_encrypted TEXT,            -- Encrypted phone number
    medical_data_encrypted TEXT,     -- HIPAA-compliant encrypted medical data
    preferences_encrypted TEXT,      -- Encrypted user preferences
    emergency_contacts_encrypted TEXT, -- Encrypted emergency contacts
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    encryption_version INTEGER DEFAULT 1, -- For key rotation tracking
    is_verified BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Indexes for performance
    CONSTRAINT user_id_format CHECK (user_id ~ '^user_[a-f0-9]{32}$'),
    CONSTRAINT email_hash_format CHECK (char_length(email_hash) = 64)
);

-- Create indexes for encrypted user profiles
CREATE INDEX IF NOT EXISTS idx_encrypted_user_profiles_email_hash ON encrypted_user_profiles(email_hash);
CREATE INDEX IF NOT EXISTS idx_encrypted_user_profiles_user_id ON encrypted_user_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_encrypted_user_profiles_created_at ON encrypted_user_profiles(created_at);
CREATE INDEX IF NOT EXISTS idx_encrypted_user_profiles_last_login ON encrypted_user_profiles(last_login);
CREATE INDEX IF NOT EXISTS idx_encrypted_user_profiles_is_active ON encrypted_user_profiles(is_active);

-- ==================== ENCRYPTED SESSIONS TABLE ====================
CREATE TABLE IF NOT EXISTS encrypted_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id TEXT UNIQUE NOT NULL,
    user_id TEXT NOT NULL,
    session_data_encrypted TEXT NOT NULL, -- Encrypted session payload
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    ip_address_hash TEXT,               -- Hashed IP for privacy
    user_agent_hash TEXT,               -- Hashed user agent
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Foreign key relationship
    FOREIGN KEY (user_id) REFERENCES encrypted_user_profiles(user_id) ON DELETE CASCADE,
    
    -- Constraints
    CONSTRAINT session_id_format CHECK (session_id ~ '^session_[a-f0-9]{64}$'),
    CONSTRAINT expires_at_future CHECK (expires_at > created_at)
);

-- Create indexes for encrypted sessions
CREATE INDEX IF NOT EXISTS idx_encrypted_sessions_session_id ON encrypted_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_encrypted_sessions_user_id ON encrypted_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_encrypted_sessions_expires_at ON encrypted_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_encrypted_sessions_is_active ON encrypted_sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_encrypted_sessions_created_at ON encrypted_sessions(created_at);

-- ==================== SECURITY AUDIT LOG TABLE ====================
CREATE TABLE IF NOT EXISTS security_audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type TEXT NOT NULL,
    user_email_hash TEXT,              -- Hashed email for privacy
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    details_encrypted TEXT,            -- Encrypted event details
    ip_address_hash TEXT,              -- Hashed IP address
    user_agent_hash TEXT,              -- Hashed user agent
    risk_level TEXT DEFAULT 'low',     -- low, medium, high, critical
    
    -- Constraints
    CONSTRAINT event_type_valid CHECK (event_type IN (
        'login_success', 'login_failed', 'logout', 'password_change',
        'email_change', 'profile_update', 'suspicious_activity',
        'data_access', 'key_rotation'
    )),
    CONSTRAINT risk_level_valid CHECK (risk_level IN ('low', 'medium', 'high', 'critical'))
);

-- Create indexes for security audit log
CREATE INDEX IF NOT EXISTS idx_security_audit_log_event_type ON security_audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_security_audit_log_user_email_hash ON security_audit_log(user_email_hash);
CREATE INDEX IF NOT EXISTS idx_security_audit_log_timestamp ON security_audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_security_audit_log_risk_level ON security_audit_log(risk_level);

-- ==================== ENCRYPTED MEDICAL RECORDS TABLE ====================
CREATE TABLE IF NOT EXISTS encrypted_medical_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    record_type TEXT NOT NULL,         -- prescription, lab_result, image_analysis, etc.
    medical_data_encrypted TEXT NOT NULL, -- HIPAA-compliant encrypted medical data
    metadata_encrypted TEXT,           -- Encrypted metadata (doctor, facility, etc.)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    encryption_version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    consent_given BOOLEAN DEFAULT FALSE, -- Explicit user consent for medical data
    
    -- Foreign key relationship
    FOREIGN KEY (user_id) REFERENCES encrypted_user_profiles(user_id) ON DELETE CASCADE,
    
    -- Constraints
    CONSTRAINT record_type_valid CHECK (record_type IN (
        'prescription', 'lab_result', 'image_analysis', 'vital_signs',
        'medical_history', 'allergy', 'medication', 'appointment',
        'emergency_contact', 'insurance_info'
    )),
    CONSTRAINT consent_required CHECK (consent_given = TRUE) -- Enforce consent
);

-- Create indexes for encrypted medical records
CREATE INDEX IF NOT EXISTS idx_encrypted_medical_records_user_id ON encrypted_medical_records(user_id);
CREATE INDEX IF NOT EXISTS idx_encrypted_medical_records_record_type ON encrypted_medical_records(record_type);
CREATE INDEX IF NOT EXISTS idx_encrypted_medical_records_created_at ON encrypted_medical_records(created_at);
CREATE INDEX IF NOT EXISTS idx_encrypted_medical_records_is_active ON encrypted_medical_records(is_active);

-- ==================== ENCRYPTION KEY MANAGEMENT TABLE ====================
CREATE TABLE IF NOT EXISTS encryption_key_management (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_id TEXT UNIQUE NOT NULL,
    key_purpose TEXT NOT NULL,         -- user_data, medical_data, sessions, audit
    key_version INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    rotated_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    algorithm TEXT DEFAULT 'ChaCha20-Poly1305',
    
    -- Constraints
    CONSTRAINT key_purpose_valid CHECK (key_purpose IN (
        'user_data', 'medical_data', 'sessions', 'audit', 'master'
    )),
    CONSTRAINT algorithm_valid CHECK (algorithm IN (
        'AES-256-GCM', 'ChaCha20-Poly1305', 'AES-128-CBC-HMAC'
    ))
);

-- Create indexes for encryption key management
CREATE INDEX IF NOT EXISTS idx_encryption_key_management_key_id ON encryption_key_management(key_id);
CREATE INDEX IF NOT EXISTS idx_encryption_key_management_key_purpose ON encryption_key_management(key_purpose);
CREATE INDEX IF NOT EXISTS idx_encryption_key_management_is_active ON encryption_key_management(is_active);

-- ==================== RATE LIMITING TABLE ====================
CREATE TABLE IF NOT EXISTS rate_limiting (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    identifier TEXT NOT NULL,          -- IP address hash, user ID, email hash
    identifier_type TEXT NOT NULL,     -- ip, user, email
    request_count INTEGER DEFAULT 1,
    window_start TIMESTAMPTZ DEFAULT NOW(),
    window_end TIMESTAMPTZ DEFAULT NOW() + INTERVAL '1 hour',
    is_blocked BOOLEAN DEFAULT FALSE,
    
    -- Constraints
    CONSTRAINT identifier_type_valid CHECK (identifier_type IN ('ip', 'user', 'email')),
    CONSTRAINT window_valid CHECK (window_end > window_start)
);

-- Create indexes for rate limiting
CREATE INDEX IF NOT EXISTS idx_rate_limiting_identifier ON rate_limiting(identifier);
CREATE INDEX IF NOT EXISTS idx_rate_limiting_window_end ON rate_limiting(window_end);
CREATE INDEX IF NOT EXISTS idx_rate_limiting_is_blocked ON rate_limiting(is_blocked);

-- ==================== TRIGGERS FOR AUTOMATIC UPDATES ====================

-- Trigger function to update 'updated_at' timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to all relevant tables
CREATE TRIGGER update_encrypted_user_profiles_updated_at
    BEFORE UPDATE ON encrypted_user_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_encrypted_sessions_updated_at
    BEFORE UPDATE ON encrypted_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_encrypted_medical_records_updated_at
    BEFORE UPDATE ON encrypted_medical_records
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ==================== ROW LEVEL SECURITY (RLS) ====================

-- Enable RLS on all tables
ALTER TABLE encrypted_user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE encrypted_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE security_audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE encrypted_medical_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE encryption_key_management ENABLE ROW LEVEL SECURITY;
ALTER TABLE rate_limiting ENABLE ROW LEVEL SECURITY;

-- ==================== SECURITY POLICIES ====================

-- User profiles: Users can only access their own data
CREATE POLICY "Users can view own profile" ON encrypted_user_profiles
    FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can update own profile" ON encrypted_user_profiles
    FOR UPDATE USING (auth.uid()::text = user_id);

-- Sessions: Users can only access their own sessions
CREATE POLICY "Users can view own sessions" ON encrypted_sessions
    FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can update own sessions" ON encrypted_sessions
    FOR UPDATE USING (auth.uid()::text = user_id);

-- Medical records: Users can only access their own medical data
CREATE POLICY "Users can view own medical records" ON encrypted_medical_records
    FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can insert own medical records" ON encrypted_medical_records
    FOR INSERT WITH CHECK (auth.uid()::text = user_id);

CREATE POLICY "Users can update own medical records" ON encrypted_medical_records
    FOR UPDATE USING (auth.uid()::text = user_id);

-- ==================== CLEANUP FUNCTIONS ====================

-- Function to clean up expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM encrypted_sessions 
    WHERE expires_at < NOW() AND is_active = FALSE;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Also cleanup very old active sessions (30 days)
    UPDATE encrypted_sessions 
    SET is_active = FALSE, updated_at = NOW()
    WHERE created_at < NOW() - INTERVAL '30 days' AND is_active = TRUE;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old audit logs (keep 1 year)
CREATE OR REPLACE FUNCTION cleanup_old_audit_logs()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM security_audit_log 
    WHERE timestamp < NOW() - INTERVAL '1 year';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up rate limiting data
CREATE OR REPLACE FUNCTION cleanup_rate_limiting()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM rate_limiting 
    WHERE window_end < NOW() - INTERVAL '24 hours';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ==================== SCHEDULED CLEANUP (if pg_cron is available) ====================

-- Note: These require the pg_cron extension to be installed
-- Uncomment if pg_cron is available in your Supabase instance

-- Clean up expired sessions daily at 2 AM
-- SELECT cron.schedule('cleanup-expired-sessions', '0 2 * * *', 'SELECT cleanup_expired_sessions();');

-- Clean up old audit logs weekly on Sundays at 3 AM  
-- SELECT cron.schedule('cleanup-old-audit-logs', '0 3 * * 0', 'SELECT cleanup_old_audit_logs();');

-- Clean up rate limiting data every 6 hours
-- SELECT cron.schedule('cleanup-rate-limiting', '0 */6 * * *', 'SELECT cleanup_rate_limiting();');

-- ==================== VIEWS FOR ANALYTICS ====================

-- Security analytics view (aggregated, no sensitive data)
CREATE OR REPLACE VIEW security_analytics AS
SELECT 
    event_type,
    DATE_TRUNC('hour', timestamp) as event_hour,
    COUNT(*) as event_count,
    COUNT(DISTINCT user_email_hash) as unique_users,
    risk_level
FROM security_audit_log
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY event_type, DATE_TRUNC('hour', timestamp), risk_level
ORDER BY event_hour DESC;

-- User activity summary view
CREATE OR REPLACE VIEW user_activity_summary AS
SELECT 
    user_id,
    created_at as registration_date,
    last_login,
    is_active,
    is_verified,
    CASE 
        WHEN last_login >= NOW() - INTERVAL '7 days' THEN 'active'
        WHEN last_login >= NOW() - INTERVAL '30 days' THEN 'inactive'
        ELSE 'dormant'
    END as activity_status
FROM encrypted_user_profiles;

-- ==================== GRANT PERMISSIONS ====================

-- Grant appropriate permissions to authenticated users
GRANT SELECT, INSERT, UPDATE ON encrypted_user_profiles TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON encrypted_sessions TO authenticated;
GRANT SELECT, INSERT ON security_audit_log TO authenticated;
GRANT SELECT, INSERT, UPDATE ON encrypted_medical_records TO authenticated;
GRANT SELECT ON security_analytics TO authenticated;
GRANT SELECT ON user_activity_summary TO authenticated;

-- Grant permissions for cleanup functions
GRANT EXECUTE ON FUNCTION cleanup_expired_sessions() TO authenticated;
GRANT EXECUTE ON FUNCTION cleanup_old_audit_logs() TO authenticated;
GRANT EXECUTE ON FUNCTION cleanup_rate_limiting() TO authenticated;

-- ==================== COMMENTS ====================

COMMENT ON TABLE encrypted_user_profiles IS 'Encrypted user profiles with zero-knowledge sensitive data storage';
COMMENT ON TABLE encrypted_sessions IS 'Encrypted session management with automatic expiration';
COMMENT ON TABLE security_audit_log IS 'Comprehensive security event logging with encrypted details';
COMMENT ON TABLE encrypted_medical_records IS 'HIPAA-compliant encrypted medical data storage';
COMMENT ON TABLE encryption_key_management IS 'Encryption key lifecycle management and rotation tracking';
COMMENT ON TABLE rate_limiting IS 'Rate limiting and abuse prevention system';

COMMENT ON COLUMN encrypted_user_profiles.email_hash IS 'SHA-256 hash of email for indexing and search';
COMMENT ON COLUMN encrypted_user_profiles.email_encrypted IS 'AES-256 encrypted email address (zero-knowledge)';
COMMENT ON COLUMN encrypted_user_profiles.medical_data_encrypted IS 'ChaCha20-Poly1305 encrypted medical data (HIPAA-compliant)';
COMMENT ON COLUMN encrypted_sessions.session_data_encrypted IS 'Encrypted session payload with user permissions';
COMMENT ON COLUMN security_audit_log.details_encrypted IS 'Encrypted security event details';
COMMENT ON COLUMN encrypted_medical_records.consent_given IS 'Explicit user consent required for medical data processing';

-- ==================== SECURITY REMINDER ====================

-- 🔐 IMPORTANT SECURITY NOTES:
-- 1. Never store encryption keys in the database
-- 2. Use environment variables for encryption keys
-- 3. Implement key rotation every 90 days
-- 4. Monitor security_audit_log for suspicious activity
-- 5. Regular backup of encrypted data with key management
-- 6. Ensure HTTPS only for all API communications
-- 7. Implement proper session timeout mechanisms
-- 8. Use prepared statements to prevent SQL injection
-- 9. Regular security audits and penetration testing
-- 10. Comply with HIPAA, GDPR, and other applicable regulations

SELECT 'Intelligent Supabase encrypted authentication schema created successfully! 🔐' as status;