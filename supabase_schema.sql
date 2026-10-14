-- Users Table for Authentication and User Management
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(32) UNIQUE NOT NULL, -- Generated from email hash
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(200),
    picture TEXT, -- User profile image URL
    auth_provider VARCHAR(50) DEFAULT 'google', -- google, facebook, etc.
    is_active BOOLEAN DEFAULT true,
    last_login TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for users table
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_user_id ON users(user_id);
CREATE INDEX IF NOT EXISTS idx_users_auth_provider ON users(auth_provider);

-- Patient Profiles Table for MedBot
CREATE TABLE IF NOT EXISTS patient_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    phone_number VARCHAR(20),
    date_of_birth DATE,
    gender VARCHAR(20),
    
    -- Physical Measurements (MANDATORY)
    weight DECIMAL(5,2) NOT NULL,
    height DECIMAL(5,2) NOT NULL,
    blood_type VARCHAR(10),
    systolic_bp INTEGER,
    diastolic_bp INTEGER,
    
    -- Sleep Cycle (MANDATORY)
    sleep_hours DECIMAL(3,1) NOT NULL,
    sleep_quality VARCHAR(20) NOT NULL,
    bedtime TIME NOT NULL,
    wake_time TIME NOT NULL,
    
    -- Lifestyle Factors
    smoking_status VARCHAR(20),
    alcohol_consumption VARCHAR(20),
    exercise_frequency VARCHAR(20),
    diet_type VARCHAR(30),
    
    -- Emergency Contact (MANDATORY)
    emergency_contact_name VARCHAR(100) NOT NULL,
    emergency_contact_phone VARCHAR(20) NOT NULL,
    emergency_relationship VARCHAR(30) NOT NULL,
    emergency_contact_email VARCHAR(255),
    emergency_contact_address TEXT,
    medical_authorization BOOLEAN NOT NULL DEFAULT false,
    
    -- Medical History
    chronic_conditions JSONB DEFAULT '[]'::jsonb,
    allergies JSONB DEFAULT '[]'::jsonb,
    medications JSONB DEFAULT '[]'::jsonb,
    
    -- Calculated Health Metrics
    bmi DECIMAL(4,2),
    bmi_category VARCHAR(20),
    health_score INTEGER,
    
    -- Raw Data Storage
    raw_form_data JSONB,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_weight CHECK (weight > 0 AND weight < 1000),
    CONSTRAINT valid_height CHECK (height > 0 AND height < 300),
    CONSTRAINT valid_sleep_hours CHECK (sleep_hours > 0 AND sleep_hours <= 12),
    CONSTRAINT valid_bmi CHECK (bmi IS NULL OR (bmi > 0 AND bmi < 100))
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_patient_profiles_email ON patient_profiles(email);
CREATE INDEX IF NOT EXISTS idx_patient_profiles_created_at ON patient_profiles(created_at);
CREATE INDEX IF NOT EXISTS idx_patient_profiles_health_score ON patient_profiles(health_score);

-- Enable Row Level Security
ALTER TABLE patient_profiles ENABLE ROW LEVEL SECURITY;

-- Create policy for users to only access their own data
CREATE POLICY IF NOT EXISTS "Users can only access their own profile" ON patient_profiles
    FOR ALL USING (auth.email() = email);

-- Create trigger for updating the updated_at column
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER IF NOT EXISTS update_patient_profiles_updated_at 
    BEFORE UPDATE ON patient_profiles 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Chat History Table (for storing conversations with patient context)
CREATE TABLE IF NOT EXISTS chat_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patient_profiles(id) ON DELETE CASCADE,
    user_message TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    session_id VARCHAR(100),
    message_metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Index for better query performance
    INDEX idx_chat_history_patient_id (patient_id),
    INDEX idx_chat_history_session_id (session_id),
    INDEX idx_chat_history_created_at (created_at)
);

-- Enable RLS for chat history
ALTER TABLE chat_history ENABLE ROW LEVEL SECURITY;

-- Policy for chat history access
CREATE POLICY IF NOT EXISTS "Users can only access their own chat history" ON chat_history
    FOR ALL USING (
        patient_id IN (
            SELECT id FROM patient_profiles WHERE email = auth.email()
        )
    );