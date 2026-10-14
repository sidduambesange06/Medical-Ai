"""
Intelligent Greeting System - User, Login, and Time Aware
Provides personalized greetings based on:
- User authentication status
- Time of day
- User's name and profile
- Previous interactions
- Medical context
"""
import json
import logging
from datetime import datetime, time
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

class IntelligentGreetingSystem:
    """
    Intelligent Greeting System.

    Generates personalized greetings based on user authentication status,
    time of day, user profile, and previous interaction history.
    """
    
    def __init__(self):
        self.greeting_cache = {}
        logger.info("[GREETING-INIT] Intelligent Greeting System initialized")
    
    def generate_intelligent_greeting(self, user_context: Dict = None, session_data: Dict = None) -> Dict:
        """
        🎯 MAIN INTELLIGENT GREETING GENERATOR
        
        Args:
            user_context: User data from database/session
            session_data: Session-specific data
            
        Returns:
            Dict with greeting message and metadata
        """
        try:
            # Get current time context
            time_context = self._get_time_context()
            
            # Determine user status
            user_status = self._analyze_user_status(user_context, session_data)
            
            # Generate personalized greeting
            greeting_data = self._generate_contextual_greeting(time_context, user_status, user_context)
            
            logger.info(f"[GREETING] Generated {user_status['type']} greeting for user: {user_status.get('name', 'Anonymous')}")
            
            return {
                "has_greeting": True,
                "message": greeting_data["message"],
                "user_type": user_status["type"],
                "time_context": time_context["period"],
                "personalized": user_status["is_authenticated"],
                "metadata": greeting_data["metadata"]
            }
            
        except Exception as e:
            logger.error(f"[GREETING-ERROR] Greeting generation failed: {e}")
            return self._get_fallback_greeting()
    
    def _get_time_context(self) -> Dict:
        """🕐 GET TIME-AWARE CONTEXT"""
        now = datetime.now()
        current_time = now.time()
        
        # Define time periods
        if time(5, 0) <= current_time < time(12, 0):
            period = "morning"
            greeting_prefix = "Good morning"
            energy_level = "fresh"
        elif time(12, 0) <= current_time < time(17, 0):
            period = "afternoon"
            greeting_prefix = "Good afternoon"
            energy_level = "productive"
        elif time(17, 0) <= current_time < time(21, 0):
            period = "evening"
            greeting_prefix = "Good evening"
            energy_level = "relaxed"
        else:
            period = "night"
            greeting_prefix = "Good evening"
            energy_level = "calm"
        
        return {
            "period": period,
            "greeting_prefix": greeting_prefix,
            "energy_level": energy_level,
            "hour": now.hour,
            "timestamp": now.isoformat()
        }
    
    def _analyze_user_status(self, user_context: Dict = None, session_data: Dict = None) -> Dict:
        """👤 ANALYZE USER AUTHENTICATION AND CONTEXT STATUS with LOGIN SCENARIO DETECTION"""
        
        status = {
            "is_authenticated": False,
            "type": "guest",
            "name": None,
            "email": None,
            "has_medical_profile": False,
            "returning_user": False,
            "auth_provider": None,
            "login_scenario": "unknown",
            "session_age": "unknown"
        }
        
        # Check session data first
        if session_data:
            status["is_authenticated"] = session_data.get('authenticated', False)
            status["email"] = session_data.get('user_email')
            
            # Detect login scenario based on session state
            greeting_shown = session_data.get('greeting_shown', False)
            last_greeting = session_data.get('last_greeting_time')
            
            if not greeting_shown:
                status["login_scenario"] = "fresh_login"  # New login or first visit
            elif last_greeting:
                from datetime import datetime
                try:
                    last_time = datetime.fromisoformat(last_greeting)
                    time_diff = datetime.now() - last_time
                    
                    if time_diff.total_seconds() < 1800:  # Less than 30 minutes
                        status["login_scenario"] = "same_session"
                        status["session_age"] = "recent"
                    elif time_diff.total_seconds() < 14400:  # Less than 4 hours
                        status["login_scenario"] = "re_login"
                        status["session_age"] = "medium"
                    else:
                        status["login_scenario"] = "new_day_login"
                        status["session_age"] = "old"
                except:
                    status["login_scenario"] = "fresh_login"
            
        # Check user context from database
        if user_context:
            status["is_authenticated"] = True
            status["name"] = user_context.get('name', '').split()[0] if user_context.get('name') else None  # First name only
            status["email"] = user_context.get('email')
            status["auth_provider"] = user_context.get('auth_provider', 'unknown')
            status["returning_user"] = bool(user_context.get('last_login'))
            
            # Enhanced profile detection
            medical_keys = ['medical_conditions', 'allergies', 'medications', 'chronic_conditions', 'health_history']
            if any(key in user_context and user_context.get(key) for key in medical_keys):
                status["has_medical_profile"] = True
        
        # Determine user type with scenario awareness
        if status["is_authenticated"]:
            if status["has_medical_profile"]:
                if status["login_scenario"] == "fresh_login":
                    status["type"] = "returning_user_with_profile"
                else:
                    status["type"] = "authenticated_with_profile"
            else:
                if status["login_scenario"] == "fresh_login" and status["returning_user"]:
                    status["type"] = "returning_user_new"
                elif status["login_scenario"] == "fresh_login":
                    status["type"] = "first_time_user"
                else:
                    status["type"] = "authenticated_new"
        else:
            status["type"] = "guest"
        
        return status
    
    def _generate_contextual_greeting(self, time_context: Dict, user_status: Dict, user_context: Dict = None) -> Dict:
        """🎨 GENERATE CONTEXTUAL GREETING BASED ON ALL FACTORS"""
        
        time_prefix = time_context["greeting_prefix"]
        user_type = user_status["type"]
        
        if user_type == "guest":
            # Guest user greeting
            message = f"""{time_prefix}! 👋

I'm **MedAI**, your intelligent medical diagnostic assistant. I'm here to help you with:

🩺 **Symptom Analysis** - Describe your symptoms for professional diagnostic insights
📚 **Medical Information** - Get evidence-based information from medical literature  
🔍 **Health Guidance** - Understand medical conditions and health concerns
🏥 **Professional Support** - Information to discuss with your healthcare provider

**To get started:**
Simply describe your symptoms or ask any medical question. I'll provide comprehensive analysis based on medical knowledge.

⚠️ **Important**: I provide diagnostic support and medical information only. For treatment and medication advice, always consult your healthcare provider.

What health concerns can I help you analyze today?"""
            
            metadata = {
                "greeting_type": "guest_welcome",
                "features_highlighted": ["symptom_analysis", "medical_info", "health_guidance"],
                "call_to_action": "describe_symptoms"
            }
            
        elif user_type == "authenticated_new":
            # New authenticated user
            name = user_status.get("name", "there")
            auth_provider = (user_status.get("auth_provider") or "").title()
            
            message = f"""{time_prefix}, {name}! 🌟

Welcome to **MedAI**! I'm your personalized medical diagnostic assistant.

Since this is your first visit, here's how I can help you:

🏥 **Personalized Health Analysis** - I can learn your medical history for tailored insights
📋 **Comprehensive Symptom Assessment** - Advanced diagnostic questioning like a real doctor
🔍 **Evidence-Based Information** - Medical knowledge from authoritative textbooks
👨‍⚕️ **Professional Healthcare Support** - Information to enhance your medical consultations

**Getting Started:**
1. Tell me about any current symptoms or health concerns
2. Optionally, you can build your health profile for more personalized insights
3. Ask any medical questions - I'll provide thorough, evidence-based responses

⚠️ **Medical Disclaimer**: I provide diagnostic support and educational information only. Always consult healthcare professionals for treatment decisions.

What would you like to discuss about your health today?"""
            
            metadata = {
                "greeting_type": "new_user_onboarding",
                "auth_provider": auth_provider,
                "next_steps": ["health_profile_creation", "first_consultation"],
                "personalization_level": "basic"
            }
            
        elif user_type == "authenticated_with_profile":
            # Returning user with medical profile
            name = user_status.get("name", "there")
            
            # Get medical context if available
            medical_summary = ""
            if user_context:
                conditions = user_context.get('medical_conditions', [])
                allergies = user_context.get('allergies', [])
                medications = user_context.get('medications', [])
                
                profile_parts = []
                if conditions:
                    profile_parts.append(f"{len(conditions)} documented condition(s)")
                if allergies:
                    profile_parts.append(f"{len(allergies)} known allergy(ies)")
                if medications:
                    profile_parts.append(f"{len(medications)} current medication(s)")
                
                if profile_parts:
                    medical_summary = f"\\n\\n🏥 **Your Health Profile**: {' | '.join(profile_parts)}"
            
            message = f"""{time_prefix}, {name}! 👨‍⚕️

Welcome back to **MedAI**! I'm ready to provide personalized medical insights based on your health profile.{medical_summary}

**Today I can help you with:**
🔍 **Symptom Analysis** - Tailored to your medical history and risk factors
📊 **Health Monitoring** - Track patterns in your health concerns  
🩺 **Condition Management** - Information relevant to your existing conditions
💡 **Personalized Insights** - Medical guidance considering your unique profile

**What's New:**
• Enhanced diagnostic questioning based on your health history
• Personalized risk assessment for symptoms
• Context-aware medical information retrieval
• Allergy and medication interaction awareness

⚠️ **Personalized Care**: All responses consider your health profile. For medication changes or urgent concerns, consult your healthcare provider immediately.

How can I assist with your health today? Any new symptoms or concerns to discuss?"""
            
            metadata = {
                "greeting_type": "returning_user_personalized",
                "has_medical_profile": True,
                "personalization_level": "advanced",
                "profile_elements": ["conditions", "allergies", "medications"] if user_context else []
            }
            
        elif user_type == "first_time_user":
            # Brand new user - first time ever
            name = user_status.get("name", "there")
            auth_provider = (user_status.get("auth_provider") or "").title()
            
            message = f"""{time_prefix}, {name}! 🎆

**Welcome to MedAI** - Your Intelligent Medical Diagnostic Assistant!

I'm excited to meet you! As your personal AI medical companion, I'm here to:

🩺 **Analyze Symptoms** - Describe any health concerns for professional diagnostic insights
📚 **Provide Medical Knowledge** - Access evidence-based information from medical literature
🔬 **Offer Health Guidance** - Understand conditions, treatments, and preventive care
🏥 **Support Healthcare Decisions** - Information to discuss with your healthcare provider

**Let's Get Started:**
• Share any current symptoms or health questions
• Ask about medical conditions, medications, or treatments
• Get guidance on when to seek medical attention

⚠️ **Important Note**: I provide diagnostic support and medical education. For treatment decisions and prescriptions, always consult qualified healthcare professionals.

What brings you to seek medical guidance today?"""
            
            metadata = {
                "greeting_type": "first_time_user_welcome",
                "auth_provider": auth_provider,
                "login_scenario": "first_visit",
                "next_steps": ["symptom_discussion", "medical_question", "health_profile_setup"]
            }
            
        elif user_type == "returning_user_new":
            # Returning user without medical profile
            name = user_status.get("name", "there")
            scenario = user_status.get("login_scenario", "unknown")
            
            message = f"""{time_prefix}, {name}! 🙏

Welcome back to **MedAI**! Great to see you again.

🔄 **Ready for Your Next Consultation**

How can I assist with your health today?

🩺 **Quick Medical Help:**
• Describe new symptoms for analysis
• Ask about medications or treatments
• Get information about medical conditions
• Understand test results or procedures

📋 **Profile Enhancement:** Consider building your health profile for more personalized medical insights tailored to your specific needs.

⚠️ **Medical Guidance**: All responses are for informational and diagnostic support purposes. Consult healthcare professionals for treatment decisions.

What health concerns would you like to discuss today?"""
            
            metadata = {
                "greeting_type": "returning_user_basic",
                "login_scenario": scenario,
                "personalization_level": "returning",
                "profile_suggestion": True
            }
            
        elif user_type == "returning_user_with_profile":
            # Returning user with full medical profile
            name = user_status.get("name", "there")
            scenario = user_status.get("login_scenario", "unknown")
            
            # Get medical context if available
            medical_summary = ""
            if user_context:
                conditions = user_context.get('medical_conditions', [])
                allergies = user_context.get('allergies', [])
                medications = user_context.get('medications', [])
                
                profile_parts = []
                if conditions:
                    profile_parts.append(f"{len(conditions)} documented condition(s)")
                if allergies:
                    profile_parts.append(f"{len(allergies)} known allergy(ies)")
                if medications:
                    profile_parts.append(f"{len(medications)} current medication(s)")
                
                if profile_parts:
                    medical_summary = f"\\n\\n🏥 **Your Health Profile Active**: {' | '.join(profile_parts)}"
            
            if scenario == "fresh_login":
                greeting_action = "Welcome back"
                session_note = "Fresh login detected - ready for new consultation"
            elif scenario == "re_login":
                greeting_action = "Welcome back"
                session_note = "Continuing your medical consultations"
            elif scenario == "new_day_login":
                greeting_action = "Good to see you again"
                session_note = "New day, new opportunities for health guidance"
            else:
                greeting_action = "Welcome back"
                session_note = "Ready to continue our medical discussions"
            
            message = f"""{time_prefix}, {name}! 👨‍⚕️

{greeting_action} to **MedAI**! {session_note}.{medical_summary}

🔬 **Personalized Medical Analysis Ready:**
• Symptom assessment considering your health history
• Medication interaction awareness
• Condition-specific medical guidance
• Risk factor analysis for your profile

🎯 **Today's Medical Support:**
• New symptoms or changes in existing conditions
• Medication questions or side effects
• Treatment progress discussions
• Preventive care recommendations

⚠️ **Personalized Care Notice**: All responses consider your documented health profile. For urgent symptoms or medication changes, contact your healthcare provider immediately.

How can I help with your health today? Any new symptoms or medical questions?"""
            
            metadata = {
                "greeting_type": "returning_user_personalized",
                "login_scenario": scenario,
                "has_medical_profile": True,
                "personalization_level": "advanced",
                "session_note": session_note
            }
        
        else:
            # Enhanced fallback greeting with scenario detection
            scenario = user_status.get("login_scenario", "unknown")
            name = user_status.get("name", "there") if user_status.get("name") else "there"
            
            message = f"""{time_prefix}, {name}! 👋

**Welcome to MedAI** - Your Intelligent Medical Assistant!

I'm here to provide professional medical guidance and diagnostic support.

🩺 **How I Can Help:**
• Analyze symptoms and health concerns
• Provide evidence-based medical information
• Explain conditions, treatments, and medications
• Guide you on when to seek medical care

⚠️ **Important**: I provide diagnostic support and medical education only. Always consult healthcare professionals for treatment decisions.

What health concerns would you like to discuss?"""
            
            metadata = {
                "greeting_type": "fallback_intelligent",
                "login_scenario": scenario,
                "reason": "unknown_user_type"
            }
        
        return {
            "message": message,
            "metadata": metadata
        }
    
    def _get_fallback_greeting(self) -> Dict:
        """🔧 FALLBACK GREETING FOR ERROR CASES"""
        return {
            "has_greeting": True,
            "message": """Hello! I'm MedAI, your intelligent medical diagnostic assistant.

🩺 **I can help you with:**
• Symptom analysis and diagnostic insights
• Medical information from authoritative sources
• Health guidance and condition explanations
• Professional healthcare support

⚠️ **Important**: I provide diagnostic support and medical education only. Always consult healthcare professionals for treatment decisions.

What symptoms or health concerns would you like to discuss?""",
            "user_type": "unknown",
            "time_context": "unknown",
            "personalized": False,
            "metadata": {"greeting_type": "fallback", "reason": "error_recovery"}
        }
    
    def should_show_greeting(self, session_data: Dict = None, user_context: Dict = None) -> bool:
        """🤔 INTELLIGENT DECISION: Should we show greeting for ALL login scenarios?"""
        try:
            from datetime import datetime, timedelta
            
            # Always show greeting for completely new sessions (no session data)
            if not session_data:
                logger.info("[GREETING-DECISION] New session - showing greeting")
                return True
            
            user_email = session_data.get('user_email')
            is_authenticated = session_data.get('authenticated', False)
            
            # SCENARIO 1: Guest users - show greeting once per session
            if not is_authenticated or not user_email:
                greeting_shown = session_data.get('greeting_shown', False)
                logger.info(f"[GREETING-DECISION] Guest user - greeting_shown: {greeting_shown}")
                return not greeting_shown
            
            # SCENARIO 2: Authenticated users - more intelligent logic
            current_time = datetime.now()
            
            # Check if this is a fresh login (no greeting_shown yet in session)
            session_greeting_shown = session_data.get('greeting_shown', False)
            last_session_greeting = session_data.get('last_greeting_time')
            
            # If no greeting shown in this session yet, it's a new login
            if not session_greeting_shown:
                logger.info("[GREETING-DECISION] New login session - showing greeting")
                return True
            
            # If greeting was shown in this session, check timing rules
            if last_session_greeting:
                try:
                    last_time = datetime.fromisoformat(last_session_greeting)
                    time_since_last = current_time - last_time
                    
                    # Show greeting again for re-login scenarios:
                    # 1. If it's been more than 30 minutes (likely new login)
                    # 2. If it's a different day
                    # 3. If it's been more than 4 hours (new session)
                    
                    if time_since_last > timedelta(hours=4):
                        logger.info("[GREETING-DECISION] Long time gap (4+ hours) - showing greeting")
                        return True
                    
                    if last_time.date() != current_time.date():
                        logger.info("[GREETING-DECISION] Different day - showing greeting")
                        return True
                    
                    if time_since_last > timedelta(minutes=30):
                        logger.info(f"[GREETING-DECISION] Re-login detected ({time_since_last.total_seconds()//60:.0f} min gap) - showing greeting")
                        return True
                        
                except Exception as time_error:
                    logger.warning(f"[GREETING-DECISION] Time parsing error: {time_error} - defaulting to show greeting")
                    return True
            
            # Default: Don't show if already shown recently in same session
            logger.info("[GREETING-DECISION] Recent greeting exists - not showing")
            return False
            
        except Exception as e:
            logger.error(f"[GREETING-DECISION] Error in greeting decision: {e}")
            return True  # Default to showing greeting on error