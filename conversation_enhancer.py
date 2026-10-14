"""
Conversation enhancer for MedBot.
Compatible with existing MedBot architecture.
Enhances conversation context without conflicts.
"""

import os
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from functools import wraps

logger = logging.getLogger(__name__)

class ConversationEnhancer:
    """
    Advanced conversation enhancer that integrates with existing systems
    NO CONFLICTS - Only enhances existing functionality
    """
    
    def __init__(self):
        self.response_cache = {}
        self.conversation_contexts = {}
        self.error_patterns = {}
        self.enhancement_stats = {
            'total_enhancements': 0,
            'cache_hits': 0,
            'fallback_prevented': 0,
            'context_improvements': 0
        }
        
        # Load existing AI system if available
        self.existing_ai = None
        try:
            from ai_engine.intelligent_medical_system import IntelligentMedicalResponseSystem
            self.existing_ai = IntelligentMedicalResponseSystem()
            logger.info("✅ Integrated with existing AI system")
        except Exception as e:
            logger.warning(f"AI system not available: {e}")
        
        logger.info("🚀 Conversation Enhancer initialized - NO CONFLICTS")
    
    def enhance_medical_response(self, original_response_func):
        """
        DECORATOR: Enhance existing response functions without changing them
        This wraps your existing functions to make them smarter
        """
        @wraps(original_response_func)
        def enhanced_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Get user input (first argument typically)
                user_input = args[0] if args else kwargs.get('user_input', '')
                user_id = kwargs.get('user_id', 'anonymous')
                
                # PRE-PROCESSING ENHANCEMENT
                enhanced_input = self._preprocess_input(user_input, user_id)
                
                # Call original function with enhanced input
                if args:
                    # Replace first arg with enhanced input
                    enhanced_args = (enhanced_input,) + args[1:]
                    original_response = original_response_func(*enhanced_args, **kwargs)
                else:
                    kwargs['user_input'] = enhanced_input
                    original_response = original_response_func(**kwargs)
                
                # POST-PROCESSING ENHANCEMENT
                enhanced_response = self._postprocess_response(
                    original_response, user_input, user_id
                )
                
                # Update stats
                self.enhancement_stats['total_enhancements'] += 1
                processing_time = time.time() - start_time
                
                logger.info(f"✅ Enhanced response in {processing_time:.2f}s")
                return enhanced_response
                
            except Exception as e:
                logger.error(f"Enhancement error: {e}")
                # FALLBACK: Return original function result
                try:
                    return original_response_func(*args, **kwargs)
                except:
                    return self._get_safe_fallback_response(args, kwargs)
        
        return enhanced_wrapper
    
    def _preprocess_input(self, user_input: str, user_id: str) -> str:
        """
        Enhance user input before processing
        - Fix common typos in medical terms
        - Add context from conversation history
        - Normalize medical terminology
        """
        try:
            if not user_input or not isinstance(user_input, str):
                return user_input
            
            enhanced_input = user_input
            
            # Fix common medical typos
            medical_fixes = {
                'migrane': 'migraine',
                'stomache': 'stomach',
                'diarhea': 'diarrhea',
                'nausia': 'nausea',
                'symtoms': 'symptoms',
                'desease': 'disease',
                'heatache': 'headache',
                'faver': 'fever',
                'cought': 'cough',
                'diabetic': 'diabetes'
            }
            
            for typo, correction in medical_fixes.items():
                if typo in enhanced_input.lower():
                    enhanced_input = enhanced_input.replace(typo, correction)
                    logger.info(f"🔧 Fixed typo: {typo} → {correction}")
            
            # Add conversation context if available
            context = self.conversation_contexts.get(user_id, {})
            if context and context.get('previous_symptoms'):
                # Don't modify input, but store context for AI
                context['last_input'] = enhanced_input
                context['timestamp'] = datetime.now().isoformat()
            
            return enhanced_input
            
        except Exception as e:
            logger.error(f"Input preprocessing failed: {e}")
            return user_input
    
    def _postprocess_response(self, original_response: str, user_input: str, user_id: str) -> str:
        """
        Enhance the AI response after generation
        - Add medical disclaimers if missing
        - Improve formatting and readability
        - Add emergency warnings where needed
        - Cache good responses
        """
        try:
            if not original_response:
                return self._generate_smart_fallback(user_input)
            
            enhanced_response = str(original_response)
            
            # Check if response is too generic or fallback
            if self._is_fallback_response(enhanced_response):
                logger.warning("🔄 Detected fallback response - enhancing")
                enhanced_response = self._improve_fallback_response(enhanced_response, user_input)
                self.enhancement_stats['fallback_prevented'] += 1
            
            # Add medical disclaimers if missing
            if self._is_medical_response(enhanced_response) and not self._has_medical_disclaimer(enhanced_response):
                enhanced_response += "\\n\\n⚠️ **Important Medical Disclaimer**: This information is for educational purposes only and should not replace professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment."
            
            # Add emergency warnings for serious symptoms
            if self._requires_emergency_warning(user_input):
                emergency_warning = "\\n\\n🚨 **URGENT**: If you're experiencing severe symptoms, chest pain, difficulty breathing, or any life-threatening condition, please seek immediate medical attention or call emergency services."
                if emergency_warning not in enhanced_response:
                    enhanced_response = emergency_warning + "\\n\\n" + enhanced_response
            
            # Improve formatting
            enhanced_response = self._improve_formatting(enhanced_response)
            
            # Update conversation context
            self._update_conversation_context(user_id, user_input, enhanced_response)
            
            # Cache good responses
            if len(enhanced_response) > 100 and self._is_quality_response(enhanced_response):
                cache_key = self._generate_cache_key(user_input)
                self.response_cache[cache_key] = {
                    'response': enhanced_response,
                    'timestamp': datetime.now().isoformat(),
                    'user_input': user_input
                }
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Response postprocessing failed: {e}")
            return original_response or self._generate_smart_fallback(user_input)
    
    def _is_fallback_response(self, response: str) -> bool:
        """Check if response is a generic fallback"""
        fallback_indicators = [
            "I'm sorry, I don't understand",
            "Could you please rephrase",
            "I'm not sure what you're asking",
            "Can you be more specific",
            "I don't have information about that",
            "Please try again",
            "I cannot help with that"
        ]
        
        response_lower = response.lower()
        return any(indicator.lower() in response_lower for indicator in fallback_indicators)
    
    def _improve_fallback_response(self, fallback_response: str, user_input: str) -> str:
        """Convert generic fallback into helpful medical response"""
        user_input_lower = user_input.lower()
        
        # Detect medical intent
        if any(word in user_input_lower for word in ['pain', 'ache', 'hurt', 'headache']):
            return f"""I understand you're experiencing pain or discomfort. Let me help you with that.

**For pain-related concerns, I can help with:**
• Identifying potential causes of your pain
• Understanding different types of pain (acute vs chronic)
• Recognizing when pain requires immediate medical attention
• General pain management principles

**Please tell me more about:**
• Where exactly is the pain located?
• How long have you been experiencing it?
• How would you describe the pain (sharp, dull, throbbing, burning)?
• What makes it better or worse?

This information will help me provide you with more specific insights about your condition.

⚠️ **Important**: For severe or persistent pain, please consult a healthcare provider for proper evaluation."""

        elif any(word in user_input_lower for word in ['fever', 'temperature', 'hot', 'chills']):
            return f"""I can definitely help you understand fever and temperature-related symptoms.

**Fever Information I can provide:**
• Normal vs abnormal body temperatures
• Common causes of fever
• When fever becomes concerning
• Fever management principles
• Associated symptoms to monitor

**Please share more details:**
• What is your current temperature?
• How long have you had the fever?
• Any other symptoms (chills, sweating, body aches)?
• Have you taken any medications?

**🚨 Seek immediate care if:**
• Temperature over 103°F (39.4°C)
• Fever with severe headache
• Difficulty breathing
• Persistent vomiting
• Signs of dehydration

What specific fever-related questions can I help answer?"""

        elif any(word in user_input_lower for word in ['nausea', 'vomiting', 'sick', 'throw up']):
            return f"""I can help you understand nausea and vomiting symptoms.

**I can provide information about:**
• Common causes of nausea and vomiting
• When these symptoms are concerning
• Dehydration prevention
• Foods and remedies that may help

**Please tell me more:**
• How long have you been experiencing nausea/vomiting?
• Any specific triggers you've noticed?
• Are you able to keep fluids down?
• Any other symptoms (fever, abdominal pain, diarrhea)?

**🚨 Seek medical attention if:**
• Unable to keep fluids down for 24+ hours
• Signs of severe dehydration
• Blood in vomit
• Severe abdominal pain
• High fever with vomiting

What specific aspects of your nausea/vomiting would you like me to address?"""

        else:
            return f"""I'm here to help with your medical questions and health concerns.

**I specialize in:**
• Symptom analysis and assessment
• Medical condition information
• Health education and guidance
• When to seek medical care

**To provide the best help, please tell me:**
• What specific symptoms are you experiencing?
• How long have they been present?
• What's your main health concern today?

**Examples of questions I can help with:**
• "I have a headache that won't go away"
• "What could cause chest pain?"
• "I'm experiencing fatigue and dizziness"
• "Tell me about high blood pressure"

What specific health question or symptom would you like me to analyze?"""
    
    def _is_medical_response(self, response: str) -> bool:
        """Check if response contains medical content"""
        medical_terms = [
            'symptom', 'diagnosis', 'treatment', 'condition', 'disease',
            'medical', 'health', 'doctor', 'patient', 'medication',
            'therapy', 'clinical', 'healthcare', 'hospital'
        ]
        
        response_lower = response.lower()
        return any(term in response_lower for term in medical_terms)
    
    def _has_medical_disclaimer(self, response: str) -> bool:
        """Check if response already has medical disclaimer"""
        disclaimer_phrases = [
            'medical disclaimer', 'consult', 'healthcare provider',
            'professional medical advice', 'not replace', 'educational purposes'
        ]
        
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in disclaimer_phrases)
    
    def _requires_emergency_warning(self, user_input: str) -> bool:
        """Check if user input suggests emergency situation"""
        emergency_keywords = [
            'chest pain', 'can\'t breathe', 'difficulty breathing', 'severe pain',
            'blood', 'unconscious', 'emergency', 'urgent', 'severe',
            'crushing pain', 'heart attack', 'stroke'
        ]
        
        user_input_lower = user_input.lower()
        return any(keyword in user_input_lower for keyword in emergency_keywords)
    
    def _improve_formatting(self, response: str) -> str:
        """Improve response formatting for better readability"""
        try:
            # Add proper spacing around headers
            response = response.replace('**', '\\n**').replace('\\n\\n**', '\\n**')
            
            # Ensure lists have proper spacing
            response = response.replace('• ', '\\n• ').replace('\\n\\n• ', '\\n• ')
            
            # Clean up excessive newlines
            while '\\n\\n\\n' in response:
                response = response.replace('\\n\\n\\n', '\\n\\n')
            
            return response.strip()
            
        except:
            return response
    
    def _generate_smart_fallback(self, user_input: str) -> str:
        """Generate intelligent fallback based on user input"""
        return f"""I understand you're asking about: "{user_input}"

Let me help you with your medical question in the best way I can.

**I can assist with:**
• Analyzing symptoms and health concerns
• Providing medical information and education
• Explaining conditions and diseases
• Guidance on when to seek medical care

**For the best help, please:**
• Describe your specific symptoms
• Mention how long you've been experiencing them
• Share any relevant medical history
• Ask specific questions about your health

**If this is urgent:**
• Severe symptoms → Seek immediate medical care
• Non-urgent concerns → Schedule with your doctor
• General questions → I'm here to help explain

What specific medical information or symptom analysis can I provide for you?"""
    
    def _update_conversation_context(self, user_id: str, user_input: str, response: str):
        """Update conversation context for better future responses"""
        try:
            if user_id not in self.conversation_contexts:
                self.conversation_contexts[user_id] = {
                    'messages': [],
                    'detected_symptoms': [],
                    'medical_topics': [],
                    'last_updated': None
                }
            
            context = self.conversation_contexts[user_id]
            
            # Add to message history (keep last 10)
            context['messages'].append({
                'timestamp': datetime.now().isoformat(),
                'input': user_input,
                'response': response[:200] + '...' if len(response) > 200 else response
            })
            context['messages'] = context['messages'][-10:]  # Keep last 10
            
            # Extract and store symptoms mentioned
            symptoms = self._extract_symptoms(user_input)
            for symptom in symptoms:
                if symptom not in context['detected_symptoms']:
                    context['detected_symptoms'].append(symptom)
            
            # Keep only recent symptoms (last 20)
            context['detected_symptoms'] = context['detected_symptoms'][-20:]
            
            context['last_updated'] = datetime.now().isoformat()
            self.enhancement_stats['context_improvements'] += 1
            
        except Exception as e:
            logger.error(f"Context update failed: {e}")
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract medical symptoms from text"""
        symptoms = []
        symptom_keywords = [
            'headache', 'migraine', 'fever', 'nausea', 'vomiting', 'fatigue',
            'dizziness', 'cough', 'chest pain', 'back pain', 'stomach pain',
            'joint pain', 'muscle pain', 'shortness of breath', 'rash',
            'itching', 'swelling', 'bleeding', 'discharge'
        ]
        
        text_lower = text.lower()
        for symptom in symptom_keywords:
            if symptom in text_lower:
                symptoms.append(symptom)
        
        return symptoms
    
    def _is_quality_response(self, response: str) -> bool:
        """Check if response is high quality for caching"""
        quality_indicators = [
            len(response) > 100,
            '**' in response,  # Has formatting
            any(word in response.lower() for word in ['symptom', 'condition', 'medical']),
            not self._is_fallback_response(response),
            'disclaimer' in response.lower() or 'consult' in response.lower()
        ]
        
        return sum(quality_indicators) >= 3
    
    def _generate_cache_key(self, user_input: str) -> str:
        """Generate cache key for response caching"""
        # Normalize input for better cache hits
        normalized = user_input.lower().strip()
        normalized = ''.join(char for char in normalized if char.isalnum() or char.isspace())
        return normalized[:100]  # Limit key length
    
    def check_cached_response(self, user_input: str) -> Optional[str]:
        """Check if we have a cached response for similar input"""
        try:
            cache_key = self._generate_cache_key(user_input)
            cached = self.response_cache.get(cache_key)
            
            if cached:
                # Check if cache is still fresh (24 hours)
                cache_time = datetime.fromisoformat(cached['timestamp'])
                if (datetime.now() - cache_time.replace(tzinfo=None)).hours < 24:
                    self.enhancement_stats['cache_hits'] += 1
                    logger.info(f"✅ Cache hit for: {user_input[:30]}...")
                    return cached['response']
                else:
                    # Remove expired cache
                    del self.response_cache[cache_key]
            
            return None
            
        except Exception as e:
            logger.error(f"Cache check failed: {e}")
            return None
    
    def _get_safe_fallback_response(self, args, kwargs) -> str:
        """Ultra-safe fallback when everything fails"""
        return """Hello! I'm your medical AI assistant. I'm here to help with your health questions and medical concerns.

**I can help you with:**
• Analyzing symptoms and health issues
• Providing medical information and education
• Explaining medical conditions
• Guidance on healthcare decisions

**Please describe:**
• Your symptoms or health concern
• How long you've been experiencing it
• Any specific questions you have

**For emergencies:** Please contact emergency services or visit your nearest emergency room.

How can I assist you with your health today?"""
    
    def get_enhancement_stats(self) -> Dict:
        """Get enhancement statistics"""
        return {
            **self.enhancement_stats,
            'active_contexts': len(self.conversation_contexts),
            'cached_responses': len(self.response_cache),
            'ai_system_available': self.existing_ai is not None,
            'timestamp': datetime.now().isoformat()
        }
    
    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        logger.info("🧹 Response cache cleared")
    
    def clear_old_contexts(self, hours: int = 24):
        """Clear old conversation contexts"""
        try:
            current_time = datetime.now()
            expired_contexts = []
            
            for user_id, context in self.conversation_contexts.items():
                if context.get('last_updated'):
                    last_update = datetime.fromisoformat(context['last_updated'])
                    if (current_time - last_update).total_seconds() > (hours * 3600):
                        expired_contexts.append(user_id)
            
            for user_id in expired_contexts:
                del self.conversation_contexts[user_id]
            
            logger.info(f"🧹 Cleared {len(expired_contexts)} old conversation contexts")
            
        except Exception as e:
            logger.error(f"Context cleanup failed: {e}")

# Global enhancer instance
_conversation_enhancer = None

def get_conversation_enhancer() -> ConversationEnhancer:
    """Get global conversation enhancer"""
    global _conversation_enhancer
    if _conversation_enhancer is None:
        _conversation_enhancer = ConversationEnhancer()
    return _conversation_enhancer

# Easy integration decorators
def enhance_medical_ai(func):
    """DECORATOR: Enhance any existing medical AI function"""
    enhancer = get_conversation_enhancer()
    return enhancer.enhance_medical_response(func)

def smart_cache(func):
    """DECORATOR: Add smart caching to any function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        enhancer = get_conversation_enhancer()
        
        # Try cache first
        user_input = args[0] if args else kwargs.get('user_input', '')
        cached_response = enhancer.check_cached_response(str(user_input))
        if cached_response:
            return cached_response
        
        # Call original function
        result = func(*args, **kwargs)
        return result
    
    return wrapper

if __name__ == "__main__":
    # Test the enhancer
    enhancer = ConversationEnhancer()
    
    # Test response enhancement
    original_response = "I don't understand your question."
    enhanced = enhancer._improve_fallback_response(original_response, "I have a headache")
    print("Enhanced Response:")
    print(enhanced)
    
    # Test stats
    stats = enhancer.get_enhancement_stats()
    print(f"\\nStats: {json.dumps(stats, indent=2)}")