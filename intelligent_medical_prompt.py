"""
Advanced Intelligent Medical Chatbot System
Implements doctor-like conversational flow with intelligent query detection and diagnosis
"""

import re
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class IntelligentMedicalChatbot:
    def __init__(self):
        self.conversation_state = {}
        self.medical_knowledge_base = self._load_medical_knowledge()
        self.symptom_analyzer = SymptomAnalyzer()
        self.diagnosis_engine = DiagnosisEngine()
        
    def _load_medical_knowledge(self):
        """Load comprehensive medical knowledge base"""
        return {
            'diseases': {
                'migraine': {
                    'symptoms': ['headache', 'nausea', 'sensitivity to light', 'visual aura', 'throbbing pain'],
                    'triggers': ['stress', 'hormonal changes', 'certain foods', 'sleep changes'],
                    'treatment': ['rest in dark room', 'pain medication', 'preventive medication'],
                    'prevention': ['identify triggers', 'regular sleep', 'stress management']
                },
                'diabetes': {
                    'symptoms': ['excessive thirst', 'frequent urination', 'fatigue', 'blurred vision', 'slow healing'],
                    'types': ['type 1', 'type 2', 'gestational'],
                    'treatment': ['insulin therapy', 'medication', 'diet control', 'exercise'],
                    'complications': ['heart disease', 'kidney damage', 'nerve damage']
                },
                'hypertension': {
                    'symptoms': ['headaches', 'dizziness', 'chest pain', 'shortness of breath'],
                    'causes': ['genetics', 'diet', 'stress', 'obesity', 'age'],
                    'treatment': ['medication', 'lifestyle changes', 'diet modification'],
                    'prevention': ['exercise', 'low sodium diet', 'stress management']
                }
            },
            'symptoms': {
                'headache': ['migraine', 'tension headache', 'cluster headache', 'hypertension'],
                'fatigue': ['diabetes', 'anemia', 'thyroid disorders', 'depression'],
                'chest_pain': ['heart disease', 'anxiety', 'muscle strain', 'acid reflux']
            }
        }

class ConversationFlow:
    @staticmethod
    def get_greeting_response(user_input: str) -> str:
        """Intelligent greeting response"""
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        
        if any(greeting in user_input.lower() for greeting in greetings):
            return """Hello! I'm Med-AI, your intelligent medical assistant. 

🏥 **How I can help you today:**
- Analyze your symptoms and provide medical insights
- Help identify possible conditions based on your symptoms  
- Provide treatment recommendations and preventive measures
- Answer medical questions with evidence-based information

**Please describe your medical concern or symptoms, and I'll guide you through a comprehensive analysis.**

What brings you here today? Are you experiencing any specific symptoms or health concerns?"""
        
        return None

class SymptomAnalyzer:
    def __init__(self):
        self.symptom_patterns = {
            'pain': {
                'keywords': ['pain', 'ache', 'hurt', 'sore', 'discomfort'],
                'follow_up': 'Can you describe the pain? Is it sharp, dull, throbbing, or burning? Where exactly is it located and when did it start?'
            },
            'headache': {
                'keywords': ['headache', 'migraine', 'head pain', 'skull pain'],
                'follow_up': 'Tell me more about your headache: Is it throbbing or constant? Do you have any nausea, sensitivity to light, or visual changes? How long have you been experiencing this?'
            },
            'fatigue': {
                'keywords': ['tired', 'exhausted', 'fatigue', 'weakness', 'energy'],
                'follow_up': 'How long have you been feeling fatigued? Is it constant or does it come and go? Have you noticed any other symptoms like changes in appetite, sleep, or mood?'
            },
            'fever': {
                'keywords': ['fever', 'temperature', 'hot', 'chills', 'burning'],
                'follow_up': 'What is your current temperature? When did the fever start? Are you experiencing any other symptoms like cough, sore throat, or body aches?'
            }
        }
    
    def analyze_symptoms(self, user_input: str) -> Dict:
        """Analyze user input for symptoms"""
        identified_symptoms = []
        follow_up_questions = []
        
        for symptom_type, info in self.symptom_patterns.items():
            if any(keyword in user_input.lower() for keyword in info['keywords']):
                identified_symptoms.append(symptom_type)
                follow_up_questions.append(info['follow_up'])
        
        return {
            'symptoms': identified_symptoms,
            'follow_up_questions': follow_up_questions,
            'needs_clarification': len(identified_symptoms) > 0
        }

class DiagnosisEngine:
    def __init__(self):
        self.diagnostic_criteria = {
            'migraine': {
                'primary_symptoms': ['headache', 'throbbing pain', 'nausea'],
                'secondary_symptoms': ['light sensitivity', 'visual aura', 'vomiting'],
                'questions': [
                    'Is the headache on one side of your head?',
                    'Do you feel nauseous or have you vomited?',
                    'Are you sensitive to light or sound?',
                    'Does the pain feel like throbbing or pulsing?'
                ]
            },
            'diabetes': {
                'primary_symptoms': ['excessive thirst', 'frequent urination', 'fatigue'],
                'secondary_symptoms': ['blurred vision', 'slow healing', 'weight loss'],
                'questions': [
                    'Have you been drinking more water than usual?',
                    'Are you urinating more frequently?',
                    'Have you experienced unexplained weight loss?',
                    'Do you have a family history of diabetes?'
                ]
            }
        }
    
    def generate_diagnostic_questions(self, suspected_conditions: List[str]) -> List[str]:
        """Generate diagnostic questions based on suspected conditions"""
        questions = []
        for condition in suspected_conditions:
            if condition in self.diagnostic_criteria:
                questions.extend(self.diagnostic_criteria[condition]['questions'])
        return questions[:3]  # Limit to 3 questions

class IntelligentMedicalResponseSystem:
    def __init__(self):
        self.chatbot = IntelligentMedicalChatbot()
        self.conversation_flow = ConversationFlow()
        self.user_sessions = {}
    
    def process_intelligent_query(self, user_input: str, user_id: str = 'default', medical_content: str = '') -> str:
        """Main intelligent processing function"""
        
        # Initialize user session if not exists
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'stage': 'initial',
                'symptoms': [],
                'suspected_conditions': [],
                'questions_asked': [],
                'context': {}
            }
        
        session = self.user_sessions[user_id]
        
        # Stage 1: Greeting and Introduction
        greeting_response = self.conversation_flow.get_greeting_response(user_input)
        if greeting_response:
            session['stage'] = 'greeting_complete'
            return greeting_response
        
        # Stage 2: Medical Query Detection and Symptom Analysis
        if self._is_medical_query(user_input):
            return self._handle_medical_query(user_input, session, medical_content)
        
        # Non-medical query
        return self._handle_non_medical_query(user_input)
    
    def _is_medical_query(self, query: str) -> bool:
        """Advanced medical query detection"""
        medical_indicators = [
            # Symptoms
            'pain', 'ache', 'hurt', 'headache', 'migraine', 'fever', 'nausea', 'vomiting',
            'fatigue', 'tired', 'dizzy', 'cough', 'cold', 'flu', 'infection',
            
            # Medical terms
            'symptoms', 'disease', 'condition', 'treatment', 'medicine', 'medication',
            'doctor', 'hospital', 'diagnosis', 'cure', 'therapy',
            
            # Phrases
            'i have', 'experiencing', 'suffering from', 'what causes', 'how to treat',
            'feeling sick', 'not well', 'health problem', 'medical advice'
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in medical_indicators)
    
    def _handle_medical_query(self, user_input: str, session: Dict, medical_content: str) -> str:
        """Handle medical query with intelligent flow"""
        
        # Analyze symptoms
        symptom_analysis = self.chatbot.symptom_analyzer.analyze_symptoms(user_input)
        
        if symptom_analysis['symptoms']:
            session['symptoms'].extend(symptom_analysis['symptoms'])
            session['stage'] = 'symptom_analysis'
            
            # Generate intelligent response based on symptoms
            return self._generate_symptom_response(symptom_analysis, session, medical_content)
        
        # General medical query
        return self._generate_general_medical_response(user_input, medical_content)
    
    def _generate_symptom_response(self, symptom_analysis: Dict, session: Dict, medical_content: str) -> str:
        """Generate intelligent symptom-based response"""
        
        primary_symptom = symptom_analysis['symptoms'][0] if symptom_analysis['symptoms'] else 'general'
        
        # Start with empathy and acknowledgment
        response_parts = [
            f"I understand you're experiencing {', '.join(symptom_analysis['symptoms'])}. Let me help you with a comprehensive analysis."
        ]
        
        # Add medical knowledge if available
        if medical_content:
            response_parts.append(f"\n**Medical Information:**\n{medical_content[:300]}...")
        
        # Generate follow-up questions for better diagnosis
        if symptom_analysis['follow_up_questions']:
            response_parts.append(f"\n**To better understand your condition, please tell me:**")
            for i, question in enumerate(symptom_analysis['follow_up_questions'][:2], 1):
                response_parts.append(f"{i}. {question}")
        
        # Add preliminary guidance
        response_parts.append(self._get_preliminary_guidance(symptom_analysis['symptoms']))
        
        return '\n'.join(response_parts)
    
    def _get_preliminary_guidance(self, symptoms: List[str]) -> str:
        """Get preliminary medical guidance based on symptoms"""
        
        guidance_map = {
            'headache': """
**Immediate Care Recommendations:**
• Rest in a quiet, dark room
• Apply cold or warm compress to head/neck
• Stay hydrated
• Consider over-the-counter pain relief if appropriate

**When to seek immediate care:** Sudden severe headache, fever with headache, vision changes, or confusion.""",
            
            'pain': """
**General Pain Management:**
• Rest the affected area
• Apply ice for acute injuries, heat for muscle tension
• Over-the-counter pain medication if appropriate
• Gentle movement when possible

**Seek medical attention if:** Severe pain, signs of infection, or pain persists.""",
            
            'fatigue': """
**Energy Management:**
• Ensure adequate sleep (7-9 hours)
• Maintain regular sleep schedule
• Stay hydrated and eat nutritious meals
• Light exercise if possible

**Monitor for:** Persistent fatigue lasting weeks, accompanied by fever, weight loss, or other concerning symptoms."""
        }
        
        for symptom in symptoms:
            if symptom in guidance_map:
                return guidance_map[symptom]
        
        return """
**General Health Recommendations:**
• Monitor your symptoms closely
• Maintain good hydration and nutrition
• Rest as needed
• Contact healthcare provider if symptoms worsen or persist

**Important:** This is preliminary guidance. Always consult with healthcare professionals for proper diagnosis and treatment."""
    
    def _generate_general_medical_response(self, user_input: str, medical_content: str) -> str:
        """Generate response for general medical queries"""
        
        if medical_content:
            return f"""Based on your query about "{user_input}", here's what the medical literature says:

**Medical Information:**
{medical_content}

**Additional Guidance:**
This information provides a foundation for understanding your condition. For personalized medical advice, diagnosis, and treatment plans, please consult with qualified healthcare professionals.

**Would you like me to:**
1. Explain any specific aspect in more detail?
2. Discuss potential symptoms to watch for?
3. Provide preventive measures and lifestyle recommendations?

⚠️ **Important:** This information is for educational purposes. Always consult healthcare professionals for medical decisions."""
        
        else:
            return f"""I understand you're asking about "{user_input}". While I don't have specific textbook information available right now, I can provide some general guidance.

**For comprehensive information about your medical concern, I recommend:**
• Consulting with your primary care physician
• Speaking with a specialist if needed
• Referring to current medical literature and guidelines

**If you're experiencing symptoms, please describe them in detail so I can provide more targeted assistance.**

Would you like to tell me more about any specific symptoms or concerns you're having?"""
    
    def _handle_non_medical_query(self, user_input: str) -> str:
        """Handle non-medical queries"""
        return """I'm Med-AI, a specialized medical assistant designed to help with health-related questions and concerns.

**I can help you with:**
• Symptom analysis and medical guidance
• Information about diseases and conditions  
• Treatment options and preventive measures
• Health and wellness advice

**For the best assistance, please share:**
• Any symptoms you're experiencing
• Specific medical questions or concerns
• Health conditions you'd like to learn about

What medical topic would you like to discuss today?"""

def create_intelligent_medical_prompt(user_query: str, medical_content: str = '', user_id: str = 'default') -> str:
    """Create intelligent medical response using the advanced system"""
    
    system = IntelligentMedicalResponseSystem()
    return system.process_intelligent_query(user_query, user_id, medical_content)

def get_fast_medical_response(query: str, medical_content: str = '') -> str:
    """Optimized for fast responses while maintaining intelligence"""
    
    # Quick medical query classification
    if any(word in query.lower() for word in ['hello', 'hi', 'hey']):
        return """Hello! I'm Med-AI, your intelligent medical assistant. 

I'm here to help analyze your symptoms, provide medical insights, and guide you through health concerns.

**What would you like to discuss today?**
• Describe any symptoms you're experiencing
• Ask about a specific medical condition
• Get guidance on treatment options

Please share your medical concern, and I'll provide comprehensive assistance."""
    
    # Medical symptom detection
    symptoms = ['pain', 'headache', 'migraine', 'fever', 'nausea', 'fatigue', 'cough']
    detected_symptom = None
    for symptom in symptoms:
        if symptom in query.lower():
            detected_symptom = symptom
            break
    
    if detected_symptom:
        responses = {
            'migraine': """I understand you're experiencing migraine symptoms. Let me help you with comprehensive guidance.

**Based on medical knowledge:**
{content}

**To better assist you, please tell me:**
1. Is the headache throbbing and on one side of your head?
2. Are you experiencing nausea or sensitivity to light?
3. How long have you had this headache?

**Immediate relief measures:**
• Rest in a quiet, dark room
• Apply cold compress to forehead
• Stay hydrated
• Avoid known triggers

**Seek medical attention if:** This is your first severe headache, fever accompanies the headache, or you have vision changes.

Would you like me to explain more about migraine triggers or treatment options?""",
            
            'headache': """I understand you have a headache. Let me provide comprehensive guidance to help you.

**Medical Information:**
{content}

**To better understand your condition:**
1. How severe is the pain (1-10 scale)?
2. Where exactly is the pain located?
3. Is it throbbing, sharp, or dull?
4. Any accompanying symptoms like nausea or light sensitivity?

**Immediate care:**
• Rest in a quiet environment
• Apply cold or warm compress
• Stay well hydrated
• Consider appropriate pain relief

**Red flags - Seek immediate care if:**
• Sudden, severe headache unlike any before
• Headache with fever, neck stiffness, or rash
• Headache following head injury

What other symptoms are you experiencing with your headache?"""
        }
        
        template = responses.get(detected_symptom, responses['headache'])
        return template.format(content=medical_content[:200] if medical_content else "Relevant medical information from textbooks.")
    
    # General medical response
    if medical_content:
        return f"""Based on your query about "{query}", here's comprehensive medical guidance:

**Medical Knowledge:**
{medical_content}

**Clinical Approach:**
I'll help you understand this condition thoroughly. 

**Next steps for you:**
1. Are you currently experiencing any symptoms related to this condition?
2. Do you have any family history of this condition?
3. Are you looking for preventive measures or treatment information?

**Important:** This information is evidence-based medical knowledge. For personalized care, diagnosis, and treatment, always consult with qualified healthcare professionals.

What specific aspect would you like me to explain in more detail?"""
    
    return f"""I understand you're asking about "{query}". Let me provide you with helpful medical guidance.

While I don't have specific textbook information available for this query, I can help guide you through the medical approach to your concern.

**Please tell me more:**
• Are you experiencing specific symptoms?
• Is this for preventive information or current health concerns?
• Do you have any related medical history?

**For comprehensive information:**
• Consult with your healthcare provider
• Consider seeing a specialist if needed
• I can help analyze any symptoms you describe

Would you like to describe any symptoms or specific concerns you're having?"""

# Export main functions
__all__ = [
    'IntelligentMedicalResponseSystem',
    'create_intelligent_medical_prompt', 
    'get_fast_medical_response'
]