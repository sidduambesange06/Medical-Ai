"""
Intelligent Medical Response System.
Extracted from core application logic.

Handles symptom analysis, disease pattern recognition, and medical information
retrieval from indexed textbooks. Avoids medication recommendations for legal safety.
"""
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class IntelligentMedicalResponseSystem:
    """
    Intelligent Medical Diagnosis and Support System.

    Focused on medical diagnosis support and symptom analysis.
    Does not provide medication recommendations (legal safety constraint).
    """
    
    def __init__(self):
        self.user_sessions = {}
        logger.info("[AI-INIT] Intelligent Medical Diagnosis System initialized")
    
    def process_intelligent_query_with_patient_context(self, user_input: str, user_id: str = 'default', 
                                                     medical_content: str = '', patient_context: Dict = None) -> str:
        """
        🧠 PATIENT-AWARE DIAGNOSTIC PROCESSING
        Enhanced diagnostic processing with patient medical context integration
        """
        
        # Initialize user session with patient context
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'stage': 'initial',
                'symptoms': [],
                'context': {},
                'patient_context': patient_context or {},
                'conversation_history': []
            }
        else:
            # Update existing session with patient context
            self.user_sessions[user_id]['patient_context'] = patient_context or {}
        
        session = self.user_sessions[user_id]
        query_lower = user_input.lower().strip()
        
        # Enhanced greeting with patient personalization
        greeting_words = ['hello', 'hi', 'hey', 'hiii', 'good morning', 'good afternoon', 'good evening']
        is_greeting = any(word in query_lower for word in greeting_words)
        
        has_medical_history = len(session.get('conversation_history', [])) > 1
        already_greeted = session.get('first_greeting_shown', False)
        
        should_show_greeting = (
            is_greeting and (
                not already_greeted or
                (len(query_lower.split()) <= 3 and query_lower.strip() in [word.lower() for word in greeting_words])
            )
        )
        
        if should_show_greeting:
            session['stage'] = 'greeted'
            session['first_greeting_shown'] = True
            
            # Personalized greeting with patient context
            greeting = "Hello! I'm Med-AI, your intelligent medical diagnostic assistant."
            
            if patient_context:
                patient_summary = patient_context.get('patient_summary', '')
                if patient_summary:
                    greeting += f"\\n\\nBased on your health profile ({patient_summary}), I'm ready to provide personalized diagnostic insights."
                
                risk_factors = patient_context.get('risk_factors', [])
                if risk_factors:
                    greeting += f"\\n\\n⚠️ **Important**: I note you have the following risk factors: {', '.join(risk_factors)}. I'll consider these in my diagnostic analysis."
            
            greeting += f"""

🏥 **How I can assist you today:**
- Analyze your symptoms and identify potential conditions
- Provide diagnostic insights based on medical literature
- Help recognize symptom patterns and disease indicators
- Support your healthcare decisions with evidence-based information
- Retrieve relevant medical textbook information

**Please describe your symptoms or medical concern, and I'll provide comprehensive diagnostic insights.**

⚠️ **Important**: I provide diagnostic support and medical information only. For medication advice, please consult your healthcare provider.

What symptoms or health concerns would you like me to analyze?"""
            
            return greeting
        
        # Medical query processing with patient context
        if self._is_medical_query(user_input):
            return self._handle_medical_query_with_patient_context(user_input, session, medical_content, patient_context)
        
        # Non-medical redirect with patient awareness
        response = "I'm Med-AI, your personalized medical diagnostic assistant."
        
        if patient_context:
            response += f" I have your health profile on file and can provide tailored diagnostic insights."
        
        response += """

**I can help you with:**
• Comprehensive symptom analysis based on your medical history
• Disease identification and pattern recognition
• Medical information retrieval from textbooks
• Diagnostic insights tailored to your risk factors
• Evidence-based health information

Please share your symptoms or health concern so I can provide diagnostic insights."""
        
        return response
        
    def process_intelligent_query(self, user_input: str, user_id: str = 'default', medical_content: str = '') -> str:
        """
        🧠 MAIN DIAGNOSTIC PROCESSING
        Main intelligent processing focused on diagnosis and symptom analysis
        """
        
        # Initialize user session
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'stage': 'initial',
                'symptoms': [],
                'context': {},
                'conversation_history': []
            }
        
        session = self.user_sessions[user_id]
        query_lower = user_input.lower().strip()
        
        # SMART GREETING DETECTION
        greeting_words = ['hello', 'hi', 'hey', 'hiii', 'good morning', 'good afternoon', 'good evening']
        is_greeting = any(word in query_lower for word in greeting_words)
        
        has_medical_history = len(session.get('conversation_history', [])) > 1
        already_greeted = session.get('first_greeting_shown', False)
        
        should_show_greeting = (
            is_greeting and (
                not already_greeted or
                (len(query_lower.split()) <= 3 and query_lower.strip() in [word.lower() for word in greeting_words])
            )
        )
        
        if should_show_greeting:
            session['stage'] = 'greeted'
            session['first_greeting_shown'] = True
            return """Hello! I'm Med-AI, your intelligent medical diagnostic assistant. 

🏥 **How I can help you today:**
- Analyze symptoms and identify potential conditions
- Provide diagnostic insights from medical literature
- Help recognize disease patterns and indicators
- Support healthcare decisions with evidence-based information

**Please describe your symptoms or medical concern, and I'll provide comprehensive diagnostic analysis.**

⚠️ **Important**: I provide diagnostic support and medical information only. For treatment and medication advice, please consult your healthcare provider.

What symptoms or health concerns would you like me to analyze?"""
        
        # MEDICAL QUERY DETECTION
        if self._is_medical_query(user_input):
            return self._handle_medical_query(user_input, session, medical_content)
        
        # NON-MEDICAL QUERY REDIRECT
        return """I'm Med-AI, a specialized medical diagnostic assistant focused on symptom analysis and disease identification.

**I can help you with:**
• Symptom analysis and pattern recognition
• Disease identification based on symptoms
• Medical information retrieval from textbooks
• Evidence-based diagnostic insights

**I do NOT provide:**
• Medication recommendations or dosages
• Treatment prescriptions
• Medical advice requiring professional consultation

**Please share your symptoms:**
• Any symptoms you're experiencing
• Medical questions about conditions or diseases
• Health concerns you'd like analyzed

What symptoms or medical topic would you like me to analyze today?"""
    
    def _is_medical_query(self, query: str) -> bool:
        """🔍 ENHANCED MEDICAL QUERY DETECTION - Diagnosis Focused"""
        medical_indicators = [
            # Symptoms - FOCUS AREA
            'pain', 'ache', 'hurt', 'headache', 'migraine', 'fever', 'nausea', 'vomiting',
            'fatigue', 'tired', 'dizzy', 'cough', 'cold', 'flu', 'infection', 'suffering',
            'chest pain', 'stomach pain', 'back pain', 'joint pain', 'muscle pain',
            'shortness of breath', 'difficulty breathing', 'rapid heartbeat', 'palpitations',
            'rash', 'itching', 'swelling', 'bruising', 'bleeding', 'discharge',
            
            # Medical terms - DIAGNOSIS FOCUSED
            'symptoms', 'disease', 'condition', 'diagnosis', 'disorder',
            'doctor', 'hospital', 'medical', 'health', 'illness',
            
            # Question patterns - INFORMATION RETRIEVAL
            'what causes', 'symptoms of', 'what is', 'tell me about', 
            'i have', 'experiencing', 'suffering from', 'feeling sick', 'not well',
            'information about', 'how does', 'why does'
            
            # REMOVED: medication, treatment, dosage related terms for legal safety
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in medical_indicators)
    
    def _classify_medical_query_type(self, query: str) -> str:
        """🏷️ ADVANCED: Classify medical query type - DIAGNOSIS FOCUSED"""
        query_lower = query.lower().strip()
        
        # 1. DIAGNOSTIC QUERIES - MAIN FOCUS
        diagnostic_patterns = [
            'i have', 'i am experiencing', 'i feel', 'suffering from', 'experiencing',
            'feeling sick', 'not well', 'something wrong', 'what is wrong',
            'help me', 'what could be', 'could this be', 'is this normal',
            'symptoms are', 'pain in', 'ache in'
        ]
        
        # 2. MEDICAL INFORMATION QUERIES - KNOWLEDGE RETRIEVAL
        info_patterns = [
            'what is', 'what are', 'tell me about', 'information about',
            'explain', 'describe', 'definition of', 'meaning of',
            'how does', 'why does', 'what causes', 'symptoms of',
            'types of', 'stages of', 'prognosis of', 'risk factors'
        ]
        
        # 3. CONDITION/DISEASE QUERIES - PATTERN RECOGNITION
        condition_patterns = [
            'diabetes', 'hypertension', 'asthma', 'arthritis', 'cancer',
            'infection', 'disease', 'condition', 'disorder', 'syndrome'
        ]
        
        # REMOVED: Medicine/treatment patterns for legal safety
        
        # Classify based on patterns
        if any(pattern in query_lower for pattern in diagnostic_patterns):
            return 'diagnostic'
        elif any(pattern in query_lower for pattern in condition_patterns):
            return 'condition_info'
        elif any(pattern in query_lower for pattern in info_patterns):
            return 'medical_info'
        else:
            return 'general_medical'
    
    def _detect_symptoms(self, user_input: str) -> list:
        """🔍 ENHANCED: Comprehensive symptom detection with pattern matching"""
        symptom_patterns = {
            'headache': ['headache', 'migraine', 'head pain', 'migrane', 'head ache', 'skull pain'],
            'fever': ['fever', 'temperature', 'hot', 'chills', 'burning up', 'feverish', 'high temp'],
            'pain': ['pain', 'ache', 'hurt', 'sore', 'discomfort', 'aching', 'painful', 'throbbing'],
            'chest_pain': ['chest pain', 'chest ache', 'heart pain', 'chest discomfort', 'chest pressure'],
            'stomach_pain': ['stomach pain', 'stomach ache', 'belly pain', 'abdominal pain', 'tummy pain'],
            'back_pain': ['back pain', 'back ache', 'spine pain', 'lower back', 'upper back'],
            'fatigue': ['tired', 'exhausted', 'fatigue', 'weakness', 'weary', 'drained', 'worn out'],
            'nausea': ['nausea', 'nauseous', 'sick', 'vomiting', 'queasy', 'throw up', 'morning sickness'],
            'breathing': ['shortness of breath', 'difficulty breathing', 'breathless', 'can\'t breathe', 'wheezing'],
            'dizziness': ['dizzy', 'dizziness', 'lightheaded', 'vertigo', 'spinning', 'unsteady'],
            'cough': ['cough', 'coughing', 'persistent cough', 'dry cough', 'wet cough'],
            'skin': ['rash', 'itching', 'redness', 'swelling', 'bumps', 'hives', 'irritation'],
            'digestive': ['diarrhea', 'constipation', 'bloating', 'gas', 'indigestion', 'heartburn']
        }
        
        detected = []
        query_lower = user_input.lower()
        
        for symptom, keywords in symptom_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                detected.append(symptom)
        
        return detected
    
    def _generate_diagnostic_questions(self, symptoms: list, user_input: str) -> str:
        """👩‍⚕️ ADVANCED: Generate doctor-like diagnostic questions based on symptoms"""
        
        if not symptoms:
            return """1. Can you describe your main symptoms in detail?
2. When did these symptoms first start?
3. Have they gotten better, worse, or stayed the same?
4. Is there anything that makes the symptoms better or worse?
5. Do you have any known allergies or medical conditions?
6. Have you experienced these symptoms before?"""
        
        primary_symptom = symptoms[0]
        questions = []
        
        # General questions for all symptoms
        questions.extend([
            f"1. How long have you been experiencing {primary_symptom}?",
            "2. On a scale of 1-10, how severe would you rate your symptoms?",
            "3. Have you noticed any triggers that make symptoms worse?"
        ])
        
        # Symptom-specific diagnostic questions (MEDICAL ONLY - NO TREATMENT)
        if 'fever' in symptoms:
            questions.extend([
                "4. Have you measured your temperature? What was the reading?",
                "5. Are you experiencing chills, sweating, or body aches?",
                "6. Any recent travel or exposure to illness?"
            ])
        elif 'headache' in symptoms:
            questions.extend([
                "4. Is the headache on one side or both sides of your head?",
                "5. Does light or sound make the headache worse?",
                "6. Have you experienced nausea or vomiting with the headache?"
            ])
        elif 'chest_pain' in symptoms:
            questions.extend([
                "4. Is the chest pain sharp, dull, burning, or pressure-like?",
                "5. Does the pain spread to your arms, neck, jaw, or back?",
                "6. Does the pain worsen with activity or deep breathing?",
                "⚠️ **URGENT**: If experiencing severe chest pain, seek immediate medical attention!"
            ])
        elif 'stomach_pain' in symptoms:
            questions.extend([
                "4. Where exactly is the stomach pain located?",
                "5. Is the pain crampy, sharp, or burning?",
                "6. Have you had any changes in bowel movements or appetite?"
            ])
        elif 'breathing' in symptoms:
            questions.extend([
                "4. When do you notice the breathing difficulty most?",
                "5. Is it worse when lying down or with activity?",
                "6. Any chest pain or wheezing along with breathlessness?",
                "⚠️ **URGENT**: If experiencing severe breathing difficulty, seek immediate medical attention!"
            ])
        else:
            questions.extend([
                "4. Are there any other symptoms accompanying this?",
                "5. Have you experienced these symptoms before?",
                "6. Is this the first time you've had these concerns?"
            ])
        
        return "\\n".join(questions)
    
    def _handle_medical_query_with_patient_context(self, user_input: str, session: dict, 
                                                  medical_content: str, patient_context: Dict = None) -> str:
        """🧠 PATIENT-AWARE: Handle medical query with patient context integration - DIAGNOSIS FOCUSED"""
        
        logger.info(f"[PATIENT-AWARE] Processing diagnostic query with patient context: {bool(patient_context)}")
        
        # STEP 1: Classify query type for optimal response
        query_type = self._classify_medical_query_type(user_input)
        
        # STEP 2: Enhanced response with patient context
        if medical_content and len(medical_content.strip()) > 10:
            return self._format_response_with_patient_context(query_type, user_input, medical_content, session, patient_context)
        
        # STEP 3: Handle queries without textbook content but with patient awareness
        return self._handle_query_without_content_with_patient_context(query_type, user_input, session, patient_context)
    
    def _format_response_with_patient_context(self, query_type: str, user_input: str, 
                                            medical_content: str, session: dict, patient_context: Dict = None) -> str:
        """📋 Format diagnostic response with patient context awareness - NO MEDICATION ADVICE"""
        
        # Get base response from medical content
        base_response = self._format_response_by_type(query_type, user_input, medical_content, session)
        
        if not patient_context:
            return base_response
        
        # Add patient-specific diagnostic considerations
        patient_additions = []
        
        # Check for relevant risk factors
        risk_factors = patient_context.get('risk_factors', [])
        if risk_factors:
            relevant_risks = []
            for risk in risk_factors:
                if any(keyword in user_input.lower() for keyword in [
                    'heart', 'chest', 'blood pressure', 'hypertension'
                ]) and 'hypertension' in risk.lower():
                    relevant_risks.append(risk)
                elif any(keyword in user_input.lower() for keyword in [
                    'weight', 'diet', 'diabetes', 'sugar'
                ]) and 'obesity' in risk.lower():
                    relevant_risks.append(risk)
            
            if relevant_risks:
                patient_additions.append(f"⚠️ **Important for your profile**: Given your risk factors ({', '.join(relevant_risks)}), these symptoms may require special attention.")
        
        # Check for known conditions relevance
        medical_conditions = patient_context.get('medical_conditions', [])
        if medical_conditions:
            patient_additions.append(f"🏥 **Medical History Consideration**: I note you have {len(medical_conditions)} known condition(s) in your profile that may be relevant to these symptoms.")
        
        # Check for allergies (important for diagnostic context)
        allergies = patient_context.get('allergies', [])
        if allergies:
            patient_additions.append(f"🚨 **Allergy Alert**: Your profile shows known allergies. Always inform healthcare providers about your allergies during consultations.")
        
        # Add patient summary context
        patient_summary = patient_context.get('patient_summary', '')
        if patient_summary:
            patient_additions.append(f"👤 **Personalized Analysis**: Based on your profile ({patient_summary}), this diagnostic analysis is tailored to your specific health context.")
        
        # Combine base response with patient-specific additions
        if patient_additions:
            enhanced_response = base_response + "\\n\\n" + "\\n\\n".join(patient_additions)
        else:
            enhanced_response = base_response + f"\\n\\n👤 **Personalized Analysis**: This diagnostic response considers your health profile for the most relevant insights."
        
        return enhanced_response
    
    def _format_response_by_type(self, query_type: str, user_input: str, medical_content: str, session: dict) -> str:
        """📋 ADVANCED: Format diagnostic response based on query type - LEGALLY COMPLIANT"""
        
        if query_type == 'diagnostic':
            # Patient describing symptoms - needs diagnostic approach
            symptoms = self._detect_symptoms(user_input)
            diagnostic_questions = self._generate_diagnostic_questions(symptoms, user_input)
            
            return f"""🩺 **MEDICAL DIAGNOSTIC ANALYSIS**

**Based on your symptoms, here's what I found in medical literature:**
{medical_content}

**🔍 SYMPTOM ANALYSIS:**
Detected symptoms: {', '.join(symptoms) if symptoms else 'General health concern'}

**👩‍⚕️ DIAGNOSTIC QUESTIONS:**
To better understand your condition, please consider these questions:
{diagnostic_questions}

**⚠️ IMPORTANT DISCLAIMERS:**
• This is diagnostic support information only
• For treatment and medication advice, consult your healthcare provider
• If symptoms are severe or worsening, seek immediate medical attention
• This analysis is based on medical literature and your symptom description

**📋 NEXT STEPS:**
1. Consider the diagnostic questions above
2. Monitor your symptoms and their progression
3. Consult with a healthcare professional for proper diagnosis
4. Keep a symptom diary if symptoms persist"""
        
        elif query_type == 'condition_info':
            # Information about specific medical condition
            return f"""🏥 **MEDICAL CONDITION INFORMATION**

**Here's what medical literature says about your query:**
{medical_content}

**📚 KEY INFORMATION:**
This information is compiled from medical textbooks and literature to help you understand the condition better.

**⚠️ IMPORTANT:**
• This is educational information only
• Every individual case is unique
• For personalized medical advice, consult your healthcare provider
• Treatment decisions should always be made with professional medical guidance

**💡 FOR HEALTHCARE PROFESSIONALS:**
This information can serve as a reference point for your clinical decision-making process."""
        
        elif query_type == 'medical_info':
            # General medical information request
            return f"""📖 **MEDICAL INFORMATION RETRIEVAL**

**Here's the medical information you requested:**
{medical_content}

**🎯 INFORMATION SOURCE:**
This information is retrieved from medical textbooks and authoritative medical literature.

**📋 EDUCATIONAL PURPOSE:**
• This content is for educational and informational purposes
• Helps in understanding medical concepts and terminology
• Provides evidence-based medical knowledge
• Supports informed healthcare discussions

**⚠️ DISCLAIMER:**
This information does not constitute medical advice. Always consult healthcare professionals for medical decisions."""
        
        else:  # general_medical
            return f"""🔬 **GENERAL MEDICAL GUIDANCE**

**Medical Information:**
{medical_content}

**🎯 PURPOSE:**
This information is provided to enhance your understanding of medical topics and support informed healthcare decisions.

**📚 EVIDENCE-BASED:**
All information is sourced from medical literature and textbooks.

**⚠️ ALWAYS REMEMBER:**
• Consult healthcare professionals for medical advice
• This is educational information only
• Individual medical cases vary significantly
• Professional medical evaluation is essential for diagnosis and treatment"""
    
    def _handle_medical_query(self, user_input: str, session: dict, medical_content: str) -> str:
        """🧠 ENHANCED: Handle medical query with intelligent type-based formatting - DIAGNOSIS FOCUSED"""
        
        logger.info(f"[DEBUG] Processing diagnostic query with medical content length: {len(medical_content)}")
        
        # STEP 1: Classify query type for optimal response
        query_type = self._classify_medical_query_type(user_input)
        logger.info(f"[DEBUG] Query classified as: {query_type}")
        
        # STEP 2: Format response based on query type and available content
        if medical_content and len(medical_content.strip()) > 10:
            logger.info(f"[DEBUG] Using medical textbook content for {query_type} query")
            return self._format_response_by_type(query_type, user_input, medical_content, session)
        
        # STEP 3: Handle queries without textbook content
        return self._handle_query_without_content(query_type, user_input, session)
    
    def _handle_query_without_content(self, query_type: str, user_input: str, session: dict) -> str:
        """📚 Handle diagnostic queries without medical textbook content"""
        
        if query_type == 'diagnostic':
            symptoms = self._detect_symptoms(user_input)
            diagnostic_questions = self._generate_diagnostic_questions(symptoms, user_input)
            
            return f"""🩺 **SYMPTOM ANALYSIS** (Based on AI Medical Knowledge)

**🔍 DETECTED SYMPTOMS:**
{', '.join(symptoms) if symptoms else 'General health concern reported'}

**👩‍⚕️ DIAGNOSTIC ASSESSMENT:**
Based on the symptoms you've described, here's a preliminary analysis to help guide your next steps.

**DIAGNOSTIC QUESTIONS TO CONSIDER:**
{diagnostic_questions}

**⚠️ IMPORTANT - SEEK PROFESSIONAL CARE:**
• This is preliminary symptom analysis only
• A healthcare professional should evaluate your symptoms
• If symptoms are severe, seek immediate medical attention
• Consider scheduling an appointment with your doctor

**📋 RECOMMENDED ACTIONS:**
1. Answer the diagnostic questions above
2. Monitor symptom progression
3. Consult with a healthcare professional
4. Keep track of when symptoms occur and what triggers them"""
        
        else:
            return f"""🏥 **MEDICAL INFORMATION SUPPORT**

I understand you're looking for information about: "{user_input}"

**📚 WHAT I CAN HELP WITH:**
• Symptom analysis and pattern recognition
• Disease information from medical literature
• Diagnostic support and medical education
• Healthcare guidance and information retrieval

**⚠️ CURRENT LIMITATION:**
I don't have specific textbook content for your exact query at the moment, but I can help with:

**🎯 ALTERNATIVE SUPPORT:**
• Describe your symptoms in detail for analysis
• Ask about specific medical conditions or diseases
• Request information about diagnostic procedures
• Inquire about medical terminology or concepts

**💡 FOR BEST RESULTS:**
Please provide more specific symptoms or medical questions, and I'll do my best to provide comprehensive diagnostic support based on medical knowledge."""
    
    def _handle_query_without_content_with_patient_context(self, query_type: str, user_input: str, 
                                                         session: dict, patient_context: Dict = None) -> str:
        """📋 Handle queries without medical content but with patient awareness - DIAGNOSIS FOCUSED"""
        
        base_response = self._handle_query_without_content(query_type, user_input, session)
        
        if not patient_context:
            return base_response
        
        # Enhance with patient context for personalized diagnostic support
        patient_info = []
        
        risk_factors = patient_context.get('risk_factors', [])
        if risk_factors:
            patient_info.append(f"Risk factors to consider: {', '.join(risk_factors)}")
        
        medical_conditions = patient_context.get('medical_conditions', [])
        if medical_conditions:
            patient_info.append(f"Known medical conditions: {len(medical_conditions)} documented")
        
        allergies = patient_context.get('allergies', [])
        if allergies:
            patient_info.append(f"Known allergies: Important for healthcare consultations")
        
        if patient_info:
            enhanced_response = base_response + f"\\n\\n👤 **YOUR HEALTH PROFILE CONTEXT**: {' | '.join(patient_info)}"
            enhanced_response += "\\n\\nI'll keep these factors in mind when providing diagnostic insights. For the most accurate analysis, please provide more specific details about your current symptoms or concerns."
        else:
            enhanced_response = base_response
        
        return enhanced_response
    
    # ==================== ULTRA-ADVANCED MEDICAL CAPABILITIES ====================
    
    def enhanced_symptom_analysis(self, symptoms: str, patient_context: Dict = None) -> Dict:
        """🧠 ULTRA-ADVANCED: Enhanced symptom analysis with risk stratification"""
        
        analysis = {
            'detected_symptoms': [],
            'symptom_clusters': [],
            'severity_score': 0,
            'urgency_level': 'routine',
            'red_flags': [],
            'differential_diagnosis': [],
            'recommended_actions': [],
            'patient_specific_risks': []
        }
        
        # Advanced symptom pattern recognition
        symptom_patterns = {
            'cardiovascular': {
                'keywords': ['chest pain', 'palpitations', 'shortness of breath', 'dizziness', 'fainting'],
                'severity_multiplier': 3,
                'red_flags': ['crushing chest pain', 'severe shortness of breath', 'syncope']
            },
            'neurological': {
                'keywords': ['headache', 'confusion', 'weakness', 'numbness', 'vision changes'],
                'severity_multiplier': 2.5,
                'red_flags': ['sudden severe headache', 'speech changes', 'paralysis']
            },
            'respiratory': {
                'keywords': ['cough', 'wheeze', 'sputum', 'breathing difficulty'],
                'severity_multiplier': 2,
                'red_flags': ['severe breathing difficulty', 'cyanosis', 'hemoptysis']
            },
            'gastrointestinal': {
                'keywords': ['nausea', 'vomiting', 'diarrhea', 'abdominal pain'],
                'severity_multiplier': 1.5,
                'red_flags': ['severe abdominal pain', 'blood in vomit', 'severe dehydration']
            }
        }
        
        symptoms_lower = symptoms.lower()
        
        for category, pattern in symptom_patterns.items():
            found_symptoms = [kw for kw in pattern['keywords'] if kw in symptoms_lower]
            if found_symptoms:
                analysis['detected_symptoms'].extend(found_symptoms)
                analysis['symptom_clusters'].append(category)
                analysis['severity_score'] += len(found_symptoms) * pattern['severity_multiplier']
                
                # Check for red flags
                red_flags = [rf for rf in pattern['red_flags'] if rf in symptoms_lower]
                analysis['red_flags'].extend(red_flags)
        
        # Calculate urgency level
        if analysis['red_flags']:
            analysis['urgency_level'] = 'emergency'
        elif analysis['severity_score'] > 10:
            analysis['urgency_level'] = 'urgent'
        elif analysis['severity_score'] > 5:
            analysis['urgency_level'] = 'semi-urgent'
        
        # Patient-specific risk assessment
        if patient_context:
            risk_factors = patient_context.get('risk_factors', [])
            medical_conditions = patient_context.get('medical_conditions', [])
            
            for condition in medical_conditions:
                condition_lower = str(condition).lower()
                if 'cardiovascular' in analysis['symptom_clusters'] and 'heart' in condition_lower:
                    analysis['patient_specific_risks'].append('Pre-existing cardiac condition increases concern')
                    analysis['severity_score'] += 2
                elif 'respiratory' in analysis['symptom_clusters'] and 'asthma' in condition_lower:
                    analysis['patient_specific_risks'].append('Asthma history requires careful monitoring')
                    analysis['severity_score'] += 1.5
        
        return analysis
    
    def calculate_health_risk_score(self, patient_data: Dict) -> Dict:
        """📊 ULTRA-ADVANCED: Calculate comprehensive health risk score"""
        
        risk_assessment = {
            'overall_score': 0,
            'risk_level': 'low',
            'contributing_factors': [],
            'protective_factors': [],
            'recommendations': [],
            'areas_of_concern': []
        }
        
        base_score = 0
        
        # Age-based risk
        age = patient_data.get('age', 0)
        if age > 75:
            base_score += 15
            risk_assessment['contributing_factors'].append('Advanced age (>75)')
        elif age > 65:
            base_score += 10
            risk_assessment['contributing_factors'].append('Senior age (65-75)')
        elif age > 50:
            base_score += 5
        elif age < 65 and age > 18:
            risk_assessment['protective_factors'].append('Young to middle age')
        
        # Medical conditions
        conditions = patient_data.get('medical_conditions', [])
        for condition in conditions:
            condition_str = str(condition).lower()
            if any(term in condition_str for term in ['diabetes', 'heart', 'cardiovascular']):
                base_score += 20
                risk_assessment['contributing_factors'].append(f'High-risk condition: {condition}')
            elif any(term in condition_str for term in ['hypertension', 'high blood pressure']):
                base_score += 15
                risk_assessment['contributing_factors'].append(f'Moderate-risk condition: {condition}')
            else:
                base_score += 5
                risk_assessment['contributing_factors'].append(f'Medical condition: {condition}')
        
        # Medications (indicator of health issues)
        medications = patient_data.get('medications', [])
        if len(medications) > 5:
            base_score += 10
            risk_assessment['contributing_factors'].append('Multiple medications (>5)')
        elif len(medications) > 2:
            base_score += 5
        
        # Risk level determination
        risk_assessment['overall_score'] = base_score
        
        if base_score >= 40:
            risk_assessment['risk_level'] = 'high'
            risk_assessment['areas_of_concern'] = ['Multiple risk factors present', 'Requires close monitoring']
        elif base_score >= 20:
            risk_assessment['risk_level'] = 'moderate'
            risk_assessment['areas_of_concern'] = ['Some risk factors present']
        else:
            risk_assessment['risk_level'] = 'low'
            risk_assessment['protective_factors'].append('Few risk factors identified')
        
        # Generate recommendations
        if risk_assessment['risk_level'] == 'high':
            risk_assessment['recommendations'] = [
                'Regular medical monitoring recommended',
                'Consider comprehensive health checkup',
                'Discuss symptom patterns with healthcare provider'
            ]
        elif risk_assessment['risk_level'] == 'moderate':
            risk_assessment['recommendations'] = [
                'Annual health checkups recommended',
                'Monitor existing conditions closely'
            ]
        else:
            risk_assessment['recommendations'] = [
                'Maintain current health practices',
                'Routine preventive care sufficient'
            ]
        
        return risk_assessment
    
    def generate_smart_medical_context(self, patient_data: Dict, symptoms: str = None) -> str:
        """🎯 ULTRA-SMART: Generate intelligent medical context for AI processing"""
        
        context_elements = []
        
        # Patient basics
        if patient_data.get('name'):
            context_elements.append(f"Patient: {patient_data['name']}")
        
        age = patient_data.get('age')
        if age:
            age_category = 'pediatric' if age < 18 else 'adult' if age < 65 else 'geriatric'
            context_elements.append(f"Age: {age} ({age_category})")
        
        # Medical conditions with risk stratification
        conditions = patient_data.get('medical_conditions', [])
        if conditions:
            high_risk = []
            moderate_risk = []
            low_risk = []
            
            for condition in conditions:
                condition_str = str(condition).lower()
                if any(term in condition_str for term in ['heart', 'diabetes', 'cancer', 'stroke']):
                    high_risk.append(str(condition))
                elif any(term in condition_str for term in ['hypertension', 'asthma', 'arthritis']):
                    moderate_risk.append(str(condition))
                else:
                    low_risk.append(str(condition))
            
            if high_risk:
                context_elements.append(f"HIGH-RISK CONDITIONS: {', '.join(high_risk)}")
            if moderate_risk:
                context_elements.append(f"Moderate-risk conditions: {', '.join(moderate_risk)}")
            if low_risk:
                context_elements.append(f"Other conditions: {', '.join(low_risk)}")
        
        # Current medications (important for interactions and current treatment)
        medications = patient_data.get('medications', [])
        if medications:
            med_count = len(medications)
            if med_count > 5:
                context_elements.append(f"MULTIPLE MEDICATIONS ({med_count}): Complex medication profile")
            else:
                context_elements.append(f"Current medications: {med_count} documented")
        
        # Allergies (critical safety information)
        allergies = patient_data.get('allergies', [])
        if allergies:
            context_elements.append(f"ALLERGIES: {', '.join(str(a) for a in allergies)}")
        
        # Current symptoms context
        if symptoms:
            symptom_analysis = self.enhanced_symptom_analysis(symptoms, patient_data)
            if symptom_analysis['urgency_level'] in ['emergency', 'urgent']:
                context_elements.append(f"URGENT SYMPTOMS: {symptom_analysis['urgency_level'].upper()} level")
        
        return " | ".join(context_elements) if context_elements else "Basic patient profile available"
    
    def emergency_triage_assessment(self, symptoms: str, patient_context: Dict = None) -> Dict:
        """🚨 ULTRA-CRITICAL: Emergency triage assessment with immediate recommendations"""
        
        triage = {
            'triage_level': 1,  # 1=Emergency, 2=Urgent, 3=Semi-urgent, 4=Standard, 5=Non-urgent
            'immediate_action': '',
            'time_sensitivity': '',
            'emergency_keywords': [],
            'patient_risk_factors': [],
            'clinical_reasoning': []
        }
        
        symptoms_lower = symptoms.lower()
        
        # Emergency indicators (Level 1)
        emergency_indicators = [
            'chest pain', 'heart attack', 'stroke', 'can\'t breathe', 'severe bleeding',
            'unconscious', 'seizure', 'severe head injury', 'poisoning', 'overdose',
            'severe allergic reaction', 'suicide', 'severe burns'
        ]
        
        # Urgent indicators (Level 2)
        urgent_indicators = [
            'difficulty breathing', 'high fever', 'severe pain', 'persistent vomiting',
            'severe headache', 'vision loss', 'weakness one side', 'confusion'
        ]
        
        for indicator in emergency_indicators:
            if indicator in symptoms_lower:
                triage['emergency_keywords'].append(indicator)
                triage['triage_level'] = 1
                triage['immediate_action'] = 'CALL EMERGENCY SERVICES IMMEDIATELY (911)'
                triage['time_sensitivity'] = 'IMMEDIATE - Minutes matter'
                triage['clinical_reasoning'].append(f'Emergency indicator detected: {indicator}')
        
        if triage['triage_level'] > 1:  # Check urgent if not emergency
            for indicator in urgent_indicators:
                if indicator in symptoms_lower:
                    triage['emergency_keywords'].append(indicator)
                    triage['triage_level'] = 2
                    triage['immediate_action'] = 'Seek immediate medical care'
                    triage['time_sensitivity'] = 'Within 1-2 hours'
                    triage['clinical_reasoning'].append(f'Urgent indicator detected: {indicator}')
        
        # Patient-specific risk escalation
        if patient_context:
            conditions = patient_context.get('medical_conditions', [])
            for condition in conditions:
                condition_str = str(condition).lower()
                if 'cardiovascular' in symptoms_lower or 'chest' in symptoms_lower:
                    if 'heart' in condition_str:
                        triage['triage_level'] = min(triage['triage_level'], 1)
                        triage['patient_risk_factors'].append('Pre-existing cardiac condition')
                        triage['clinical_reasoning'].append('Cardiac symptoms + cardiac history = HIGH RISK')
        
        # Default assessment if no emergency/urgent indicators
        if triage['triage_level'] > 2:
            triage['triage_level'] = 3
            triage['immediate_action'] = 'Monitor symptoms and consider medical consultation'
            triage['time_sensitivity'] = 'Within 24-48 hours if symptoms persist'
        
        return triage
    
    def process_advanced_medical_query(self, query: str, patient_context: Dict = None) -> str:
        """🧠 ULTRA-ADVANCED: Process medical queries with all advanced capabilities integrated"""
        
        try:
            # Step 1: Emergency triage assessment
            triage = self.emergency_triage_assessment(query, patient_context)
            
            # Step 2: Enhanced symptom analysis
            symptom_analysis = self.enhanced_symptom_analysis(query, patient_context)
            
            # Step 3: Generate smart medical context
            if patient_context:
                medical_context = self.generate_smart_medical_context(patient_context, query)
                risk_assessment = self.calculate_health_risk_score(patient_context)
            else:
                medical_context = "Limited patient context available"
                risk_assessment = {"risk_level": "unknown", "recommendations": []}
            
            # Step 4: Construct comprehensive response
            response = f"""🏥 **ULTRA-ADVANCED MEDICAL ANALYSIS**

🚨 **TRIAGE ASSESSMENT**: Level {triage['triage_level']} - {triage['immediate_action']}
⏰ **TIME SENSITIVITY**: {triage['time_sensitivity']}

🧠 **SYMPTOM ANALYSIS**:
• Detected symptoms: {', '.join(symptom_analysis['detected_symptoms']) if symptom_analysis['detected_symptoms'] else 'Multiple indicators analyzed'}
• Severity score: {symptom_analysis['severity_score']}/20
• Urgency level: {symptom_analysis['urgency_level'].upper()}

👤 **PATIENT CONTEXT**: {medical_context}
📊 **RISK LEVEL**: {risk_assessment['risk_level'].upper()}

🩺 **CLINICAL REASONING**:"""
            
            # Add clinical reasoning
            for reason in triage['clinical_reasoning']:
                response += f"\n• {reason}"
            
            for risk in symptom_analysis['patient_specific_risks']:
                response += f"\n• {risk}"
            
            # Emergency handling
            if triage['triage_level'] == 1:
                response += f"\n\n🚨 **EMERGENCY PROTOCOL ACTIVATED**"
                response += f"\n{triage['immediate_action']}"
                response += f"\n\n⚠️ **CRITICAL**: Do not delay medical care. This assessment suggests immediate professional evaluation is needed."
            
            # Red flags
            if symptom_analysis['red_flags']:
                response += f"\n\n🔴 **RED FLAGS DETECTED**: {', '.join(symptom_analysis['red_flags'])}"
            
            # Recommendations
            response += f"\n\n📋 **RECOMMENDATIONS**:"
            for rec in risk_assessment['recommendations']:
                response += f"\n• {rec}"
            
            response += f"\n\n⚠️ **MEDICAL DISCLAIMER**: This is an AI-powered diagnostic support tool. It provides preliminary analysis but cannot replace professional medical evaluation. Always consult healthcare providers for definitive diagnosis and treatment."
            
            return response
            
        except Exception as e:
            logger.error(f"Advanced medical query processing error: {e}")
            return "I apologize, but I encountered an error processing your medical query. Please rephrase your question or consult a healthcare provider directly."