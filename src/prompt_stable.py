


"""
Medical Prompt System - Production Ready and Stable
Based on proven reference architecture with enhanced medical consultation capabilities
"""

import re
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def get_production_medical_prompt(medical_content: str, sources: list, query: str, context_info: Dict = None) -> str:
    """
    Stable production medical prompt that generates professional, consultative responses
    """
    
    context_section = ""
    if context_info and context_info.get('has_context'):
        context_section = f"""
CONVERSATION CONTEXT:
{context_info.get('context_summary', 'Previous conversation context available')}

CONTEXT INTEGRATION:
- Build upon previous discussions when relevant
- Reference earlier topics if they relate to current query
- Maintain conversation continuity
- Acknowledge user's ongoing medical concerns
"""

    return f"""You are Dr. MedBot, a professional medical AI assistant conducting a medical consultation. You have access to medical textbooks and should provide expert medical guidance with appropriate follow-up questions.

PATIENT QUERY: {query}

MEDICAL TEXTBOOK INFORMATION:
{medical_content}
{context_section}

CONSULTATION APPROACH:
1. MEDICAL ASSESSMENT: Acknowledge the patient's concern and provide relevant medical information from the textbook sources
2. FOLLOW-UP QUESTIONS: Ask 2-3 specific diagnostic questions to gather more details:
   - Duration and timeline of symptoms
   - Severity and impact on daily activities  
   - Associated symptoms or triggers
   - Medical history or current medications (when relevant)
3. RECOMMENDATIONS: Provide appropriate medical guidance and next steps
4. PROFESSIONAL CARE: Advise when to seek medical consultation

RESPONSE STRUCTURE:
**Medical Assessment**: [Brief acknowledgment + relevant medical information from textbooks]
**Follow-up Questions**: [2-3 specific questions to gather diagnostic information]
**Recommendations**: [Evidence-based guidance and next steps]

GUIDELINES:
• Keep response professional and concise (200-350 words)
• Use medical textbook information as primary source
• Ask relevant follow-up questions like a doctor would
• Provide practical medical guidance
• Reference textbook sources when available
• Maintain empathetic, professional tone

End with: "⚠️ This information combines medical textbook evidence with clinical guidance. Always consult healthcare professionals for proper medical evaluation and treatment."
"""

def get_medical_no_textbook_prompt(query: str, context_info: Dict = None) -> str:
    """
    Stable fallback prompt when no textbook information is available
    """
    
    context_awareness = ""
    if context_info and context_info.get('has_context'):
        context_awareness = f"""
CONVERSATION CONTEXT: Building on our previous medical discussions about {context_info.get('previous_topics', 'your health concerns')}.
"""

    return f"""You are Dr. MedBot, a medical AI assistant. A patient asked: "{query}"

SITUATION: No specific information found in the medical textbook database.
{context_awareness}

YOUR RESPONSE SHOULD:
1. Acknowledge their concern professionally
2. Ask relevant diagnostic questions to understand their situation:
   - What specific symptoms are they experiencing?
   - How long has this been going on?
   - What prompted them to ask this question?
   - Any associated symptoms or concerns?
   - Relevant medical history or current medications
3. Provide general medical guidance based on the symptom category
4. Recommend appropriate medical consultation

RESPONSE FORMAT:
**Understanding Your Concern**: [Professional acknowledgment]
**Diagnostic Questions**: [2-3 specific questions to gather information]
**General Guidance**: [Appropriate medical advice for the symptom category]
**Next Steps**: [Recommendation for medical consultation]

Example: "I understand your concern about [symptom/condition]. To provide better guidance, could you tell me: [specific questions]? Based on your answers, I can provide more targeted guidance and determine the urgency of medical consultation needed."

⚠️ For accurate medical information and proper diagnosis, please consult qualified healthcare professionals.
"""

def get_general_knowledge_prompt(context_info: Dict = None) -> str:
    """
    Stable prompt for non-medical queries
    """
    
    context_section = ""
    if context_info and context_info.get('has_context'):
        context_section = """
CONVERSATION CONTEXT: Consider our previous discussion when providing your response.

"""

    return f"""You are a helpful AI assistant responding to a general (non-medical) question.

{context_section}RESPONSE GUIDELINES:
• Provide clear, accurate, and helpful information
• Keep responses focused and well-organized
• Use appropriate tone for the question type
• Include relevant examples when helpful
• Be conversational yet informative
• Reference previous conversation when relevant

Since this is not a medical query, you can use your full knowledge base to provide a comprehensive answer. No medical disclaimers are needed unless the topic touches on health matters.
"""

def get_greeting_introduction_prompt(context_info: Dict = None) -> str:
    """
    Stable greeting prompt with context awareness
    """
    
    returning_user = context_info and context_info.get('is_returning_user', False)
    
    if returning_user:
        return """You are Dr. MedBot, a specialized medical AI assistant welcoming back a returning user.

RETURNING USER GREETING:
🏥 **Welcome Back!** I remember our previous medical discussions and I'm here to continue helping with your health questions.

INTRODUCTION:
📚 **Medical Knowledge**: I provide evidence-based medical information from authoritative medical textbooks
🔄 **Conversation Memory**: I can reference our previous discussions for better context
🔍 **How I Work**: I search medical literature and ask follow-up questions like a doctor would
📝 **Citations**: All medical responses include textbook sources and references
🎯 **Professional Care**: I guide you toward appropriate medical consultation when needed
⚠️ **Important**: I provide educational information - always consult healthcare professionals for medical advice

RESPONSE STYLE:
• Warm and welcoming tone acknowledging previous interactions
• Brief explanation of capabilities
• Keep welcome concise (3-4 sentences)
• Invite medical questions

Example: "Welcome back! I remember our previous medical discussions and I'm ready to continue helping with your health questions. What would you like to explore today?"
"""
    
    else:
        return """You are Dr. MedBot, a specialized medical AI assistant powered by medical textbook knowledge.

INTRODUCE YOURSELF WITH:
🏥 **Purpose**: I provide evidence-based medical information from authoritative medical textbooks
📚 **Knowledge Source**: Medical textbooks and clinical literature
🔍 **How I Work**: I search medical literature and ask follow-up questions to provide better guidance
📝 **Citations**: All medical responses include textbook sources and page references
🎯 **Professional Approach**: I conduct consultations with appropriate diagnostic questions
⚠️ **Important**: I provide educational information - always consult healthcare professionals for medical advice

RESPONSE STYLE:
• Professional yet approachable
• Briefly explain your medical consultation approach
• Mention the medical textbook database
• Keep introduction concise (3-4 sentences)
• Invite medical questions

Example: "Hello! I'm Dr. MedBot, your medical AI assistant that provides evidence-based information from medical textbooks. I ask follow-up questions like a doctor would to provide better guidance. What medical topic would you like to learn about today?"
"""

def create_smart_medical_classifier(query: str, context_info: Dict = None) -> dict:
    """
    Stable medical query classifier with confidence scoring
    """
    
    query_lower = query.lower()
    
    # Enhanced medical keywords organized by categories
    medical_keywords = {
        'core_medical': [
            'medical', 'health', 'healthcare', 'clinical', 'diagnosis', 'treatment',
            'therapy', 'cure', 'healing', 'recovery', 'medicine', 'medication',
            'drug', 'pharmaceutical', 'prescription', 'dosage', 'side effect'
        ],
        'diseases_conditions': [
            'diabetes', 'hypertension', 'cancer', 'tumor', 'heart disease', 'stroke',
            'asthma', 'copd', 'pneumonia', 'bronchitis', 'tuberculosis', 'covid',
            'flu', 'influenza', 'cold', 'fever', 'infection', 'virus', 'bacteria',
            'arthritis', 'osteoporosis', 'alzheimer', 'parkinson', 'epilepsy',
            'depression', 'anxiety', 'bipolar', 'schizophrenia', 'ptsd',
            'hepatitis', 'cirrhosis', 'kidney disease', 'liver disease',
            'anemia', 'leukemia', 'lymphoma', 'migraine', 'headache'
        ],
        'symptoms': [
            'pain', 'ache', 'hurt', 'discomfort', 'soreness', 'burning',
            'headache', 'migraine', 'dizziness', 'vertigo', 'nausea',
            'vomiting', 'diarrhea', 'constipation', 'bloating', 'cramps',
            'cough', 'wheeze', 'shortness of breath', 'chest pain',
            'back pain', 'neck pain', 'joint pain', 'muscle pain',
            'fatigue', 'weakness', 'tiredness', 'exhaustion',
            'fever', 'chills', 'sweating', 'rash', 'itching', 'swelling',
            'bleeding', 'bruising', 'numbness', 'tingling',
            'insomnia', 'sleep disorder', 'appetite loss', 'weight loss'
        ],
        'anatomy': [
            'heart', 'cardiac', 'blood pressure', 'circulation',
            'lung', 'respiratory', 'breathing', 'airway',
            'brain', 'neurological', 'nervous system', 'spine',
            'liver', 'kidney', 'stomach', 'intestine', 'digestive',
            'blood', 'bone', 'muscle', 'skin', 'eye', 'ear'
        ],
        'question_patterns': [
            'what is', 'what are', 'what causes', 'how to treat',
            'symptoms of', 'signs of', 'treatment for', 'cure for',
            'medicine for', 'prevention of', 'risk factors'
        ]
    }
    
    # Calculate keyword matches and scores
    keyword_score = 0
    category_matches = {}
    
    for category, keywords in medical_keywords.items():
        category_matches[category] = []
        for keyword in keywords:
            if keyword in query_lower:
                category_matches[category].append(keyword)
        
        # Weight different categories
        weights = {
            'core_medical': 3,
            'diseases_conditions': 4,
            'symptoms': 3,
            'anatomy': 2,
            'question_patterns': 1
        }
        
        if category_matches[category]:
            keyword_score += len(category_matches[category]) * weights.get(category, 1)
    
    # Pattern matching for medical queries
    medical_patterns = [
        r'\b(what|how|why)\s+.*\b(disease|symptom|treatment|medicine|cure)\b',
        r'\b(symptoms?|treatment|cure|medicine)\s+(of|for)\b',
        r'\b(pain|ache|hurt)\s+(in|of)\b',
        r'\b(how\s+to\s+(treat|cure|prevent))\b',
        r'\b(side\s+effects?|dosage)\b'
    ]
    
    pattern_matches = sum(1 for pattern in medical_patterns 
                        if re.search(pattern, query_lower))
    
    # Context boost for users with medical history
    context_boost = 0
    if context_info and context_info.get('has_medical_history'):
        context_boost = 3
    
    # Calculate final confidence score
    total_score = keyword_score + (pattern_matches * 3) + context_boost
    confidence = min(100, total_score * 6)
    
    is_medical = confidence >= 20
    
    return {
        'is_medical': is_medical,
        'confidence': confidence,
        'category_matches': {k: v for k, v in category_matches.items() if v},
        'pattern_matches': pattern_matches,
        'context_boost': context_boost,
        'classification': 'medical' if is_medical else 'general',
        'primary_categories': [cat for cat, matches in category_matches.items() if matches][:3]
    }

def select_optimal_prompt(query: str, has_textbook_knowledge: bool = False, 
                        medical_content: str = "", sources: list = [], 
                        context_info: Dict = None) -> str:
    """
    Stable prompt selection based on query type and available knowledge
    """
    
    classification = create_smart_medical_classifier(query, context_info)
    
    if classification['is_medical']:
        if has_textbook_knowledge and medical_content:
            # Full medical response with textbook evidence
            return get_production_medical_prompt(medical_content, sources, query, context_info)
        else:
            # No textbook knowledge but medical query
            return get_medical_no_textbook_prompt(query, context_info)
    else:
        # Check if it's a greeting/introduction
        greeting_keywords = ['hello', 'hi', 'hey', 'who are you', 'what are you', 'introduce']
        if any(keyword in query.lower() for keyword in greeting_keywords):
            return get_greeting_introduction_prompt(context_info)
        else:
            return get_general_knowledge_prompt(context_info)

def validate_medical_response_quality(response: str, query: str, context_info: Dict = None) -> dict:
    """
    Stable quality validation for medical responses
    """
    
    quality_metrics = {
        'length_appropriate': 100 <= len(response) <= 500,
        'has_medical_disclaimer': '⚠️' in response and 'healthcare professionals' in response.lower(),
        'addresses_query': any(term in response.lower() for term in query.lower().split()),
        'professional_tone': not any(word in response.lower() for word in ['awesome', 'cool', 'wow', 'amazing']),
        'has_structure': '**' in response,  # Check for structured response
        'concise': response.count('\n\n') <= 4,
        'no_repetition': len(set(response.split('. '))) > len(response.split('. ')) * 0.8
    }
    
    # Check for follow-up questions (key feature)
    has_followup = any(indicator in response.lower() for indicator in [
        'could you tell me', 'can you describe', 'how long', 'what specific',
        'any other symptoms', 'have you noticed', 'do you have'
    ])
    quality_metrics['has_followup_questions'] = has_followup
    
    # Medical terminology check
    medical_terms = [
        'diagnosis', 'treatment', 'symptoms', 'condition', 'medication', 'therapy',
        'clinical', 'medical', 'health', 'patient', 'disease'
    ]
    medical_term_count = sum(1 for term in medical_terms if term in response.lower())
    quality_metrics['adequate_medical_terminology'] = medical_term_count >= 2
    
    quality_score = sum(quality_metrics.values()) / len(quality_metrics) * 100
    
    return {
        'quality_score': round(quality_score, 1),
        'metrics': quality_metrics,
        'recommendation': 'approved' if quality_score >= 70 else 'needs_improvement',
        'word_count': len(response.split()),
        'medical_term_count': medical_term_count,
        'has_followup': has_followup
    }

def optimize_medical_content_extraction(chunks: list, query: str, context_info: Dict = None, max_sentences: int = 6) -> str:
    """
    Stable content extraction focused on relevance and clarity
    """
    
    query_terms = set(query.lower().split())
    optimized_sections = []
    
    for chunk in chunks[:3]:  # Focus on top 3 chunks
        content = chunk.get('content', '')
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
        
        # Score sentences by relevance
        scored_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            relevance_score = sum(1 for term in query_terms if term in sentence_lower)
            
            # Boost for medical keywords
            medical_boost = sum(1 for keyword in ['treatment', 'symptom', 'cause', 'diagnosis', 'medication'] 
                            if keyword in sentence_lower)
            
            total_score = relevance_score + medical_boost
            scored_sentences.append((sentence, total_score))
        
        # Select top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        selected_sentences = [s[0] for s in scored_sentences[:3] if s[1] > 0]
        
        if selected_sentences:
            section = '. '.join(selected_sentences)
            if not section.endswith('.'):
                section += '.'
            
            # Add citation
            book_name = chunk.get('book_name', 'Medical Textbook')
            page = chunk.get('page', 'N/A')
            section += f" [{book_name}, p.{page}]"
            
            optimized_sections.append(section)
    
    return '\n\n'.join(optimized_sections)

def create_context_summary(recent_messages: List[Dict], max_length: int = 150) -> str:
    """
    Stable context summary creation
    """
    if not recent_messages:
        return ""
    
    # Extract key medical terms from recent messages
    medical_terms = []
    for msg in recent_messages[-3:]:  # Last 3 messages
        content = msg.get('message', '').lower()
        medical_keywords = ['symptom', 'treatment', 'diagnosis', 'medication', 'condition', 'pain', 'health']
        found_terms = [term for term in medical_keywords if term in content]
        medical_terms.extend(found_terms)
    
    unique_terms = list(set(medical_terms))
    
    if unique_terms:
        summary = "Previous discussion covered: " + ", ".join(unique_terms[:3])
        return summary[:max_length] + "..." if len(summary) > max_length else summary
    
    return "Continuing medical discussion"

# Export key functions for main application
__all__ = [
    'get_production_medical_prompt',
    'get_medical_no_textbook_prompt',
    'get_general_knowledge_prompt',
    'get_greeting_introduction_prompt',
    'create_smart_medical_classifier',
    'select_optimal_prompt',
    'validate_medical_response_quality',
    'optimize_medical_content_extraction',
    'create_context_summary'
]

def test_stable_prompt_system():
    """
    Test function to validate the stable prompt system
    """
    
    test_queries = [
        "What are the symptoms of diabetes?",
        "I have a headache that won't go away",
        "How to treat high blood pressure?",
        "Hello, who are you?",
        "What's the weather like today?"
    ]
    
    results = {}
    
    for query in test_queries:
        classification = create_smart_medical_classifier(query)
        
        if classification['is_medical']:
            prompt = get_production_medical_prompt("Sample medical content", [], query)
            prompt_type = "medical"
        else:
            prompt_type = "general"
            if any(keyword in query.lower() for keyword in ['hello', 'who are you']):
                prompt = get_greeting_introduction_prompt()
                prompt_type = "greeting"
            else:
                prompt = get_general_knowledge_prompt()
        
        results[query] = {
            'classification': classification,
            'prompt_type': prompt_type,
            'prompt_length': len(prompt.split()),
            'confidence': classification['confidence']
        }
    
    return results

if __name__ == "__main__":
    # Run test
    test_results = test_stable_prompt_system()
    print("Stable Prompt System Test Results:")
    print("=" * 50)
    
    for query, result in test_results.items():
        print(f"\nQuery: {query}")
        print(f"Classification: {result['classification']['classification']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print(f"Prompt Type: {result['prompt_type']}")
        print(f"Prompt Length: {result['prompt_length']} words")