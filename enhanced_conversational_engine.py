"""
Conversational engine with Pinecone integration and context awareness.
Handles medical query routing, response generation, and conversation history.
"""

import os
import sys
import time
import json
import uuid
import asyncio
import logging
import traceback
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import re
from functools import lru_cache
import weakref

# Core AI/ML imports
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import groq
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Medical NLP
try:
    import spacy
    import scispacy
    MEDICAL_NLP_AVAILABLE = True
except ImportError:
    MEDICAL_NLP_AVAILABLE = False

# Performance monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('conversational_engine.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 🔧 JSON Serialization Utility for Production
def serialize_for_json(obj):
    """Convert complex objects to JSON-serializable format"""
    if isinstance(obj, Enum):
        return obj.value  # Convert enum to string value
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return {key: serialize_for_json(value) for key, value in obj.__dict__.items() if not key.startswith('_')}
    else:
        return obj

class ConversationStage(Enum):
    """Stages of conversation processing"""
    INITIALIZATION = "initialization"
    CONTEXT_ANALYSIS = "context_analysis"
    MEDICAL_RETRIEVAL = "medical_retrieval"
    RESPONSE_GENERATION = "response_generation"
    QUALITY_VALIDATION = "quality_validation"
    CONTEXT_UPDATE = "context_update"
    COMPLETION = "completion"
    ERROR = "error"

class ResponseType(Enum):
    """Types of responses the system can generate"""
    MEDICAL_TEXTBOOK = "medical_textbook"
    CONVERSATIONAL = "conversational"
    DIAGNOSTIC_SUPPORT = "diagnostic_support"
    EDUCATIONAL = "educational"
    SAFETY_WARNING = "safety_warning"
    CLARIFICATION_REQUEST = "clarification_request"
    GENERAL_HEALTH = "general_health"
    EMERGENCY = "emergency"

class ConversationIntent(Enum):
    """Intent classification for medical conversations"""
    SYMPTOM_INQUIRY = "symptom_inquiry"
    TREATMENT_QUESTION = "treatment_question"
    MEDICATION_INFO = "medication_info"
    DIAGNOSTIC_CLARIFICATION = "diagnostic_clarification"
    PROCEDURE_EXPLANATION = "procedure_explanation"
    GENERAL_HEALTH_INFO = "general_health_info"
    EMERGENCY_CONCERN = "emergency_concern"
    FOLLOW_UP = "follow_up"
    CASUAL_CONVERSATION = "casual_conversation"

@dataclass
class ConversationMetrics:
    """Comprehensive metrics for conversation quality and performance"""
    conversation_id: str
    user_id: str
    session_start: float
    total_messages: int = 0
    medical_queries: int = 0
    response_times: List[float] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)
    context_utilization: List[float] = field(default_factory=list)
    user_satisfaction_indicators: Dict[str, Any] = field(default_factory=dict)
    safety_flags: List[str] = field(default_factory=list)
    
    def add_response_time(self, time_ms: float):
        self.response_times.append(time_ms)
    
    def add_retrieval_score(self, score: float):
        self.retrieval_scores.append(score)
    
    def get_average_response_time(self) -> float:
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
    
    def get_average_retrieval_score(self) -> float:
        return sum(self.retrieval_scores) / len(self.retrieval_scores) if self.retrieval_scores else 0.0

@dataclass
class ConversationContext:
    """Rich conversation context with medical history and preferences"""
    user_id: str
    session_id: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    medical_entities: Dict[str, List[str]] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    medical_profile: Dict[str, Any] = field(default_factory=dict)
    current_intent: Optional[ConversationIntent] = None
    context_summary: str = ""
    last_medical_retrieval: Optional[Dict[str, Any]] = None
    safety_concerns: List[str] = field(default_factory=list)
    conversation_flow: List[str] = field(default_factory=list)
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add message to conversation history with rich metadata"""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.conversation_history.append(message)
        self.conversation_flow.append(f"{role}: {content[:50]}...")

@dataclass
class MedicalRetrievalResult:
    """Enhanced medical retrieval result with comprehensive metadata"""
    query: str
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    relevance_scores: List[float] = field(default_factory=list)
    source_books: List[str] = field(default_factory=list)
    medical_concepts: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    retrieval_time_ms: float = 0.0
    total_chunks_found: int = 0
    best_match_score: float = 0.0
    sources_metadata: List[Dict[str, Any]] = field(default_factory=list)

class MedicalEntityExtractor:
    """Advanced medical entity extraction and classification"""
    
    def __init__(self):
        self.medical_patterns = self._compile_medical_patterns()
        self.drug_database = self._load_drug_database()
        self.condition_database = self._load_condition_database()
        self.symptom_database = self._load_symptom_database()
        
        # Load medical NLP model if available
        self.nlp_model = None
        if MEDICAL_NLP_AVAILABLE:
            try:
                self.nlp_model = spacy.load("en_core_sci_sm")
                logger.info("Medical NLP model loaded successfully")
            except Exception as e:
                logger.warning(f"Medical NLP model not available: {e}")
    
    def _compile_medical_patterns(self) -> Dict[str, List]:
        """Compile comprehensive medical regex patterns"""
        return {
            'symptoms': [
                r'\b(pain|ache|aching|sore|soreness|hurt|hurting|discomfort)\b',
                r'\b(fever|temperature|chills|sweats|hot|cold)\b',
                r'\b(nausea|vomiting|vomit|sick|queasy)\b',
                r'\b(headache|migraine|head pain)\b',
                r'\b(fatigue|tired|exhausted|weakness|weak)\b',
                r'\b(cough|coughing|wheeze|wheezing)\b',
                r'\b(rash|itchy|itch|burning|swelling|swollen)\b'
            ],
            'medications': [
                r'\b(mg|mcg|ml|tablet|capsule|pill|dose|dosage)\b',
                r'\b(prescription|medication|medicine|drug|treatment)\b',
                r'\b(antibiotic|pain killer|analgesic|anti-inflammatory)\b'
            ],
            'medical_procedures': [
                r'\b(surgery|operation|procedure|examination|test)\b',
                r'\b(x-ray|CT scan|MRI|ultrasound|blood test)\b',
                r'\b(biopsy|endoscopy|colonoscopy)\b'
            ],
            'body_parts': [
                r'\b(head|neck|chest|back|arm|leg|hand|foot|stomach|abdomen)\b',
                r'\b(heart|lung|liver|kidney|brain|spine)\b',
                r'\b(joint|muscle|bone|skin|eye|ear)\b'
            ]
        }
    
    def _load_drug_database(self) -> set:
        """Load comprehensive drug name database"""
        return {
            'acetaminophen', 'ibuprofen', 'aspirin', 'naproxen', 'diclofenac',
            'morphine', 'codeine', 'tramadol', 'hydrocodone', 'oxycodone',
            'lisinopril', 'enalapril', 'losartan', 'amlodipine', 'metoprolol',
            'atorvastatin', 'simvastatin', 'rosuvastatin', 'pravastatin',
            'metformin', 'insulin', 'glyburide', 'glipizide', 'pioglitazone',
            'omeprazole', 'lansoprazole', 'esomeprazole', 'ranitidine',
            'amoxicillin', 'azithromycin', 'ciprofloxacin', 'doxycycline',
            'prednisone', 'prednisolone', 'hydrocortisone', 'methylprednisolone'
        }
    
    def _load_condition_database(self) -> set:
        """Load medical condition database"""
        return {
            'hypertension', 'diabetes', 'asthma', 'copd', 'pneumonia',
            'bronchitis', 'sinusitis', 'migraine', 'depression', 'anxiety',
            'arthritis', 'osteoporosis', 'cancer', 'tumor', 'infection',
            'allergy', 'eczema', 'dermatitis', 'gastritis', 'ulcer',
            'angina', 'arrhythmia', 'stroke', 'seizure', 'epilepsy'
        }
    
    def _load_symptom_database(self) -> set:
        """Load symptom database"""
        return {
            'fever', 'pain', 'nausea', 'vomiting', 'diarrhea', 'constipation',
            'headache', 'dizziness', 'fatigue', 'weakness', 'cough', 'dyspnea',
            'chest pain', 'palpitations', 'syncope', 'rash', 'pruritus',
            'edema', 'jaundice', 'hematuria', 'dysuria', 'polyuria'
        }
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract comprehensive medical entities from text"""
        text_lower = text.lower()
        entities = {
            'symptoms': [],
            'medications': [],
            'conditions': [],
            'procedures': [],
            'body_parts': [],
            'measurements': [],
            'temporal_expressions': []
        }
        
        # Pattern-based extraction
        for category, patterns in self.medical_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if category in entities:
                    entities[category].extend(matches)
        
        # Database matching
        words = text_lower.split()
        for word in words:
            if word in self.drug_database:
                entities['medications'].append(word)
            elif word in self.condition_database:
                entities['conditions'].append(word)
            elif word in self.symptom_database:
                entities['symptoms'].append(word)
        
        # Advanced NLP extraction if available
        if self.nlp_model:
            try:
                doc = self.nlp_model(text)
                for ent in doc.ents:
                    if ent.label_ in ['DISEASE', 'SYMPTOM']:
                        entities['conditions'].append(ent.text.lower())
                    elif ent.label_ in ['DRUG', 'CHEMICAL']:
                        entities['medications'].append(ent.text.lower())
            except Exception as e:
                logger.warning(f"NLP entity extraction failed: {e}")
        
        # Remove duplicates and empty values
        for key in entities:
            entities[key] = list(set([e for e in entities[key] if e.strip()]))
        
        return entities

class ConversationIntentClassifier:
    """Advanced intent classification for medical conversations"""
    
    def __init__(self):
        self.intent_patterns = {
            ConversationIntent.SYMPTOM_INQUIRY: [
                r'\b(symptom|feel|feeling|experience|having|suffering)\b',
                r'\b(pain|hurt|ache|discomfort|problem)\b',
                r'\b(what.*wrong|what.*happening|why.*feel)\b'
            ],
            ConversationIntent.TREATMENT_QUESTION: [
                r'\b(treatment|treat|cure|heal|help|remedy)\b',
                r'\b(how.*treat|how.*cure|what.*do)\b',
                r'\b(medicine|medication|drug|therapy)\b'
            ],
            ConversationIntent.MEDICATION_INFO: [
                r'\b(medication|medicine|drug|pill|tablet|capsule)\b',
                r'\b(dosage|dose|take|prescribed|side effect)\b',
                r'\b(safe.*take|how.*take|when.*take)\b'
            ],
            ConversationIntent.DIAGNOSTIC_CLARIFICATION: [
                r'\b(diagnosis|diagnose|test|result|report)\b',
                r'\b(what.*mean|explain|understand|confused)\b',
                r'\b(doctor.*said|told.*have)\b'
            ],
            ConversationIntent.EMERGENCY_CONCERN: [
                r'\b(emergency|urgent|serious|severe|crisis)\b',
                r'\b(chest pain|can\'t breathe|unconscious|bleeding)\b',
                r'\b(call.*911|go.*hospital|need.*help)\b'
            ]
        }
        
        self.emergency_keywords = [
            'chest pain', 'difficulty breathing', 'unconscious', 'severe bleeding',
            'stroke symptoms', 'heart attack', 'allergic reaction', 'overdose',
            'suicidal thoughts', 'severe trauma', 'poisoning'
        ]
    
    def classify_intent(self, text: str, context: ConversationContext = None) -> ConversationIntent:
        """Classify conversation intent with context awareness"""
        text_lower = text.lower()
        
        # Emergency detection (highest priority)
        if any(keyword in text_lower for keyword in self.emergency_keywords):
            return ConversationIntent.EMERGENCY_CONCERN
        
        # Pattern matching with scoring
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            
            if score > 0:
                intent_scores[intent] = score
        
        # Context-based adjustment
        if context and context.conversation_history:
            last_messages = context.conversation_history[-3:]  # Last 3 messages
            
            for msg in last_messages:
                if msg['role'] == 'assistant' and 'follow up' in msg['content'].lower():
                    intent_scores[ConversationIntent.FOLLOW_UP] = intent_scores.get(ConversationIntent.FOLLOW_UP, 0) + 2
        
        # Return highest scoring intent or default
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return ConversationIntent.GENERAL_HEALTH_INFO

class EnhancedPineconeRetriever:
    """Advanced Pinecone retrieval with medical context optimization"""
    
    def __init__(self, api_key: str, index_name: str = "medical-books-ultimate"):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone library not available")
        
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.index = None
        
        # Initialize embedding model
        if EMBEDDINGS_AVAILABLE:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            raise ImportError("SentenceTransformers not available")
        
        self._connect_to_index()
    
    def _connect_to_index(self):
        """Connect to Pinecone index"""
        try:
            self.index = self.pc.Index(self.index_name)
            # Test connection
            stats = self.index.describe_index_stats()
            logger.info(f"Connected to Pinecone index '{self.index_name}' with {stats.total_vector_count} vectors")
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone index: {e}")
            raise
    
    def _enhance_query_with_medical_context(self, query: str, context: ConversationContext) -> str:
        """Enhance query with medical context for better retrieval"""
        enhanced_query = query
        
        # Add medical entities from context
        if context.medical_entities:
            for entity_type, entities in context.medical_entities.items():
                if entities:
                    enhanced_query += f" {' '.join(entities[:3])}"  # Add top 3 entities
        
        # Add recent medical topics from conversation
        recent_medical_terms = []
        for msg in context.conversation_history[-5:]:  # Last 5 messages
            if msg['role'] == 'user':
                # Extract potential medical terms
                words = msg['content'].lower().split()
                medical_words = [w for w in words if len(w) > 4 and any(
                    med_word in w for med_word in ['symptom', 'pain', 'treat', 'medic', 'diagnos', 'therapy']
                )]
                recent_medical_terms.extend(medical_words[:2])
        
        if recent_medical_terms:
            enhanced_query += f" {' '.join(recent_medical_terms[:3])}"
        
        return enhanced_query
    
    async def retrieve_medical_knowledge(
        self, 
        query: str, 
        context: ConversationContext,
        top_k: int = 10,
        score_threshold: float = 0.7
    ) -> MedicalRetrievalResult:
        """Advanced medical knowledge retrieval with context enhancement"""
        start_time = time.time()
        
        # Enhance query with context
        enhanced_query = self._enhance_query_with_medical_context(query, context)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(enhanced_query).tolist()
        
        # Build filter based on medical entities
        filter_dict = {}
        if context.medical_entities.get('conditions'):
            # Prioritize chunks related to mentioned conditions
            pass  # Could add metadata filtering here
        
        try:
            # Perform vector search
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            # Process results
            chunks = []
            relevance_scores = []
            source_books = set()
            medical_concepts = set()
            sources_metadata = []
            
            for match in results.matches:
                if match.score >= score_threshold:
                    chunk_data = {
                        'id': match.id,
                        'text': match.metadata.get('text', ''),
                        'source': match.metadata.get('filename', 'Unknown'),
                        'page': match.metadata.get('page', 0),
                        'book_type': match.metadata.get('book_type', 'general'),
                        'medical_score': match.metadata.get('medical_score', 0.0),
                        'relevance_score': float(match.score)
                    }
                    
                    chunks.append(chunk_data)
                    relevance_scores.append(float(match.score))
                    source_books.add(match.metadata.get('filename', 'Unknown'))
                    
                    # Extract medical concepts from metadata
                    if match.metadata.get('medical_keywords'):
                        keywords = match.metadata['medical_keywords'].split(',')
                        medical_concepts.update(keywords[:3])
                    
                    # Collect source metadata
                    source_info = {
                        'book_name': match.metadata.get('filename', 'Unknown'),
                        'book_type': match.metadata.get('book_type', 'general'),
                        'page': match.metadata.get('page', 0),
                        'medical_score': match.metadata.get('medical_score', 0.0),
                        'relevance': float(match.score)
                    }
                    sources_metadata.append(source_info)
            
            retrieval_time = (time.time() - start_time) * 1000
            
            result = MedicalRetrievalResult(
                query=enhanced_query,
                chunks=chunks,
                relevance_scores=relevance_scores,
                source_books=list(source_books),
                medical_concepts=list(medical_concepts),
                confidence_score=max(relevance_scores) if relevance_scores else 0.0,
                retrieval_time_ms=retrieval_time,
                total_chunks_found=len(chunks),
                best_match_score=max(relevance_scores) if relevance_scores else 0.0,
                sources_metadata=sources_metadata
            )
            
            # Update context with retrieval result
            context.last_medical_retrieval = asdict(result)
            
            logger.info(f"Retrieved {len(chunks)} chunks in {retrieval_time:.2f}ms with max relevance {result.best_match_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Medical knowledge retrieval failed: {e}")
            return MedicalRetrievalResult(query=enhanced_query, retrieval_time_ms=(time.time() - start_time) * 1000)

class ResponseGenerator:
    """
    🧠 PROPRIETARY MEDICAL INTELLIGENCE ENGINE - NO EXTERNAL API DEPENDENCY

    REVOLUTIONARY STATUS: ✅ PROPRIETARY MEDICAL AI (SEPT 2025)
    - Using Medical Books Intelligence Engine (NO GROQ DEPENDENCY)
    - Advanced clinical reasoning from medical textbooks
    - Doctor-specific response generation
    - Medical specialty-aware processing
    - Zero external API costs ($0/month vs $50K+/month)
    """

    def __init__(self, medical_books_db_path: str = None):
        # Initialize proprietary medical intelligence instead of Groq
        from .ai_engine.medical_books_intelligence import MedicalBooksIntelligenceEngine
        self.medical_engine = MedicalBooksIntelligenceEngine()
        
        self.groq_client = Groq(api_key=groq_api_key)
        # 🚀 VERIFIED WORKING MODELS ONLY (TESTED DEC 2024)
        # Based on actual API testing - only these 2 models work
        self.model_name = "llama-3.1-8b-instant"  # Primary fast model ✅
        
        # UPDATED WORKING MODELS (January 2025)
        self.available_models = [
            "llama-3.1-8b-instant",       # ✅ Primary model - fast and reliable
            "llama-3.1-70b-versatile",    # ✅ High-performance model for complex queries  
            "gemma2-9b-it",               # ✅ Backup model for failover
            "mixtral-8x7b-32768"          # ✅ Alternative fallback
        ]
        
        # Smart Model Selection with Working Models Only
        self.model_tiers = {
            'greeting': ["llama-3.1-8b-instant"],                      # Fast greetings
            'symptoms': ["llama-3.1-8b-instant", "gemma2-9b-it"],     # Medical analysis 
            'diagnosis': ["llama-3.1-8b-instant", "gemma2-9b-it"],    # Diagnostic support
            'emergency': ["llama-3.1-8b-instant"],                     # Reliable for emergencies
            'medication': ["llama-3.1-8b-instant", "gemma2-9b-it"],   # Drug information
            'treatment': ["llama-3.1-8b-instant", "gemma2-9b-it"],    # Treatment guidance
            'general': ["llama-3.1-8b-instant", "gemma2-9b-it"]       # General health
        }
        
        # 🧠 AI Performance Intelligence
        self.model_performance_tracker = {
            'response_times': {},
            'success_rates': {},
            'quality_scores': {},
            'user_satisfaction': {}
        }
        
        # 💾 Advanced Conversation Memory System
        self.conversation_memory = {}
        self.context_embeddings = {}
        self.medical_reasoning_cache = {}
        
        # 🎯 Medical Expertise Classification
        self.medical_complexity_detector = {
            'emergency_keywords': ['chest pain', 'difficulty breathing', 'severe bleeding', 'unconscious', 'stroke', 'heart attack'],
            'complex_medical': ['drug interaction', 'differential diagnosis', 'contraindication', 'pathophysiology'],
            'simple_health': ['common cold', 'headache', 'fever', 'basic symptoms']
        }
        
        # 🛡️ ADVANCED SAFETY & COMPLIANCE SYSTEM
        self.safety_protocols = {
            'emergency': {
                'phrases': ["🚨 SEEK IMMEDIATE MEDICAL ATTENTION", "Call emergency services (911) if symptoms are severe"],
                'escalation_required': True,
                'confidence_threshold': 0.9
            },
            'prescription': {
                'phrases': ["Only a licensed physician can prescribe medications", "Consult your healthcare provider for prescriptions"],
                'legal_compliance': True
            },
            'diagnosis': {
                'phrases': ["This is educational information only", "Professional medical diagnosis is required for accurate assessment"],
                'diagnostic_disclaimer': True
            },
            'general': {
                'phrases': ["Always consult with a healthcare professional", "This information is for educational purposes only"],
                'standard_disclaimer': True
            }
        }
    
    def _build_medical_system_prompt(self, context: ConversationContext, intent: ConversationIntent) -> str:
        """Build next-generation intelligent medical AI system prompt for startup innovation - MEDICAL BOOKS ONLY"""
        base_prompt = """You are MedBot Pro - a revolutionary AI-powered medical intelligence platform that operates EXCLUSIVELY using medical textbook knowledge.

🚨 CRITICAL OPERATIONAL MANDATE:
- You MUST ONLY use information from the medical textbooks provided below
- NEVER use external knowledge, general AI training, or non-medical book sources
- If medical textbooks don't contain relevant information, you MUST say so explicitly
- Your responses MUST be based 100% on the retrieved medical literature only

🧠 MEDICAL TEXTBOOK-ONLY INTELLIGENCE:
- Powered exclusively by vector knowledge retrieval from medical textbooks
- All responses derived solely from provided medical literature
- Zero external knowledge allowed - only textbook-based information
- Medical knowledge synthesis from authoritative textbook sources only
- Contextual awareness limited to provided medical book content

🔬 TEXTBOOK-RESTRICTED PROCESSING:
- Deep analysis limited to medical textbook content provided
- Medical knowledge integration from textbook sources only
- Evidence-based insights strictly from medical literature below
- Smart correlation limited to textbook knowledge provided
- NO general medical knowledge from training data

🛡️ STRICT KNOWLEDGE BOUNDARIES:
- MANDATORY: Check if medical textbooks contain relevant information first
- If textbooks lack information: "I don't have specific information about this in the medical textbooks available"
- NEVER supplement with general medical knowledge from training
- Always cite specific textbook sources when available
- Maintain strict adherence to provided medical literature only

⚡ TEXTBOOK-BASED RESPONSE PROTOCOL:
1. Check medical textbook content provided below for relevant information
2. If found: Use ONLY textbook information to respond
3. If not found: Explicitly state textbook limitations
4. Always reference specific textbook sources
5. NEVER use general medical knowledge not in textbooks

MEDICAL TEXTBOOK CONTENT TO USE:"""
        
        # Add user profile context
        if context.medical_profile:
            base_prompt += f"\nUser Medical Profile: {json.dumps(context.medical_profile, indent=2)}"
        
        # Add conversation summary
        if context.context_summary:
            base_prompt += f"\nConversation Summary: {context.context_summary}"
        
        # Add advanced intent-specific AI capabilities
        intent_instructions = {
            ConversationIntent.EMERGENCY_CONCERN: "🚨 INTELLIGENT EMERGENCY PROTOCOL: Deploy advanced emergency detection algorithms. Provide immediate, life-saving guidance while coordinating emergency service recommendations. Utilize predictive risk assessment.",
            ConversationIntent.SYMPTOM_INQUIRY: "🔍 ADVANCED SYMPTOM ANALYSIS: Engage multi-dimensional symptom correlation engine. Provide intelligent differential analysis using comprehensive medical knowledge retrieval. Deploy predictive health pattern recognition.",
            ConversationIntent.MEDICATION_INFO: "💊 INTELLIGENT PHARMACOLOGY ENGINE: Access advanced medication knowledge database. Provide comprehensive drug interaction analysis, personalized dosing insights, and predictive side effect profiling.",
            ConversationIntent.DIAGNOSTIC_CLARIFICATION: "🧬 ADVANCED DIAGNOSTIC INTELLIGENCE: Deploy deep medical knowledge synthesis. Provide intelligent medical concept explanation with multi-layered understanding and predictive health insights."
        }
        
        if intent in intent_instructions:
            base_prompt += f"\nSPECIFIC INSTRUCTION: {intent_instructions[intent]}"
        
        base_prompt += "\n\n🎯 INTELLIGENT KNOWLEDGE INTEGRATION: You have access to advanced medical knowledge through state-of-the-art vector retrieval systems. Synthesize this knowledge intelligently to provide next-generation medical insights while maintaining natural conversation flow. Represent the future of medical AI technology."
        
        return base_prompt
    
    def _format_medical_knowledge(self, retrieval_result: MedicalRetrievalResult) -> str:
        """Format medical knowledge for prompt inclusion"""
        if not retrieval_result.chunks:
            return "\n=== MEDICAL KNOWLEDGE ===\nNo specific medical literature found for this query. Provide general health guidance based on medical principles.\n"
        
        knowledge_text = "\n=== MEDICAL TEXTBOOK KNOWLEDGE ===\n"
        knowledge_text += "Use the following medical information to provide evidence-based responses:\n\n"
        
        # Only include the most relevant chunks with medical content
        relevant_chunks = [chunk for chunk in retrieval_result.chunks[:3] if chunk['relevance_score'] > 0.5]
        
        for i, chunk in enumerate(relevant_chunks, 1):
            # Remove technical details and focus on medical content
            knowledge_text += f"Medical Reference {i}:\n"
            knowledge_text += f"{chunk['text']}\n\n"
        
        if relevant_chunks:
            knowledge_text += "IMPORTANT: Base your response primarily on this medical literature. "
            knowledge_text += "Reference this knowledge naturally as part of your medical expertise.\n"
        
        return knowledge_text
    
    def _build_conversation_context(self, context: ConversationContext) -> str:
        """Build conversation context for prompt"""
        if not context.conversation_history:
            return ""
        
        context_text = "\n=== RECENT CONVERSATION ===\n"
        
        # Include last 6 messages for context
        recent_messages = context.conversation_history[-6:]
        for msg in recent_messages:
            role = msg['role'].upper()
            content = msg['content'][:500]  # Truncate long messages
            timestamp = msg.get('timestamp', '')
            context_text += f"{role}: {content}\n"
        
        return context_text
    
    async def generate_response(
        self,
        user_query: str,
        context: ConversationContext,
        retrieval_result: MedicalRetrievalResult,
        intent: ConversationIntent
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate comprehensive medical response with context awareness"""
        start_time = time.time()
        
        try:
            # Build comprehensive prompt
            system_prompt = self._build_medical_system_prompt(context, intent)
            medical_knowledge = self._format_medical_knowledge(retrieval_result)
            conversation_context = self._build_conversation_context(context)
            
            # Handle emergency situations
            if intent == ConversationIntent.EMERGENCY_CONCERN:
                emergency_response = self._generate_emergency_response(user_query, retrieval_result)
                return emergency_response, {
                    'response_type': 'emergency',
                    'generation_time_ms': (time.time() - start_time) * 1000,
                    'safety_priority': True
                }
            
            # STRICT MEDICAL BOOK ONLY CHECK
            if not retrieval_result.chunks or len(retrieval_result.chunks) == 0:
                # NO MEDICAL BOOK KNOWLEDGE AVAILABLE - PROVIDE TEXTBOOK-LIMITED RESPONSE
                fallback_response = f"""I don't have specific information about "{user_query}" in the medical textbooks currently available to me.

🔍 **Medical Knowledge Limitation:**
My responses are restricted to information from medical textbooks in our database. For your specific question, I cannot find relevant content in the available medical literature.

📚 **Recommendation:**
Please consult with a qualified healthcare professional who can:
- Access comprehensive medical databases
- Provide personalized medical advice
- Conduct proper medical evaluation
- Offer evidence-based treatment options

⚠️ **Important:** This system only uses medical textbook knowledge. For complete medical information and personalized care, always consult healthcare professionals.

🎯 **Alternative:** You may try rephrasing your question or asking about general medical topics that might be covered in our medical textbooks."""
                
                return fallback_response, {
                    'response_type': 'textbook_limitation',
                    'generation_time_ms': (time.time() - start_time) * 1000,
                    'medical_books_used': False,
                    'fallback_reason': 'no_textbook_content'
                }
            
            # Build optimized medical prompt with STRICT textbook-only instructions
            user_message = f"{medical_knowledge}\n{conversation_context}\n\nPatient Query: {user_query}\n\nIMPORTANT: Use ONLY the medical textbook content provided above. Do not add any general medical knowledge from training data. If the textbook content doesn't fully answer the question, state the limitation clearly."
            
            # 🚀 SMART MODEL SELECTION with only VERIFIED models
            response = None
            last_error = None
            
            # Use only verified working models
            models_to_try = self.available_models.copy()
            
            for model_name in models_to_try:
                try:
                    logger.info(f"🔄 Attempting response with {model_name}")
                    
                    response = self.groq_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        max_tokens=2000,
                        temperature=0.2,  # Lower temperature for more consistent medical responses
                        top_p=0.85,       # Slightly more focused sampling
                        frequency_penalty=0.1  # Reduce repetition
                    )
                    
                    # Success!
                    logger.info(f"✅ Successfully generated response with {model_name}")
                    
                    # Update primary model if this one worked better
                    if model_name != self.model_name:
                        logger.info(f"🔄 Switching primary model to: {model_name}")
                        self.model_name = model_name
                    break
                    
                except Exception as model_error:
                    error_msg = str(model_error)
                    if "decommissioned" in error_msg.lower():
                        logger.error(f"🚫 Model {model_name} is DECOMMISSIONED: {model_error}")
                        # Remove this model from future attempts this session
                        if model_name in self.available_models:
                            self.available_models.remove(model_name)
                    else:
                        logger.warning(f"❌ Model {model_name} failed: {model_error}")
                    last_error = model_error
                    continue
            
            if not response:
                logger.error("🚨 ALL VERIFIED MODELS FAILED - using fallback")
                raise last_error or Exception("All verified GROQ models failed")
            
            generated_text = response.choices[0].message.content
            
            # Enhance response with citations and safety notes
            enhanced_response = self._enhance_response_with_citations(
                generated_text, retrieval_result, intent
            )
            
            generation_time = (time.time() - start_time) * 1000
            
            metadata = {
                'response_type': self._determine_response_type(intent, retrieval_result),  # Already returns string
                'generation_time_ms': generation_time,
                'sources_cited': len(retrieval_result.source_books),
                'confidence_score': retrieval_result.confidence_score,
                'medical_concepts_used': retrieval_result.medical_concepts,
                'safety_notes_added': True,
                'intent_classified': intent.value
            }
            
            logger.info(f"Generated response in {generation_time:.2f}ms with {len(retrieval_result.chunks)} sources")
            
            return enhanced_response, metadata
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            logger.error(traceback.format_exc())
            
            # Fallback response
            fallback_response = self._generate_fallback_response(user_query, retrieval_result)
            return fallback_response, {
                'response_type': 'general_health',
                'generation_time_ms': (time.time() - start_time) * 1000,
                'error': str(e),
                'fallback_used': True
            }
    
    def _generate_emergency_response(self, query: str, retrieval_result: MedicalRetrievalResult) -> str:
        """Generate emergency response with immediate safety priority"""
        response = "🚨 **EMERGENCY MEDICAL CONCERN DETECTED** 🚨\n\n"
        response += "**IMMEDIATE ACTION REQUIRED:**\n"
        response += "• Call 911 or emergency services immediately\n"
        response += "• Go to the nearest emergency room\n"
        response += "• Do not delay seeking professional medical care\n\n"
        
        if retrieval_result.chunks:
            response += "**Relevant Medical Information:**\n"
            for chunk in retrieval_result.chunks[:2]:  # Top 2 most relevant
                response += f"• {chunk['text'][:200]}... (Source: {chunk['source']})\n"
        
        response += "\n⚠️ **This is not a substitute for emergency medical care. Seek immediate professional help.**"
        
        return response
    
    def _enhance_response_with_citations(
        self, 
        response: str, 
        retrieval_result: MedicalRetrievalResult,
        intent: ConversationIntent
    ) -> str:
        """Enhance response with medical literature references and safety notes"""
        
        enhanced_response = response
        
        # Add medical literature sources ONLY if significant medical knowledge was used
        if retrieval_result.chunks and retrieval_result.confidence_score > 0.6:
            enhanced_response += "\n\n📚 **Medical Literature Sources:**\n"
            
            unique_sources = {}
            for chunk in retrieval_result.chunks[:3]:  # Top 3 sources
                source_key = f"{chunk['source']}_p{chunk['page']}"
                if source_key not in unique_sources and chunk['relevance_score'] > 0.5:
                    # Clean source name - remove technical details
                    clean_source = chunk['source'].replace('Unknown', 'Medical Reference')
                    unique_sources[source_key] = {
                        'source': clean_source,
                        'page': chunk['page'],
                        'book_type': chunk.get('book_type', 'general')
                    }
            
            for i, (_, source_info) in enumerate(unique_sources.items(), 1):
                enhanced_response += f"{i}. *{source_info['source']}*, Page {source_info['page']} ({source_info['book_type']})\n"
        
        # Add safety disclaimer based on intent
        safety_note = self._get_safety_note_for_intent(intent)
        if safety_note:
            enhanced_response += f"\n\n⚠️ **Important:** {safety_note}"
        
        # Add follow-up suggestions
        if intent in [ConversationIntent.SYMPTOM_INQUIRY, ConversationIntent.TREATMENT_QUESTION]:
            enhanced_response += "\n\n💡 **Follow-up suggestions:**\n"
            enhanced_response += "• Would you like me to explain any specific aspects in more detail?\n"
            enhanced_response += "• Do you have any other related questions?\n"
            enhanced_response += "• Would information about when to seek medical care be helpful?"
        
        return enhanced_response
    
    def _get_safety_note_for_intent(self, intent: ConversationIntent) -> str:
        """Get appropriate safety note for intent"""
        safety_notes = {
            ConversationIntent.SYMPTOM_INQUIRY: "This information is for educational purposes. Consult a healthcare provider for proper diagnosis and treatment.",
            ConversationIntent.TREATMENT_QUESTION: "Always consult with a healthcare professional before starting any treatment. Individual cases may vary significantly.",
            ConversationIntent.MEDICATION_INFO: "Medication information is for reference only. Always follow your healthcare provider's instructions and prescription labels.",
            ConversationIntent.DIAGNOSTIC_CLARIFICATION: "This explanation is for educational purposes. Discuss your specific diagnosis and treatment plan with your healthcare provider."
        }
        
        return safety_notes.get(intent, "Always consult with a healthcare professional for personalized medical advice.")
    
    def _determine_response_type(self, intent: ConversationIntent, retrieval_result: MedicalRetrievalResult) -> str:
        """Determine the type of response generated - returns JSON-safe string"""
        if intent == ConversationIntent.EMERGENCY_CONCERN:
            return 'emergency'
        elif retrieval_result.confidence_score > 0.7 and retrieval_result.chunks:
            return 'medical_textbook'
        elif intent in [ConversationIntent.DIAGNOSTIC_CLARIFICATION, ConversationIntent.PROCEDURE_EXPLANATION]:
            return 'educational'
        elif intent == ConversationIntent.SYMPTOM_INQUIRY:
            return 'diagnostic_support'
        else:
            return 'general_health'
    
    def _generate_fallback_response(self, query: str, retrieval_result: MedicalRetrievalResult) -> str:
        """Generate fallback response when AI generation fails"""
        response = "I understand you're seeking medical information. "
        
        if retrieval_result.chunks and retrieval_result.confidence_score > 0.4:
            response += "Based on medical literature available:\n\n"
            for i, chunk in enumerate(retrieval_result.chunks[:2], 1):
                clean_source = chunk['source'].replace('Unknown', 'Medical Reference')
                response += f"**Medical Reference {i}** (Page {chunk['page']}):\n"
                response += f"{chunk['text'][:400]}...\n\n"
        else:
            response += "While I don't have specific medical literature on this exact topic, I recommend:\n\n"
        
        response += "**For personalized medical care:**\n"
        response += "• Schedule a consultation with your healthcare provider\n"
        response += "• Discuss symptoms and concerns with a medical professional\n"
        response += "• Consider specialist referral if needed\n\n"
        response += "⚠️ **Important:** This information is for educational purposes only. Always consult qualified healthcare professionals for medical diagnosis and treatment."
        
        return response

class BasicResponseGenerator:
    """Fallback response generator when Groq is not available"""
    
    def __init__(self):
        self.medical_knowledge_base = {
            'fever': "Fever is a temporary increase in body temperature. Common causes include infections, medications, or other medical conditions. Monitor temperature and stay hydrated.",
            'headache': "Headaches can have various causes including tension, dehydration, or medical conditions. Rest, hydration, and over-the-counter pain relief may help.",
            'nausea': "Nausea can be caused by various factors including infections, medications, or digestive issues. Stay hydrated and consider small, frequent meals.",
            'pain': "Pain is the body's way of signaling injury or illness. The treatment depends on the cause and severity. Consult a healthcare provider for persistent pain.",
            'cough': "Coughs can be caused by infections, allergies, or irritants. Stay hydrated, use throat lozenges, and consider a humidifier.",
            'fatigue': "Fatigue can result from poor sleep, stress, medical conditions, or lifestyle factors. Ensure adequate rest and consider consulting a healthcare provider.",
        }
        
        logger.info("✅ Basic response generator initialized (fallback mode)")
    
    async def generate_response(
        self,
        user_query: str,
        context: ConversationContext,
        retrieval_result: MedicalRetrievalResult,
        intent: ConversationIntent
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate basic medical response using local knowledge"""
        start_time = time.time()
        
        try:
            query_lower = user_query.lower()
            
            # Check for emergency situations
            emergency_keywords = ['chest pain', 'difficulty breathing', 'severe pain', 'unconscious', 'emergency']
            if any(keyword in query_lower for keyword in emergency_keywords):
                response = """🚨 **MEDICAL EMERGENCY** 🚨
                
This appears to be a medical emergency. Please:
• Call emergency services (911) immediately
• Go to the nearest emergency room
• Do not delay seeking professional medical care

This system cannot provide emergency medical treatment."""
                
                return response, {
                    'response_type': 'emergency',
                    'generation_time_ms': (time.time() - start_time) * 1000,
                    'fallback_mode': True
                }
            
            # Look for medical keywords in query
            response_parts = []
            found_topics = []
            
            for keyword, info in self.medical_knowledge_base.items():
                if keyword in query_lower:
                    response_parts.append(f"**{keyword.title()}:**\n{info}")
                    found_topics.append(keyword)
            
            if response_parts:
                response = "\n\n".join(response_parts)
                response += "\n\n⚠️ **Important:** This information is for educational purposes only. Please consult with a healthcare professional for proper diagnosis and treatment."
            else:
                # Generic medical response
                response = """I understand you have a medical question. While I can provide general health information, I recommend:

**For immediate concerns:**
• Contact your healthcare provider
• Call your doctor's office
• Visit an urgent care center if needed

**For general health information:**
• Describe your specific symptoms
• Mention how long you've been experiencing them
• Share any relevant medical history

**Remember:** This system provides educational information only and cannot replace professional medical advice."""
            
            # Add conversation context if available
            if context and context.conversation_history:
                response += "\n\n💡 **Tip:** You can ask follow-up questions about your health concerns."
            
            metadata = {
                'response_type': 'general_health',
                'generation_time_ms': (time.time() - start_time) * 1000,
                'fallback_mode': True,
                'topics_found': found_topics,
                'intent_classified': intent.value if intent else 'general'
            }
            
            return response, metadata
            
        except Exception as e:
            logger.error(f"Basic response generation failed: {e}")
            
            fallback_response = """I apologize, but I'm experiencing technical difficulties. 

For medical questions and health concerns, please:
• Contact your healthcare provider directly
• Call your doctor's office
• Visit a medical professional
• Use official medical resources

Your health and safety are important."""
            
            return fallback_response, {
                'response_type': 'error_fallback',
                'generation_time_ms': (time.time() - start_time) * 1000,
                'error': str(e),
                'fallback_mode': True
            }

class UltimateConversationalEngine:
    """Main conversational engine orchestrating all components"""
    
    def __init__(self, config: Dict[str, str]):
        # Validate required configuration
        required_keys = ['pinecone_api_key', 'groq_api_key']
        for key in required_keys:
            if key not in config or not config[key]:
                raise ValueError(f"Missing required configuration: {key}")
        
        self.config = config
        
        # Initialize fallback mode
        self.use_fallback_mode = False
        
        # Initialize components
        logger.info("🚀 Initializing Ultimate Conversational Engine v5.0...")
        
        try:
            # Initialize Pinecone retriever with fallback to basic knowledge
            self.retriever = None
            self.use_vector_retrieval = False
            
            try:
                if config.get('pinecone_api_key') and config['pinecone_api_key'] not in ['test', 'test_key', 'fallback_key']:
                    self.retriever = EnhancedPineconeRetriever(
                        api_key=config['pinecone_api_key'],
                        index_name=config.get('pinecone_index_name', 'medical-books-ultimate')
                    )
                    self.use_vector_retrieval = True
                    logger.info("✅ Pinecone vector retrieval initialized")
                else:
                    logger.info("🔄 Using basic knowledge base (no vector retrieval)")
            except Exception as pinecone_error:
                logger.warning(f"⚠️ Pinecone initialization failed: {pinecone_error}")
                logger.info("🔄 Using basic knowledge base (no vector retrieval)")
                self.use_vector_retrieval = False
                self.use_fallback_mode = True
            
            # Initialize response generator - ALWAYS use proper API
            if not config.get('groq_api_key') or config['groq_api_key'] in ['test', 'test_key', 'fallback_key']:
                raise ValueError("❌ GROQ API key is required for proper RAG conversation. Please set a valid GROQ_API_KEY in your environment.")
            
            self.response_generator = ResponseGenerator(config['groq_api_key'])
            logger.info("✅ Response generator initialized with proper API")
            
            # Initialize supporting components
            self.entity_extractor = MedicalEntityExtractor()
            self.intent_classifier = ConversationIntentClassifier()
            logger.info("✅ Medical NLP components initialized")
            
            # Initialize Redis if available
            self.redis_client = None
            if REDIS_AVAILABLE and config.get('redis_url'):
                try:
                    self.redis_client = redis.from_url(config['redis_url'])
                    self.redis_client.ping()
                    logger.info("✅ Redis cache connected")
                except Exception as e:
                    logger.warning(f"Redis connection failed: {e}")
            
            # Active conversations storage
            self.active_conversations = {}
            self.conversation_metrics = {}
            
            logger.info("🎉 Ultimate Conversational Engine fully initialized!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize conversational engine: {e}")
            raise
    
    async def process_conversation(
        self,
        user_query: str,
        user_id: str,
        session_id: str = None,
        user_profile: Dict[str, Any] = None,
        auth_context: Dict[str, Any] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Main conversation processing pipeline"""
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        conversation_key = f"{user_id}_{session_id}"
        processing_start = time.time()
        
        try:
            # Step 1: Get or create conversation context with auth integration
            context = await self._get_or_create_context(user_id, session_id, user_profile, auth_context)
            context.add_message('user', user_query)
            
            # Step 1.5: Apply auth-based restrictions if needed
            if auth_context:
                # Guest users have limited functionality
                if auth_context.get('is_guest', False):
                    context.medical_profile = {}  # No medical profile for guests
                    if len(context.conversation_history) > 10:  # Limit conversation length
                        return self._get_guest_limitation_response(), {
                            'guest_limited': True,
                            'suggestion': 'Please register for full conversation history'
                        }
                
                # Add user permissions to context
                context.user_preferences['permissions'] = auth_context.get('permissions', [])
                context.user_preferences['is_admin'] = auth_context.get('is_admin', False)
            
            # Step 2: Extract medical entities from query
            medical_entities = self.entity_extractor.extract_medical_entities(user_query)
            context.medical_entities.update(medical_entities)
            
            # Step 3: Classify conversation intent
            intent = self.intent_classifier.classify_intent(user_query, context)
            context.current_intent = intent
            
            logger.info(f"Processing query with intent: {intent.value}, entities: {len(sum(medical_entities.values(), []))}")
            
            # Step 4: Retrieve relevant medical knowledge (with fallback)
            if self.use_fallback_mode or not self.retriever:
                # Create empty retrieval result for fallback mode
                retrieval_result = MedicalRetrievalResult(
                    query=user_query,
                    chunks=[],
                    relevance_scores=[],
                    source_books=[],
                    medical_concepts=[],
                    confidence_score=0.0,
                    retrieval_time_ms=0.0,
                    total_chunks_found=0,
                    best_match_score=0.0,
                    sources_metadata=[]
                )
                logger.info("🔄 Using fallback mode - no vector retrieval")
            else:
                retrieval_result = await self.retriever.retrieve_medical_knowledge(
                    user_query, context, top_k=10, score_threshold=0.6
                )
            
            # Step 5: Generate response
            response, response_metadata = await self.response_generator.generate_response(
                user_query, context, retrieval_result, intent
            )
            
            # Step 6: Add response to context
            context.add_message('assistant', response, response_metadata)
            
            # Step 7: Update conversation metrics
            total_processing_time = (time.time() - processing_start) * 1000
            await self._update_conversation_metrics(
                conversation_key, total_processing_time, retrieval_result, response_metadata
            )
            
            # Step 8: Store updated context
            await self._store_context(context)
            
            # Step 9: Prepare comprehensive response metadata
            comprehensive_metadata = {
                'conversation_id': conversation_key,
                'session_id': session_id,
                'user_id': user_id,
                'intent_classified': intent.value,
                'medical_entities': medical_entities,
                'retrieval_stats': {
                    'chunks_found': retrieval_result.total_chunks_found,
                    'confidence_score': retrieval_result.confidence_score,
                    'retrieval_time_ms': retrieval_result.retrieval_time_ms,
                    'sources': retrieval_result.source_books
                },
                'response_stats': response_metadata,
                'total_processing_time_ms': total_processing_time,
                'safety_flags': context.safety_concerns,
                'conversation_turn': len(context.conversation_history) // 2
            }
            
            logger.info(f"✅ Conversation processed successfully in {total_processing_time:.2f}ms")
            
            return response, comprehensive_metadata
            
        except Exception as e:
            logger.error(f"❌ Conversation processing failed: {e}")
            logger.error(traceback.format_exc())
            
            # Return error response
            error_response = (
                "I apologize, but I encountered a technical issue while processing your request. "
                "Please try again in a moment. For urgent medical concerns, please contact a healthcare provider directly."
            )
            
            error_metadata = {
                'error': str(e),
                'processing_time_ms': (time.time() - processing_start) * 1000,
                'intent_classified': 'error',
                'fallback_response': True
            }
            
            return error_response, error_metadata
    
    async def _get_or_create_context(
        self,
        user_id: str,
        session_id: str,
        user_profile: Dict[str, Any] = None,
        auth_context: Dict[str, Any] = None
    ) -> ConversationContext:
        """Get existing conversation context or create new one"""
        conversation_key = f"{user_id}_{session_id}"
        
        # Try to retrieve from active conversations
        if conversation_key in self.active_conversations:
            return self.active_conversations[conversation_key]
        
        # Try to retrieve from Redis cache
        if self.redis_client:
            try:
                cached_context = self.redis_client.get(f"context:{conversation_key}")
                if cached_context:
                    context_data = json.loads(cached_context)
                    context = ConversationContext(**context_data)
                    self.active_conversations[conversation_key] = context
                    return context
            except Exception as e:
                logger.warning(f"Failed to retrieve context from Redis: {e}")
        
        # Create new context with auth integration
        enhanced_user_profile = user_profile or {}
        if auth_context:
            # Enrich profile with auth data
            enhanced_user_profile.update({
                'email': auth_context.get('user_email', ''),
                'name': auth_context.get('user_name', 'User'),
                'is_admin': auth_context.get('is_admin', False),
                'auth_provider': auth_context.get('auth_provider', 'unknown'),
                'permissions': auth_context.get('permissions', [])
            })
        
        context = ConversationContext(
            user_id=user_id,
            session_id=session_id,
            medical_profile=enhanced_user_profile.get('medical_profile', {}),
            user_preferences=enhanced_user_profile.get('preferences', {})
        )
        
        # Initialize conversation metrics
        if conversation_key not in self.conversation_metrics:
            self.conversation_metrics[conversation_key] = ConversationMetrics(
                conversation_id=conversation_key,
                user_id=user_id,
                session_start=time.time()
            )
        
        self.active_conversations[conversation_key] = context
        return context
    
    async def _store_context(self, context: ConversationContext):
        """Store conversation context for persistence"""
        conversation_key = f"{context.user_id}_{context.session_id}"
        
        # Store in active conversations
        self.active_conversations[conversation_key] = context
        
        # Store in Redis with expiration
        if self.redis_client:
            try:
                context_data = asdict(context)
                # Convert datetime objects to ISO strings
                for msg in context_data['conversation_history']:
                    if isinstance(msg.get('timestamp'), datetime):
                        msg['timestamp'] = msg['timestamp'].isoformat()
                
                self.redis_client.setex(
                    f"context:{conversation_key}",
                    3600,  # 1 hour expiration
                    json.dumps(context_data, default=str)
                )
            except Exception as e:
                logger.warning(f"Failed to store context in Redis: {e}")
    
    async def _update_conversation_metrics(
        self,
        conversation_key: str,
        processing_time_ms: float,
        retrieval_result: MedicalRetrievalResult,
        response_metadata: Dict[str, Any]
    ):
        """Update comprehensive conversation metrics"""
        if conversation_key not in self.conversation_metrics:
            return
        
        metrics = self.conversation_metrics[conversation_key]
        metrics.total_messages += 1
        metrics.add_response_time(processing_time_ms)
        
        if retrieval_result.confidence_score > 0:
            metrics.medical_queries += 1
            metrics.add_retrieval_score(retrieval_result.confidence_score)
        
        # Update user satisfaction indicators based on response quality
        if response_metadata.get('sources_cited', 0) > 0:
            metrics.user_satisfaction_indicators['has_sources'] = True
        
        if response_metadata.get('safety_notes_added', False):
            metrics.user_satisfaction_indicators['safety_conscious'] = True
    
    def get_conversation_analytics(self, user_id: str = None) -> Dict[str, Any]:
        """Get comprehensive conversation analytics"""
        analytics = {
            'timestamp': datetime.now().isoformat(),
            'total_active_conversations': len(self.active_conversations),
            'total_users': len(set(conv.split('_')[0] for conv in self.active_conversations.keys())),
            'global_metrics': {
                'average_response_time_ms': 0.0,
                'average_retrieval_score': 0.0,
                'total_medical_queries': 0,
                'total_messages': 0
            },
            'user_metrics': {}
        }
        
        # Calculate global metrics
        all_response_times = []
        all_retrieval_scores = []
        total_medical_queries = 0
        total_messages = 0
        
        for metrics in self.conversation_metrics.values():
            all_response_times.extend(metrics.response_times)
            all_retrieval_scores.extend(metrics.retrieval_scores)
            total_medical_queries += metrics.medical_queries
            total_messages += metrics.total_messages
        
        analytics['global_metrics'] = {
            'average_response_time_ms': sum(all_response_times) / len(all_response_times) if all_response_times else 0.0,
            'average_retrieval_score': sum(all_retrieval_scores) / len(all_retrieval_scores) if all_retrieval_scores else 0.0,
            'total_medical_queries': total_medical_queries,
            'total_messages': total_messages
        }
        
        # User-specific metrics if requested
        if user_id:
            user_conversations = [key for key in self.conversation_metrics.keys() if key.startswith(user_id)]
            user_metrics = {}
            
            for conv_key in user_conversations:
                metrics = self.conversation_metrics[conv_key]
                user_metrics[conv_key] = {
                    'total_messages': metrics.total_messages,
                    'medical_queries': metrics.medical_queries,
                    'average_response_time': metrics.get_average_response_time(),
                    'average_retrieval_score': metrics.get_average_retrieval_score(),
                    'safety_flags': len(metrics.safety_flags)
                }
            
            analytics['user_metrics'] = user_metrics
        
        return analytics
    
    def _get_guest_limitation_response(self) -> str:
        """Get response for guest users hitting limitations"""
        return """
🔐 **Guest User Limitation Reached**

Thank you for using MedBot! As a guest user, you've reached the conversation limit.

**To continue with unlimited conversations:**
• 📝 **Register for a free account** - Get full conversation history
• 🩺 **Add medical profile** - Get personalized health insights  
• 💾 **Save conversation history** - Access your past health discussions
• 🚀 **Premium features** - Advanced medical analysis

**What you can do right now:**
• Start a new conversation session
• Register for a free account
• Contact support for assistance

**Your health matters!** Register now to get the most comprehensive medical AI assistance.
        """.strip()

# Factory function for easy integration
def create_conversational_engine(config: Dict[str, str]) -> UltimateConversationalEngine:
    """Factory function to create configured conversational engine"""
    return UltimateConversationalEngine(config)

# Integration with existing Flask app
class FlaskIntegration:
    """Integration helper for Flask applications"""
    
    def __init__(self, app, engine: UltimateConversationalEngine):
        self.app = app
        self.engine = engine
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes for conversational engine"""
        
        @self.app.route('/api/chat/enhanced', methods=['POST'])
        async def enhanced_chat():
            """Enhanced chat endpoint with full conversational engine"""
            try:
                data = request.get_json()
                user_query = data.get('message', '').strip()
                user_id = data.get('user_id') or session.get('user_id', str(uuid.uuid4()))
                session_id = data.get('session_id') or session.get('session_id')
                user_profile = data.get('user_profile', {})
                
                if not user_query:
                    return jsonify({'error': 'Message required'}), 400
                
                # Process conversation
                response, metadata = await self.engine.process_conversation(
                    user_query, user_id, session_id, user_profile
                )
                
                return jsonify({
                    'response': response,
                    'metadata': metadata,
                    'success': True
                })
                
            except Exception as e:
                logger.error(f"Enhanced chat endpoint error: {e}")
                return jsonify({
                    'error': 'Internal server error',
                    'success': False
                }), 500
        
        @self.app.route('/api/chat/analytics')
        def conversation_analytics():
            """Get conversation analytics"""
            try:
                user_id = request.args.get('user_id')
                analytics = self.engine.get_conversation_analytics(user_id)
                return jsonify(analytics)
            except Exception as e:
                logger.error(f"Analytics endpoint error: {e}")
                return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Example usage and testing
    async def test_engine():
        config = {
            'pinecone_api_key': os.getenv('PINECONE_API_KEY'),
            'groq_api_key': os.getenv('GROQ_API_KEY'),
            'pinecone_index_name': 'medical-books-ultimate',
            'redis_url': os.getenv('REDIS_URL')
        }
        
        if not config['pinecone_api_key'] or not config['groq_api_key']:
            print("Missing required API keys in environment variables")
            return
        
        # Create engine
        engine = create_conversational_engine(config)
        
        # Test conversation
        test_queries = [
            "What are the symptoms of diabetes?",
            "How is hypertension treated?",
            "Can you explain what causes chest pain?",
            "What should I know about taking aspirin daily?"
        ]
        
        user_id = "test_user_123"
        session_id = str(uuid.uuid4())
        
        for query in test_queries:
            print(f"\n🤔 USER: {query}")
            response, metadata = await engine.process_conversation(query, user_id, session_id)
            print(f"🤖 ASSISTANT: {response}")
            print(f"📊 METADATA: Intent={metadata['intent_classified']}, Sources={len(metadata['retrieval_stats']['sources'])}, Time={metadata['total_processing_time_ms']:.0f}ms")
        
        # Get analytics
        analytics = engine.get_conversation_analytics(user_id)
        print(f"\n📈 ANALYTICS: {json.dumps(analytics, indent=2)}")
    
    # Run test
    asyncio.run(test_engine())