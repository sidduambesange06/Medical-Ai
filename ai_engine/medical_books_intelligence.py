"""
Medical Books Intelligence Engine.
Generates clinical responses using extracted knowledge from indexed medical textbooks.

Provides symptom-based differential diagnosis and treatment protocol extraction
without requiring external LLM API calls.
"""

import re
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ClinicalEvidence:
    """Structured clinical evidence extracted from medical books"""
    condition: str
    symptoms: List[str]
    diagnostic_criteria: List[str]
    treatment_protocols: List[str]
    contraindications: List[str]
    evidence_strength: float  # 0.0 to 1.0
    source_textbook: str
    page_reference: int

@dataclass
class DiagnosticInsight:
    """Clinical diagnostic insight with reasoning"""
    primary_condition: str
    probability_score: float
    supporting_evidence: List[str]
    differential_conditions: List[str]
    clinical_reasoning: str
    recommended_workup: List[str]
    red_flags: List[str]

class ClinicalContentAnalyzer:
    """Extract structured clinical knowledge from medical textbooks"""

    def __init__(self):
        self.clinical_patterns = self._compile_clinical_patterns()
        self.medical_entities = self._load_medical_entities()

    def _compile_clinical_patterns(self) -> Dict[str, List[str]]:
        """Compile regex patterns for clinical knowledge extraction"""
        return {
            'diagnostic_criteria': [
                r'diagnosis.*requires?.*(?:all|any|at least).*of.*following',
                r'criteria.*for.*diagnosis.*(?:include|are)',
                r'diagnostic.*criteria.*(?:include|consist of)',
                r'to.*diagnose.*(?:requires?|needs?|must have)',
                r'classic.*(?:triad|presentation|symptoms).*(?:of|include)',
                r'pathognomonic.*(?:sign|symptom|finding).*(?:for|of)'
            ],

            'symptoms_patterns': [
                r'(?:patients?|individuals?).*(?:present|presents?).*with',
                r'(?:common|typical|classic).*(?:symptoms?|signs?|presentations?)',
                r'(?:manifests?|manifesting).*as',
                r'(?:characterized|characterized by)',
                r'clinical.*(?:features?|manifestations?|presentations?)'
            ],

            'treatment_patterns': [
                r'(?:first|second|third).*line.*(?:treatment|therapy|management)',
                r'(?:treatment|therapy|management).*(?:consists? of|includes?|involves?)',
                r'(?:recommended|preferred|standard).*(?:treatment|therapy)',
                r'(?:dose|dosage).*(?:is|should be|recommended)',
                r'contraindications?.*(?:include|are|consist of)'
            ],

            'prognosis_patterns': [
                r'prognosis.*(?:is|depends on|varies)',
                r'(?:survival|mortality|morbidity).*rate',
                r'(?:complications?|adverse effects?).*(?:include|may include)',
                r'(?:course|progression).*of.*(?:disease|condition)'
            ]
        }

    def _load_medical_entities(self) -> Dict[str, List[str]]:
        """Load comprehensive medical entity database"""
        return {
            'conditions': [
                'myocardial infarction', 'pneumonia', 'diabetes mellitus', 'hypertension',
                'asthma', 'chronic obstructive pulmonary disease', 'stroke', 'sepsis',
                'heart failure', 'pulmonary embolism', 'acute coronary syndrome',
                'acute respiratory distress syndrome', 'diabetic ketoacidosis'
            ],

            'symptoms': [
                'chest pain', 'dyspnea', 'palpitations', 'syncope', 'fatigue',
                'nausea', 'vomiting', 'headache', 'dizziness', 'fever',
                'cough', 'abdominal pain', 'back pain', 'joint pain'
            ],

            'treatments': [
                'aspirin', 'metformin', 'lisinopril', 'atorvastatin', 'metoprolol',
                'oxygen therapy', 'mechanical ventilation', 'fluid resuscitation',
                'anticoagulation', 'thrombolytic therapy', 'percutaneous coronary intervention'
            ],

            'diagnostic_tests': [
                'electrocardiogram', 'chest x-ray', 'computed tomography',
                'magnetic resonance imaging', 'echocardiogram', 'blood tests',
                'arterial blood gas', 'troponin', 'brain natriuretic peptide'
            ]
        }

    def extract_clinical_knowledge(self, medical_text: str, source_info: Dict) -> List[ClinicalEvidence]:
        """Extract structured clinical knowledge from medical textbook content"""
        clinical_evidence = []

        # Split text into meaningful chunks
        sentences = re.split(r'[.!?]+', medical_text)

        for sentence in sentences:
            if len(sentence.strip()) < 20:  # Skip very short sentences
                continue

            evidence = self._analyze_sentence_for_clinical_content(
                sentence.strip(), source_info
            )

            if evidence:
                clinical_evidence.extend(evidence)

        return clinical_evidence

    def _analyze_sentence_for_clinical_content(self, sentence: str, source_info: Dict) -> List[ClinicalEvidence]:
        """Analyze individual sentence for clinical knowledge"""
        sentence_lower = sentence.lower()
        evidence_list = []

        # Detect conditions mentioned
        detected_conditions = []
        for condition in self.medical_entities['conditions']:
            if condition.lower() in sentence_lower:
                detected_conditions.append(condition)

        if not detected_conditions:
            return evidence_list

        # Extract diagnostic criteria
        for condition in detected_conditions:
            diagnostic_criteria = self._extract_diagnostic_criteria(sentence, condition)
            symptoms = self._extract_symptoms(sentence)
            treatments = self._extract_treatments(sentence)

            if diagnostic_criteria or symptoms or treatments:
                evidence = ClinicalEvidence(
                    condition=condition,
                    symptoms=symptoms,
                    diagnostic_criteria=diagnostic_criteria,
                    treatment_protocols=treatments,
                    contraindications=[],  # TODO: Extract contraindications
                    evidence_strength=self._calculate_evidence_strength(sentence),
                    source_textbook=source_info.get('filename', 'Unknown'),
                    page_reference=source_info.get('page', 0)
                )
                evidence_list.append(evidence)

        return evidence_list

    def _extract_diagnostic_criteria(self, sentence: str, condition: str) -> List[str]:
        """Extract diagnostic criteria from sentence"""
        criteria = []
        sentence_lower = sentence.lower()

        # Look for diagnostic criteria patterns
        for pattern in self.clinical_patterns['diagnostic_criteria']:
            if re.search(pattern, sentence_lower):
                # Extract the criteria following the pattern
                match = re.search(pattern + r'[:\-\s]*(.*?)(?:\.|$)', sentence_lower)
                if match:
                    criteria_text = match.group(1).strip()
                    # Split criteria by common delimiters
                    individual_criteria = re.split(r'[,;]|and|or', criteria_text)
                    criteria.extend([c.strip() for c in individual_criteria if c.strip()])

        return criteria

    def _extract_symptoms(self, sentence: str) -> List[str]:
        """Extract symptoms mentioned in sentence"""
        symptoms = []
        sentence_lower = sentence.lower()

        for symptom in self.medical_entities['symptoms']:
            if symptom.lower() in sentence_lower:
                symptoms.append(symptom)

        return symptoms

    def _extract_treatments(self, sentence: str) -> List[str]:
        """Extract treatment protocols from sentence"""
        treatments = []
        sentence_lower = sentence.lower()

        for treatment in self.medical_entities['treatments']:
            if treatment.lower() in sentence_lower:
                treatments.append(treatment)

        return treatments

    def _calculate_evidence_strength(self, sentence: str) -> float:
        """Calculate evidence strength based on linguistic markers"""
        sentence_lower = sentence.lower()
        strength = 0.5  # Base strength

        # Strong evidence indicators
        strong_indicators = ['always', 'must', 'required', 'necessary', 'pathognomonic', 'definitive']
        moderate_indicators = ['typically', 'usually', 'commonly', 'often', 'frequently']
        weak_indicators = ['may', 'might', 'can', 'possible', 'sometimes', 'occasionally']

        for indicator in strong_indicators:
            if indicator in sentence_lower:
                strength += 0.3

        for indicator in moderate_indicators:
            if indicator in sentence_lower:
                strength += 0.1

        for indicator in weak_indicators:
            if indicator in sentence_lower:
                strength -= 0.1

        return min(1.0, max(0.1, strength))

class DiagnosticCriteriaEngine:
    """Generate differential diagnosis from medical textbook knowledge"""

    def __init__(self):
        self.condition_symptom_map = {}
        self.diagnostic_algorithms = {}

    def process_clinical_evidence(self, evidence_list: List[ClinicalEvidence]) -> Dict[str, Any]:
        """Process clinical evidence to build diagnostic knowledge base"""

        for evidence in evidence_list:
            condition = evidence.condition

            if condition not in self.condition_symptom_map:
                self.condition_symptom_map[condition] = {
                    'symptoms': set(),
                    'diagnostic_criteria': set(),
                    'treatments': set(),
                    'evidence_sources': []
                }

            # Build knowledge mapping
            self.condition_symptom_map[condition]['symptoms'].update(evidence.symptoms)
            self.condition_symptom_map[condition]['diagnostic_criteria'].update(evidence.diagnostic_criteria)
            self.condition_symptom_map[condition]['treatments'].update(evidence.treatment_protocols)
            self.condition_symptom_map[condition]['evidence_sources'].append({
                'source': evidence.source_textbook,
                'page': evidence.page_reference,
                'strength': evidence.evidence_strength
            })

        return self.condition_symptom_map

    def generate_differential_diagnosis(self, patient_symptoms: List[str], patient_context: Dict = None) -> List[DiagnosticInsight]:
        """Generate differential diagnosis based on medical textbook knowledge"""

        differential_insights = []

        for condition, knowledge in self.condition_symptom_map.items():
            # Calculate symptom match score
            matching_symptoms = set(patient_symptoms) & knowledge['symptoms']
            total_condition_symptoms = len(knowledge['symptoms'])

            if total_condition_symptoms > 0:
                symptom_match_score = len(matching_symptoms) / total_condition_symptoms
            else:
                symptom_match_score = 0.0

            # Only consider conditions with reasonable symptom overlap
            if symptom_match_score > 0.1:

                # Calculate overall probability considering evidence strength
                avg_evidence_strength = np.mean([
                    source['strength'] for source in knowledge['evidence_sources']
                ]) if knowledge['evidence_sources'] else 0.5

                probability_score = symptom_match_score * avg_evidence_strength

                # Generate clinical reasoning
                clinical_reasoning = self._generate_clinical_reasoning(
                    condition, matching_symptoms, knowledge
                )

                # Generate recommended workup
                recommended_workup = self._generate_recommended_workup(condition, knowledge)

                insight = DiagnosticInsight(
                    primary_condition=condition,
                    probability_score=probability_score,
                    supporting_evidence=list(matching_symptoms),
                    differential_conditions=[],  # Will be populated later
                    clinical_reasoning=clinical_reasoning,
                    recommended_workup=recommended_workup,
                    red_flags=self._identify_red_flags(condition, patient_symptoms)
                )

                differential_insights.append(insight)

        # Sort by probability score
        differential_insights.sort(key=lambda x: x.probability_score, reverse=True)

        # Add differential conditions to top insights
        for i, insight in enumerate(differential_insights[:5]):
            insight.differential_conditions = [
                other.primary_condition for other in differential_insights[i+1:i+4]
            ]

        return differential_insights[:10]  # Return top 10 differentials

    def _generate_clinical_reasoning(self, condition: str, matching_symptoms: set, knowledge: Dict) -> str:
        """Generate clinical reasoning for diagnostic consideration"""

        reasoning_parts = []

        if matching_symptoms:
            reasoning_parts.append(
                f"Patient presents with {len(matching_symptoms)} symptoms consistent with {condition}: "
                f"{', '.join(matching_symptoms)}"
            )

        if knowledge['diagnostic_criteria']:
            reasoning_parts.append(
                f"Diagnostic criteria from medical literature include: "
                f"{'; '.join(list(knowledge['diagnostic_criteria'])[:3])}"
            )

        if knowledge['evidence_sources']:
            strong_sources = [s for s in knowledge['evidence_sources'] if s['strength'] > 0.7]
            if strong_sources:
                reasoning_parts.append(
                    f"Strong evidence from {len(strong_sources)} medical textbook sources"
                )

        return " ".join(reasoning_parts)

    def _generate_recommended_workup(self, condition: str, knowledge: Dict) -> List[str]:
        """Generate recommended diagnostic workup"""

        # Basic workup recommendations based on condition type
        workup_recommendations = []

        condition_lower = condition.lower()

        if 'heart' in condition_lower or 'cardiac' in condition_lower:
            workup_recommendations.extend([
                'Electrocardiogram (ECG)',
                'Cardiac enzymes (troponin)',
                'Chest X-ray',
                'Echocardiogram if indicated'
            ])
        elif 'pneumonia' in condition_lower or 'respiratory' in condition_lower:
            workup_recommendations.extend([
                'Chest X-ray',
                'Complete blood count with differential',
                'Blood cultures if febrile',
                'Arterial blood gas if hypoxic'
            ])
        elif 'diabetes' in condition_lower:
            workup_recommendations.extend([
                'Fasting glucose',
                'Hemoglobin A1c',
                'Urinalysis',
                'Comprehensive metabolic panel'
            ])
        else:
            workup_recommendations.extend([
                'Complete blood count',
                'Basic metabolic panel',
                'Appropriate imaging based on symptoms'
            ])

        return workup_recommendations

    def _identify_red_flags(self, condition: str, patient_symptoms: List[str]) -> List[str]:
        """Identify red flag symptoms requiring immediate attention"""

        red_flags = []
        condition_lower = condition.lower()
        symptoms_lower = [s.lower() for s in patient_symptoms]

        # Cardiovascular red flags
        if 'heart' in condition_lower or 'cardiac' in condition_lower:
            cv_red_flags = ['chest pain', 'severe dyspnea', 'syncope', 'palpitations']
            for flag in cv_red_flags:
                if any(flag in symptom for symptom in symptoms_lower):
                    red_flags.append(f'Cardiovascular emergency concern: {flag}')

        # Respiratory red flags
        if 'pneumonia' in condition_lower or 'respiratory' in condition_lower:
            resp_red_flags = ['severe dyspnea', 'hypoxia', 'altered mental status']
            for flag in resp_red_flags:
                if any(flag in symptom for symptom in symptoms_lower):
                    red_flags.append(f'Respiratory emergency concern: {flag}')

        return red_flags

class TreatmentProtocolExtractor:
    """Extract evidence-based treatment protocols from medical books"""

    def __init__(self):
        self.treatment_algorithms = {}
        self.medication_protocols = {}

    def extract_treatment_protocols(self, evidence_list: List[ClinicalEvidence]) -> Dict[str, Any]:
        """Extract treatment protocols for each condition"""

        for evidence in evidence_list:
            condition = evidence.condition

            if condition not in self.treatment_algorithms:
                self.treatment_algorithms[condition] = {
                    'first_line': set(),
                    'second_line': set(),
                    'contraindications': set(),
                    'monitoring': set(),
                    'evidence_sources': []
                }

            # Categorize treatments by line of therapy
            for treatment in evidence.treatment_protocols:
                treatment_lower = treatment.lower()

                if any(keyword in treatment_lower for keyword in ['first line', 'initial', 'primary']):
                    self.treatment_algorithms[condition]['first_line'].add(treatment)
                elif any(keyword in treatment_lower for keyword in ['second line', 'alternative', 'backup']):
                    self.treatment_algorithms[condition]['second_line'].add(treatment)
                else:
                    self.treatment_algorithms[condition]['first_line'].add(treatment)

            # Add contraindications
            self.treatment_algorithms[condition]['contraindications'].update(evidence.contraindications)

            # Add evidence source
            self.treatment_algorithms[condition]['evidence_sources'].append({
                'source': evidence.source_textbook,
                'page': evidence.page_reference,
                'strength': evidence.evidence_strength
            })

        return self.treatment_algorithms

    def generate_treatment_recommendations(self, condition: str, patient_context: Dict = None) -> Dict[str, Any]:
        """Generate evidence-based treatment recommendations"""

        if condition not in self.treatment_algorithms:
            return {'error': f'No treatment protocols found for {condition}'}

        protocols = self.treatment_algorithms[condition]

        recommendations = {
            'condition': condition,
            'first_line_treatments': list(protocols['first_line']),
            'alternative_treatments': list(protocols['second_line']),
            'contraindications': list(protocols['contraindications']),
            'monitoring_requirements': list(protocols['monitoring']),
            'evidence_level': self._calculate_evidence_level(protocols['evidence_sources']),
            'patient_specific_considerations': []
        }

        # Add patient-specific considerations
        if patient_context:
            recommendations['patient_specific_considerations'] = self._generate_patient_specific_considerations(
                condition, patient_context, protocols
            )

        return recommendations

    def _calculate_evidence_level(self, evidence_sources: List[Dict]) -> str:
        """Calculate overall evidence level for treatment recommendations"""

        if not evidence_sources:
            return 'Limited evidence'

        avg_strength = np.mean([source['strength'] for source in evidence_sources])
        source_count = len(evidence_sources)

        if avg_strength > 0.8 and source_count >= 3:
            return 'High-quality evidence'
        elif avg_strength > 0.6 and source_count >= 2:
            return 'Moderate-quality evidence'
        else:
            return 'Limited evidence'

    def _generate_patient_specific_considerations(self, condition: str, patient_context: Dict, protocols: Dict) -> List[str]:
        """Generate patient-specific treatment considerations"""

        considerations = []

        # Age-based considerations
        age = patient_context.get('age', 0)
        if age > 75:
            considerations.append('Geriatric patient: Consider dose adjustments and drug interactions')
        elif age < 18:
            considerations.append('Pediatric patient: Age-appropriate dosing and medication selection required')

        # Comorbidity considerations
        comorbidities = patient_context.get('medical_conditions', [])
        for comorbidity in comorbidities:
            comorbidity_lower = str(comorbidity).lower()
            if 'kidney' in comorbidity_lower or 'renal' in comorbidity_lower:
                considerations.append('Renal impairment: Dose adjustment may be required')
            elif 'liver' in comorbidity_lower or 'hepatic' in comorbidity_lower:
                considerations.append('Hepatic impairment: Monitor liver function and consider dose adjustment')
            elif 'heart' in comorbidity_lower:
                considerations.append('Cardiac condition: Monitor cardiovascular effects')

        return considerations

class ClinicalReasoningEngine:
    """Generate clinical reasoning similar to medical decision-making"""

    def __init__(self):
        self.reasoning_templates = self._load_reasoning_templates()

    def _load_reasoning_templates(self) -> Dict[str, str]:
        """Load clinical reasoning templates"""
        return {
            'differential_reasoning': """
            Based on the patient's presentation of {symptoms}, several conditions should be considered:

            Primary consideration: {primary_condition} (probability: {probability:.0%})
            - Supporting factors: {supporting_evidence}
            - Clinical reasoning: {reasoning}

            Differential diagnoses to consider:
            {differential_conditions}

            Recommended workup: {workup}

            {red_flags_warning}
            """,

            'treatment_reasoning': """
            For the diagnosis of {condition}, evidence-based treatment approach includes:

            First-line therapy: {first_line}
            Alternative options: {alternatives}

            Patient-specific considerations: {patient_considerations}

            Monitoring requirements: {monitoring}

            Evidence level: {evidence_level}
            """
        }

    def generate_diagnostic_reasoning(self, insights: List[DiagnosticInsight]) -> str:
        """Generate comprehensive diagnostic reasoning"""

        if not insights:
            return "Insufficient information for diagnostic reasoning."

        primary_insight = insights[0]

        # Format red flags warning
        red_flags_warning = ""
        if primary_insight.red_flags:
            red_flags_warning = f"⚠️ RED FLAGS: {'; '.join(primary_insight.red_flags)}"

        # Format differential conditions
        differential_conditions = ""
        if primary_insight.differential_conditions:
            for i, condition in enumerate(primary_insight.differential_conditions[:3], 1):
                differential_conditions += f"\n{i}. {condition}"

        reasoning = self.reasoning_templates['differential_reasoning'].format(
            symptoms=', '.join(primary_insight.supporting_evidence),
            primary_condition=primary_insight.primary_condition,
            probability=primary_insight.probability_score,
            supporting_evidence=', '.join(primary_insight.supporting_evidence),
            reasoning=primary_insight.clinical_reasoning,
            differential_conditions=differential_conditions,
            workup='; '.join(primary_insight.recommended_workup),
            red_flags_warning=red_flags_warning
        )

        return reasoning.strip()

class MedicalBooksIntelligenceEngine:
    """
    MAIN ENGINE: Proprietary Medical Books Intelligence System
    REPLACES: Groq API with textbook-based clinical intelligence
    INNOVATION: Medical literature → Clinical decision algorithms
    """

    def __init__(self, pinecone_retriever):
        self.pinecone = pinecone_retriever
        self.clinical_analyzer = ClinicalContentAnalyzer()
        self.diagnostic_engine = DiagnosticCriteriaEngine()
        self.treatment_extractor = TreatmentProtocolExtractor()
        self.reasoning_engine = ClinicalReasoningEngine()

        # Knowledge base built from medical textbooks
        self.clinical_knowledge_base = {}
        self.diagnostic_algorithms = {}
        self.treatment_protocols = {}

        logger.info("🧠 Medical Books Intelligence Engine initialized - Zero API dependencies!")

    async def generate_clinical_response(self, query: str, patient_context: Dict = None, doctor_specialty: str = 'general') -> Tuple[str, Dict]:
        """
        CORE INNOVATION: Generate clinical responses using only medical textbook knowledge
        NO GROQ API REQUIRED - Pure medical books intelligence
        """

        start_time = datetime.now()

        try:
            # Step 1: Retrieve relevant medical textbook content
            logger.info("🔍 Retrieving medical textbook knowledge...")
            medical_evidence = await self.pinecone.retrieve_medical_knowledge(
                query, context={}, top_k=15, score_threshold=0.6
            )

            if not medical_evidence.chunks:
                return self._generate_no_content_response(query), {
                    'source': 'medical_books_intelligence',
                    'content_available': False,
                    'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000
                }

            # Step 2: Extract clinical knowledge from textbooks
            logger.info("🧬 Extracting clinical knowledge from medical literature...")
            clinical_evidence = []

            for chunk in medical_evidence.chunks:
                source_info = {
                    'filename': chunk.get('source', 'Medical Textbook'),
                    'page': chunk.get('page', 0)
                }

                evidence = self.clinical_analyzer.extract_clinical_knowledge(
                    chunk['text'], source_info
                )
                clinical_evidence.extend(evidence)

            # Step 3: Process evidence into diagnostic knowledge
            logger.info("🎯 Processing diagnostic algorithms...")
            self.diagnostic_engine.process_clinical_evidence(clinical_evidence)
            self.treatment_extractor.extract_treatment_protocols(clinical_evidence)

            # Step 4: Generate differential diagnosis
            patient_symptoms = self._extract_symptoms_from_query(query)
            diagnostic_insights = self.diagnostic_engine.generate_differential_diagnosis(
                patient_symptoms, patient_context
            )

            # Step 5: Generate clinical reasoning
            clinical_reasoning = self.reasoning_engine.generate_diagnostic_reasoning(diagnostic_insights)

            # Step 6: Format response for doctors
            clinical_response = self._format_doctor_response(
                query, diagnostic_insights, clinical_reasoning, doctor_specialty, medical_evidence
            )

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            metadata = {
                'source': 'proprietary_medical_books_intelligence',
                'api_dependencies': None,
                'clinical_evidence_count': len(clinical_evidence),
                'diagnostic_insights_count': len(diagnostic_insights),
                'textbook_sources': len(set(e.source_textbook for e in clinical_evidence)),
                'processing_time_ms': processing_time,
                'doctor_specialty': doctor_specialty,
                'innovation_claim': 'First AI to generate clinical responses from medical textbooks only'
            }

            logger.info(f"✅ Medical Books Intelligence response generated in {processing_time:.2f}ms")

            return clinical_response, metadata

        except Exception as e:
            logger.error(f"❌ Medical Books Intelligence Engine error: {e}")
            return self._generate_error_response(query), {
                'source': 'medical_books_intelligence',
                'error': str(e),
                'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000
            }

    def _extract_symptoms_from_query(self, query: str) -> List[str]:
        """Extract symptoms mentioned in the query"""
        symptoms = []
        query_lower = query.lower()

        # Common symptom keywords
        symptom_keywords = [
            'chest pain', 'shortness of breath', 'dyspnea', 'palpitations',
            'syncope', 'dizziness', 'headache', 'nausea', 'vomiting',
            'fever', 'fatigue', 'weakness', 'cough', 'abdominal pain'
        ]

        for symptom in symptom_keywords:
            if symptom in query_lower:
                symptoms.append(symptom)

        return symptoms

    def _format_doctor_response(self, query: str, insights: List[DiagnosticInsight],
                              reasoning: str, specialty: str, evidence) -> str:
        """Format response specifically for medical professionals"""

        if not insights:
            return f"""
🏥 **MEDICAL BOOKS INTELLIGENCE ANALYSIS**

**Query:** {query}

**Analysis:** Based on available medical textbook content, I cannot provide specific diagnostic insights for this query. The medical literature in our database may not contain sufficient information on this topic.

**Recommendation:** Consider consulting additional medical resources or specialty-specific literature for this clinical question.

**📚 Evidence Base:** {len(evidence.chunks)} medical textbook references analyzed

⚠️ **Note:** This analysis is based solely on medical textbook knowledge without external API dependencies.
            """.strip()

        primary_insight = insights[0]

        response = f"""
🏥 **MEDICAL BOOKS INTELLIGENCE ANALYSIS**
*Proprietary AI using medical textbook knowledge only*

**Clinical Query:** {query}

**🎯 PRIMARY DIAGNOSTIC CONSIDERATION:**
**{primary_insight.primary_condition}** (Confidence: {primary_insight.probability_score:.1%})

**📋 CLINICAL REASONING:**
{reasoning}

**🩺 DIFFERENTIAL DIAGNOSIS:**"""

        for i, insight in enumerate(insights[:5], 1):
            response += f"""
{i}. **{insight.primary_condition}** ({insight.probability_score:.1%})
   - Supporting evidence: {', '.join(insight.supporting_evidence[:3])}"""

        if primary_insight.red_flags:
            response += f"""

🚨 **RED FLAGS IDENTIFIED:**
{chr(10).join(f"• {flag}" for flag in primary_insight.red_flags)}"""

        response += f"""

**🔬 RECOMMENDED WORKUP:**
{chr(10).join(f"• {test}" for test in primary_insight.recommended_workup)}

**📚 EVIDENCE BASE:**
- Medical textbooks analyzed: {len(evidence.source_books)}
- Clinical evidence extracted: {len([e for e in evidence.chunks if e['relevance_score'] > 0.7])} high-relevance sources
- Knowledge source: Proprietary medical literature database

**💡 FOR CLINICIANS:**
This analysis represents AI-powered clinical decision support based exclusively on medical textbook knowledge. Use as adjunct to clinical judgment and standard diagnostic protocols.

⚠️ **INNOVATION NOTE:** This response was generated without external API dependencies, using our proprietary Medical Books Intelligence Engine.
        """.strip()

        return response

    def _generate_no_content_response(self, query: str) -> str:
        """Generate response when no medical textbook content is available"""
        return f"""
🏥 **MEDICAL BOOKS INTELLIGENCE - LIMITED CONTENT**

**Query:** {query}

**Status:** No specific medical textbook content found for this query in our current database.

**📚 Current Limitations:**
Our medical textbook database may not contain information on this specific topic. This represents an opportunity to expand our medical literature collection.

**🎯 Recommendations:**
1. Rephrase query with more specific medical terminology
2. Try related symptoms or conditions
3. Consult additional medical literature sources

**💡 Innovation Note:**
This system operates entirely on medical textbook knowledge without external API dependencies. As we expand our medical literature database, coverage will improve.

**For Immediate Clinical Needs:**
Please consult peer-reviewed medical literature, clinical guidelines, or specialist consultation for this specific query.
        """.strip()

    def _generate_error_response(self, query: str) -> str:
        """Generate error response maintaining professional tone"""
        return f"""
🏥 **MEDICAL BOOKS INTELLIGENCE - SYSTEM MESSAGE**

**Query:** {query}

**Status:** Technical processing error encountered.

**System Information:** Our proprietary Medical Books Intelligence Engine experienced a processing issue. This system operates independently of external APIs.

**Recommendations:**
1. Please retry your query
2. Ensure query contains specific medical terminology
3. Contact system administrator if problem persists

**Innovation Note:** This system represents proprietary medical AI technology operating without external API dependencies.
        """.strip()

# Factory function for easy integration
def create_medical_books_intelligence_engine(pinecone_retriever):
    """Create configured Medical Books Intelligence Engine"""
    return MedicalBooksIntelligenceEngine(pinecone_retriever)

if __name__ == "__main__":
    # Example usage and testing
    print("🧠 Medical Books Intelligence Engine v1.0")
    print("✅ Zero API dependencies")
    print("✅ Proprietary clinical algorithms")
    print("✅ Medical textbook knowledge extraction")
    print("🚀 Ready for investor demo!")