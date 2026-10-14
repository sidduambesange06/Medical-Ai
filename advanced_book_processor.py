"""
Medical Book Processor.
End-to-end book upload and Pinecone indexing pipeline.

Handles PDF extraction, chunking, embedding generation, and vector storage
with real-time progress monitoring.
"""

import os
import sys
import time
import json
import hashlib
import asyncio
import threading
import logging
import traceback
import subprocess
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import weakref
import gc

# Core dependencies
import numpy as np
import pymupdf as fitz
from PIL import Image
import io

# AI/ML Libraries
try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# LangChain for text processing (using modern imports)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# OCR Libraries
try:
    import pytesseract
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Medical NLP
try:
    import spacy
    import scispacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Performance monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('advanced_book_processor.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class ProcessingStage(Enum):
    """Book processing stages for progress tracking"""
    VALIDATION = "validation"
    EXTRACTION = "extraction"
    CHUNKING = "chunking"
    EMBEDDINGS = "embeddings"
    VECTORIZATION = "vectorization"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"

class BookType(Enum):
    """Medical book categories for specialized processing"""
    ANATOMY = "anatomy"
    PHYSIOLOGY = "physiology"
    PATHOLOGY = "pathology"
    PHARMACOLOGY = "pharmacology"
    SURGERY = "surgery"
    INTERNAL_MEDICINE = "internal_medicine"
    PEDIATRICS = "pediatrics"
    PSYCHIATRY = "psychiatry"
    RADIOLOGY = "radiology"
    EMERGENCY = "emergency"
    GENERAL = "general"

class ContentType(Enum):
    """Types of content for specialized processing"""
    TEXT = "text"
    IMAGES = "images"
    TABLES = "tables"
    DIAGRAMS = "diagrams"
    MIXED = "mixed"

@dataclass
class BookProcessingConfig:
    """Configuration for book processing"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    quality_threshold: float = 0.7
    use_ocr: bool = True
    use_gpu: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 100
    max_workers: int = 4
    retry_attempts: int = 3

@dataclass
class ProcessingMetrics:
    """Comprehensive processing metrics"""
    book_id: str
    filename: str
    file_size: int
    book_type: BookType
    processing_stage: ProcessingStage
    start_time: float
    current_step_time: float
    progress_percent: float = 0.0
    
    # Content metrics
    total_pages: int = 0
    processed_pages: int = 0
    extracted_text_length: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
    embedded_chunks: int = 0
    indexed_chunks: int = 0
    
    # Quality metrics
    content_quality_score: float = 0.0
    medical_terminology_score: float = 0.0
    readability_score: float = 0.0
    
    # Performance metrics
    extraction_time: float = 0.0
    chunking_time: float = 0.0
    embedding_time: float = 0.0
    indexing_time: float = 0.0
    total_processing_time: float = 0.0
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Success/failure rates
    success_rate: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization"""
        return asdict(self)
    
    def update_progress(self, stage: ProcessingStage, percent: float, message: str = ""):
        """Update processing progress"""
        self.processing_stage = stage
        self.progress_percent = percent
        self.current_step_time = time.time()
        if message:
            logger.info(f"{self.filename}: {message}")

class MedicalTerminologyExtractor:
    """Extract and analyze medical terminology"""
    
    def __init__(self):
        self.medical_terms = self._load_medical_terms()
        self.drug_names = self._load_drug_names()
        self.condition_patterns = self._compile_condition_patterns()
        
    def _load_medical_terms(self) -> set:
        """Load comprehensive medical terminology"""
        # Basic medical terms - in production, load from comprehensive database
        terms = {
            'diagnosis', 'treatment', 'therapy', 'medication', 'dosage', 'symptom',
            'syndrome', 'pathology', 'etiology', 'prognosis', 'clinical', 'patient',
            'disease', 'condition', 'disorder', 'infection', 'inflammation',
            'acute', 'chronic', 'malignant', 'benign', 'carcinoma', 'sarcoma',
            'hypertension', 'diabetes', 'pneumonia', 'myocardial', 'infarction',
            'fracture', 'hemorrhage', 'ischemia', 'necrosis', 'metastasis',
            'antibiotics', 'analgesics', 'antihypertensive', 'insulin', 'morphine',
            'surgery', 'surgical', 'operative', 'anesthesia', 'incision', 'suture'
        }
        return terms
    
    def _load_drug_names(self) -> set:
        """Load drug names database"""
        drugs = {
            'aspirin', 'ibuprofen', 'acetaminophen', 'morphine', 'insulin',
            'metformin', 'lisinopril', 'atorvastatin', 'omeprazole', 'warfarin',
            'metoprolol', 'amlodipine', 'prednisone', 'azithromycin', 'cephalexin'
        }
        return drugs
    
    def _compile_condition_patterns(self) -> List:
        """Compile regex patterns for medical conditions"""
        import re
        patterns = [
            r'\b\w*itis\b',  # Inflammatory conditions
            r'\b\w*osis\b',  # Pathological conditions
            r'\b\w*emia\b',  # Blood conditions
            r'\b\w*uria\b',  # Urinary conditions
            r'\b\w*pathy\b', # Disease conditions
            r'\b\w*trophy\b' # Growth conditions
        ]
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for medical content"""
        text_lower = text.lower()
        words = text.split()
        
        # Count medical terms
        medical_term_count = sum(1 for word in words if word.lower() in self.medical_terms)
        drug_count = sum(1 for word in words if word.lower() in self.drug_names)
        
        # Find pattern matches
        condition_matches = []
        for pattern in self.condition_patterns:
            matches = pattern.findall(text)
            condition_matches.extend(matches)
        
        # Calculate medical relevance score
        total_words = len(words)
        if total_words == 0:
            medical_score = 0.0
        else:
            medical_score = (medical_term_count + drug_count + len(condition_matches)) / total_words * 100
        
        return {
            'medical_terms_found': medical_term_count,
            'drugs_found': drug_count,
            'medical_conditions': len(condition_matches),
            'medical_relevance_score': min(medical_score, 100.0),
            'condition_examples': condition_matches[:5],
            'is_medical_content': medical_score > 5.0
        }

class BookTypeClassifier:
    """Intelligent book type classification"""
    
    TYPE_KEYWORDS = {
        BookType.ANATOMY: [
            'anatomy', 'anatomical', 'structure', 'morphology', 'histology',
            'gray', 'netter', 'bone', 'muscle', 'organ', 'tissue', 'cell'
        ],
        BookType.PHYSIOLOGY: [
            'physiology', 'physiological', 'function', 'mechanism', 'process',
            'guyton', 'homeostasis', 'metabolism', 'circulation', 'respiration'
        ],
        BookType.PATHOLOGY: [
            'pathology', 'pathological', 'disease', 'pathogenesis', 'etiology',
            'robbins', 'pathophysiology', 'lesion', 'neoplasm', 'cancer'
        ],
        BookType.PHARMACOLOGY: [
            'pharmacology', 'drug', 'medication', 'pharmaceutical', 'dosage',
            'goodman', 'gilman', 'therapeutic', 'pharmacokinetics', 'toxicity'
        ],
        BookType.SURGERY: [
            'surgery', 'surgical', 'operative', 'procedure', 'technique',
            'schwartz', 'operation', 'suture', 'anesthesia', 'trauma'
        ],
        BookType.INTERNAL_MEDICINE: [
            'internal', 'medicine', 'clinical', 'diagnosis', 'treatment',
            'harrison', 'cecil', 'cardiology', 'gastroenterology', 'nephrology'
        ],
        BookType.PEDIATRICS: [
            'pediatric', 'children', 'child', 'infant', 'neonatal',
            'nelson', 'developmental', 'growth', 'vaccination', 'adolescent'
        ],
        BookType.PSYCHIATRY: [
            'psychiatry', 'psychiatric', 'mental', 'psychological', 'behavior',
            'dsm', 'depression', 'anxiety', 'schizophrenia', 'therapy'
        ],
        BookType.RADIOLOGY: [
            'radiology', 'imaging', 'xray', 'ct', 'mri', 'ultrasound',
            'radiographic', 'scan', 'diagnostic', 'contrast'
        ],
        BookType.EMERGENCY: [
            'emergency', 'critical', 'trauma', 'urgent', 'acute',
            'tintinalli', 'resuscitation', 'shock', 'cardiac arrest'
        ]
    }
    
    @classmethod
    def classify(cls, filename: str, content_sample: str = "", 
                 page_count: int = 0) -> Tuple[BookType, float]:
        """Classify book type with confidence score"""
        combined_text = f"{filename} {content_sample}".lower()
        
        scores = {}
        for book_type, keywords in cls.TYPE_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in combined_text:
                    # Weight keywords found in filename higher
                    weight = 2 if keyword in filename.lower() else 1
                    score += weight
            
            # Normalize score
            if score > 0:
                scores[book_type] = score / len(keywords)
        
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])
            return best_type[0], best_type[1]
        
        return BookType.GENERAL, 0.0

class AdvancedTextExtractor:
    """Multi-engine text extraction with OCR fallback"""
    
    def __init__(self, config: BookProcessingConfig):
        self.config = config
        self.ocr_reader = None
        if OCR_AVAILABLE and config.use_ocr:
            try:
                self.ocr_reader = easyocr.Reader(['en'])
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {e}")
                self.ocr_reader = None
    
    async def extract_from_pdf(self, pdf_path: Path, metrics: ProcessingMetrics) -> List[Dict[str, Any]]:
        """Extract text and metadata from PDF with multiple methods"""
        logger.info(f"Starting text extraction for {pdf_path.name}")
        start_time = time.time()
        
        pages_data = []
        
        try:
            # Primary extraction with PyMuPDF
            doc = fitz.open(str(pdf_path))
            metrics.total_pages = len(doc)
            
            for page_num in range(len(doc)):
                if page_num % 10 == 0:
                    progress = (page_num / len(doc)) * 100
                    metrics.update_progress(
                        ProcessingStage.EXTRACTION, 
                        progress * 0.4,  # Extraction is 40% of total
                        f"Extracting page {page_num + 1}/{len(doc)}"
                    )
                
                page = doc[page_num]
                page_data = {
                    'page_number': page_num + 1,
                    'text': '',
                    'images': [],
                    'tables': [],
                    'metadata': {}
                }
                
                # Extract text using multiple methods
                text_methods = []
                
                # Method 1: Standard text extraction
                try:
                    text1 = page.get_text()
                    text_methods.append(('standard', text1, len(text1)))
                except Exception as e:
                    logger.warning(f"Standard extraction failed for page {page_num + 1}: {e}")
                
                # Method 2: Layout preservation
                try:
                    text2 = page.get_text("dict")
                    extracted_text2 = self._extract_from_dict(text2)
                    text_methods.append(('layout', extracted_text2, len(extracted_text2)))
                except Exception as e:
                    logger.warning(f"Layout extraction failed for page {page_num + 1}: {e}")
                
                # Method 3: OCR for images (if available and needed)
                if self.config.use_ocr and self.ocr_reader:
                    try:
                        # Check if page has images or is image-heavy
                        image_list = page.get_images()
                        if image_list or (text_methods and max(t[2] for t in text_methods) < 100):
                            ocr_text = await self._ocr_extract_page(page)
                            if ocr_text:
                                text_methods.append(('ocr', ocr_text, len(ocr_text)))
                    except Exception as e:
                        logger.warning(f"OCR extraction failed for page {page_num + 1}: {e}")
                
                # Select best extraction method
                if text_methods:
                    best_method = max(text_methods, key=lambda x: x[2])
                    page_data['text'] = best_method[1]
                    page_data['extraction_method'] = best_method[0]
                    page_data['text_length'] = best_method[2]
                
                # Extract images and metadata
                page_data['images'] = self._extract_images(page)
                page_data['metadata'] = self._extract_page_metadata(page)
                
                pages_data.append(page_data)
                metrics.processed_pages += 1
                metrics.extracted_text_length += len(page_data['text'])
            
            doc.close()
            
            extraction_time = time.time() - start_time
            metrics.extraction_time = extraction_time
            
            logger.info(f"Text extraction completed: {len(pages_data)} pages, "
                       f"{metrics.extracted_text_length:,} characters in {extraction_time:.2f}s")
            
            return pages_data
            
        except Exception as e:
            logger.error(f"Text extraction failed for {pdf_path.name}: {e}")
            metrics.errors.append(f"Text extraction failed: {str(e)}")
            raise
    
    def _extract_from_dict(self, page_dict: Dict) -> str:
        """Extract text from PyMuPDF dictionary format"""
        text_parts = []
        
        if 'blocks' in page_dict:
            for block in page_dict['blocks']:
                if 'lines' in block:
                    for line in block['lines']:
                        if 'spans' in line:
                            line_text = ''.join(span.get('text', '') for span in line['spans'])
                            if line_text.strip():
                                text_parts.append(line_text)
        
        return '\n'.join(text_parts)
    
    async def _ocr_extract_page(self, page) -> str:
        """Extract text using OCR"""
        try:
            # Convert page to image
            mat = fitz.Matrix(2, 2)  # High resolution
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Run OCR
            result = self.ocr_reader.readtext(img_data, detail=0)
            ocr_text = ' '.join(result)
            
            return ocr_text
        except Exception as e:
            logger.warning(f"OCR processing failed: {e}")
            return ""
    
    def _extract_images(self, page) -> List[Dict]:
        """Extract image information from page"""
        images = []
        try:
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                xref = img[0]
                image_info = {
                    'index': img_index,
                    'xref': xref,
                    'width': img[2],
                    'height': img[3],
                    'colorspace': img[4] if len(img) > 4 else None,
                    'size_estimate': img[2] * img[3] * 3  # Rough size estimate
                }
                images.append(image_info)
        except Exception as e:
            logger.warning(f"Image extraction failed: {e}")
        
        return images
    
    def _extract_page_metadata(self, page) -> Dict:
        """Extract metadata from page"""
        try:
            rect = page.rect
            return {
                'width': rect.width,
                'height': rect.height,
                'rotation': page.rotation,
                'has_links': len(page.get_links()) > 0,
                'has_annotations': len(page.annots()) > 0
            }
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            return {}

class IntelligentChunker:
    """Advanced chunking with medical context awareness"""
    
    def __init__(self, config: BookProcessingConfig):
        self.config = config
        self.medical_extractor = MedicalTerminologyExtractor()
        
        # Initialize text splitter
        if LANGCHAIN_AVAILABLE:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
            )
        else:
            self.splitter = None
    
    async def create_chunks(self, pages_data: List[Dict], book_metadata: Dict, 
                          metrics: ProcessingMetrics) -> List[Document]:
        """Create intelligent chunks from extracted text"""
        logger.info("Starting intelligent chunking process")
        start_time = time.time()
        
        chunks = []
        total_text = ""
        
        # Combine all text with page markers
        for page_data in pages_data:
            page_text = page_data['text']
            if page_text.strip():
                page_marker = f"\n[PAGE {page_data['page_number']}]\n"
                total_text += page_marker + page_text + "\n"
        
        if not total_text.strip():
            raise ValueError("No text content found for chunking")
        
        try:
            # Method 1: Use LangChain splitter if available
            if self.splitter:
                raw_chunks = self.splitter.split_text(total_text)
            else:
                # Fallback chunking method
                raw_chunks = self._fallback_chunking(total_text)
            
            logger.info(f"Created {len(raw_chunks)} raw chunks")
            
            # Process chunks with medical context awareness
            processed_chunks = []
            for i, chunk_text in enumerate(raw_chunks):
                if i % 20 == 0:
                    progress = 40 + (i / len(raw_chunks)) * 30  # Chunking is 30% after extraction
                    metrics.update_progress(
                        ProcessingStage.CHUNKING,
                        progress,
                        f"Processing chunk {i + 1}/{len(raw_chunks)}"
                    )
                
                # Skip very small chunks
                if len(chunk_text.strip()) < self.config.min_chunk_size:
                    continue
                
                # Analyze medical content
                medical_analysis = self.medical_extractor.analyze_text(chunk_text)
                
                # Extract page number from chunk
                page_num = self._extract_page_number(chunk_text)
                
                # Create enhanced metadata
                chunk_metadata = {
                    'filename': book_metadata.get('filename', 'unknown'),
                    'book_type': book_metadata.get('book_type', 'general'),
                    'page': page_num,
                    'chunk_index': len(processed_chunks),
                    'chunk_size': len(chunk_text),
                    'word_count': len(chunk_text.split()),
                    'medical_score': medical_analysis['medical_relevance_score'],
                    'medical_terms': medical_analysis['medical_terms_found'],
                    'drugs_found': medical_analysis['drugs_found'],
                    'medical_conditions': medical_analysis['medical_conditions'],
                    'is_medical_content': medical_analysis['is_medical_content'],
                    'processing_timestamp': datetime.now().isoformat(),
                    'extraction_method': 'intelligent_chunking',
                    'quality_indicators': {
                        'has_medical_terms': medical_analysis['medical_terms_found'] > 0,
                        'has_drugs': medical_analysis['drugs_found'] > 0,
                        'has_conditions': medical_analysis['medical_conditions'] > 0,
                        'sufficient_length': len(chunk_text) >= self.config.min_chunk_size
                    }
                }
                
                # Create Document object
                if LANGCHAIN_AVAILABLE:
                    chunk_doc = Document(
                        page_content=chunk_text.strip(),
                        metadata=chunk_metadata
                    )
                else:
                    # Fallback Document creation
                    chunk_doc = type('Document', (), {
                        'page_content': chunk_text.strip(),
                        'metadata': chunk_metadata
                    })()
                
                processed_chunks.append(chunk_doc)
                metrics.processed_chunks += 1
            
            # Filter high-quality chunks
            quality_chunks = [
                chunk for chunk in processed_chunks
                if chunk.metadata['medical_score'] > 1.0 or
                   chunk.metadata['word_count'] > 50
            ]
            
            chunking_time = time.time() - start_time
            metrics.chunking_time = chunking_time
            metrics.total_chunks = len(quality_chunks)
            
            logger.info(f"Chunking completed: {len(quality_chunks)} quality chunks "
                       f"from {len(raw_chunks)} raw chunks in {chunking_time:.2f}s")
            
            return quality_chunks
            
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            metrics.errors.append(f"Chunking failed: {str(e)}")
            raise
    
    def _fallback_chunking(self, text: str) -> List[str]:
        """Fallback chunking method when LangChain is not available"""
        chunks = []
        current_chunk = ""
        sentences = text.split('. ')
        
        for sentence in sentences:
            sentence = sentence.strip() + '. '
            
            if len(current_chunk) + len(sentence) <= self.config.chunk_size:
                current_chunk += sentence
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _extract_page_number(self, text: str) -> int:
        """Extract page number from chunk text"""
        import re
        page_match = re.search(r'\[PAGE (\d+)\]', text)
        return int(page_match.group(1)) if page_match else 0

class EmbeddingEngine:
    """GPU-accelerated embedding generation"""
    
    def __init__(self, config: BookProcessingConfig):
        self.config = config
        self.model = None
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def initialize(self) -> int:
        """Initialize embedding model"""
        logger.info(f"Initializing embedding model: {self.config.embedding_model}")
        
        try:
            if EMBEDDINGS_AVAILABLE:
                device = DEVICE if self.config.use_gpu else "cpu"
                self.model = SentenceTransformer(
                    self.config.embedding_model,
                    device=device
                )
                
                # Test embedding to get dimension
                test_embedding = self.model.encode("medical test")
                dimension = len(test_embedding)
                
                logger.info(f"Embedding model loaded on {device}, dimension: {dimension}")
                return dimension
            else:
                raise ImportError("Sentence transformers not available")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    async def generate_embeddings(self, chunks: List[Document], 
                                metrics: ProcessingMetrics) -> List[Tuple[Document, List[float]]]:
        """Generate embeddings for chunks with GPU acceleration"""
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        start_time = time.time()
        
        embedded_chunks = []
        
        try:
            # Batch processing for efficiency
            batch_size = self.config.batch_size
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_texts = [chunk.page_content for chunk in batch]
                
                # Progress update
                progress = 70 + (i / len(chunks)) * 20  # Embeddings are 20% of total
                metrics.update_progress(
                    ProcessingStage.EMBEDDINGS,
                    progress,
                    f"Generating embeddings batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}"
                )
                
                # Generate embeddings
                batch_embeddings = await self._generate_batch_embeddings(batch_texts)
                
                # Pair chunks with embeddings
                for chunk, embedding in zip(batch, batch_embeddings):
                    embedded_chunks.append((chunk, embedding))
                    metrics.embedded_chunks += 1
            
            embedding_time = time.time() - start_time
            metrics.embedding_time = embedding_time
            
            logger.info(f"Embedding generation completed: {len(embedded_chunks)} embeddings "
                       f"in {embedding_time:.2f}s")
            
            return embedded_chunks
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            metrics.errors.append(f"Embedding generation failed: {str(e)}")
            raise
    
    async def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        try:
            if self.model:
                embeddings = self.model.encode(texts, convert_to_numpy=True)
                return [emb.tolist() for emb in embeddings]
            else:
                # Fallback: return zero vectors
                return [[0.0] * 384] * len(texts)  # MiniLM dimension
                
        except Exception as e:
            logger.warning(f"Batch embedding generation failed: {e}")
            return [[0.0] * 384] * len(texts)

class PineconeManager:
    """Advanced Pinecone vector database management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get('api_key')
        self.index_name = config.get('index_name', 'medical-books-ultimate')
        self.environment = config.get('environment', 'us-east-1')
        self.pc = None
        self.index = None
        
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone library not available")
        
        if not self.api_key:
            raise ValueError("Pinecone API key not provided")
    
    async def initialize(self, embedding_dimension: int) -> bool:
        """Initialize Pinecone index"""
        logger.info("Initializing Pinecone connection")
        
        try:
            self.pc = Pinecone(api_key=self.api_key)
            
            # Check existing indexes
            existing_indexes = self.pc.list_indexes().names()
            
            if self.index_name in existing_indexes:
                logger.info(f"Using existing index: {self.index_name}")
            else:
                # Create new index
                logger.info(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=embedding_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.environment
                    )
                )
                
                # Wait for index to be ready
                await asyncio.sleep(30)
            
            self.index = self.pc.Index(self.index_name)
            
            # Test connection
            stats = self.index.describe_index_stats()
            logger.info(f"Pinecone index ready. Current vectors: {stats.total_vector_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pinecone initialization failed: {e}")
            raise
    
    async def upsert_vectors(self, embedded_chunks: List[Tuple[Document, List[float]]], 
                           book_id: str, metrics: ProcessingMetrics) -> bool:
        """Upsert vectors to Pinecone with retry logic"""
        logger.info(f"Upserting {len(embedded_chunks)} vectors to Pinecone")
        start_time = time.time()
        
        try:
            batch_size = 100  # Pinecone optimal batch size
            successful_upserts = 0
            
            for i in range(0, len(embedded_chunks), batch_size):
                batch = embedded_chunks[i:i + batch_size]
                
                # Progress update
                progress = 90 + (i / len(embedded_chunks)) * 10  # Indexing is final 10%
                metrics.update_progress(
                    ProcessingStage.INDEXING,
                    progress,
                    f"Indexing batch {i//batch_size + 1}/{(len(embedded_chunks)-1)//batch_size + 1}"
                )
                
                # Prepare vectors for upsert
                vectors = []
                for j, (chunk, embedding) in enumerate(batch):
                    vector_id = f"{book_id}_chunk_{i + j}"
                    
                    # Prepare metadata (Pinecone has size limits)
                    metadata = {
                        'text': chunk.page_content[:1000],  # Limit text size
                        'filename': chunk.metadata.get('filename', ''),
                        'book_type': chunk.metadata.get('book_type', ''),
                        'page': chunk.metadata.get('page', 0),
                        'chunk_index': chunk.metadata.get('chunk_index', 0),
                        'medical_score': chunk.metadata.get('medical_score', 0.0),
                        'medical_terms': chunk.metadata.get('medical_terms', 0),
                        'word_count': chunk.metadata.get('word_count', 0),
                        'is_medical': chunk.metadata.get('is_medical_content', False),
                        'indexed_at': datetime.now().isoformat()
                    }
                    
                    vectors.append({
                        'id': vector_id,
                        'values': embedding,
                        'metadata': metadata
                    })
                
                # Upsert with retry logic
                retry_count = 0
                while retry_count < self.config.retry_attempts:
                    try:
                        response = self.index.upsert(vectors=vectors)
                        successful_upserts += len(vectors)
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= self.config.retry_attempts:
                            logger.error(f"Batch upsert failed after {retry_count} retries: {e}")
                            metrics.errors.append(f"Upsert batch failed: {str(e)}")
                        else:
                            logger.warning(f"Upsert retry {retry_count}: {e}")
                            await asyncio.sleep(2 ** retry_count)
                
                metrics.indexed_chunks = successful_upserts
            
            indexing_time = time.time() - start_time
            metrics.indexing_time = indexing_time
            
            success_rate = (successful_upserts / len(embedded_chunks)) * 100
            metrics.success_rate = success_rate
            
            logger.info(f"Vector indexing completed: {successful_upserts}/{len(embedded_chunks)} "
                       f"vectors indexed ({success_rate:.1f}% success) in {indexing_time:.2f}s")
            
            return success_rate > 90.0  # Consider successful if >90% indexed
            
        except Exception as e:
            logger.error(f"Vector upsert failed: {e}")
            metrics.errors.append(f"Vector upsert failed: {str(e)}")
            raise

class UltimateMedicalBookProcessor:
    """Main book processor orchestrating all components"""
    
    def __init__(self, config: BookProcessingConfig, pinecone_config: Dict[str, Any],
                 redis_client=None):
        self.config = config
        self.pinecone_config = pinecone_config
        self.redis_client = redis_client
        
        # Initialize components
        self.text_extractor = AdvancedTextExtractor(config)
        self.chunker = IntelligentChunker(config)
        self.embedding_engine = EmbeddingEngine(config)
        self.pinecone_manager = PineconeManager(pinecone_config)
        self.medical_extractor = MedicalTerminologyExtractor()
        
        # Progress tracking
        self.active_processes = {}
        self.process_lock = threading.RLock()
    
    def generate_book_id(self, filename: str, file_path: Path) -> str:
        """Generate unique book ID"""
        file_hash = hashlib.md5(f"{filename}_{file_path.stat().st_mtime}".encode()).hexdigest()
        return f"book_{file_hash[:12]}"
    
    async def validate_file(self, file_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
        """Comprehensive file validation"""
        logger.info(f"Validating file: {file_path.name}")
        
        try:
            # Basic checks
            if not file_path.exists():
                return False, "File does not exist", {}
            
            if file_path.suffix.lower() != '.pdf':
                return False, "Only PDF files are supported", {}
            
            file_size = file_path.stat().st_size
            if file_size > 500 * 1024 * 1024:  # 500MB limit
                return False, "File size exceeds 500MB limit", {}
            
            # PDF structure validation
            try:
                doc = fitz.open(str(file_path))
                page_count = len(doc)
                
                if page_count == 0:
                    doc.close()
                    return False, "PDF contains no pages", {}
                
                # Sample first few pages for content analysis
                sample_text = ""
                max_sample_pages = min(5, page_count)
                
                for page_num in range(max_sample_pages):
                    page = doc[page_num]
                    page_text = page.get_text()
                    sample_text += page_text + " "
                
                doc.close()
                
                # Medical content analysis
                medical_analysis = self.medical_extractor.analyze_text(sample_text)
                
                # Book type classification
                book_type, confidence = BookTypeClassifier.classify(
                    file_path.name, sample_text, page_count
                )
                
                validation_result = {
                    'file_size': file_size,
                    'page_count': page_count,
                    'sample_text_length': len(sample_text),
                    'medical_relevance_score': medical_analysis['medical_relevance_score'],
                    'is_medical_content': medical_analysis['is_medical_content'],
                    'book_type': book_type.value,
                    'book_type_confidence': confidence,
                    'estimated_processing_time': self._estimate_processing_time(file_size, page_count),
                    'recommended_config': self._get_recommended_config(book_type, page_count)
                }
                
                # Validation criteria
                is_valid = (
                    medical_analysis['is_medical_content'] or
                    medical_analysis['medical_relevance_score'] > 2.0 or
                    confidence > 0.3
                )
                
                message = "File validation successful" if is_valid else "Low medical content detected"
                
                return is_valid, message, validation_result
                
            except Exception as e:
                return False, f"PDF validation failed: {str(e)}", {}
                
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False, f"Validation error: {str(e)}", {}
    
    def _estimate_processing_time(self, file_size: int, page_count: int) -> Dict[str, float]:
        """Estimate processing time based on file characteristics"""
        # Time per page estimates (in seconds)
        base_time_per_page = 0.5
        gpu_speedup = 0.3 if self.config.use_gpu and TORCH_AVAILABLE else 1.0
        
        extraction_time = page_count * base_time_per_page * gpu_speedup
        chunking_time = page_count * 0.1
        embedding_time = (page_count / 10) * 2.0 * gpu_speedup
        indexing_time = (page_count / 10) * 1.0
        
        total_time = extraction_time + chunking_time + embedding_time + indexing_time
        
        return {
            'extraction_minutes': extraction_time / 60,
            'chunking_minutes': chunking_time / 60,
            'embedding_minutes': embedding_time / 60,
            'indexing_minutes': indexing_time / 60,
            'total_minutes': total_time / 60
        }
    
    def _get_recommended_config(self, book_type: BookType, page_count: int) -> BookProcessingConfig:
        """Get recommended configuration for book type"""
        config = BookProcessingConfig()
        
        # Adjust based on book type
        type_configs = {
            BookType.ANATOMY: {'chunk_size': 800, 'chunk_overlap': 150},
            BookType.PHYSIOLOGY: {'chunk_size': 1000, 'chunk_overlap': 200},
            BookType.PATHOLOGY: {'chunk_size': 1200, 'chunk_overlap': 250},
            BookType.PHARMACOLOGY: {'chunk_size': 900, 'chunk_overlap': 180},
            BookType.SURGERY: {'chunk_size': 1100, 'chunk_overlap': 220},
        }
        
        if book_type in type_configs:
            for key, value in type_configs[book_type].items():
                setattr(config, key, value)
        
        # Adjust based on page count
        if page_count > 1000:
            config.batch_size = 150
            config.max_workers = min(8, config.max_workers * 2)
        elif page_count > 500:
            config.batch_size = 120
        
        return config
    
    async def process_uploaded_file(self, file_path: str, filename: str, 
                                   uploaded_by: str = None, auth_context: Dict = None) -> Dict[str, Any]:
        """Enhanced book upload with auth integration and admin logging"""
        start_time = time.time()
        
        try:
            logger.info(f"🔄 Processing uploaded file: {filename} by {uploaded_by}")
            
            # Enhanced auth validation
            if auth_context and not auth_context.get('is_admin', False):
                return {
                    'success': False,
                    'message': "Admin privileges required for book upload",
                    'filename': filename,
                    'processing_time': time.time() - start_time,
                    'error_code': "INSUFFICIENT_PRIVILEGES"
                }
            
            # Convert to Path object for processing
            file_path_obj = Path(file_path)
            
            # Generate book ID
            book_id = f"book_{hashlib.md5(f'{filename}_{uploaded_by}_{time.time()}'.encode()).hexdigest()[:12]}"
            
            # Process the book using existing method
            processing_result = await self.process_book(file_path_obj, book_id)
            
            # Log admin action if auth context available
            if uploaded_by and hasattr(self, '_log_upload_action'):
                self._log_upload_action(uploaded_by, filename, processing_result, auth_context)
            
            # Return enhanced result
            return {
                'success': processing_result.success,
                'message': f"Successfully processed {filename}" if processing_result.success else processing_result.error_message,
                'filename': filename,
                'book_id': book_id,
                'chunks_processed': processing_result.total_chunks,
                'vectors_stored': processing_result.vectors_stored,
                'processing_time': processing_result.total_processing_time,
                'uploaded_by': uploaded_by,
                'medical_score': getattr(processing_result, 'medical_relevance_score', 0.0),
                'categories': getattr(processing_result, 'detected_categories', []),
                'pages_processed': getattr(processing_result, 'total_pages', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Book upload processing failed: {e}")
            return {
                'success': False,
                'message': f"Processing failed: {str(e)}",
                'filename': filename,
                'processing_time': time.time() - start_time,
                'error_code': "PROCESSING_ERROR"
            }
    
    def _log_upload_action(self, uploaded_by: str, filename: str, result: Any, auth_context: Dict = None):
        """Log book upload action for admin audit"""
        try:
            log_entry = {
                'admin_email': uploaded_by,
                'action': 'book_upload',
                'filename': filename,
                'success': getattr(result, 'success', False),
                'chunks_processed': getattr(result, 'total_chunks', 0),
                'processing_time': getattr(result, 'total_processing_time', 0),
                'timestamp': datetime.now().isoformat(),
                'user_agent': auth_context.get('user_agent', 'unknown') if auth_context else 'unknown'
            }
            
            # In a real implementation, this would log to database
            logger.info(f"📊 Book upload logged: {json.dumps(log_entry, indent=2)}")
            
        except Exception as e:
            logger.error(f"Upload action logging failed: {e}")
    
    async def process_book(self, file_path: Path, book_id: str = None) -> ProcessingMetrics:
        """Process a single book end-to-end"""
        if book_id is None:
            book_id = self.generate_book_id(file_path.name, file_path)
        
        logger.info(f"Starting book processing: {file_path.name} (ID: {book_id})")
        
        # Initialize metrics
        metrics = ProcessingMetrics(
            book_id=book_id,
            filename=file_path.name,
            file_size=file_path.stat().st_size,
            book_type=BookType.GENERAL,  # Will be updated during processing
            processing_stage=ProcessingStage.VALIDATION,
            start_time=time.time(),
            current_step_time=time.time()
        )
        
        # Store in active processes
        with self.process_lock:
            self.active_processes[book_id] = metrics
        
        try:
            # Step 1: File Validation
            metrics.update_progress(ProcessingStage.VALIDATION, 5, "Validating file")
            is_valid, message, validation_data = await self.validate_file(file_path)
            
            if not is_valid:
                raise ValueError(f"File validation failed: {message}")
            
            # Update book type from validation
            metrics.book_type = BookType(validation_data['book_type'])
            logger.info(f"Book classified as: {metrics.book_type.value}")
            
            # Step 2: Initialize components
            metrics.update_progress(ProcessingStage.VALIDATION, 10, "Initializing processing engines")
            embedding_dimension = await self.embedding_engine.initialize()
            await self.pinecone_manager.initialize(embedding_dimension)
            
            # Step 3: Text Extraction
            metrics.update_progress(ProcessingStage.EXTRACTION, 15, "Extracting text and images")
            pages_data = await self.text_extractor.extract_from_pdf(file_path, metrics)
            
            if not pages_data:
                raise ValueError("No content extracted from PDF")
            
            # Step 4: Intelligent Chunking
            metrics.update_progress(ProcessingStage.CHUNKING, 45, "Creating intelligent chunks")
            book_metadata = {
                'filename': file_path.name,
                'book_type': metrics.book_type.value,
                'book_id': book_id,
                'page_count': len(pages_data)
            }
            
            chunks = await self.chunker.create_chunks(pages_data, book_metadata, metrics)
            
            if not chunks:
                raise ValueError("No chunks created from extracted text")
            
            # Step 5: Generate Embeddings
            metrics.update_progress(ProcessingStage.EMBEDDINGS, 75, "Generating vector embeddings")
            embedded_chunks = await self.embedding_engine.generate_embeddings(chunks, metrics)
            
            # Step 6: Index in Pinecone
            metrics.update_progress(ProcessingStage.INDEXING, 90, "Indexing vectors in database")
            success = await self.pinecone_manager.upsert_vectors(embedded_chunks, book_id, metrics)
            
            if not success:
                metrics.warnings.append("Indexing completed with some failures")
            
            # Final metrics calculation
            metrics.total_processing_time = time.time() - metrics.start_time
            metrics.processing_stage = ProcessingStage.COMPLETED
            metrics.progress_percent = 100.0
            
            # Calculate quality scores
            if metrics.extracted_text_length > 0:
                metrics.content_quality_score = min(
                    (metrics.indexed_chunks / metrics.total_chunks) * 100, 100.0
                ) if metrics.total_chunks > 0 else 0
            
            # Store processing results in Redis if available
            if self.redis_client:
                await self._store_processing_results(book_id, metrics)
            
            logger.info(f"Book processing completed successfully: {file_path.name}")
            logger.info(f"Metrics: {metrics.total_chunks} chunks, {metrics.indexed_chunks} indexed, "
                       f"{metrics.total_processing_time:.2f}s total")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Book processing failed: {e}")
            logger.error(traceback.format_exc())
            
            metrics.processing_stage = ProcessingStage.FAILED
            metrics.errors.append(str(e))
            metrics.total_processing_time = time.time() - metrics.start_time
            
            return metrics
            
        finally:
            # Clean up
            with self.process_lock:
                if book_id in self.active_processes:
                    del self.active_processes[book_id]
    
    async def _store_processing_results(self, book_id: str, metrics: ProcessingMetrics):
        """Store processing results in Redis"""
        try:
            if self.redis_client:
                key = f"book_processing:{book_id}"
                data = metrics.to_dict()
                self.redis_client.setex(key, 86400, json.dumps(data, default=str))  # 24 hours
                logger.info(f"Processing results stored in Redis: {key}")
        except Exception as e:
            logger.warning(f"Failed to store results in Redis: {e}")
    
    def get_processing_status(self, book_id: str) -> Optional[ProcessingMetrics]:
        """Get current processing status for a book"""
        with self.process_lock:
            return self.active_processes.get(book_id)
    
    def get_all_active_processes(self) -> Dict[str, ProcessingMetrics]:
        """Get all currently active processes"""
        with self.process_lock:
            return self.active_processes.copy()

# Factory function for easy integration
def create_book_processor(pinecone_api_key: str, redis_client=None, 
                         custom_config: Dict[str, Any] = None) -> AdvancedBookProcessor:
    """Factory function to create configured book processor"""
    
    # Default configuration
    config = BookProcessingConfig()
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Pinecone configuration
    pinecone_config = {
        'api_key': pinecone_api_key,
        'index_name': os.getenv('PINECONE_INDEX_NAME', 'medical-books-ultimate'),
        'environment': os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
    }
    
    return AdvancedBookProcessor(config, pinecone_config, redis_client)

if __name__ == "__main__":
    # Example usage
    async def main():
        # Configuration
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            print("Error: PINECONE_API_KEY environment variable not set")
            return
        
        # Create processor
        processor = create_book_processor(pinecone_api_key)
        
        # Process a book
        test_file = Path("data/sample_medical_book.pdf")
        if test_file.exists():
            metrics = await processor.process_book(test_file)
            print(f"Processing completed: {metrics.processing_stage}")
            print(f"Success rate: {metrics.success_rate:.1f}%")
        else:
            print(f"Test file not found: {test_file}")
    
    # Run example
    asyncio.run(main())