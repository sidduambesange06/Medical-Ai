#!/usr/bin/env python3
"""
Advanced medical vision processing system.
Multi-engine OCR for medical images (prescriptions, lab reports, X-rays, etc.)
with medical entity extraction and document classification.
"""

import os
import sys
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json
import re
import base64
from pathlib import Path

# Optional advanced imports with fallbacks
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False

try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

try:
    from skimage import feature, filters, measure, morphology, segmentation
    from skimage.restoration import denoise_nl_means, estimate_sigma
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """OCR result with confidence and metadata"""
    text: str
    confidence: float
    engine: str
    bbox: Optional[List[Tuple[int, int]]] = None
    language: Optional[str] = None
    processing_time: Optional[float] = None

@dataclass
class MedicalEntity:
    """Extracted medical entity"""
    entity: str
    category: str  # medication, condition, dosage, etc.
    confidence: float
    context: str

@dataclass  
class MedicalImageAnalysis:
    """Complete medical image analysis result"""
    image_type: str
    ocr_results: List[OCRResult]
    medical_entities: List[MedicalEntity]
    extracted_text: str
    confidence_score: float
    recommendations: List[str]
    processing_metadata: Dict
    warnings: List[str]

class UltraAdvancedMedicalVision:
    """World's most advanced medical image processing system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Initialize all available OCR engines
        self.ocr_engines = {}
        self.setup_ocr_engines()
        
        # Medical knowledge patterns
        self.medical_patterns = self.load_medical_patterns()
        
        # Image preprocessing configurations
        self.preprocessing_configs = self.load_preprocessing_configs()
        
        self.logger.info("🏥 Ultra-Advanced Medical Vision System initialized")
    
    def setup_logging(self):
        """Setup advanced logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _find_tesseract_advanced(self):
        """Advanced Tesseract detection system"""
        import os
        import subprocess
        import glob
        import platform
        
        self.logger.info("🔍 Running advanced Tesseract detection...")
        
        # Priority-ordered search locations
        search_locations = [
            # Standard Windows locations
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            
            # Third-party app bundles (like Sejda PDF)
            r"C:\Program Files\Sejda PDF Desktop\resources\vendor\tesseract-windows-x64\tesseract.exe",
            r"C:\Program Files (x86)\Sejda PDF Desktop\resources\vendor\tesseract-windows-x64\tesseract.exe",
            
            # Common alternative locations
            r"C:\tesseract\tesseract.exe",
            r"C:\tools\tesseract\tesseract.exe",
            
            # PATH environment variable
            "tesseract"
        ]
        
        # Test each location
        for path in search_locations:
            try:
                self.logger.info(f"🔍 Testing Tesseract path: {path}")
                
                if path == "tesseract":
                    # Test if tesseract is in PATH
                    result = subprocess.run(
                        ["tesseract", "--version"], 
                        capture_output=True, 
                        text=True, 
                        timeout=5
                    )
                    if result.returncode == 0:
                        version_info = result.stdout.strip().split('\n')[0] if result.stdout else "Unknown"
                        self.logger.info(f"✅ Found Tesseract in PATH: {version_info}")
                        return "tesseract"
                else:
                    # Test specific file path
                    if os.path.isfile(path):
                        result = subprocess.run(
                            [path, "--version"], 
                            capture_output=True, 
                            text=True, 
                            timeout=5
                        )
                        if result.returncode == 0:
                            version_info = result.stdout.strip().split('\n')[0] if result.stdout else "Unknown"
                            self.logger.info(f"✅ Found working Tesseract: {path}")
                            self.logger.info(f"   Version: {version_info}")
                            return path
                        else:
                            self.logger.warning(f"⚠️ File exists but failed version test: {path}")
                    else:
                        self.logger.debug(f"📂 Path not found: {path}")
                        
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
                self.logger.debug(f"❌ Test failed for {path}: {e}")
                continue
            except Exception as e:
                self.logger.warning(f"⚠️ Unexpected error testing {path}: {e}")
                continue
        
        # Advanced search: scan Program Files directories
        self.logger.info("🔍 Performing deep scan of Program Files...")
        program_files_dirs = [
            r"C:\Program Files",
            r"C:\Program Files (x86)"
        ]
        
        for base_dir in program_files_dirs:
            if os.path.exists(base_dir):
                try:
                    # Use glob to find tesseract.exe recursively
                    pattern = os.path.join(base_dir, "**/tesseract.exe")
                    found_paths = glob.glob(pattern, recursive=True)
                    
                    for path in found_paths:
                        try:
                            self.logger.info(f"🔍 Testing discovered path: {path}")
                            result = subprocess.run(
                                [path, "--version"], 
                                capture_output=True, 
                                text=True, 
                                timeout=5
                            )
                            if result.returncode == 0:
                                version_info = result.stdout.strip().split('\n')[0] if result.stdout else "Unknown"
                                self.logger.info(f"✅ Found working Tesseract via deep scan: {path}")
                                self.logger.info(f"   Version: {version_info}")
                                return path
                        except Exception as e:
                            self.logger.debug(f"❌ Deep scan test failed for {path}: {e}")
                            continue
                            
                except Exception as e:
                    self.logger.warning(f"⚠️ Error scanning {base_dir}: {e}")
        
        self.logger.error("❌ Tesseract not found in any location")
        self.logger.info("💡 To install Tesseract:")
        self.logger.info("   1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        self.logger.info("   2. Install to: C:\\Program Files\\Tesseract-OCR")
        self.logger.info("   3. Or run: python install_tesseract_windows.py")
        
        return None
    
    def _configure_tessdata(self, tesseract_path):
        """Configure tessdata path for Tesseract OCR"""
        import os
        
        self.logger.info(f"🔍 Configuring tessdata for Tesseract at: {tesseract_path}")
        
        # Get the directory containing tesseract.exe
        tesseract_dir = os.path.dirname(tesseract_path)
        
        # Common tessdata locations relative to tesseract executable
        potential_tessdata_paths = [
            os.path.join(tesseract_dir, "tessdata"),                    # Same directory
            os.path.join(tesseract_dir, "..", "tessdata"),              # Parent directory  
            os.path.join(tesseract_dir, "..", "share", "tessdata"),     # Standard Linux/Unix layout
            os.path.join(tesseract_dir, "share", "tessdata"),           # Some Windows installations
            os.path.join(tesseract_dir, "data"),                        # Alternative name
            os.path.join(tesseract_dir, "traineddata"),                 # Alternative name
        ]
        
        # Also check some system-wide locations
        system_tessdata_paths = [
            r"C:\Program Files\Tesseract-OCR\tessdata",
            r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
            os.path.join(os.environ.get('PROGRAMFILES', ''), 'Tesseract-OCR', 'tessdata'),
            os.path.join(os.environ.get('PROGRAMFILES(X86)', ''), 'Tesseract-OCR', 'tessdata'),
        ]
        
        all_paths = potential_tessdata_paths + system_tessdata_paths
        
        # Test each potential tessdata path
        for tessdata_path in all_paths:
            if os.path.exists(tessdata_path) and os.path.isdir(tessdata_path):
                self.logger.info(f"📂 Found tessdata directory: {tessdata_path}")
                
                # Verify it contains language files
                eng_file = os.path.join(tessdata_path, "eng.traineddata")
                if os.path.exists(eng_file):
                    self.logger.info(f"✅ Found English language file: {eng_file}")
                    
                    # List all available language files
                    try:
                        language_files = [f for f in os.listdir(tessdata_path) if f.endswith('.traineddata')]
                        self.logger.info(f"📚 Available languages: {', '.join([f.replace('.traineddata', '') for f in language_files])}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not list language files: {e}")
                    
                    return tessdata_path
                else:
                    self.logger.warning(f"⚠️ Tessdata directory found but missing eng.traineddata: {tessdata_path}")
        
        # If no tessdata found, try to download it
        self.logger.warning("❌ No valid tessdata directory found")
        self.logger.info("💡 Tessdata configuration suggestions:")
        self.logger.info("   1. Download language data from: https://github.com/tesseract-ocr/tessdata")
        self.logger.info("   2. Extract to a 'tessdata' folder next to tesseract.exe")
        self.logger.info("   3. Ensure eng.traineddata file is present")
        
        # Try to create tessdata directory and download basic English data
        try:
            tessdata_dir = os.path.join(tesseract_dir, "tessdata")
            if not os.path.exists(tessdata_dir):
                os.makedirs(tessdata_dir)
                self.logger.info(f"📁 Created tessdata directory: {tessdata_dir}")
                
                # Try to download English language data
                import urllib.request
                eng_url = "https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata"
                eng_file = os.path.join(tessdata_dir, "eng.traineddata")
                
                self.logger.info("⬇️ Attempting to download English language data...")
                urllib.request.urlretrieve(eng_url, eng_file)
                
                if os.path.exists(eng_file) and os.path.getsize(eng_file) > 1000000:  # Should be several MB
                    self.logger.info("✅ Successfully downloaded English language data")
                    return tessdata_dir
                else:
                    self.logger.warning("⚠️ Downloaded file seems incomplete")
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Could not auto-download tessdata: {e}")
        
        return None
    
    def setup_ocr_engines(self):
        """Initialize all available OCR engines"""
        self.logger.info("🔧 Initializing OCR engines...")
        
        # 1. Tesseract OCR (Primary) - Advanced Auto-Detection
        try:
            tesseract_found = False
            tesseract_path = self._find_tesseract_advanced()
            
            if tesseract_path:
                try:
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path
                    
                    # Configure tessdata path if needed
                    tessdata_path = self._configure_tessdata(tesseract_path)
                    if tessdata_path:
                        os.environ['TESSDATA_PREFIX'] = tessdata_path
                        self.logger.info(f"📁 Set TESSDATA_PREFIX to: {tessdata_path}")
                    
                    # Test if it works with a simple image
                    test_img = Image.new('RGB', (100, 30), color='white')
                    test_result = pytesseract.image_to_string(test_img)
                    tesseract_found = True
                    self.ocr_engines['tesseract'] = 'available'
                    self.logger.info(f"✅ Tesseract OCR successfully initialized: {tesseract_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Tesseract test failed for {tesseract_path}: {e}")
            
            if not tesseract_found:
                self.logger.warning("❌ Tesseract not found or not working - medical OCR will use alternative engines")
                self.ocr_engines['tesseract'] = 'unavailable'
        except Exception as e:
            self.logger.error(f"❌ Tesseract setup failed: {e}")
            self.ocr_engines['tesseract'] = 'unavailable'
        
        # 2. EasyOCR (Excellent for multiple languages)
        if EASYOCR_AVAILABLE:
            try:
                self.easy_reader = easyocr.Reader(['en', 'es', 'fr', 'de', 'it'])  # Multi-language support
                self.ocr_engines['easyocr'] = 'available'
                self.logger.info("✅ EasyOCR initialized")
            except Exception as e:
                self.logger.error(f"❌ EasyOCR setup failed: {e}")
                self.ocr_engines['easyocr'] = 'unavailable'
        else:
            self.ocr_engines['easyocr'] = 'not_installed'
        
        # 3. PaddleOCR (Excellent for Asian languages and handwriting)
        if PADDLEOCR_AVAILABLE:
            try:
                self.paddle_ocr = PaddleOCR(use_textline_orientation=True, lang='en')
                self.ocr_engines['paddleocr'] = 'available'
                self.logger.info("✅ PaddleOCR initialized")
            except Exception as e:
                self.logger.error(f"❌ PaddleOCR setup failed: {e}")
                self.ocr_engines['paddleocr'] = 'unavailable'
        else:
            self.ocr_engines['paddleocr'] = 'not_installed'
        
        # 4. TrOCR (Transformer-based OCR for handwriting)
        if TROCR_AVAILABLE:
            try:
                # Try alternative TrOCR models that are more accessible
                model_options = [
                    'microsoft/trocr-base-printed',  # Try printed text model first
                    'microsoft/trocr-small-printed',  # Smaller model
                    'microsoft/trocr-base-handwritten'  # Original model as fallback
                ]
                
                self.trocr_processor = None
                self.trocr_model = None
                
                for model_name in model_options:
                    try:
                        self.logger.info(f"🔍 Trying TrOCR model: {model_name}")
                        self.trocr_processor = TrOCRProcessor.from_pretrained(model_name)
                        self.trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
                        self.ocr_engines['trocr'] = 'available'
                        self.logger.info(f"✅ TrOCR initialized with {model_name}")
                        break
                    except Exception as model_e:
                        self.logger.warning(f"⚠️ TrOCR model {model_name} failed: {str(model_e)[:100]}")
                        continue
                
                if self.trocr_processor is None:
                    raise Exception("All TrOCR models failed to load")
                    
            except Exception as e:
                self.logger.error(f"❌ TrOCR setup failed: {e}")
                self.ocr_engines['trocr'] = 'unavailable'
        else:
            self.ocr_engines['trocr'] = 'not_installed'
        
        available_engines = [k for k, v in self.ocr_engines.items() if v == 'available']
        self.logger.info(f"🚀 OCR Engines Ready: {len(available_engines)} / {len(self.ocr_engines)}")
        self.logger.info(f"Available: {', '.join(available_engines)}")
    
    def load_medical_patterns(self) -> Dict:
        """Load comprehensive medical patterns and knowledge"""
        return {
            'medications': [
                r'\b(?:mg|ml|mcg|units?|tablets?|capsules?)\b',
                r'\b\d+(?:\.\d+)?\s*(?:mg|ml|mcg|units?)\b',
                r'\bonce\s+daily\b|\btwo\s+times?\s+daily\b|\bthree\s+times?\s+daily\b',
                r'\bBID\b|\bTID\b|\bQID\b|\bQD\b|\bPRN\b'
            ],
            'vital_signs': [
                r'\b(?:BP|Blood\s+Pressure)[\s:]*\d+\/\d+\b',
                r'\b(?:HR|Heart\s+Rate)[\s:]*\d+\b',
                r'\b(?:Temp|Temperature)[\s:]*\d+(?:\.\d+)?°?[CF]?\b',
                r'\b(?:O2\s+Sat|SpO2)[\s:]*\d+%?\b'
            ],
            'lab_values': [
                r'\b(?:Glucose|HbA1c|Cholesterol|HDL|LDL|Triglycerides)[\s:]*\d+(?:\.\d+)?\b',
                r'\b(?:WBC|RBC|Hgb|Hct|Platelets)[\s:]*\d+(?:\.\d+)?\b'
            ],
            'dates': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b'
            ],
            'dosages': [
                r'\b\d+(?:\.\d+)?\s*(?:mg|ml|mcg|units?|IU|mEq)\b',
                r'\b(?:take|give|administer)\s+\d+(?:\.\d+)?\s*(?:mg|ml|tablets?)\b'
            ]
        }
    
    def load_preprocessing_configs(self) -> Dict:
        """Load image preprocessing configurations for different image types"""
        return {
            'prescription': {
                'denoise': True,
                'enhance_contrast': 1.5,
                'enhance_sharpness': 1.2,
                'binarization': True,
                'deskew': True
            },
            'lab_report': {
                'denoise': True,
                'enhance_contrast': 1.3,
                'enhance_sharpness': 1.1,
                'binarization': True,
                'table_detection': True
            },
            'xray': {
                'clahe': True,
                'gamma_correction': 1.2,
                'edge_enhancement': True,
                'denoise': False  # Preserve medical details
            },
            'handwritten': {
                'denoise': True,
                'enhance_contrast': 1.4,
                'enhance_sharpness': 1.3,
                'morphological_ops': True,
                'dilation': True
            }
        }
    
    def detect_image_type(self, image: np.ndarray) -> str:
        """Advanced image type detection using multiple methods"""
        height, width = image.shape[:2]
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate image statistics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Detect if image is predominantly dark (like X-rays)
        if mean_intensity < 60:
            return 'xray'
        
        # Detect if image has table-like structure (lab reports)
        if self._detect_table_structure(gray):
            return 'lab_report'
        
        # Detect handwritten content
        if self._detect_handwriting(gray):
            return 'handwritten'
        
        # Default to prescription/document
        return 'prescription'
    
    def _detect_table_structure(self, gray_image: np.ndarray) -> bool:
        """Detect table-like structures in images"""
        try:
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, vertical_kernel)
            
            h_line_count = np.sum(horizontal_lines > 0)
            v_line_count = np.sum(vertical_lines > 0)
            
            # If significant lines detected, likely a table
            threshold = gray_image.size * 0.01  # 1% of image pixels
            return h_line_count > threshold and v_line_count > threshold
        except:
            return False
    
    def _detect_handwriting(self, gray_image: np.ndarray) -> bool:
        """Detect handwritten content using edge analysis"""
        try:
            # Calculate edge density and irregularity
            edges = cv2.Canny(gray_image, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Handwritten text typically has higher edge density and more irregular patterns
            return edge_density > 0.1  # Threshold for handwriting detection
        except:
            return False
    
    def preprocess_image(self, image: np.ndarray, image_type: str) -> np.ndarray:
        """Advanced image preprocessing based on image type"""
        config = self.preprocessing_configs.get(image_type, self.preprocessing_configs['prescription'])
        
        if len(image.shape) == 3:
            # Convert BGR to RGB for PIL
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        
        processed_image = pil_image
        
        try:
            # Noise reduction
            if config.get('denoise', False) and SKIMAGE_AVAILABLE:
                # Convert to numpy for scikit-image
                img_array = np.array(processed_image)
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                
                # Estimate noise and apply non-local means denoising
                sigma_est = estimate_sigma(img_array, average_sigmas=True, channel_axis=None)
                denoised = denoise_nl_means(img_array, h=0.8 * sigma_est, sigma=sigma_est,
                                          fast_mode=True, patch_size=5, patch_distance=3)
                processed_image = Image.fromarray((denoised * 255).astype(np.uint8))
            
            # Contrast enhancement
            if config.get('enhance_contrast'):
                enhancer = ImageEnhance.Contrast(processed_image)
                processed_image = enhancer.enhance(config['enhance_contrast'])
            
            # Sharpness enhancement
            if config.get('enhance_sharpness'):
                enhancer = ImageEnhance.Sharpness(processed_image)
                processed_image = enhancer.enhance(config['enhance_sharpness'])
            
            # Binarization for text documents
            if config.get('binarization', False):
                # Convert to grayscale first
                if processed_image.mode != 'L':
                    processed_image = processed_image.convert('L')
                
                # Apply adaptive thresholding
                img_array = np.array(processed_image)
                binary = cv2.adaptiveThreshold(img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
                processed_image = Image.fromarray(binary)
            
            # CLAHE for X-rays
            if config.get('clahe', False):
                if processed_image.mode != 'L':
                    processed_image = processed_image.convert('L')
                
                img_array = np.array(processed_image)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(img_array)
                processed_image = Image.fromarray(enhanced)
            
            # Deskewing for documents
            if config.get('deskew', False):
                processed_image = self._deskew_image(processed_image)
            
            return np.array(processed_image)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Preprocessing failed: {e}, using original")
            return np.array(pil_image)
    
    def _deskew_image(self, image: Image.Image) -> Image.Image:
        """Deskew image to correct orientation"""
        try:
            # Convert to grayscale numpy array
            gray = np.array(image.convert('L'))
            
            # Apply Hough line transform to detect dominant lines
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Calculate the median angle
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = np.degrees(theta) - 90
                    if abs(angle) < 45:  # Only consider reasonable angles
                        angles.append(angle)
                
                if angles:
                    median_angle = np.median(angles)
                    if abs(median_angle) > 0.5:  # Only rotate if significant skew
                        return image.rotate(median_angle, expand=True, fillcolor='white')
            
            return image
        except:
            return image  # Return original if deskewing fails
    
    def ocr_tesseract(self, image: np.ndarray, config: str = 'medical') -> OCRResult:
        """Advanced Tesseract OCR with medical configuration"""
        if self.ocr_engines.get('tesseract') != 'available':
            raise Exception("Tesseract not available")
        
        try:
            pil_image = Image.fromarray(image)
            
            # Medical-optimized Tesseract configuration
            medical_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()_+-=[]{}|;:,.<>/? '
            
            if config == 'prescription':
                # Optimized for prescription text
                custom_config = '--oem 3 --psm 6 -c preserve_interword_spaces=1'
            elif config == 'handwritten':
                # For handwritten text (though Tesseract isn't best for this)
                custom_config = '--oem 3 --psm 8'
            else:
                custom_config = medical_config
            
            start_time = datetime.now()
            
            # Get text with confidence data
            data = pytesseract.image_to_data(pil_image, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Extract text and calculate confidence
            words = []
            confidences = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:  # Only include confident results
                    words.append(data['text'][i])
                    confidences.append(int(data['conf'][i]))
            
            text = ' '.join(words)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return OCRResult(
                text=text.strip(),
                confidence=avg_confidence / 100.0,  # Convert to 0-1 scale
                engine='tesseract',
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"❌ Tesseract OCR failed: {e}")
            return OCRResult(text="", confidence=0.0, engine='tesseract_failed')
    
    def ocr_easyocr(self, image: np.ndarray) -> OCRResult:
        """EasyOCR processing"""
        if self.ocr_engines.get('easyocr') != 'available':
            raise Exception("EasyOCR not available")
        
        try:
            start_time = datetime.now()
            results = self.easy_reader.readtext(image)
            
            # Combine all detected text
            text_parts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Only include confident results
                    text_parts.append(text)
                    confidences.append(confidence)
            
            combined_text = ' '.join(text_parts)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return OCRResult(
                text=combined_text,
                confidence=avg_confidence,
                engine='easyocr',
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"❌ EasyOCR failed: {e}")
            return OCRResult(text="", confidence=0.0, engine='easyocr_failed')
    
    def ocr_paddleocr(self, image: np.ndarray) -> OCRResult:
        """PaddleOCR processing"""
        if self.ocr_engines.get('paddleocr') != 'available':
            raise Exception("PaddleOCR not available")
        
        try:
            start_time = datetime.now()
            results = self.paddle_ocr.ocr(image)
            
            # Extract text and confidence
            text_parts = []
            confidences = []
            
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]
                        confidence = line[1][1]
                        
                        if confidence > 0.5:  # Only confident results
                            text_parts.append(text)
                            confidences.append(confidence)
            
            combined_text = ' '.join(text_parts)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return OCRResult(
                text=combined_text,
                confidence=avg_confidence,
                engine='paddleocr',
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"❌ PaddleOCR failed: {e}")
            return OCRResult(text="", confidence=0.0, engine='paddleocr_failed')
    
    def ocr_trocr(self, image: np.ndarray) -> OCRResult:
        """TrOCR (Transformer OCR) for handwritten text"""
        if self.ocr_engines.get('trocr') != 'available':
            raise Exception("TrOCR not available")
        
        try:
            start_time = datetime.now()
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image).convert('RGB')
            
            # Process with TrOCR
            pixel_values = self.trocr_processor(pil_image, return_tensors="pt").pixel_values
            generated_ids = self.trocr_model.generate(pixel_values)
            generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # TrOCR doesn't provide confidence scores, so we estimate based on text length and coherence
            confidence = min(0.9, len(generated_text) / 50.0) if generated_text else 0.0
            
            return OCRResult(
                text=generated_text,
                confidence=confidence,
                engine='trocr',
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"❌ TrOCR failed: {e}")
            return OCRResult(text="", confidence=0.0, engine='trocr_failed')
    
    def extract_medical_entities(self, text: str) -> List[MedicalEntity]:
        """Extract medical entities using pattern matching and NLP"""
        entities = []
        
        # Extract medications
        for pattern in self.medical_patterns['medications']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(MedicalEntity(
                    entity=match.group(),
                    category='medication',
                    confidence=0.8,
                    context=text[max(0, match.start()-20):match.end()+20]
                ))
        
        # Extract vital signs
        for pattern in self.medical_patterns['vital_signs']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(MedicalEntity(
                    entity=match.group(),
                    category='vital_sign',
                    confidence=0.9,
                    context=text[max(0, match.start()-20):match.end()+20]
                ))
        
        # Extract lab values
        for pattern in self.medical_patterns['lab_values']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(MedicalEntity(
                    entity=match.group(),
                    category='lab_value',
                    confidence=0.85,
                    context=text[max(0, match.start()-20):match.end()+20]
                ))
        
        # Extract dates
        for pattern in self.medical_patterns['dates']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(MedicalEntity(
                    entity=match.group(),
                    category='date',
                    confidence=0.95,
                    context=text[max(0, match.start()-10):match.end()+10]
                ))
        
        return entities
    
    def analyze_medical_image(self, image_path: str) -> MedicalImageAnalysis:
        """Complete medical image analysis using all available engines"""
        start_time = datetime.now()
        
        try:
            # Load and validate image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Detect image type
            image_type = self.detect_image_type(image)
            self.logger.info(f"🔍 Detected image type: {image_type}")
            
            # Preprocess image
            processed_image = self.preprocess_image(image, image_type)
            
            # Run all available OCR engines
            ocr_results = []
            
            # Tesseract
            if self.ocr_engines.get('tesseract') == 'available':
                try:
                    result = self.ocr_tesseract(processed_image, image_type)
                    ocr_results.append(result)
                except Exception as e:
                    self.logger.warning(f"⚠️ Tesseract failed: {e}")
            
            # EasyOCR
            if self.ocr_engines.get('easyocr') == 'available':
                try:
                    result = self.ocr_easyocr(processed_image)
                    ocr_results.append(result)
                except Exception as e:
                    self.logger.warning(f"⚠️ EasyOCR failed: {e}")
            
            # PaddleOCR
            if self.ocr_engines.get('paddleocr') == 'available':
                try:
                    result = self.ocr_paddleocr(processed_image)
                    ocr_results.append(result)
                except Exception as e:
                    self.logger.warning(f"⚠️ PaddleOCR failed: {e}")
            
            # TrOCR (for handwritten content)
            if image_type == 'handwritten' and self.ocr_engines.get('trocr') == 'available':
                try:
                    result = self.ocr_trocr(processed_image)
                    ocr_results.append(result)
                except Exception as e:
                    self.logger.warning(f"⚠️ TrOCR failed: {e}")
            
            if not ocr_results:
                raise Exception("No OCR engines available")
            
            # Select best result based on confidence
            best_result = max(ocr_results, key=lambda x: x.confidence)
            combined_text = best_result.text
            
            # If multiple engines available, try to combine results
            if len(ocr_results) > 1:
                # Use ensemble approach - combine results from multiple engines
                all_texts = [r.text for r in ocr_results if r.confidence > 0.3]
                if len(all_texts) > 1:
                    # For now, use the highest confidence result
                    # In future, could implement more sophisticated text combination
                    combined_text = best_result.text
            
            # Extract medical entities
            medical_entities = self.extract_medical_entities(combined_text)
            
            # Generate recommendations based on image type and content
            recommendations = self._generate_recommendations(image_type, combined_text, medical_entities)
            
            # Calculate overall confidence
            overall_confidence = best_result.confidence
            
            # Generate warnings
            warnings = self._generate_warnings(image_type, ocr_results, medical_entities)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return MedicalImageAnalysis(
                image_type=image_type,
                ocr_results=ocr_results,
                medical_entities=medical_entities,
                extracted_text=combined_text,
                confidence_score=overall_confidence,
                recommendations=recommendations,
                processing_metadata={
                    'processing_time': processing_time,
                    'engines_used': [r.engine for r in ocr_results],
                    'image_dimensions': image.shape,
                    'preprocessing_applied': image_type
                },
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"❌ Medical image analysis failed: {e}")
            raise
    
    def _generate_recommendations(self, image_type: str, text: str, entities: List[MedicalEntity]) -> List[str]:
        """Generate medical recommendations based on analysis"""
        recommendations = []
        
        if image_type == 'prescription':
            if entities:
                recommendations.append("✅ Prescription information detected successfully")
                medication_entities = [e for e in entities if e.category == 'medication']
                if medication_entities:
                    recommendations.append(f"📋 {len(medication_entities)} medication(s) identified")
            else:
                recommendations.append("⚠️ No clear medication information found - verify prescription details")
        
        elif image_type == 'lab_report':
            lab_entities = [e for e in entities if e.category == 'lab_value']
            if lab_entities:
                recommendations.append(f"🧪 {len(lab_entities)} lab value(s) detected")
                recommendations.append("📊 Consider tracking these values over time")
            else:
                recommendations.append("⚠️ No clear lab values found - check image quality")
        
        elif image_type == 'xray':
            recommendations.append("🏥 X-ray image detected - for interpretation, consult radiologist")
            recommendations.append("📸 Ensure image quality is sufficient for medical review")
        
        elif image_type == 'handwritten':
            recommendations.append("✍️ Handwritten content detected")
            recommendations.append("🔍 Consider manual verification of extracted text")
        
        # General recommendations
        if text and len(text) > 50:
            recommendations.append("✅ Substantial text content extracted")
        elif text:
            recommendations.append("⚠️ Limited text extracted - consider improving image quality")
        else:
            recommendations.append("❌ No text extracted - check image quality and OCR setup")
        
        return recommendations
    
    def _generate_warnings(self, image_type: str, ocr_results: List[OCRResult], entities: List[MedicalEntity]) -> List[str]:
        """Generate warnings based on analysis"""
        warnings = []
        
        # Check OCR confidence
        if ocr_results:
            avg_confidence = np.mean([r.confidence for r in ocr_results])
            if avg_confidence < 0.5:
                warnings.append("⚠️ Low OCR confidence - verify extracted text manually")
        
        # Check for critical medical information
        if image_type == 'prescription':
            dosage_entities = [e for e in entities if 'mg' in e.entity.lower() or 'ml' in e.entity.lower()]
            if not dosage_entities:
                warnings.append("⚠️ No dosage information clearly detected")
        
        # Check for multiple engine disagreement
        if len(ocr_results) > 1:
            texts = [r.text for r in ocr_results]
            if len(set(texts)) == len(texts):  # All different
                warnings.append("⚠️ OCR engines produced different results - manual verification recommended")
        
        return warnings
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            'engines_status': self.ocr_engines,
            'available_engines': [k for k, v in self.ocr_engines.items() if v == 'available'],
            'system_capabilities': {
                'tesseract': self.ocr_engines.get('tesseract') == 'available',
                'easyocr': self.ocr_engines.get('easyocr') == 'available',
                'paddleocr': self.ocr_engines.get('paddleocr') == 'available',
                'trocr': self.ocr_engines.get('trocr') == 'available',
                'dicom_support': DICOM_AVAILABLE,
                'advanced_preprocessing': SKIMAGE_AVAILABLE,
                'mediapipe': MEDIAPIPE_AVAILABLE
            },
            'supported_formats': ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
            'medical_specialties': [
                'prescription_analysis', 'lab_report_processing',
                'xray_preprocessing', 'handwritten_notes', 'medical_charts'
            ]
        }

# Global instance
ultra_medical_vision = None

def get_medical_vision_system() -> UltraAdvancedMedicalVision:
    """Get the global medical vision system instance"""
    global ultra_medical_vision
    if ultra_medical_vision is None:
        ultra_medical_vision = UltraAdvancedMedicalVision()
    return ultra_medical_vision

if __name__ == "__main__":
    # Test the system
    system = UltraAdvancedMedicalVision()
    status = system.get_system_status()
    
    print("🏥 Ultra-Advanced Medical Vision System")
    print("=" * 50)
    print(f"Available OCR Engines: {len(status['available_engines'])}")
    for engine in status['available_engines']:
        print(f"  ✅ {engine}")
    
    unavailable = [k for k, v in status['engines_status'].items() if v != 'available']
    if unavailable:
        print(f"Unavailable Engines: {len(unavailable)}")
        for engine in unavailable:
            print(f"  ❌ {engine}")
    
    print("\n🚀 System ready for medical image processing!")
    
    # Installation recommendations
    if 'tesseract' in unavailable:
        print("\n⚠️ To install Tesseract:")
        print("   Run: python install_tesseract_windows.py")
    
    if 'easyocr' in unavailable:
        print("\n⚠️ To install EasyOCR:")
        print("   pip install easyocr")
    
    if 'paddleocr' in unavailable:
        print("\n⚠️ To install PaddleOCR:")
        print("   pip install paddlepaddle paddleocr")