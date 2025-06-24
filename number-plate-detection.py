#!/usr/bin/env python3
"""
Complete Municipal Waste Detection System with Number Plate Detection
Advanced AI-powered waste throwing detection + Vehicle identification for municipal enforcement
Version 2.1 - With License Plate Recognition
"""

import sys
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import sqlite3
import threading
import time
import json
import logging
import hashlib
import base64
import shutil
from datetime import datetime, timedelta
from collections import defaultdict, deque
import subprocess
import math
import uuid
import re

# Advanced AI/ML imports with fallbacks
try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
    print("✅ YOLO/PyTorch available")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available. Basic detection will be used.")

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# OCR for license plate reading
try:
    import pytesseract
    OCR_AVAILABLE = True
    print("✅ Tesseract OCR available")
except ImportError:
    try:
        import easyocr
        OCR_AVAILABLE = True
        print("✅ EasyOCR available")
    except ImportError:
        OCR_AVAILABLE = False
        print("⚠️ No OCR available. Install pytesseract or easyocr for plate reading")

# GPS and location services
try:
    import geocoder
    GPS_AVAILABLE = True
except ImportError:
    GPS_AVAILABLE = False

# Advanced image processing
try:
    from skimage import feature, measure, morphology
    import scipy.ndimage as ndi
    ADVANCED_PROCESSING = True
except ImportError:
    ADVANCED_PROCESSING = False

# Email notifications
try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.base import MIMEBase
    from email import encoders
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

# Analytics
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import seaborn as sns
    import pandas as pd
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False

# Suppress OpenCV warnings
os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('municipal_waste_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "system_name": "Municipal Waste Detection System with Vehicle Tracking",
    "version": "2.1",
    "municipality": "Your City Name",
    "cameras": {
        "camera_0": {
            "name": "Main Street Camera",
            "location": "Main St & 1st Ave",
            "coordinates": [40.7128, -74.0060],
            "enforcement_zone": "downtown"
        }
    },
    "detection_settings": {
        "confidence_threshold": 0.7,
        "auto_citation_threshold": 0.85,
        "recording_duration": 30,
        "plate_detection_enabled": True,
        "plate_confidence_threshold": 0.6
    },
    "municipal_settings": {
        "base_fine_amounts": {
            "plastic": 150.0,
            "glass": 200.0,
            "metal": 100.0,
            "paper": 50.0,
            "organic": 75.0,
            "hazardous": 500.0
        },
        "enforcement_hours": {
            "start": "06:00",
            "end": "22:00"
        },
        "vehicle_tracking": {
            "enabled": True,
            "store_images": True,
            "link_to_incidents": True
        }
    },
    "weather_api_key": "",
    "database_path": "municipal_waste_detection.db"
}

class ConfigManager:
    """Configuration management for the system"""
    
    def __init__(self):
        self.config_file = "config.json"
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file or create default"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info("Configuration loaded from file")
                return config
            else:
                self.save_config(DEFAULT_CONFIG)
                logger.info("Default configuration created")
                return DEFAULT_CONFIG
        except Exception as e:
            logger.error(f"Config load error: {e}")
            return DEFAULT_CONFIG
    
    def save_config(self, config=None):
        """Save configuration to file"""
        try:
            if config is None:
                config = self.config
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info("Configuration saved")
        except Exception as e:
            logger.error(f"Config save error: {e}")
    
    def get(self, key, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

class NumberPlateDetector:
    """Advanced number plate detection and recognition system"""
    
    def __init__(self):
        self.plate_cascade = None
        self.setup_plate_detection()
        self.ocr_reader = None
        self.setup_ocr()
        
        # Plate validation patterns for different regions
        self.plate_patterns = {
            'US': [
                r'^[A-Z]{1,3}[0-9]{1,4}$',  # ABC123
                r'^[0-9]{1,3}[A-Z]{1,4}$',  # 123ABC
                r'^[A-Z]{2,3}[0-9]{2,4}[A-Z]{0,2}$',  # AB123C
            ],
            'EU': [
                r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$',  # A123BC
                r'^[0-9]{1,4}[A-Z]{2,4}$',  # 1234AB
            ],
            'INDIA': [
                r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{1,4}$',  # KL07AB1234
                r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$',  # 20BH1234A
            ]
        }
        
        logger.info("Number plate detector initialized")
    
    def setup_plate_detection(self):
        """Setup license plate detection"""
        try:
            # Try to load custom trained cascade
            cascade_paths = [
                'models/haarcascade_license_plate.xml',
                'haarcascade_russian_plate_number.xml',
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # Fallback
            ]
            
            for cascade_path in cascade_paths:
                if os.path.exists(cascade_path):
                    self.plate_cascade = cv2.CascadeClassifier(cascade_path)
                    if not self.plate_cascade.empty():
                        logger.info(f"Plate cascade loaded: {cascade_path}")
                        break
            
            if self.plate_cascade is None or self.plate_cascade.empty():
                logger.warning("No license plate cascade found, using contour-based detection")
                
        except Exception as e:
            logger.error(f"Plate cascade setup error: {e}")
    
    def setup_ocr(self):
        """Setup OCR for plate reading"""
        try:
            if OCR_AVAILABLE:
                try:
                    # Try EasyOCR first (more accurate for plates)
                    import easyocr
                    self.ocr_reader = easyocr.Reader(['en'])
                    logger.info("EasyOCR initialized for plate reading")
                except ImportError:
                    # Fallback to Tesseract
                    import pytesseract
                    self.ocr_reader = 'tesseract'
                    logger.info("Tesseract OCR initialized for plate reading")
            else:
                logger.warning("No OCR available - plate text recognition disabled")
                
        except Exception as e:
            logger.error(f"OCR setup error: {e}")
    
    def detect_license_plates(self, frame):
        """Detect license plates in frame"""
        try:
            detected_plates = []
            
            if frame is None or frame.size == 0:
                return frame, detected_plates
            
            # Use cascade detection if available
            if self.plate_cascade and not self.plate_cascade.empty():
                plates = self.detect_with_cascade(frame)
                detected_plates.extend(plates)
            
            # Always try contour-based detection for better coverage
            contour_plates = self.detect_with_contours(frame)
            detected_plates.extend(contour_plates)
            
            # Remove duplicate detections
            detected_plates = self.remove_duplicate_plates(detected_plates)
            
            # Read text from detected plates
            for plate in detected_plates:
                plate_text = self.read_plate_text(frame, plate)
                plate['text'] = plate_text
                plate['is_valid'] = self.validate_plate_text(plate_text)
            
            # Draw detections
            self.draw_plate_detections(frame, detected_plates)
            
            return frame, detected_plates
            
        except Exception as e:
            logger.error(f"License plate detection error: {e}")
            return frame, []
    
    def detect_with_cascade(self, frame):
        """Detect plates using Haar cascade"""
        plates = []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect plates
            detected = self.plate_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(100, 30),
                maxSize=(400, 150)
            )
            
            for (x, y, w, h) in detected:
                # Verify aspect ratio (plates are typically wider than tall)
                aspect_ratio = w / h
                if 2.0 < aspect_ratio < 6.0:
                    plates.append({
                        'method': 'cascade',
                        'bbox': (x, y, w, h),
                        'confidence': 0.8,
                        'aspect_ratio': aspect_ratio
                    })
                    
        except Exception as e:
            logger.debug(f"Cascade detection error: {e}")
        
        return plates
    
    def detect_with_contours(self, frame):
        """Detect plates using contour analysis"""
        plates = []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter to reduce noise
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Edge detection
            edges = cv2.Canny(gray, 30, 200)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
            
            for contour in contours:
                # Approximate contour
                epsilon = 0.018 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Look for rectangular shapes (4 corners)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    
                    # Check size and aspect ratio
                    area = w * h
                    aspect_ratio = w / h
                    
                    if (2000 < area < 30000 and 
                        2.0 < aspect_ratio < 6.0 and 
                        w > 80 and h > 20):
                        
                        # Additional validation using edge density
                        roi_edges = edges[y:y+h, x:x+w]
                        edge_density = np.sum(roi_edges > 0) / (w * h)
                        
                        if edge_density > 0.1:  # Plates have good edge density
                            confidence = min(0.9, edge_density * 2)
                            plates.append({
                                'method': 'contour',
                                'bbox': (x, y, w, h),
                                'confidence': confidence,
                                'aspect_ratio': aspect_ratio,
                                'edge_density': edge_density
                            })
                            
        except Exception as e:
            logger.debug(f"Contour detection error: {e}")
        
        return plates
    
    def remove_duplicate_plates(self, plates):
        """Remove duplicate plate detections"""
        try:
            if len(plates) <= 1:
                return plates
            
            # Sort by confidence
            plates.sort(key=lambda x: x['confidence'], reverse=True)
            
            filtered_plates = []
            for plate in plates:
                x1, y1, w1, h1 = plate['bbox']
                
                # Check overlap with existing plates
                is_duplicate = False
                for existing in filtered_plates:
                    x2, y2, w2, h2 = existing['bbox']
                    
                    # Calculate overlap
                    overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                    overlap_area = overlap_x * overlap_y
                    
                    # If significant overlap, it's a duplicate
                    if overlap_area > 0.5 * min(w1 * h1, w2 * h2):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    filtered_plates.append(plate)
            
            return filtered_plates
            
        except Exception as e:
            logger.error(f"Duplicate removal error: {e}")
            return plates
    
    def read_plate_text(self, frame, plate):
        """Read text from detected license plate"""
        try:
            if not self.ocr_reader:
                return "OCR_UNAVAILABLE"
            
            x, y, w, h = plate['bbox']
            
            # Extract plate region with some padding
            padding = 5
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            plate_roi = frame[y1:y2, x1:x2]
            
            if plate_roi.size == 0:
                return "INVALID_ROI"
            
            # Preprocess for better OCR
            plate_processed = self.preprocess_plate_for_ocr(plate_roi)
            
            # Read text using available OCR
            if hasattr(self.ocr_reader, 'readtext'):  # EasyOCR
                results = self.ocr_reader.readtext(plate_processed)
                if results:
                    # Combine all detected text
                    text_parts = [result[1] for result in results if result[2] > 0.5]
                    plate_text = ''.join(text_parts).upper()
                else:
                    plate_text = "NO_TEXT_DETECTED"
            else:  # Tesseract
                import pytesseract
                # Configure for license plates
                config = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                plate_text = pytesseract.image_to_string(plate_processed, config=config).strip().upper()
            
            # Clean up the text
            plate_text = self.clean_plate_text(plate_text)
            
            return plate_text if plate_text else "UNREADABLE"
            
        except Exception as e:
            logger.error(f"Plate text reading error: {e}")
            return "ERROR_READING"
    
    def preprocess_plate_for_ocr(self, plate_roi):
        """Preprocess plate image for better OCR accuracy"""
        try:
            # Convert to grayscale
            if len(plate_roi.shape) == 3:
                gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_roi.copy()
            
            # Resize for better OCR (minimum 200px width)
            h, w = gray.shape
            if w < 200:
                scale = 200 / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            
            # Top hat to enhance text
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            
            # Blackhat to enhance dark text on light background
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            
            # Combine
            enhanced = cv2.add(gray, tophat)
            enhanced = cv2.subtract(enhanced, blackhat)
            
            # Apply threshold
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Remove noise
            denoised = cv2.medianBlur(binary, 3)
            
            return denoised
            
        except Exception as e:
            logger.error(f"Plate preprocessing error: {e}")
            return plate_roi
    
    def clean_plate_text(self, text):
        """Clean and normalize plate text"""
        try:
            if not text:
                return ""
            
            # Remove non-alphanumeric characters
            cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
            
            # Remove common OCR errors
            replacements = {
                'O': '0',  # Common confusion
                'I': '1',  # Common confusion
                'S': '5',  # Sometimes confused
                'G': '6',  # Sometimes confused
            }
            
            # Only apply replacements if it makes the plate more valid
            for old, new in replacements.items():
                test_text = cleaned.replace(old, new)
                if self.validate_plate_text(test_text) and not self.validate_plate_text(cleaned):
                    cleaned = test_text
                    break
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Plate text cleaning error: {e}")
            return text
    
    def validate_plate_text(self, text):
        """Validate if text looks like a real license plate"""
        try:
            if not text or len(text) < 3 or len(text) > 10:
                return False
            
            # Check against common patterns
            for region, patterns in self.plate_patterns.items():
                for pattern in patterns:
                    if re.match(pattern, text):
                        return True
            
            # Basic validation: mix of letters and numbers
            has_letter = any(c.isalpha() for c in text)
            has_number = any(c.isdigit() for c in text)
            
            return has_letter and has_number
            
        except Exception as e:
            logger.error(f"Plate validation error: {e}")
            return False
    
    def draw_plate_detections(self, frame, plates):
        """Draw license plate detections on frame"""
        try:
            for plate in plates:
                x, y, w, h = plate['bbox']
                confidence = plate['confidence']
                plate_text = plate.get('text', 'DETECTING...')
                is_valid = plate.get('is_valid', False)
                
                # Color based on validity and confidence
                if is_valid and confidence > 0.7:
                    color = (0, 255, 0)  # Green for valid plates
                elif confidence > 0.5:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 165, 255)  # Orange for low confidence
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw confidence
                conf_text = f"{confidence:.2f}"
                cv2.putText(frame, conf_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw plate text
                if plate_text and plate_text != "DETECTING...":
                    # Background for text
                    text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x, y + h + 5), (x + text_size[0] + 10, y + h + 25), 
                                 (0, 0, 0), -1)
                    
                    # Text
                    text_color = (0, 255, 0) if is_valid else (255, 255, 255)
                    cv2.putText(frame, plate_text, (x + 5, y + h + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                
                # Add validity indicator
                if is_valid:
                    cv2.putText(frame, "✓", (x + w - 20, y + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
        except Exception as e:
            logger.error(f"Plate drawing error: {e}")

class InstantRecordingSystem:
    """Instant video and image recording system"""
    
    def __init__(self):
        self.active_recordings = {}
        self.recording_duration = 30  # seconds
        self.pre_recording_buffer = deque(maxlen=90)  # 3 seconds at 30fps
        
        # Ensure directories exist
        os.makedirs("recordings", exist_ok=True)
        os.makedirs("screenshots", exist_ok=True)
        os.makedirs("plate_captures", exist_ok=True)  # For license plate images
        logger.info("Recording system initialized")
    
    def start_incident_recording(self, camera_id, incident_type, frame):
        """Start recording immediately when incident detected"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # Create filenames
            video_filename = f"recordings/incident_{incident_type}_{camera_id}_{timestamp}.avi"
            image_filename = f"screenshots/incident_{incident_type}_{camera_id}_{timestamp}.jpg"
            
            # Save immediate screenshot
            if frame is not None and frame.size > 0:
                cv2.imwrite(image_filename, frame)
                logger.info(f"Screenshot saved: {image_filename}")
            
            # Setup video recording
            if frame is not None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                height, width = frame.shape[:2]
                video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (width, height))
                
                if video_writer.isOpened():
                    # Write pre-recording buffer
                    for buffered_frame in self.pre_recording_buffer:
                        if buffered_frame is not None:
                            video_writer.write(buffered_frame)
                    
                    # Store recording info
                    self.active_recordings[camera_id] = {
                        'writer': video_writer,
                        'video_filename': video_filename,
                        'image_filename': image_filename,
                        'start_time': time.time(),
                        'incident_type': incident_type,
                        'frame_count': 0
                    }
                    
                    logger.info(f"Recording started: {video_filename}")
                    return video_filename, image_filename
                else:
                    logger.error("Failed to start video recording")
                    return None, image_filename
            
            return None, image_filename
                
        except Exception as e:
            logger.error(f"Recording start error: {e}")
            return None, None
    
    def save_plate_capture(self, frame, plate_data, camera_id):
        """Save license plate capture"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            x, y, w, h = plate_data['bbox']
            
            # Extract plate region
            plate_img = frame[y:y+h, x:x+w]
            
            # Create filename
            plate_text = plate_data.get('text', 'UNKNOWN')
            clean_text = re.sub(r'[^A-Z0-9]', '', plate_text)
            
            plate_filename = f"plate_captures/plate_{clean_text}_{camera_id}_{timestamp}.jpg"
            
            # Save plate image
            if plate_img.size > 0:
                cv2.imwrite(plate_filename, plate_img)
                logger.info(f"Plate capture saved: {plate_filename}")
                return plate_filename
            
        except Exception as e:
            logger.error(f"Plate capture save error: {e}")
        
        return None
    
    def add_frame_to_buffer(self, frame):
        """Add frame to pre-recording buffer"""
        try:
            if frame is not None and frame.size > 0:
                self.pre_recording_buffer.append(frame.copy())
        except Exception as e:
            logger.debug(f"Buffer error: {e}")
    
    def update_recordings(self, camera_id, frame):
        """Update active recordings"""
        try:
            if camera_id in self.active_recordings and frame is not None:
                recording = self.active_recordings[camera_id]
                
                # Write frame
                recording['writer'].write(frame)
                recording['frame_count'] += 1
                
                # Check if recording duration exceeded
                if time.time() - recording['start_time'] > self.recording_duration:
                    self.stop_recording(camera_id)
                    
        except Exception as e:
            logger.error(f"Recording update error: {e}")
    
    def stop_recording(self, camera_id):
        """Stop recording for camera"""
        try:
            if camera_id in self.active_recordings:
                recording = self.active_recordings[camera_id]
                recording['writer'].release()
                
                logger.info(f"Recording stopped: {recording['video_filename']} ({recording['frame_count']} frames)")
                
                del self.active_recordings[camera_id]
                return recording['video_filename']
                
        except Exception as e:
            logger.error(f"Recording stop error: {e}")
        
        return None
    
    def stop_all_recordings(self):
        """Stop all active recordings"""
        for camera_id in list(self.active_recordings.keys()):
            self.stop_recording(camera_id)

class AdvancedWasteDetector:
    """Advanced AI-powered waste detection system"""
    
    def __init__(self):
        self.models_loaded = False
        self.yolo_model = None
        
        # Waste categories with municipal data
        self.waste_categories = {
            'plastic': {
                'items': ['bottle', 'bag', 'cup', 'container', 'wrapper', 'straw'],
                'fine_amount': 150.0,
                'severity': 'high',
                'color': (0, 0, 255)  # Red
            },
            'organic': {
                'items': ['food', 'fruit', 'vegetable', 'banana', 'apple'],
                'fine_amount': 75.0,
                'severity': 'medium',
                'color': (0, 255, 0)  # Green
            },
            'metal': {
                'items': ['can', 'bottle_cap', 'foil'],
                'fine_amount': 100.0,
                'severity': 'high',
                'color': (255, 0, 0)  # Blue
            },
            'paper': {
                'items': ['newspaper', 'tissue', 'napkin'],
                'fine_amount': 50.0,
                'severity': 'low',
                'color': (0, 255, 255)  # Yellow
            },
            'glass': {
                'items': ['bottle', 'jar'],
                'fine_amount': 200.0,
                'severity': 'very_high',
                'color': (255, 0, 255)  # Magenta
            }
        }
        
        self.setup_detection()
    
    def setup_detection(self):
        """Setup detection models"""
        try:
            logger.info("Setting up waste detection models...")
            
            if YOLO_AVAILABLE:
                try:
                    # Try to load YOLOv8 (will download if not present)
                    self.yolo_model = YOLO('yolov8n.pt')
                    logger.info("✅ YOLOv8 model loaded successfully")
                except Exception as e:
                    logger.warning(f"YOLOv8 loading failed: {e}")
                    self.yolo_model = None
            
            self.models_loaded = True
            logger.info("✅ Detection system initialized")
            
        except Exception as e:
            logger.error(f"Detection setup error: {e}")
            self.models_loaded = True  # Continue with basic detection
    
    def detect_waste_objects(self, frame):
        """Detect waste objects in frame"""
        try:
            if frame is None or frame.size == 0:
                return frame, []
                
            if self.yolo_model and YOLO_AVAILABLE:
                return self.detect_with_yolo(frame)
            else:
                return self.detect_with_opencv(frame)
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return frame, []
    
    def detect_with_yolo(self, frame):
        """YOLO-based detection"""
        try:
            detected_objects = []
            
            # Run YOLO inference
            results = self.yolo_model(frame, conf=0.3, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        try:
                            # Get box data
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            # Get class name
                            class_name = self.yolo_model.names[class_id]
                            
                            # Check if relevant to waste detection
                            if self.is_waste_relevant(class_name):
                                # Classify waste type
                                waste_type = self.classify_waste_type(class_name)
                                
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                detected_object = {
                                    'type': class_name,
                                    'waste_category': waste_type,
                                    'material': waste_type,
                                    'confidence': confidence,
                                    'bbox': (x1, y1, x2-x1, y2-y1),
                                    'center': ((x1+x2)//2, (y1+y2)//2),
                                    'area': (x2-x1) * (y2-y1),
                                    'fine_amount': self.waste_categories.get(waste_type, {}).get('fine_amount', 100),
                                    'severity': self.waste_categories.get(waste_type, {}).get('severity', 'medium')
                                }
                                
                                detected_objects.append(detected_object)
                                
                                # Draw detection
                                color = self.waste_categories.get(waste_type, {}).get('color', (255, 255, 255))
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                
                                label = f'{class_name}: {confidence:.2f}'
                                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                                fine_label = f'${detected_object["fine_amount"]:.0f}'
                                cv2.putText(frame, fine_label, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        except Exception as e:
                            logger.debug(f"Box processing error: {e}")
                            continue
            
            return frame, detected_objects
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return self.detect_with_opencv(frame)
    
    def detect_with_opencv(self, frame):
        """OpenCV-based fallback detection"""
        try:
            detected_objects = []
            
            if frame is None or frame.size == 0:
                return frame, detected_objects
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect bottles (plastic objects)
            bottles = self.detect_bottles(frame, hsv)
            detected_objects.extend(bottles)
            
            # Detect cans (metal objects)
            cans = self.detect_cans(frame, hsv)
            detected_objects.extend(cans)
            
            # Draw detections
            for obj in detected_objects:
                x, y, w, h = obj['bbox']
                color = self.waste_categories.get(obj['waste_category'], {}).get('color', (255, 255, 255))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                label = f"{obj['type']}: {obj['confidence']:.2f}"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                fine_label = f"${obj['fine_amount']:.0f}"
                cv2.putText(frame, fine_label, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            return frame, detected_objects
            
        except Exception as e:
            logger.error(f"OpenCV detection error: {e}")
            return frame, []
    
    def detect_bottles(self, frame, hsv):
        """Detect bottle-like objects"""
        bottles = []
        try:
            # Detect clear/transparent objects (bottles)
            lower_clear = np.array([0, 0, 200])
            upper_clear = np.array([180, 30, 255])
            mask = cv2.inRange(hsv, lower_clear, upper_clear)
            
            # Morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 15000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = h / w if w > 0 else 0
                    
                    if 1.2 < aspect_ratio < 4.0:  # Bottle shape
                        bottles.append({
                            'type': 'bottle',
                            'waste_category': 'plastic',
                            'material': 'plastic',
                            'confidence': min(0.8, area / 10000),
                            'bbox': (x, y, w, h),
                            'center': (x + w//2, y + h//2),
                            'area': area,
                            'fine_amount': self.waste_categories['plastic']['fine_amount'],
                            'severity': 'high'
                        })
        except Exception as e:
            logger.debug(f"Bottle detection error: {e}")
        
        return bottles
    
    def detect_cans(self, frame, hsv):
        """Detect can-like objects"""
        cans = []
        try:
            # Detect metallic objects
            lower_metal = np.array([0, 0, 150])
            upper_metal = np.array([180, 50, 255])
            mask = cv2.inRange(hsv, lower_metal, upper_metal)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 200 < area < 5000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = h / w if w > 0 else 0
                    
                    if 0.8 < aspect_ratio < 1.5:  # Can shape
                        cans.append({
                            'type': 'can',
                            'waste_category': 'metal',
                            'material': 'aluminum',
                            'confidence': min(0.7, area / 3000),
                            'bbox': (x, y, w, h),
                            'center': (x + w//2, y + h//2),
                            'area': area,
                            'fine_amount': self.waste_categories['metal']['fine_amount'],
                            'severity': 'high'
                        })
        except Exception as e:
            logger.debug(f"Can detection error: {e}")
        
        return cans
    
    def is_waste_relevant(self, class_name):
        """Check if object is relevant for waste detection"""
        waste_objects = [
            'bottle', 'cup', 'bowl', 'banana', 'apple', 'orange',
            'sandwich', 'handbag', 'backpack', 'book', 'cell phone',
            'laptop', 'mouse', 'remote', 'keyboard', 'scissors'
        ]
        return class_name.lower() in waste_objects
    
    def classify_waste_type(self, class_name):
        """Classify detected object into waste category"""
        class_name_lower = class_name.lower()
        
        for waste_type, info in self.waste_categories.items():
            if any(item in class_name_lower for item in info['items']):
                return waste_type
        
        return 'plastic'  # Default classification

class AdvancedThrowingDetector:
    """Advanced throwing behavior detection"""
    
    def __init__(self):
        self.motion_tracker = {}
        self.track_id_counter = 0
        self.throwing_patterns = deque(maxlen=100)
        logger.info("Throwing detector initialized")
    
    def detect_throwing_motion(self, objects, frame, timestamp):
        """Detect throwing behavior from object motion"""
        try:
            throwing_incidents = []
            
            if not objects:
                return throwing_incidents
            
            # Update object tracking
            tracked_objects = self.update_object_tracking(objects, timestamp)
            
            # Analyze each tracked object
            for track_id, track_data in tracked_objects.items():
                throwing_analysis = self.analyze_throwing_pattern(track_data)
                
                if throwing_analysis['is_throwing']:
                    throwing_incidents.append({
                        'track_id': track_id,
                        'confidence': throwing_analysis['confidence'],
                        'trajectory': throwing_analysis['trajectory'],
                        'object_type': track_data.get('object_type', 'unknown'),
                        'material': track_data.get('material', 'unknown'),
                        'fine_amount': track_data.get('fine_amount', 100),
                        'severity': throwing_analysis['severity']
                    })
            
            return throwing_incidents
            
        except Exception as e:
            logger.error(f"Throwing detection error: {e}")
            return []
    
    def update_object_tracking(self, detected_objects, timestamp):
        """Update object tracking"""
        try:
            current_tracks = {}
            
            for obj in detected_objects:
                center = obj['center']
                
                # Find matching track or create new one
                matched_track_id = self.find_matching_track(center)
                
                if matched_track_id:
                    # Update existing track
                    track_data = self.motion_tracker[matched_track_id]
                    track_data['positions'].append(center)
                    track_data['timestamps'].append(timestamp)
                    track_data['last_seen'] = timestamp
                    current_tracks[matched_track_id] = track_data
                else:
                    # Create new track
                    track_id = self.track_id_counter
                    self.track_id_counter += 1
                    
                    track_data = {
                        'track_id': track_id,
                        'positions': deque([center], maxlen=20),
                        'timestamps': deque([timestamp], maxlen=20),
                        'object_type': obj['type'],
                        'material': obj['material'],
                        'fine_amount': obj['fine_amount'],
                        'first_seen': timestamp,
                        'last_seen': timestamp
                    }
                    
                    self.motion_tracker[track_id] = track_data
                    current_tracks[track_id] = track_data
            
            # Clean up old tracks
            self.cleanup_old_tracks(timestamp)
            
            return current_tracks
            
        except Exception as e:
            logger.error(f"Object tracking error: {e}")
            return {}
    
    def find_matching_track(self, center, max_distance=50):
        """Find matching track for detected object"""
        try:
            best_match = None
            min_distance = float('inf')
            
            for track_id, track_data in self.motion_tracker.items():
                if len(track_data['positions']) > 0:
                    last_pos = track_data['positions'][-1]
                    distance = np.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)
                    
                    if distance < max_distance and distance < min_distance:
                        min_distance = distance
                        best_match = track_id
            
            return best_match
        except Exception as e:
            logger.debug(f"Track matching error: {e}")
            return None
    
    def analyze_throwing_pattern(self, track_data):
        """Analyze if motion pattern indicates throwing"""
        try:
            positions = list(track_data['positions'])
            timestamps = list(track_data['timestamps'])
            
            if len(positions) < 5:
                return {'is_throwing': False, 'confidence': 0.0, 'severity': 'low', 'trajectory': positions}
            
            # Calculate velocities
            velocities = []
            for i in range(1, len(positions)):
                dt = timestamps[i] - timestamps[i-1]
                if dt > 0:
                    dx = positions[i][0] - positions[i-1][0]
                    dy = positions[i][1] - positions[i-1][1]
                    velocity = np.sqrt(dx*dx + dy*dy) / dt
                    velocities.append(velocity)
            
            if not velocities:
                return {'is_throwing': False, 'confidence': 0.0, 'severity': 'low', 'trajectory': positions}
            
            # Throwing characteristics
            max_velocity = max(velocities)
            velocity_variance = np.var(velocities) if len(velocities) > 1 else 0
            
            # Simple throwing detection based on rapid motion
            confidence = 0.0
            if max_velocity > 20:  # Rapid motion detected
                confidence += 0.4
            
            if velocity_variance > 100:  # Variable motion (acceleration/deceleration)
                confidence += 0.3
            
            # Trajectory analysis
            if len(positions) >= 3:
                # Check for parabolic motion (up then down)
                y_positions = [pos[1] for pos in positions]
                if len(set(y_positions)) > 1:  # Check for variation in y positions
                    confidence += 0.3
            
            is_throwing = confidence > 0.6
            severity = self.determine_severity(track_data, max_velocity)
            
            return {
                'is_throwing': is_throwing,
                'confidence': min(confidence, 0.95),
                'severity': severity,
                'trajectory': positions,
                'max_velocity': max_velocity
            }
            
        except Exception as e:
            logger.error(f"Throwing analysis error: {e}")
            return {'is_throwing': False, 'confidence': 0.0, 'severity': 'low', 'trajectory': []}
    
    def determine_severity(self, track_data, max_velocity):
        """Determine severity based on object and motion"""
        try:
            material = track_data.get('material', 'unknown')
            
            if material == 'glass':
                return 'very_high'
            elif material == 'metal' and max_velocity > 30:
                return 'high'
            elif material == 'plastic':
                return 'high'
            else:
                return 'medium'
        except Exception as e:
            logger.debug(f"Severity determination error: {e}")
            return 'medium'
    
    def cleanup_old_tracks(self, current_time, max_age=3.0):
        """Remove old tracks"""
        try:
            tracks_to_remove = []
            for track_id, track_data in self.motion_tracker.items():
                if current_time - track_data['last_seen'] > max_age:
                    tracks_to_remove.append(track_id)
            
            for track_id in tracks_to_remove:
                del self.motion_tracker[track_id]
        except Exception as e:
            logger.error(f"Track cleanup error: {e}")

class MunicipalDatabase:
    """Municipal database management with vehicle tracking"""
    
    def __init__(self, db_path="municipal_waste_detection.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize municipal database with vehicle tracking tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Main incidents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS waste_incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_uuid TEXT UNIQUE,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    incident_type TEXT NOT NULL,
                    waste_category TEXT,
                    material_type TEXT,
                    confidence_score REAL,
                    throwing_confidence REAL,
                    location_description TEXT,
                    camera_id TEXT,
                    face_detected BOOLEAN DEFAULT 0,
                    face_count INTEGER DEFAULT 0,
                    screenshot_path TEXT,
                    video_path TEXT,
                    evidence_hash TEXT,
                    fine_amount REAL,
                    severity_level TEXT,
                    status TEXT DEFAULT 'pending',
                    citation_number TEXT,
                    reviewed BOOLEAN DEFAULT 0,
                    created_by TEXT DEFAULT 'system'
                )
            ''')
            
            # Vehicle/License plate detection table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detected_vehicles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id INTEGER,
                    plate_number TEXT,
                    plate_confidence REAL,
                    plate_is_valid BOOLEAN DEFAULT 0,
                    vehicle_type TEXT,
                    vehicle_color TEXT,
                    plate_image_path TEXT,
                    detection_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    camera_id TEXT,
                    bbox_coordinates TEXT,
                    plate_region TEXT,
                    verified BOOLEAN DEFAULT 0,
                    owner_notified BOOLEAN DEFAULT 0,
                    FOREIGN KEY (incident_id) REFERENCES waste_incidents (id)
                )
            ''')
            
            # Vehicle owner information (for municipal records)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vehicle_owners (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT UNIQUE,
                    owner_name TEXT,
                    owner_address TEXT,
                    owner_phone TEXT,
                    owner_email TEXT,
                    vehicle_make TEXT,
                    vehicle_model TEXT,
                    vehicle_year INTEGER,
                    vehicle_color TEXT,
                    registration_date DATE,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # License plate alerts and blacklist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS plate_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT,
                    alert_type TEXT,  -- 'wanted', 'repeat_offender', 'stolen', 'alert'
                    alert_reason TEXT,
                    created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_date DATETIME,
                    active BOOLEAN DEFAULT 1,
                    priority_level INTEGER DEFAULT 1
                )
            ''')
            
            # System configuration
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_config (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    category TEXT,
                    description TEXT
                )
            ''')
            
            # Daily statistics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date DATE PRIMARY KEY,
                    total_incidents INTEGER DEFAULT 0,
                    plastic_incidents INTEGER DEFAULT 0,
                    metal_incidents INTEGER DEFAULT 0,
                    glass_incidents INTEGER DEFAULT 0,
                    paper_incidents INTEGER DEFAULT 0,
                    organic_incidents INTEGER DEFAULT 0,
                    total_fines REAL DEFAULT 0,
                    vehicles_detected INTEGER DEFAULT 0,
                    plates_read INTEGER DEFAULT 0,
                    valid_plates INTEGER DEFAULT 0
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Municipal database with vehicle tracking initialized")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def add_incident(self, incident_data):
        """Add waste incident to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            incident_uuid = str(uuid.uuid4())
            
            cursor.execute('''
                INSERT INTO waste_incidents (
                    incident_uuid, incident_type, waste_category, material_type,
                    confidence_score, throwing_confidence, location_description,
                    camera_id, face_detected, face_count, screenshot_path,
                    video_path, evidence_hash, fine_amount, severity_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                incident_uuid,
                incident_data.get('incident_type', 'unknown'),
                incident_data.get('waste_category', 'unknown'),
                incident_data.get('material_type', 'unknown'),
                incident_data.get('confidence_score', 0.0),
                incident_data.get('throwing_confidence', 0.0),
                incident_data.get('location_description', ''),
                str(incident_data.get('camera_id', 0)),
                incident_data.get('face_detected', False),
                incident_data.get('face_count', 0),
                incident_data.get('screenshot_path', ''),
                incident_data.get('video_path', ''),
                incident_data.get('evidence_hash', ''),
                incident_data.get('fine_amount', 0.0),
                incident_data.get('severity_level', 'medium')
            ))
            
            incident_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            # Update daily statistics
            self.update_daily_stats(incident_data)
            
            return incident_id, incident_uuid
            
        except Exception as e:
            logger.error(f"Error adding incident: {e}")
            return None, None
    
    def add_vehicle_detection(self, incident_id, vehicle_data):
        """Add vehicle/license plate detection to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO detected_vehicles (
                    incident_id, plate_number, plate_confidence, plate_is_valid,
                    vehicle_type, vehicle_color, plate_image_path, camera_id,
                    bbox_coordinates, plate_region
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                incident_id,
                vehicle_data.get('plate_number', ''),
                vehicle_data.get('plate_confidence', 0.0),
                vehicle_data.get('plate_is_valid', False),
                vehicle_data.get('vehicle_type', 'unknown'),
                vehicle_data.get('vehicle_color', 'unknown'),
                vehicle_data.get('plate_image_path', ''),
                str(vehicle_data.get('camera_id', 0)),
                json.dumps(vehicle_data.get('bbox_coordinates', [])),
                vehicle_data.get('plate_region', 'unknown')
            ))
            
            vehicle_id = cursor.lastrowid
            
            # Check if this plate is on any alert lists
            plate_number = vehicle_data.get('plate_number', '')
            if plate_number:
                cursor.execute('''
                    SELECT alert_type, alert_reason, priority_level 
                    FROM plate_alerts 
                    WHERE plate_number = ? AND active = 1 
                    AND (expires_date IS NULL OR expires_date > datetime('now'))
                ''', (plate_number,))
                
                alerts = cursor.fetchall()
                if alerts:
                    logger.warning(f"ALERT: License plate {plate_number} is on alert list!")
                    for alert in alerts:
                        logger.warning(f"Alert type: {alert[0]}, Reason: {alert[1]}, Priority: {alert[2]}")
            
            conn.commit()
            conn.close()
            
            return vehicle_id
            
        except Exception as e:
            logger.error(f"Error adding vehicle detection: {e}")
            return None
    
    def check_plate_alerts(self, plate_number):
        """Check if a license plate is on any alert lists"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT alert_type, alert_reason, priority_level, created_date
                FROM plate_alerts 
                WHERE plate_number = ? AND active = 1 
                AND (expires_date IS NULL OR expires_date > datetime('now'))
                ORDER BY priority_level DESC
            ''', (plate_number,))
            
            alerts = cursor.fetchall()
            conn.close()
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking plate alerts: {e}")
            return []
    
    def update_daily_stats(self, incident_data):
        """Update daily statistics including vehicle data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            today = datetime.now().date()
            waste_category = incident_data.get('waste_category', 'unknown')
            fine_amount = incident_data.get('fine_amount', 0.0)
            
            # Insert or update daily stats
            cursor.execute('''
                INSERT OR IGNORE INTO daily_stats (date) VALUES (?)
            ''', (today,))
            
            cursor.execute('''
                UPDATE daily_stats SET 
                    total_incidents = total_incidents + 1,
                    total_fines = total_fines + ?
                WHERE date = ?
            ''', (fine_amount, today))
            
            # Update category-specific count
            if waste_category in ['plastic', 'metal', 'glass', 'paper', 'organic']:
                cursor.execute(f'''
                    UPDATE daily_stats SET 
                        {waste_category}_incidents = {waste_category}_incidents + 1
                    WHERE date = ?
                ''', (today,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Daily stats update error: {e}")
    
    def update_vehicle_stats(self, plates_detected=0, valid_plates=0):
        """Update daily vehicle detection statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            today = datetime.now().date()
            
            cursor.execute('''
                INSERT OR IGNORE INTO daily_stats (date) VALUES (?)
            ''', (today,))
            
            cursor.execute('''
                UPDATE daily_stats SET 
                    vehicles_detected = vehicles_detected + 1,
                    plates_read = plates_read + ?,
                    valid_plates = valid_plates + ?
                WHERE date = ?
            ''', (plates_detected, valid_plates, today))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Vehicle stats update error: {e}")
    
    def get_incidents(self, limit=100, status=None):
        """Get incidents from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM waste_incidents"
            params = []
            
            if status:
                query += " WHERE status = ?"
                params.append(status)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            incidents = cursor.fetchall()
            conn.close()
            
            return incidents
            
        except Exception as e:
            logger.error(f"Error getting incidents: {e}")
            return []
    
    def get_vehicle_detections(self, incident_id=None, limit=100):
        """Get vehicle detections"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if incident_id:
                cursor.execute('''
                    SELECT * FROM detected_vehicles 
                    WHERE incident_id = ? 
                    ORDER BY detection_timestamp DESC
                ''', (incident_id,))
            else:
                cursor.execute('''
                    SELECT * FROM detected_vehicles 
                    ORDER BY detection_timestamp DESC 
                    LIMIT ?
                ''', (limit,))
            
            vehicles = cursor.fetchall()
            conn.close()
            
            return vehicles
            
        except Exception as e:
            logger.error(f"Error getting vehicle detections: {e}")
            return []
    
    def get_daily_stats(self, date=None):
        """Get daily statistics including vehicle data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if date is None:
                date = datetime.now().date()
            
            cursor.execute("SELECT * FROM daily_stats WHERE date = ?", (date,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'total_incidents': result[1],
                    'plastic_incidents': result[2],
                    'metal_incidents': result[3],
                    'glass_incidents': result[4],
                    'paper_incidents': result[5],
                    'organic_incidents': result[6],
                    'total_fines': result[7],
                    'vehicles_detected': result[8] if len(result) > 8 else 0,
                    'plates_read': result[9] if len(result) > 9 else 0,
                    'valid_plates': result[10] if len(result) > 10 else 0
                }
            else:
                return {
                    'total_incidents': 0,
                    'plastic_incidents': 0,
                    'metal_incidents': 0,
                    'glass_incidents': 0,
                    'paper_incidents': 0,
                    'organic_incidents': 0,
                    'total_fines': 0.0,
                    'vehicles_detected': 0,
                    'plates_read': 0,
                    'valid_plates': 0
                }
                
        except Exception as e:
            logger.error(f"Daily stats error: {e}")
            return {}

class NotificationService:
    """Notification service for municipal alerts including vehicle alerts"""
    
    def __init__(self):
        self.email_enabled = EMAIL_AVAILABLE
        self.webhook_enabled = True
        logger.info("Notification service initialized")
        
    def send_incident_alert(self, incident_data):
        """Send alert for waste incident"""
        try:
            confidence = incident_data.get('confidence_score', 0.0)
            if confidence >= 0.8:
                logger.info(f"High priority alert: {incident_data.get('incident_uuid', 'unknown')}")
                # In a real system, this would send emails/SMS/webhooks
                
        except Exception as e:
            logger.error(f"Notification error: {e}")
    
    def send_vehicle_alert(self, vehicle_data, alert_type="vehicle_detected"):
        """Send alert for vehicle/license plate detection"""
        try:
            plate_number = vehicle_data.get('plate_number', 'UNKNOWN')
            camera_id = vehicle_data.get('camera_id', 'Unknown')
            
            if alert_type == "plate_alert":
                logger.warning(f"🚨 PLATE ALERT: {plate_number} detected at Camera {camera_id}")
            else:
                logger.info(f"🚗 Vehicle detected: {plate_number} at Camera {camera_id}")
                
        except Exception as e:
            logger.error(f"Vehicle notification error: {e}")

class MunicipalWasteDetectionSystem:
    """Main municipal waste detection system with vehicle tracking"""
    
    def __init__(self):
        # Load configuration
        self.config_manager = ConfigManager()
        
        # Core components
        self.db_manager = MunicipalDatabase()
        self.waste_detector = AdvancedWasteDetector()
        self.throwing_detector = AdvancedThrowingDetector()
        self.recording_system = InstantRecordingSystem()
        self.notification_service = NotificationService()
        
        # Vehicle/License plate detection
        self.plate_detector = NumberPlateDetector()
        self.plate_detection_enabled = self.config_manager.get('detection_settings.plate_detection_enabled', True)
        
        # Camera management
        self.cameras = {}
        self.available_cameras = self.detect_cameras()
        self.current_camera_index = 0
        
        # System state
        self.is_monitoring = False
        self.detection_active = True
        
        # Performance metrics
        self.performance_metrics = {
            'detection_accuracy': 0.85,
            'false_positive_rate': 0.05,
            'average_processing_time': 0.03,
            'current_fps': 0.0,
            'plate_detection_rate': 0.0
        }
        
        # Statistics
        self.daily_stats = defaultdict(int)
        self.recent_incidents = deque(maxlen=100)
        self.recent_vehicles = deque(maxlen=50)
        
        # FPS calculation
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        logger.info("Municipal Waste Detection System with Vehicle Tracking initialized")
    
    def detect_cameras(self):
        """Detect available cameras"""
        available_cameras = []
        
        # Suppress OpenCV camera warnings temporarily
        old_level = cv2.getLogLevel()
        cv2.setLogLevel(0)
        
        try:
            for i in range(6):  # Check first 6 camera indices
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                            available_cameras.append({
                                'index': i,
                                'name': f"Camera {i}",
                                'resolution': f"{width}x{height}",
                                'status': 'available'
                            })
                            
                            logger.info(f"Found camera {i}: {width}x{height}")
                    
                    cap.release()
                    
                except Exception as e:
                    logger.debug(f"Camera {i} check failed: {e}")
        finally:
            cv2.setLogLevel(old_level)
        
        if not available_cameras:
            # Add a dummy camera for testing
            available_cameras.append({
                'index': 0,
                'name': "Test Camera (Simulated)",
                'resolution': "640x480",
                'status': 'simulated'
            })
            logger.info("No real cameras found, using simulated camera for testing")
        
        return available_cameras
    
    def start_monitoring(self):
        """Start monitoring system"""
        try:
            if not self.available_cameras:
                return False, "No cameras available"
            
            if not self.start_camera(self.current_camera_index):
                return False, f"Failed to start camera {self.current_camera_index}"
            
            self.is_monitoring = True
            logger.info("Municipal monitoring with vehicle tracking started")
            return True, "Monitoring started successfully"
            
        except Exception as e:
            logger.error(f"Start monitoring error: {e}")
            return False, f"Failed to start monitoring: {str(e)}"
    
    def start_camera(self, camera_index):
        """Start specific camera"""
        try:
            if camera_index in self.cameras:
                return True
            
            # Check if this is a simulated camera
            camera_info = next((cam for cam in self.available_cameras if cam['index'] == camera_index), None)
            if camera_info and camera_info['status'] == 'simulated':
                # Create simulated camera
                self.cameras[camera_index] = {
                    'camera': None,  # No real camera
                    'last_frame': self.create_test_frame(),
                    'total_incidents': 0,
                    'simulated': True
                }
                logger.info(f"Simulated camera {camera_index} started")
                return True
            
            # Try to open real camera
            camera = cv2.VideoCapture(camera_index)
            if not camera.isOpened():
                return False
            
            # Set camera properties
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Test frame
            ret, frame = camera.read()
            if ret and frame is not None:
                self.cameras[camera_index] = {
                    'camera': camera,
                    'last_frame': frame,
                    'total_incidents': 0,
                    'simulated': False
                }
                
                logger.info(f"Camera {camera_index} started")
                return True
            
            camera.release()
            return False
            
        except Exception as e:
            logger.error(f"Camera start error: {e}")
            return False
    
    def create_test_frame(self):
        """Create a test frame for simulation"""
        try:
            # Create a simple test frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (50, 50, 50)  # Dark gray background
            
            # Add some text
            cv2.putText(frame, "SIMULATED CAMERA FEED", (150, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Connect real camera for live detection", (100, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, "Waste + Vehicle Detection Ready", (120, 280),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            return frame
        except Exception as e:
            logger.error(f"Test frame creation error: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def stop_monitoring(self):
        """Stop monitoring system"""
        try:
            self.is_monitoring = False
            self.recording_system.stop_all_recordings()
            
            for camera_index in list(self.cameras.keys()):
                if camera_index in self.cameras:
                    camera_data = self.cameras[camera_index]
                    if not camera_data.get('simulated', False) and camera_data['camera']:
                        camera_data['camera'].release()
                    del self.cameras[camera_index]
            
            logger.info("Monitoring stopped")
            
        except Exception as e:
            logger.error(f"Stop monitoring error: {e}")
    
    def get_frame(self, camera_index):
        """Get frame from camera"""
        if camera_index not in self.cameras or not self.is_monitoring:
            return None
        
        try:
            camera_data = self.cameras[camera_index]
            
            if camera_data.get('simulated', False):
                # Return updated test frame
                frame = self.create_test_frame()
            else:
                camera = camera_data['camera']
                ret, frame = camera.read()
                
                if not ret or frame is None:
                    return None
            
            self.cameras[camera_index]['last_frame'] = frame
            
            # Update FPS
            self.fps_counter += 1
            current_time = time.time()
            if current_time - self.fps_start_time >= 1.0:
                self.performance_metrics['current_fps'] = self.fps_counter / (current_time - self.fps_start_time)
                self.fps_counter = 0
                self.fps_start_time = current_time
            
            # Process frame
            processed_frame, incidents = self.process_frame_municipal(camera_index, frame)
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            return None
    
    def process_frame_municipal(self, camera_id, frame):
        """Process frame for municipal detection with vehicle tracking"""
        try:
            if not self.detection_active or frame is None:
                return frame, []
            
            processed_frame = frame.copy()
            detected_incidents = []
            
            # Add frame to recording buffer
            self.recording_system.add_frame_to_buffer(frame)
            
            # Detect waste objects
            processed_frame, waste_objects = self.waste_detector.detect_waste_objects(processed_frame)
            
            # Detect license plates if enabled
            detected_plates = []
            if self.plate_detection_enabled:
                processed_frame, detected_plates = self.plate_detector.detect_license_plates(processed_frame)
                
                # Process plate detections
                for plate in detected_plates:
                    if plate.get('is_valid', False):
                        # Save plate capture
                        plate_image_path = self.recording_system.save_plate_capture(frame, plate, camera_id)
                        
                        # Check for alerts
                        plate_alerts = self.db_manager.check_plate_alerts(plate.get('text', ''))
                        if plate_alerts:
                            self.notification_service.send_vehicle_alert(plate, "plate_alert")
                        
                        # Add to recent vehicles
                        vehicle_detection = {
                            'timestamp': datetime.now(),
                            'plate_number': plate.get('text', 'UNKNOWN'),
                            'confidence': plate.get('confidence', 0.0),
                            'camera_id': camera_id,
                            'is_valid': plate.get('is_valid', False),
                            'has_alerts': len(plate_alerts) > 0,
                            'image_path': plate_image_path
                        }
                        self.recent_vehicles.append(vehicle_detection)
                        
                        # Update vehicle stats
                        self.db_manager.update_vehicle_stats(
                            plates_detected=1, 
                            valid_plates=1 if plate.get('is_valid', False) else 0
                        )
            
            # Detect throwing behavior
            throwing_incidents = self.throwing_detector.detect_throwing_motion(
                waste_objects, frame, time.time()
            )
            
            # Process incidents
            for throwing_incident in throwing_incidents:
                if throwing_incident['confidence'] > 0.6:
                    # Create incident record
                    incident_data = self.create_incident_record(
                        throwing_incident, waste_objects, camera_id, frame
                    )
                    
                    # Add to database
                    incident_id, incident_uuid = self.db_manager.add_incident(incident_data)
                    
                    if incident_id:
                        # Link detected vehicles to incident
                        for plate in detected_plates:
                            if plate.get('is_valid', False):
                                vehicle_data = {
                                    'plate_number': plate.get('text', ''),
                                    'plate_confidence': plate.get('confidence', 0.0),
                                    'plate_is_valid': plate.get('is_valid', False),
                                    'camera_id': camera_id,
                                    'bbox_coordinates': plate.get('bbox', []),
                                    'plate_image_path': self.recording_system.save_plate_capture(frame, plate, camera_id)
                                }
                                self.db_manager.add_vehicle_detection(incident_id, vehicle_data)
                        
                        # Start recording
                        video_path, image_path = self.recording_system.start_incident_recording(
                            camera_id, incident_data['waste_category'], frame
                        )
                        
                        # Send notifications
                        self.notification_service.send_incident_alert(incident_data)
                        
                        incident_summary = {
                            'incident_id': incident_id,
                            'incident_uuid': incident_uuid,
                            'timestamp': datetime.now(),
                            'confidence': throwing_incident['confidence'],
                            'waste_category': incident_data['waste_category'],
                            'material_type': incident_data['material_type'],
                            'fine_amount': incident_data['fine_amount'],
                            'severity': incident_data['severity_level'],
                            'camera_id': camera_id,
                            'vehicle_count': len([p for p in detected_plates if p.get('is_valid', False)]),
                            'plate_numbers': [p.get('text', '') for p in detected_plates if p.get('is_valid', False)]
                        }
                        
                        self.recent_incidents.append(incident_summary)
                        detected_incidents.append(incident_summary)
                        
                        logger.warning(f"🚨 INCIDENT: {incident_uuid} - {incident_data['waste_category']} - ${incident_data['fine_amount']:.2f}")
                        if incident_summary['vehicle_count'] > 0:
                            logger.warning(f"🚗 Associated vehicles: {', '.join(incident_summary['plate_numbers'])}")
            
            # Update recordings
            self.recording_system.update_recordings(camera_id, frame)
            
            # Add overlays
            self.add_municipal_overlays(processed_frame, camera_id, waste_objects, detected_plates, detected_incidents)
            
            return processed_frame, detected_incidents
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame, []
    
    def create_incident_record(self, throwing_incident, waste_objects, camera_id, frame):
        """Create incident record"""
        try:
            primary_object = waste_objects[0] if waste_objects else {}
            
            # Calculate fine
            base_fine = primary_object.get('fine_amount', 100.0)
            severity_multiplier = {
                'low': 0.5, 'medium': 1.0, 'high': 1.5, 
                'very_high': 2.0, 'critical': 3.0
            }.get(throwing_incident['severity'], 1.0)
            
            fine_amount = base_fine * severity_multiplier
            
            # Generate evidence hash
            evidence_string = f"{time.time()}{camera_id}{throwing_incident['confidence']}"
            evidence_hash = hashlib.sha256(evidence_string.encode()).hexdigest()
            
            incident_data = {
                'incident_type': 'waste_throwing',
                'waste_category': primary_object.get('waste_category', 'plastic'),
                'material_type': primary_object.get('material', 'plastic'),
                'confidence_score': throwing_incident['confidence'],
                'throwing_confidence': throwing_incident['confidence'],
                'camera_id': str(camera_id),
                'location_description': f"Camera {camera_id}",
                'face_detected': False,
                'face_count': 0,
                'evidence_hash': evidence_hash,
                'fine_amount': fine_amount,
                'severity_level': throwing_incident['severity']
            }
            
            return incident_data
            
        except Exception as e:
            logger.error(f"Incident record creation error: {e}")
            return {}
    
    def add_municipal_overlays(self, frame, camera_id, waste_objects, detected_plates, incidents):
        """Add municipal overlays to frame including vehicle information"""
        try:
            if frame is None or frame.size == 0:
                return
                
            height, width = frame.shape[:2]
            
            # Header
            cv2.rectangle(frame, (0, 0), (width, 80), (0, 0, 0), -1)
            cv2.putText(frame, "MUNICIPAL WASTE ENFORCEMENT + VEHICLE TRACKING", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Status
            status_text = "ACTIVE MONITORING" if self.is_monitoring else "STANDBY"
            status_color = (0, 255, 0) if self.is_monitoring else (0, 0, 255)
            cv2.putText(frame, status_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            
            # Plate detection status
            plate_status = "PLATE DETECTION: ON" if self.plate_detection_enabled else "PLATE DETECTION: OFF"
            cv2.putText(frame, plate_status, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Camera info
            cv2.putText(frame, f"Camera {camera_id}", (width - 150, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # FPS
            fps_text = f"FPS: {self.performance_metrics['current_fps']:.1f}"
            cv2.putText(frame, fps_text, (width - 150, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Statistics
            valid_plates = len([p for p in detected_plates if p.get('is_valid', False)])
            stats_text = f"Objects: {len(waste_objects)} | Plates: {len(detected_plates)} ({valid_plates} valid) | Today: {self.daily_stats.get('total_incidents', 0)}"
            cv2.putText(frame, stats_text, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Active incidents
            if incidents:
                incident_text = f"🚨 {len(incidents)} ACTIVE INCIDENT(S)"
                cv2.putText(frame, incident_text, (10, height - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Show vehicle info for incidents
                for i, incident in enumerate(incidents[:2]):  # Show first 2 incidents
                    if incident.get('vehicle_count', 0) > 0:
                        vehicle_text = f"Vehicle(s): {', '.join(incident['plate_numbers'][:2])}"
                        cv2.putText(frame, vehicle_text, (10, height - 35 + i * 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Show recent valid plates
            if valid_plates > 0:
                recent_plates = [p.get('text', 'UNKNOWN') for p in detected_plates if p.get('is_valid', False)][:3]
                plates_text = f"Recent Plates: {', '.join(recent_plates)}"
                cv2.putText(frame, plates_text, (10, height - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
            
            # Recording indicator
            if camera_id in self.recording_system.active_recordings:
                cv2.circle(frame, (width - 30, 65), 10, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (width - 60, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
            
            # Timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (width - 200, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
        except Exception as e:
            logger.error(f"Overlay error: {e}")
    
    def get_system_statistics(self):
        """Get system statistics including vehicle data"""
        try:
            daily_stats = self.db_manager.get_daily_stats()
            
            return {
                'monitoring_status': self.is_monitoring,
                'cameras_active': len(self.cameras),
                'current_fps': self.performance_metrics['current_fps'],
                'detection_accuracy': self.performance_metrics['detection_accuracy'],
                'active_recordings': len(self.recording_system.active_recordings),
                'plate_detection_enabled': self.plate_detection_enabled,
                'recent_vehicles_count': len(self.recent_vehicles),
                **daily_stats
            }
            
        except Exception as e:
            logger.error(f"Statistics error: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup system resources"""
        try:
            self.stop_monitoring()
            self.recording_system.stop_all_recordings()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

class MunicipalWasteDetectionGUI:
    """Municipal GUI interface with vehicle tracking"""
    
    def __init__(self):
        # Initialize user info FIRST
        self.current_user = "Administrator"
        self.user_permissions = ["view", "operate", "review", "admin"]
        
        # Initialize system
        self.system = MunicipalWasteDetectionSystem()
        
        # Initialize GUI
        self.root = tk.Tk()
        self.setup_gui()
        
        # Threading
        self.update_thread = None
        self.is_updating = False
        
        logger.info("Municipal GUI with vehicle tracking initialized")
        
    def setup_gui(self):
        """Setup GUI"""
        self.root.title("Municipal Waste Enforcement System v2.1 - With Vehicle Tracking")
        self.root.geometry("1800x1100")
        self.root.configure(bg='#f0f0f0')
        
        self.create_main_layout()
        self.create_video_panel()
        self.create_control_panel()
        self.create_statistics_panel()
        self.create_incidents_panel()
        self.create_vehicles_panel()
        self.create_status_bar()
        
        self.update_displays()
    
    def create_main_layout(self):
        """Create main layout"""
        # Title bar
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="🏛️ MUNICIPAL WASTE ENFORCEMENT + 🚗 VEHICLE TRACKING", 
                bg='#2c3e50', fg='white', font=('Arial', 16, 'bold')).pack(side='left', padx=20, pady=15)
        
        self.system_status_label = tk.Label(title_frame, text="● OFFLINE", 
                                           bg='#2c3e50', fg='#e74c3c', font=('Arial', 12, 'bold'))
        self.system_status_label.pack(side='right', padx=20, pady=15)
        
        # Main content
        self.main_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel (video)
        self.left_panel = tk.Frame(self.main_frame, bg='#2c3e50', relief='raised', bd=2)
        self.left_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Right panel (controls and stats)
        self.right_panel = tk.Frame(self.main_frame, bg='#2c3e50', width=450, relief='raised', bd=2)
        self.right_panel.pack(side='right', fill='y', padx=(5, 0))
        self.right_panel.pack_propagate(False)
        
        # Bottom panel (incidents and vehicles)
        self.bottom_panel = tk.Frame(self.root, bg='#2c3e50', height=300, relief='raised', bd=2)
        self.bottom_panel.pack(fill='x', padx=10, pady=(0, 10))
        self.bottom_panel.pack_propagate(False)
    
    def create_video_panel(self):
        """Create video display panel"""
        # Video header
        video_header = tk.Frame(self.left_panel, bg='#2c3e50', height=40)
        video_header.pack(fill='x', padx=5, pady=5)
        video_header.pack_propagate(False)
        
        tk.Label(video_header, text="📹 LIVE MONITORING + 🚗 VEHICLE DETECTION", 
                bg='#2c3e50', fg='white', font=('Arial', 12, 'bold')).pack(side='left', pady=10)
        
        # Camera selection
        tk.Label(video_header, text="Camera:", bg='#2c3e50', fg='white').pack(side='right', padx=(20, 5), pady=10)
        
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(video_header, textvariable=self.camera_var, state='readonly', width=15)
        self.camera_combo.pack(side='right', pady=10)
        self.camera_combo.bind('<<ComboboxSelected>>', self.on_camera_changed)
        
        # Video display
        self.video_canvas = tk.Canvas(self.left_panel, bg='black')
        self.video_canvas.pack(fill='both', expand=True, padx=5, pady=(0, 5))
        
        self.update_camera_list()
    
    def create_control_panel(self):
        """Create control panel"""
        control_header = tk.Frame(self.right_panel, bg='#2c3e50', height=40)
        control_header.pack(fill='x', padx=5, pady=5)
        control_header.pack_propagate(False)
        
        tk.Label(control_header, text="⚙️ CONTROLS", 
                bg='#2c3e50', fg='white', font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Control buttons
        controls_frame = tk.Frame(self.right_panel, bg='#2c3e50')
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        self.start_btn = tk.Button(controls_frame, text="▶️ START MONITORING", 
                                  command=self.start_monitoring, bg='#27ae60', fg='white', 
                                  font=('Arial', 10, 'bold'), height=2)
        self.start_btn.pack(fill='x', pady=2)
        
        self.stop_btn = tk.Button(controls_frame, text="⏹️ STOP MONITORING", 
                                 command=self.stop_monitoring, bg='#e74c3c', fg='white', 
                                 font=('Arial', 10, 'bold'), height=2)
        self.stop_btn.pack(fill='x', pady=2)
        
        self.emergency_btn = tk.Button(controls_frame, text="🚨 EMERGENCY ALERT",
                                       command=self.emergency_alert, bg='#c0392b', fg='white', 
                                      font=('Arial', 10, 'bold'))
        self.emergency_btn.pack(fill='x', pady=5)
        
        # Detection settings
        settings_frame = tk.LabelFrame(self.right_panel, text="Detection Settings", 
                                      bg='#2c3e50', fg='white', font=('Arial', 10, 'bold'))
        settings_frame.pack(fill='x', padx=10, pady=10)
        
        # Waste detection sensitivity
        tk.Label(settings_frame, text="Waste Detection Sensitivity:", 
                bg='#2c3e50', fg='white').pack(anchor='w', padx=5)
        
        self.sensitivity_var = tk.IntVar(value=75)
        self.sensitivity_scale = tk.Scale(settings_frame, from_=30, to=100, orient='horizontal', 
                                        variable=self.sensitivity_var, bg='#2c3e50', fg='white')
        self.sensitivity_scale.pack(fill='x', padx=5, pady=2)
        
        # Plate detection toggle
        self.plate_detection_var = tk.BooleanVar(value=self.system.plate_detection_enabled)
        tk.Checkbutton(settings_frame, text="Enable License Plate Detection", 
                      variable=self.plate_detection_var, bg='#2c3e50', fg='white',
                      selectcolor='#2c3e50', activebackground='#2c3e50',
                      command=self.toggle_plate_detection).pack(anchor='w', padx=5, pady=5)
        
        # Vehicle tracking settings
        vehicle_frame = tk.LabelFrame(self.right_panel, text="Vehicle Tracking", 
                                     bg='#2c3e50', fg='white', font=('Arial', 10, 'bold'))
        vehicle_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(vehicle_frame, text="🚗 View Vehicle Log", command=self.view_vehicle_log,
                 bg='#3498db', fg='white', font=('Arial', 9, 'bold')).pack(fill='x', padx=5, pady=2)
        
        tk.Button(vehicle_frame, text="🔍 Search Plate", command=self.search_plate,
                 bg='#9b59b6', fg='white', font=('Arial', 9, 'bold')).pack(fill='x', padx=5, pady=2)
        
        tk.Button(vehicle_frame, text="⚠️ Plate Alerts", command=self.manage_plate_alerts,
                 bg='#e67e22', fg='white', font=('Arial', 9, 'bold')).pack(fill='x', padx=5, pady=2)
    
    def create_statistics_panel(self):
        """Create statistics panel with vehicle data"""
        stats_header = tk.Frame(self.right_panel, bg='#2c3e50', height=40)
        stats_header.pack(fill='x', padx=5, pady=(20, 5))
        stats_header.pack_propagate(False)
        
        tk.Label(stats_header, text="📊 STATISTICS", 
                bg='#2c3e50', fg='white', font=('Arial', 11, 'bold')).pack(pady=10)
        
        # Statistics display
        stats_frame = tk.Frame(self.right_panel, bg='#2c3e50')
        stats_frame.pack(fill='x', padx=10)
        
        self.stats_labels = {}
        
        stats_data = [
            ("Total Incidents", "total_incidents", "#e74c3c"),
            ("Plastic Waste", "plastic_incidents", "#e67e22"),
            ("Metal Objects", "metal_incidents", "#95a5a6"),
            ("Glass Items", "glass_incidents", "#9b59b6"),
            ("Total Fines", "total_fines", "#2ecc71"),
            ("Vehicles Detected", "vehicles_detected", "#3498db"),
            ("Valid Plates", "valid_plates", "#f39c12"),
            ("Active Cameras", "cameras_active", "#1abc9c")
        ]
        
        for i, (label, key, color) in enumerate(stats_data):
            row = i // 2
            col = i % 2
            
            stat_frame = tk.Frame(stats_frame, bg='#34495e', relief='raised', bd=1)
            stat_frame.grid(row=row, column=col, padx=2, pady=2, sticky='ew')
            
            tk.Label(stat_frame, text=label, bg='#34495e', fg='white', 
                    font=('Arial', 8)).pack(pady=(2, 0))
            
            value_label = tk.Label(stat_frame, text="0", bg='#34495e', fg=color, 
                                  font=('Arial', 12, 'bold'))
            value_label.pack(pady=(0, 2))
            
            self.stats_labels[key] = value_label
        
        stats_frame.grid_columnconfigure(0, weight=1)
        stats_frame.grid_columnconfigure(1, weight=1)
    
    def create_incidents_panel(self):
        """Create incidents panel"""
        # Split bottom panel into two parts
        incidents_frame = tk.Frame(self.bottom_panel, bg='#2c3e50', width=900)
        incidents_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        incidents_header = tk.Frame(incidents_frame, bg='#2c3e50', height=30)
        incidents_header.pack(fill='x', pady=(0, 5))
        incidents_header.pack_propagate(False)
        
        tk.Label(incidents_header, text="🚨 RECENT INCIDENTS", 
                bg='#2c3e50', fg='white', font=('Arial', 12, 'bold')).pack(side='left', pady=5)
        
        tk.Button(incidents_header, text="🔄 Refresh", command=self.refresh_incidents,
                 bg='#3498db', fg='white', font=('Arial', 8, 'bold')).pack(side='right', pady=5)
        
        # Incidents table
        incidents_table_frame = tk.Frame(incidents_frame, bg='#2c3e50')
        incidents_table_frame.pack(fill='both', expand=True)
        
        columns = ('ID', 'Time', 'Camera', 'Type', 'Material', 'Confidence', 'Fine', 'Vehicles', 'Status')
        self.incidents_tree = ttk.Treeview(incidents_table_frame, columns=columns, show='headings', height=8)
        
        # Configure columns
        column_widths = {'ID': 60, 'Time': 80, 'Camera': 60, 'Type': 80, 'Material': 70, 
                        'Confidence': 70, 'Fine': 60, 'Vehicles': 80, 'Status': 80}
        
        for col in columns:
            self.incidents_tree.heading(col, text=col)
            self.incidents_tree.column(col, width=column_widths.get(col, 80))
        
        # Scrollbar
        incidents_scrollbar = ttk.Scrollbar(incidents_table_frame, orient="vertical", command=self.incidents_tree.yview)
        self.incidents_tree.configure(yscrollcommand=incidents_scrollbar.set)
        
        self.incidents_tree.pack(side='left', fill='both', expand=True)
        incidents_scrollbar.pack(side='right', fill='y')
        
        self.incidents_tree.bind('<Double-1>', self.view_incident_details)
    
    def create_vehicles_panel(self):
        """Create vehicles detection panel"""
        vehicles_frame = tk.Frame(self.bottom_panel, bg='#2c3e50', width=450)
        vehicles_frame.pack(side='right', fill='y', padx=5, pady=5)
        vehicles_frame.pack_propagate(False)
        
        vehicles_header = tk.Frame(vehicles_frame, bg='#2c3e50', height=30)
        vehicles_header.pack(fill='x', pady=(0, 5))
        vehicles_header.pack_propagate(False)
        
        tk.Label(vehicles_header, text="🚗 DETECTED VEHICLES", 
                bg='#2c3e50', fg='white', font=('Arial', 12, 'bold')).pack(side='left', pady=5)
        
        tk.Button(vehicles_header, text="🔄", command=self.refresh_vehicles,
                 bg='#3498db', fg='white', font=('Arial', 8, 'bold')).pack(side='right', pady=5)
        
        # Vehicles table
        vehicles_table_frame = tk.Frame(vehicles_frame, bg='#2c3e50')
        vehicles_table_frame.pack(fill='both', expand=True)
        
        vehicle_columns = ('Time', 'Plate', 'Confidence', 'Camera', 'Valid', 'Alert')
        self.vehicles_tree = ttk.Treeview(vehicles_table_frame, columns=vehicle_columns, show='headings', height=8)
        
        # Configure vehicle columns
        vehicle_widths = {'Time': 60, 'Plate': 80, 'Confidence': 70, 'Camera': 50, 'Valid': 50, 'Alert': 50}
        
        for col in vehicle_columns:
            self.vehicles_tree.heading(col, text=col)
            self.vehicles_tree.column(col, width=vehicle_widths.get(col, 60))
        
        # Vehicle scrollbar
        vehicles_scrollbar = ttk.Scrollbar(vehicles_table_frame, orient="vertical", command=self.vehicles_tree.yview)
        self.vehicles_tree.configure(yscrollcommand=vehicles_scrollbar.set)
        
        self.vehicles_tree.pack(side='left', fill='both', expand=True)
        vehicles_scrollbar.pack(side='right', fill='y')
        
        self.vehicles_tree.bind('<Double-1>', self.view_vehicle_details)
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = tk.Frame(self.root, bg='#34495e', height=25, relief='sunken', bd=1)
        self.status_bar.pack(fill='x', side='bottom')
        self.status_bar.pack_propagate(False)
        
        tk.Label(self.status_bar, text=f"User: {self.current_user}", 
                bg='#34495e', fg='white', font=('Arial', 8)).pack(side='left', padx=10, pady=2)
        
        # Plate detection status
        self.plate_status_label = tk.Label(self.status_bar, text="Plate Detection: ON", 
                                          bg='#34495e', fg='#27ae60', font=('Arial', 8))
        self.plate_status_label.pack(side='left', padx=20, pady=2)
        
        self.time_label = tk.Label(self.status_bar, text="", 
                                  bg='#34495e', fg='white', font=('Arial', 8))
        self.time_label.pack(side='right', padx=10, pady=2)
        
        self.update_time()
    
    def update_time(self):
        """Update time display"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.time_label.configure(text=current_time)
            self.root.after(1000, self.update_time)
        except Exception as e:
            logger.error(f"Time update error: {e}")
            self.root.after(1000, self.update_time)
    
    def toggle_plate_detection(self):
        """Toggle license plate detection"""
        try:
            self.system.plate_detection_enabled = self.plate_detection_var.get()
            status_text = "Plate Detection: ON" if self.system.plate_detection_enabled else "Plate Detection: OFF"
            status_color = '#27ae60' if self.system.plate_detection_enabled else '#e74c3c'
            self.plate_status_label.configure(text=status_text, fg=status_color)
            logger.info(f"Plate detection {'enabled' if self.system.plate_detection_enabled else 'disabled'}")
        except Exception as e:
            logger.error(f"Plate detection toggle error: {e}")
    
    def start_monitoring(self):
        """Start monitoring"""
        try:
            success, message = self.system.start_monitoring()
            
            if success:
                self.is_updating = True
                self.update_thread = threading.Thread(target=self.update_video_feed, daemon=True)
                self.update_thread.start()
                
                self.system_status_label.configure(text="● ONLINE", fg='#27ae60')
                messagebox.showinfo("System Started", "Municipal waste detection with vehicle tracking is now active!")
                
                logger.info("GUI monitoring started")
            else:
                messagebox.showerror("Start Error", f"Failed to start: {message}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Start error: {str(e)}")
            logger.error(f"GUI start error: {e}")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        try:
            if messagebox.askyesno("Stop Monitoring", "Stop the monitoring system?"):
                self.system.stop_monitoring()
                self.is_updating = False
                
                if self.update_thread:
                    self.update_thread.join(timeout=2)
                
                self.system_status_label.configure(text="● OFFLINE", fg='#e74c3c')
                
                # Clear video
                self.video_canvas.delete("all")
                self.video_canvas.create_text(
                    400, 300,
                    text="🏛️\nMUNICIPAL WASTE DETECTION\n+ VEHICLE TRACKING\nClick Start to begin monitoring",
                    fill='white', font=('Arial', 16), justify='center'
                )
                
                messagebox.showinfo("System Stopped", "Monitoring system stopped")
                logger.info("GUI monitoring stopped")
                
        except Exception as e:
            messagebox.showerror("Error", f"Stop error: {str(e)}")
            logger.error(f"GUI stop error: {e}")
    
    def update_video_feed(self):
        """Update video feed"""
        while self.is_updating and self.system.is_monitoring:
            try:
                camera_id = self.get_current_camera_id()
                frame = self.system.get_frame(camera_id)
                
                if frame is not None:
                    # Convert for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    
                    # Resize
                    canvas_width = self.video_canvas.winfo_width()
                    canvas_height = self.video_canvas.winfo_height()
                    
                    if canvas_width > 1 and canvas_height > 1:
                        frame_pil = frame_pil.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                        frame_tk = ImageTk.PhotoImage(frame_pil)
                        
                        self.root.after(0, self.update_video_display, frame_tk)
                        self.root.after(0, self.update_displays)
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Video feed error: {e}")
                time.sleep(0.1)
    
    def update_video_display(self, frame_tk):
        """Update video display"""
        try:
            self.video_canvas.delete("all")
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                self.video_canvas.create_image(canvas_width//2, canvas_height//2, image=frame_tk)
                self.video_canvas.image = frame_tk
                
        except Exception as e:
            logger.error(f"Display update error: {e}")
    
    def update_displays(self):
        """Update all displays"""
        try:
            stats = self.system.get_system_statistics()
            
            # Update statistics
            self.stats_labels['total_incidents'].config(text=str(stats.get('total_incidents', 0)))
            self.stats_labels['plastic_incidents'].config(text=str(stats.get('plastic_incidents', 0)))
            self.stats_labels['metal_incidents'].config(text=str(stats.get('metal_incidents', 0)))
            self.stats_labels['glass_incidents'].config(text=str(stats.get('glass_incidents', 0)))
            self.stats_labels['total_fines'].config(text=f"${stats.get('total_fines', 0):.0f}")
            self.stats_labels['vehicles_detected'].config(text=str(stats.get('vehicles_detected', 0)))
            self.stats_labels['valid_plates'].config(text=str(stats.get('valid_plates', 0)))
            self.stats_labels['cameras_active'].config(text=str(stats.get('cameras_active', 0)))
            
            # Update tables
            self.update_incidents_table()
            self.update_vehicles_table()
            
        except Exception as e:
            logger.error(f"Display update error: {e}")
    
    def update_incidents_table(self):
        """Update incidents table"""
        try:
            # Clear existing
            for item in self.incidents_tree.get_children():
                self.incidents_tree.delete(item)
            
            # Get recent incidents
            for incident in list(self.system.recent_incidents)[-20:]:
                vehicle_info = f"{incident.get('vehicle_count', 0)} vehicles"
                if incident.get('plate_numbers'):
                    vehicle_info = ', '.join(incident['plate_numbers'][:2])
                    if len(incident['plate_numbers']) > 2:
                        vehicle_info += f" +{len(incident['plate_numbers'])-2}"
                
                self.incidents_tree.insert('', 0, values=(
                    str(incident['incident_id'])[:6],
                    incident['timestamp'].strftime('%H:%M:%S'),
                    f"C{incident['camera_id']}",
                    incident['waste_category'].title(),
                    incident['material_type'].title(),
                    f"{incident['confidence']:.2f}",
                    f"${incident['fine_amount']:.0f}",
                    vehicle_info,
                    "Pending"
                ))
                
        except Exception as e:
            logger.error(f"Incidents table update error: {e}")
    
    def update_vehicles_table(self):
        """Update vehicles table"""
        try:
            # Clear existing
            for item in self.vehicles_tree.get_children():
                self.vehicles_tree.delete(item)
            
            # Get recent vehicles
            for vehicle in list(self.system.recent_vehicles)[-20:]:
                valid_icon = "✓" if vehicle.get('is_valid', False) else "✗"
                alert_icon = "⚠️" if vehicle.get('has_alerts', False) else "—"
                
                self.vehicles_tree.insert('', 0, values=(
                    vehicle['timestamp'].strftime('%H:%M:%S'),
                    vehicle.get('plate_number', 'UNKNOWN'),
                    f"{vehicle.get('confidence', 0):.2f}",
                    f"C{vehicle['camera_id']}",
                    valid_icon,
                    alert_icon
                ))
                
        except Exception as e:
            logger.error(f"Vehicles table update error: {e}")
    
    def get_current_camera_id(self):
        """Get current camera ID"""
        try:
            camera_str = self.camera_var.get()
            if camera_str and "Camera" in camera_str:
                return int(camera_str.split()[1])
        except:
            pass
        return 0
    
    def update_camera_list(self):
        """Update camera list"""
        try:
            camera_names = [f"Camera {cam['index']}" for cam in self.system.available_cameras]
            self.camera_combo['values'] = camera_names
            if camera_names:
                self.camera_combo.set(camera_names[0])
        except Exception as e:
            logger.error(f"Camera list error: {e}")
    
    def on_camera_changed(self, event=None):
        """Handle camera change"""
        try:
            camera_id = self.get_current_camera_id()
            self.system.current_camera_index = camera_id
            logger.info(f"Camera changed to {camera_id}")
        except Exception as e:
            logger.error(f"Camera change error: {e}")
    
    def emergency_alert(self):
        """Send emergency alert"""
        try:
            if messagebox.askyesno("Emergency Alert", "Send emergency alert?"):
                # Create emergency incident
                incident_data = {
                    'incident_type': 'emergency',
                    'waste_category': 'emergency',
                    'material_type': 'emergency',
                    'confidence_score': 1.0,
                    'camera_id': str(self.get_current_camera_id()),
                    'fine_amount': 0.0,
                    'severity_level': 'critical'
                }
                
                incident_id, incident_uuid = self.system.db_manager.add_incident(incident_data)
                if incident_uuid:
                    messagebox.showinfo("Alert Sent", f"Emergency alert sent! ID: {incident_uuid[:8]}")
                else:
                    messagebox.showerror("Error", "Failed to create emergency alert")
                
        except Exception as e:
            messagebox.showerror("Error", f"Emergency alert error: {str(e)}")
            logger.error(f"Emergency alert error: {e}")
    
    def view_vehicle_log(self):
        """View vehicle detection log"""
        try:
            log_window = tk.Toplevel(self.root)
            log_window.title("Vehicle Detection Log")
            log_window.geometry("800x600")
            log_window.configure(bg='#2c3e50')
            
            tk.Label(log_window, text="🚗 Vehicle Detection Log", 
                    bg='#2c3e50', fg='white', font=('Arial', 16, 'bold')).pack(pady=20)
            
            # Create detailed vehicle list
            log_frame = tk.Frame(log_window, bg='#2c3e50')
            log_frame.pack(fill='both', expand=True, padx=20, pady=20)
            
            columns = ('Timestamp', 'Plate Number', 'Confidence', 'Camera', 'Valid', 'Alerts')
            log_tree = ttk.Treeview(log_frame, columns=columns, show='headings', height=20)
            
            for col in columns:
                log_tree.heading(col, text=col)
                log_tree.column(col, width=120)
            
            # Add data
            vehicles = self.system.db_manager.get_vehicle_detections(limit=500)
            for vehicle in vehicles:
                log_tree.insert('', 'end', values=(
                    vehicle[8] if len(vehicle) > 8 else 'Unknown',  # timestamp
                    vehicle[1] if len(vehicle) > 1 else 'Unknown',  # plate_number
                    f"{vehicle[2]:.2f}" if len(vehicle) > 2 else '0.00',  # confidence
                    vehicle[9] if len(vehicle) > 9 else 'Unknown',  # camera_id
                    "Yes" if len(vehicle) > 3 and vehicle[3] else "No",  # is_valid
                    "Check" if len(vehicle) > 1 else "None"  # alerts (would need to check)
                ))
            
            scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=log_tree.yview)
            log_tree.configure(yscrollcommand=scrollbar.set)
            
            log_tree.pack(side='left', fill='both', expand=True)
            scrollbar.pack(side='right', fill='y')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open vehicle log: {str(e)}")
            logger.error(f"Vehicle log error: {e}")
    
    def search_plate(self):
        """Search for specific license plate"""
        try:
            search_window = tk.Toplevel(self.root)
            search_window.title("License Plate Search")
            search_window.geometry("500x400")
            search_window.configure(bg='#2c3e50')
            
            tk.Label(search_window, text="🔍 License Plate Search", 
                    bg='#2c3e50', fg='white', font=('Arial', 16, 'bold')).pack(pady=20)
            
            # Search input
            search_frame = tk.Frame(search_window, bg='#2c3e50')
            search_frame.pack(pady=20)
            
            tk.Label(search_frame, text="Enter Plate Number:", 
                    bg='#2c3e50', fg='white', font=('Arial', 12)).pack(pady=5)
            
            self.search_entry = tk.Entry(search_frame, font=('Arial', 14), width=20)
            self.search_entry.pack(pady=5)
            
            tk.Button(search_frame, text="🔍 Search", command=self.perform_plate_search,
                     bg='#3498db', fg='white', font=('Arial', 12, 'bold')).pack(pady=10)
            
            # Results area
            self.search_results = tk.Text(search_window, bg='#34495e', fg='white', 
                                         font=('Arial', 10), width=60, height=15)
            self.search_results.pack(fill='both', expand=True, padx=20, pady=20)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open search window: {str(e)}")
            logger.error(f"Plate search window error: {e}")
    
    def perform_plate_search(self):
        """Perform the actual plate search"""
        try:
            plate_number = self.search_entry.get().upper().strip()
            if not plate_number:
                messagebox.showwarning("Warning", "Please enter a plate number to search")
                return
            
            # Clear previous results
            self.search_results.delete('1.0', tk.END)
            
            # Search in database
            vehicles = self.system.db_manager.get_vehicle_detections(limit=1000)
            matches = [v for v in vehicles if v[1] and plate_number in v[1].upper()]
            
            if matches:
                self.search_results.insert(tk.END, f"Found {len(matches)} detection(s) for plate: {plate_number}\n\n")
                
                for i, match in enumerate(matches[:10], 1):  # Show first 10 matches
                    timestamp = match[8] if len(match) > 8 else 'Unknown'
                    camera = match[9] if len(match) > 9 else 'Unknown'
                    confidence = match[2] if len(match) > 2 else 0
                    
                    self.search_results.insert(tk.END, f"{i}. Detection on {timestamp}\n")
                    self.search_results.insert(tk.END, f"   Camera: {camera}\n")
                    self.search_results.insert(tk.END, f"   Confidence: {confidence:.2f}\n")
                    
                    # Check for alerts
                    alerts = self.system.db_manager.check_plate_alerts(plate_number)
                    if alerts:
                        self.search_results.insert(tk.END, f"   ⚠️ ALERTS: {len(alerts)} active alert(s)\n")
                    
                    self.search_results.insert(tk.END, "\n")
                
                if len(matches) > 10:
                    self.search_results.insert(tk.END, f"... and {len(matches) - 10} more detections\n")
            else:
                self.search_results.insert(tk.END, f"No detections found for plate: {plate_number}\n")
                self.search_results.insert(tk.END, "This plate has not been detected by the system.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Search failed: {str(e)}")
            logger.error(f"Plate search error: {e}")
    
    def manage_plate_alerts(self):
        """Manage license plate alerts"""
        try:
            alert_window = tk.Toplevel(self.root)
            alert_window.title("License Plate Alerts Management")
            alert_window.geometry("700x500")
            alert_window.configure(bg='#2c3e50')
            
            tk.Label(alert_window, text="⚠️ License Plate Alerts Management", 
                    bg='#2c3e50', fg='white', font=('Arial', 16, 'bold')).pack(pady=20)
            
            # Add new alert section
            add_frame = tk.LabelFrame(alert_window, text="Add New Alert", 
                                     bg='#2c3e50', fg='white', font=('Arial', 12, 'bold'))
            add_frame.pack(fill='x', padx=20, pady=10)
            
            # Input fields
            input_frame = tk.Frame(add_frame, bg='#2c3e50')
            input_frame.pack(fill='x', padx=10, pady=10)
            
            tk.Label(input_frame, text="Plate Number:", bg='#2c3e50', fg='white').grid(row=0, column=0, sticky='w')
            self.alert_plate_entry = tk.Entry(input_frame, width=15)
            self.alert_plate_entry.grid(row=0, column=1, padx=5)
            
            tk.Label(input_frame, text="Alert Type:", bg='#2c3e50', fg='white').grid(row=0, column=2, sticky='w', padx=(20,0))
            self.alert_type_var = tk.StringVar(value="wanted")
            alert_type_combo = ttk.Combobox(input_frame, textvariable=self.alert_type_var,
                                           values=["wanted", "repeat_offender", "stolen", "alert"],
                                           state='readonly', width=15)
            alert_type_combo.grid(row=0, column=3, padx=5)
            
            tk.Label(input_frame, text="Reason:", bg='#2c3e50', fg='white').grid(row=1, column=0, sticky='w', pady=(10,0))
            self.alert_reason_entry = tk.Entry(input_frame, width=40)
            self.alert_reason_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=(10,0), sticky='ew')
            
            tk.Button(input_frame, text="Add Alert", command=self.add_plate_alert,
                     bg='#e74c3c', fg='white', font=('Arial', 10, 'bold')).grid(row=1, column=3, padx=5, pady=(10,0))
            
            # Current alerts list
            alerts_frame = tk.LabelFrame(alert_window, text="Current Alerts", 
                                        bg='#2c3e50', fg='white', font=('Arial', 12, 'bold'))
            alerts_frame.pack(fill='both', expand=True, padx=20, pady=10)
            
            # Alerts table
            alerts_columns = ('Plate', 'Type', 'Reason', 'Created', 'Priority')
            self.alerts_tree = ttk.Treeview(alerts_frame, columns=alerts_columns, show='headings', height=10)
            
            for col in alerts_columns:
                self.alerts_tree.heading(col, text=col)
                self.alerts_tree.column(col, width=120)
            
            alerts_scrollbar = ttk.Scrollbar(alerts_frame, orient="vertical", command=self.alerts_tree.yview)
            self.alerts_tree.configure(yscrollcommand=alerts_scrollbar.set)
            
            self.alerts_tree.pack(side='left', fill='both', expand=True, padx=10, pady=10)
            alerts_scrollbar.pack(side='right', fill='y', pady=10)
            
            # Load existing alerts
            self.refresh_alerts_list()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open alerts management: {str(e)}")
            logger.error(f"Alerts management error: {e}")
    
    def add_plate_alert(self):
        """Add a new plate alert"""
        try:
            plate_number = self.alert_plate_entry.get().upper().strip()
            alert_type = self.alert_type_var.get()
            reason = self.alert_reason_entry.get().strip()
            
            if not plate_number or not reason:
                messagebox.showwarning("Warning", "Please fill in all fields")
                return
            
            # Add to database (simplified - in real system would use proper database method)
            conn = sqlite3.connect(self.system.db_manager.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO plate_alerts (plate_number, alert_type, alert_reason, priority_level)
                VALUES (?, ?, ?, ?)
            ''', (plate_number, alert_type, reason, 1))
            
            conn.commit()
            conn.close()
            
            # Clear fields
            self.alert_plate_entry.delete(0, tk.END)
            self.alert_reason_entry.delete(0, tk.END)
            
            # Refresh list
            self.refresh_alerts_list()
            
            messagebox.showinfo("Success", f"Alert added for plate: {plate_number}")
            logger.info(f"Plate alert added: {plate_number} - {alert_type}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add alert: {str(e)}")
            logger.error(f"Add plate alert error: {e}")
    
    def refresh_alerts_list(self):
        """Refresh the alerts list"""
        try:
            # Clear existing
            for item in self.alerts_tree.get_children():
                self.alerts_tree.delete(item)
            
            # Get alerts from database
            conn = sqlite3.connect(self.system.db_manager.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT plate_number, alert_type, alert_reason, created_date, priority_level
                FROM plate_alerts 
                WHERE active = 1 
                ORDER BY created_date DESC
            ''')
            
            alerts = cursor.fetchall()
            conn.close()
            
            # Add to tree
            for alert in alerts:
                created_date = alert[3][:10] if alert[3] else 'Unknown'  # Just date part
                self.alerts_tree.insert('', 'end', values=(
                    alert[0],  # plate_number
                    alert[1],  # alert_type
                    alert[2][:30] + "..." if len(alert[2]) > 30 else alert[2],  # reason (truncated)
                    created_date,  # created_date
                    alert[4]   # priority_level
                ))
                
        except Exception as e:
            logger.error(f"Refresh alerts error: {e}")
    
    def refresh_incidents(self):
        """Refresh incidents display"""
        try:
            self.update_incidents_table()
            messagebox.showinfo("Refresh", "Incidents list refreshed")
        except Exception as e:
            logger.error(f"Refresh error: {e}")
    
    def refresh_vehicles(self):
        """Refresh vehicles display"""
        try:
            self.update_vehicles_table()
            messagebox.showinfo("Refresh", "Vehicles list refreshed")
        except Exception as e:
            logger.error(f"Refresh vehicles error: {e}")
    
    def view_incident_details(self, event):
        """View incident details"""
        try:
            selection = self.incidents_tree.selection()
            if selection:
                item = self.incidents_tree.item(selection[0])
                incident_id = item['values'][0]
                
                # Create details window
                details_window = tk.Toplevel(self.root)
                details_window.title("Incident Details")
                details_window.geometry("700x500")
                details_window.configure(bg='#2c3e50')
                
                tk.Label(details_window, text=f"Incident Details - ID: {incident_id}", 
                        bg='#2c3e50', fg='white', font=('Arial', 14, 'bold')).pack(pady=20)
                
                details_text = tk.Text(details_window, bg='#34495e', fg='white', font=('Arial', 10))
                details_text.pack(fill='both', expand=True, padx=20, pady=20)
                
                # Get vehicle detections for this incident
                vehicles = self.system.db_manager.get_vehicle_detections(incident_id=incident_id)
                
                details_info = f"""Incident ID: {incident_id}
Status: Under Review
Evidence: Available
Location: Camera {self.get_current_camera_id()}
Type: {item['values'][3]}
Material: {item['values'][4]}
Confidence: {item['values'][5]}
Fine Amount: {item['values'][6]}
Time: {item['values'][1]}
Associated Vehicles: {item['values'][7]}

Vehicle Details:
"""
                
                if vehicles:
                    for i, vehicle in enumerate(vehicles, 1):
                        details_info += f"\n{i}. Plate: {vehicle[1] if len(vehicle) > 1 else 'Unknown'}"
                        details_info += f"\n   Confidence: {vehicle[2] if len(vehicle) > 2 else 0:.2f}"
                        details_info += f"\n   Valid: {'Yes' if len(vehicle) > 3 and vehicle[3] else 'No'}"
                else:
                    details_info += "\nNo vehicles detected for this incident."
                
                details_info += f"""

This incident is pending review by municipal authorities.
Evidence files have been automatically saved and secured.
Vehicle information can be used for owner notification and enforcement.
"""
                
                details_text.insert('1.0', details_info)
                details_text.config(state='disabled')
                
        except Exception as e:
            logger.error(f"View details error: {e}")
    
    def view_vehicle_details(self, event):
        """View vehicle details"""
        try:
            selection = self.vehicles_tree.selection()
            if selection:
                item = self.vehicles_tree.item(selection[0])
                plate_number = item['values'][1]
                
                # Create vehicle details window
                vehicle_window = tk.Toplevel(self.root)
                vehicle_window.title("Vehicle Details")
                vehicle_window.geometry("600x400")
                vehicle_window.configure(bg='#2c3e50')
                
                tk.Label(vehicle_window, text=f"Vehicle Details - {plate_number}", 
                        bg='#2c3e50', fg='white', font=('Arial', 14, 'bold')).pack(pady=20)
                
                vehicle_text = tk.Text(vehicle_window, bg='#34495e', fg='white', font=('Arial', 10))
                vehicle_text.pack(fill='both', expand=True, padx=20, pady=20)
                
                # Check for alerts
                alerts = self.system.db_manager.check_plate_alerts(plate_number)
                
                vehicle_info = f"""License Plate: {plate_number}
Detection Time: {item['values'][0]}
Camera: {item['values'][3]}
Confidence: {item['values'][2]}
Valid Plate: {item['values'][4]}

Alert Status: {item['values'][5]}
"""
                
                if alerts:
                    vehicle_info += f"\n⚠️ ACTIVE ALERTS ({len(alerts)}):\n"
                    for alert in alerts:
                        vehicle_info += f"\n- Type: {alert[0]}"
                        vehicle_info += f"\n  Reason: {alert[1]}"
                        vehicle_info += f"\n  Priority: {alert[2]}"
                        vehicle_info += f"\n  Date: {alert[3][:10] if alert[3] else 'Unknown'}\n"
                else:
                    vehicle_info += "\nNo active alerts for this vehicle."
                
                vehicle_info += "\n\nThis information can be used for municipal enforcement and owner notification."
                
                vehicle_text.insert('1.0', vehicle_info)
                vehicle_text.config(state='disabled')
                
        except Exception as e:
            logger.error(f"View vehicle details error: {e}")
    
    def on_closing(self):
        """Handle application closing"""
        try:
            if messagebox.askyesno("Exit", "Exit the Municipal Waste Detection System?"):
                self.is_updating = False
                self.system.cleanup()
                self.root.destroy()
                logger.info("Application closed")
        except Exception as e:
            logger.error(f"Closing error: {e}")
            self.root.destroy()
    
    def run(self):
        """Run the application"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Initial display
            self.video_canvas.create_text(
                400, 300,
                text="🏛️ MUNICIPAL WASTE DETECTION SYSTEM\n+ 🚗 VEHICLE TRACKING\nClick Start Monitoring to begin",
                fill='white', font=('Arial', 16), justify='center'
            )
            
            logger.info("Starting Municipal Waste Detection GUI with Vehicle Tracking")
            self.root.mainloop()
            
        except Exception as e:
            logger.error(f"GUI error: {e}")

def check_system_requirements():
    """Check system requirements"""
    requirements = {
        'opencv': False,
        'numpy': False,
        'PIL': False,
        'tkinter': False
    }
    
    try:
        import cv2
        requirements['opencv'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        requirements['numpy'] = True
    except ImportError:
        pass
    
    try:
        from PIL import Image
        requirements['PIL'] = True
    except ImportError:
        pass
    
    try:
        import tkinter
        requirements['tkinter'] = True
    except ImportError:
        pass
    
    missing = [req for req, available in requirements.items() if not available]
    
    if missing:
        print("❌ Missing requirements:")
        for req in missing:
            print(f"  - {req}")
        print("\n💡 Install with: pip install opencv-python numpy pillow")
        return False
    
    return True

def main():
    """Main entry point"""
    try:
        print("🏛️ Municipal Waste Detection System v2.1")
        print("+ 🚗 Vehicle License Plate Detection")
        print("=" * 60)
        print("Initializing system...")
        
        # Check requirements
        if not check_system_requirements():
            print("❌ Missing required dependencies")
            print("Please install the missing packages and try again.")
            return
        
        # Create directories
        directories = [
            "recordings", "screenshots", "plate_captures", "reports", "logs", 
            "evidence", "exports", "models", "config"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print("✅ System requirements met")
        print("✅ Directories created")
        
        # Check for OCR availability
        if OCR_AVAILABLE:
            print("✅ OCR available for license plate reading")
        else:
            print("⚠️ OCR not available - install pytesseract or easyocr for plate text reading")
            print("   pip install pytesseract easyocr")
        
        print("🚀 Starting Municipal Waste Detection System with Vehicle Tracking...")
        
        # Run application
        app = MunicipalWasteDetectionGUI()
        app.run()
        
    except KeyboardInterrupt:
        print("\n🛑 System shutdown requested")
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"❌ System error: {e}")
        print("Please check the log file for details.")
    finally:
        print("🏛️ Municipal Waste Detection System shutdown complete")

if __name__ == "__main__":
    main()