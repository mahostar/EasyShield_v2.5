"""
Anti-Spoofing Test Program for Raspberry Pi

A streamlined version of the anti-spoofing model test app optimized for Pi,
featuring burst-mode image capture and confidence comparison.
"""

import os
import sys
import cv2
import torch
import numpy as np
import gc
import time
from datetime import datetime
from collections import deque
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QLineEdit, QMessageBox, QComboBox,
    QSpinBox, QCheckBox, QStatusBar, QFileDialog, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO

# Suppress excessive YOLO logging
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.WARNING)

class EasyShieldTest(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.cap = None
        self.timer = None
        self.frame_count = 0
        self.fps = 0
        self.last_time = cv2.getTickCount()
        
        # Default settings
        self.model_path = "E:/projects/anti-spoofing deep learning Model/models/anti_spoofing_20250313_1104422/weights/best.pt"
        self.camera_url = "http://192.168.1.13:8080/video"
        self.confidence_threshold = 0.5
        
        # Face detector
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_scale_factor = 1.2
        self.face_min_neighbors = 4
        self.face_min_size = (60, 60)
        
        # Scan mode settings
        self.scan_in_progress = False
        self.scan_frames = []
        self.scan_results = []
        self.scan_timer = QTimer()
        self.scan_timer.timeout.connect(self.capture_scan_frame)
        self.scan_count = 0
        self.max_scan_frames = 5  # Number of frames to capture
        self.scan_delay = 150  # Delay between captures in ms
        
        # Allocate space for face preview
        self.face_preview = None
        
        # Configure GPU
        self.configure_gpu()
        
        self.init_ui()
        self.load_model()
    
    def configure_gpu(self):
        """Configure GPU for optimal performance"""
        try:
            if torch.cuda.is_available():
                # Force GPU usage
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # Print GPU information
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"Using GPU: {gpu_name} with {gpu_mem:.2f} GB memory")
                self.device = torch.device('cuda')
            else:
                print("CUDA is not available. Using CPU instead.")
                self.device = torch.device('cpu')
        except Exception as e:
            print(f"GPU configuration error: {e}")
            self.device = torch.device('cpu')
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("EasyShield Anti-Spoofing Test")
        self.setGeometry(100, 100, 1000, 700)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Left panel for controls
        left_panel = QVBoxLayout()
        
        # Model selection
        model_group = QWidget()
        model_layout = QVBoxLayout(model_group)
        
        model_layout.addWidget(QLabel("Model File:"))
        self.model_path_input = QLineEdit(self.model_path)
        self.model_path_input.setReadOnly(True)
        model_layout.addWidget(self.model_path_input)
        
        model_buttons = QHBoxLayout()
        self.browse_model_btn = QPushButton("Browse...")
        self.browse_model_btn.clicked.connect(self.select_model_file)
        model_buttons.addWidget(self.browse_model_btn)
        
        self.load_model_btn = QPushButton("Load Selected Model")
        self.load_model_btn.clicked.connect(self.reload_model)
        model_buttons.addWidget(self.load_model_btn)
        
        model_layout.addLayout(model_buttons)
        left_panel.addWidget(model_group)
        
        # Camera settings
        camera_group = QWidget()
        camera_layout = QVBoxLayout(camera_group)
        
        # Camera selection
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("IP Camera")
        self.camera_combo.addItem("Webcam 0")
        self.camera_combo.addItem("Webcam 1")
        self.camera_combo.addItem("Raspberry Pi Camera")
        camera_layout.addWidget(QLabel("Camera Source:"))
        camera_layout.addWidget(self.camera_combo)
        
        # IP Camera URL
        self.url_input = QLineEdit(self.camera_url)
        camera_layout.addWidget(QLabel("IP Camera URL:"))
        camera_layout.addWidget(self.url_input)
        
        # Processing settings
        self.confidence_spin = QSpinBox()
        self.confidence_spin.setRange(1, 100)
        self.confidence_spin.setValue(int(self.confidence_threshold * 100))
        self.confidence_spin.setSuffix("%")
        camera_layout.addWidget(QLabel("Confidence Threshold:"))
        camera_layout.addWidget(self.confidence_spin)
        
        # Face crop preview
        self.face_preview_label = QLabel("Face Crop Preview")
        self.face_preview_label.setAlignment(Qt.AlignCenter)
        self.face_preview_label.setMinimumSize(150, 150)
        self.face_preview_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        camera_layout.addWidget(self.face_preview_label)
        
        # Add camera group to left panel
        left_panel.addWidget(camera_group)
        
        # Control buttons
        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.toggle_camera)
        left_panel.addWidget(self.start_btn)
        
        # Scan button
        self.scan_btn = QPushButton("SCAN (5 Images)")
        self.scan_btn.clicked.connect(self.start_scan)
        self.scan_btn.setEnabled(False)
        self.scan_btn.setStyleSheet("background-color: #0078d7; color: white; font-weight: bold; font-size: 14px; padding: 8px;")
        left_panel.addWidget(self.scan_btn)
        
        # Scan progress bar
        self.scan_progress = QProgressBar()
        self.scan_progress.setRange(0, self.max_scan_frames)
        self.scan_progress.setValue(0)
        left_panel.addWidget(self.scan_progress)
        
        # Result label
        self.result_label = QLabel("Result: Waiting for scan")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        left_panel.addWidget(self.result_label)
        
        # Status display
        self.status_label = QLabel("Status: Not running")
        left_panel.addWidget(self.status_label)
        
        # Add stretch to push everything up
        left_panel.addStretch()
        
        # Right panel for video display
        right_panel = QVBoxLayout()
        
        # Video display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: black;")
        right_panel.addWidget(self.image_label)
        
        # Add panels to main layout
        layout.addLayout(left_panel, 1)
        layout.addLayout(right_panel, 3)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Create timer for video update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Show the window
        self.show()
    
    def select_model_file(self):
        """Open file dialog to select a model file"""
        options = QFileDialog.Options()
        model_file, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Model File", 
            os.path.dirname(self.model_path),  # Start in the current model's directory
            "Model Files (*.pt *.pth *.weights);;All Files (*)",
            options=options
        )
        
        if model_file:
            self.model_path_input.setText(model_file)
    
    def reload_model(self):
        """Load the model from the selected path"""
        new_model_path = self.model_path_input.text()
        
        if not os.path.exists(new_model_path):
            QMessageBox.critical(self, "Error", f"Model file not found: {new_model_path}")
            return
        
        try:
            # Save camera state
            camera_was_running = False
            if self.cap is not None and self.timer.isActive():
                camera_was_running = True
                self.toggle_camera()  # Stop camera
            
            # Unload existing model and clear memory
            if self.model is not None:
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Update model path
            self.model_path = new_model_path
            
            # Load new model
            self.load_model()
            
            # Restart camera if it was running
            if camera_was_running:
                self.toggle_camera()  # Start camera again
            
            self.statusBar().showMessage(f"Model loaded successfully: {os.path.basename(new_model_path)}")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self.statusBar().showMessage("Failed to load model")
    
    def load_model(self):
        """Load the YOLO model"""
        try:
            # Force using the latest model
            self.model = YOLO(self.model_path)
            
            # Set model to inference mode immediately to prepare it
            if self.model is not None:
                # Force model to GPU if available
                if torch.cuda.is_available():
                    self.model.to('cuda')
                    print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
                else:
                    self.model.to('cpu')
                    print("Model loaded on CPU")
                self.model.eval()
            
            # Check device model is running on
            device = self.model.device
            self.statusBar().showMessage(f"Model loaded successfully on {device}")
            print(f"Model loaded on: {device}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self.statusBar().showMessage("Failed to load model")
    
    def toggle_camera(self):
        """Start or stop the camera"""
        if self.timer.isActive():
            # Stop camera
            self.timer.stop()
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.start_btn.setText("Start Camera")
            self.status_label.setText("Status: Stopped")
            self.scan_btn.setEnabled(False)
            self.statusBar().showMessage("Camera stopped")
        else:
            # Start camera
            if self.camera_combo.currentText() == "IP Camera":
                camera_source = self.url_input.text()
                self.cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
                if not self.cap.isOpened():
                    # Fallback to default backend
                    self.cap = cv2.VideoCapture(camera_source)
            elif self.camera_combo.currentText() == "Raspberry Pi Camera":
                # Special handling for Pi camera if available
                try:
                    # Try using Pi camera with OpenCV
                    self.cap = cv2.VideoCapture(0)
                    # You might need to add specific settings for Pi camera here
                except Exception as e:
                    QMessageBox.warning(self, "Warning", f"Error opening Pi camera: {str(e)}\nFalling back to default camera.")
                    self.cap = cv2.VideoCapture(0)
            else:
                camera_id = int(self.camera_combo.currentText().split(" ")[1])
                self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Failed to open camera")
                return
            
            # Optimize camera settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            
            # Start timer with appropriate frequency
            self.timer.start(33)  # ~30 FPS
            
            self.start_btn.setText("Stop Camera")
            self.status_label.setText("Status: Running")
            self.scan_btn.setEnabled(True)
            self.statusBar().showMessage("Camera started")
    
    def get_face_crop(self, frame, x, y, w, h):
        """Extract and process face region from the frame"""
        # Add padding to face crop to capture context
        padding = int(w * 1.5)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        # Extract face region with context
        face_img = frame[y1:y2, x1:x2]
        
        # Ensure we got a valid crop
        if face_img.size == 0:
            return None
        
        # Update face preview
        if not self.scan_in_progress or self.scan_count == 0:
            try:
                if face_img is not None and face_img.size > 0:
                    # Ensure the face image is not too large for preview
                    max_size = 150
                    h, w = face_img.shape[:2]
                    scale = min(max_size/w, max_size/h)
                    if scale < 1:
                        preview_img = cv2.resize(face_img.copy(), (int(w*scale), int(h*scale)))
                    else:
                        preview_img = face_img.copy()
                    
                    # Convert to RGB for Qt
                    preview_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
                    h, w = preview_rgb.shape[:2]
                    bytes_per_line = 3 * w
                    q_img = QImage(preview_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    preview_pixmap = QPixmap.fromImage(q_img)
                    self.face_preview_label.setPixmap(preview_pixmap)
                    
                    # Save for scan mode
                    self.face_preview = face_img
            except Exception as e:
                print(f"Error updating face preview: {e}")
        
        # Resize to expected input size for model (640x640)
        try:
            resized_face = cv2.resize(face_img, (640, 640))
            return resized_face
        except Exception as e:
            print(f"Error preparing face crop: {e}")
            return None
    
    def update_frame(self):
        """Update the video frame and run detection"""
        if self.cap is None:
            return
        
        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Performance monitoring
        current_time = cv2.getTickCount()
        time_diff = (current_time - self.last_time) / cv2.getTickFrequency()
        self.fps = 1.0 / time_diff if time_diff > 0 else 0
        self.last_time = current_time
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Detect faces using OpenCV Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=self.face_scale_factor,
            minNeighbors=self.face_min_neighbors,
            minSize=self.face_min_size
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Get face crop for classification
            face_img = self.get_face_crop(frame, x, y, w, h)
            
            if face_img is not None and not self.scan_in_progress:
                # Draw rectangle around face (green during normal mode)
                padding = int(w * 0.2)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(display_frame.shape[1], x + w + padding)
                y2 = min(display_frame.shape[0], y + h + padding)
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # If no faces detected, disable scan button
        if len(faces) == 0:
            self.scan_btn.setEnabled(False)
            self.face_preview = None
        elif not self.scan_in_progress:
            self.scan_btn.setEnabled(True)
        
        # Add FPS and status to frame
        cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add scan mode indicator
        if self.scan_in_progress:
            cv2.putText(display_frame, f"SCANNING: {self.scan_count}/{self.max_scan_frames}", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Convert and display frame
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation
        ))
        
        # Increment frame counter
        self.frame_count += 1
    
    def start_scan(self):
        """Start the multi-frame scanning process"""
        # Make sure we have a face
        if self.face_preview is None:
            QMessageBox.warning(self, "Warning", "No face detected for scanning!")
            return
        
        # Reset scan data
        self.scan_frames = []
        self.scan_results = []
        self.scan_count = 0
        self.scan_progress.setValue(0)
        self.result_label.setText("Scanning...")
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px; color: blue;")
        
        # Disable scan button during scan
        self.scan_btn.setEnabled(False)
        self.scan_in_progress = True
        
        # Start the scan timer to capture frames with delay
        self.scan_timer.start(self.scan_delay)
    
    def capture_scan_frame(self):
        """Capture a frame for scanning"""
        # Stop if we've collected enough frames or camera stopped
        if self.scan_count >= self.max_scan_frames or self.cap is None:
            self.scan_timer.stop()
            self.process_scan_results()
            return
        
        # Get a frame from the camera
        ret, frame = self.cap.read()
        if not ret:
            self.scan_timer.stop()
            self.scan_in_progress = False
            self.scan_btn.setEnabled(True)
            QMessageBox.warning(self, "Error", "Failed to capture frame during scan!")
            return
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=self.face_scale_factor,
            minNeighbors=self.face_min_neighbors,
            minSize=self.face_min_size
        )
        
        # If we found faces, process the largest one
        if len(faces) > 0:
            # Find the largest face
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            x, y, w, h = largest_face
            
            # Get face crop
            face_img = self.get_face_crop(frame, x, y, w, h)
            
            if face_img is not None:
                # Add to our scan frames
                self.scan_frames.append(face_img)
                
                # Update progress
                self.scan_count += 1
                self.scan_progress.setValue(self.scan_count)
                
                # Print scanning status
                print(f"Captured scan frame {self.scan_count}/{self.max_scan_frames}")
        else:
            # No face detected, still increment counter but don't add to frames
            self.scan_count += 1
            self.scan_progress.setValue(self.scan_count)
            print(f"No face detected in scan frame {self.scan_count}")
    
    def process_scan_results(self):
        """Process the captured frames and determine if real or fake"""
        self.scan_in_progress = False
        
        # Check if we have any valid frames
        if len(self.scan_frames) == 0:
            self.result_label.setText("Scan Failed: No faces detected!")
            self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px; color: red;")
            self.scan_btn.setEnabled(True)
            return
        
        print(f"Processing {len(self.scan_frames)} scan frames...")
        self.statusBar().showMessage(f"Processing {len(self.scan_frames)} scan frames...")
        
        # Check if model is loaded
        if self.model is None:
            self.result_label.setText("Error: Model not properly loaded!")
            self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px; background-color: red; color: white;")
            self.scan_btn.setEnabled(True)
            self.statusBar().showMessage("Error: Failed to process frames - model not loaded")
            return
            
        # Process each frame with the model
        real_count = 0
        fake_count = 0
        confidence_scores = []
        
        for i, face_img in enumerate(self.scan_frames):
            # Run anti-spoofing classification
            with torch.no_grad():
                results = self.model(
                    face_img,
                    conf=self.confidence_spin.value() / 100,
                    verbose=False
                )
            
            # Process results if any detections
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get the prediction result
                result = results[0]
                conf = float(result.boxes.conf[0])
                cls = int(result.boxes.cls[0])
                
                confidence_scores.append(conf)
                
                # Count real/fake predictions
                if cls == 0:  # Real
                    real_count += 1
                else:  # Fake
                    fake_count += 1
                
                print(f"Frame {i+1}: {'REAL' if cls == 0 else 'FAKE'} with confidence {conf:.2f}")
        
        # Decision logic - majority vote with confidence threshold
        if real_count > fake_count:
            self.result_label.setText("✅ REAL PERSON DETECTED")
            self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px; background-color: green; color: white;")
        else:
            self.result_label.setText("❌ ATTACK DETECTED\nEasyShield Guard Defense Activated")
            self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px; background-color: red; color: white;")
        
        # Show detailed results in status bar
        avg_conf = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        self.statusBar().showMessage(f"Results: {real_count} real, {fake_count} fake. Avg confidence: {avg_conf:.2f}")
        
        # Re-enable scan button
        self.scan_btn.setEnabled(True)
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.cap is not None:
            self.cap.release()
        
        # Clean up resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear the terminal line
        print("\rApplication closed.")
        event.accept()

def main():
    # Enable high DPI scaling
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    window = EasyShieldTest()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 