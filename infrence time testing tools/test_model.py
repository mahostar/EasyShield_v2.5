"""
Anti-Spoofing Test Program

A high-performance GUI application for testing the trained anti-spoofing model
using an IP webcam or local camera.
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
    QSpinBox, QCheckBox, QStatusBar, QSlider, QDoubleSpinBox, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO

# Suppress excessive YOLO logging
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.WARNING)

class AntiSpoofingTest(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.cap = None
        self.timer = None
        self.frame_count = 0
        self.fps = 0
        self.last_time = cv2.getTickCount()
        self.fps_history = deque(maxlen=30)  # For smoother FPS display
        
        # Default settings
        self.model_path = "E:/projects/anti-spoofing deep learning Model/models/anti_spoofing_20250313_1104422/weights/best.pt"
        self.camera_url = "http://192.168.1.13:8080/video"
        self.confidence_threshold = 0.5
        self.process_every_n_frames = 2  # Process every nth frame for better performance
        
        # Performance settings
        self.target_fps = 30  # Target FPS - reduced from 60 to 30 for stability
        self.last_terminal_update = 0
        self.terminal_update_interval = 2.0  # Update terminal every 2 seconds
        
        # Face detector (using OpenCV's Haar cascade like in gt.py)
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Performance optimization: Add face detector parameters
        self.face_scale_factor = 1.2  # Default is 1.1, higher is faster but less accurate
        self.face_min_neighbors = 4  # Default is 5, lower is faster
        self.face_min_size = (60, 60)  # Larger minimum size is faster
        
        # Cache the last detected faces to reduce processing
        self.last_faces = []
        self.face_cache_frames = 5  # Only detect faces every N frames
        
        # Performance monitoring
        self.performance_history = deque(maxlen=60)
        self.last_gc_time = time.time()
        self.gc_interval = 10.0  # Run garbage collection every 10 seconds
        
        # RGB frame allocation
        self.rgb_frame = None
        
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
        self.setWindowTitle("Anti-Spoofing Test")
        self.setGeometry(100, 100, 1200, 800)
        
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
        
        # Display options
        self.show_fps_cb = QCheckBox("Show FPS")
        self.show_fps_cb.setChecked(True)
        camera_layout.addWidget(self.show_fps_cb)
        
        self.show_confidence_cb = QCheckBox("Show Confidence")
        self.show_confidence_cb.setChecked(True)
        camera_layout.addWidget(self.show_confidence_cb)
        
        # Add camera group to left panel
        left_panel.addWidget(camera_group)
        
        # Control buttons
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.toggle_camera)
        left_panel.addWidget(self.start_btn)
        
        # Status display
        self.status_label = QLabel("Status: Not running")
        left_panel.addWidget(self.status_label)
        
        self.fps_label = QLabel("FPS: -")
        left_panel.addWidget(self.fps_label)
        
        # Add stretch to push everything up
        left_panel.addStretch()
        
        # Right panel for video display
        right_panel = QVBoxLayout()
        
        # Video display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("background-color: black;")
        right_panel.addWidget(self.image_label)
        
        # Add panels to main layout
        layout.addLayout(left_panel, 1)
        layout.addLayout(right_panel, 4)
        
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
            self.start_btn.setText("Start")
            self.status_label.setText("Status: Stopped")
            self.statusBar().showMessage("Camera stopped")
        else:
            # Start camera
            if self.camera_combo.currentText() == "IP Camera":
                camera_source = self.url_input.text()
                
                # For IP cameras, try to use FFMPEG backend for better performance
                self.cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
                if not self.cap.isOpened():
                    # Fallback to default backend
                    self.cap = cv2.VideoCapture(camera_source)
            else:
                camera_id = int(self.camera_combo.currentText().split(" ")[1])
                camera_source = camera_id
                self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Failed to open camera")
                return
            
            # Optimize camera settings for performance
            if self.camera_combo.currentText() != "IP Camera":
                # For local webcams
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Use smaller resolution for better performance
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            else:
                # For IP cameras, try to use a larger resolution if available
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
            # Common settings for all cameras
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            
            # For IP cameras, set MJPEG format for better performance
            if self.camera_combo.currentText() == "IP Camera":
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            # Clear any accumulated performance data
            self.fps_history.clear()
            self.last_faces = []
            
            # Start timer with appropriate frequency based on target FPS
            frame_interval = int(1000 / self.target_fps)
            self.timer.start(frame_interval)
            
            self.start_btn.setText("Stop")
            self.status_label.setText("Status: Running")
            self.statusBar().showMessage("Camera started")
    
    def get_face_crop(self, frame, x, y, w, h):
        """Extract and process face region from the frame (like in gt.py)"""
        # Add significantly more padding to face crop to capture context (phones, papers, etc.)
        padding = int(w * 1.5)  # Increased from 0.2 to 1.5 (750% increase in padding)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        # Extract face region with much more context
        face_img = frame[y1:y2, x1:x2]
        
        # Ensure we got a valid crop
        if face_img.size == 0:
            return None
        
        # Update face preview in GUI only occasionally
        if self.frame_count % 5 == 0:  # Only update preview every 5 frames
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
            except Exception as e:
                print(f"Error updating face preview: {e}")
        
        # Resize to expected input size for model (640x640)
        try:
            # For better performance, use direct resize to model input size
            # Skip intermediate processing
            resized_face = cv2.resize(face_img, (640, 640))
            return resized_face
        except Exception as e:
            print(f"Error preparing face crop: {e}")
            return None
    
    def update_frame(self):
        """Update the video frame and run detection"""
        if self.cap is None:
            return
        
        # Read frame with timeout
        ret, frame = self.cap.read()
        if not ret:
            # Don't spam the status bar, just try again
            return
        
        # Performance monitoring
        current_time = cv2.getTickCount()
        time_diff = (current_time - self.last_time) / cv2.getTickFrequency()
        current_fps = 1.0 / time_diff if time_diff > 0 else 0
        self.fps_history.append(current_fps)
        self.fps = sum(self.fps_history) / len(self.fps_history)
        self.last_time = current_time
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Process frames - only do detection on some frames for performance
        if self.frame_count % self.process_every_n_frames == 0:
            # Resize frame for faster processing (half size)
            proc_frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces using OpenCV Haar Cascade
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=self.face_scale_factor,
                minNeighbors=self.face_min_neighbors,
                minSize=self.face_min_size
            )
            
            # Convert small frame detections back to original frame size
            self.last_faces = [(x*2, y*2, w*2, h*2) for (x, y, w, h) in faces]
        
        # Process each detected face
        for (x, y, w, h) in self.last_faces:
            # Get face crop for classification
            face_img = self.get_face_crop(frame, x, y, w, h)
            
            if face_img is not None:
                # Run anti-spoofing classification
                with torch.no_grad():  # Disable gradient computation for inference
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
                    
                    # Calculate padded coordinates for display
                    padding = int(w * 0.2)
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(display_frame.shape[1], x + w + padding)
                    y2 = min(display_frame.shape[0], y + h + padding)
                    
                    # Set color based on classification (green for real, red for fake)
                    # FIX: Reverse the labels - class 0 is real, class 1 is fake
                    color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                    label = "REAL" if cls == 0 else "FAKE"
                    
                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with confidence
                    if self.show_confidence_cb.isChecked():
                        label_text = f"{label} {conf:.2f}"
                    else:
                        label_text = label
                    
                    # Add text above the rectangle
                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(display_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), color, -1)
                    cv2.putText(display_frame, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add FPS to frame (only if enabled)
        if self.show_fps_cb.isChecked():
            cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Update terminal output less frequently
        current_time = datetime.now().timestamp()
        if current_time - self.last_terminal_update >= self.terminal_update_interval:
            print(f"\rFPS: {self.fps:.1f} | Faces: {len(self.last_faces)}", end="")
            self.last_terminal_update = current_time
        
        # Update FPS label less frequently
        if self.frame_count % 30 == 0 and self.show_fps_cb.isChecked():
            self.fps_label.setText(f"FPS: {self.fps:.1f}")
        
        # Convert and display frame
        h, w = display_frame.shape[:2]
        bytes_per_line = 3 * w
        
        # Allocate RGB frame only once
        if self.rgb_frame is None or self.rgb_frame.shape[:2] != display_frame.shape[:2]:
            self.rgb_frame = np.empty((h, w, 3), dtype=np.uint8)
        
        # Convert BGR to RGB in-place
        cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB, dst=self.rgb_frame)
        
        # Create QImage without copying data
        qt_image = QImage(self.rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale image efficiently
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation  # Use faster scaling
        )
        
        # Update image
        self.image_label.setPixmap(scaled_pixmap)
        
        # Periodic memory cleanup to avoid slowdown
        if time.time() - self.last_gc_time > self.gc_interval:
            # Run garbage collection to prevent memory leaks
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.last_gc_time = time.time()
        
        # Increment frame counter
        self.frame_count += 1
    
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
    window = AntiSpoofingTest()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 