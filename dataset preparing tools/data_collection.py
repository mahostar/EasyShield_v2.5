"""
DepthFusion-ViT Data Collection Tool
-----------------------------------
This script provides a GUI application for collecting face data to train 
an anti-spoofing model. It supports:
1. Camera capture
2. Screen region capture (for collecting from videos/streams)
"""

import cv2
import os
import time
import numpy as np
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageGrab
import threading
import queue
from datetime import datetime
import urllib.parse
import torch
import torchvision
import torch.nn as nn
from collections import deque
import socket
import pygetwindow as gw
import keyboard

# Optional import of MTCNN - will fall back to OpenCV if not available
try:
    from facenet_pytorch import MTCNN
    print("Successfully imported MTCNN from facenet_pytorch")
    MTCNN_AVAILABLE = True
except ImportError:
    print("MTCNN not available, using OpenCV for face detection")
    MTCNN_AVAILABLE = False

class ScreenCaptureWindow:
    def __init__(self, parent, callback):
        self.parent = parent
        self.callback = callback
        self.window = tk.Toplevel(parent)
        self.window.title("Screen Region Selector")
        self.window.attributes('-alpha', 0.3)  # Semi-transparent
        self.window.attributes('-topmost', True)
        
        # Variables for region selection
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.selection_done = False
        
        # Create canvas for drawing selection rectangle
        self.canvas = tk.Canvas(self.window, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # Instructions label
        self.label = ttk.Label(self.window, 
                             text="Click and drag to select screen region.\nPress ESC to cancel.")
        self.label.pack(pady=10)
        
        # Bind ESC key to cancel
        self.window.bind("<Escape>", self.cancel)
        
        # Center window
        self.window.update_idletasks()
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        self.window.geometry(f"{screen_width}x{screen_height}+0+0")
    
    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
    
    def on_drag(self, event):
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y, outline="red"
        )
    
    def on_release(self, event):
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)
        
        # Get screen coordinates
        window_x = self.window.winfo_x()
        window_y = self.window.winfo_y()
        
        # Calculate absolute screen coordinates
        screen_x1 = window_x + x1
        screen_y1 = window_y + y1
        screen_x2 = window_x + x2
        screen_y2 = window_y + y2
        
        self.selection_done = True
        self.window.destroy()
        
        # Call callback with selected region
        self.callback((screen_x1, screen_y1, screen_x2, screen_y2))
    
    def cancel(self, event=None):
        self.window.destroy()
        self.callback(None)

class CameraThread(threading.Thread):
    """Dedicated thread for camera reading to maximize frame rate"""
    def __init__(self, camera_url, frame_buffer, stop_event, timeout=10.0):
        super().__init__()
        self.daemon = True
        self.camera_url = camera_url
        self.frame_buffer = frame_buffer
        self.stop_event = stop_event
        self.timeout = timeout
        self.fps = 0
        self.cap = None
        print(f"Initializing camera thread with URL: {camera_url}")
        
    def run(self):
        # Set timeout for socket operations
        socket.setdefaulttimeout(self.timeout)
        print(f"Socket timeout set to {self.timeout} seconds")
        
        try:
            print("Attempting to open camera connection...")
            # Configure capture parameters for better performance
            if isinstance(self.camera_url, int):
                # Local camera
                self.cap = cv2.VideoCapture(self.camera_url)
            else:
                # IP camera - use FFMPEG backend
                self.cap = cv2.VideoCapture(self.camera_url, cv2.CAP_FFMPEG)
            
            if not self.cap.isOpened():
                print(f"Failed to open camera: {self.camera_url}")
                print("Camera error code:", self.cap.get(cv2.CAP_PROP_ERROR_STRING) if hasattr(cv2, 'CAP_PROP_ERROR_STRING') else "Unknown error")
                print("Checking camera URL format...")
                if isinstance(self.camera_url, str) and 'http' in self.camera_url:
                    print("URL validation:")
                    print(f"- Protocol: {'https' if 'https://' in self.camera_url else 'http'}")
                    print(f"- Ends with /video: {'Yes' if self.camera_url.endswith('/video') else 'No'}")
                    print("Attempting to ping host...")
                    try:
                        host = urllib.parse.urlparse(self.camera_url).netloc.split(':')[0]
                        response = os.system(f"ping -n 1 {host}")
                        print(f"Ping response: {'Success' if response == 0 else 'Failed'}")
                    except Exception as e:
                        print(f"Ping test failed: {str(e)}")
                return
            
            print("Camera opened successfully!")
            print(f"Camera backend: {self.cap.getBackendName() if hasattr(self.cap, 'getBackendName') else 'Unknown'}")
            print(f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"FPS setting: {self.cap.get(cv2.CAP_PROP_FPS)}")
            
            # MJPEG optimizations for better high-resolution performance
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Small buffer for low latency
            
            # Attempt to set high-resolution mode
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            if width < 1280:
                print("Attempting to increase camera resolution...")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                # Check if it worked
                new_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                new_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"Updated resolution: {int(new_width)}x{int(new_height)}")
            
            frame_times = deque(maxlen=30)
            last_frame_time = time.time()
            
            while not self.stop_event.is_set():
                start_time = time.time()
                
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Camera read failed, error info:")
                    print(f"- Is opened: {self.cap.isOpened()}")
                    print(f"- Last error: {self.cap.get(cv2.CAP_PROP_ERROR_STRING) if hasattr(self.cap, 'CAP_PROP_ERROR_STRING') else 'Unknown'}")
                    print("Attempting to reconnect...")
                    time.sleep(0.5)
                    # Recreate capture
                    self.cap.release()
                    if isinstance(self.camera_url, int):
                        self.cap = cv2.VideoCapture(self.camera_url)
                    else:
                        self.cap = cv2.VideoCapture(self.camera_url, cv2.CAP_FFMPEG)
                    if not self.cap.isOpened():
                        print("Reconnection failed")
                    continue
                
                # Calculate FPS
                current_time = time.time()
                frame_times.append(current_time)
                
                # Update FPS every 5 frames
                if len(frame_times) > 5:
                    time_diff = frame_times[-1] - frame_times[0]
                    if time_diff > 0:
                        self.fps = len(frame_times) / time_diff
                
                # If the buffer is full, remove oldest frame
                if len(self.frame_buffer) >= self.frame_buffer.maxlen:
                    self.frame_buffer.popleft()
                    
                # Add new frame to buffer
                self.frame_buffer.append(frame)
                
                # Throttle if needed to avoid excessive CPU usage
                elapsed = time.time() - start_time
                if elapsed < 0.01:  # Target up to 100 FPS for capture
                    time.sleep(0.01 - elapsed)
                    
        except Exception as e:
            print(f"Camera thread critical error:")
            print(f"- Error type: {type(e).__name__}")
            print(f"- Error message: {str(e)}")
            print(f"- URL used: {self.camera_url}")
            if isinstance(e, socket.timeout):
                print("Connection timed out - check if the IP camera is accessible")
            elif isinstance(e, urllib.error.URLError):
                print("URL Error - check if the IP camera address is correct")
            elif isinstance(e, ConnectionRefusedError):
                print("Connection refused - check if the IP camera is running and accepting connections")
        finally:
            # Clean up
            if self.cap and self.cap.isOpened():
                self.cap.release()
            print("Camera thread stopped")


class FaceDataCollectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DepthFusion-ViT Data Collection Tool")
        self.root.geometry("1200x720")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Check CUDA availability
        self.cuda_available = False
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.cuda_available = self.device.type == 'cuda'
            print(f"Using device: {self.device}")
            if self.cuda_available:
                print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"GPU detection error: {str(e)}")
            self.device = torch.device('cpu')
        
        # Variables
        self.save_location = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "DepthFusionViT_Data"))
        self.is_recording = False
        self.is_collecting_real = False
        self.is_collecting_fake = False
        self.camera_source = tk.StringVar(value="0")
        self.camera_url = tk.StringVar(value="http://192.168.1.13:8080/video")
        self.real_count = 0
        self.fake_count = 0
        self.frame_count = 0
        self.processed_count = 0
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        # Default to OpenCV as it's generally more reliable
        self.detector_type = tk.StringVar(value="OpenCV" if not MTCNN_AVAILABLE else "MTCNN")
        self.fps_display = tk.StringVar(value="Camera: 0 FPS | Processing: 0 FPS")
        self.resolution = tk.StringVar(value="640x480")
        self.detect_faces = tk.BooleanVar(value=True)
        self.process_every_n_frames = tk.IntVar(value=1)  # Process every n frames
        self.mtcnn_error_count = 0  # Track MTCNN errors to avoid console spam
        
        # Frame buffer for the camera thread
        self.frame_buffer = deque(maxlen=10)
        
        # Camera thread
        self.camera_thread = None
        
        # Initialize face detector
        self.initialize_detector()
        
        # Initialize video capture
        self.cap = None
        
        # Create GUI components
        self.create_widgets()
        
        # No startup popup message
    
    def initialize_detector(self):
        """Initialize the appropriate face detector"""
        if self.detector_type.get() == "MTCNN" and MTCNN_AVAILABLE:
            try:
                self.detector_backend = "MTCNN"
                self.face_detector = MTCNN(
                    keep_all=True,
                    device=self.device,
                    select_largest=False,
                    min_face_size=60,  # Decreased min face size to detect smaller faces
                    thresholds=[0.6, 0.7, 0.7],  # More lenient thresholds
                    post_process=False
                )
                print("Using MTCNN face detector")
            except Exception as e:
                print(f"Error initializing MTCNN: {str(e)}")
                self.detector_type.set("OpenCV")
                self.detector_backend = "OpenCV"
                self.face_detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                print("Falling back to OpenCV face detector")
        else:
            # OpenCV's detector
            self.detector_backend = "OpenCV"
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("Using OpenCV face detector")

    def create_widgets(self):
        # Main frame layout
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Camera selection
        camera_frame = ttk.LabelFrame(control_frame, text="Camera Settings")
        camera_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Camera type selection
        type_frame = ttk.Frame(camera_frame)
        type_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(type_frame, text="Camera Type:").pack(side=tk.LEFT)
        self.camera_type = ttk.Combobox(type_frame, values=["Local Camera", "IP Webcam", "Screen Capture"])
        self.camera_type.set("IP Webcam") # Default to IP webcam
        self.camera_type.pack(side=tk.LEFT, padx=5)
        self.camera_type.bind('<<ComboboxSelected>>', self.on_camera_type_change)
        
        # Local camera selection
        self.local_frame = ttk.Frame(camera_frame)
        ttk.Label(self.local_frame, text="Camera ID:").pack(side=tk.LEFT)
        camera_dropdown = ttk.Combobox(self.local_frame, textvariable=self.camera_source, 
                                     values=["0", "1", "2", "3"])
        camera_dropdown.pack(side=tk.LEFT, padx=5)
        
        # IP webcam URL
        self.ip_frame = ttk.Frame(camera_frame)
        self.ip_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(self.ip_frame, text="IP Webcam URL:").pack(side=tk.LEFT)
        ttk.Entry(self.ip_frame, textvariable=self.camera_url, width=40).pack(side=tk.LEFT, padx=5)
        
        # Screen capture controls
        self.screen_frame = ttk.Frame(camera_frame)
        ttk.Button(self.screen_frame, text="Select Region", 
                  command=self.select_screen_region).pack(side=tk.LEFT, padx=5)
        
        # Screen capture variables
        self.capture_region = None
        self.screen_capture_active = False
        self.screen_capture_thread = None
        
        # Resolution selection
        res_frame = ttk.Frame(camera_frame)
        res_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(res_frame, text="Resolution:").pack(side=tk.LEFT)
        res_dropdown = ttk.Combobox(res_frame, textvariable=self.resolution, 
                                  values=["640x480", "1280x720", "1920x1080"])
        res_dropdown.set("1920x1080")  # Default to high resolution 1080p
        res_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Performance options
        perf_frame = ttk.Frame(camera_frame)
        perf_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Face detection toggle
        ttk.Checkbutton(perf_frame, text="Enable Face Detection", 
                      variable=self.detect_faces).pack(side=tk.LEFT, padx=5)
        
        # Frame processing rate
        proc_frame = ttk.Frame(camera_frame)
        proc_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(proc_frame, text="Process every N frames:").pack(side=tk.LEFT)
        ttk.Spinbox(proc_frame, from_=1, to=5, textvariable=self.process_every_n_frames, 
                  width=5).pack(side=tk.LEFT, padx=5)
        self.process_every_n_frames.set(1)  # Default to processing every frame
        
        # Capture Rate
        capture_frame = ttk.Frame(camera_frame)
        capture_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(capture_frame, text="Capture every N frames:").pack(side=tk.LEFT)
        self.capture_every_n_frames = tk.IntVar(value=5)  # Default to saving every 5 frames (3-5 images/sec)
        ttk.Spinbox(capture_frame, from_=1, to=15, textvariable=self.capture_every_n_frames, 
                  width=5).pack(side=tk.LEFT, padx=5)
                  
        # Face detector selection
        detector_frame = ttk.Frame(camera_frame)
        detector_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(detector_frame, text="Face Detector:").pack(side=tk.LEFT)
        
        # Only show MTCNN option if available
        detector_values = ["OpenCV"]
        if MTCNN_AVAILABLE:
            detector_values.append("MTCNN")
            
        detector_dropdown = ttk.Combobox(detector_frame, textvariable=self.detector_type, 
                                       values=detector_values)
        detector_dropdown.pack(side=tk.LEFT, padx=5)
        detector_dropdown.bind('<<ComboboxSelected>>', self.on_detector_change)
        
        # Save location
        location_frame = ttk.LabelFrame(control_frame, text="Save Location")
        location_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Entry(location_frame, textvariable=self.save_location, width=30).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(location_frame, text="Browse", command=self.browse_location).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Camera control
        camera_control_frame = ttk.LabelFrame(control_frame, text="Camera Control")
        camera_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.camera_button = ttk.Button(camera_control_frame, text="Start Camera", command=self.toggle_camera)
        self.camera_button.pack(padx=5, pady=5, fill=tk.X)
        
        # Context capture ratio
        context_frame = ttk.Frame(camera_control_frame)
        context_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(context_frame, text="Context Capture Ratio:").pack(side=tk.LEFT)
        self.context_ratio = tk.DoubleVar(value=2.0)  # Capture 2x the face area by default
        context_scale = ttk.Scale(context_frame, from_=1.0, to=4.0, variable=self.context_ratio)
        context_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(context_frame, text="(Higher = more context)").pack(side=tk.LEFT, padx=5)
        
        # Minimal blur check instead of threshold
        self.min_blur_check = tk.BooleanVar(value=True)
        ttk.Checkbutton(camera_control_frame, text="Enable Minimal Blur Check", 
                      variable=self.min_blur_check).pack(fill=tk.X, padx=5, pady=5)
        
        # Collection controls
        collection_frame = ttk.LabelFrame(control_frame, text="Data Collection")
        collection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Real data collection
        real_frame = ttk.Frame(collection_frame)
        real_frame.pack(fill=tk.X, padx=5, pady=5)
        self.real_button = ttk.Button(real_frame, text="Collect Real", command=self.toggle_real_collection)
        self.real_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.real_label = ttk.Label(real_frame, text="Real: 0")
        self.real_label.pack(side=tk.LEFT, padx=5)
        
        # Fake data collection
        fake_frame = ttk.Frame(collection_frame)
        fake_frame.pack(fill=tk.X, padx=5, pady=5)
        self.fake_button = ttk.Button(fake_frame, text="Collect Fake", command=self.toggle_fake_collection)
        self.fake_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.fake_label = ttk.Label(fake_frame, text="Fake: 0")
        self.fake_label.pack(side=tk.LEFT, padx=5)
        
        # Status frame
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(status_frame, textvariable=self.fps_display).pack(side=tk.RIGHT, padx=5)
        
        # Right panel - Video display
        display_frame = ttk.LabelFrame(main_frame, text="Camera Feed")
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(display_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
    
    def browse_location(self):
        """Open file dialog to select save location"""
        folder = filedialog.askdirectory(initialdir=self.save_location.get())
        if folder:
            self.save_location.set(folder)
            
            # Create subdirectories
            for subdir in ["real", "fake"]:
                os.makedirs(os.path.join(folder, subdir), exist_ok=True)
            
            self.status_var.set(f"Save location set to: {folder}")
    
    def toggle_camera(self):
        """Start or stop the camera"""
        if not self.is_recording:
            self.start_camera()
            self.camera_button.config(text="Stop Camera")
        else:
            self.stop_camera()
            self.camera_button.config(text="Start Camera")
    
    def toggle_real_collection(self):
        """Toggle real data collection"""
        if not self.is_recording:
            messagebox.showinfo("Info", "Please start the camera first")
            return
        
        self.is_collecting_real = not self.is_collecting_real
        self.is_collecting_fake = False  # Stop fake collection if active
        
        if self.is_collecting_real:
            self.real_button.config(text="Stop Real")
            self.fake_button.config(state=tk.DISABLED)
            self.status_var.set("Collecting REAL samples...")
        else:
            self.real_button.config(text="Collect Real")
            self.fake_button.config(state=tk.NORMAL)
            self.status_var.set("Stopped collecting REAL samples")
    
    def toggle_fake_collection(self):
        """Toggle fake data collection"""
        if not self.is_recording:
            messagebox.showinfo("Info", "Please start the camera first")
            return
        
        self.is_collecting_fake = not self.is_collecting_fake
        self.is_collecting_real = False  # Stop real collection if active
        
        if self.is_collecting_fake:
            self.fake_button.config(text="Stop Fake")
            self.real_button.config(state=tk.DISABLED)
            self.status_var.set("Collecting FAKE samples...")
        else:
            self.fake_button.config(text="Collect Fake")
            self.real_button.config(state=tk.NORMAL)
            self.status_var.set("Stopped collecting FAKE samples")
    
    def select_screen_region(self):
        """Open screen region selector window"""
        if self.is_recording:
            self.stop_camera()
        
        # Minimize main window temporarily
        self.root.iconify()
        time.sleep(0.5)  # Give time for window to minimize
        
        # Create region selector
        selector = ScreenCaptureWindow(self.root, self.on_region_selected)
        
        # Wait for selector to close
        self.root.wait_window(selector.window)
        
        # Restore main window
        self.root.deiconify()
    
    def on_region_selected(self, region):
        """Handle selected screen region"""
        if region:
            self.capture_region = region
            self.status_var.set(f"Selected region: {region}")
            
            # Start capture if in screen capture mode
            if self.camera_type.get() == "Screen Capture":
                self.start_camera()
    
    def on_camera_type_change(self, event=None):
        """Handle camera type selection change"""
        if self.camera_type.get() == "Local Camera":
            self.local_frame.pack(fill=tk.X, padx=5, pady=5)
            self.ip_frame.pack_forget()
            self.screen_frame.pack_forget()
        elif self.camera_type.get() == "IP Webcam":
            self.local_frame.pack_forget()
            self.ip_frame.pack(fill=tk.X, padx=5, pady=5)
            self.screen_frame.pack_forget()
        else:  # Screen Capture
            self.local_frame.pack_forget()
            self.ip_frame.pack_forget()
            self.screen_frame.pack(fill=tk.X, padx=5, pady=5)
    
    def start_screen_capture(self):
        """Start screen region capture thread"""
        self.screen_capture_active = True
        self.screen_capture_thread = threading.Thread(target=self.screen_capture_loop)
        self.screen_capture_thread.daemon = True
        self.screen_capture_thread.start()
    
    def screen_capture_loop(self):
        """Continuously capture selected screen region"""
        while self.screen_capture_active and not self.stop_event.is_set():
            try:
                if self.capture_region:
                    # Capture screen region
                    screenshot = ImageGrab.grab(bbox=self.capture_region)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Add to frame buffer
                    if len(self.frame_buffer) >= self.frame_buffer.maxlen:
                        self.frame_buffer.popleft()
                    self.frame_buffer.append(frame)
                    
                    time.sleep(1/30)  # Limit to ~30 FPS
            except Exception as e:
                print(f"Screen capture error: {str(e)}")
                time.sleep(0.1)
    
    def start_camera(self):
        """Start the camera or screen capture"""
        try:
            self.stop_event.clear()
            self.frame_buffer.clear()
            
            if self.camera_type.get() == "Screen Capture":
                if not self.capture_region:
                    messagebox.showerror("Error", "Please select a screen region first")
                    return
                self.start_screen_capture()
            else:
                # Get resolution
                width, height = map(int, self.resolution.get().split('x'))
                
                # Original camera capture code
                if self.camera_type.get() == "Local Camera":
                    camera_src = int(self.camera_source.get())
                else:
                    url = self.camera_url.get()
                    if not url.endswith('/video'):
                        url = url.rstrip('/') + '/video'
                    camera_src = url
                
                self.camera_thread = CameraThread(
                    camera_src, 
                    self.frame_buffer,
                    self.stop_event
                )
                
                self.camera_thread.start()
                time.sleep(0.5)  # Wait a bit for camera to start
                
                # Try to set resolution if camera has started
                if self.camera_thread.cap and self.camera_thread.cap.isOpened():
                    self.camera_thread.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self.camera_thread.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    print(f"Camera resolution set to: {width}x{height}")
            
            # Start processing
            self.is_recording = True
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self.process_camera)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            # Start display thread
            self.display_thread = threading.Thread(target=self.update_display)
            self.display_thread.daemon = True
            self.display_thread.start()
            
            self.status_var.set("Capture started")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.is_recording = False
    
    def stop_camera(self):
        """Stop camera or screen capture"""
        if self.is_recording:
            self.stop_event.set()
            self.is_recording = False
            self.screen_capture_active = False
            
            # Wait for threads to finish
            if hasattr(self, 'processing_thread'):
                self.processing_thread.join(timeout=1.0)
            
            if hasattr(self, 'display_thread'):
                self.display_thread.join(timeout=1.0)
            
            if hasattr(self, 'camera_thread') and self.camera_thread:
                self.camera_thread.join(timeout=1.0)
            
            self.status_var.set("Capture stopped")
            self.canvas.delete("all")
            self.camera_button.config(text="Start Camera")
    
    def process_camera(self):
        """Process frames and detect faces"""
        # For FPS calculation
        frame_times = deque(maxlen=30)
        last_mtcnn_error_time = 0
        
        # Minimal blur threshold - only reject extremely blurry images
        MIN_BLUR_THRESHOLD = 15  
        
        while not self.stop_event.is_set():
            try:
                # Skip if no frames available
                if not self.frame_buffer:
                    time.sleep(0.01)  # Short sleep to avoid CPU spinning
                    continue
                
                # Get the latest frame
                frame = self.frame_buffer[-1].copy()
                
                # Skip frames if needed for performance
                self.frame_count += 1
                if self.frame_count % self.process_every_n_frames.get() != 0:
                    # Still display frame but skip detection
                    if not self.frame_queue.full():
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                        self.frame_queue.put(rgb_frame)
                    continue
                
                self.processed_count += 1
                
                # Update FPS calculation
                current_time = time.time()
                frame_times.append(current_time)
                
                # Keep only recent frames for FPS calculation
                if len(frame_times) > 1:
                    fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                    if self.camera_thread:
                        camera_fps = self.camera_thread.fps
                    else:
                        camera_fps = 0
                    # Update FPS display every 10 frames
                    if self.processed_count % 10 == 0:
                        self.fps_display.set(f"Camera: {camera_fps:.1f} FPS | Processing: {fps:.1f} FPS")
                
                # Skip face detection if disabled (just save whole frames)
                faces = []
                if self.detect_faces.get():
                    # Resize for faster processing if needed
                    h, w = frame.shape[:2]
                    process_frame = frame
                    
                    # Downsample large frames for faster processing
                    if max(h, w) > 800:
                        scale = 800 / max(h, w)
                        process_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                    
                    # Detect faces using the selected detector
                    if self.detector_backend == "MTCNN":
                        try:
                            # Convert to RGB for PyTorch
                            frame_rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                            
                            # For facenet_pytorch MTCNN, detect method returns:
                            # boxes, probs, landmarks
                            # Where boxes is a list of [x1, y1, x2, y2] face bounding boxes
                            mtcnn_result = self.face_detector.detect(frame_rgb)
                            
                            # If using facenet_pytorch and it detects faces
                            if mtcnn_result and len(mtcnn_result) >= 2:
                                boxes, probs = mtcnn_result[0], mtcnn_result[1]
                                
                                if boxes is not None and len(boxes) > 0:
                                    # Process each detected face
                                    for i, box in enumerate(boxes):
                                        # Make sure confidence is high enough (above 0.85)
                                        if probs[i] < 0.85:
                                            continue
                                            
                                        # Convert to [x, y, w, h] format
                                        x1, y1, x2, y2 = [int(b) for b in box]
                                        w_box = x2 - x1
                                        h_box = y2 - y1
                                        
                                        # Only consider reasonably sized faces
                                        if w_box < 20 or h_box < 20:
                                            continue
                                            
                                        # Rescale if the frame was resized
                                        if process_frame is not frame:
                                            scale_x = w / process_frame.shape[1]
                                            scale_y = h / process_frame.shape[0]
                                            x1 = int(x1 * scale_x)
                                            y1 = int(y1 * scale_y)
                                            w_box = int(w_box * scale_x)
                                            h_box = int(h_box * scale_y)
                                        
                                        # Add the face to our list
                                        faces.append((x1, y1, w_box, h_box))
                            
                        except Exception as e:
                            # Limit error reporting to avoid console spam
                            self.mtcnn_error_count += 1
                            if current_time - last_mtcnn_error_time > 5.0:  # Only print every 5 seconds
                                print(f"MTCNN error: {str(e)} (suppressing further errors for 5 seconds)")
                                last_mtcnn_error_time = current_time
                            
                            # Fall back to OpenCV detector when MTCNN fails
                            faces = self.opencv_face_detection(process_frame, frame, w, h)
                    else:
                        # OpenCV's detector
                        faces = self.opencv_face_detection(process_frame, frame, w, h)
                
                # Create a display frame (with visual markers for display only)
                display_frame = frame.copy()
                
                # If there are no faces but we're collecting data, just save the whole frame
                if not faces and (self.is_collecting_real or self.is_collecting_fake):
                    # Save whole frame every N frames
                    if self.processed_count % self.capture_every_n_frames.get() == 0:
                        # Only check if frame is extremely blurry if blur check is enabled
                        save_whole_frame = True
                        if self.min_blur_check.get():
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            blur = cv2.Laplacian(gray, cv2.CV_64F).var()
                            save_whole_frame = blur >= MIN_BLUR_THRESHOLD
                        
                        if save_whole_frame:
                            self.save_whole_frame(frame)
                
                # Process detected faces
                for (x, y, w, h) in faces:
                    # Draw rectangle ON THE DISPLAY FRAME ONLY (not the saved image)
                    color = (0, 255, 0) if self.is_collecting_real else (
                        (0, 0, 255) if self.is_collecting_fake else (128, 128, 128)
                    )
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Check blur and save if collecting
                    if self.is_collecting_real or self.is_collecting_fake:
                        # Save every N frames
                        if self.processed_count % self.capture_every_n_frames.get() == 0:
                            # Extract a wider area around the face for context
                            context_ratio = self.context_ratio.get()
                            
                            # Calculate context box dimensions
                            context_w = int(w * context_ratio)
                            context_h = int(h * context_ratio)
                            
                            # Center the context box on the face
                            context_x = max(0, x + w//2 - context_w//2)
                            context_y = max(0, y + h//2 - context_h//2)
                            
                            # Make sure box doesn't go outside the frame
                            context_x2 = min(frame.shape[1], context_x + context_w)
                            context_y2 = min(frame.shape[0], context_y + context_h)
                            context_w = context_x2 - context_x
                            context_h = context_y2 - context_y
                            
                            # Extract the context area from ORIGINAL frame
                            context_img = frame[context_y:context_y2, context_x:context_x2]
                            
                            # Draw the context region on the display frame (not the saved image)
                            cv2.rectangle(display_frame, (context_x, context_y), 
                                        (context_x + context_w, context_y + context_h),
                                        (255, 255, 0), 2)
                            
                            # Check minimal blur if enabled
                            save_image = True
                            if self.min_blur_check.get():
                                gray_face = cv2.cvtColor(context_img, cv2.COLOR_BGR2GRAY)
                                blur = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                                save_image = blur >= MIN_BLUR_THRESHOLD
                                
                                # Show blur value on display frame only
                                cv2.putText(display_frame, f"Blur: {int(blur)}", (x, y-10),
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                         (0, 255, 0) if blur >= MIN_BLUR_THRESHOLD else (0, 0, 255),
                                         2)
                                
                            if save_image:
                                self.save_face(context_img)
                
                # Add collection status to display frame only
                if self.is_collecting_real:
                    cv2.putText(display_frame, "Collecting: REAL", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif self.is_collecting_fake:
                    cv2.putText(display_frame, "Collecting: FAKE", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add FPS and counters to display frame only
                if self.camera_thread:
                    fps_text = f"Camera: {self.camera_thread.fps:.1f} FPS | Processing: {fps:.1f} FPS"
                else:
                    fps_text = f"Processing: {fps:.1f} FPS"
                cv2.putText(display_frame, fps_text, (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add counter information
                cv2.putText(display_frame, f"Real: {self.real_count} | Fake: {self.fake_count}", 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Convert to RGB for tkinter
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Add to display queue
                if not self.frame_queue.full():
                    self.frame_queue.put(rgb_frame)
                
                # Update counters periodically
                if self.processed_count % 30 == 0:
                    self.update_counters()
                    
            except Exception as e:
                print(f"Frame processing error: {str(e)}")
                time.sleep(0.1)  # Don't flood console with errors

    def update_display(self):
        """Update the video display"""
        while not self.stop_event.is_set():
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    # Get canvas dimensions
                    canvas_width = self.canvas.winfo_width()
                    canvas_height = self.canvas.winfo_height()
                    
                    if canvas_width > 1 and canvas_height > 1:
                        # Resize frame to fit canvas
                        aspect_ratio = frame.shape[1] / frame.shape[0]
                        
                        if canvas_width / canvas_height > aspect_ratio:
                            new_height = canvas_height
                            new_width = int(new_height * aspect_ratio)
                        else:
                            new_width = canvas_width
                            new_height = int(new_width / aspect_ratio)
                        
                        frame = cv2.resize(frame, (new_width, new_height))
                        
                        # Convert to PhotoImage
                        image = Image.fromarray(frame)
                        photo = ImageTk.PhotoImage(image=image)
                        
                        # Update canvas
                        self.canvas.config(width=new_width, height=new_height)
                        self.canvas.create_image(new_width//2, new_height//2, 
                                              image=photo, anchor=tk.CENTER)
                        self.canvas.image = photo
            
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Display error: {str(e)}")
            
            time.sleep(0.03)  # ~30 FPS
    
    def save_whole_frame(self, frame):
        """Save the entire frame when no faces are detected"""
        try:
            # Determine label and update counter
            if self.is_collecting_real:
                label = "real"
                self.real_count += 1
            elif self.is_collecting_fake:
                label = "fake"
                self.fake_count += 1
            else:
                return
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_full.jpg"
            
            # Save directory
            save_dir = os.path.join(self.save_location.get(), label)
            os.makedirs(save_dir, exist_ok=True)
            
            # Crop to 640x640 without black borders
            h, w = frame.shape[:2]
            
            # Take center crop
            if w > h:
                # Landscape orientation - crop width
                start_x = (w - h) // 2
                cropped = frame[:, start_x:start_x+h]
            else:
                # Portrait orientation - crop height
                start_y = (h - w) // 2
                cropped = frame[start_y:start_y+w, :]
            
            # Now resize to exactly 640x640
            resized = cv2.resize(cropped, (640, 640))
            
            # Verify final dimensions
            assert resized.shape[0] == 640 and resized.shape[1] == 640, f"Image dimensions are {resized.shape[1]}x{resized.shape[0]}, expected 640x640"
            
            # Save the properly sized image
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, resized)
            
        except Exception as e:
            print(f"Error saving full frame: {str(e)}")
        
    def save_face(self, face_img):
        """Save face image with appropriate label"""
        try:
            # Determine label and update counter
            if self.is_collecting_real:
                label = "real"
                self.real_count += 1
            elif self.is_collecting_fake:
                label = "fake"
                self.fake_count += 1
            else:
                return
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}.jpg"
            
            # Save directory
            save_dir = os.path.join(self.save_location.get(), label)
            os.makedirs(save_dir, exist_ok=True)
            
            # Crop to 640x640 without black borders
            h, w = face_img.shape[:2]
            
            # Take center crop
            if w > h:
                # Landscape orientation - crop width
                start_x = (w - h) // 2
                cropped = face_img[:, start_x:start_x+h]
            else:
                # Portrait orientation - crop height
                start_y = (h - w) // 2
                cropped = face_img[start_y:start_y+w, :]
            
            # Now resize to exactly 640x640
            resized = cv2.resize(cropped, (640, 640))
            
            # Verify final dimensions
            assert resized.shape[0] == 640 and resized.shape[1] == 640, f"Image dimensions are {resized.shape[1]}x{resized.shape[0]}, expected 640x640"
            
            # Save the properly sized image
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, resized)
            
            # Update display in the GUI
            self.update_counters()
            
        except Exception as e:
            print(f"Error saving image: {str(e)}")
    
    def update_counters(self):
        """Update the sample counters"""
        self.real_label.config(text=f"Real: {self.real_count}")
        self.fake_label.config(text=f"Fake: {self.fake_count}")
    
    def on_close(self):
        """Handle window close"""
        if self.is_recording:
            self.stop_camera()
        self.root.destroy()

    def on_detector_change(self, event=None):
        """Handle detector type change"""
        if self.is_recording:
            # If camera is running, show a message that detector will change on restart
            messagebox.showinfo("Detector Change", 
                              "The face detector will change after you restart the camera.")
        else:
            # If camera is not running, just update the detector
            self.initialize_detector()
            self.status_var.set(f"Changed detector to {self.detector_type.get()}")

    def opencv_face_detection(self, process_frame, original_frame, w, h):
        """Helper method to detect faces using OpenCV"""
        faces = []
        try:
            gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
            
            # Use OpenCV's Haar cascade
            cv_faces = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            ).detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            # Process OpenCV results
            if len(cv_faces) > 0:
                # Rescale if necessary
                if process_frame is not original_frame:
                    scale_x = w / process_frame.shape[1]
                    scale_y = h / process_frame.shape[0]
                    for (x, y, w, h) in cv_faces:
                        faces.append((int(x*scale_x), int(y*scale_y), 
                                    int(w*scale_x), int(h*scale_y)))
                else:
                    faces = [(x, y, w, h) for (x, y, w, h) in cv_faces]
        except Exception as e:
            print(f"OpenCV face detection error: {str(e)}")
            
        return faces


def main():
    root = tk.Tk()
    app = FaceDataCollectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main() 