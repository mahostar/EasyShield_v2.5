import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
import numpy as np
from tkinterdnd2 import DND_FILES, TkinterDnD
from threading import Thread
import queue
from datetime import datetime
from PIL import Image
import imghdr

class FaceExtractor:
    def __init__(self):
        self.root = TkinterDnD.Tk()
        self.root.title("Face Extractor Tool")
        self.root.geometry("900x800")
        
        # Supported formats
        self.supported_formats = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp',
                                 '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
        
        self.files = []
        self.output_folder = ""
        self.processing_queue = queue.Queue()
        
        # Face detection and extraction parameters
        self.detect_faces = tk.BooleanVar(value=True)
        self.context_ratio = tk.DoubleVar(value=2.0)  # Capture 2x the face area by default
        self.extract_all_frames = tk.BooleanVar(value=False)  # Extract all frames vs selected frames
        self.frames_interval = tk.IntVar(value=10)  # Extract every Nth frame
        self.category = tk.StringVar(value="real")  # Default category (real or fake)
        
        # Initialize face detector
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title_label = ttk.Label(self.root, text="Face Extractor Tool", font=('Helvetica', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left frame for files
        left_frame = ttk.LabelFrame(main_frame, text="Input Files", padding=10)
        left_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        # Supported formats label
        formats_text = ", ".join([f"*{fmt}" for fmt in self.supported_formats])
        formats_label = ttk.Label(left_frame, text=f"Supported formats: {formats_text}")
        formats_label.pack(pady=5)
        
        # Drop area
        self.drop_area = tk.Text(left_frame, height=15, width=40)
        self.drop_area.pack(pady=5, fill='both', expand=True)
        self.drop_area.insert('1.0', "Drag and drop video or image files here...")
        self.drop_area.configure(state='disabled')
        
        # Enable drag and drop
        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind('<<Drop>>', self.on_drop)
        
        # Clear button
        clear_btn = ttk.Button(left_frame, text="Clear Files", command=self.clear_files)
        clear_btn.pack(pady=5)
        
        # Right frame for settings
        right_frame = ttk.LabelFrame(main_frame, text="Extraction Settings", padding=10)
        right_frame.pack(side='right', fill='both', padx=5)
        
        # Face detection toggle
        face_detect_check = ttk.Checkbutton(right_frame, text="Enable Face Detection", 
                                         variable=self.detect_faces)
        face_detect_check.pack(fill='x', pady=5)
        
        # Context ratio for face extraction
        context_frame = ttk.Frame(right_frame)
        context_frame.pack(fill='x', pady=5)
        ttk.Label(context_frame, text="Context Ratio:").pack(side='left')
        context_scale = ttk.Scale(context_frame, from_=1.0, to=4.0, 
                                variable=self.context_ratio, orient='horizontal')
        context_scale.pack(side='left', fill='x', expand=True, padx=5)
        ttk.Label(context_frame, text="(Higher = more context)").pack(side='right')
        
        # Video frame extraction options
        video_frame = ttk.LabelFrame(right_frame, text="Video Options")
        video_frame.pack(fill='x', pady=5)
        
        # Extract all frames toggle
        all_frames_check = ttk.Checkbutton(video_frame, text="Extract All Frames", 
                                         variable=self.extract_all_frames)
        all_frames_check.pack(fill='x', pady=5)
        
        # Frame interval
        interval_frame = ttk.Frame(video_frame)
        interval_frame.pack(fill='x', pady=5)
        ttk.Label(interval_frame, text="Extract every N frames:").pack(side='left')
        interval_entry = ttk.Entry(interval_frame, textvariable=self.frames_interval, width=5)
        interval_entry.pack(side='right')
        
        # Category selection
        category_frame = ttk.Frame(right_frame)
        category_frame.pack(fill='x', pady=10)
        ttk.Label(category_frame, text="Category:").pack(side='left')
        real_radio = ttk.Radiobutton(category_frame, text="Real", value="real", 
                                   variable=self.category)
        real_radio.pack(side='left', padx=10)
        fake_radio = ttk.Radiobutton(category_frame, text="Fake", value="fake", 
                                   variable=self.category)
        fake_radio.pack(side='left', padx=10)
        
        # Help text
        help_text = """
Tool Help:
- Drag and drop video files or images
- Face Detection: Detects and extracts faces
- Context Ratio: How much area around the face to include
- All frames will be resized to 640x640
- Images will be saved to real/fake subfolders
        """
        help_label = ttk.Label(right_frame, text=help_text, justify='left')
        help_label.pack(pady=10)
        
        # Output folder selection
        folder_frame = ttk.Frame(self.root)
        folder_frame.pack(fill='x', padx=10, pady=5)
        
        self.folder_label = ttk.Label(folder_frame, text="Output folder: Not selected")
        self.folder_label.pack(side='left', padx=5)
        
        self.folder_btn = ttk.Button(folder_frame, text="Select Output Folder", 
                                   command=self.select_output_folder)
        self.folder_btn.pack(side='right', padx=5)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(self.root, text="Progress", padding=10)
        progress_frame.pack(fill='x', padx=10, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', pady=5)
        
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.pack(pady=5)
        
        # Process button
        self.process_btn = ttk.Button(self.root, text="Extract Faces", command=self.start_processing)
        self.process_btn.pack(pady=10)
        
    def detect_face_opencv(self, frame):
        """Detect faces using OpenCV's Haar cascade"""
        faces = []
        try:
            h, w = frame.shape[:2]
            process_frame = frame
            
            # Downsample large frames for faster processing
            if max(h, w) > 800:
                scale = 800 / max(h, w)
                process_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            
            gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            cv_faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            # Process results
            if len(cv_faces) > 0:
                # Rescale if necessary
                if process_frame is not frame:
                    scale_x = w / process_frame.shape[1]
                    scale_y = h / process_frame.shape[0]
                    for (x, y, w, h) in cv_faces:
                        faces.append((int(x*scale_x), int(y*scale_y), 
                                    int(w*scale_x), int(h*scale_y)))
                else:
                    faces = [(x, y, w, h) for (x, y, w, h) in cv_faces]
        except Exception as e:
            print(f"Face detection error: {str(e)}")
            
        return faces
        
    def process_image(self, image_path, category):
        """Process a single image file"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return False, "Failed to read image"
            
            # Create output subfolder
            save_dir = os.path.join(self.output_folder, category)
            os.makedirs(save_dir, exist_ok=True)
            
            # Detect faces if enabled
            faces = []
            if self.detect_faces.get():
                faces = self.detect_face_opencv(image)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # If faces found, extract each face
            if faces:
                for i, (x, y, w, h) in enumerate(faces):
                    # Extract a wider area around the face for context
                    context_ratio = self.context_ratio.get()
                    
                    # Calculate context box dimensions
                    context_w = int(w * context_ratio)
                    context_h = int(h * context_ratio)
                    
                    # Center the context box on the face
                    context_x = max(0, x + w//2 - context_w//2)
                    context_y = max(0, y + h//2 - context_h//2)
                    
                    # Make sure box doesn't go outside the frame
                    context_x2 = min(image.shape[1], context_x + context_w)
                    context_y2 = min(image.shape[0], context_y + context_h)
                    context_w = context_x2 - context_x
                    context_h = context_y2 - context_y
                    
                    # Extract the context area
                    face_img = image[context_y:context_y2, context_x:context_x2]
                    
                    # Process to 640x640 square
                    self.save_as_square(face_img, save_dir, f"{timestamp}_face_{i}.jpg")
            else:
                # No faces found or detection disabled, process whole image
                self.save_as_square(image, save_dir, f"{timestamp}.jpg")
                
            return True, None
        except Exception as e:
            return False, str(e)
    
    def save_as_square(self, image, save_dir, filename):
        """Crop and resize image to 640x640 square"""
        try:
            h, w = image.shape[:2]
            
            # Take center crop to make it square
            if w > h:
                # Landscape orientation - crop width
                start_x = (w - h) // 2
                cropped = image[:, start_x:start_x+h]
            else:
                # Portrait orientation - crop height
                start_y = (h - w) // 2
                cropped = image[start_y:start_y+w, :]
            
            # Resize to exactly 640x640
            resized = cv2.resize(cropped, (640, 640))
            
            # Save the image
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, resized)
            
            return True
        except Exception as e:
            print(f"Error saving square image: {str(e)}")
            return False
    
    def process_video(self, video_path, category):
        """Process a video file frame by frame"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "Failed to open video file"
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Create output subfolder
            save_dir = os.path.join(self.output_folder, category)
            os.makedirs(save_dir, exist_ok=True)
            
            frame_count = 0
            saved_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Only process selected frames
                if not self.extract_all_frames.get() and frame_count % self.frames_interval.get() != 0:
                    frame_count += 1
                    continue
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                
                # Detect faces if enabled
                faces = []
                if self.detect_faces.get():
                    faces = self.detect_face_opencv(frame)
                
                # If faces found, extract each face
                if faces:
                    for i, (x, y, w, h) in enumerate(faces):
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
                        
                        # Extract the context area
                        face_img = frame[context_y:context_y2, context_x:context_x2]
                        
                        # Process to 640x640 square
                        if self.save_as_square(face_img, save_dir, f"{timestamp}_frame_{frame_count}_face_{i}.jpg"):
                            saved_count += 1
                else:
                    # No faces found or detection disabled, process whole frame
                    if self.save_as_square(frame, save_dir, f"{timestamp}_frame_{frame_count}.jpg"):
                        saved_count += 1
                
                frame_count += 1
                
                # Update progress
                if total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                    self.processing_queue.put(("progress", progress))
                    self.processing_queue.put(("status", f"Processing video: Frame {frame_count}/{total_frames} - Saved: {saved_count}"))
            
            cap.release()
            return True, saved_count
        except Exception as e:
            return False, str(e)
    
    def process_files(self):
        """Process all files in the queue"""
        try:
            total_files = len(self.files)
            processed_files = 0
            total_saved = 0
            
            for file_path in self.files:
                file_name = os.path.basename(file_path)
                self.processing_queue.put(("status", f"Processing {file_name}..."))
                
                # Check if it's an image or video
                is_image = False
                try:
                    # Try to determine if it's an image
                    if os.path.isfile(file_path):
                        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
                            is_image = True
                        elif imghdr.what(file_path) is not None:
                            is_image = True
                except:
                    # If error determining, assume by extension
                    is_image = file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'))
                
                category = self.category.get()
                
                if is_image:
                    success, result = self.process_image(file_path, category)
                    if success:
                        total_saved += 1
                    else:
                        self.processing_queue.put(("status", f"Error processing {file_name}: {result}"))
                else:
                    # Assume it's a video
                    success, result = self.process_video(file_path, category)
                    if success:
                        total_saved += result
                    else:
                        self.processing_queue.put(("status", f"Error processing {file_name}: {result}"))
                
                processed_files += 1
                progress = (processed_files / total_files) * 100
                self.processing_queue.put(("progress", progress))
            
            self.processing_queue.put(("complete", f"Processed {processed_files} files. Saved {total_saved} images."))
            
        except Exception as e:
            self.processing_queue.put(("error", f"Error: {str(e)}"))
    
    def clear_files(self):
        """Clear the list of files"""
        self.files = []
        self.update_drop_area()
        
    def on_drop(self, event):
        """Handle drag and drop of files"""
        files = self.root.tk.splitlist(event.data)
        valid_files = [f for f in files if f.lower().endswith(self.supported_formats)]
        
        if not valid_files:
            messagebox.showwarning("Invalid Files", 
                                 "No valid files were dropped.\n\nSupported formats: " + 
                                 ", ".join(self.supported_formats))
            return
            
        self.files.extend(valid_files)
        self.update_drop_area()
        
    def update_drop_area(self):
        """Update the drop area text with current files"""
        self.drop_area.configure(state='normal')
        self.drop_area.delete('1.0', tk.END)
        if self.files:
            self.drop_area.insert('1.0', "Added files:\n\n")
            for i, file in enumerate(self.files, 1):
                self.drop_area.insert(tk.END, f"{i}. {os.path.basename(file)}\n")
        else:
            self.drop_area.insert('1.0', "Drag and drop video or image files here...")
        self.drop_area.configure(state='disabled')
        
    def select_output_folder(self):
        """Select the output folder"""
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder = folder
            self.folder_label.config(text=f"Output folder: {folder}")
            
            # Create subdirectories
            for subdir in ["real", "fake"]:
                os.makedirs(os.path.join(folder, subdir), exist_ok=True)
            
    def update_ui(self):
        """Update the UI with processing status"""
        try:
            while True:
                msg_type, value = self.processing_queue.get_nowait()
                if msg_type == "progress":
                    self.progress_var.set(value)
                elif msg_type == "status":
                    self.status_label.config(text=value)
                elif msg_type == "complete":
                    self.status_label.config(text=value)
                    self.process_btn.config(state='normal')
                    messagebox.showinfo("Complete", value)
                    return
                elif msg_type == "error":
                    self.status_label.config(text=value)
                    self.process_btn.config(state='normal')
                    messagebox.showerror("Error", value)
                    return
                
                self.root.update()
        except queue.Empty:
            self.root.after(100, self.update_ui)
            
    def start_processing(self):
        """Start the processing thread"""
        if not self.files:
            messagebox.showwarning("No Files", "Please add some files first!")
            return
            
        if not self.output_folder:
            messagebox.showwarning("No Output Folder", "Please select an output folder first!")
            return
            
        self.process_btn.config(state='disabled')
        self.status_label.config(text="Processing...")
        self.progress_var.set(0)
        
        process_thread = Thread(target=self.process_files)
        process_thread.daemon = True
        process_thread.start()
        
        self.update_ui()
        
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = FaceExtractor()
    app.run()