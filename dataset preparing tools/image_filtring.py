import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import glob
import torch
from facenet_pytorch import MTCNN
import cv2
import numpy as np
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import shutil
from datetime import datetime


class ImageFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Filter Tool")
        
        # Maximize window
        self.root.state('zoomed')
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Threading control
        self.exit_event = threading.Event()
        self.thread_pool = None
        
        # Initialize CUDA if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize MTCNN
        try:
            self.mtcnn = MTCNN(
                keep_all=True,
                device=self.device,
                select_largest=False,
                min_face_size=80,
                thresholds=[0.6, 0.7, 0.7]
            )
            print(f"MTCNN initialized on {self.device}")
        except Exception as e:
            print(f"Error initializing MTCNN: {e}")
            self.mtcnn = None
        
        # Variables
        self.dataset_path = tk.StringVar()
        self.current_index = 0
        self.image_files = []
        self.current_images = []
        self.selected_images = set()
        self.total_images = 0
        self.processed_count = 0
        self.deleted_count = 0
        self.images_per_page = tk.IntVar(value=8)
        self.enable_suggestions = tk.BooleanVar(value=True)
        self.processing_queue = queue.Queue()
        self.image_widgets = []
        self.face_detection_results = {}
        self.detection_thread = None
        self.backup_folder = None
        self.deleted_images = []  # Keep track of deleted images for recovery
        
        # Create GUI components
        self.create_widgets()
        
        # Bind keyboard events
        self.root.bind("<space>", self.delete_selected)
        self.root.bind("<Right>", self.next_page)
        self.root.bind("<Left>", self.prev_page)
        self.root.bind("<Escape>", lambda e: self.clear_selection())
        self.root.bind("u", self.undo_last_deletion)
        
    def create_widgets(self):
        # Main layout
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top control panel
        control_frame = ttk.LabelFrame(main_frame, text="Dataset Controls", padding="10")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Folder selection
        folder_frame = ttk.Frame(control_frame)
        folder_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(folder_frame, text="Dataset Folder:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(folder_frame, textvariable=self.dataset_path, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(folder_frame, text="Browse", command=self.browse_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(folder_frame, text="Load Dataset", command=self.load_dataset).pack(side=tk.LEFT, padx=5)
        
        # Settings frame
        settings_frame = ttk.Frame(control_frame)
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Images per page setting
        ttk.Label(settings_frame, text="Images per page:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(settings_frame, from_=4, to=32, width=5, 
                    textvariable=self.images_per_page,
                    command=self.on_images_per_page_change).pack(side=tk.LEFT, padx=5)
        
        # Face detection toggle
        ttk.Checkbutton(settings_frame, text="Enable face detection suggestions",
                       variable=self.enable_suggestions).pack(side=tk.LEFT, padx=20)
        
        # Instructions
        ttk.Label(settings_frame, 
                 text="Click to select/deselect. Space to delete & next page. 'U' to undo deletion.").pack(side=tk.LEFT, padx=20)
        
        # Stats frame
        stats_frame = ttk.Frame(control_frame)
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.stats_label = ttk.Label(stats_frame, text="No dataset loaded")
        self.stats_label.pack(side=tk.LEFT, padx=5)
        
        self.progress_var = tk.StringVar(value="0/0")
        ttk.Label(stats_frame, textvariable=self.progress_var).pack(side=tk.RIGHT, padx=5)
        
        # Navigation buttons
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(nav_frame, text="Previous Page (←)", command=self.prev_page).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next Page (→/Space)", command=lambda: self.next_page()).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Undo Delete (U)", command=self.undo_last_deletion).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Delete Selected & Next", command=lambda: self.delete_selected()).pack(side=tk.RIGHT, padx=5)
        ttk.Button(nav_frame, text="Clear Selection", command=self.clear_selection).pack(side=tk.RIGHT, padx=5)
        ttk.Button(nav_frame, text="Verify Dataset", command=self.verify_dataset).pack(side=tk.RIGHT, padx=5)
        
        # Image grid frame
        self.grid_frame = ttk.Frame(main_frame)
        self.grid_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready. Select a dataset folder to begin.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
        # Create image grid
        self.create_image_grid()
    
    def create_image_grid(self):
        # Clear existing grid
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        
        # Calculate grid dimensions
        images_per_page = self.images_per_page.get()
        cols = min(4, images_per_page)
        rows = (images_per_page + cols - 1) // cols
        
        # Create frames for each image slot
        self.image_widgets = []
        self.image_frames = []
        self.selection_indicators = []
        
        for i in range(images_per_page):
            row = i // cols
            col = i % cols
            
            # Create frame for image
            frame = ttk.Frame(self.grid_frame, borderwidth=5, relief="solid")
            frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # Calculate frame size based on screen dimensions
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            frame_width = (screen_width - 100) // cols
            frame_height = (screen_height - 260) // rows
            frame.configure(width=frame_width, height=frame_height)
            
            # Create canvas for image
            canvas = tk.Canvas(frame, bg="black", highlightthickness=0)
            canvas.pack(fill=tk.BOTH, expand=True)
            
            # Create a selection indicator (colored rectangle around the frame)
            # Initial state - gray border (unselected)
            frame.configure(style="Unselected.TFrame")
            
            # Bind click events to both frame and canvas
            canvas.bind("<Button-1>", lambda e, idx=i: self.toggle_selection(idx))
            frame.bind("<Button-1>", lambda e, idx=i: self.toggle_selection(idx))
            
            self.image_widgets.append(canvas)
            self.image_frames.append(frame)
        
        # Configure grid weights
        for i in range(rows):
            self.grid_frame.grid_rowconfigure(i, weight=1)
        for i in range(cols):
            self.grid_frame.grid_columnconfigure(i, weight=1)
        
        # Create styles for selected/unselected frames
        self.style = ttk.Style()
        self.style.configure("Selected.TFrame", borderwidth=5, relief="solid", background="blue")
        self.style.configure("Unselected.TFrame", borderwidth=1, relief="solid", background="gray")
        self.style.configure("Suggested.TFrame", borderwidth=5, relief="solid", background="yellow")
    
    def on_images_per_page_change(self):
        self.create_image_grid()
        self.display_current_images()
    
    def toggle_selection(self, index):
        if index >= len(self.current_images):
            return
            
        img_path = self.current_images[index]
        frame = self.image_frames[index]
        
        if img_path in self.selected_images:
            self.selected_images.remove(img_path)
            frame.configure(style="Unselected.TFrame")
            # If it was suggested, restore the yellow highlight
            if self.enable_suggestions.get() and self.face_detection_results.get(img_path, False):
                frame.configure(style="Suggested.TFrame")
        else:
            self.selected_images.add(img_path)
            frame.configure(style="Selected.TFrame")
        
        # Update status
        self.status_var.set(f"Selected: {len(self.selected_images)} images")
    
    def clear_selection(self):
        self.selected_images.clear()
        for i, frame in enumerate(self.image_frames):
            if i < len(self.current_images):
                img_path = self.current_images[i]
                if self.enable_suggestions.get() and self.face_detection_results.get(img_path, False):
                    frame.configure(style="Suggested.TFrame")
                else:
                    frame.configure(style="Unselected.TFrame")
        self.status_var.set("Selection cleared")
    
    def detect_faces(self, image_path):
        if self.exit_event.is_set():
            return False
            
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            # Convert to RGB for MTCNN
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            boxes, _ = self.mtcnn.detect(image_rgb)
            
            # Return True if no faces detected
            return boxes is None or len(boxes) == 0
            
        except Exception as e:
            print(f"Error detecting faces in {image_path}: {e}")
            return False
    
    def process_images_batch(self, image_paths):
        if self.exit_event.is_set():
            return []
            
        results = []
        for img_path in image_paths:
            if self.exit_event.is_set():
                break
            results.append(self.detect_faces(img_path))
        return results
    
    def verify_dataset(self):
        """Check all images in the dataset folder to make sure they're properly loaded"""
        if not self.dataset_path.get() or not os.path.isdir(self.dataset_path.get()):
            messagebox.showerror("Error", "Please select a valid dataset folder first")
            return
            
        # Find all images in the directory
        path = self.dataset_path.get()
        all_images = set(glob.glob(os.path.join(path, "*.jpg")))
        current_loaded = set(self.image_files)
        
        # Find missing images
        missing_images = all_images - current_loaded
        
        if missing_images:
            # Add missing images to our list
            self.image_files.extend(list(missing_images))
            self.image_files.sort()  # Re-sort to ensure order
            self.total_images = len(self.image_files)
            self.update_stats()
            
            messagebox.showinfo("Dataset Verification", 
                f"Found {len(missing_images)} images that weren't loaded before.\n"
                f"Total images now: {self.total_images}")
            
            # Refresh the current view
            self.display_current_images()
        else:
            messagebox.showinfo("Dataset Verification", 
                "All images in the folder are properly loaded.")
    
    def load_dataset(self):
        # Stop any existing processing
        self.exit_event.set()
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
        self.exit_event.clear()
        
        path = self.dataset_path.get()
        if not path or not os.path.isdir(path):
            messagebox.showerror("Error", "Please select a valid dataset folder")
            return
        
        # Create backup folder if it doesn't exist
        self.backup_folder = os.path.join(path, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.backup_folder, exist_ok=True)
        
        # Find all JPG files
        self.image_files = sorted(glob.glob(os.path.join(path, "*.jpg")))
        self.total_images = len(self.image_files)
        self.current_index = 0
        self.processed_count = 0
        self.deleted_count = 0
        self.face_detection_results = {}
        self.selected_images.clear()
        self.deleted_images = []
        
        if self.total_images > 0:
            self.status_var.set(f"Loaded {self.total_images} images from {path}")
            self.update_stats()
            self.display_current_images()
            
            # Start face detection in background if enabled
            if self.enable_suggestions.get() and self.mtcnn is not None:
                self.detection_thread = threading.Thread(target=self.background_face_detection, daemon=True)
                self.detection_thread.start()
        else:
            self.status_var.set(f"No images found in {path}")
            messagebox.showinfo("Info", "No JPG images found in the selected folder")
    
    def background_face_detection(self):
        batch_size = 8
        try:
            for i in range(0, len(self.image_files), batch_size):
                if self.exit_event.is_set():
                    break
                    
                batch = self.image_files[i:i + batch_size]
                results = self.process_images_batch(batch)
                
                for img_path, is_no_face in zip(batch, results):
                    if self.exit_event.is_set():
                        break
                    self.face_detection_results[img_path] = is_no_face
                    
                # Update display if current batch is visible
                if i <= self.current_index < i + batch_size and not self.exit_event.is_set():
                    self.root.after(0, self.display_current_images)
                
                # Update status periodically
                if i % (batch_size * 4) == 0:
                    self.root.after(0, lambda: self.status_var.set(
                        f"Processing face detection: {min(i + batch_size, len(self.image_files))}/{len(self.image_files)}"
                    ))
                    
            if not self.exit_event.is_set():
                self.root.after(0, lambda: self.status_var.set(f"Face detection complete. Found {sum(self.face_detection_results.values())} images without faces"))
        except Exception as e:
            print(f"Error in background face detection: {e}")
    
    def display_current_images(self):
        if not self.image_files:
            return
        
        # Get current batch of images
        start_idx = self.current_index
        end_idx = min(start_idx + self.images_per_page.get(), len(self.image_files))
        self.current_images = self.image_files[start_idx:end_idx]
        
        # Clear all canvases
        for canvas in self.image_widgets:
            canvas.delete("all")
        
        # Reset all frames to unselected style
        for frame in self.image_frames:
            frame.configure(style="Unselected.TFrame")
        
        # Display images
        for i, img_path in enumerate(self.current_images):
            try:
                # Load and resize image
                image = Image.open(img_path)
                canvas = self.image_widgets[i]
                frame = self.image_frames[i]
                canvas_width = canvas.winfo_width()
                canvas_height = canvas.winfo_height()
                
                # Resize image to fit canvas
                img_width, img_height = image.size
                scale = min(canvas_width/img_width, canvas_height/img_height) * 0.95  # 5% margin
                new_size = (int(img_width * scale), int(img_height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(image)
                canvas.image = photo
                
                # Calculate position to center image
                x = canvas_width // 2
                y = canvas_height // 2
                canvas.create_image(x, y, image=photo)
                
                # Set frame style based on selection/suggestion status
                if img_path in self.selected_images:
                    frame.configure(style="Selected.TFrame")
                elif self.enable_suggestions.get() and self.face_detection_results.get(img_path, False):
                    frame.configure(style="Suggested.TFrame")
                
                # Add image label (optional)
                filename = os.path.basename(img_path)
                canvas.create_text(10, 10, text=filename, anchor=tk.NW, fill="white")
                
                # Add image dimensions
                size_text = f"{img_width}x{img_height}"
                canvas.create_text(10, 30, text=size_text, anchor=tk.NW, fill="white")
                
            except Exception as e:
                print(f"Error displaying image {img_path}: {e}")
                # Show error message in canvas
                canvas = self.image_widgets[i]
                canvas.create_text(canvas.winfo_width()//2, canvas.winfo_height()//2, 
                                  text=f"Error: {str(e)}", fill="red")
        
        # Update progress
        self.progress_var.set(f"{start_idx + 1}-{end_idx}/{self.total_images}")
    
    def backup_file(self, file_path):
        """Create a backup of the file before deletion"""
        try:
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                backup_path = os.path.join(self.backup_folder, filename)
                shutil.copy2(file_path, backup_path)
                return backup_path
        except Exception as e:
            print(f"Error backing up {file_path}: {e}")
        return None

    def delete_selected(self, event=None):
        deleted_count = 0
        deleted_batch = []
        
        try:
            # Delete any selected images
            if self.selected_images:
                for img_path in list(self.selected_images):  # Use list to make a copy for safe iteration
                    # Delete image and corresponding JSON
                    json_path = os.path.splitext(img_path)[0] + ".json"
                    
                    # Backup files before deletion
                    img_backup = self.backup_file(img_path)
                    json_backup = self.backup_file(json_path)
                    
                    # If backup successful, proceed with deletion
                    if img_backup:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                            deleted_count += 1
                        if os.path.exists(json_path):
                            os.remove(json_path)
                        
                        # Track deleted image for potential recovery
                        deleted_batch.append({
                            'image_path': img_path,
                            'json_path': json_path,
                            'image_backup': img_backup,
                            'json_backup': json_backup
                        })
                        
                        self.deleted_count += 1
                        
                        # Remove from lists
                        if img_path in self.image_files:
                            self.image_files.remove(img_path)
                        self.selected_images.remove(img_path)
                
                if deleted_count > 0:
                    self.status_var.set(f"Deleted {deleted_count} images (saved to backup)")
                    # Add this batch to deleted_images for potential recovery
                    self.deleted_images.append(deleted_batch)
            
            # Update stats
            self.update_stats()
            
            # Go to next page, but don't skip images
            # Instead of calling next_page which would increment by images_per_page,
            # just display the current page which will show the next batch of images
            # since we deleted from the current page
            self.display_current_images()
            
            # If nothing was deleted or if user wants to advance to next page anyway
            # call next_page to move forward
            if deleted_count == 0:
                self.next_page()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete images: {str(e)}")
    
    def undo_last_deletion(self, event=None):
        """Recover the last batch of deleted images"""
        if not self.deleted_images:
            messagebox.showinfo("Info", "No deleted images to recover")
            return
        
        try:
            # Get the last deleted batch
            last_batch = self.deleted_images.pop()
            recovered_count = 0
            
            for item in last_batch:
                # Restore image file
                if os.path.exists(item['image_backup']):
                    shutil.copy2(item['image_backup'], item['image_path'])
                    recovered_count += 1
                    
                    # Add back to image_files list
                    if item['image_path'] not in self.image_files:
                        self.image_files.append(item['image_path'])
                
                # Restore JSON file if it exists
                if item['json_backup'] and os.path.exists(item['json_backup']):
                    shutil.copy2(item['json_backup'], item['json_path'])
            
            # Re-sort image files
            self.image_files.sort()
            
            # Update stats and display
            self.deleted_count -= recovered_count
            self.update_stats()
            self.display_current_images()
            
            messagebox.showinfo("Recovery", f"Recovered {recovered_count} images from the last deletion")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to recover images: {str(e)}")
    
    def next_page(self, event=None):
        if not self.image_files:
            return
        
        if self.current_index + self.images_per_page.get() < len(self.image_files):
            self.current_index += self.images_per_page.get()
            self.selected_images.clear()
            self.display_current_images()
            self.status_var.set(f"Showing images {self.current_index + 1} to {min(self.current_index + self.images_per_page.get(), len(self.image_files))}")
        else:
            if len(self.image_files) > 0:
                self.status_var.set("Reached the end of the dataset")
                messagebox.showinfo("End of Dataset", "You've reached the end of the dataset.")
    
    def prev_page(self, event=None):
        if not self.image_files:
            return
        
        if self.current_index > 0:
            self.current_index = max(0, self.current_index - self.images_per_page.get())
            self.selected_images.clear()
            self.display_current_images()
            self.status_var.set(f"Showing images {self.current_index + 1} to {min(self.current_index + self.images_per_page.get(), len(self.image_files))}")
    
    def update_stats(self):
        if hasattr(self, 'stats_label'):  # Check if stats_label exists
            self.stats_label.config(
                text=f"Total: {self.total_images} | Remaining: {len(self.image_files)} | Deleted: {self.deleted_count}"
            )
    
    def browse_folder(self):
        folder = filedialog.askdirectory(initialdir=os.path.expanduser("~"))
        if folder:
            self.dataset_path.set(folder)
            self.status_var.set(f"Selected folder: {folder}")
    
    def on_close(self):
        # Signal threads to stop
        self.exit_event.set()
        
        # Wait for processing thread to finish
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
            
        if messagebox.askokcancel("Quit", 
            f"You've processed {self.processed_count} images and deleted {self.deleted_count}. Are you sure you want to quit?"):
            self.root.destroy()


def main():
    root = tk.Tk()
    app = ImageFilterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main() 