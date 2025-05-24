import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance
import random
from datetime import datetime
from tkinterdnd2 import DND_FILES, TkinterDnD
import threading
import queue
from pathlib import Path

class ImageAugmentor:
    def __init__(self):
        self.root = TkinterDnD.Tk()
        self.root.title("Image Augmentor")
        self.root.geometry("1000x600")
        
        # Initialize variables
        self.input_folders = []
        self.output_folder = ""
        self.is_processing = False
        self.total_images = 0
        self.processed_images = 0
        self.processing_queue = queue.Queue()
        
        # Create GUI
        self.create_gui()
        
    def create_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Input folders
        left_frame = ttk.LabelFrame(main_frame, text="Input Folders", padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Drop area
        self.drop_area = tk.Text(left_frame, height=20, width=50)
        self.drop_area.pack(fill=tk.BOTH, expand=True, pady=5)
        self.drop_area.insert('1.0', "Drag and drop folders here...\n\nSupported formats:\n.jpg, .jpeg, .png, .bmp")
        self.drop_area.configure(state='disabled')
        
        # Enable drag and drop
        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind('<<Drop>>', self.on_drop)
        
        # Buttons frame
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Add Folders", command=self.add_folders).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear All", command=self.clear_folders).pack(side=tk.LEFT, padx=5)
        
        # Right panel - Settings and controls
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)
        
        # Output folder selection
        output_frame = ttk.LabelFrame(right_frame, text="Output Folder", padding="5")
        output_frame.pack(fill=tk.X, pady=5)
        
        self.output_path = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_path, state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(output_frame, text="Select", command=self.select_output).pack(side=tk.RIGHT, padx=5)
        
        # Augmentation settings
        settings_frame = ttk.LabelFrame(right_frame, text="Augmentation Settings", padding="5")
        settings_frame.pack(fill=tk.X, pady=5)
        
        # Number of variations per image
        var_frame = ttk.Frame(settings_frame)
        var_frame.pack(fill=tk.X, pady=5)
        ttk.Label(var_frame, text="Variations per image:").pack(side=tk.LEFT)
        self.min_variations = tk.IntVar(value=4)
        self.max_variations = tk.IntVar(value=8)
        ttk.Spinbox(var_frame, from_=1, to=10, width=5, textvariable=self.min_variations).pack(side=tk.LEFT, padx=5)
        ttk.Label(var_frame, text="to").pack(side=tk.LEFT)
        ttk.Spinbox(var_frame, from_=1, to=10, width=5, textvariable=self.max_variations).pack(side=tk.LEFT, padx=5)
        
        # Effect strength settings
        strength_frame = ttk.LabelFrame(settings_frame, text="Effect Strength", padding="5")
        strength_frame.pack(fill=tk.X, pady=5)
        
        # Light adjustment range
        light_frame = ttk.Frame(strength_frame)
        light_frame.pack(fill=tk.X, pady=2)
        ttk.Label(light_frame, text="Light adjustment (±%):").pack(side=tk.LEFT)
        self.light_strength = tk.DoubleVar(value=20.0)
        ttk.Entry(light_frame, textvariable=self.light_strength, width=8).pack(side=tk.RIGHT)
        
        # Contrast adjustment range
        contrast_frame = ttk.Frame(strength_frame)
        contrast_frame.pack(fill=tk.X, pady=2)
        ttk.Label(contrast_frame, text="Contrast adjustment (±%):").pack(side=tk.LEFT)
        self.contrast_strength = tk.DoubleVar(value=20.0)
        ttk.Entry(contrast_frame, textvariable=self.contrast_strength, width=8).pack(side=tk.RIGHT)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(right_frame, text="Progress", padding="5")
        progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.status_var, wraplength=300).pack(fill=tk.X)
        
        # Process button
        self.process_btn = ttk.Button(right_frame, text="Start Processing", command=self.toggle_processing)
        self.process_btn.pack(pady=10, fill=tk.X)
        
    def on_drop(self, event):
        self.drop_area.configure(state='normal')
        
        # Get dropped paths
        paths = self.root.tk.splitlist(event.data)
        
        # Process each dropped item
        for path in paths:
            path = path.strip('"{}')
            if os.path.isdir(path) and path not in self.input_folders:
                self.input_folders.append(path)
        
        self.update_folder_list()
        self.update_total_images()
        self.drop_area.configure(state='disabled')
    
    def add_folders(self):
        folders = filedialog.askdirectory(multiple=True)
        if folders:
            for folder in folders:
                if folder not in self.input_folders:
                    self.input_folders.append(folder)
            self.update_folder_list()
            self.update_total_images()
    
    def clear_folders(self):
        self.input_folders.clear()
        self.update_folder_list()
        self.update_total_images()
    
    def update_folder_list(self):
        self.drop_area.configure(state='normal')
        self.drop_area.delete('1.0', tk.END)
        
        if self.input_folders:
            self.drop_area.insert('1.0', "Added folders:\n\n")
            for i, folder in enumerate(self.input_folders, 1):
                self.drop_area.insert(tk.END, f"{i}. {folder}\n")
        else:
            self.drop_area.insert('1.0', "Drag and drop folders here...\n\nSupported formats:\n.jpg, .jpeg, .png, .bmp")
        
        self.drop_area.configure(state='disabled')
    
    def update_total_images(self):
        self.total_images = 0
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for folder in self.input_folders:
            for root, _, files in os.walk(folder):
                self.total_images += sum(1 for f in files if os.path.splitext(f)[1].lower() in supported_formats)
        
        self.status_var.set(f"Found {self.total_images} images")
    
    def select_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder = folder
            self.output_path.set(folder)
    
    def toggle_processing(self):
        if not self.is_processing:
            if not self.input_folders:
                messagebox.showwarning("Warning", "Please add at least one input folder.")
                return
            if not self.output_folder:
                messagebox.showwarning("Warning", "Please select an output folder.")
                return
            
            self.start_processing()
        else:
            self.stop_processing()
    
    def start_processing(self):
        self.is_processing = True
        self.process_btn.config(text="Stop Processing")
        self.processed_images = 0
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_images)
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def stop_processing(self):
        self.is_processing = False
        self.process_btn.config(text="Start Processing")
        self.status_var.set("Processing stopped")
    
    def apply_augmentation(self, image):
        """Apply random augmentation to the image"""
        try:
            # Convert to PIL Image for easier processing
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Random light adjustment
            if random.random() < 0.7:
                factor = 1.0 + random.uniform(-self.light_strength.get()/100, 
                                            self.light_strength.get()/100)
                image = ImageEnhance.Brightness(image).enhance(factor)
            
            # Random contrast adjustment
            if random.random() < 0.7:
                factor = 1.0 + random.uniform(-self.contrast_strength.get()/100, 
                                            self.contrast_strength.get()/100)
                image = ImageEnhance.Contrast(image).enhance(factor)
            
            # Random rotation
            if random.random() < 0.7:
                angle = random.uniform(-12, 12)
                image = image.rotate(angle, expand=True)
            
            # Random aspect ratio change
            if random.random() < 0.5:
                w, h = image.size
                stretch_factor = random.uniform(0.98, 1.02)
                new_width = int(w * stretch_factor)
                image = image.resize((new_width, h), Image.Resampling.LANCZOS)
            
            # Convert back to OpenCV format
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return image
            
        except Exception as e:
            print(f"Error in augmentation: {str(e)}")
            return None
    
    def process_images(self):
        try:
            supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
            
            for folder in self.input_folders:
                if not self.is_processing:
                    break
                
                for root, _, files in os.walk(folder):
                    for file in files:
                        if not self.is_processing:
                            break
                        
                        if os.path.splitext(file)[1].lower() in supported_formats:
                            input_path = os.path.join(root, file)
                            
                            try:
                                # Read image
                                image = cv2.imread(input_path)
                                if image is None:
                                    continue
                                
                                # Create variations
                                num_variations = random.randint(
                                    self.min_variations.get(),
                                    self.max_variations.get()
                                )
                                
                                # Create relative output path
                                rel_path = os.path.relpath(root, folder)
                                output_dir = os.path.join(self.output_folder, rel_path)
                                os.makedirs(output_dir, exist_ok=True)
                                
                                # Generate variations
                                for i in range(num_variations):
                                    if not self.is_processing:
                                        break
                                        
                                    # Apply random augmentations
                                    augmented = self.apply_augmentation(image)
                                    if augmented is None:
                                        continue
                                    
                                    # Save augmented image
                                    base_name = os.path.splitext(file)[0]
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                    output_path = os.path.join(output_dir, 
                                                            f"{base_name}_aug_{i}_{timestamp}.jpg")
                                    
                                    cv2.imwrite(output_path, augmented)
                                
                                self.processed_images += 1
                                progress = (self.processed_images / self.total_images) * 100
                                self.progress_var.set(progress)
                                self.status_var.set(f"Processing: {input_path}")
                                
                            except Exception as e:
                                print(f"Error processing {input_path}: {str(e)}")
                                continue
            
            if self.is_processing:
                self.is_processing = False
                self.process_btn.config(text="Start Processing")
                self.status_var.set("Processing complete!")
                messagebox.showinfo("Complete", f"Processed {self.processed_images} images")
                
        except Exception as e:
            self.is_processing = False
            self.process_btn.config(text="Start Processing")
            self.status_var.set("Error occurred!")
            messagebox.showerror("Error", str(e))
    
    def run(self):
        self.root.mainloop()

def main():
    app = ImageAugmentor()
    app.run()

if __name__ == "__main__":
    main() 