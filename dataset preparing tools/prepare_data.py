"""
Prepare Anti-Spoofing Dataset for YOLO Training

This script prepares clean images (without visual markers) for YOLO training,
creating proper annotation files without relying on color-based markers.
"""

import os
import json
import shutil
import random
from pathlib import Path
import cv2
import numpy as np

# Define constants
OUTPUT_DATASET = "yolo_dataset"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def ask_for_folders():
    """Ask the user for the real and fake folder locations using command line input"""
    print("\nPlease enter the full path to the folder containing REAL face images:")
    print("Example: E:/projects/anti-spoofing deep learning Model/dataset/real")
    
    real_folder = input("Real faces folder path: ").strip()
    if not os.path.exists(real_folder):
        print(f"Error: Folder '{real_folder}' does not exist.")
        exit()
    
    print(f"Selected real folder: {real_folder}")
    
    print("\nPlease enter the full path to the folder containing FAKE face images:")
    print("Example: E:/projects/anti-spoofing deep learning Model/dataset/fake")
    
    fake_folder = input("Fake faces folder path: ").strip()
    if not os.path.exists(fake_folder):
        print(f"Error: Folder '{fake_folder}' does not exist.")
        exit()
    
    print(f"Selected fake folder: {fake_folder}")
    
    return real_folder, fake_folder

def create_yolo_directory_structure():
    """Create the directory structure required for YOLO training"""
    print("Creating YOLO directory structure...")
    
    # Remove old dataset if exists
    if os.path.exists(OUTPUT_DATASET):
        print(f"Removing old dataset at {OUTPUT_DATASET}")
        shutil.rmtree(OUTPUT_DATASET)
    
    os.makedirs(f"{OUTPUT_DATASET}/images/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_DATASET}/images/val", exist_ok=True)
    os.makedirs(f"{OUTPUT_DATASET}/images/test", exist_ok=True)
    os.makedirs(f"{OUTPUT_DATASET}/labels/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_DATASET}/labels/val", exist_ok=True)
    os.makedirs(f"{OUTPUT_DATASET}/labels/test", exist_ok=True)
    
    print("Directory structure created successfully.")

def detect_face(image_path):
    """
    Detect face in the image and return its coordinates
    Returns None if no face is detected
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    # Get image dimensions
    img_height, img_width = img.shape[:2]
    
    # Try to use DNN face detector first
    try:
        # Check if model files exist, download if not
        prototxt_path = "deploy.prototxt"
        model_path = "res10_300x300_ssd_iter_140000.caffemodel"
        
        if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
            print("Downloading face detection model...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                prototxt_path
            )
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                model_path
            )
        
        # Initialize DNN face detector
        face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        
        # Prepare image for DNN
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        # Detect faces
        face_net.setInput(blob)
        detections = face_net.forward()
        
        # Get the face with highest confidence
        best_confidence = 0
        best_box = None
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:  # Minimum confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([img_width, img_height, img_width, img_height])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                x2 = max(0, min(x2, img_width))
                y2 = max(0, min(y2, img_height))
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_box = (x1, y1, x2, y2)
        
        if best_box is not None:
            return best_box, (img_width, img_height)
    
    except Exception as e:
        print(f"DNN face detection failed: {e}. Falling back to Haar Cascade.")
    
    # Fallback to Haar Cascade if DNN fails
    try:
        # Load Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            # Get the largest face
            largest_area = 0
            largest_face = None
            
            for (x, y, w, h) in faces:
                area = w * h
                if area > largest_area:
                    largest_area = area
                    largest_face = (x, y, w, h)
            
            x, y, w, h = largest_face
            return (x, y, x+w, y+h), (img_width, img_height)
    
    except Exception as e:
        print(f"Haar Cascade face detection failed: {e}")
    
    return None, None

def create_yolo_annotation(box, img_size, is_real):
    """
    Create YOLO format annotation from face coordinates
    YOLO format: class_id x_center y_center width height (normalized)
    """
    if box is None or img_size is None:
        return None
    
    x1, y1, x2, y2 = box
    img_width, img_height = img_size
    
    # Calculate normalized center coordinates and dimensions
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # Class ID: 0 for fake, 1 for real
    class_id = 1 if is_real else 0
    
    # Ensure values are within [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_dataset(real_folder, fake_folder):
    """Process the dataset and create YOLO format annotations"""
    print("Processing dataset...")
    
    all_data = []
    
    # Process real images
    print("Processing real images...")
    real_count = 0
    for img_file in Path(real_folder).glob("*.jpg"):
        all_data.append((str(img_file), True))
        real_count += 1
    
    # Process fake images
    print("Processing fake images...")
    fake_count = 0
    for img_file in Path(fake_folder).glob("*.jpg"):
        all_data.append((str(img_file), False))
        fake_count += 1
    
    print(f"Found {len(all_data)} total images: {real_count} real, {fake_count} fake")
    
    if len(all_data) == 0:
        print("No images found! Make sure your folders contain .jpg images.")
        
        # Check if there are any files at all
        real_files = list(os.listdir(real_folder))
        fake_files = list(os.listdir(fake_folder))
        
        if real_files:
            print(f"Files found in real folder: {', '.join(real_files[:5])}" + 
                  (f" and {len(real_files)-5} more" if len(real_files) > 5 else ""))
        else:
            print("No files found in real folder.")
            
        if fake_files:
            print(f"Files found in fake folder: {', '.join(fake_files[:5])}" + 
                  (f" and {len(fake_files)-5} more" if len(fake_files) > 5 else ""))
        else:
            print("No files found in fake folder.")
            
        return
    
    # Shuffle the data
    random.shuffle(all_data)
    
    # Split into train, validation, and test sets
    train_count = int(len(all_data) * TRAIN_RATIO)
    val_count = int(len(all_data) * VAL_RATIO)
    
    train_data = all_data[:train_count]
    val_data = all_data[train_count:train_count + val_count]
    test_data = all_data[train_count + val_count:]
    
    print(f"Dataset split: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test images")
    
    # Process each set
    process_data_subset(train_data, "train")
    process_data_subset(val_data, "val")
    process_data_subset(test_data, "test")
    
    # Create dataset.yaml file
    create_dataset_yaml(len(train_data), len(val_data), len(test_data))
    
    print("Dataset processing completed.")

def process_data_subset(data_subset, subset_name):
    """Process a subset of the data (train, val, or test)"""
    print(f"Processing {subset_name} set...")
    
    successful_count = 0
    skipped_count = 0
    
    for img_path, is_real in data_subset:
        try:
            # Create a new filename based on class and original name
            img_filename = os.path.basename(img_path)
            base_name = os.path.splitext(img_filename)[0]
            prefix = "real" if is_real else "fake"
            new_img_filename = f"{prefix}_{base_name}.jpg"
            
            # Destination paths
            dst_img_path = os.path.join(OUTPUT_DATASET, "images", subset_name, new_img_filename)
            dst_label_path = os.path.join(OUTPUT_DATASET, "labels", subset_name, new_img_filename.replace(".jpg", ".txt"))
            
            # Detect face and create YOLO annotation
            box, img_size = detect_face(img_path)
            if box is None:
                print(f"No face detected in {img_path}")
                skipped_count += 1
                continue
            
            yolo_annotation = create_yolo_annotation(box, img_size, is_real)
            if yolo_annotation is None:
                print(f"Failed to create annotation for {img_path}")
                skipped_count += 1
                continue
            
            # Copy image to destination
            shutil.copy2(img_path, dst_img_path)
            
            # Save YOLO annotation
            with open(dst_label_path, "w") as f:
                f.write(yolo_annotation)
            
            successful_count += 1
            
            # Progress update
            if successful_count % 10 == 0:
                print(f"Processed {successful_count} images...")
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            skipped_count += 1
    
    print(f"Processed {successful_count} images for {subset_name} set. Skipped {skipped_count} images.")

def create_dataset_yaml(train_count, val_count, test_count):
    """Create the dataset.yaml file for YOLO training"""
    yaml_content = f"""# YOLOv8 Anti-Spoofing Dataset Configuration
path: {os.path.abspath(OUTPUT_DATASET)}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: fake
  1: real

# Statistics
nc: 2  # number of classes
train_count: {train_count}
val_count: {val_count}
test_count: {test_count}
"""
    
    with open(f"{OUTPUT_DATASET}/dataset.yaml", "w") as f:
        f.write(yaml_content)
    
    print(f"Created dataset.yaml at {OUTPUT_DATASET}/dataset.yaml")

def suggest_folders():
    """Suggest possible folders in the current directory"""
    current_dir = os.getcwd()
    print(f"\nLooking for possible dataset folders in: {current_dir}")
    
    # Look for common dataset folder names
    possible_folders = []
    
    # Check if dataset_for_test exists
    dataset_test = os.path.join(current_dir, "dataset_for_test")
    if os.path.exists(dataset_test):
        real_folder = os.path.join(dataset_test, "real")
        fake_folder = os.path.join(dataset_test, "fake")
        
        if os.path.exists(real_folder) and os.path.exists(fake_folder):
            print("\nFound dataset_for_test with real and fake subfolders!")
            print(f"  Real folder: {real_folder}")
            print(f"  Fake folder: {fake_folder}")
            
            use_found = input("\nUse these folders? (y/n): ").strip().lower()
            if use_found == 'y':
                return real_folder, fake_folder
    
    # Look for any folder with "dataset" in the name
    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        if os.path.isdir(item_path) and "dataset" in item.lower():
            # Check if it has real and fake subfolders
            real_folder = os.path.join(item_path, "real")
            fake_folder = os.path.join(item_path, "fake")
            
            if os.path.exists(real_folder) and os.path.exists(fake_folder):
                print(f"\nFound potential dataset folder: {item}")
                print(f"  Real folder: {real_folder}")
                print(f"  Fake folder: {fake_folder}")
                
                use_found = input("\nUse these folders? (y/n): ").strip().lower()
                if use_found == 'y':
                    return real_folder, fake_folder
    
    print("\nNo suitable dataset folders found automatically.")
    return None, None

if __name__ == "__main__":
    print("Starting dataset preparation for YOLO training...")
    print("WARNING: This script expects clean images without visual markers!")
    print("Make sure your dataset contains original images without boxes or scares.")
    
    # Ask for confirmation
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        exit()
    
    # Try to suggest folders first
    real_folder, fake_folder = suggest_folders()
    
    # If no folders were found or user rejected suggestions, ask manually
    if real_folder is None or fake_folder is None:
        real_folder, fake_folder = ask_for_folders()
    
    # Create directory structure
    create_yolo_directory_structure()
    
    # Process the dataset
    process_dataset(real_folder, fake_folder)
    
    print("Dataset preparation completed.") 