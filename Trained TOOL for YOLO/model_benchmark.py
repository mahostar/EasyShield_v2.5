"""
Anti-Spoofing Model Benchmarking System

A comprehensive benchmarking and visualization tool for comparing multiple
anti-spoofing models, generating performance metrics, and creating publication-quality
visualizations for academic documentation.
"""

import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
import shutil
import json
import time
from tqdm import tqdm
from scipy import stats
import torch.nn.functional as F

# Optional GradCAM import
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Warning: pytorch-grad-cam not available. Some visualizations will be disabled.")

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QLineEdit, QMessageBox, QComboBox,
    QSpinBox, QFileDialog, QProgressBar, QScrollArea, QFrame,
    QGroupBox, QFormLayout, QTabWidget, QSplitter, QDesktopWidget,
    QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QFont, QIcon, QPixmap, QPalette, QColor, QDesktopServices
from ultralytics import YOLO

# Suppress excessive logging
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.WARNING)

# Configure matplotlib for non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid thread issues

class ModelInfo:
    """Class to hold information about each model being benchmarked"""
    def __init__(self):
        self.version = ""
        self.weight_path = ""
        self.metrics_folder = ""
        self.model = None
        self.results = {}
        self.inference_times = []
        self.predictions = []
        self.confidences = []
        self.features = []
        self.flip_labels = False  # New flag for label flipping
        
    def clear_memory(self):
        """Clear model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class BenchmarkWorker(QThread):
    """Worker thread for running benchmarks"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, models, test_dataset_path, output_path):
        super().__init__()
        self.models = models
        self.test_dataset_path = Path(test_dataset_path)
        self.output_path = Path(output_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.temp_path = self.output_path / 'temp'
        self.temp_path.mkdir(exist_ok=True)
        self.total_benchmark_path = self.output_path / 'total_benchmark'
        self.total_benchmark_path.mkdir(exist_ok=True)
    
    def run(self):
        try:
            self.status.emit("Starting benchmark process...")
            self.benchmark_models()
            self.status.emit("Benchmark complete!")
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
    
    def load_test_dataset(self):
        """Load and prepare test dataset"""
        real_path = self.test_dataset_path / 'real'
        fake_path = self.test_dataset_path / 'fake'
        
        if not real_path.exists() or not fake_path.exists():
            raise ValueError("Test dataset must contain 'real' and 'fake' folders")
        
        real_images = list(real_path.glob('*.jpg')) + list(real_path.glob('*.png'))
        fake_images = list(fake_path.glob('*.jpg')) + list(fake_path.glob('*.png'))
        
        return real_images, fake_images
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Resize to model input size
        img = cv2.resize(img, (640, 640))
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize and convert to tensor
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        
        return img
    
    def benchmark_models(self):
        """Run benchmarking process for all models"""
        # Load test dataset
        self.status.emit("Loading test dataset...")
        real_images, fake_images = self.load_test_dataset()
        total_images = len(real_images) + len(fake_images)
        
        # Process each model sequentially
        for i, (model_name, model_info) in enumerate(self.models.items()):
            self.status.emit(f"Processing model: {model_name}")
            
            # Create model-specific output directory
            model_path = Path(self.output_path) / model_name
            model_path.mkdir(exist_ok=True)
            
            # Create temporary results directory for this model
            model_temp_path = self.temp_path / model_name
            model_temp_path.mkdir(exist_ok=True)
            
            try:
                # Load model
                self.status.emit(f"Loading model: {model_name}")
                model = YOLO(model_info.weight_path)
                model.to(self.device)
                model.eval()
                
                # Initialize results storage
                predictions = []
                confidences = []
                true_labels = []
                inference_times = []
                features = []
                
                # Process real images
                self.status.emit(f"Processing real images for {model_name}")
                for img_path in real_images:
                    try:
                        img = self.preprocess_image(img_path)
                        img = img.to(self.device)
                        
                        # Time inference
                        start_time = time.perf_counter()
                        with torch.no_grad():
                            results = model(img, verbose=False)
                        inference_time = (time.perf_counter() - start_time) * 1000
                        
                        if len(results) > 0 and len(results[0].boxes) > 0:
                            conf = float(results[0].boxes.conf[0])
                            cls = int(results[0].boxes.cls[0])
                            
                            # Apply label flipping if enabled for this model
                            if model_info.flip_labels:
                                cls = 1 - cls  # Flip 0 to 1 and 1 to 0
                            
                            predictions.append(cls)
                            confidences.append(conf)
                            true_labels.append(1)  # Real = 1
                            inference_times.append(inference_time)
                            
                            if hasattr(results[0], 'features'):
                                features.append(results[0].features[0].cpu().numpy())
                        
                        # Save intermediate results
                        np.save(model_temp_path / 'predictions.npy', np.array(predictions))
                        np.save(model_temp_path / 'confidences.npy', np.array(confidences))
                        np.save(model_temp_path / 'true_labels.npy', np.array(true_labels))
                        np.save(model_temp_path / 'inference_times.npy', np.array(inference_times))
                        if features:
                            np.save(model_temp_path / 'features.npy', np.array(features))
                        
                    except Exception as e:
                        print(f"Error processing image {img_path}: {str(e)}")
                        continue
                
                # Process fake images
                self.status.emit(f"Processing fake images for {model_name}")
                for img_path in fake_images:
                    try:
                        img = self.preprocess_image(img_path)
                        img = img.to(self.device)
                        
                        start_time = time.perf_counter()
                        with torch.no_grad():
                            results = model(img, verbose=False)
                        inference_time = (time.perf_counter() - start_time) * 1000
                        
                        if len(results) > 0 and len(results[0].boxes) > 0:
                            conf = float(results[0].boxes.conf[0])
                            cls = int(results[0].boxes.cls[0])
                            
                            # Apply label flipping if enabled for this model
                            if model_info.flip_labels:
                                cls = 1 - cls  # Flip 0 to 1 and 1 to 0
                            
                            predictions.append(cls)
                            confidences.append(conf)
                            true_labels.append(0)  # Fake = 0
                            inference_times.append(inference_time)
                            
                            if hasattr(results[0], 'features'):
                                features.append(results[0].features[0].cpu().numpy())
                        
                        # Save intermediate results
                        np.save(model_temp_path / 'predictions.npy', np.array(predictions))
                        np.save(model_temp_path / 'confidences.npy', np.array(confidences))
                        np.save(model_temp_path / 'true_labels.npy', np.array(true_labels))
                        np.save(model_temp_path / 'inference_times.npy', np.array(inference_times))
                        if features:
                            np.save(model_temp_path / 'features.npy', np.array(features))
                            
                    except Exception as e:
                        print(f"Error processing image {img_path}: {str(e)}")
                        continue
                
                # Calculate metrics
                predictions = np.array(predictions)
                confidences = np.array(confidences)
                true_labels = np.array(true_labels)
                inference_times = np.array(inference_times)
                if features:
                    features = np.array(features)
                
                model_info.results = {
                    'predictions': predictions,
                    'confidences': confidences,
                    'true_labels': true_labels,
                    'inference_times': inference_times,
                    'features': features,
                    'accuracy_score': accuracy_score(true_labels, predictions),
                    'precision_score': precision_score(true_labels, predictions),
                    'recall_score': recall_score(true_labels, predictions),
                    'f1_score': f1_score(true_labels, predictions),
                    'auc_score': roc_auc_score(true_labels, confidences)
                }
                
                # Calculate EER
                fpr, tpr, thresholds = roc_curve(true_labels, confidences)
                fnr = 1 - tpr
                eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
                model_info.results['eer'] = eer
                
                # Generate model-specific visualizations
                self.status.emit(f"Generating visualizations for {model_name}")
                
                from visualization_utils import (
                    plot_confusion_matrices,
                    plot_feature_space,
                    create_prediction_cards
                )
                
                # Confusion matrix
                plot_confusion_matrices({model_name: model_info.results}, model_path)
                
                # Feature space visualization if features are available
                if len(features) > 0:
                    plot_feature_space({model_name: model_info.results}, model_path)
                
                # Create prediction cards with example images
                create_prediction_cards(
                    model_info.results,
                    real_images,
                    fake_images,
                    model_path,
                    model_name
                )
                
                # Grad-CAM visualization
                if len(real_images) > 0:
                    try:
                        # Instead of GradCAM, save a sample image with prediction overlay
                        sample_img_path = real_images[0]
                        img = cv2.imread(str(sample_img_path))
                        img = cv2.resize(img, (640, 640))
                        
                        # Run inference to get prediction
                        input_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                        input_tensor = input_tensor.unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            results = model(input_tensor, verbose=False)
                        
                        # Draw prediction on image
                        if len(results) > 0 and len(results[0].boxes) > 0:
                            conf = float(results[0].boxes.conf[0])
                            cls = int(results[0].boxes.cls[0])
                            label = "Real" if cls == 1 else "Fake"
                            color = (0, 255, 0) if cls == 1 else (0, 0, 255)  # Green for real, red for fake
                            
                            # Add text with prediction and confidence
                            cv2.putText(img, f"{label}: {conf:.2f}", (20, 40), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            
                            # Save the visualization
                            cv2.imwrite(str(model_path / 'sample_prediction.png'), img)
                            
                            # Only attempt GradCAM if available
                            if GRADCAM_AVAILABLE:
                                try:
                                    # GradCAM visualization code here
                                    grad_cam = GradCAM(model=model)
                                    cam_image = grad_cam(input_tensor)
                                    visualization = show_cam_on_image(img / 255.0, cam_image[0])
                                    cv2.imwrite(str(model_path / 'gradcam_visualization.png'), visualization)
                                except Exception as e:
                                    print(f"GradCAM visualization failed: {str(e)}")
                        
                        # Update progress
                        total_processed = i + 1
                        progress = int((total_processed / len(self.models)) * 100)
                        self.progress.emit(progress)
                        
                    except Exception as e:
                        print(f"Error generating visualization for {model_name}: {str(e)}")
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
                
                # Copy historical data if available
                if model_info.metrics_folder:
                    hist_path = model_path / 'historical'
                    hist_path.mkdir(exist_ok=True)
                    for file in Path(model_info.metrics_folder).glob('*'):
                        shutil.copy2(file, hist_path)
                
                # Clean up model
                model.cpu()
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                self.error.emit(f"Error processing model {model_name}: {str(e)}")
                continue
            
            finally:
                # Ensure model is cleaned up even if an error occurs
                if 'model' in locals():
                    model.cpu()
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Generate comparative visualizations after all models are processed
        self.status.emit("Generating comparative visualizations...")
        
        try:
            from visualization_utils import (
                plot_roc_curves, plot_precision_recall_curves,
                plot_det_curves, plot_eer_curves, plot_confidence_distribution,
                plot_model_evolution, plot_inference_times, create_summary_report,
                plot_confidence_separation
            )
            
            # Create dictionary of results
            models_results = {name: info.results for name, info in self.models.items()}
            
            # Generate all comparative plots
            plot_roc_curves(models_results, self.total_benchmark_path)
            plot_precision_recall_curves(models_results, self.total_benchmark_path)
            plot_det_curves(models_results, self.total_benchmark_path)
            plot_eer_curves(models_results, self.total_benchmark_path)
            plot_confidence_distribution(models_results, self.total_benchmark_path)
            plot_model_evolution(models_results, self.total_benchmark_path)
            plot_inference_times(models_results, self.total_benchmark_path)
            
            # Create the separation visualization
            plot_confidence_separation(models_results, self.total_benchmark_path)
            
            # Create summary report
            create_summary_report(models_results, self.total_benchmark_path)
            
            # Export benchmark data for AI analysis
            from visualization_utils import export_benchmark_data
            json_path, csv_path = export_benchmark_data(models_results, self.total_benchmark_path)
            self.status.emit(f"Benchmark data exported to {json_path}")
            
            # Generate AI analysis if requested
            if hasattr(self, 'generate_ai_report') and self.generate_ai_report:
                self.status.emit("Generating AI analysis report...")
                try:
                    from gemini_analysis import GeminiAnalyzer
                    analyzer = GeminiAnalyzer()
                    report_path = analyzer.create_comprehensive_report(json_path)
                    if report_path:
                        self.status.emit(f"AI analysis report generated at {report_path}")
                    else:
                        self.status.emit("Failed to generate AI analysis report")
                except Exception as e:
                    self.status.emit(f"Error generating AI report: {str(e)}")
            
        except Exception as e:
            self.error.emit(f"Error generating comparative visualizations: {str(e)}")
        
        finally:
            # Clean up temp directory
            shutil.rmtree(self.temp_path)
            
        self.status.emit("Benchmark completed successfully!")

class ModelBenchmark(QMainWindow):
    """Main window for the model benchmarking system"""
    def __init__(self):
        super().__init__()
        self.models = []  # List of ModelInfo objects
        self.model_cells = []  # List of UI elements for each model
        self.test_dataset_path = ""
        self.output_path = ""
        self.worker = None
        
        # Configure GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Anti-Spoofing Model Benchmark Suite")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create tab widget for different sections
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Add tabs
        self.setup_model_tab()
        self.setup_dataset_tab()
        self.setup_results_tab()
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Show the window
        self.show()
    
    def setup_model_tab(self):
        """Setup the model selection and configuration tab"""
        model_tab = QWidget()
        layout = QVBoxLayout(model_tab)
        
        # Model count selection
        count_group = QGroupBox("Number of Models to Benchmark")
        count_layout = QHBoxLayout()
        self.model_count_spin = QSpinBox()
        self.model_count_spin.setRange(1, 20)
        self.model_count_spin.setValue(1)
        count_layout.addWidget(QLabel("Select number of models:"))
        count_layout.addWidget(self.model_count_spin)
        count_layout.addStretch()
        count_group.setLayout(count_layout)
        layout.addWidget(count_group)
        
        # Create scrollable area for model cells
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Container for model cells
        self.model_container = QWidget()
        self.model_layout = QVBoxLayout(self.model_container)
        self.model_layout.addStretch()
        
        scroll.setWidget(self.model_container)
        layout.addWidget(scroll)
        
        # Add model button
        update_btn = QPushButton("Update Model Count")
        update_btn.clicked.connect(self.update_model_count)
        layout.addWidget(update_btn)
        
        self.tab_widget.addTab(model_tab, "Model Selection")
    
    def setup_dataset_tab(self):
        """Setup the dataset and output configuration tab"""
        dataset_tab = QWidget()
        layout = QVBoxLayout(dataset_tab)
        
        # Test dataset selection
        dataset_group = QGroupBox("Test Dataset Configuration")
        dataset_layout = QFormLayout()
        
        self.dataset_path = QLineEdit()
        self.dataset_path.setReadOnly(True)
        browse_dataset_btn = QPushButton("Browse...")
        browse_dataset_btn.clicked.connect(self.select_test_dataset)
        
        dataset_row = QHBoxLayout()
        dataset_row.addWidget(self.dataset_path)
        dataset_row.addWidget(browse_dataset_btn)
        
        dataset_layout.addRow("Test Dataset Path:", dataset_row)
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # Output folder selection
        output_group = QGroupBox("Output Configuration")
        output_layout = QFormLayout()
        
        self.output_path = QLineEdit()
        self.output_path.setReadOnly(True)
        browse_output_btn = QPushButton("Browse...")
        browse_output_btn.clicked.connect(self.select_output_folder)
        
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_path)
        output_row.addWidget(browse_output_btn)
        
        output_layout.addRow("Output Folder:", output_row)
        
        # AI Analysis option
        self.ai_analysis_checkbox = QCheckBox("Generate AI Analysis Report")
        self.ai_analysis_checkbox.setChecked(True)
        self.ai_analysis_checkbox.setToolTip("Use Google's Gemini AI to analyze benchmark results and generate detailed reports")
        output_layout.addRow("AI Analysis:", self.ai_analysis_checkbox)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Start benchmark button
        self.start_btn = QPushButton("Start Benchmark")
        self.start_btn.clicked.connect(self.start_benchmark)
        self.start_btn.setEnabled(False)
        layout.addWidget(self.start_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        self.tab_widget.addTab(dataset_tab, "Dataset & Output")
    
    def setup_results_tab(self):
        """Setup the results visualization tab"""
        results_tab = QWidget()
        layout = QVBoxLayout(results_tab)
        
        # Results will be shown here after benchmark
        self.results_label = QLabel("Run benchmark to see results")
        self.results_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.results_label)
        
        self.tab_widget.addTab(results_tab, "Results")
    
    def create_model_cell(self):
        """Create a UI group for a single model"""
        cell = QGroupBox("Model Configuration")
        layout = QFormLayout()
        
        # Version/name input
        version_input = QLineEdit()
        layout.addRow("Model Version:", version_input)
        
        # Model weights file selection
        weights_path = QLineEdit()
        weights_path.setReadOnly(True)
        browse_weights = QPushButton("Browse...")
        weights_layout = QHBoxLayout()
        weights_layout.addWidget(weights_path)
        weights_layout.addWidget(browse_weights)
        layout.addRow("Model Weights:", weights_layout)
        
        # Metrics folder selection
        metrics_path = QLineEdit()
        metrics_path.setReadOnly(True)
        browse_metrics = QPushButton("Browse...")
        metrics_layout = QHBoxLayout()
        metrics_layout.addWidget(metrics_path)
        metrics_layout.addWidget(browse_metrics)
        layout.addRow("Metrics Folder:", metrics_layout)
        
        # Add flip labels checkbox
        flip_labels = QCheckBox("Flip Labels (0=Real, 1=Fake)")
        flip_labels.setStyleSheet("color: white;")
        layout.addRow("Label Configuration:", flip_labels)
        
        cell.setLayout(layout)
        return cell, version_input, weights_path, metrics_path, browse_weights, browse_metrics, flip_labels
    
    def update_model_count(self):
        """Update the number of model configuration cells"""
        count = self.model_count_spin.value()
        
        # Clear existing cells
        for i in reversed(range(self.model_layout.count())):
            widget = self.model_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        self.model_cells = []
        
        # Add new cells
        for i in range(count):
            cell, version, weights, metrics, browse_w, browse_m, flip_labels = self.create_model_cell()
            cell.setTitle(f"Model {i+1} Configuration")
            
            # Connect browse buttons
            browse_w.clicked.connect(lambda checked, x=weights: self.browse_weights(x))
            browse_m.clicked.connect(lambda checked, x=metrics: self.browse_metrics(x))
            
            self.model_layout.insertWidget(i, cell)
            self.model_cells.append((cell, version, weights, metrics, flip_labels))
        
        # Add stretch at the end
        self.model_layout.addStretch()
    
    def browse_weights(self, line_edit):
        """Open file dialog to select model weights"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Weights",
            "",
            "Model Files (*.pt *.pth);;All Files (*)"
        )
        if file_path:
            line_edit.setText(file_path)
            self.check_start_button()
    
    def browse_metrics(self, line_edit):
        """Open folder dialog to select metrics folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Metrics Folder"
        )
        if folder_path:
            line_edit.setText(folder_path)
            self.check_start_button()
    
    def select_test_dataset(self):
        """Open folder dialog to select test dataset"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Test Dataset Folder"
        )
        if folder_path:
            self.dataset_path.setText(folder_path)
            self.check_start_button()
    
    def select_output_folder(self):
        """Open folder dialog to select output location"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder"
        )
        if folder_path:
            self.output_path.setText(folder_path)
            self.check_start_button()
    
    def check_start_button(self):
        """Check if all required fields are filled to enable start button"""
        # Check if test dataset and output paths are set
        if not self.dataset_path.text() or not self.output_path.text():
            self.start_btn.setEnabled(False)
            return
        
        # Check if all model cells are properly filled
        for _, version, weights, metrics, flip_labels in self.model_cells:
            if not version.text() or not weights.text():
                self.start_btn.setEnabled(False)
                return
        
        # Enable start button if all checks pass
        self.start_btn.setEnabled(True)
    
    def start_benchmark(self):
        """Start the benchmarking process"""
        # Disable UI elements
        self.start_btn.setEnabled(False)
        self.tab_widget.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Collect model information
        models = {}
        for _, version, weights, metrics, flip_labels in self.model_cells:
            model_info = ModelInfo()
            model_info.version = version.text()
            model_info.weight_path = weights.text()
            model_info.metrics_folder = metrics.text() if metrics.text() else ""
            model_info.flip_labels = flip_labels.isChecked()  # Store flip labels setting
            models[version.text()] = model_info
        
        # Create and start worker thread
        self.worker = BenchmarkWorker(
            models,
            self.dataset_path.text(),
            self.output_path.text()
        )
        
        # Pass AI analysis preference to worker
        self.worker.generate_ai_report = self.ai_analysis_checkbox.isChecked()
        
        # Connect signals
        self.worker.progress.connect(self.update_progress)
        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(self.benchmark_finished)
        self.worker.error.connect(self.benchmark_error)
        
        # Start processing
        self.worker.start()
    
    def benchmark_finished(self):
        """Handle benchmark completion"""
        # Re-enable UI elements
        self.start_btn.setEnabled(True)
        self.tab_widget.setEnabled(True)
        
        # Switch to results tab
        self.tab_widget.setCurrentIndex(2)  # Results tab
        
        # Update results view
        self.update_results_view()
        
        # Show completion message
        QMessageBox.information(
            self,
            "Benchmark Complete",
            "Benchmark process has completed successfully!\n\n"
            f"Results have been saved to:\n{self.output_path.text()}"
        )
    
    def update_results_view(self):
        """Update the results tab with links to generated visualizations"""
        # Clear previous content
        results_tab = self.tab_widget.widget(2)
        layout = results_tab.layout()
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().deleteLater()
        
        # Create scrollable area for results
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        # Add title
        title = QLabel("Benchmark Results")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        content_layout.addWidget(title)
        
        # Add path to results
        path_label = QLabel(f"Results saved to: {self.output_path.text()}")
        path_label.setWordWrap(True)
        content_layout.addWidget(path_label)
        
        # Check if AI report exists and add a link
        ai_report_path = Path(self.output_path.text()) / "total_benchmark" / "comprehensive_report.html"
        if ai_report_path.exists():
            ai_report_group = QGroupBox("AI Analysis Report")
            ai_report_layout = QVBoxLayout()
            
            ai_label = QLabel("A comprehensive AI-generated analysis report is available:")
            ai_report_layout.addWidget(ai_label)
            
            open_report_btn = QPushButton("Open AI Analysis Report")
            open_report_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(ai_report_path))))
            ai_report_layout.addWidget(open_report_btn)
            
            ai_report_group.setLayout(ai_report_layout)
            content_layout.addWidget(ai_report_group)
        
        # Add comparative results section
        comp_group = QGroupBox("Comparative Analysis")
        comp_layout = QVBoxLayout()
        
        plots = [
            ("ROC Curves", "roc_curves.png"),
            ("Precision-Recall Curves", "precision_recall_curves.png"),
            ("DET Curves", "det_curves.png"),
            ("EER Curves", "eer_curves.png"),
            ("Confidence Distribution", "confidence_distribution.png"),
            ("Real vs Fake Separation Plot", "confidence_separation.png"),
            ("Model Evolution", "model_evolution.png"),
            ("Inference Times", "inference_times.png")
        ]
        
        for title, filename in plots:
            path = Path(self.output_path.text()) / "total_benchmark" / filename
            if path.exists():
                label = QLabel(title)
                pixmap = QPixmap(str(path))
                img_label = QLabel()
                img_label.setPixmap(pixmap.scaled(
                    800, 600,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))
                comp_layout.addWidget(label)
                comp_layout.addWidget(img_label)
        
        comp_group.setLayout(comp_layout)
        content_layout.addWidget(comp_group)
        
        # Add individual model results
        for model_name in os.listdir(self.output_path.text()):
            model_path = Path(self.output_path.text()) / model_name
            if model_path.is_dir() and model_name != "total_benchmark" and model_name != "temp":
                model_group = QGroupBox(f"Model: {model_name}")
                model_layout = QVBoxLayout()
                
                # Add model-specific visualizations
                for plot in ["confusion_matrix", "tsne_features", "sample_prediction", "prediction_cards"]:
                    for ext in [".png", "_real.png", "_fake.png"]:
                        path = model_path / f"{plot}{ext}"
                        if path.exists():
                            label = QLabel(plot.replace("_", " ").title())
                            pixmap = QPixmap(str(path))
                            img_label = QLabel()
                            img_label.setPixmap(pixmap.scaled(
                                400, 300,
                                Qt.KeepAspectRatio,
                                Qt.SmoothTransformation
                            ))
                            model_layout.addWidget(label)
                            model_layout.addWidget(img_label)
                
                model_group.setLayout(model_layout)
                content_layout.addWidget(model_group)
        
        scroll.setWidget(content)
        layout.addWidget(scroll)
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop worker if running
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        
        # Clean up resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        event.accept()

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def update_status(self, message):
        """Update status bar message"""
        self.statusBar().showMessage(message)
    
    def benchmark_error(self, error_message):
        """Handle benchmark error"""
        QMessageBox.critical(self, "Error", f"Benchmark failed: {error_message}")
        self.start_btn.setEnabled(True)
        self.tab_widget.setEnabled(True)
        self.progress_bar.setValue(0)
    
    def browse_file(self, line_edit, title="Select File", file_filter="All Files (*.*)"):
        """Open file browser dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            "",
            file_filter
        )
        if file_path:
            line_edit.setText(file_path)
            self.check_start_button()
    
    def browse_folder(self, line_edit, title="Select Folder"):
        """Open folder browser dialog"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            title,
            ""
        )
        if folder_path:
            line_edit.setText(folder_path)
            self.check_start_button()

def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Set dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    # Set stylesheet
    app.setStyleSheet("""
        QMainWindow {
            background-color: #353535;
        }
        QLabel {
            color: white;
        }
        QPushButton {
            background-color: #2a82da;
            border: none;
            color: white;
            padding: 5px 15px;
            border-radius: 3px;
            font-weight: bold;
        }
        QPushButton:disabled {
            background-color: #666666;
        }
        QPushButton:hover {
            background-color: #3d95ed;
        }
        QProgressBar {
            border: 2px solid grey;
            border-radius: 5px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #2a82da;
            width: 10px;
            margin: 0.5px;
        }
        QGroupBox {
            border: 1px solid #666666;
            border-radius: 5px;
            margin-top: 1ex;
            font-weight: bold;
            color: white;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 3px;
        }
        QLineEdit {
            padding: 5px;
            border: 1px solid #666666;
            border-radius: 3px;
            background-color: #252525;
            color: white;
        }
        QScrollArea {
            border: none;
        }
    """)
    
    window = ModelBenchmark()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 