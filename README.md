<div align="center">

<img src="screenshots/EasyShield_Logo.png" width="100" alt="…"/>

# EasyShield : A State-of-the-Art Face Anti-Spoofing Deep Learning Model

_optimized for Edge Devices_

[![Source Code](https://img.shields.io/badge/source%20code-GitHub-181717.svg?logo=github)](https://github.com/mahostar/EasyShield-Anti-Spoofing-AI-Model)
[![image](https://img.shields.io/pypi/l/magentic_ui.svg)]([https://pypi.python.org/pypi/magentic_ui](https://github.com/mahostar/EasyShield-Anti-Spoofing-AI-Model/blob/main/LICENSE))

[![Paper](https://img.shields.io/badge/paper-coming--soon-lightgrey?logo=latex&logoColor=white)](https://github.com/mahostar/EasyShield-Anti-Spoofing-AI-Model)


</div>


<div align="center">
<img src="screenshots/infrence.png" width="400" alt="…"/>
</div>


## Table of Contents
- [1. Introduction](#introduction)
- [2. Features](#features)
- [3. System Pipeline Overview](#system-pipeline-overview)
- [4. Tools Overview](#tools-overview)
  - [4.1. Face Extractor Tool](#face-extractor-tool)
  - [4.2. Image Augmentor Tool](#image-augmentor-tool)
  - [4.3. Image Filtering Tool](#image-filtering-tool)
  - [4.4. Dataset Preparation Script (`prepare_data.py`)](#dataset-preparation-script-prepare_datapy)
  - [4.5. Easy Spoof Trainer Tool (`Easy_Spoof_Trainer.py`)](#easy-spoof-trainer-tool-easy_spoof_trainerpy)
  - [4.6. Model Testing Tool (`test_model.py`)](#model-testing-tool-test_modelpy)
- [5. Setup and Installation](#setup-and-installation)
  - [5.1. Windows Development Environment Setup](#windows-development-environment-setup)
  - [5.2. Linux/Edge Device Setup](#linuxedge-device-setup)
- [6. Dataset Workflow](#dataset-workflow)
- [7. How to Train a New Model](#how-to-train-a-new-model)
- [8. How to Test a Trained Model](#how-to-test-a-trained-model)
- [9. Model Performance and Analysis](#model-performance-and-analysis)
- [10. Project Structure](#project-structure)
- [11. Ethical & Practical Considerations](#ethical--practical-considerations)
- [12. Conclusion](#conclusion)
- [13. References](#references)


## 1. Introduction

Facial recognition technology has become increasingly prevalent in various applications, from unlocking smartphones to verifying identities for financial transactions. However, the vulnerability of these systems to spoofing attacks—where an impostor presents a fake facial biometric (e.g., a printed photo, a video replay) to deceive the system—poses a significant security risk. Consequently, robust face anti-spoofing mechanisms are essential to ensure the reliability and integrity of facial recognition systems.

The performance of EasyShield has been rigorously evaluated and compared against several leading anti-spoofing models. As illustrated in **Figure 1** (Accuracy Comparison) and detailed in **Table 1** (Performance Comparison), EasyShield v2.5 demonstrates superior accuracy, precision, recall, F1 score, AUC, and EER compared to models like MN3_antispoof, DeePixBiS, and WENDGOUNDI. Specifically, EasyShield achieved an accuracy of 92.30% and an EER of 6.25% on our challenging 8000-image test dataset.

<div align="center">
<img src="screenshots/accuracy_comparison.png" width="600" alt="…"/>
</div>
Figure 1: Accuracy comparison between EasyShield v2.5 and competing models.


| Metric                   | EasyShield v2.5 | MN3\_antispoof | DeePixBiS | WENDGOUNDI |
|--------------------------|-----------------|----------------|-----------|------------|
| Accuracy                 | **92.30%**      | 66.60%         | 54.20%    | 58.33%     |
| Precision                | **88.32%**      | 81.74%         | 52.50%    | 55.83%     |
| Recall                   | **97.50%**      | 42.75%         | 88.30%    | 79.75%     |
| F1 Score                 | **92.68%**      | 56.14%         | 65.85%    | 65.68%     |
| AUC                      | **98.61%**      | 81.11%         | 40.29%    | 61.79%     |
| EER                      | **6.25%**       | 27.55%         | 58.37%    | 38.65%     |
| APCER                    | **6.25%**       | 9.55%          | 79.90%    | 57.05%     |
| BPCER                    | **2.50%**       | 57.25%         | 11.70%    | 20.25%     |
| ACER                     | **4.38%**       | 33.40%         | 45.80%    | 38.65%     |
| Avg. Inference Time (ms) | 75.47           | 9.36           | 59.20     | **6.93**   |


*Table 1: Performance Comparison of Anti-Spoofing Models on 8000-Image Test Dataset. EasyShield v2.5 shows superior overall performance.*

## 2. Features
*   **Lightweight and Fast:** Optimized for deployment on edge devices with limited computational resources, achieving an average inference time of 75.47 ms.
*   **High Accuracy:** Demonstrates 92.30% accuracy in detecting presentation attacks (e.g., print and replay attacks).
*   **Robust Binary Classification:** Effectively classifies faces as "Real" or "Fake".
*   **Comprehensive Toolkit:** Includes a suite of tools for dataset preparation, image augmentation, model training, and real-time testing.
*   **State-of-the-Art Model:** Utilizes the YOLOv8 nano architecture for efficient and accurate detection.
*   **Strong Performance Metrics:** Achieves an EER of 6.25% and an AUC of 98.61%.

## 3. System Pipeline Overview
The EasyShield system employs an end-to-end workflow, illustrated in the figure below, which encompasses data acquisition and preprocessing, model training and evaluation, and finally, deployment for real-time inference. The core of the system is a YOLOv8 nano model, trained to classify input face images (resized to 640x640 pixels) as either "Real" or "Fake." Specialized Python-based tools, equipped with graphical user interfaces (GUIs) where appropriate, are provided to facilitate each step of this pipeline. These tools range from extracting face crops from raw video/image data and augmenting the dataset for improved robustness, to training the model and testing its performance.


<div align="center">
<img src="screenshots/piplineEasySpoof.png" width="600" alt="…"/>
</div>
*Figure 4: Overview of the EasyShield system pipeline from data collection to inference.*

## 4. Tools Overview

EasyShield includes a comprehensive suite of Python-based tools designed to streamline the entire lifecycle of developing and deploying the face anti-spoofing model. These tools facilitate dataset creation, augmentation, training, and testing.

### 4.1. Face Extractor Tool
The **Face Extractor tool** (`videos_and_images_face_extractor.py`), located in the `dataset preparing tools/` directory, is designed to process raw input videos (AVI, MP4, MKV) and images (JPEG, PNG) to extract contextual face crops. It utilizes advanced face detection algorithms (e.g., MTCNN or a similar robust detector) to identify faces and then crops them with a consistent margin around the detected bounding box. These crops are automatically resized to 640x640 pixels, the standard input size required by the EasyShield YOLOv8 nano model, ensuring uniformity across the dataset. The tool features a graphical user interface (GUI) for ease of use, allowing users to select input files/folders and specify output directories.


<div align="center">
<img src="screenshots/FaceExtractorTool.png" width="600" alt="…"/>
</div>
*Figure 5: Graphical User Interface of the Face Extractor tool.*

### 4.2. Image Augmentor Tool
The **Image Augmentor tool** (`image_augmentor.py`), also found in `dataset preparing tools/`, applies a variety of data augmentation techniques to the extracted face crops. Data augmentation is a crucial step for increasing the diversity of the training dataset, which significantly helps in improving the model's generalization capabilities and its robustness against unseen variations in real-world scenarios. Techniques employed may include random rotations, brightness adjustments, contrast changes, Gaussian blur, and horizontal flips. Users can typically configure the types and intensity of augmentations to be applied.

### 4.3. Image Filtering Tool
The **Image Filtering tool** (`image_filtring.py`), part of the `dataset preparing tools/` suite, allows for manual review and filtering of the augmented dataset. This tool often integrates a face detector like MTCNN (Multi-task Cascaded Convolutional Networks) to provide suggestions or automatically flag potentially problematic images. These could include images with misaligned faces, multiple faces where a single face is expected, non-face images that were incorrectly cropped, or images with poor quality (e.g., excessive blur or low resolution not introduced by augmentation). This quality control step is paramount for ensuring that the training data is of high quality, which directly impacts the accuracy and reliability of the final model.

### 4.4. Dataset Preparation Script (`prepare_data.py`)
The `prepare_data.py` script, located in `dataset preparing tools/`, is responsible for organizing the filtered and augmented images into the specific directory structure and format required by the YOLO (You Only Look Once) training framework. It typically involves creating `train/` and `valid/` (and optionally `test/`) subdirectories, each containing `images/` and `labels/` folders. For this binary classification task (Real vs. Fake), it would sort images into 'Real' and 'Fake' source folders and then prepare them accordingly. Crucially, this script also generates a `dataset.yaml` file. This YAML configuration file defines the paths to the training and validation image sets, specifies the number of classes (2, in this case), and lists the class names (e.g., "Real", "Fake").

### 4.5. Easy Spoof Trainer Tool (`Easy_Spoof_Trainer.py`)
The **Easy Spoof Trainer tool** (`Easy_Spoof_Trainer.py`), located in the `Trained TOOL for YOLO/` directory, is the core component for training the EasyShield model. It takes the dataset prepared in the YOLO format (referenced by the `dataset.yaml` file) and trains the specified YOLOv8 nano model for the binary face anti-spoofing classification task. The tool manages the training loop, allows configuration of hyperparameters (e.g., learning rate, number of epochs, batch size), and leverages the Ultralytics YOLO framework. Upon completion of training, it provides extensive evaluation metrics, including accuracy, precision, recall, F1-score, Area Under the Curve (AUC), and Equal Error Rate (EER). It also typically outputs confusion matrices, ROC curves, and saves the trained model weights (e.g., `best.pt`).

### 4.6. Model Testing Tool (`test_model.py`)
The `test_model.py` script, available in `Testing Code (windows)/` and `Testing Code (Linux)/` directories, provides a GUI for real-time testing and qualitative assessment of the trained EasyShield model. Users can input live video feeds (e.g., from a webcam by specifying source `0`) or individual images/videos by providing their file paths. The tool loads the trained YOLOv8 model weights (a `.pt` file) and performs inference on the input. It then displays the model's prediction (Real or Fake) overlaid on the video feed or image, usually along with a confidence score for the prediction. This allows for immediate visual feedback on the model's performance in practical scenarios.

## 5. Setup and Installation

This section guides you through setting up the EasyShield development environment on both Windows and Linux-based systems, including edge devices.

### 5.1. Windows Development Environment Setup

Setting up the EasyShield development environment on Windows requires careful attention to specific versions of Python, NVIDIA CUDA, NVIDIA cuDNN, and several Python packages.

**Prerequisites:**
*   NVIDIA GPU with CUDA support.
*   Administrator privileges for some installation steps.

**Installation Steps:**
1.  **Install NVIDIA CUDA Toolkit:**
    *   Download and install **CUDA Toolkit 11.8** from the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive).

2.  **Install NVIDIA cuDNN:**
    *   Download **cuDNN v8.9.6 for CUDA 11.x** from the [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive). You will need to join the NVIDIA Developer Program (free) to download.

3.  **Create a Python Virtual Environment (Highly Recommended):**
    *   Open a Command Prompt or PowerShell.
    *   Navigate to your project directory (e.g., `cd path\to\EasyShield-Anti-Spoofing-AI-Model`).
    *   Create the virtual environment:
        ```bash
        python -m venv easyshield_env
        ```
    *   Activate the virtual environment:
        ```bash
        easyshield_env\Scripts\activate
        ```
        Your command prompt should now be prefixed with `(easyshield_env)`.

7.  **Install Python Dependencies:**
    *   Ensure pip is up-to-date:
        ```bash
        python -m pip install --upgrade pip
        ```
    *   Install PyTorch (ensure this version matches your CUDA 11.8 setup):
        ```bash
        pip install ultralytics opencv-python numpy PyQt5 torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
        ```
    *   Install FaceNet-PyTorch (without its own PyTorch dependency, as it's already installed):
        ```bash
        pip install facenet-pytorch --no-deps
        ```
    *   Install all other dependencies from the `requirements.txt` file:
        ```bash
        pip install -r requirements.txt
        ```

### 5.2. Linux/Edge Device Setup

Setting up EasyShield on Linux or an edge device (e.g., NVIDIA Jetson, Raspberry Pi - though RPi performance will be limited) involves similar principles but different commands and package considerations.

**Prerequisites:**
*   Python 3.8+ (Python 3.10 recommended).
*   For GPU acceleration on devices like NVIDIA Jetson, ensure NVIDIA JetPack SDK is installed, which includes CUDA, cuDNN, and TensorRT.

**Installation Steps:**

1.  **Install Python (if needed):**
    *   Most Linux distributions come with Python 3. Check with `python3 --version`.
    *   If you need a specific version or it's missing, install it using your distribution's package manager (e.g., `sudo apt update && sudo apt install python3.10 python3.10-venv`).

2.  **Create a Python Virtual Environment (Highly Recommended):**
    *   Open a terminal.
    *   Navigate to your project directory.
    *   Create the virtual environment:
        ```bash
        python3 -m venv easyshield_env
        ```
    *   Activate the virtual environment:
        ```bash
        source easyshield_env/bin/activate
        ```
        Your terminal prompt should now be prefixed with `(easyshield_env)`.

3.  **Install Python Dependencies:**
    *   Ensure pip is up-to-date:
        ```bash
        python -m pip install --upgrade pip
        ```
    *   **Install remaining dependencies** from `requirements_linux.txt`:
        This file should contain packages specific to EasyShield that are not covered by the above, or versions pinned for Linux that differ from the main `requirements.txt`.
        ```bash
        pip install -r requirements_linux.txt
        ```

## 6. Dataset Workflow

The creation of a high-quality dataset is fundamental to the success and robustness of the EasyShield anti-spoofing model. The workflow involves a sequence of steps, each facilitated by specialized tools provided within this project:

1.  **Data Collection (Manual):**
    *   Gather raw video footage and still images. This data should include a diverse set of:
        *   **Real faces:** Various individuals, different ethnicities, ages, genders, lighting conditions, and camera qualities.
        *   **Spoofing attacks:**
            *   **Print attacks:** High-quality printed photos of faces, photos displayed on digital screens.
            *   **Replay attacks:** Videos of faces played back on tablets, smartphones, or monitors.

2.  **Face Extraction (`videos_and_images_face_extractor.py`):**
    *   **Purpose:** To detect and isolate face regions from the collected media.
    *   **Process:** Use the Face Extractor tool (see Section 4.1) to process your raw videos and images. The tool detects faces, crops them with an appropriate margin, and resizes them to 640x640 pixels.
    *   **Output:** A structured dataset of cropped face images, typically separated into 'real' and 'fake' preliminary folders.

3.  **Image Augmentation (`image_augmentor.py`):**
    *   **Purpose:** To artificially increase the size and diversity of the training dataset, improving the model's ability to generalize.
    *   **Process:** Use the Image Augmentor tool (see Section 4.2) on the extracted face crops. Apply various augmentations like rotations, brightness/contrast adjustments, blur, etc.
    *   **Output:** An expanded dataset of augmented face images.

4.  **Image Filtering (`image_filtring.py`):**
    *   **Purpose:** To ensure the quality of the dataset by removing or correcting poorly cropped, mislabeled, or irrelevant images.
    *   **Process:** Use the Image Filtering tool (see Section 4.3) to manually review the augmented dataset. The integrated MTCNN can help flag images that might be problematic (e.g., no face detected, multiple faces).
    *   **Output:** A curated, high-quality dataset of face images.

5.  **Dataset Preparation (`prepare_data.py`):**
    *   **Purpose:** To organize the curated dataset into the specific format required by the YOLOv12 training framework.
    *   **Process:** Run the `prepare_data.py` script (see Section 4.4). This script will typically:
        *   Split the data into training and validation sets (e.g., 80% train, 20% valid).
        *   Create the necessary directory structure (e.g., `Dataset_Demo_Exemple/train/images`, `Dataset_Demo_Exemple/train/labels`, etc.).
        *   Generate a `dataset.yaml` file that defines the paths to train/validation sets and class names ("Real", "Fake").
    *   **Output:** A YOLO-formatted dataset ready for training.

![Dataset Example](screenshots/datasetExemple.png)
*Figure 6: Example of the dataset structure after preparation, showing 'real' and 'fake' images categorized for training and validation (actual structure may vary based on `prepare_data.py` script).*

## 7. How to Train a New Model

Training a new EasyShield anti-spoofing model involves using the `Easy_Spoof_Trainer.py` tool, which leverages the Ultralytics YOLOv8 framework.

**Prerequisites:**
*   A fully prepared dataset in YOLO format, created using the [Dataset Workflow](#6-dataset-workflow). This includes a `dataset.yaml` file and the corresponding image/label directories.
*   The EasyShield environment correctly set up with all dependencies installed as per Section [5. Setup and Installation](#5-setup-and-installation).

**Steps to Train:**

1.  **Navigate to the Trainer Tool Directory:**
    Open your terminal or command prompt, activate your virtual environment (`easyshield_env`), and change to the directory containing the trainer script:
    ```bash
    cd path/to/EasyShield-Anti-Spoofing-AI-Model/"Trained TOOL for YOLO/"
    ```

2.  **Run the Training Script:**
    Execute `Easy_Spoof_Trainer.py` with appropriate command-line arguments. Here's a basic example:
    ```bash
    python Easy_Spoof_Trainer.py
    ```
3.  **Monitor Training and Results:**
    *   The training progress will be displayed in the terminal, including metrics like loss, accuracy, precision, and recall for each epoch.
    *   Upon completion, the trained model (usually `best.pt` and `last.pt`), along with various performance charts (e.g., confusion matrix, ROC curve) and logs, will be saved in the `runs/train/YourExperimentName/` directory. The `best.pt` model is typically the one with the best validation performance.

## 8. How to Test a Trained Model

After training your EasyShield model, you can test its performance on live webcam feed, video files, or individual images using the `test_model.py` script.

**Prerequisites:**
*   A trained EasyShield model file (e.g., `best.pt` from your training output).
*   The EasyShield environment correctly set up as per Section [5. Setup and Installation](#5-setup-and-installation).

**Steps to Test:**

1.  **Navigate to the Testing Script Directory:**
    Open your terminal or command prompt, activate your virtual environment (`easyshield_env`), and navigate to the appropriate testing code directory:
    *   For Windows:
        ```bash
        cd path/to/EasyShield-Anti-Spoofing-AI-Model/"Testing Code (windows)/"
        ```
    *   For Linux:
        ```bash
        cd path/to/EasyShield-Anti-Spoofing-AI-Model/"Testing Code (Linux)/"
        ```
    *(Ensure the `test_model.py` script is present in this directory.)*

2.  **Run the Testing Script:**
    Execute `test_model.py` with arguments specifying the trained model weights and the input source.
    ```bash
    python test_model.py
    ```

## 9. Model Performance and Analysis

EasyShield has undergone several iterations of development, with each version incorporating improvements in dataset curation, augmentation strategies, and model training techniques. **Figure 7** (formerly Figure X) illustrates the performance trajectory across these development versions, highlighting the progressive enhancements in metrics such as accuracy and EER.

![Model Performance Evolution](screenshots/ModelsPerformence.png)
*Figure 7: Model Performance Evolution of EasyShield across different development versions, showcasing improvements in key anti-spoofing metrics.*

## 10. Project Structure

The EasyShield project is organized as follows. This structure ensures that all code, tools, datasets, and documentation are easily accessible.

```plaintext
EasyShield-Anti-Spoofing-AI-Model/
├── Dataset_Demo_Exemple/             # Example dataset with 'fake' and 'real' subdirectories
│   ├── fake/
│   │   └── (image files: .jpg, .png, etc.)
│   └── real/
│       └── (image files: .jpg, .png, etc.)
├── EasyShield weights/                 # Contains pre-trained model weights
│   ├── EasyShield V2.5 - nano (well mixed dataset) 120 epochs 16 batch/ # Latest recommended version
│   │   └── best.pt                     # Example of a trained model file
│   └── old versions (less acurate - less performence)/                 # Older model versions for reference
│       └── (similar structure with .pt files)
├── Original YOLO v12 Models For training/ # Base YOLO models (Note: Project primarily uses YOLOv8, this dir might be for experimentation or legacy)
│   └── (YOLO model files like .pt or .yaml)
├── Testing Code (Linux)/               # Model testing scripts optimized or specific to Linux environments
│   └── test_model.py                   # GUI-based model testing script
├── Testing Code (windows)/             # Model testing scripts optimized or specific to Windows environments
│   └── test_model.py                   # GUI-based model testing script
├── Trained TOOL for YOLO/              # Core training script for the EasyShield model
│   └── Easy_Spoof_Trainer.py           # Python script to train the YOLOv8 model
├── dataset preparing tools/            # Suite of tools for dataset creation, augmentation, and management
│   ├── data_collection.py              # (Potentially) Script for initial data gathering or organization (if used)
│   ├── image_augmentor.py              # Tool for applying data augmentation techniques
│   ├── image_filtring.py               # Tool for manual dataset review and filtering (with MTCNN assistance)
│   ├── prepare_data.py                 # Script to format the dataset for YOLO training and generate dataset.yaml
│   └── videos_and_images_face_extractor.py # Tool to extract faces from videos and images
├── screenshots/                        # Contains all images and diagrams used in this README and other documentation
│   ├── accuracy_comparison.png
│   ├── curveCompareson.png
│   ├── datasetExemple.png
│   ├── eer_comparison.png
│   ├── FaceExtractorTool.png
│   ├── infrence.png
│   ├── ModelsPerformence.png
│   ├── overall_performance.png
│   ├── piplineEasySpoof.png
│   └── separationDigram.png
├── README.md                           # This comprehensive documentation file
├── requirements.txt                    # Python package dependencies for Windows and general use
└── requirements_linux.txt              # Python package dependencies tailored for Linux/edge device setups
```

## 11. Ethical & Practical Considerations

While EasyShield offers significant advancements in face anti-spoofing, it is crucial to consider the ethical implications and practical limitations inherent in such technology. Responsible development and deployment are paramount.

1.  **Bias and Fairness:**
    *   Deep learning models can inadvertently learn and perpetuate biases present in their training data. If the dataset is not sufficiently diverse (e.g., in terms of age, gender, ethnicity, skin tone, and environmental conditions like lighting), the model's performance may vary significantly across different user groups. This can lead to some groups being more susceptible to false positives or false negatives.
    *   **Mitigation:** Continuous efforts are needed to curate balanced, representative datasets and to test for and mitigate bias in model performance.

2.  **Privacy:**
    *   The collection, storage, and use of facial data are subject to stringent privacy regulations (e.g., GDPR, CCPA) and ethical guidelines. Facial images are considered sensitive personal information.
    *   **Mitigation:** Systems using EasyShield must ensure user consent is obtained where required, data is anonymized or pseudonymized if possible, stored securely, and processed only for the intended purposes. Transparency with users about data handling practices is essential.

## 12. Conclusion

EasyShield presents a significant step forward in the domain of face anti-spoofing technology, particularly for deployment on resource-constrained edge devices. By synergizing a lightweight YOLOv12 nano architecture with a meticulously designed dataset preparation pipeline and specialized training methodologies, EasyShield achieves state-of-the-art performance. It demonstrably outperforms several existing models in crucial metrics including accuracy (92.30%), Equal Error Rate (EER: 6.25%), and Area Under the Curve (AUC: 98.61%).
The comprehensive suite of tools provided with EasyShield facilitates the entire development lifecycle, from initial data collection and augmentation to model training, evaluation, and real-time testing. This holistic approach empowers developers and researchers to adapt and further enhance the system.

## 13. References

The development of EasyShield and the information presented in this document draw upon knowledge from various sources in the fields of machine learning, computer vision, and face anti-spoofing. Key resources and technologies include:

*   **Ultralytics YOLOv8:** The core object detection and classification framework used. [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
*   **PyTorch:** The deep learning framework used by YOLOv8 and for model development. [https://pytorch.org/](https://pytorch.org/)
*   **MTCNN (Multi-task Cascaded Convolutional Networks):** Often used for face detection. A popular implementation can be found at [https://github.com/ipazc/mtcnn](https://github.com/ipazc/mtcnn) (though the specific face detector in the tools may vary).
*   **Face Recognition and Anti-Spoofing Research:**
    *   Rosebrock, A. (2021). *PyImageSearch Gurus course*. PyImageSearch. (General computer vision and deep learning resource).
    *   **CelebA-Spoof Dataset:** A large-scale face anti-spoofing dataset. Zhang, Y., et al. (2020). CelebA-Spoof: Large-Scale CelebFace Anti-Spoofing Dataset with Rich Annotations. [https://github.com/Davidzhangyuanhan/CelebA-Spoof](https://github.com/Davidzhangyuanhan/CelebA-Spoof)
    *   Boulkenafet, Z., Komulainen, J., & Hadid, A. (2016). Face Spoofing Detection Using Colour Texture Analysis. *IEEE Transactions on Image Processing, 25*(7), 3321-3334. [DOI: 10.1109/TIP.2016.2555286](https://doi.org/10.1109/TIP.2016.2555286) (Example of foundational work in the field).
    *   Chingovska, I., Anjos, A., & Marcel, S. (2012). Anti-spoofing in action: A face spoofing database and a study on generalisation. *2012 IEEE International Joint Conference on Biometrics (IJCB)*, 1-7. [DOI: 10.1109/IJCB.2012.6199762](https://doi.org/10.1109/IJCB.2012.6199762) (Highlights importance of databases and generalization).
