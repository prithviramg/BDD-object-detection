# BDD-object-detection

This repository demonstrates an end-to-end object detection pipeline using a pre-trained Faster R-CNN model (ResNet-50 backbone) from PyTorch/TorchVision. It covers data processing, model loading, inference, evaluation (qualitative and basic quantitative), and visualization, with a focus on clear structure, documentation, and containerization of the data processing step.

## Project Goal

To detect common objects (based on COCO dataset classes) in input images using a pre-trained deep learning model.

## Setup & Installation

**Prerequisites:**

* Python 3.9+
* `pip` (Python package installer)
* Docker (for running the containerized data task)
* Git

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd your-object-detection-repo
    ```

2.  **Create a Python virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Installing PyTorch (`torch`, `torchvision`) might require specific commands depending on your OS and CUDA version if using GPU. Refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for details.*

4.  **Place Input Data:**
    Put the `.jpg`, `.jpeg`, or `.png` images you want to process into the `data/raw/` directory.

## How to Run

### 1. Data Processing Task (Containerized)

This step validates input images and copies them to the `data/processed/` directory, ready for the model.

**a. Build the Docker Image:**

Navigate to the `deployment/data_task/` directory and run:
```bash
cd deployment/data_task
docker build -t object-detection-data-processor .
cd ../.. # Go back to the project root