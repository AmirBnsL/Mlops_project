# CIFAR-10 MLOps Pipeline Report

This report documents the end-to-end MLOps pipeline implemented for the CIFAR-10 image classification project. The pipeline is designed to be modular, reproducible, and automated, using **Weights & Biases (W&B)** as the central platform for experiment tracking, artifact versioning, and model registry.

The project is divided into 5 distinct stages.

---

## 1. Data Preparation & Versioning
**Notebook:** `01_data_preparation.ipynb`

### Objective
To establish a single, immutable source of truth for the dataset to ensure reproducibility across all future experiments and training runs.

### Process
1.  **Download Source**: The pipeline downloads the raw CIFAR-10 dataset from Torchvision.
2.  **Splitting**: 
    -   **Train**: Standard training set.
    -   **Test**: A fixed subset (indices 0-8000) for model evaluation.
    -   **Simulation**: A separate subset (indices 8000+) reserved strictly for simulating "production" traffic.
3.  **Artifact Creation**: The raw data and the split indices are bundled together.
4.  **Versioning**: This bundle is uploaded to W&B as an Artifact (`cifar10_dataset:v1`).

**Why this matters**: We never download from the internet again. Every subsequent step uses this exact versioned artifact, guaranteeing that every model sees the exact same data.

---

## 2. Model Training & Hyperparameter Sweep
**Notebook:** `02_model_training_sweep.ipynb`

### Objective
To explore the hyperparameter space and identify the optimal model configuration.

### Process
1.  **Fetch Data**: Downloads the `cifar10_dataset` artifact.
2.  **Define architectures**:
    -   **Standard**: ResNet18 on 32x32 images.
    -   **Upsample**: ResNet18 on 224x224 images (using Resize transform).
    -   **Modified**: ResNet18 with a custom first layer (small kernel) for low-res inputs.
3.  **Bayesian Sweep**: Facilitated by W&B Sweeps, the agent explores combinations of:
    -   Learning Rate
    -   Batch Size
    -   Optimizer (SGD vs Adam)
    -   Architecture
4.  **Model Registry**: The best performing model (based on validation accuracy) is saved and uploaded as a W&B Artifact (`model-<run_id>:v0`).

---

## 3. Model Evaluation
**Notebook:** `03_model_evaluation.ipynb`

### Objective
To rigorously assess the best model's performance on the held-out test set.

### Process
1.  **Dependency Resolution**: Automatically identifies the "Best Run" from the sweep and downloads its specific model artifact.
2.  **Evaluation**: Runs the model on the **Test Set** (indices created in Step 1).
3.  **Visualization**: Generates and logs a **Confusion Matrix** to W&B, allowing us to see which classes (e.g., Cat vs. Dog) the model confuses most often.

---

## 4. Deployment & Monitoring Loop
**Notebook:** `04_model_deployment_monitoring.ipynb`

### Objective
To simulate a production inference environment and capture failure cases for improvement.

### Process
1.  **Model Serving**: Starts a local **FastAPI** server that exposes a `/predict` endpoint.
2.  **Traffic Simulation**: A script acts as a "client," sending images from the **Simulation Set** (indices 8000+) to the API.
3.  **Feedback Loop**:
    -   The system compares the API's prediction against the ground truth.
    -   **Correct Predictions**: Logged to a W&B Table for monitoring confidence.
    -   **Incorrect Predictions**: Captured as "Feedback" data.
4.  **Artifact Upload**: A file containing the IDs of these failed images is uploaded as a new artifact (`cifar10-feedback:v1`).

---

## 5. Automated Retraining (Continuous Training)
**Notebook:** `05_automated_retraining.ipynb`

### Objective
To implement a "Data-Centric AI" workflow where the model automatically improves based on captured failures.

### Process
1.  **Check for Feedback**: The system checks W&B for the existence of a `cifar10-feedback` artifact.
2.  **Data Merging**:
    -   Downloads the **Original Dataset**.
    -   Downloads the **Feedback Data** (Hard Examples).
    -   Creates a weighted dataset: `[Original Training Data] + [Hard Examples]`.
3.  **Fine-Tuning**:
    -   Loads the previous "Best Model".
    -   Re-initializes the optimizer with the exact same hyperparameters (Learning Rate, Optimizer Type).
    -   Trains for **5 Epochs** to adapt to the hard examples.
4.  **Validation**: Evaluates the new model after every epoch to ensure no regression in general accuracy.
5.  **Versioning V2**: The retrained model is saved and uploaded as a new version (`retrained-model:v1`), ready for the next deployment cycle.
