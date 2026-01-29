# MLOps Project: CIFAR-10 Classification with Transfer Learning

This project implements an MLOps pipeline for classifying CIFAR-10 images using transfer learning. It utilizes Weights & Biases (W&B) for experiment tracking, artifact management, and model registry.

## Project Structure

- `notebooks/`: Contains Jupyter notebooks for each stage of the pipeline.
    - `01_data_preparation.ipynb`: Downloads data, preprocesses it, and logs it as a W&B artifact.
    - `02_model_training_sweep.ipynb`: Runs hyperparameter sweeps using transfer learning (e.g., ResNet) and logs models.
    - `03_model_evaluation.ipynb`: Fetches the best model from the registry and evaluates it on the test set.
- `artifacts/`: Local storage for downloaded or generated artifacts (data, models).
- `src/`: Source code for reusable python modules (if needed).
- `requirements.txt`: List of dependencies.

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Login to Weights & Biases:
    ```bash
    wandb login
    ```

## Usage

Follow the notebooks in order:
1.  Run `notebooks/01_data_preparation.ipynb`
2.  Run `notebooks/02_model_training_sweep.ipynb`
3.  Run `notebooks/03_model_evaluation.ipynb`
