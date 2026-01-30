# MLOps Project: CIFAR-10 Classification with Transfer Learning

This project implements an end-to-end MLOps pipeline for CIFAR-10 classification using transfer learning, with **Weights & Biases (W&B)** for experiment tracking, artifact management, and model registry.

## Project Structure

- `notebooks/`: Standalone notebooks for each pipeline stage.
  - `01_data_preparation.ipynb`: **Only stage that downloads CIFAR-10**; creates train/test/sim splits and logs a W&B dataset artifact.
  - `02_model_training_sweep.ipynb`: Runs W&B sweeps (standard/upsample/modified) and logs the best model artifact.
  - `03_model_evaluation.ipynb`: Loads the best model + dataset artifact and evaluates on the fixed test split.
  - `04_model_deployment_monitoring.ipynb`: Simulates production inference via FastAPI; logs wrong predictions as feedback.
  - `05_automated_retraining.ipynb`: Retrains using the feedback data and logs validation accuracy per epoch.
- `artifacts/`: Local storage for downloaded or generated artifacts.
- `src/`: Python modules mirroring the notebook logic for reuse in scripts.
- `requirements.txt`: Python dependencies.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Authenticate with W&B:
   ```bash
   wandb login
   ```

### Optional: `.env` for `src/`
The `src/` modules read credentials from `.env`.
Create a `.env` file with:
```
WANDB_API_KEY=YOUR_KEY
PROJECT_NAME=cifar10_mlops_project
ENTITY=esi-sba-dz
```

> Note: The **notebooks do not use `.env`**; they contain explicit credentials for isolated execution.

## Usage

Run notebooks in order:
1. `notebooks/01_data_preparation.ipynb`
2. `notebooks/02_model_training_sweep.ipynb`
3. `notebooks/03_model_evaluation.ipynb`
4. `notebooks/04_model_deployment_monitoring.ipynb`
5. `notebooks/05_automated_retraining.ipynb`
