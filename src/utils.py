from dotenv import load_dotenv
import os

load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
PROJECT_NAME = os.getenv("PROJECT_NAME", "cifar10_mlops_project")
ENTITY = os.getenv("ENTITY", "esi-sba-dz")

def get_config():
    """Returns project configuration from .env file."""
    return {
        "WANDB_API_KEY": WANDB_API_KEY,
        "PROJECT_NAME": PROJECT_NAME,
        "ENTITY": ENTITY,
    }
