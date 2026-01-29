from dotenv import load_dotenv
import os

def load_env_vars():
    load_dotenv()
    return {
        "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
        "WANDB_ENTITY": os.getenv("WANDB_ENTITY"),
        "WANDB_PROJECT": os.getenv("WANDB_PROJECT"),
    }
