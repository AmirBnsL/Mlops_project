import os
import json
import glob
import time
import threading
import functools
import requests
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from fastapi import FastAPI
import uvicorn
import nest_asyncio

from src.dataset import Cifar10DataManager
from src.training import run_training_sweep, train_epoch, validate
from src.model import build_model
from src.utils import get_config

# --- Setup ---
cfg = get_config()
WANDB_API_KEY = cfg["WANDB_API_KEY"]
PROJECT_NAME = "cifar10_mlops_project2"
ENTITY = cfg["ENTITY"]

wandb.login(key=WANDB_API_KEY)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nest_asyncio.apply()

# --- 1. Data Preparation ---
dm = Cifar10DataManager(data_dir="./data")
_ = dm.prepare_initial_split()
run = wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type="data_preparation", name="cifar10_v1")
dataset_artifact = wandb.Artifact(
    name="cifar10_dataset",
    type="dataset",
    description="CIFAR-10 Raw Data + Split Indices"
)
dataset_artifact.add_dir("./data")
run.log_artifact(dataset_artifact)
run.finish()
print("Step 1 Complete: Dataset v1 logged.")

# --- 2. Model Training & Sweep ---
prep_run = wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type="training_prep")
prep_run.use_artifact(f'{ENTITY}/{PROJECT_NAME}/cifar10_dataset:latest', type='dataset').download(root="./data")
prep_run.finish()

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'min': 0.001, 'max': 0.1},
        'batch_size': {'values': [64, 128]},
        'optimizer': {'values': ['adam', 'sgd']},
        'architecture_option': {'values': ['standard', 'upsample', 'modified']},
        'epochs': {'value': 5}
    }
}
sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME, entity=ENTITY)
train_func = functools.partial(run_training_sweep, data_dir="./data")
wandb.agent(sweep_id, train_func, count=5, project=PROJECT_NAME, entity=ENTITY)

api = wandb.Api()
best_run = api.sweep(f"{ENTITY}/{PROJECT_NAME}/{sweep_id}").best_run()
os.makedirs("artifacts", exist_ok=True)
with open("artifacts/best_config.json", "w") as f:
    json.dump(best_run.config, f)
print("Step 2 Complete: Sweep finished and best config saved.")

# --- 3. Evaluation ---
run = wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type="evaluation")
run.use_artifact(f'{ENTITY}/{PROJECT_NAME}/cifar10_dataset:latest', type='dataset').download(root="./data")
sweeps = api.project(PROJECT_NAME, entity=ENTITY).sweeps()
sweep_id = sweeps[0].id if sweeps else None
best_run = api.sweep(f"{ENTITY}/{PROJECT_NAME}/{sweep_id}").best_run()
config = best_run.config
model_dir = best_run.logged_artifacts()[0].download(root="./models")
model_path = glob.glob(os.path.join(model_dir, "*.pth"))[0]

dm = Cifar10DataManager(data_dir="./data")
transform_list = [transforms.ToTensor(), transforms.Normalize(dm.mean, dm.std)]
if config['architecture_option'] == 'upsample':
    test_transform = transforms.Compose([transforms.Resize(224)] + transform_list)
else:
    test_transform = transforms.Compose(transform_list)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=test_transform)
indices_path = os.path.join("./data", "processed", "test_indices.npy")
real_test_set = Subset(test_set, np.load(indices_path))
test_loader = DataLoader(real_test_set, batch_size=100, shuffle=False)

model = build_model(config['architecture_option']).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        y_true=all_labels,
        preds=all_preds,
        class_names=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    )
})
run.finish()
print("Step 3 Complete: Evaluation results logged.")

# --- 4. Deployment & Simulation ---
prep_run = wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type="deploy_prep")
prep_run.use_artifact(f'{ENTITY}/{PROJECT_NAME}/cifar10_dataset:latest').download("./data")
sweeps = api.project(PROJECT_NAME, entity=ENTITY).sweeps()
best_run = api.sweep(f"{ENTITY}/{PROJECT_NAME}/{sweeps[0].id}").best_run()
config = best_run.config
model_dir = best_run.logged_artifacts()[0].download(root="./models")
model_path = glob.glob(os.path.join(model_dir, "*.pth"))[0]
prep_run.finish()

model = build_model(config['architecture_option']).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
transform_list = [transforms.ToTensor(), transforms.Normalize(mean, std)]
if config['architecture_option'] == 'upsample':
    val_transform = transforms.Compose([transforms.Resize(224)] + transform_list)
else:
    val_transform = transforms.Compose(transform_list)
sim_data = Cifar10DataManager("./data").get_simulation_data()

app = FastAPI()
@app.post("/predict")
def predict(payload: dict):
    idx = payload.get("index")
    image, _ = sim_data[idx]
    tensor = val_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        conf, pred = torch.max(torch.nn.functional.softmax(output, dim=1), 1)
    return {"prediction": int(pred.item()), "confidence": float(conf.item())}

server_thread = threading.Thread(
    target=lambda: uvicorn.run(app, host="127.0.0.1", port=8005, log_level="error"),
    daemon=True
)
server_thread.start()
time.sleep(3)

run = wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type="deployment_simulation")
feedback_data = []
table = wandb.Table(columns=["index", "pred", "truth", "conf", "correct"])
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for idx in np.random.choice(len(sim_data), 30, replace=False):
    _, gt = sim_data[idx]
    resp = requests.post("http://127.0.0.1:8005/predict", json={"index": int(idx)}).json()
    pred = resp["prediction"]
    correct = (pred == gt)
    table.add_data(idx, classes[pred], classes[gt], resp["confidence"], correct)
    if not correct:
        feedback_data.append((int(idx), int(gt)))
wandb.log({"simulation_results": table})
if feedback_data:
    np.save("feedback_v1.npy", feedback_data)
    art = wandb.Artifact("cifar10-feedback", type="dataset")
    art.add_file("feedback_v1.npy")
    wandb.log_artifact(art)
wandb.finish()
print("Step 4 Complete: Feedback gathered and logged.")

# --- 5. Automated Retraining ---
run = wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type="retrain", tags=["retrain"])
try:
    print("Downloading Feedback...")
    f_art = run.use_artifact(f'{ENTITY}/{PROJECT_NAME}/cifar10-feedback:latest').download(root=".")
    feedback = np.load(os.path.join(f_art, "feedback_v1.npy"))
except:
    print("No feedback found.")
    feedback = []
if len(feedback) > 0:
    print("Downloading Baseline Data...")
    run.use_artifact(f'{ENTITY}/{PROJECT_NAME}/cifar10_dataset:latest').download("./data")
    sweeps = api.project(PROJECT_NAME, entity=ENTITY).sweeps()
    best_run = api.sweep(f"{ENTITY}/{PROJECT_NAME}/{sweeps[0].id}").best_run()
    config = best_run.config
    print("Downloading Baseline Model...")
    m_dir = best_run.logged_artifacts()[0].download(root="./models")
    m_path = glob.glob(os.path.join(m_dir, "*.pth"))[0]
    dm = Cifar10DataManager()
    tf_list = [transforms.ToTensor(), transforms.Normalize(dm.mean, dm.std)]
    train_tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)] + tf_list)
    if config['architecture_option'] == 'upsample':
        train_tf = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip()] + tf_list)
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=train_tf)
    raw_sim = torchvision.datasets.CIFAR10(root="./data", train=False, download=False)
    class FeedbackDS(torch.utils.data.Dataset):
        def __init__(self, raw, inds, tf):
            self.raw = raw; self.inds = [int(i[0]) for i in inds]; self.tf = tf
        def __len__(self): return len(self.inds)
        def __getitem__(self, i):
            img, label = self.raw[self.inds[i]]
            return self.tf(img), label
    fb_ds = FeedbackDS(raw_sim, feedback, train_tf)
    loader = DataLoader(ConcatDataset([train_set, fb_ds]), batch_size=config['batch_size'], shuffle=True)
    test_tf = transforms.Compose(tf_list)
    if config['architecture_option'] == 'upsample':
        test_tf = transforms.Compose([transforms.Resize(224)] + tf_list)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=test_tf)
    test_indices = np.load(os.path.join("./data", "processed", "test_indices.npy"))
    test_loader = DataLoader(Subset(test_set, test_indices), batch_size=config['batch_size'], shuffle=False)
    model = build_model(config['architecture_option']).to(device)
    model.load_state_dict(torch.load(m_path, map_location=device))
    lr = config.get('learning_rate', 0.001)
    if config.get('optimizer') == 'sgd':
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    print("Fine-tuning...")
    for e in range(5):
        train_loss = train_epoch(model, loader, crit, opt, device)
        val_loss, val_acc = validate(model, test_loader, crit, device)
        print(f"Epoch {e+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        wandb.log({"retrain_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc, "epoch": e})
    torch.save(model.state_dict(), "retrained.pth")
    art = wandb.Artifact("retrained-model", type="model")
    art.add_file("retrained.pth")
    run.log_artifact(art)
    print("Retraining Complete.")
else:
    print("Skipping.")
run.finish()
print("Step 5 Complete: Automated retraining finished.")
