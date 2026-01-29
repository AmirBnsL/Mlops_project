import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import os
from pathlib import Path
from .dataset import Cifar10DataManager
from .model import build_model

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(loader), 100 * correct / total

def run_training_sweep(config=None, data_dir="./data"):
    # This function is called by wandb.agent
    with wandb.init(config=config):
        cfg = wandb.config
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data
        dm = Cifar10DataManager(data_dir=data_dir)
        train_loader, test_loader = dm.get_loaders(cfg.batch_size, architecture_option=cfg.architecture_option)
        
        # Model
        model = build_model(cfg.architecture_option).to(device)
        
        # Optimization
        criterion = nn.CrossEntropyLoss()
        if cfg.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)
        else:
            optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
            
        best_acc = 0.0
        
        for epoch in range(cfg.epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, test_loader, criterion, device)
            
            wandb.log({
                "epoch": epoch, 
                "train_loss": train_loss, 
                "val_loss": val_loss, 
                "val_acc": val_acc
            })
            
            if val_acc > best_acc:
                best_acc = val_acc
                # Save Best Model State locally
                os.makedirs("models", exist_ok=True)
                model_path = f"models/model_{wandb.run.id}.pth"
                torch.save(model.state_dict(), model_path)
                
                # Log as W&B Artifact
                artifact = wandb.Artifact(f"model-{wandb.run.id}", type="model")
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)
