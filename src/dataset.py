import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import wandb
from torch.utils.data import Subset, DataLoader

class Cifar10DataManager:
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)

    def get_transforms(self, architecture_option='standard'):
        # Base transforms
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ]
        
        train_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)
        ] + transform_list

        if architecture_option == 'upsample':
            transform_list.insert(0, transforms.Resize(224))
            train_transforms.insert(0, transforms.Resize(224))

        return transforms.Compose(train_transforms), transforms.Compose(transform_list)

    def prepare_initial_split(self):
        """
        Downloads CIFAR-10.
        Splits Test set (10k) into:
        - Test (8k): For model evaluation
        - Simulation (2k): For live traffic simulation (Holdout)
        Logs artifacts to W&B.
        """
        # Download raw data
        train_set = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        test_set = torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True)
        
        # Split Test Set
        indices = list(range(len(test_set)))
        # Shuffle deterministically for reproducibility
        np.random.seed(42)
        np.random.shuffle(indices)
        
        test_indices = indices[:8000]
        sim_indices = indices[8000:]
        
        # Save indices to disk to ensure we load the same split later
        os.makedirs(os.path.join(self.data_dir, "processed"), exist_ok=True)
        np.save(os.path.join(self.data_dir, "processed", "test_indices.npy"), test_indices)
        np.save(os.path.join(self.data_dir, "processed", "sim_indices.npy"), sim_indices)
        
        return train_set, test_set, test_indices, sim_indices

    def log_dataset_artifact(self, run, name, description, filepath=None, type="dataset"):
        artifact = wandb.Artifact(name, type=type, description=description)
        if filepath:
            artifact.add_file(filepath)
        else:
            artifact.add_dir(self.data_dir) # Simplified: logging whole data dir
        run.log_artifact(artifact)
        
    def get_loaders(self, batch_size, architecture_option='standard', version="v1"):
        train_transform, test_transform = self.get_transforms(architecture_option)
        
        train_set = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=test_transform)
        
        # Load indices
        test_indices = np.load(os.path.join(self.data_dir, "processed", "test_indices.npy"))
        
        # Create Subsets
        real_test_set = Subset(test_set, test_indices)
        
        # If version is v2, we might append new data to train_set here
        # (Implementation of connecting feedback loop data would go here)
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(real_test_set, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, test_loader

    def get_simulation_data(self):
        """Returns the raw Simulation subset (PIL images) for inference"""
        test_set_raw = torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=None) # No transform for raw
        sim_indices = np.load(os.path.join(self.data_dir, "processed", "sim_indices.npy"))
        return Subset(test_set_raw, sim_indices)
