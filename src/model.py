import torch
import torch.nn as nn
import torchvision

def build_model(architecture_option='standard', num_classes=10, pretrained=True):
    model = torchvision.models.resnet18(pretrained=pretrained)
    
    if architecture_option == 'modified':
        # Option B: Pro Way - Small kernels for small images
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    
    # Replace Head
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
