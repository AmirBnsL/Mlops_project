import torch
import torch.nn as nn
import torchvision

def build_model(architecture_option='standard', num_classes=10, pretrained=True):
    """
    Builds the model based on the architecture option.
    Options:
    - standard: ResNet18 (32x32 images)
    - upsample: ResNet18 (224x224 images via transforms)
    - modified: ResNet18 with modified first layer for small inputs
    """
    model = torchvision.models.resnet18(pretrained=pretrained)
    
    if architecture_option == 'modified':
        # Pro Way - Small kernels for small images
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif architecture_option == 'upsample':
        # Upsample logic is handled in transforms; model remains standard
        pass
    
    # Replace Head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model
