
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def build_model(num_classes=2, device='cpu'):
    """
    Builds MobileNetV3 Small for binary plant disease classification.

    Args:
        num_classes : number of output classes (default 2)
        device      : 'cuda' or 'cpu'

    Returns:
        model : MobileNetV3 ready for training
    """
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    model = model.to(device)
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = build_model(num_classes=2, device=device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model built on {device}')
    print(f'Total params    : {total:,}')
    print(f'Trainable params: {trainable:,}')
