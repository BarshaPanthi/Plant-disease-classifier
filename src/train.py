
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total_samples = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item() * images.size(0)
        correct       += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)
    return total_loss / total_samples, correct / total_samples * 100


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            loss           = criterion(outputs, labels)
            total_loss    += loss.item() * images.size(0)
            correct       += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)
    return total_loss / total_samples, correct / total_samples * 100


def get_optimizer(model, lr=0.001):
    return Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


def get_scheduler(optimizer):
    return StepLR(optimizer, step_size=5, gamma=0.1)
