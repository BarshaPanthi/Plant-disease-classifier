
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_binary_label(original_label, class_names):
    return 0 if "healthy" in class_names[original_label].lower() else 1

class PlantDiseaseDataset(torch.utils.data.Dataset):
    def __init__(self, subset, class_names):
        self.subset = subset
        self.class_names = class_names
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        image, original_label = self.subset[idx]
        return image, get_binary_label(original_label, self.class_names)

def get_dataloaders(data_dir="./data/PlantVillage", batch_size=32):
    train_folder = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    val_folder   = datasets.ImageFolder(root=data_dir, transform=val_transforms)
    class_names  = train_folder.classes
    all_indices  = list(range(len(train_folder)))
    all_labels   = train_folder.targets
    train_idx, temp_idx = train_test_split(all_indices, test_size=0.30, stratify=all_labels, random_state=42)
    temp_labels  = [all_labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=temp_labels, random_state=42)
    train_ds = PlantDiseaseDataset(Subset(train_folder, train_idx), class_names)
    val_ds   = PlantDiseaseDataset(Subset(val_folder,   val_idx),   class_names)
    test_ds  = PlantDiseaseDataset(Subset(val_folder,   test_idx),  class_names)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    print(f"Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")
    return train_loader, val_loader, test_loader, class_names
