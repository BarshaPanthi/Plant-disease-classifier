
import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

sys.path.append('./src')
from dataset  import get_dataloaders
from model    import build_model
from train    import train_one_epoch, validate
from evaluate import evaluate, print_metrics


# Config
DATA_DIR   = './data/PlantVillage'
MODEL_PATH = './outputs/best_model.pth'
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR         = 0.001

os.makedirs('./outputs/results', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load data
print('Loading data...')
train_loader, val_loader, test_loader, class_names = get_dataloaders(
    data_dir=DATA_DIR, batch_size=BATCH_SIZE
)

# Build model
print('Building model...')
model     = build_model(num_classes=2, device=device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Train
print(f'Training for {NUM_EPOCHS} epochs...')
best_val_acc = 0.0
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss,   val_acc   = validate(model, val_loader, criterion, device)
    scheduler.step()
    print(f'Epoch {epoch}/{NUM_EPOCHS}  Train Acc: {train_acc:.2f}%  Val Acc: {val_acc:.2f}%')
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f'  Best model saved.')

# Evaluate on test set
print()
print('Evaluating on test set...')
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
preds, labels = evaluate(model, test_loader, device)
print_metrics(preds, labels)
