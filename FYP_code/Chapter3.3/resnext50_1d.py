import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# --- Import the scheduler --- ##
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from datetime import datetime

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"  -> EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# 1D-ResNeXt Model and Dataset Definitions ---
class Conv1DBlock(nn.Module):
    """A basic module containing Conv-BN-ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlockBottleneck1D(nn.Module):
    """The core Bottleneck block for ResNeXt"""
    def __init__(self, in_channels, bottleneck_channels, out_channels, stride, cardinality):
        super().__init__()
        self.conv1 = Conv1DBlock(in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = Conv1DBlock(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1)
        # --- Grouped Convolution ---
        self.conv2.conv = nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.conv3 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        # --- Shortcut Connection ---
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn3(self.conv3(x))
        x += shortcut
        return self.relu(x)

class ResNeXt1D(nn.Module):
    """The complete ResNeXt-1D model"""
    def __init__(self, block, layers_config, num_classes, in_channels, cardinality):
        super().__init__()
        self.cardinality = cardinality
        self.in_channels = 64 # Initial in_channels for the first layer

        # Stem
        self.stem = nn.Sequential(
            Conv1DBlock(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # ResNeXt Layers
        self.layer1 = self._make_layer(block, layers_config[0], bottleneck_channels=128, out_channels=256, stride=1)
        self.layer2 = self._make_layer(block, layers_config[1], bottleneck_channels=256, out_channels=512, stride=2)
        self.layer3 = self._make_layer(block, layers_config[2], bottleneck_channels=512, out_channels=1024, stride=2)
        self.layer4 = self._make_layer(block, layers_config[3], bottleneck_channels=1024, out_channels=2048, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, block, num_blocks, bottleneck_channels, out_channels, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, bottleneck_channels, out_channels, s, self.cardinality))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNeXt50_1D(num_classes=2, in_channels=8, cardinality=32):
    """Constructs a ResNeXt-50 (32x4d) model for 1D data."""
    # [3, 4, 6, 3] corresponds to the number of blocks in each layer for ResNet-50
    return ResNeXt1D(ResidualBlockBottleneck1D, [3, 4, 6, 3], num_classes, in_channels, cardinality)

class EEG_1D_Dataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.label_mapping = {'Baseline': 0, 'Ictal': 1}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        data = np.load(path) # Shape (Channels, Length), e.g., (8, 512)
        tensor_data = torch.from_numpy(data).float()

        label_str = path.split(os.sep)[-2] 
        label = self.label_mapping[label_str]
        
        return tensor_data, label


data_list_dir = '/home/student/s230005071/shu/split_data'
train_path_list = os.path.join(data_list_dir, 'train_1d.txt')
val_path_list = os.path.join(data_list_dir, 'val_1d.txt')
test_path_list = os.path.join(data_list_dir, 'test_1d.txt')

model_save_dir = '/home/student/s230005071/shu/cnn_models_1d'
os.makedirs(model_save_dir, exist_ok=True)


run_dir_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_resnext50_1d"
run_save_dir = os.path.join(model_save_dir, run_dir_name)
os.makedirs(run_save_dir, exist_ok=True)


BATCH_SIZE = 16 
LEARNING_RATE = 0.01 
EPOCHS = 100
NUM_CLASSES = 2
INPUT_CHANNELS = 8 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10 

print(f"Using device: {DEVICE}, Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}, Optimizer: SGD (Mom=0.9)")

with open(train_path_list, 'r') as f:
    train_paths = [line.strip() for line in f]
with open(val_path_list, 'r') as f:
    val_paths = [line.strip() for line in f]
with open(test_path_list, 'r') as f:
    test_paths = [line.strip() for line in f]

train_dataset = EEG_1D_Dataset(train_paths)
val_dataset = EEG_1D_Dataset(val_paths)
test_dataset = EEG_1D_Dataset(test_paths)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Model, Optimizer, Loss Function
model = ResNeXt50_1D(num_classes=NUM_CLASSES, in_channels=INPUT_CHANNELS).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS) 
criterion = nn.CrossEntropyLoss()
early_stopping = EarlyStopping(patience=PATIENCE) 

best_val_loss = float('inf')
best_model_path = os.path.join(run_save_dir, 'best_model.pth')

print("--- Starting Training (SGD) ---")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
    val_loss /= len(val_loader.dataset)
    
    scheduler.step() 

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    early_stopping(val_loss) 
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"  -> Val loss decreased. Saved best model.")
    
    if early_stopping.early_stop: 
        print(f"Early stopping triggered at epoch {epoch+1}.")
        break

print("\n--- Training finished. Starting testing on the best model. ---")
model.load_state_dict(torch.load(best_model_path))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="[Testing]"):
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print("\n--- Test Results ---")
print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
print("Confusion Matrix:\n", cm)

results_df = pd.DataFrame({
    "accuracy": [acc], "f1_score": [f1], "precision": [precision], "recall": [recall]
})
results_df.to_csv(os.path.join(run_save_dir, "test_metrics.csv"), index=False)
pd.DataFrame(cm).to_csv(os.path.join(run_save_dir, "confusion_matrix.csv"), index=False)
