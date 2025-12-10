import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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

# 1D ResNet-50
def conv3x1(in_planes, out_planes, stride=1):
    """1D 3x1 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)

class Bottleneck1D(nn.Module):
    # ResNet-50/101/152  Bottleneck，expansion =4
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        # 1x1 conv
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        
        # 3x3 conv
        self.conv2 = conv3x1(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        
        # 1x1 conv (expansion)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet50_1D(nn.Module):
    def __init__(self, num_classes=2, in_channels=8):
        super(ResNet50_1D, self).__init__()
        self.inplanes = 64
        
        # Stem: 7x1 conv, stride 2
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Layers: [3, 4, 6, 3] blocks for ResNet-50
        self.layer1 = self._make_layer(Bottleneck1D, 64, 3)
        self.layer2 = self._make_layer(Bottleneck1D, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck1D, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck1D, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * Bottleneck1D.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class EEG_1D_Dataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.label_mapping = {'Baseline': 0, 'Ictal': 1}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        data = np.load(path) 
        tensor_data = torch.from_numpy(data).float()
        
        label_str = path.split(os.sep)[-2] 
        label = self.label_mapping[label_str]
        return tensor_data, label

split_data_dir = '/home/student/s230005071/shu/split_data_42'
train_path_list = os.path.join(split_data_dir, 'train_1d.txt')
val_path_list = os.path.join(split_data_dir, 'val_1d.txt')
test_path_list = os.path.join(split_data_dir, 'test_1d.txt')

model_save_dir = '/home/student/s230005071/shu/cnn_models_1d_1206'
os.makedirs(model_save_dir, exist_ok=True)

BATCH_SIZE = 16
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
EPOCHS = 100
NUM_CLASSES = 2
INPUT_CHANNELS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10

run_dir_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_resnet50_1d_b{BATCH_SIZE}_sgd"
run_save_dir = os.path.join(model_save_dir, run_dir_name)
os.makedirs(run_save_dir, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}, Weight Decay: {WEIGHT_DECAY}")

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

model = ResNet50_1D(num_classes=NUM_CLASSES, in_channels=INPUT_CHANNELS).to(DEVICE)

#SGD
optimizer = optim.SGD(
    model.parameters(), 
    lr=LEARNING_RATE, 
    momentum=0.9, 
    weight_decay=WEIGHT_DECAY
)

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()
early_stopping = EarlyStopping(patience=PATIENCE)

best_val_loss = float('inf')
best_model_path = os.path.join(run_save_dir, 'best_resnet50_1d.pth')

print("\n--- Starting Training (ResNet-50 1D) ---")
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
        print("  -> Saved best model.")
    
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

print("\n--- Test Results (ResNet-50 1D) ---")
print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
print("Confusion Matrix:\n", cm)

pd.DataFrame({
    "accuracy": [acc], "f1_score": [f1], "precision": [precision], "recall": [recall]
}).to_csv(os.path.join(run_save_dir, "test_metrics.csv"), index=False)

pd.DataFrame(cm).to_csv(os.path.join(run_save_dir, "confusion_matrix.csv"), index=False)
