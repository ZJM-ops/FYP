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

class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class Inception1D(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception1D, self).__init__()
        
        # 分支 1: 1x1 卷积
        self.branch1 = BasicConv1d(in_channels, ch1x1, kernel_size=1)

        # 分支 2: 1x1 卷积 -> 3x3 卷积
        self.branch2 = nn.Sequential(
            BasicConv1d(in_channels, ch3x3red, kernel_size=1),
            BasicConv1d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        # 分支 3: 1x1 卷积 -> 5x5 卷积
        self.branch3 = nn.Sequential(
            BasicConv1d(in_channels, ch5x5red, kernel_size=1),
            BasicConv1d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        # 分支 4: 3x3 池化 -> 1x1 卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            BasicConv1d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # dim=1
        return torch.cat([branch1, branch2, branch3, branch4], 1)

class GoogleNet1D(nn.Module):
    def __init__(self, num_classes=2, in_channels=8):
        super(GoogleNet1D, self).__init__()
        
        # Stem 
        self.conv1 = BasicConv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool1d(3, stride=2, padding=1)
        
        self.conv2 = BasicConv1d(64, 64, kernel_size=1)
        self.conv3 = BasicConv1d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool1d(3, stride=2, padding=1)

        #Inception 模块
        self.inception3a = Inception1D(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception1D(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool1d(3, stride=2, padding=1)

        self.inception4a = Inception1D(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception1D(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception1D(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception1D(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception1D(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool1d(3, stride=2, padding=1)

        self.inception5a = Inception1D(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception1D(832, 384, 192, 384, 48, 128, 128)

        # 分类
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # N x 8 x Length
        x = self.conv1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
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
        # Shape: Channels, Length
        data = np.load(path) 
        tensor_data = torch.from_numpy(data).float()
        label_str = path.split(os.sep)[-2] 
        label = self.label_mapping[label_str]
        
        return tensor_data, label

split_data_dir = '/home/student/s230005071/shu/split_data'
train_path_list = os.path.join(split_data_dir, 'train_1d.txt')
val_path_list = os.path.join(split_data_dir, 'val_1d.txt')
test_path_list = os.path.join(split_data_dir, 'test_1d.txt')

model_save_dir = '/home/student/s230005071/shu/cnn_models_1d'
os.makedirs(model_save_dir, exist_ok=True)

BATCH_SIZE = 16
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
EPOCHS = 100
NUM_CLASSES = 2
INPUT_CHANNELS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10

run_dir_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_googlenet_1d_b{BATCH_SIZE}_sgd"
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

model = GoogleNet1D(num_classes=NUM_CLASSES, in_channels=INPUT_CHANNELS).to(DEVICE)

# SGD
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
best_model_path = os.path.join(run_save_dir, 'best_googlenet_1d.pth')

print("\n--- Starting Training (GoogleNet 1D) ---")
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

print("\n--- Test Results (GoogleNet 1D) ---")
print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
print("Confusion Matrix:\n", cm)

pd.DataFrame({
    "accuracy": [acc], "f1_score": [f1], "precision": [precision], "recall": [recall]
}).to_csv(os.path.join(run_save_dir, "test_metrics.csv"), index=False)

pd.DataFrame(cm).to_csv(os.path.join(run_save_dir, "confusion_matrix.csv"), index=False)
