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

# --- 1. Early Stopping 类 ---
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

# --- 2. 1D Inception-ResNet-v2 模型定义 ---

class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class InceptionResNetA(nn.Module):
    """
    Inception-ResNet-A 模块 (1D Version)
    """
    def __init__(self, in_channels, scale=1.0):
        super(InceptionResNetA, self).__init__()
        self.scale = scale
        
        # Branch 0: 1x1
        self.branch0 = BasicConv1d(in_channels, 32, kernel_size=1)
        
        # Branch 1: 1x1 -> 3x3
        self.branch1 = nn.Sequential(
            BasicConv1d(in_channels, 32, kernel_size=1),
            BasicConv1d(32, 32, kernel_size=3, padding=1)
        )
        
        # Branch 2: 1x1 -> 3x3 -> 3x3
        self.branch2 = nn.Sequential(
            BasicConv1d(in_channels, 32, kernel_size=1),
            BasicConv1d(32, 48, kernel_size=3, padding=1),
            BasicConv1d(48, 64, kernel_size=3, padding=1)
        )
        
        # 1x1 conv to match input channels (32+32+64 = 128 -> in_channels)
        # Note: Standard Inception-ResNet-v2 keeps dimensions consistent
        self.conv2d = nn.Conv1d(128, in_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        return self.relu(out)

class InceptionResNetB(nn.Module):
    """
    Inception-ResNet-B 模块 (1D Version)
    """
    def __init__(self, in_channels, scale=1.0):
        super(InceptionResNetB, self).__init__()
        self.scale = scale
        
        # Branch 0: 1x1
        self.branch0 = BasicConv1d(in_channels, 192, kernel_size=1)
        
        # Branch 1: 1x1 -> 1x7 -> 7x1 (Simulated as 1x5 -> 1x5 in 1D or just larger kernels)
        # We use standard 1D kernels to approximate receptive field
        self.branch1 = nn.Sequential(
            BasicConv1d(in_channels, 128, kernel_size=1),
            BasicConv1d(128, 160, kernel_size=5, padding=2), # Approx
            BasicConv1d(160, 192, kernel_size=5, padding=2)  # Approx
        )
        
        self.conv2d = nn.Conv1d(384, in_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        return self.relu(out)

class InceptionResNetC(nn.Module):
    """
    Inception-ResNet-C 模块 (1D Version)
    """
    def __init__(self, in_channels, scale=1.0):
        super(InceptionResNetC, self).__init__()
        self.scale = scale
        
        # Branch 0: 1x1
        self.branch0 = BasicConv1d(in_channels, 192, kernel_size=1)
        
        # Branch 1: 1x1 -> 1x3 -> 3x1 (Simulated as 1x3 in 1D)
        self.branch1 = nn.Sequential(
            BasicConv1d(in_channels, 192, kernel_size=1),
            BasicConv1d(192, 224, kernel_size=3, padding=1),
            BasicConv1d(224, 256, kernel_size=3, padding=1)
        )
        
        self.conv2d = nn.Conv1d(448, in_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        return self.relu(out)

class ReductionA(nn.Module):
    def __init__(self, in_channels, k, l, m, n):
        super(ReductionA, self).__init__()
        # Branch 0: MaxPool
        self.branch0 = nn.MaxPool1d(3, stride=2, padding=1)
        
        # Branch 1: 3x3 Conv stride 2
        self.branch1 = BasicConv1d(in_channels, n, kernel_size=3, stride=2, padding=1)
        
        # Branch 2: 1x1 -> 3x3 -> 3x3 stride 2
        self.branch2 = nn.Sequential(
            BasicConv1d(in_channels, k, kernel_size=1),
            BasicConv1d(k, l, kernel_size=3, padding=1),
            BasicConv1d(l, m, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return torch.cat((x0, x1, x2), 1)

class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super(ReductionB, self).__init__()
        self.branch0 = nn.MaxPool1d(3, stride=2, padding=1)
        
        self.branch1 = nn.Sequential(
            BasicConv1d(in_channels, 256, kernel_size=1),
            BasicConv1d(256, 384, kernel_size=3, stride=2, padding=1)
        )
        
        self.branch2 = nn.Sequential(
            BasicConv1d(in_channels, 256, kernel_size=1),
            BasicConv1d(256, 288, kernel_size=3, stride=2, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            BasicConv1d(in_channels, 256, kernel_size=1),
            BasicConv1d(256, 288, kernel_size=3, padding=1),
            BasicConv1d(288, 320, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        return torch.cat((x0, x1, x2, x3), 1)

class InceptionResNetV2_1D(nn.Module):
    def __init__(self, num_classes=2, in_channels=8):
        super(InceptionResNetV2_1D, self).__init__()
        
        # Stem
        self.conv1 = BasicConv1d(in_channels, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv1d(32, 32, kernel_size=3)
        self.conv3 = BasicConv1d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool1d(3, stride=2)
        self.conv4 = BasicConv1d(64, 80, kernel_size=1)
        self.conv5 = BasicConv1d(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool1d(3, stride=2)
        
        # Blocks
        self.mixed_5b = ReductionA(192, 192, 208, 256, 384) # Result channels: 192+384+256 = 832 (approx)
        # Note: To simplify and match tensor sizes without complex math, we use a simpler linear stem->block transition
        # Let's align channels to standard Inception-ResNet sizes
        self.adjust_channels = BasicConv1d(832, 320, kernel_size=1) # Adjust to 320 for Block A input

        self.repeat_a = nn.Sequential(
            InceptionResNetA(320, 0.17),
            InceptionResNetA(320, 0.17),
            InceptionResNetA(320, 0.17),
            InceptionResNetA(320, 0.17),
            InceptionResNetA(320, 0.17)
        )
        
        self.reduction_a = ReductionA(320, 192, 192, 256, 384) # 320 + 384 + 256 = 960 (Output channels)
        
        self.repeat_b = nn.Sequential(
            InceptionResNetB(960, 0.10),
            InceptionResNetB(960, 0.10),
            InceptionResNetB(960, 0.10),
            InceptionResNetB(960, 0.10),
            InceptionResNetB(960, 0.10)
        )
        
        self.reduction_b = ReductionB(960) # 960 + 384 + 288 + 320 = 1952 (approx 2048 in original)
        
        self.repeat_c = nn.Sequential(
            InceptionResNetC(1952, 0.20),
            InceptionResNetC(1952, 0.20),
            InceptionResNetC(1952, 0.20)
        )
        
        self.conv_last = BasicConv1d(1952, 1536, kernel_size=1)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1536, num_classes)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool2(x)
        
        x = self.mixed_5b(x)
        x = self.adjust_channels(x) # Bridge to Block A
        
        x = self.repeat_a(x)
        x = self.reduction_a(x)
        x = self.repeat_b(x)
        x = self.reduction_b(x)
        x = self.repeat_c(x)
        x = self.conv_last(x)
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# --- 3. 数据集定义 (读取 .npy) ---
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

# --- 4. 配置 ---
split_data_dir = '/home/student/s230005071/shu/split_data_42'
train_path_list = os.path.join(split_data_dir, 'train_1d.txt')
val_path_list = os.path.join(split_data_dir, 'val_1d.txt')
test_path_list = os.path.join(split_data_dir, 'test_1d.txt')

model_save_dir = '/home/student/s230005071/shu/cnn_models_1d_1206'
os.makedirs(model_save_dir, exist_ok=True)

# 超参数
BATCH_SIZE = 16
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
EPOCHS = 100
NUM_CLASSES = 2
INPUT_CHANNELS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10

# 运行目录
run_dir_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_inception_resnet_v2_1d_b{BATCH_SIZE}_sgd"
run_save_dir = os.path.join(model_save_dir, run_dir_name)
os.makedirs(run_save_dir, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}, Weight Decay: {WEIGHT_DECAY}")

# --- 5. 数据加载 ---
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

# --- 6. 模型与训练设置 ---
model = InceptionResNetV2_1D(num_classes=NUM_CLASSES, in_channels=INPUT_CHANNELS).to(DEVICE)

# 优化器 SGD
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
best_model_path = os.path.join(run_save_dir, 'best_inception_resnet_v2_1d.pth')

# --- 7. 训练循环 ---
print("\n--- Starting Training (Inception-ResNet-v2 1D) ---")
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

    # 验证
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

# --- 8. 测试 ---
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

# 计算指标
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print("\n--- Test Results (Inception-ResNet-v2 1D) ---")
print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
print("Confusion Matrix:\n", cm)

# 保存结果
pd.DataFrame({
    "accuracy": [acc], "f1_score": [f1], "precision": [precision], "recall": [recall]
}).to_csv(os.path.join(run_save_dir, "test_metrics.csv"), index=False)

pd.DataFrame(cm).to_csv(os.path.join(run_save_dir, "confusion_matrix.csv"), index=False)

print(f"\n--- Script Finished. Results saved in {run_save_dir} ---")