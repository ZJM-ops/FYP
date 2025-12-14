import timm
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime

class MultiChannelDataset(Dataset):
    def __init__(self, file_paths, transform=None, single_channel_index=None):
        self.segs = {}
        for p in file_paths:
            parts = p.split(os.sep)
            label_folder = parts[-2]
            filename = parts[-1]

            seg_id_parts = filename.split('_seg')
            if len(seg_id_parts) > 1:
                base_name = '_seg'.join(seg_id_parts[:-1])
                seg_num = seg_id_parts[-1].split('_')[0]
                seg_id = f"{label_folder}_{base_name}_seg{seg_num}"
            else:
                seg_id = filename.split('_ch')[0]

            if seg_id not in self.segs:
                self.segs[seg_id] = []
            self.segs[seg_id].append(p)

        self.seg_ids = [k for k, v in self.segs.items() if len(v) == 8]
        for k in self.seg_ids:
            self.segs[k] = sorted(self.segs[k], key=lambda x: int(x.split("ch")[-1].split(".")[0]))

        self.transform = transform
        self.label_mapping = {'Baseline': 0, 'Ictal': 1}
        self.single_channel_index = single_channel_index

    def __len__(self):
        return len(self.seg_ids)

    def __getitem__(self, idx):
        seg_id = self.seg_ids[idx]
        img_paths = self.segs[seg_id]

        if len(img_paths) != 8:
            raise ValueError(f"Sample {seg_id} does not have 8 channels, found {len(img_paths)}")

        if self.single_channel_index is not None:
            img_path = img_paths[self.single_channel_index - 1]
            imgs = Image.open(img_path).convert("RGB")
            if self.transform:
                imgs = self.transform(imgs)
        else:
            imgs = [Image.open(p).convert("RGB") for p in img_paths]
            if self.transform:
                imgs = [self.transform(img) for img in imgs]
            imgs = torch.cat(imgs, dim=0)

        label = self.label_mapping[img_paths[0].split(os.sep)[-2]]
        return imgs, label

split_data_dir = '/home/student/s230005071/shu/split_data_42'
train_images_path = os.path.join(split_data_dir, 'train_images.txt')
val_images_path = os.path.join(split_data_dir, 'val_images.txt')
test_images_path = os.path.join(split_data_dir, 'test_images.txt')

model_save_dir = '/home/student/s230005071/shu/cnn_models_1206'
os.makedirs(model_save_dir, exist_ok=True)

BATCH_SIZE = 16
LEARNING_RATE = 0.1
WEIGHT_DECAY = 1e-4
EPOCHS = 100
NUM_CLASSES = 2
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10

run_dir_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_resnext101_b{BATCH_SIZE}_sgd{LEARNING_RATE}_wd{WEIGHT_DECAY}"
run_save_dir = os.path.join(model_save_dir, run_dir_name)
os.makedirs(run_save_dir, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}, Weight Decay: {WEIGHT_DECAY}")
print(f"Early Stopping Patience: {PATIENCE}, Image Size: {IMAGE_SIZE}")

with open(train_images_path, 'r') as f:
    train_paths = [line.strip() for line in f]
with open(val_images_path, 'r') as f:
    val_paths = [line.strip() for line in f]
with open(test_images_path, 'r') as f:
    test_paths = [line.strip() for line in f]

base_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def create_resnext101_32x4d(num_classes=2, pretrained=False):
    """ 使用 timm 库创建 ResNeXt-101 (32x4d) 模型。 """
    model = timm.create_model('resnext101_32x4d', pretrained=pretrained, num_classes=num_classes)
    return model.to(DEVICE)

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
            print(f"EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

all_test_probs = []
test_labels = []
channel_results = []

for i in range(1, 9): # 8个通道
    print(f"\n--- Training ResNeXt-101 for Channel {i} (SGD LR={LEARNING_RATE}, WD={WEIGHT_DECAY}) ---")

    train_dataset = MultiChannelDataset(train_paths, transform=base_transform, single_channel_index=i)
    val_dataset = MultiChannelDataset(val_paths, transform=base_transform, single_channel_index=i)
    test_dataset = MultiChannelDataset(test_paths, transform=base_transform, single_channel_index=i)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = create_resnext101_32x4d(num_classes=NUM_CLASSES, pretrained=False)

    optimizer = optim.SGD(
        model.parameters(), 
        lr=LEARNING_RATE, 
        momentum=0.9, 
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_model_path = os.path.join(run_save_dir, f'best_resnext101_32x4d_ch{i}_sgd.pth')
    
    early_stopping = EarlyStopping(patience=PATIENCE)

   for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Ch{i} Training"):
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

        print(f"Epoch {epoch+1}/{EPOCHS}, Ch{i} Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        early_stopping(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1} for channel {i}")
            break

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    channel_probs = []
    channel_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            channel_probs.extend(probs.cpu().numpy())
            channel_labels.extend(labels.cpu().numpy())

    all_test_probs.append(channel_probs)
    if not test_labels:
        test_labels = channel_labels

    preds = (np.array(channel_probs) > 0.5).astype(int)
    acc = accuracy_score(channel_labels, preds)
    f1 = f1_score(channel_labels, preds)
    precision = precision_score(channel_labels, preds)
    recall = recall_score(channel_labels, preds)
    fpr, tpr, _ = roc_curve(channel_labels, np.array(channel_probs))
    roc_auc = auc(fpr, tpr)

    channel_results.append({
        'channel': i,
        'accuracy': acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'auc': roc_auc
    })

    print(f"Channel {i} Metrics: Acc={acc:.4f}, F1={f1:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, AUC={roc_auc:.4f}")

    del model, optimizer, scheduler, criterion, early_stopping
    del train_dataset, val_dataset, test_dataset
    del train_loader, val_loader, test_loader
    torch.cuda.empty_cache()

channel_results_df = pd.DataFrame(channel_results)
channel_results_df.to_csv(os.path.join(run_save_dir, "individual_channel_metrics.csv"), index=False)
print("\n--- Individual channel metrics saved. ---")

print("\n--- All 8 models trained. Performing Ensemble. ---")
all_test_probs = np.array(all_test_probs)
ensemble_probs = np.mean(all_test_probs, axis=0)
ensemble_preds = (ensemble_probs > 0.5).astype(int)

acc = accuracy_score(test_labels, ensemble_preds)
f1 = f1_score(test_labels, ensemble_preds)
precision = precision_score(test_labels, ensemble_preds)
recall = recall_score(test_labels, ensemble_preds)
cm = confusion_matrix(test_labels, ensemble_preds)

print(f"Test Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
print("Confusion Matrix:\n", cm)

pd.DataFrame({
    "acc": [acc],
    "f1": [f1],
    "precision": [precision],
    "recall": [recall]
}).to_csv(os.path.join(run_save_dir, "test_metrics_ensemble.csv"), index=False)

pd.DataFrame(cm).to_csv(os.path.join(run_save_dir, "confusion_matrix_ensemble.csv"), index=False)
