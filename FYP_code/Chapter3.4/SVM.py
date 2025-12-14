import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

train_path = r"/Users/jannie/Desktop/train_data.csv"
val_path   = r"/Users/jannie/Desktop/val_data.csv"
test_path  = r"/Users/jannie/Desktop/test_data.csv"

train_df = pd.read_csv(train_path)
val_df   = pd.read_csv(val_path)
test_df  = pd.read_csv(test_path)

print(f"Train : {len(train_df)}")
print(f"Val   : {len(val_df)}")
print(f"Test  : {len(test_df)}")

label_encoder = LabelEncoder()
train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])
test_df['label_encoded'] = label_encoder.transform(test_df['label'])

for i, class_name in enumerate(label_encoder.classes_):
    print(f"  {class_name} → {i}")
#SVM
channels = sorted(train_df['channel'].unique())
models = {}
scalers = {}


for ch in channels:
    print(f"\nChannal{ch} SVM ...")

    train_ch = train_df[train_df['channel'] == ch]
    if train_ch.empty:
        continue

    drop_cols = ['animal_id', 'group', 'label', 'channel', 'segment_id', 'label_encoded']
    X_train = train_ch.drop(columns=drop_cols, errors='ignore').apply(pd.to_numeric, errors='coerce').fillna(0)
    y_train = train_ch['label_encoded']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    scalers[ch] = scaler

    # SVM
    model = SVC(kernel="rbf", C=0.01, gamma="scale", probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    models[ch] = model

print("\nSVM finish")

def ensemble_predict(df, models, scalers, channels, label_encoder):
    segment_results = []
    prob_results = []
    

    group_keys = ['animal_id', 'segment_id', 'label']

    skipped_segments = []
    
    for (animal, seg_id, lbl_str), seg in df.groupby(group_keys):
        present_channels = sorted(seg['channel'].unique())
  
        if len(present_channels) != len(channels) or set(present_channels) != set(channels):
            skipped_segments.append((animal, seg_id, lbl_str, len(present_channels)))
            continue
   
        channel_probs = []
        for ch in channels:
            seg_ch = seg[seg['channel'] == ch]
            X = seg_ch.drop(columns=['animal_id', 'group', 'label', 'channel', 'segment_id', 'label_encoded'], 
                           errors='ignore')
            X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
            X_scaled = scalers[ch].transform(X)
            
            prob = models[ch].predict_proba(X_scaled)[0]
            channel_probs.append(prob)
        
        if len(channel_probs) == len(channels):
            avg_prob = np.mean(channel_probs, axis=0)
            
            ref_model = models[channels[0]]
            pred_encoded = ref_model.classes_[np.argmax(avg_prob)]

            pred_str = label_encoder.inverse_transform([pred_encoded])[0]
            true_str = lbl_str 
            
            segment_results.append([true_str, pred_str, pred_encoded])
            prob_results.append(avg_prob)
    
    if skipped_segments:
        print(f"  跳过了 {len(skipped_segments)} 个segment（通道不全）")
    
    return segment_results, prob_results


test_results, test_probs = ensemble_predict(test_df, models, scalers, channels, label_encoder)

if len(test_results) == 0:
    print("no fitting segment。")
else:
    y_true_str = [x[0] for x in test_results]  
    y_pred_str = [x[1] for x in test_results]  
    y_true_encoded = [x[2] for x in test_results]  
    y_probs_test = np.array(test_probs)

    accuracy_test  = accuracy_score(y_true_str, y_pred_str)
    f1_test        = f1_score(y_true_str, y_pred_str, average="weighted")
    precision_test = precision_score(y_true_str, y_pred_str, average="weighted")
    recall_test    = recall_score(y_true_str, y_pred_str, average="weighted")
    
    print("\n test data:")
    print(f"Accuracy : {accuracy_test:.4f}")
    print(f"F1 Score : {f1_test:.4f}")
    print(f"Precision: {precision_test:.4f}")
    print(f"Recall   : {recall_test:.4f}")
    print(f"sample : {len(y_true_str)}")
    
    for label in sorted(set(y_true_str)):
        count = sum(1 for y in y_true_str if y == label)
        print(f"  {label}: {count}个样本 ({count/len(y_true_str):.1%})")

if len(test_results) > 0:
    cm = confusion_matrix(y_true_str, y_pred_str, labels=label_encoder.classes_)
    plt.figure(figsize=(8, 6))
 
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted label", fontsize=12)
    plt.ylabel("True label", fontsize=12)
    plt.title("Confusion Matrix - SVM (Test Set)", fontsize=14)
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(8, 6))
    
    n_classes = len(label_encoder.classes_)
    
    if n_classes == 2:
        pos_label = 'Ictal' if 'Ictal' in label_encoder.classes_ else label_encoder.classes_[1]
        pos_label_encoded = label_encoder.transform([pos_label])[0]率
        y_scores = y_probs_test[:, pos_label_encoded]
        y_true_binary = np.array([1 if y == pos_label else 0 for y in y_true_str])
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Guessing')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title(f'ROC Curve - {pos_label} vs Others (Test Set)', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_idx]
        
        print(f"最佳阈值: {best_threshold:.4f}")
        print(f"Sensitivity {tpr[best_idx]:.4f}")
        print(f"Specificity{1-fpr[best_idx]:.4f}")
        
    else:
        from sklearn.preprocessing import label_binariz
        y_true_encoded_array = np.array(y_true_encoded)
        y_true_bin = label_binarize(y_true_encoded_array, classes=range(n_classes))
        
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_probs_test.ravel())
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'Micro-average ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Guessing')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Micro-average ROC Curve - SVM (Test Set)', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    
single_accs = []
for ch in channels:
    if ch in models and ch in scalers:
        test_ch = test_df[test_df['channel'] == ch]
        if not test_ch.empty:
            X_test = test_ch.drop(columns=['animal_id', 'group', 'label', 'channel', 'segment_id', 'label_encoded'], 
                                 errors='ignore')
            X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
            X_test_scaled = scalers[ch].transform(X_test)
            y_test_encoded = test_ch['label_encoded']
            
            y_pred_encoded = models[ch].predict(X_test_scaled)
            y_pred_str = label_encoder.inverse_transform(y_pred_encoded)
            y_true_str = test_ch['label'].values
            
            test_acc_single = accuracy_score(y_true_str, y_pred_str)
            single_accs.append(test_acc_single)
            print(f"{ch}\t{test_acc_single:.4f}")
      


if len(test_results) > 0 and single_accs:
    print("\n" + "="*60)
    print("SVM")
    print("="*60)
    
    avg_single_acc = np.mean(single_accs)
    improvement = accuracy_test - avg_single_acc
    improvement_pct = (improvement / avg_single_acc) * 100 if avg_single_acc > 0 else 0
    
    print(f"单通道平均准确率: {avg_single_acc:.4f}")
    print(f"8通道集成准确率: {accuracy_test:.4f}")
    print(f"性能提升: {improvement:.4f} (+{improvement_pct:.1f}%)")
    
   