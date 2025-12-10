import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv("/Users/jannie/Desktop/train_data.csv")
val_df   = pd.read_csv("/Users/jannie/Desktop/val_data.csv")
test_df  = pd.read_csv("/Users/jannie/Desktop/test_data.csv")


label_encoder = LabelEncoder()
train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])
test_df['label_encoded']  = label_encoder.transform(test_df['label'])

channels = sorted(train_df['channel'].unique())
models = {}
scalers = {}

for ch in channels:
    train_ch = train_df[train_df['channel'] == ch]
    if train_ch.empty:
        continue

    drop_cols = ['animal_id','group','label','channel','segment_id','label_encoded']
    X_train = train_ch.drop(columns=drop_cols, errors='ignore').apply(pd.to_numeric, errors='coerce').fillna(0)
    y_train = train_ch['label_encoded']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    scalers[ch] = scaler

    model = LogisticRegression(
        C=1.0, penalty='l2', solver='lbfgs', max_iter=1000,
        random_state=42, multi_class='ovr'
    )
    model.fit(X_train_scaled, y_train)
    models[ch] = model

def ensemble_predict(df, models, scalers, channels):
    segment_results = []
    prob_results = []
    group_keys = ['animal_id', 'segment_id', 'label']

    for (animal, seg_id, lbl_str), seg in df.groupby(group_keys):

        present_channels = sorted(seg['channel'].unique())
        if set(present_channels) != set(channels):
            continue

        channel_probs = []
        for ch in channels:
            seg_ch = seg[seg['channel'] == ch]
            X = seg_ch.drop(columns=[
                'animal_id','group','label','channel','segment_id','label_encoded'
            ], errors='ignore')

            X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
            X_scaled = scalers[ch].transform(X)
            prob = models[ch].predict_proba(X_scaled)[0]
            channel_probs.append(prob)

        avg_prob = np.mean(channel_probs, axis=0)
        ref_model = models[channels[0]]
        pred_encoded = ref_model.classes_[np.argmax(avg_prob)]
        pred_str = label_encoder.inverse_transform([pred_encoded])[0]

        segment_results.append([lbl_str, pred_str, pred_encoded])
        prob_results.append(avg_prob)

    return segment_results, prob_results


test_results, test_probs = ensemble_predict(test_df, models, scalers, channels)


if len(test_results) == 0:
    print("测试集没有有效 segment（8 通道不全）。")
else:
    y_true_str  = [x[0] for x in test_results]
    y_pred_str  = [x[1] for x in test_results]
    y_probs     = np.array(test_probs)

    print(f"Accuracy : {accuracy_score(y_true_str, y_pred_str):.4f}")
    print(f"F1 Score : {f1_score(y_true_str, y_pred_str, average='weighted'):.4f}")
    print(f"Precision: {precision_score(y_true_str, y_pred_str, average='weighted'):.4f}")
    print(f"Recall   : {recall_score(y_true_str, y_pred_str, average='weighted'):.4f}")
    print(f"样本数   : {len(y_true_str)}")

    cm = confusion_matrix(y_true_str, y_pred_str, labels=label_encoder.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.show()


    n_classes = len(label_encoder.classes_)

    if n_classes == 2:
        pos_label = label_encoder.classes_[1]
        pos_enc = label_encoder.transform([pos_label])[0]

        y_true_bin = np.array([1 if y == pos_label else 0 for y in y_true_str])
        y_scores = y_probs[:, pos_enc]

        fpr, tpr, _ = roc_curve(y_true_bin, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
        plt.plot([0,1],[0,1],"--",lw=2)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()

    else:
        from sklearn.preprocessing import label_binarize
        y_true_encoded = [x[2] for x in test_results]
        y_bin = label_binarize(y_true_encoded, classes=range(n_classes))

        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_probs.ravel())
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, lw=2, label=f"micro-AUC={roc_auc:.3f}")
        plt.plot([0,1],[0,1],"--",lw=2)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC (Micro-average)")
        plt.legend()
        plt.tight_layout()
        plt.show()
