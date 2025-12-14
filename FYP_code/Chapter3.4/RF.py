import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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

    drop_cols = ['animal_id', 'group', 'label', 'channel', 'segment_id', 'label_encoded']
    X_train = train_ch.drop(columns=drop_cols, errors='ignore').apply(pd.to_numeric, errors='coerce').fillna(0)
    y_train = train_ch['label_encoded']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    scalers[ch] = scaler

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    models[ch] = model

def ensemble_predict(df, models, scalers, channels, label_encoder):

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
            X = seg_ch.drop(columns=['animal_id', 'group', 'label', 'channel', 'segment_id', 'label_encoded'],
                            errors='ignore').apply(pd.to_numeric, errors='coerce').fillna(0)
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


test_results, test_probs = ensemble_predict(test_df, models, scalers, channels, label_encoder)

if len(test_results) > 0:

    y_true_str = [x[0] for x in test_results]
    y_pred_str = [x[1] for x in test_results]
    y_true_encoded = [x[2] for x in test_results]
    y_probs_test = np.array(test_probs)

    accuracy_test  = accuracy_score(y_true_str, y_pred_str)
    f1_test        = f1_score(y_true_str, y_pred_str, average="weighted")
    precision_test = precision_score(y_true_str, y_pred_str, average="weighted")
    recall_test    = recall_score(y_true_str, y_pred_str, average="weighted")

    print("\n===== Test Set Performance (Random Forest Ensemble) =====")
    print(f"Accuracy : {accuracy_test:.4f}")
    print(f"F1 Score : {f1_test:.4f}")
    print(f"Precision: {precision_test:.4f}")
    print(f"Recall   : {recall_test:.4f}")
    print(f"Samples  : {len(y_true_str)}")

    cm = confusion_matrix(y_true_str, y_pred_str, labels=label_encoder.classes_)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Random Forest Ensemble)")
    plt.tight_layout()
    plt.show()
    # ROC

    n_classes = len(label_encoder.classes_)
    plt.figure(figsize=(8, 6))
    if n_classes == 2:

        pos_label = label_encoder.classes_[1]
        pos_encoded = label_encoder.transform([pos_label])[0]

        y_true_bin = np.array([1 if y == pos_label else 0 for y in y_true_str])
        y_scores = y_probs_test[:, pos_encoded]

        fpr, tpr, _ = roc_curve(y_true_bin, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", lw=2)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Binary)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    else:
        from sklearn.preprocessing import label_binarize

        y_bin = label_binarize(np.array(y_true_encoded), classes=range(n_classes))
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_probs_test.ravel())
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f"Micro-Average AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", lw=2)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Micro-average)")
        plt.legend()
        plt.tight_layout()
        plt.show()
