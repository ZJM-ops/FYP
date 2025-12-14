import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re

warnings.filterwarnings('ignore')

train_df = pd.read_csv("/Users/jannie/Desktop/train_data.csv")
val_df   = pd.read_csv("/Users/jannie/Desktop/val_data.csv")
test_df  = pd.read_csv("/Users/jannie/Desktop/test_data.csv")

def get_band_index(col_name):
    m = re.match(r"band(\d+)_", col_name)
    return int(m.group(1)) if m else None


def filter_to_band4_7(df):
    band_cols = [c for c in df.columns if c.startswith("band")]
    keep_cols = [c for c in band_cols if get_band_index(c) is not None and get_band_index(c) >= 4]
    drop_cols = [c for c in band_cols if c not in keep_cols]
    return df.drop(columns=drop_cols)


train_df = filter_to_band4_7(train_df)
val_df   = filter_to_band4_7(val_df)
test_df  = filter_to_band4_7(test_df)

label_encoder = LabelEncoder()
train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])
test_df['label_encoded']  = label_encoder.transform(test_df['label'])

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
            feature_cols = [c for c in seg_ch.columns if c.startswith("band")]
            X = seg_ch[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

            X_scaled = scalers[ch].transform(X)
            prob = models[ch].predict_proba(X_scaled)[0]
            channel_probs.append(prob)

        avg_prob = np.mean(channel_probs, axis=0)
        pred_encoded = np.argmax(avg_prob)
        pred_str = label_encoder.inverse_transform([pred_encoded])[0]

        segment_results.append([lbl_str, pred_str])
        prob_results.append(avg_prob)

    return segment_results, np.array(prob_results)

channels = sorted(train_df['channel'].unique())

models_config = {
    'SVM': {
        'class': SVC,
        'params': {'kernel': 'rbf', 'C': 0.01, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    },
    'Logistic Regression': {
        'class': LogisticRegression,
        'params': {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42}
    },
    'Random Forest': {
        'class': RandomForestClassifier,
        'params': {
            'n_estimators': 100, 'max_depth': None,
            'min_samples_split': 2, 'min_samples_leaf': 1,
            'max_features': 'sqrt', 'random_state': 42, 'n_jobs': -1
        }
    }
}

all_results = {}

for model_name, config in models_config.items():

    models, scalers = {}, {}

    for ch in channels:
        train_ch = train_df[train_df['channel'] == ch]

        feature_cols = [c for c in train_ch.columns if c.startswith("band")]
        X_train = train_ch[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        y_train = train_ch['label_encoded']

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        scalers[ch] = scaler

        model = config['class'](**config['params'])
        model.fit(X_train_scaled, y_train)
        models[ch] = model

    test_results, test_probs = ensemble_predict(test_df, models, scalers, channels, label_encoder)

    y_true = [t[0] for t in test_results]
    y_pred = [t[1] for t in test_results]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=label_encoder.classes_[1])
    rec = recall_score(y_true, y_pred, pos_label=label_encoder.classes_[1])
    f1v = f1_score(y_true, y_pred, pos_label=label_encoder.classes_[1])
    cm = confusion_matrix(y_true, y_pred, labels=label_encoder.classes_)

    all_results[model_name] = {
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": test_probs,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1v,
        "cm": cm
    }


print("\n\n===== Model Performance Comparison =====")
print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
print("-" * 75)

for name, r in all_results.items():
    print(f"{name:<25} {r['accuracy']:.4f}     {r['precision']:.4f}     {r['recall']:.4f}     {r['f1']:.4f}")


# Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
class_names = label_encoder.classes_

for idx, (model_name, r) in enumerate(all_results.items()):
    ax = axes[idx]
    sns.heatmap(r['cm'], annot=True, fmt='d', cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(model_name)

plt.tight_layout()
plt.show()


# ROC
plt.figure(figsize=(8, 6))
pos_label = label_encoder.classes_[1]
pos_idx = label_encoder.transform([pos_label])[0]

plt.plot([0, 1], [0, 1], '--', color='gray')

colors = {"SVM": "red", "Logistic Regression": "blue", "Random Forest": "green"}
auc_values = {}

for model_name, r in all_results.items():
    y_true_bin = np.array([1 if y == pos_label else 0 for y in r['y_true']])
    y_scores = r['probs'][:, pos_idx]

    fpr, tpr, _ = roc_curve(y_true_bin, y_scores)
    roc_auc = auc(fpr, tpr)
    auc_values[model_name] = roc_auc

    plt.plot(fpr, tpr, lw=2, color=colors[model_name],
             label=f"{model_name} (AUC={roc_auc:.3f})")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


print("\n===== AUC Comparison =====")
for m in auc_values:
    print(f"{m:<25} AUC = {auc_values[m]:.4f}")
