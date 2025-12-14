import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


CSV_FILE = r"D:\download\thesis_all_models_predictions_final.csv"
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12


def plot_all_roc():
    df = pd.read_csv(CSV_FILE)
    y_true = df['y_true']

    plt.figure(figsize=(10, 8))

    for col in df.columns:
        if col == 'y_true':
            continue

        model_name = col.replace('_prob', '')
        y_score = df[col]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            lw=2,
            alpha=0.85,
            label=f"{model_name} (AUC = {roc_auc:.3f})"
        )

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, alpha=0.5)

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    plt.title('ROC Curve Comparison', fontsize=18)

    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()

    plt.savefig('Figure_All_Models_ROC_Uniform.png', dpi=300)


if __name__ == "__main__":
    plot_all_roc()
