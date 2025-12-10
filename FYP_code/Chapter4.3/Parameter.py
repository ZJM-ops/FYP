import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False

def plot_complexity_refined():
    data = {
        'Model': ['ResNet-18', 'ResNet-50', 'ResNeXt-50', 'ResNeXt-101', 'Inception-ResNet-v2'],
        'Params': [11.18, 23.51, 22.98, 86.75, 54.31],
        'Accuracy': [93.96, 96.31, 96.80, 95.71, 96.96]
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.scatterplot(
        data=df,
        x='Params',
        y='Accuracy',
        s=300,
        color='#4c72b0',
        alpha=0.9,
        edgecolor='white',
        linewidth=2,
        ax=ax
    )

    offsets = {
        'ResNet-18': (2, -0.1),
        'ResNet-50': (2, -0.3),
        'ResNeXt-50': (2, 0.1),
        'Inception-ResNet-v2': (2, 0),
        'ResNeXt-101': (-3, 0)
    }

    ha_dict = {'ResNeXt-101': 'right'}

    for _, row in df.iterrows():
        name = row['Model']
        x = row['Params']
        y = row['Accuracy']
        dx, dy = offsets.get(name, (2, 0))
        ha = ha_dict.get(name, 'left')

        ax.text(
            x + dx, y + dy,
            name,
            fontsize=13,
            color='#333333',
            fontweight='medium',
            horizontalalignment=ha,
            verticalalignment='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7)
        )

    ax.set_title('Accuracy vs. Model Complexity', fontsize=20, pad=20, weight='bold')
    ax.set_xlabel('Parameters (Millions)', fontsize=16, labelpad=10)
    ax.set_ylabel('Test Accuracy (%)', fontsize=16, labelpad=10)

    sns.despine(top=True, right=True)

    ax.set_xlim(-5, 100)
    ax.set_ylim(93, 98)

    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('Figure_Complexity_Refined.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_complexity_refined()
