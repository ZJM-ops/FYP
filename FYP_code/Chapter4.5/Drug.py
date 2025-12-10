import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import mannwhitneyu

# 配置路径
CSV_PATH = r"D:\test_predictions_probabilities_ensemble.xlsx"
OUTPUT_PLOT_PATH = 'drug_effect_ensemble_boxplot.png'

# 绘图风格
sns.set_theme(style="ticks", font="Arial", font_scale=1.2)

# 读取数据
df = pd.read_excel(CSV_PATH)

# 仅分析真实标签为 Ictal 的样本
ictal_df = df[df['True_Label'] == 1].copy()

# 统计检验（无打印）
def calculate_p_value(group1_name, group2_name):
    g1 = ictal_df[ictal_df['Group'] == group1_name]['Probability_Ictal']
    g2 = ictal_df[ictal_df['Group'] == group2_name]['Probability_Ictal']
    if len(g1) > 0 and len(g2) > 0:
        stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
        return p
    return None

# 自动执行 p-value 计算（不打印）
p_pilo_vpa = calculate_p_value('PILO', 'VPA')
p_pilo_mir = calculate_p_value('PILO', 'miR')

# 绘图
plt.figure(figsize=(10, 7))

target_order = ['PILO', 'VPA', 'miR', 'Sponges', 'Scramble']
plot_order = [g for g in target_order if g in ictal_df['Group'].unique()]

# 箱线图
sns.boxplot(
    x='Group',
    y='Probability_Ictal',
    data=ictal_df,
    order=plot_order,
    palette="Set2",
    width=0.5,
    linewidth=1.5,
    showfliers=False
)

# 数据点散点图
sns.stripplot(
    x='Group',
    y='Probability_Ictal',
    data=ictal_df,
    order=plot_order,
    color=".25",
    alpha=0.4,
    size=3,
    jitter=True
)

plt.ylabel('Predicted Probability of Seizure', fontsize=14)
plt.xlabel('Experimental Group', fontsize=14)

plt.ylim(0.45, 1.05)

# 添加 median 数值（不打印）
medians = ictal_df.groupby('Group')['Probability_Ictal'].median()
for i, group in enumerate(plot_order):
    if group in medians:
        median_val = medians[group]
        plt.text(
            i, median_val + 0.02, f'{median_val:.2f}',
            ha='center',
            fontsize=11,
            color='black',
            weight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        )

sns.despine()

plt.tight_layout()
plt.savefig(OUTPUT_PLOT_PATH, dpi=300, bbox_inches='tight')
plt.show()
