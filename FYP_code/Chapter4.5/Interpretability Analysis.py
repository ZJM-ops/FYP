#energy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# === 1. Read data ===
features_path = r"/Users/jannie/Desktop/features.csv"
features_df = pd.read_csv(features_path)

energy_cols = sorted([col for col in features_df.columns if 'band' in col and 'energy' in col])

# === 2. Mapping band ===
band_names = [
    "Gamma\n(1-63Hz)",
    "Gamma\n(63-126Hz)", 
    "High Gamma\n(126-188Hz)",
    "Fast Ripples\n(188-250Hz)",
    "Fast Ripples\n(250-313Hz)",
    "Fast Ripples\n(313-375Hz)",
    "Fast Ripples\n(375-438Hz)",
    "Fast Ripples\n(438-500Hz)"
]

# === 3. statistical result ===
baseline_df = features_df[features_df['label'] == 'Baseline']
ictal_df = features_df[features_df['label'] == 'Ictal']

print("="*70)
print("Statistical result (Baseline vs Ictal)")
print("="*70)
print(f"{'band':<25} {'Baseline Mean':<15} {'Ictal Mean':<15} {'变化%':<10} {'p-value':<12} {'显著?'}")
print("-" * 90)

results = []
for i, energy_col in enumerate(energy_cols):
   
    baseline_total = baseline_df[energy_cols].sum(axis=1).mean()
    ictal_total = ictal_df[energy_cols].sum(axis=1).mean()
    
    baseline_rel = (baseline_df[energy_col].mean() / baseline_total * 100) if baseline_total > 0 else 0
    ictal_rel = (ictal_df[energy_col].mean() / ictal_total * 100) if ictal_total > 0 else 0
    

    change_pct = ((ictal_rel - baseline_rel) / baseline_rel * 100) if baseline_rel > 0 else 0
    
    # Mann-Whitney U test
    try:
        _, p_value = stats.mannwhitneyu(
            baseline_df[energy_col].values, 
            ictal_df[energy_col].values, 
            alternative='two-sided'
        )
    except:
        p_value = 1.0
    
    significant = "Yes" if p_value < 0.001 else "No"
    
    print(f"{band_names[i]:<25} {baseline_rel:>8.2f}%       {ictal_rel:>8.2f}%       {change_pct:>7.1f}%       {p_value:>10.4f}    {significant}")
    
    results.append({
        'band': band_names[i],
        'baseline': baseline_rel,
        'ictal': ictal_rel,
        'change': change_pct,
        'p_value': p_value
    })

# === 4. Visualization===
morandi_colors = {
    'gamma': '#88B0AB',     
    'high_gamma': '#7AA3A8', 
    'fr_main': '#A89BB9',   
    'fr_variants': ['#B8A8C8', '#C8B5D7', '#D8C2E6', '#E8CFF5'],  
    'increase': '#D4A5A5',   
    'decrease': '#9BB7C4',   
    'significance': '#E8B4B4' 
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Figure1
ax1 = axes[0]
changes = [r['change'] for r in results]

morandi_bar_colors = []
for i, band in enumerate(band_names):
    if 'Gamma' in band:
        if 'High' in band:
            morandi_bar_colors.append(morandi_colors['high_gamma'])
        else:
            morandi_bar_colors.append(morandi_colors['gamma'])
    elif 'Fast Ripples' in band:
        
        fr_index = i - 3 
        if fr_index < len(morandi_colors['fr_variants']):
            morandi_bar_colors.append(morandi_colors['fr_variants'][fr_index])
        else:
            morandi_bar_colors.append(morandi_colors['fr_main'])

bars = ax1.bar(band_names, changes, 
               color=morandi_bar_colors, 
               alpha=0.85, 
               edgecolor='#4A4A4A', 
               linewidth=1.2)
ax1.set_xlabel('Frequency Bands', fontsize=12, fontweight='medium')
ax1.set_ylabel('Energy Change (%)', fontsize=12, fontweight='medium')
ax1.set_title('(A) Energy Change: Ictal vs Baseline', 
              fontsize=14, fontweight='bold', color='#2C3E50')
ax1.set_xticklabels(band_names, rotation=45, ha='right', fontsize=10)
ax1.grid(True, alpha=0.2, axis='y', linestyle='--', color='#999999')
ax1.axhline(y=0, color='#666666', linestyle='-', linewidth=1.2, alpha=0.7)


for i, (bar, change, r) in enumerate(zip(bars, changes, results)):
    height = bar.get_height()
    
   
    if height >= 0:
        text_y = height + 3
        va_pos = 'bottom'
        text_color = '#8B3A62'  
    else:
        text_y = height - 4
        va_pos = 'top'
        text_color = '#2C5F7D'  
    
    
    ax1.text(bar.get_x() + bar.get_width()/2., 
            text_y,
            f'{change:.1f}%', 
            ha='center', 
            va=va_pos, 
            fontsize=10, 
            fontweight='bold',
            color=text_color,
            bbox=dict(boxstyle='round,pad=0.2', 
                     facecolor='white', 
                     alpha=0.85,
                     edgecolor='none',
                     linewidth=0))
    
    
    if r['p_value'] < 0.001:
        sig_y = max(height, 0) + (12 if height >= 0 else -12)
        ax1.text(bar.get_x() + bar.get_width()/2., 
                sig_y,
                '***', 
                ha='center', 
                va='bottom' if height >= 0 else 'top', 
                fontsize=12, 
                color='#E74C3C',
                fontweight='bold')

# Figure 2
ax2 = axes[1]
increase_factors = [(r['ictal']/r['baseline']) if r['baseline'] > 0 else 0 for r in results]

bars_factor = ax2.bar(band_names, increase_factors, 
                      color=morandi_bar_colors, 
                      alpha=0.85, 
                      edgecolor='#4A4A4A', 
                      linewidth=1.2)
ax2.set_xlabel('Frequency Bands', fontsize=12, fontweight='medium')
ax2.set_ylabel('Increase Factor (Ictal/Baseline)', fontsize=12, fontweight='medium')
ax2.set_title('(B) Increase Factor in Ictal State', 
              fontsize=14, fontweight='bold', color='#2C3E50')
ax2.set_xticklabels(band_names, rotation=45, ha='right', fontsize=10)
ax2.grid(True, alpha=0.2, axis='y', linestyle='--', color='#999999')
ax2.axhline(y=1, color='#666666', linestyle='--', linewidth=1.8, alpha=0.8)


for bar, factor in zip(bars_factor, increase_factors):
    height = bar.get_height()
    
   
    if factor > 1:
        text_color = '#8B3A62'  
    else:
        text_color = '#2C5F7D'  
    
    ax2.text(bar.get_x() + bar.get_width()/2., 
            height + 0.08,
            f'{factor:.1f}x', 
            ha='center', 
            va='bottom', 
            fontsize=10, 
            fontweight='bold',
            color=text_color,
            bbox=dict(boxstyle='round,pad=0.2', 
                     facecolor='white', 
                     alpha=0.85,
                     edgecolor='none',
                     linewidth=0))

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=morandi_colors['gamma'], alpha=0.85, edgecolor='#4A4A4A', label='Gamma Band'),
    Patch(facecolor=morandi_colors['fr_main'], alpha=0.85, edgecolor='#4A4A4A', label='Fast Ripples'),
    Patch(facecolor='white', edgecolor='none', label='***: p < 0.001')
]


ax1.legend(handles=legend_elements, 
           loc='upper right',
           fontsize=9,
           framealpha=0.9,
           edgecolor='#DDDDDD')

plt.suptitle('Fast Ripples Energy Increase During Epileptic Seizures', 
             fontsize=16, fontweight='bold', color='#2C3E50', y=1.05)
plt.tight_layout()
plt.show()


#entropy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# === 1. Read data ===
features_path = r"/Users/jannie/Desktop/features.csv"
features_df = pd.read_csv(features_path)

approx_entropy_cols = sorted([col for col in features_df.columns if 'approx_entropy' in col])

# === 2. Frequency band===
band_names = [
    "Gamma\n(1-63Hz)",
    "Gamma\n(63-126Hz)", 
    "High Gamma\n(126-188Hz)",
    "Fast Ripples\n(188-250Hz)",
    "Fast Ripples\n(250-313Hz)",
    "Fast Ripples\n(313-375Hz)",
    "Fast Ripples\n(375-438Hz)",
    "Fast Ripples\n(438-500Hz)"
]

# === 3. Caculate ===
baseline_df = features_df[features_df['label'] == 'Baseline']
ictal_df = features_df[features_df['label'] == 'Ictal']

print("="*70)
print("📊 近似熵变化：高频段下降最多（信号变得更有序）")
print("="*70)
print(f"{'频段':<15} {'Baseline':<10} {'Ictal':<10} {'变化':<10} {'下降%':<10} {'p-value':<12}")
print("-" * 70)

results = []
for i, entropy_col in enumerate(approx_entropy_cols):
    baseline_mean = baseline_df[entropy_col].mean()
    ictal_mean = ictal_df[entropy_col].mean()
    change = ictal_mean - baseline_mean
    change_pct = (change / baseline_mean * 100) if baseline_mean != 0 else 0
    
    try:
        _, p_value = stats.mannwhitneyu(baseline_df[entropy_col].values, 
                                       ictal_df[entropy_col].values, 
                                       alternative='two-sided')
    except:
        p_value = 1.0
    
    p_display = f"{p_value:.4f}"
    if p_value < 0.001:
        p_display = "<0.001***"
    elif p_value < 0.01:
        p_display = f"{p_value:.4f}**"
    elif p_value < 0.05:
        p_display = f"{p_value:.4f}*"
    
    change_symbol = "↓" if change < 0 else "↑"
    
    print(f"{band_names[i]:<15} {baseline_mean:>8.4f}   {ictal_mean:>8.4f}   {change:>+8.4f}   {change_pct:>7.1f}% {change_symbol}   {p_display:<12}")
    
    results.append({
        'band': band_names[i],
        'baseline': baseline_mean,
        'ictal': ictal_mean,
        'change': change,
        'change_pct': change_pct,
        'p_value': p_value
    })

# === 4. Reult ===
print("\n" + "="*70)
print("="*70)


fr_bands = ["188-250Hz", "250-313Hz", "313-375Hz", "375-438Hz", "438-500Hz"]
fr_results = [r for r in results if r['band'] in fr_bands]

if fr_results:
    fr_sorted = sorted(fr_results, key=lambda x: x['change'])
    
    print("Fast Ripples band entropy decline ranking：")
    for i, r in enumerate(fr_sorted[:3], 1):  
        print(f"{i}. {r['band']}: decline{r['change']:+.4f} ({r['change_pct']:.1f}%)")
    
    
    fr_baseline_avg = np.mean([r['baseline'] for r in fr_results])
    fr_ictal_avg = np.mean([r['ictal'] for r in fr_results])
    fr_change_avg = np.mean([r['change'] for r in fr_results])
    
    print(f"\nFast Ripples (188-500Hz) total：")
    print(f"• Baseline average entropy: {fr_baseline_avg:.4f}")
    print(f"• Ictal average entropy: {fr_ictal_avg:.4f}")
    print(f"• mean decline: {fr_change_avg:+.4f} ({(fr_ictal_avg - fr_baseline_avg)/fr_baseline_avg*100:.1f}%)")


low_freq_bands = ["1-63Hz", "63-126Hz", "126-188Hz"]
low_freq_results = [r for r in results if r['band'] in low_freq_bands]

if low_freq_results and fr_results:
    low_change_avg = np.mean([r['change'] for r in low_freq_results])
    fr_change_avg = np.mean([r['change'] for r in fr_results])
    
    print(f"\nComparison of high-frequency bands vs. low-frequency bands：")
    print(f"• low-frequency bands(1-188Hz)average change: {low_change_avg:+.4f}")
    print(f"• high-frequency bands(188-500Hz)average change: {fr_change_avg:+.4f}")
    print(f"• The high-frequency band drops more than the low-frequency band: {abs(fr_change_avg - low_change_avg):.4f}")

# === 5. Visualization===
plt.figure(figsize=(12, 6))

uniform_colors = [
    '#7FA8C9',  
    '#7FA8C9',  
    '#7FA8C9',  
    '#D8A7B1', 
    '#D8A7B1',  
    '#D8A7B1',  
    '#D8A7B1',  
    '#D8A7B1'   
]


bars = plt.bar(range(len(results)), 
               [r['change'] for r in results], 
               color=uniform_colors,
               alpha=0.85, 
               edgecolor='#4A4A4A',
               linewidth=1.2,
               zorder=2)


plt.xlabel('Frequency Bands', fontsize=13, fontweight='medium', color='#333333')
plt.ylabel('Entropy Change\n(Ictal - Baseline)', fontsize=13, fontweight='medium', color='#333333')
plt.title('Approximate Entropy Changes Across Frequency Bands', 
          fontsize=15, fontweight='bold', color='#2C3E50', pad=20)


plt.xticks(range(len(results)), 
           band_names, 
           rotation=45, 
           ha='right', 
           fontsize=11,
           fontweight='medium',
           color='#444444')


plt.axhline(y=0, color='#666666', linestyle='-', linewidth=1.2, alpha=0.7, zorder=1)
plt.grid(True, alpha=0.2, axis='y', linestyle='--', color='#999999', zorder=0)


for i, r in enumerate(results):
    change = r['change']
    p_val = r['p_value']

    if change >= 0:
        va_pos = 'bottom'
        y_offset = 0.001
        label_color = '#2C5F7D'  
    else:
        va_pos = 'top'
        y_offset = -0.001
        label_color = '#8B3A62'  
    
    
    label_text = f'{change:+.3f}'
    plt.text(i, change + y_offset, 
             label_text, 
             ha='center', 
             va=va_pos, 
             fontsize=10, 
             fontweight='bold',
             color=label_color,
             bbox=dict(boxstyle='round,pad=0.2', 
                      facecolor='white', 
                      alpha=0.7,
                      edgecolor='none',
                      linewidth=0))
    
   
    if p_val < 0.001:
        sig_marker = '***'
        sig_color = '#E74C3C'  
        sig_y_offset = 0.007 if change >= 0 else -0.007
        plt.text(i, change + sig_y_offset, 
                 sig_marker, 
                 ha='center', 
                 va='bottom' if change >= 0 else 'top', 
                 fontsize=11, 
                 color=sig_color,
                 fontweight='bold')
    elif p_val < 0.01:
        sig_marker = '**'
        sig_color = '#E67E22'  
        sig_y_offset = 0.006 if change >= 0 else -0.006
        plt.text(i, change + sig_y_offset, 
                 sig_marker, 
                 ha='center', 
                 va='bottom' if change >= 0 else 'top', 
                 fontsize=11, 
                 color=sig_color,
                 fontweight='bold')
    elif p_val < 0.05:
        sig_marker = '*'
        sig_color = '#F39C12'  
        sig_y_offset = 0.005 if change >= 0 else -0.005
        plt.text(i, change + sig_y_offset, 
                 sig_marker, 
                 ha='center', 
                 va='bottom' if change >= 0 else 'top', 
                 fontsize=11, 
                 color=sig_color,
                 fontweight='bold')

plt.axvspan(3, 7, alpha=0.06, color='#D4A5A5', zorder=0)



legend_elements = [
    plt.Rectangle((0,0), 1, 1, facecolor=uniform_colors[0], alpha=0.85, edgecolor='#4A4A4A', label='Low Frequency (1-188 Hz)'),
    plt.Rectangle((0,0), 1, 1, facecolor=uniform_colors[3], alpha=0.85, edgecolor='#4A4A4A', label='High Frequency (188-500 Hz)'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#E74C3C', markersize=14, label='p < 0.001')
]

plt.legend(handles=legend_elements, 
           loc='upper right', 
           fontsize=9,
           framealpha=0.9,
           edgecolor='#DDDDDD',
           title='Legend',
           title_fontsize=10)


plt.tight_layout()

plt.figtext(0.5, 0.01, 
            'Negative values = decreased entropy (more ordered signals during seizures)', 
            ha='center', 
            fontsize=9, 
            color='#666666',
            style='italic')

plt.show()


