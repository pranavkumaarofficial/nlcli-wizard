"""Generate 1B vs 4B comparison bar chart for Reddit post."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.family'] = 'sans-serif'

categories = ['run', 'build', 'compose', 'exec', 'network', 'volume', 'system', 'ps/images']
one_b_best = [94.7, 90.0, 77.8, 27.3, 100.0, 100.0, 100.0, 90.0]
four_b =     [96.2, 90.0, 100.0, 84.6, 100.0, 100.0, 100.0, 87.5]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - width/2, one_b_best, width, label='Gemma 3 1B (best of 3 runs)',
               color='#ef4444', alpha=0.85, edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, four_b, width, label='Gemma 3 4B (first try)',
               color='#22c55e', alpha=0.85, edgecolor='white', linewidth=0.5)

ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Docker CLI Translation: 1B vs 4B Per-Category Accuracy', fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 115)
ax.legend(fontsize=11, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
            f'{height:.0f}%', ha='center', va='bottom', fontsize=9, color='#666')
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
            f'{height:.0f}%', ha='center', va='bottom', fontsize=9, color='#666')

# Add overall accuracy annotation
ax.annotate('Overall: 1B = 76%  |  4B = 94%',
            xy=(0.98, 0.98), xycoords='axes fraction',
            ha='right', va='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='#ccc'))

plt.tight_layout()
plt.savefig('docs/1b_vs_4b_comparison.png', dpi=150, bbox_inches='tight')
print("Saved to docs/1b_vs_4b_comparison.png")
