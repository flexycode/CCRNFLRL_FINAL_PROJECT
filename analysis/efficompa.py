"""plot_results.py - Create comparison plots"""
import matplotlib.pyplot as plt
import numpy as np

# Your results: 2,309-word model vs 8,636-word model
models = ['2,309 words\n(Original)', '8,636 words\n(ENABLE1)']
win_rates = [0.95, 0.72]
avg_guesses = [4.19, 4.94]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Win rate comparison
colors = ['#6aaa64', '#c9b458']
ax1.bar(models, win_rates, color=colors, alpha=0.8, width=0.6)
ax1.set_ylabel('Win Rate', fontsize=12)
ax1.set_ylim(0, 1.0)
ax1.set_title('DQN Agent: Win Rate Comparison', fontsize=13)
ax1.grid(axis='y', alpha=0.3)
for i, v in enumerate(win_rates):
    ax1.text(i, v + 0.02, f'{v:.0%}', ha='center', fontsize=11, fontweight='bold')

# Avg guesses comparison
ax2.bar(models, avg_guesses, color=colors, alpha=0.8, width=0.6)
ax2.set_ylabel('Avg Guesses (wins)', fontsize=12)
ax2.set_ylim(0, 6)
ax2.set_title('DQN Agent: Efficiency Comparison', fontsize=13)
ax2.grid(axis='y', alpha=0.3)
for i, v in enumerate(avg_guesses):
    ax2.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('dqn_comparison.png', dpi=300, bbox_inches='tight')
print("Saved to dqn_comparison.png")
plt.show()