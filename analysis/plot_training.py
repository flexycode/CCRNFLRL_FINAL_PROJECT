"""
Plot training metrics saved during DQN training.

Run after training completes:
    python3 train.py --n_words 200 --episodes 2000
    python3 plot_training.py

Creates multiple plots showing:
  - Win rate progression (with smoothing)
  - Average guesses progression
  - Learning curves with confidence bands
  - Epoch-by-epoch comparison
"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from pathlib import Path


def smooth_curve(data, window=20):
    """Smooth a curve using uniform filter."""
    if len(data) < window:
        return data
    return uniform_filter1d(data, size=window, mode='nearest')


def load_metrics(path=str(Path(__file__).parent.parent / "data" / "training_metrics.json")):
    """Load training metrics from JSON file."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Could not find {path}. Did you run train.py yet?")
    with open(path) as f:
        return json.load(f)


def plot_learning_curves(metrics):
    """Plot win rate and avg guesses on a 2-panel layout."""
    episodes = metrics["episodes"]
    win_rates = metrics["win_rates"]
    avg_guesses = metrics["avg_guesses"]
    n_words = metrics["n_words"]
    
    if not episodes:
        print("Warning: No training data found. Run train.py first.")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
    fig.suptitle(f"DQN Training Progress ({n_words} words)", 
                 fontsize=14, fontweight="bold", y=0.995)
    
    # Win rate plot
    win_rates_smooth = smooth_curve(win_rates, window=max(3, len(episodes)//20))
    ax1.plot(episodes, win_rates, 'o-', alpha=0.5, linewidth=1.5, 
             color='#6aaa64', markersize=4, label='Raw')
    ax1.plot(episodes, win_rates_smooth, '-', linewidth=2.5, 
             color='#2e7d32', label='Smoothed')
    ax1.fill_between(episodes, win_rates, alpha=0.15, color='#6aaa64')
    ax1.set_ylabel('Win Rate', fontsize=11, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_title('Win Rate Progression', fontsize=12, fontweight='bold', pad=8)
    
    # Final win rate annotation
    final_wr = metrics["final_win_rate"]
    ax1.text(0.02, 0.95, f'Final: {final_wr:.1%}', 
             transform=ax1.transAxes, fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#6aaa64', alpha=0.3),
             verticalalignment='top')
    
    # Avg guesses plot
    avg_guesses_smooth = smooth_curve(avg_guesses, window=max(3, len(episodes)//20))
    ax2.plot(episodes, avg_guesses, 'o-', alpha=0.5, linewidth=1.5, 
             color='#c9b458', markersize=4, label='Raw')
    ax2.plot(episodes, avg_guesses_smooth, '-', linewidth=2.5, 
             color='#a89c00', label='Smoothed')
    ax2.fill_between(episodes, avg_guesses_smooth, alpha=0.15, color='#c9b458')
    ax2.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Avg Guesses (wins)', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 6.5)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_title('Efficiency Progression (Avg Guesses to Solve)', 
                  fontsize=12, fontweight='bold', pad=8)
    
    # Final avg guesses annotation
    final_avg = metrics["final_avg_guesses"]
    ax2.text(0.02, 0.95, f'Final: {final_avg:.2f}', 
             transform=ax2.transAxes, fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#c9b458', alpha=0.3),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: training_curves.png")
    plt.show()


def plot_combined_dashboard(metrics):
    """Plot a 2x2 dashboard with learning curves and summary stats."""
    episodes = metrics["episodes"]
    win_rates = metrics["win_rates"]
    avg_guesses = metrics["avg_guesses"]
    n_words = metrics["n_words"]
    final_wr = metrics["final_win_rate"]
    final_avg = metrics["final_avg_guesses"]
    
    if not episodes:
        return
    
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # Title
    fig.suptitle(f"DQN Training Dashboard ({n_words} words, {len(episodes)} checkpoints)",
                 fontsize=15, fontweight='bold', y=0.98)
    
    # Plot 1: Win rate with confidence band
    ax1 = fig.add_subplot(gs[0, 0])
    win_rates_smooth = smooth_curve(win_rates, window=max(3, len(episodes)//20))
    ax1.plot(episodes, win_rates_smooth, linewidth=2.5, color='#2e7d32', zorder=3)
    ax1.scatter(episodes, win_rates, alpha=0.3, s=20, color='#6aaa64', zorder=2)
    ax1.fill_between(episodes, win_rates_smooth, alpha=0.15, color='#6aaa64', zorder=1)
    ax1.set_ylabel('Win Rate', fontweight='bold', fontsize=10)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.25, linestyle=':')
    ax1.set_title('Win Rate', fontweight='bold', fontsize=11)
    ax1.axhline(y=final_wr, color='#2e7d32', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Plot 2: Avg guesses
    ax2 = fig.add_subplot(gs[0, 1])
    avg_guesses_smooth = smooth_curve(avg_guesses, window=max(3, len(episodes)//20))
    ax2.plot(episodes, avg_guesses_smooth, linewidth=2.5, color='#a89c00', zorder=3)
    ax2.scatter(episodes, avg_guesses, alpha=0.3, s=20, color='#c9b458', zorder=2)
    ax2.fill_between(episodes, avg_guesses_smooth, alpha=0.15, color='#c9b458', zorder=1)
    ax2.set_ylabel('Avg Guesses', fontweight='bold', fontsize=10)
    ax2.set_ylim(0, 6.5)
    ax2.grid(alpha=0.25, linestyle=':')
    ax2.set_title('Efficiency', fontweight='bold', fontsize=11)
    ax2.axhline(y=final_avg, color='#a89c00', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Plot 3: Learning improvement (inverse of avg guesses)
    ax3 = fig.add_subplot(gs[1, 0])
    improvement = [6.5 - g for g in avg_guesses]  # lower avg guesses = more improvement
    improvement_smooth = smooth_curve(improvement, window=max(3, len(episodes)//20))
    ax3.plot(episodes, improvement_smooth, linewidth=2.5, color='#1976d2', zorder=3)
    ax3.scatter(episodes, improvement, alpha=0.3, s=20, color='#42a5f5', zorder=2)
    ax3.fill_between(episodes, improvement_smooth, alpha=0.15, color='#42a5f5', zorder=1)
    ax3.set_xlabel('Episode', fontweight='bold', fontsize=10)
    ax3.set_ylabel('Improvement Score', fontweight='bold', fontsize=10)
    ax3.grid(alpha=0.25, linestyle=':')
    ax3.set_title('Learning Progress (Higher = Better)', fontweight='bold', fontsize=11)
    
    # Plot 4: Summary stats box
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    summary_text = f"""
TRAINING SUMMARY

Final Performance:
  Win Rate: {final_wr:.1%}
  Avg Guesses: {final_avg:.2f}

Training Config:
  Word Pool: {n_words}
  Checkpoints: {len(episodes)}
  
Improvement:
  Initial Win Rate: {win_rates[0]:.1%}
  Final Win Rate: {final_wr:.1%}
  Gain: {(final_wr - win_rates[0]):.1%}
  
  Initial Avg Guesses: {avg_guesses[0]:.2f}
  Final Avg Guesses: {final_avg:.2f}
  Improvement: {(avg_guesses[0] - final_avg):.2f} guesses
"""
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8, pad=1))
    
    plt.savefig('training_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: training_dashboard.png")
    plt.show()


def plot_convergence_analysis(metrics):
    """Analyze convergence: show learning rate and stability."""
    episodes = metrics["episodes"]
    win_rates = metrics["win_rates"]
    avg_guesses = metrics["avg_guesses"]
    
    if len(episodes) < 2:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Convergence Analysis', fontsize=13, fontweight='bold')
    
    # Learning rate (slope of win rate curve)
    window = max(5, len(episodes) // 10)
    win_rate_slope = np.gradient(win_rates, episodes)
    win_rate_slope_smooth = uniform_filter1d(win_rate_slope, size=window, mode='nearest')
    
    ax1 = axes[0]
    ax1.plot(episodes, win_rate_slope_smooth * 1000, linewidth=2, color='#2e7d32', marker='o', markersize=4)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax1.set_xlabel('Episode', fontweight='bold')
    ax1.set_ylabel('Win Rate Change (×1000)', fontweight='bold')
    ax1.set_title('Learning Rate (Win Rate Derivative)', fontweight='bold')
    ax1.grid(alpha=0.3, linestyle=':')
    
    # Stability (rolling standard deviation)
    ax2 = axes[1]
    rolling_std = np.array([np.std(avg_guesses[max(0, i-window):i+1]) 
                           for i in range(len(avg_guesses))])
    ax2.plot(episodes, rolling_std, linewidth=2, color='#c9b458', marker='s', markersize=4)
    ax2.set_xlabel('Episode', fontweight='bold')
    ax2.set_ylabel('Avg Guesses Std Dev', fontweight='bold')
    ax2.set_title('Training Stability (Lower = More Stable)', fontweight='bold')
    ax2.grid(alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: convergence_analysis.png")
    plt.show()


def plot_phase_analysis(metrics):
    """Divide training into phases and analyze each."""
    episodes = metrics["episodes"]
    win_rates = metrics["win_rates"]
    avg_guesses = metrics["avg_guesses"]
    n_words = metrics["n_words"]
    
    if len(episodes) < 3:
        return
    
    # Divide into 3 phases
    phase_size = len(episodes) // 3
    phases = [
        ("Early", slice(0, phase_size)),
        ("Mid", slice(phase_size, 2 * phase_size)),
        ("Late", slice(2 * phase_size, None))
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Training Phases Analysis ({n_words} words)', 
                 fontsize=13, fontweight='bold')
    
    phase_names = []
    phase_win_rates = []
    phase_avg_guesses = []
    
    colors = ['#e8f5e9', '#a5d6a7', '#2e7d32']
    
    for (name, idx_slice), color in zip(phases, colors):
        phase_wr = np.mean(win_rates[idx_slice])
        phase_avg = np.mean(avg_guesses[idx_slice])
        phase_names.append(name)
        phase_win_rates.append(phase_wr)
        phase_avg_guesses.append(phase_avg)
    
    x_pos = np.arange(len(phase_names))
    width = 0.35
    
    # Win rates by phase
    bars1 = ax1.bar(x_pos - width/2, phase_win_rates, width, label='Avg Win Rate',
                    color='#6aaa64', alpha=0.8, edgecolor='#2e7d32', linewidth=2)
    ax1.set_ylabel('Win Rate', fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(phase_names)
    ax1.set_title('Win Rate by Phase', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle=':')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Avg guesses by phase
    bars2 = ax2.bar(x_pos - width/2, phase_avg_guesses, width, label='Avg Guesses',
                    color='#c9b458', alpha=0.8, edgecolor='#a89c00', linewidth=2)
    ax2.set_ylabel('Avg Guesses', fontweight='bold')
    ax2.set_ylim(0, 6)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(phase_names)
    ax2.set_title('Efficiency by Phase', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle=':')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('phase_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: phase_analysis.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot DQN training metrics")
    parser.add_argument("--metrics", type=str, default=str(Path(__file__).parent.parent / "data" / "training_metrics.json"),
                       help="Path to training_metrics.json")
    parser.add_argument("--all", action="store_true", 
                       help="Generate all plots (default: main dashboard only)")
    args = parser.parse_args()
    
    print("Loading training metrics...")
    metrics = load_metrics(args.metrics)
    
    print(f"\nTraining Summary:")
    print(f"  Word pool size: {metrics['n_words']}")
    print(f"  Total episodes: {metrics['total_episodes']}")
    print(f"  Checkpoints recorded: {len(metrics['episodes'])}")
    print(f"  Final win rate: {metrics['final_win_rate']:.1%}")
    print(f"  Final avg guesses: {metrics['final_avg_guesses']:.2f}")
    print("\nGenerating plots...")
    
    # Main plots
    plot_learning_curves(metrics)
    plot_combined_dashboard(metrics)
    
    # Additional analysis if requested
    if args.all:
        plot_convergence_analysis(metrics)
        plot_phase_analysis(metrics)
    
    print("\n✓ All plots generated successfully!")
    if not args.all:
        print("  Tip: Run with --all flag for additional convergence and phase analysis plots")


if __name__ == "__main__":
    main()