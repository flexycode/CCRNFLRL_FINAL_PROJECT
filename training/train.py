"""
Train the embedding-based DQN agent on WordleEnv (v2 — Revamped Pipeline).

Key improvements over v1:
    - Headless by default (no matplotlib blocking). Use --visual to opt-in.
    - tqdm progress bar with ETA, win rate, epsilon, loss
    - Curriculum learning: start small, progressively expand word pool
    - Checkpoint & resume support
    - Auto-generates result images to assets/ at the end
    - Train every N steps (not every step) for speed
    - Exponential epsilon decay for faster convergence

Usage:
    python training/train.py --n_words 200 --episodes 2000
    python training/train.py --n_words 2309 --episodes 3000 --curriculum
    python training/train.py --visual   # opt-in to live dashboard
"""
import argparse
import json
import time
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from core.wordle_env import WordleEnv, OBS_DIM, WORD_FEAT_DIM
from core.wordle_mdp import load_word_lists, encode_word
from core.dqn_agent import DQNAgent

# Try importing tqdm, fall back to a simple stub
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=None, desc="", **kwargs):
            self.iterable = iterable
            self.total = total
            self.n = 0
            self.desc = desc
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.n += 1
        def set_postfix_str(self, s):
            pass
        def set_postfix(self, **kwargs):
            pass
        def update(self, n=1):
            self.n += n
        def close(self):
            pass


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"


def evaluate(env, agent, n_eval=50):
    """Evaluate the agent's win rate and average guesses."""
    wins, total_guesses = 0, 0
    for i in range(n_eval):
        answer = env.answer_pool[i % len(env.answer_pool)]
        obs = env.reset(answer=answer)
        done = False
        steps = 0
        while not done:
            mask = env.valid_action_mask()
            word_feats, word_indices = env.valid_word_features(mask)
            action_local = agent.act(obs, word_feats, epsilon=0.0)
            action_idx = word_indices[action_local]
            obs, reward, done, info = env.step(action_idx)
            steps += 1
        if info["solved"]:
            wins += 1
            total_guesses += steps
    win_rate = wins / n_eval
    avg_guesses = total_guesses / max(1, wins)
    return win_rate, avg_guesses


def save_training_plots(metrics, save_dir):
    """Generate and save training result plots to the assets directory."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend — no window pops up
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    episodes = metrics["episodes"]
    win_rates = metrics["win_rates"]
    avg_guesses = metrics["avg_guesses"]
    n_words = metrics["n_words"]
    final_wr = metrics["final_win_rate"]
    final_avg = metrics["final_avg_guesses"]
    
    if not episodes:
        print("Warning: No checkpoint data to plot.")
        return
    
    # --- Plot 1: Learning Curves ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
    fig.suptitle(f"DQN v2 Training Progress ({n_words} words, {metrics['total_episodes']} episodes)",
                 fontsize=14, fontweight="bold", y=0.995)
    
    # Smoothing
    from scipy.ndimage import uniform_filter1d
    window = max(3, len(episodes) // 20)
    wr_smooth = uniform_filter1d(win_rates, size=window, mode='nearest')
    ag_smooth = uniform_filter1d(avg_guesses, size=window, mode='nearest')
    
    ax1.plot(episodes, win_rates, 'o-', alpha=0.4, linewidth=1.5,
             color='#6aaa64', markersize=3, label='Raw')
    ax1.plot(episodes, wr_smooth, '-', linewidth=2.5,
             color='#2e7d32', label='Smoothed')
    ax1.fill_between(episodes, win_rates, alpha=0.12, color='#6aaa64')
    ax1.set_ylabel('Win Rate', fontsize=11, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_title('Win Rate Progression', fontsize=12, fontweight='bold', pad=8)
    ax1.text(0.02, 0.95, f'Final: {final_wr:.1%}',
             transform=ax1.transAxes, fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#6aaa64', alpha=0.3),
             verticalalignment='top')
    
    ax2.plot(episodes, avg_guesses, 'o-', alpha=0.4, linewidth=1.5,
             color='#c9b458', markersize=3, label='Raw')
    ax2.plot(episodes, ag_smooth, '-', linewidth=2.5,
             color='#a89c00', label='Smoothed')
    ax2.fill_between(episodes, ag_smooth, alpha=0.12, color='#c9b458')
    ax2.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Avg Guesses (wins)', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 6.5)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_title('Efficiency Progression', fontsize=12, fontweight='bold', pad=8)
    ax2.text(0.02, 0.95, f'Final: {final_avg:.2f}',
             transform=ax2.transAxes, fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#c9b458', alpha=0.3),
             verticalalignment='top')
    
    plt.tight_layout()
    path1 = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path1, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [+] Saved: {path1}")
    
    # --- Plot 2: Training Dashboard ---
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    fig.suptitle(f"DQN v2 Training Dashboard ({n_words} words)",
                 fontsize=15, fontweight='bold', y=0.98)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(episodes, wr_smooth, linewidth=2.5, color='#2e7d32', zorder=3)
    ax1.scatter(episodes, win_rates, alpha=0.25, s=15, color='#6aaa64', zorder=2)
    ax1.fill_between(episodes, wr_smooth, alpha=0.12, color='#6aaa64', zorder=1)
    ax1.set_ylabel('Win Rate', fontweight='bold', fontsize=10)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.25, linestyle=':')
    ax1.set_title('Win Rate', fontweight='bold', fontsize=11)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(episodes, ag_smooth, linewidth=2.5, color='#a89c00', zorder=3)
    ax2.scatter(episodes, avg_guesses, alpha=0.25, s=15, color='#c9b458', zorder=2)
    ax2.fill_between(episodes, ag_smooth, alpha=0.12, color='#c9b458', zorder=1)
    ax2.set_ylabel('Avg Guesses', fontweight='bold', fontsize=10)
    ax2.set_ylim(0, 6.5)
    ax2.grid(alpha=0.25, linestyle=':')
    ax2.set_title('Efficiency', fontweight='bold', fontsize=11)
    
    ax3 = fig.add_subplot(gs[1, 0])
    losses = metrics.get("losses", [])
    if losses:
        loss_smooth = uniform_filter1d(losses, size=max(3, len(losses) // 20), mode='nearest')
        loss_episodes = np.linspace(episodes[0], episodes[-1], len(losses))
        ax3.plot(loss_episodes, loss_smooth, linewidth=2, color='#e53935', zorder=3)
        ax3.set_ylabel('Loss (smoothed)', fontweight='bold', fontsize=10)
        ax3.set_xlabel('Episode', fontweight='bold', fontsize=10)
        ax3.grid(alpha=0.25, linestyle=':')
        ax3.set_title('Training Loss', fontweight='bold', fontsize=11)
    else:
        improvement = [6.5 - g for g in avg_guesses]
        imp_smooth = uniform_filter1d(improvement, size=window, mode='nearest')
        ax3.plot(episodes, imp_smooth, linewidth=2.5, color='#1976d2', zorder=3)
        ax3.scatter(episodes, improvement, alpha=0.25, s=15, color='#42a5f5', zorder=2)
        ax3.set_xlabel('Episode', fontweight='bold', fontsize=10)
        ax3.set_ylabel('Improvement Score', fontweight='bold', fontsize=10)
        ax3.grid(alpha=0.25, linestyle=':')
        ax3.set_title('Learning Progress', fontweight='bold', fontsize=11)
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    elapsed_str = metrics.get("training_time_str", "N/A")
    summary_text = f"""
TRAINING SUMMARY (v2)

Final Performance:
  Win Rate: {final_wr:.1%}
  Avg Guesses: {final_avg:.2f}

Training Config:
  Word Pool: {n_words}
  Episodes: {metrics['total_episodes']}
  Architecture: EmbeddingDQN
  
Training Time: {elapsed_str}

Improvement:
  Initial Win Rate: {win_rates[0]:.1%}
  Final Win Rate: {final_wr:.1%}
  Gain: {(final_wr - win_rates[0]):+.1%}
"""
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8, pad=1))
    
    path2 = os.path.join(save_dir, "training_dashboard.png")
    plt.savefig(path2, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [+] Saved: {path2}")


def main():
    parser = argparse.ArgumentParser(description="Train the Embedding DQN Wordle Agent (v2)")
    parser.add_argument("--n_words", type=int, default=200,
                         help="Restrict action/answer space to this many words")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_every", type=int, default=4,
                         help="Train every N environment steps (not every step)")
    parser.add_argument("--target_update_every", type=int, default=10,
                         help="Hard update target network every N episodes")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--visual", action="store_true",
                         help="Enable live dashboard (opt-in, slower)")
    parser.add_argument("--board_speed", type=float, default=0.6,
                         help="Animation speed multiplier for the live demo board")
    parser.add_argument("--curriculum", action="store_true",
                         help="Use curriculum learning (start small, grow)")
    parser.add_argument("--resume", type=str, default=None,
                         help="Resume training from a checkpoint file")
    parser.add_argument("--device", type=str, default="auto",
                         help="Device: 'cpu', 'cuda', or 'auto'")
    parser.add_argument("--no_prioritized", action="store_true",
                         help="Disable prioritized replay (use uniform)")
    parser.add_argument("--save_name", type=str, default="dqn_wordle_v2.pt",
                         help="Name for the saved model file")
    args = parser.parse_args()
    
    # Device selection
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Load word lists
    answers, _ = load_word_lists()
    
    # Curriculum learning schedule
    if args.curriculum and args.n_words > 200:
        curriculum_stages = []
        stage_size = max(100, args.n_words // 4)
        current = min(200, args.n_words)
        while current < args.n_words:
            curriculum_stages.append(current)
            current = min(current * 2, args.n_words)
        curriculum_stages.append(args.n_words)
        print(f"Curriculum stages: {curriculum_stages}")
    else:
        curriculum_stages = [args.n_words]
    
    # Initialize with the first curriculum stage
    word_pool = answers[:curriculum_stages[0]]
    print(f"Starting training on {len(word_pool)} words (action space = {len(word_pool)})")
    
    env = WordleEnv(answer_pool=word_pool, action_words=word_pool, seed=0)
    agent = DQNAgent(obs_dim=OBS_DIM, word_dim=WORD_FEAT_DIM, device=device,
                     lr=args.lr, prioritized=not args.no_prioritized)
    
    # Resume from checkpoint
    if args.resume:
        print(f"Resuming from {args.resume}")
        agent.load(args.resume)
    
    # Live dashboard (opt-in)
    dashboard = None
    if args.visual:
        try:
            from core.board_renderer import WordleBoard
            import matplotlib.pyplot as plt
            
            class TrainingDashboard:
                def __init__(self, board_speed=1.0):
                    plt.ion()
                    self.fig = plt.figure(figsize=(11, 6))
                    gs = self.fig.add_gridspec(2, 2, width_ratios=[1, 1.4])
                    board_ax = self.fig.add_subplot(gs[:, 0])
                    board_ax.set_title("Live demo game", fontsize=11, fontweight="bold")
                    self.board = WordleBoard(board_ax, speed=board_speed)
                    self.ax1 = self.fig.add_subplot(gs[0, 1])
                    self.ax2 = self.fig.add_subplot(gs[1, 1], sharex=self.ax1)
                    (self.line1,) = self.ax1.plot([], [], "o-", color="#6aaa64")
                    self.ax1.set_ylabel("Win rate")
                    self.ax1.set_ylim(-0.05, 1.05)
                    self.ax1.grid(alpha=0.3)
                    (self.line2,) = self.ax2.plot([], [], "o-", color="#c9b458")
                    self.ax2.set_ylabel("Avg guesses")
                    self.ax2.set_xlabel("Episode")
                    self.ax2.set_ylim(0, 6.5)
                    self.ax2.grid(alpha=0.3)
                    self.status = self.fig.text(0.25, 0.02, "", ha="center", fontsize=10)
                    self.episodes, self.win_rates, self.avg_guesses = [], [], []
                    plt.tight_layout()
                    plt.show(block=False)
                    plt.pause(0.3)
                
                def checkpoint(self, ep, wr, ag):
                    self.episodes.append(ep)
                    self.win_rates.append(wr)
                    self.avg_guesses.append(ag)
                    self.line1.set_data(self.episodes, self.win_rates)
                    self.line2.set_data(self.episodes, self.avg_guesses)
                    self.ax1.set_xlim(0, max(self.episodes) * 1.05)
                    self.ax2.set_xlim(0, max(self.episodes) * 1.05)
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                
                def finish(self):
                    plt.ioff()
                    plt.show()
            
            dashboard = TrainingDashboard(board_speed=args.board_speed)
        except Exception as e:
            print(f"Warning: Could not create visual dashboard: {e}")
            dashboard = None
    
    # Epsilon schedule (exponential decay)
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_rate = 3.0 / args.episodes  # reach ~5% by 70% of training
    
    # Tracking
    all_episodes = []
    all_win_rates = []
    all_avg_guesses = []
    all_losses = []
    step_count = 0
    curriculum_idx = 0
    
    start = time.time()
    pbar = tqdm(range(1, args.episodes + 1), desc="Training", unit="ep")
    
    for ep in pbar:
        # Curriculum: expand word pool at milestones
        if len(curriculum_stages) > 1:
            progress = ep / args.episodes
            target_idx = min(int(progress * len(curriculum_stages)), len(curriculum_stages) - 1)
            if target_idx > curriculum_idx:
                curriculum_idx = target_idx
                new_size = curriculum_stages[curriculum_idx]
                word_pool = answers[:new_size]
                env = WordleEnv(answer_pool=word_pool, action_words=word_pool, seed=0)
                print(f"\n  >> Curriculum: expanded to {new_size} words at episode {ep}")
        
        # Exponential epsilon decay
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-epsilon_decay_rate * ep)
        
        obs = env.reset()
        done = False
        ep_loss = []
        
        while not done:
            mask = env.valid_action_mask()
            word_feats, word_indices = env.valid_word_features(mask)
            
            action_local = agent.act(obs, word_feats, epsilon=epsilon)
            action_idx = int(word_indices[action_local])
            chosen_word_feat = word_feats[action_local]
            
            next_obs, reward, done, info = env.step(action_idx)
            
            # Get next valid word features for the replay buffer
            if not done:
                next_mask = env.valid_action_mask()
                next_wf, _ = env.valid_word_features(next_mask)
            else:
                next_wf = np.zeros((1, WORD_FEAT_DIM), dtype=np.float32)
            
            agent.remember(obs, chosen_word_feat, reward, next_obs, done, next_wf)
            
            step_count += 1
            if step_count % args.train_every == 0:
                loss = agent.train_step(batch_size=args.batch_size)
                if loss is not None:
                    ep_loss.append(loss)
            
            obs = next_obs
        
        # Soft target update every step, hard update periodically
        agent.soft_update_target()
        if ep % args.target_update_every == 0:
            agent.hard_update_target()
        
        if ep_loss:
            all_losses.append(np.mean(ep_loss))
        
        # Evaluation checkpoint
        if ep % args.eval_every == 0:
            win_rate, avg_guesses = evaluate(env, agent, n_eval=min(50, len(word_pool)))
            elapsed = time.time() - start
            
            all_episodes.append(ep)
            all_win_rates.append(win_rate)
            all_avg_guesses.append(avg_guesses)
            
            pbar.set_postfix_str(
                f"wr={win_rate:.0%} ag={avg_guesses:.2f} ε={epsilon:.2f} "
                f"pool={len(word_pool)} {elapsed:.0f}s"
            )
            
            if dashboard:
                dashboard.checkpoint(ep, win_rate, avg_guesses)
        
        # Periodic checkpoint save
        if ep % (args.eval_every * 5) == 0:
            checkpoint_path = str(DATA_DIR / f"checkpoint_ep{ep}.pt")
            agent.save(checkpoint_path)
    
    pbar.close()
    elapsed = time.time() - start
    
    # Final evaluation on ALL training words
    print(f"\nTraining complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    
    win_rate, avg_guesses = evaluate(env, agent, n_eval=len(word_pool))
    print(f"Final eval over all {len(word_pool)} words: "
          f"win_rate={win_rate:.2%}  avg_guesses={avg_guesses:.2f}")
    
    # Save model
    model_path = str(DATA_DIR / args.save_name)
    agent.save(model_path)
    print(f"Saved trained model to {model_path}")
    
    # Save training metrics
    elapsed_str = f"{elapsed/60:.1f} min" if elapsed < 3600 else f"{elapsed/3600:.1f} hours"
    metrics = {
        "episodes": all_episodes,
        "win_rates": all_win_rates,
        "avg_guesses": all_avg_guesses,
        "losses": all_losses[-len(all_episodes):] if all_losses else [],
        "final_win_rate": float(win_rate),
        "final_avg_guesses": float(avg_guesses),
        "n_words": len(word_pool),
        "total_episodes": args.episodes,
        "training_time_seconds": elapsed,
        "training_time_str": elapsed_str,
        "architecture": "EmbeddingDQN_v2",
        "device": device,
        "curriculum_stages": curriculum_stages,
    }
    metrics_path = str(DATA_DIR / "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved training metrics to {metrics_path}")
    
    # Auto-generate result images
    print("\nGenerating result images...")
    os.makedirs(str(ASSETS_DIR), exist_ok=True)
    try:
        save_training_plots(metrics, str(ASSETS_DIR))
        print("[+] All result images saved to assets/")
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up checkpoints (keep only the final model)
    for f in DATA_DIR.glob("checkpoint_ep*.pt"):
        f.unlink()
    
    if dashboard:
        print("Close the window to exit.")
        dashboard.finish()


if __name__ == "__main__":
    main()