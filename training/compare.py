"""
Head-to-head: trained Embedding DQN agent vs. the entropy-greedy heuristic,
on the same word pool. Auto-saves comparison chart to assets/.
"""
import argparse
import os
import sys
import json
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.wordle_env import WordleEnv, OBS_DIM, WORD_FEAT_DIM
from core.wordle_mdp import load_word_lists, best_guess, filter_candidates, score_guess
from core.dqn_agent import DQNAgent

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"


def run_dqn(env, agent, word_pool):
    wins, total_guesses = 0, 0
    for answer in word_pool:
        obs = env.reset(answer=answer)
        done = False
        steps = 0
        while not done:
            mask = env.valid_action_mask()
            word_feats, word_indices = env.valid_word_features(mask)
            action_local = agent.act(obs, word_feats, epsilon=0.0)
            action_idx = int(word_indices[action_local])
            obs, reward, done, info = env.step(action_idx)
            steps += 1
        if info["solved"]:
            wins += 1
            total_guesses += steps
    return wins / len(word_pool), total_guesses / max(1, wins)


def run_heuristic(word_pool, full_answer_pool, guess_pool):
    wins, total_guesses = 0, 0
    for answer in word_pool:
        candidates = list(full_answer_pool)
        for turn in range(1, 7):
            guess = best_guess(candidates, guess_pool)
            pattern = score_guess(guess, answer)
            if pattern == (2, 2, 2, 2, 2):
                wins += 1
                total_guesses += turn
                break
            candidates = filter_candidates(candidates, guess, pattern)
        else:
            pass
    return wins / len(word_pool), total_guesses / max(1, wins)


def save_comparison_chart(dqn_wr, dqn_avg, heur_wr, heur_avg, n_words, save_dir):
    """Generate and save a comparison bar chart."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"DQN v2 vs Entropy Heuristic ({n_words} words)", 
                 fontsize=14, fontweight='bold')
    
    agents = ['DQN v2', 'Entropy\nHeuristic']
    colors_wr = ['#42a5f5', '#6aaa64']
    colors_ag = ['#42a5f5', '#c9b458']
    
    # Win rates
    bars1 = ax1.bar(agents, [dqn_wr, heur_wr], color=colors_wr, 
                    edgecolor='white', linewidth=2, width=0.5)
    ax1.set_ylabel('Win Rate', fontweight='bold', fontsize=11)
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3, linestyle=':')
    ax1.set_title('Win Rate Comparison', fontweight='bold')
    for bar, val in zip(bars1, [dqn_wr, heur_wr]):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Avg guesses
    bars2 = ax2.bar(agents, [dqn_avg, heur_avg], color=colors_ag,
                    edgecolor='white', linewidth=2, width=0.5)
    ax2.set_ylabel('Avg Guesses', fontweight='bold', fontsize=11)
    ax2.set_ylim(0, 6.5)
    ax2.grid(axis='y', alpha=0.3, linestyle=':')
    ax2.set_title('Efficiency Comparison', fontweight='bold')
    for bar, val in zip(bars2, [dqn_avg, heur_avg]):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_chart.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [+] Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_words", type=int, default=200)
    parser.add_argument("--model_path", type=str, 
                         default=str(DATA_DIR / "dqn_wordle_v2.pt"))
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    answers, guesses = load_word_lists()
    word_pool = answers[:args.n_words]

    env = WordleEnv(answer_pool=word_pool, action_words=word_pool, seed=0)
    agent = DQNAgent(obs_dim=OBS_DIM, word_dim=WORD_FEAT_DIM)
    agent.load(args.model_path)

    print(f"Comparing on {len(word_pool)} words...\n")
    
    t0 = time.time()
    dqn_win_rate, dqn_avg = run_dqn(env, agent, word_pool)
    t1 = time.time()
    print(f"DQN v2 agent:      win_rate={dqn_win_rate:.2%}  avg_guesses={dqn_avg:.2f}  ({t1-t0:.1f}s)")

    heur_win_rate, heur_avg = run_heuristic(word_pool, word_pool, word_pool)
    t2 = time.time()
    print(f"Entropy heuristic: win_rate={heur_win_rate:.2%}  avg_guesses={heur_avg:.2f}  ({t2-t1:.1f}s)")
    
    # Save comparison results
    os.makedirs(str(ASSETS_DIR), exist_ok=True)
    comparison = {
        "n_words": args.n_words,
        "dqn": {"win_rate": dqn_win_rate, "avg_guesses": dqn_avg},
        "heuristic": {"win_rate": heur_win_rate, "avg_guesses": heur_avg},
    }
    with open(str(ASSETS_DIR / "comparison_results.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    
    if not args.no_plot:
        try:
            save_comparison_chart(dqn_win_rate, dqn_avg, heur_win_rate, heur_avg,
                                  args.n_words, str(ASSETS_DIR))
        except Exception as e:
            print(f"Warning: Could not generate chart: {e}")


if __name__ == "__main__":
    main()
