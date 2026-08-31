"""
Benchmark the trained Embedding DQN agent (v2).

Features:
    - Standard benchmark on training pool words
    - Out-of-distribution (OOD) testing on words NOT in training pool
    - Auto-saves results to assets/ as both JSON and image
    - Progress bar
"""
import sys
import os
import json
import time
import argparse
from pathlib import Path
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))

from core.wordle_mdp import load_word_lists
from core.wordle_env import WordleEnv, OBS_DIM, WORD_FEAT_DIM
from core.dqn_agent import DQNAgent

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=None, desc="", **kwargs):
            self.iterable = iterable
        def __iter__(self):
            yield from self.iterable
        def close(self):
            pass

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"


def benchmark_agent(env, agent, targets, desc="Benchmarking"):
    """Run the agent on a list of target words, return stats."""
    guess_counts = Counter()
    failures = []
    
    pbar = tqdm(targets, desc=desc)
    for answer in pbar:
        obs = env.reset(answer=answer)
        done = False
        n_guesses = 0
        
        while not done:
            mask = env.valid_action_mask()
            word_feats, word_indices = env.valid_word_features(mask)
            action_local = agent.act(obs, word_feats, epsilon=0.0)
            action_idx = int(word_indices[action_local])
            obs, reward, done, info = env.step(action_idx)
            n_guesses += 1
        
        if info["solved"]:
            guess_counts[n_guesses] += 1
        else:
            failures.append(answer)
    
    pbar.close()
    total_solved = sum(guess_counts.values())
    total_guesses = sum(k * v for k, v in guess_counts.items())
    
    return {
        "total_tested": len(targets),
        "solved": total_solved,
        "failed": len(failures),
        "win_rate": total_solved / max(1, len(targets)),
        "avg_guesses": total_guesses / max(1, total_solved),
        "guess_distribution": dict(sorted(guess_counts.items())),
        "failed_words": failures[:20],  # first 20 failures
    }


def save_benchmark_plot(results, ood_results=None, save_dir=None):
    """Generate and save benchmark result visualizations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    if ood_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    
    fig.suptitle("DQN v2 Benchmark Results", fontsize=14, fontweight='bold')
    
    # In-distribution results
    dist = results["guess_distribution"]
    if dist:
        keys = sorted(dist.keys())
        values = [dist[k] for k in keys]
        colors = ['#6aaa64' if k <= 6 else '#c62828' for k in keys]
        bars = ax1.bar([str(k) for k in keys], values, color=colors, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    str(val), ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax1.set_xlabel('Number of Guesses', fontweight='bold')
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title(f'In-Distribution (Win: {results["win_rate"]:.1%}, Avg: {results["avg_guesses"]:.2f})',
                  fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle=':')
    
    # OOD results if available
    if ood_results and ax2 is not None:
        dist_ood = ood_results["guess_distribution"]
        if dist_ood:
            keys = sorted(dist_ood.keys())
            values = [dist_ood[k] for k in keys]
            colors = ['#42a5f5' if k <= 6 else '#c62828' for k in keys]
            bars = ax2.bar([str(k) for k in keys], values, color=colors, edgecolor='white', linewidth=1.5)
            for bar, val in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        str(val), ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax2.set_xlabel('Number of Guesses', fontweight='bold')
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title(f'Out-of-Distribution (Win: {ood_results["win_rate"]:.1%}, Avg: {ood_results["avg_guesses"]:.2f})',
                      fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    path = os.path.join(save_dir, "benchmark_results.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [+] Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark the trained DQN v2 agent")
    parser.add_argument("--n_words", type=int, default=200,
                         help="Word pool size used during training")
    parser.add_argument("--sample", type=int, default=None,
                         help="Only test on this many words (faster)")
    parser.add_argument("--model_path", type=str, 
                         default=str(DATA_DIR / "dqn_wordle_v2.pt"),
                         help="Path to the trained model")
    parser.add_argument("--ood", action="store_true",
                         help="Also test on out-of-distribution words")
    parser.add_argument("--no_plot", action="store_true",
                         help="Skip generating plots")
    args = parser.parse_args()
    
    answers, _ = load_word_lists()
    word_pool = answers[:args.n_words]
    
    # Load trained agent
    agent = DQNAgent(obs_dim=OBS_DIM, word_dim=WORD_FEAT_DIM)
    agent.load(args.model_path)
    print(f"Loaded model from {args.model_path}")
    
    # In-distribution benchmark
    env = WordleEnv(answer_pool=word_pool, action_words=word_pool)
    targets = word_pool if args.sample is None else word_pool[:args.sample]
    
    print(f"\n{'='*60}")
    print(f"IN-DISTRIBUTION BENCHMARK ({len(targets)} words)")
    print(f"{'='*60}")
    
    start = time.time()
    results = benchmark_agent(env, agent, targets, desc="In-dist benchmark")
    elapsed = time.time() - start
    
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Solved: {results['solved']}/{results['total_tested']}  ({results['win_rate']:.1%})")
    print(f"Failed (>6 guesses): {results['failed']}")
    print(f"Average guesses (solved): {results['avg_guesses']:.3f}")
    print("\nGuess-count distribution:")
    for k in sorted(results["guess_distribution"]):
        count = results["guess_distribution"][k]
        bar = "#" * (count * 40 // max(1, results['total_tested']))
        print(f"  {k}: {count:5d}  {bar}")
    
    # Out-of-distribution benchmark
    ood_results = None
    if args.ood and args.n_words < len(answers):
        ood_words = answers[args.n_words:args.n_words + min(200, len(answers) - args.n_words)]
        
        # For OOD testing, the agent still uses its training word pool as
        # its action space, but the SECRET words are from OUTSIDE that pool.
        # This tests whether the agent's strategy generalizes.
        env_ood = WordleEnv(answer_pool=ood_words, action_words=word_pool)
        
        print(f"\n{'='*60}")
        print(f"OUT-OF-DISTRIBUTION BENCHMARK ({len(ood_words)} unseen words)")
        print(f"{'='*60}")
        
        ood_results = benchmark_agent(env_ood, agent, ood_words, desc="OOD benchmark")
        
        print(f"\nSolved: {ood_results['solved']}/{ood_results['total_tested']}  ({ood_results['win_rate']:.1%})")
        print(f"Failed: {ood_results['failed']}")
        print(f"Average guesses (solved): {ood_results['avg_guesses']:.3f}")
        if ood_results['failed_words']:
            print(f"Failed words (first 20): {ood_results['failed_words']}")
    
    # Save results
    os.makedirs(str(ASSETS_DIR), exist_ok=True)
    all_results = {"in_distribution": results}
    if ood_results:
        all_results["out_of_distribution"] = ood_results
    
    results_path = str(ASSETS_DIR / "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Saved results to {results_path}")
    
    if not args.no_plot:
        try:
            save_benchmark_plot(results, ood_results, str(ASSETS_DIR))
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")


if __name__ == "__main__":
    main()