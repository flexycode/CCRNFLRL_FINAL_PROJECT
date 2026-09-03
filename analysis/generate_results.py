"""
One-click result generator for the DQN v2 Wordle Solver.

Loads the latest training metrics, runs benchmarks, and generates
all visualization images into assets/.

Usage:
    python analysis/generate_results.py
    python analysis/generate_results.py --model_path data/dqn_wordle_v2.pt --n_words 200
"""
import sys
import os
import json
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"


def main():
    parser = argparse.ArgumentParser(description="Generate all result images")
    parser.add_argument("--metrics", type=str, 
                         default=str(DATA_DIR / "training_metrics.json"))
    parser.add_argument("--model_path", type=str,
                         default=str(DATA_DIR / "dqn_wordle_v2.pt"))
    parser.add_argument("--n_words", type=int, default=None,
                         help="Override word pool size (auto-detected from metrics)")
    args = parser.parse_args()
    
    os.makedirs(str(ASSETS_DIR), exist_ok=True)
    
    # 1. Load training metrics
    print("=" * 60)
    print("GENERATING ALL RESULT IMAGES")
    print("=" * 60)
    
    metrics = None
    if Path(args.metrics).exists():
        with open(args.metrics) as f:
            metrics = json.load(f)
        print(f"\n[+] Loaded training metrics ({metrics.get('total_episodes', '?')} episodes)")
        
        n_words = args.n_words or metrics.get("n_words", 200)
    else:
        print(f"\n[!] No training metrics found at {args.metrics}")
        n_words = args.n_words or 200
    
    # 2. Generate training plots
    if metrics and metrics.get("episodes"):
        print("\n--- Training Plots ---")
        # Import the training plot function
        sys.path.insert(0, str(PROJECT_ROOT / "training"))
        from train import save_training_plots
        save_training_plots(metrics, str(ASSETS_DIR))
    else:
        print("\n[!] Skipping training plots (no checkpoint data)")
    
    # 3. Run benchmark if model exists
    if Path(args.model_path).exists():
        print("\n--- Benchmark ---")
        from core.wordle_mdp import load_word_lists
        from core.wordle_env import WordleEnv, OBS_DIM, WORD_FEAT_DIM
        from core.dqn_agent import DQNAgent
        
        answers, _ = load_word_lists()
        word_pool = answers[:n_words]
        
        agent = DQNAgent(obs_dim=OBS_DIM, word_dim=WORD_FEAT_DIM)
        agent.load(args.model_path)
        
        env = WordleEnv(answer_pool=word_pool, action_words=word_pool)
        
        from training.benchmark_dqn import benchmark_agent, save_benchmark_plot
        
        # In-distribution
        results = benchmark_agent(env, agent, word_pool[:min(200, len(word_pool))],
                                   desc="In-dist benchmark")
        print(f"  In-dist: {results['win_rate']:.1%} win rate, {results['avg_guesses']:.2f} avg guesses")
        
        # OOD
        ood_results = None
        if n_words < len(answers):
            ood_words = answers[n_words:n_words + min(100, len(answers) - n_words)]
            env_ood = WordleEnv(answer_pool=ood_words, action_words=word_pool)
            ood_results = benchmark_agent(env_ood, agent, ood_words, desc="OOD benchmark")
            print(f"  OOD: {ood_results['win_rate']:.1%} win rate, {ood_results['avg_guesses']:.2f} avg guesses")
        
        save_benchmark_plot(results, ood_results, str(ASSETS_DIR))
        
        # Save combined results
        all_results = {"in_distribution": results}
        if ood_results:
            all_results["out_of_distribution"] = ood_results
        with open(str(ASSETS_DIR / "benchmark_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
    else:
        print(f"\n[!] No model found at {args.model_path}, skipping benchmark")
    
    print(f"\n{'='*60}")
    print(f"[+] All results saved to: {ASSETS_DIR}")
    print(f"{'='*60}")
    
    # List generated files
    for f in sorted(ASSETS_DIR.glob("*")):
        size = f.stat().st_size
        print(f"  {f.name} ({size:,} bytes)")


if __name__ == "__main__":
    main()
