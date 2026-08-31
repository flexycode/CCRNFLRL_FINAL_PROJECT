"""
Benchmark the trained DQN agent on the 8,636-word ENABLE1 pool.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import argparse
from collections import Counter
from core.wordle_mdp import load_word_lists
from core.wordle_env import WordleEnv, OBS_DIM
from core.dqn_agent import DQNAgent

def main(n_words=8636, sample_size=None):
    answers, _ = load_word_lists()
    word_pool = answers[:n_words]
    
    # Load trained agent
    agent = DQNAgent(obs_dim=OBS_DIM, n_actions=len(word_pool))
    agent.load(str(Path(__file__).parent.parent / 'data' / 'dqn_wordle.pt'))
    
    targets = word_pool if sample_size is None else word_pool[:sample_size]
    print(f"Benchmarking DQN agent ({n_words} words) against {len(targets)} answers...\n")
    
    guess_counts = Counter()
    failures = []
    start = time.time()
    
    for i, answer in enumerate(targets):
        env = WordleEnv(answer_pool=word_pool, action_words=word_pool)
        obs = env.reset(answer=answer)
        done = False
        n_guesses = 0
        
        while not done:
            mask = env.valid_action_mask()
            action = agent.act(obs, mask, epsilon=0.0)
            obs, reward, done, info = env.step(action)
            n_guesses += 1
        
        solved = info["solved"]
        if solved:
            guess_counts[n_guesses] += 1
        else:
            failures.append(answer)
        
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(targets)} done")
    
    elapsed = time.time() - start
    total_solved = sum(guess_counts.values())
    total_guesses = sum(k * v for k, v in guess_counts.items())
    
    print(f"\nDone in {elapsed:.1f}s\n")
    print(f"Solved: {total_solved}/{len(targets)}  ({100*total_solved/len(targets):.1f}%)")
    print(f"Failed (>6 guesses): {len(failures)}")
    print(f"\nAverage guesses (solved games): {total_guesses/total_solved:.3f}")
    print("\nGuess-count distribution:")
    for k in sorted(guess_counts):
        bar = "#" * (guess_counts[k] // max(1, len(targets)//200))
        print(f"  {k}: {guess_counts[k]:5d}  {bar}")

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(n_words=8636, sample_size=n)