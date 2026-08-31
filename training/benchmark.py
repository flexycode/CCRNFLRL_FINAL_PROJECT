"""
Benchmark the entropy-greedy policy: play it against every possible answer
and report the distribution of guesses-to-solve. This is how you evaluate
an MDP policy empirically when exact value iteration is intractable.
"""
import json
import time
from collections import Counter
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import time

from core.wordle_mdp import load_word_lists, play_episode

def main(sample_size=None):
    answers, guesses = load_word_lists()

    with open(Path(__file__).parent.parent / "data" / "best_opener.json") as f:
        opener = json.load(f)["word"]

    targets = answers if sample_size is None else answers[:sample_size]
    print(f"Benchmarking policy (opener='{opener}') against {len(targets)} answers...\n")

    guess_counts = Counter()
    failures = []
    start = time.time()

    for i, answer in enumerate(targets):
        transitions = play_episode(answer, guesses, answers, first_guess=opener)
        n_guesses = len(transitions)
        solved = transitions[-1][1] == (2, 2, 2, 2, 2)
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
    if failures:
        print(f"  e.g. {failures[:10]}")
    print(f"\nAverage guesses (solved games): {total_guesses/total_solved:.3f}")
    print("\nGuess-count distribution:")
    for k in sorted(guess_counts):
        bar = "#" * (guess_counts[k] // max(1, len(targets)//200))
        print(f"  {k}: {guess_counts[k]:5d}  {bar}")

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(n)
