"""
Interactive assistant for solving a REAL Wordle game.

Play the actual game (NYT Wordle, or any clone), and after each guess,
type in the feedback pattern you see using:
    g = green, y = yellow, b = black/gray

e.g. if you guessed "raise" and got green-gray-yellow-gray-green, type: gbybg

The assistant re-applies the MDP state transition (filtering the candidate
set) and recommends the next action (guess) via the entropy-greedy policy.
"""
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.wordle_mdp import load_word_lists, best_guess, filter_candidates, pattern_to_str

PATTERN_MAP = {"g": 2, "y": 1, "b": 0}

def parse_pattern(s):
    s = s.strip().lower()
    if len(s) != 5 or any(c not in PATTERN_MAP for c in s):
        return None
    return tuple(PATTERN_MAP[c] for c in s)

def main():
    answers, guesses = load_word_lists()
    candidates = list(answers)

    with open(Path(__file__).parent.parent / "data" / "best_opener.json") as f:
        opener = json.load(f)["word"]

    print("=== Wordle Solver Assistant ===")
    print("Feedback format: g=green, y=yellow, b=gray/black (e.g. 'gbybg')\n")

    turn = 1
    suggestion = opener
    while True:
        print(f"Turn {turn}: try guessing -> {suggestion.upper()}  "
              f"({len(candidates)} candidates remaining)")
        guess = input("  Guess you actually entered (Enter to accept suggestion): ").strip().lower()
        if not guess:
            guess = suggestion
        if len(guess) != 5:
            print("  Guess must be 5 letters, try again.")
            continue

        fb = input("  Feedback pattern (e.g. gbybg), or 'solved': ").strip().lower()
        if fb in ("solved", "ggggg"):
            print(f"\nSolved in {turn} guesses! ({guess.upper()})")
            break

        pattern = parse_pattern(fb)
        if pattern is None:
            print("  Couldn't parse that pattern, use only g/y/b, 5 chars.")
            continue

        candidates = filter_candidates(candidates, guess, pattern)
        print(f"  -> {len(candidates)} candidates remain.")

        if len(candidates) == 0:
            print("  No candidates left - double check your feedback entries.")
            break
        if len(candidates) <= 10:
            print(f"  Remaining candidates: {candidates}")

        turn += 1
        if turn > 6:
            print("Out of guesses!")
            break

        suggestion = best_guess(candidates, guesses)

if __name__ == "__main__":
    main()
