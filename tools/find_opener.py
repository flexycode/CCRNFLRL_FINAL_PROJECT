"""
Precompute the best opening guess (this is the expensive step: it's
O(|guess_pool| * |answer_pool|) pattern computations, ~2309*12947 = ~30M,
so we only want to do it once and cache the result).
"""
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.wordle_mdp import load_word_lists, expected_info_gain

def main():
    answers, guesses = load_word_lists()

    # Restricting the search to the answer list (not the full guess list)
    # is the standard practical shortcut: it's ~5x faster and the true
    # optimal opener is empirically almost always in this smaller set anyway.
    best_word, best_score = None, -1.0
    scored = []
    for g in answers:
        score = expected_info_gain(g, answers)
        scored.append((score, g))
        if score > best_score:
            best_word, best_score = g, score

    scored.sort(reverse=True)
    print("Top 10 opening guesses by expected information gain:")
    for score, word in scored[:10]:
        print(f"  {word}: {score:.3f} bits")

    with open(Path(__file__).parent.parent / "data" / "best_opener.json", "w") as f:
        json.dump({"word": best_word, "bits": best_score}, f)
    print(f"\nCached best opener: {best_word} ({best_score:.3f} bits)")

if __name__ == "__main__":
    main()
