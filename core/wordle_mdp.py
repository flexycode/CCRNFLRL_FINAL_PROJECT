"""
Wordle as an MDP.

State   : the set of candidate words still consistent with all feedback so far
          (this is the sufficient statistic - two different guess histories
          that leave the same candidate set are equivalent going forward),
          plus the number of guesses used.
Action  : which 5-letter word to guess next (from the full valid-guess list).
Transition: stochastic in which of the 243 (3^5) feedback patterns occurs,
          weighted by how many remaining candidates would produce that
          pattern (assuming a uniform prior over remaining answers).
          Given a pattern, the transition to the next state is deterministic:
          filter the candidate set down to words consistent with that pattern.
Reward  : -1 per guess taken; solving is the terminal state.

Exact Bellman-optimal solving is intractable (state space = subsets of ~2315
words), so the policy implemented here is the standard near-optimal
approximation: at each state, choose the action that maximizes expected
information gain (entropy reduction) over the candidate set. This is a
one-step-greedy policy, not full value iteration, but it is what basically
every strong Wordle solver (e.g. 3Blue1Brown's) actually uses in practice.
"""

import json
import math
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Word lists
# ---------------------------------------------------------------------------

def load_word_lists():
    """Load the real NYT Wordle lists.
    ANSWERS   = words the puzzle actually picks as the secret word.
    GUESSES   = ANSWERS + accepted-but-never-secret words (full legal action space).
    """
    with open(DATA_DIR / "wordles.json") as f:
        answers = json.load(f)
    with open(DATA_DIR / "nonwordles.json") as f:
        nonanswers = json.load(f)
    answers = [w.lower() for w in answers]
    guesses = sorted(set(answers) | set(w.lower() for w in nonanswers))
    return answers, guesses


# ---------------------------------------------------------------------------
# Feedback function (the environment's transition dynamics)
# ---------------------------------------------------------------------------
# Pattern encoding: tuple of 5 ints, one per letter position.
#   2 = green (correct letter, correct spot)
#   1 = yellow (letter in word, wrong spot)
#   0 = gray (letter not in word, accounting for duplicate letters)

def score_guess(guess: str, answer: str) -> tuple:
    """Replicates Wordle's actual duplicate-letter handling."""
    result = [0] * 5
    answer_chars = list(answer)

    # First pass: greens
    for i in range(5):
        if guess[i] == answer_chars[i]:
            result[i] = 2
            answer_chars[i] = None  # consume this letter

    # Second pass: yellows (only from letters not already consumed)
    remaining = Counter(c for c in answer_chars if c is not None)
    for i in range(5):
        if result[i] == 0:
            g = guess[i]
            if remaining.get(g, 0) > 0:
                result[i] = 1
                remaining[g] -= 1

    return tuple(result)


def pattern_to_str(pattern: tuple) -> str:
    symbols = {0: "\u2b1c", 1: "\U0001f7e8", 2: "\U0001f7e9"}  # white/yellow/green squares
    return "".join(symbols[p] for p in pattern)


# ---------------------------------------------------------------------------
# State transition: filter candidates by an observed pattern
# ---------------------------------------------------------------------------

def filter_candidates(candidates, guess, pattern):
    return [w for w in candidates if score_guess(guess, w) == pattern]


# ---------------------------------------------------------------------------
# Policy: entropy-maximizing greedy action selection
# ---------------------------------------------------------------------------

def expected_info_gain(guess, candidates):
    """Expected bits of information gained by guessing `guess`, given the
    current candidate set (used as the distribution over the true answer).
    This equals the entropy of the pattern distribution: higher = the guess
    splits the candidate set into more even, more numerous groups.
    """
    n = len(candidates)
    if n <= 1:
        return 0.0
    pattern_counts = Counter(score_guess(guess, ans) for ans in candidates)
    entropy = 0.0
    for count in pattern_counts.values():
        p = count / n
        entropy -= p * math.log2(p)
    return entropy


def best_guess(candidates, guess_pool, full_search_threshold=2):
    """Pick the action (guess) maximizing expected information gain.

    When very few candidates remain, prefer guessing an actual remaining
    candidate (chance of winning immediately) over a pure-information probe.
    """
    if len(candidates) <= full_search_threshold:
        return candidates[0]

    best_word, best_score = None, -1.0
    for g in guess_pool:
        score = expected_info_gain(g, candidates)
        # tie-break: prefer a guess that could itself be the answer
        if score > best_score or (score == best_score and g in candidates):
            best_word, best_score = g, score
    return best_word


# ---------------------------------------------------------------------------
# Reward / episode simulation
# ---------------------------------------------------------------------------

def play_episode(answer, guess_pool, answer_pool, first_guess=None, max_guesses=6, verbose=False):
    """Simulate one MDP episode against a known answer. Returns list of
    (guess, pattern, reward) transitions. Reward = -1 per guess."""
    candidates = list(answer_pool)
    transitions = []

    for turn in range(1, max_guesses + 1):
        if turn == 1 and first_guess:
            guess = first_guess
        else:
            guess = best_guess(candidates, guess_pool)

        pattern = score_guess(guess, answer)
        reward = -1
        transitions.append((guess, pattern, reward))

        if verbose:
            print(f"  Turn {turn}: {guess}  {pattern_to_str(pattern)}  "
                  f"({len(candidates)} candidates before this guess)")

        if pattern == (2, 2, 2, 2, 2):
            return transitions  # solved

        candidates = filter_candidates(candidates, guess, pattern)

    return transitions  # failed to solve in max_guesses


if __name__ == "__main__":
    answers, guesses = load_word_lists()
    print(f"Loaded {len(answers)} possible answers, {len(guesses)} total legal guesses.")
