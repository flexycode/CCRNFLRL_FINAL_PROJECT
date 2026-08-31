"""
Explicit Environment / Agent split for Wordle-as-MDP.

WordleEnv follows the standard RL env interface:
    obs = env.reset()
    obs, reward, done, info = env.step(action)

The environment owns the TRUE hidden state (the secret answer) and the
ground-truth transition dynamics (score_guess). It does NOT know or care
what policy the agent uses.

State representation (obs) is a fixed-size numeric feature vector, since a
raw "set of remaining candidates" can't be fed into a neural net directly:

  For each of the 5 positions x 26 letters: one-hot "this letter is
  confirmed correct at this position" (green)                =  5*26 = 130
  For each of 26 letters: "confirmed present somewhere, but we
  don't yet know all its correct positions" (yellow-ever-seen) =    26
  For each of 26 letters: "confirmed absent from the word"     =    26
  Guesses remaining (normalized)                                =     1
  ------------------------------------------------------------------
  Total obs dim                                                 = 183

This is a lossy compression of the true state (the exact candidate set) -
that's the necessary tradeoff to make it learnable with function
approximation instead of a table.
"""
import numpy as np
import random
from core.wordle_mdp import load_word_lists, score_guess

N_POS = 5
N_LET = 26
OBS_DIM = N_POS * N_LET + N_LET + N_LET + 1  # 130 + 26 + 26 + 1 = 183
MAX_GUESSES = 6


def letter_idx(c):
    return ord(c) - ord('a')


class WordleEnv:
    """Gym-style environment. Action = index into self.action_words."""

    def __init__(self, answer_pool, action_words, seed=None):
        self.answer_pool = answer_pool          # words the env can pick as secret
        self.action_words = action_words        # legal actions (restricted list)
        self.word_to_action = {w: i for i, w in enumerate(action_words)}
        self.rng = random.Random(seed)
        self.reset()

    def reset(self, answer=None):
        self.answer = answer or self.rng.choice(self.answer_pool)
        self.guesses_used = 0
        self.done = False
        # per-position green letter (or None)
        self.green = [None] * N_POS
        # letters confirmed present somewhere (yellow-ever-seen)
        self.present = set()
        # letters confirmed fully absent
        self.absent = set()
        # candidate set, kept ONLY for reward shaping / info, not given to agent
        self._candidates = list(self.answer_pool)
        self._guessed = set()  # words already tried this episode - never worth repeating
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        offset = 0
        for pos in range(N_POS):
            if self.green[pos] is not None:
                obs[offset + pos * N_LET + letter_idx(self.green[pos])] = 1.0
        offset += N_POS * N_LET
        for c in self.present:
            obs[offset + letter_idx(c)] = 1.0
        offset += N_LET
        for c in self.absent:
            obs[offset + letter_idx(c)] = 1.0
        offset += N_LET
        obs[offset] = (MAX_GUESSES - self.guesses_used) / MAX_GUESSES
        return obs

    def valid_action_mask(self):
        """Boolean mask over action_words: True if still consistent with
        everything learned so far. Used to prevent the agent wasting guesses
        on words already known to be impossible (standard action-masking)."""
        mask = np.zeros(len(self.action_words), dtype=bool)
        for i, w in enumerate(self.action_words):
            ok = True
            for pos, g in enumerate(self.green):
                if g is not None and w[pos] != g:
                    ok = False
                    break
            if ok:
                if any(c not in w for c in self.present):
                    ok = False
                if ok and any(c in w for c in self.absent):
                    ok = False
                if ok and w in self._guessed:
                    ok = False  # never worth repeating an exact past guess
            mask[i] = ok
        return mask

    def step(self, action_idx):
        if self.done:
            raise RuntimeError("step() called after episode finished")

        guess = self.action_words[action_idx]
        pattern = score_guess(guess, self.answer)
        self.guesses_used += 1
        self._guessed.add(guess)

        prev_candidate_count = len(self._candidates)

        # update known info from pattern
        for pos, p in enumerate(pattern):
            if p == 2:
                self.green[pos] = guess[pos]
            elif p == 1:
                self.present.add(guess[pos])
            else:  # p == 0
                # only mark absent if this letter isn't confirmed present
                # elsewhere in the word (duplicate-letter safe default)
                if guess[pos] not in [self.green[i] for i in range(N_POS)] \
                        and guess[pos] not in self.present:
                    self.absent.add(guess[pos])

        # shrink the (internal-only) true candidate set for reward shaping
        self._candidates = [w for w in self._candidates if score_guess(guess, w) == pattern]

        solved = (pattern == (2, 2, 2, 2, 2))
        truncated = (self.guesses_used >= MAX_GUESSES) and not solved
        self.done = solved or truncated

        # --- reward shaping ---
        # -1 per turn (cost of a guess) is the "pure" MDP reward.
        # We add a shaped bonus for shrinking the candidate set (log-ratio,
        # so it approximates the entropy/information-gain signal) to make
        # learning tractable over a 2300-word action space, plus solve/fail
        # terminal bonuses.
        reward = -1.0
        new_count = max(1, len(self._candidates))
        info_gain = np.log2(max(1, prev_candidate_count) / new_count)
        reward += 0.5 * info_gain
        if solved:
            reward += 10.0
        elif truncated:
            reward -= 5.0

        info = {
            "pattern": pattern,
            "guess": guess,
            "solved": solved,
            "candidates_remaining": len(self._candidates),
        }
        return self._get_obs(), reward, self.done, info


if __name__ == "__main__":
    answers, guesses = load_word_lists()
    env = WordleEnv(answer_pool=answers, action_words=answers, seed=0)
    obs = env.reset(answer="knoll")
    print("obs dim:", obs.shape)
    obs, r, done, info = env.step(env.word_to_action["raise"])
    print(info, "reward:", r)
