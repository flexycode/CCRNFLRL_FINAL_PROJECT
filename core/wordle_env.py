"""
Explicit Environment / Agent split for Wordle-as-MDP (v2 — Optimized).

WordleEnv follows the standard RL env interface:
    obs = env.reset()
    obs, reward, done, info = env.step(action)

Key v2 improvements:
    - Vectorized action masking (NumPy broadcast, no Python loop)
    - Word feature encoding support for embedding-based DQN
    - Improved reward shaping with progressive solve bonus
    - Pre-computed word character arrays for fast masking
"""
import numpy as np
import random
from core.wordle_mdp import load_word_lists, score_guess, encode_word_batch

N_POS = 5
N_LET = 26
OBS_DIM = N_POS * N_LET + N_LET + N_LET + 1  # 130 + 26 + 26 + 1 = 183
WORD_FEAT_DIM = 130  # 5 × 26 one-hot encoding per word
MAX_GUESSES = 6


def letter_idx(c):
    return ord(c) - ord('a')


class WordleEnv:
    """Gym-style environment. Action = index into the VALID action set (dynamic).
    
    v2 changes:
        - Pre-computes character arrays for vectorized masking
        - Provides word feature matrices for the embedding-based DQN
        - Improved reward function
    """

    def __init__(self, answer_pool, action_words, seed=None):
        self.answer_pool = answer_pool          # words the env can pick as secret
        self.action_words = action_words        # legal actions (full list)
        self.word_to_action = {w: i for i, w in enumerate(action_words)}
        self.rng = random.Random(seed)
        
        # Pre-compute character arrays for vectorized masking
        # char_at[i, pos] = letter index (0-25) of the i-th word at position pos
        self._char_at = np.array([[ord(c) - ord('a') for c in w] for w in action_words],
                                  dtype=np.int8)  # (N, 5)
        # char_set[i, letter] = True if letter appears anywhere in word i
        self._char_set = np.zeros((len(action_words), 26), dtype=bool)
        for i, w in enumerate(action_words):
            for c in w:
                self._char_set[i, letter_idx(c)] = True
        
        # Pre-compute word feature encodings (130-dim one-hot per word)
        self._word_feats = encode_word_batch(action_words)  # (N, 130)
        
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
        self._guessed_indices = set()
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
        everything learned so far. Uses vectorized NumPy operations for speed."""
        N = len(self.action_words)
        mask = np.ones(N, dtype=bool)
        
        # Green constraints: word[pos] must equal the confirmed green letter
        for pos in range(N_POS):
            if self.green[pos] is not None:
                required = letter_idx(self.green[pos])
                mask &= (self._char_at[:, pos] == required)
        
        # Present (yellow) constraints: word must contain each confirmed-present letter
        for c in self.present:
            li = letter_idx(c)
            mask &= self._char_set[:, li]
        
        # Absent constraints: word must NOT contain any confirmed-absent letter
        for c in self.absent:
            li = letter_idx(c)
            mask &= ~self._char_set[:, li]
        
        # Never repeat a guess
        for idx in self._guessed_indices:
            mask[idx] = False
        
        return mask

    def valid_word_features(self, mask=None):
        """Return (word_feats, word_indices) for all valid candidate words.
        
        Args:
            mask: optional pre-computed mask from valid_action_mask()
            
        Returns:
            word_feats: (K, 130) numpy array of word feature vectors
            word_indices: (K,) numpy array of indices into self.action_words
        """
        if mask is None:
            mask = self.valid_action_mask()
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            # Fallback: return all words if mask is empty (shouldn't happen)
            indices = np.arange(len(self.action_words))
        return self._word_feats[indices], indices

    def step(self, action_idx):
        if self.done:
            raise RuntimeError("step() called after episode finished")

        guess = self.action_words[action_idx]
        pattern = score_guess(guess, self.answer)
        self.guesses_used += 1
        self._guessed.add(guess)
        self._guessed_indices.add(action_idx)

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

        # --- reward shaping (v2: improved) ---
        reward = -1.0
        new_count = max(1, len(self._candidates))
        info_gain = np.log2(max(1, prev_candidate_count) / new_count)
        reward += 0.5 * info_gain
        if solved:
            # Progressive bonus: solving faster = bigger reward
            guesses_saved = MAX_GUESSES - self.guesses_used
            reward += 10.0 + 2.0 * guesses_saved
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
    mask = env.valid_action_mask()
    print("valid actions:", mask.sum())
    wf, wi = env.valid_word_features(mask)
    print("word features shape:", wf.shape)
    obs, r, done, info = env.step(env.word_to_action["raise"])
    print(info, "reward:", r)
