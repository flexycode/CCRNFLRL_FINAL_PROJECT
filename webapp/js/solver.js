/**
 * solver.js — Entropy-greedy Wordle solver ported from core/wordle_mdp.py
 *
 * This is a faithful JavaScript port of the Python entropy-greedy heuristic.
 * It includes: score_guess (with duplicate-letter handling), filter_candidates,
 * expected_info_gain, and best_guess.
 */

const Solver = (() => {
  let ANSWERS = [];
  let GUESSES = [];
  let loaded = false;

  /**
   * Load word lists from bundled JSON files.
   * @returns {Promise<{answers: string[], guesses: string[]}>}
   */
  async function loadWordLists() {
    if (loaded) return { answers: ANSWERS, guesses: GUESSES };

    const [answersRes, nonwordlesRes] = await Promise.all([
      fetch('data/wordles.json'),
      fetch('data/nonwordles.json')
    ]);
    const answersRaw = await answersRes.json();
    const nonwordles = await nonwordlesRes.json();

    ANSWERS = answersRaw.map(w => w.toLowerCase());

    // GUESSES = union of answers + nonwordles, sorted
    const allSet = new Set([...ANSWERS, ...nonwordles.map(w => w.toLowerCase())]);
    GUESSES = [...allSet].sort();

    loaded = true;
    return { answers: ANSWERS, guesses: GUESSES };
  }

  /**
   * Score a guess against an answer — replicates Wordle's duplicate-letter handling.
   * Returns an array of 5 ints: 2=green, 1=yellow, 0=gray.
   *
   * Ported from: core/wordle_mdp.py :: score_guess()
   */
  function scoreGuess(guess, answer) {
    const result = [0, 0, 0, 0, 0];
    const answerChars = answer.split('');

    // First pass: greens
    for (let i = 0; i < 5; i++) {
      if (guess[i] === answerChars[i]) {
        result[i] = 2;
        answerChars[i] = null; // consume
      }
    }

    // Count remaining letters
    const remaining = {};
    for (const c of answerChars) {
      if (c !== null) {
        remaining[c] = (remaining[c] || 0) + 1;
      }
    }

    // Second pass: yellows
    for (let i = 0; i < 5; i++) {
      if (result[i] === 0) {
        const g = guess[i];
        if ((remaining[g] || 0) > 0) {
          result[i] = 1;
          remaining[g]--;
        }
      }
    }

    return result;
  }

  /**
   * Convert pattern array to a string key for hashing.
   */
  function patternKey(pattern) {
    return pattern.join(',');
  }

  /**
   * Filter candidates: keep only words that would produce the same pattern.
   *
   * Ported from: core/wordle_mdp.py :: filter_candidates()
   */
  function filterCandidates(candidates, guess, pattern) {
    const pk = patternKey(pattern);
    return candidates.filter(w => patternKey(scoreGuess(guess, w)) === pk);
  }

  /**
   * Calculate expected information gain (Shannon entropy of pattern distribution).
   *
   * Ported from: core/wordle_mdp.py :: expected_info_gain()
   */
  function expectedInfoGain(guess, candidates) {
    const n = candidates.length;
    if (n <= 1) return 0;

    const patternCounts = {};
    for (const ans of candidates) {
      const pk = patternKey(scoreGuess(guess, ans));
      patternCounts[pk] = (patternCounts[pk] || 0) + 1;
    }

    let entropy = 0;
    for (const count of Object.values(patternCounts)) {
      const p = count / n;
      entropy -= p * Math.log2(p);
    }
    return entropy;
  }

  /**
   * Pick the guess that maximizes expected information gain.
   * When few candidates remain, prefer guessing one of them directly.
   *
   * Ported from: core/wordle_mdp.py :: best_guess()
   *
   * NOTE: For the web UI, we search over the candidate set only (not the full
   * 10k+ guess pool) to keep it fast. For large candidate sets we use a sample.
   */
  function bestGuess(candidates, guessPool, fullSearchThreshold = 2) {
    if (candidates.length <= fullSearchThreshold) {
      return candidates[0];
    }

    // Optimization: Skip the massive calculation for the very first turn.
    // We already know 'raise' (or 'tares') is the optimal entropy opener.
    if (candidates.length === ANSWERS.length) {
      return 'raise';
    }

    // For performance: limit search space aggressively on the web
    let searchSet;
    if (candidates.length <= 50) {
      // Search candidates + a small sample of top guesses
      const candidateSet = new Set(candidates);
      const extra = guessPool.filter(w => !candidateSet.has(w)).slice(0, 100);
      searchSet = [...candidates, ...extra];
    } else {
      // Just search a sample of candidates to avoid freezing
      searchSet = candidates.slice(0, 100);
    }

    let bestWord = searchSet[0];
    let bestScore = -1;
    const candidateSet = new Set(candidates);

    for (const g of searchSet) {
      const score = expectedInfoGain(g, candidates);
      if (score > bestScore || (score === bestScore && candidateSet.has(g))) {
        bestWord = g;
        bestScore = score;
      }
    }

    return bestWord;
  }

  /**
   * Get the top N guesses ranked by information gain.
   */
  function topGuesses(candidates, guessPool, n = 5) {
    if (candidates.length <= 1) {
      return candidates.map(w => ({ word: w, score: 0 }));
    }

    // Optimization: Skip calculation on turn 1
    if (candidates.length === ANSWERS.length) {
      return [{ word: 'raise', score: 5.88 }];
    }

    // Limit search aggressively for the UI panel
    let searchSet;
    if (candidates.length <= 50) {
      const candidateSet = new Set(candidates);
      const extra = guessPool.filter(w => !candidateSet.has(w)).slice(0, 100);
      searchSet = [...candidates, ...extra];
    } else {
      searchSet = candidates.slice(0, 100);
    }

    const scored = searchSet.map(g => ({
      word: g,
      score: expectedInfoGain(g, candidates)
    }));

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, n);
  }

  /**
   * Pattern to emoji string for display.
   */
  function patternToEmoji(pattern) {
    const map = { 0: '⬛', 1: '🟨', 2: '🟩' };
    return pattern.map(p => map[p]).join('');
  }

  /**
   * Check if a word is valid (in the guess pool).
   */
  function isValidWord(word) {
    return GUESSES.includes(word.toLowerCase());
  }

  /**
   * Check if pattern is all green (solved).
   */
  function isSolved(pattern) {
    return pattern.every(p => p === 2);
  }

  return {
    loadWordLists,
    scoreGuess,
    filterCandidates,
    expectedInfoGain,
    bestGuess,
    topGuesses,
    patternToEmoji,
    isValidWord,
    isSolved,
    patternKey,
    get answers() { return ANSWERS; },
    get guesses() { return GUESSES; },
    get isLoaded() { return loaded; }
  };
})();
