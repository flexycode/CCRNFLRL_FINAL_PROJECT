/**
 * game.js — Wordle game engine
 *
 * Two modes:
 *  1. PLAY mode — user types guesses manually
 *  2. AI mode — entropy-greedy solver plays with animations
 */

const Game = (() => {
  let answer = '';
  let currentRow = 0;
  let currentCol = 0;
  let currentGuess = '';
  let gameOver = false;
  let candidates = [];
  let mode = 'play'; // 'play' or 'ai'
  let aiRunning = false;
  let aiSpeed = 600; // ms between AI moves
  let keyboardState = {}; // letter -> 'correct'|'present'|'absent'

  // DOM refs
  let boardEl, keyboardEl, messageEl, statsEl;

  function init(boardElement, keyboardElement, messageElement, statsElement) {
    boardEl = boardElement;
    keyboardEl = keyboardElement;
    messageEl = messageElement;
    statsEl = statsElement;
  }

  async function newGame(selectedAnswer = null) {
    const { answers } = await Solver.loadWordLists();
    answer = selectedAnswer || answers[Math.floor(Math.random() * answers.length)];
    currentRow = 0;
    currentCol = 0;
    currentGuess = '';
    gameOver = false;
    aiRunning = false;
    candidates = [...answers];
    keyboardState = {};
    mode = 'play';

    Board.create(boardEl);
    createKeyboard();
    clearMessage();
    updateStats();
  }

  function createKeyboard() {
    if (!keyboardEl) return;
    keyboardEl.innerHTML = '';
    const rows = [
      'qwertyuiop',
      'asdfghjkl',
      'zxcvbnm'
    ];

    rows.forEach((row, ri) => {
      const rowEl = document.createElement('div');
      rowEl.className = 'keyboard-row';

      if (ri === 2) {
        const enter = document.createElement('button');
        enter.className = 'key wide';
        enter.textContent = 'Enter';
        enter.addEventListener('click', () => handleKey('Enter'));
        rowEl.appendChild(enter);
      }

      for (const ch of row) {
        const key = document.createElement('button');
        key.className = 'key';
        key.textContent = ch;
        key.dataset.key = ch;
        key.addEventListener('click', () => handleKey(ch));
        rowEl.appendChild(key);
      }

      if (ri === 2) {
        const back = document.createElement('button');
        back.className = 'key wide';
        back.textContent = '⌫';
        back.addEventListener('click', () => handleKey('Backspace'));
        rowEl.appendChild(back);
      }

      keyboardEl.appendChild(rowEl);
    });
  }

  function handleKey(key) {
    if (gameOver || aiRunning) return;

    if (key === 'Enter') {
      submitGuess();
    } else if (key === 'Backspace') {
      if (currentCol > 0) {
        currentCol--;
        currentGuess = currentGuess.slice(0, -1);
        Board.setLetter(boardEl, currentRow, currentCol, '');
      }
    } else if (/^[a-z]$/i.test(key) && currentCol < 5) {
      const letter = key.toLowerCase();
      Board.setLetter(boardEl, currentRow, currentCol, letter.toUpperCase());
      currentGuess += letter;
      currentCol++;
    }
  }

  async function submitGuess() {
    if (currentGuess.length !== 5) {
      showToast('Not enough letters');
      return;
    }

    if (!Solver.isValidWord(currentGuess)) {
      showToast('Not in word list');
      return;
    }

    const pattern = Solver.scoreGuess(currentGuess, answer);
    await Board.revealRow(boardEl, currentRow, currentGuess, pattern);
    updateKeyboard(currentGuess, pattern);

    // Update candidates
    candidates = Solver.filterCandidates(candidates, currentGuess, pattern);

    if (Solver.isSolved(pattern)) {
      gameOver = true;
      showMessage(`🎉 Brilliant! Solved in ${currentRow + 1} guess${currentRow > 0 ? 'es' : ''}!`, 'win');
      saveStats(true, currentRow + 1);
    } else if (currentRow >= 5) {
      gameOver = true;
      showMessage(`The word was ${answer.toUpperCase()}`, 'lose');
      saveStats(false, 7);
    }

    currentRow++;
    currentCol = 0;
    currentGuess = '';
    updateStats();
  }

  function updateKeyboard(guess, pattern) {
    const priority = { 'correct': 3, 'present': 2, 'absent': 1 };
    const classes = ['absent', 'present', 'correct'];

    for (let i = 0; i < 5; i++) {
      const letter = guess[i];
      const cls = classes[pattern[i]];
      const current = keyboardState[letter];
      if (!current || priority[cls] > priority[current]) {
        keyboardState[letter] = cls;
      }
    }

    // Apply to keyboard
    if (keyboardEl) {
      keyboardEl.querySelectorAll('.key[data-key]').forEach(key => {
        const letter = key.dataset.key;
        if (keyboardState[letter]) {
          key.className = `key ${keyboardState[letter]}`;
        }
      });
    }
  }

  function showMessage(text, type = '') {
    if (!messageEl) return;
    messageEl.textContent = text;
    messageEl.className = `game-message ${type}`;
    messageEl.classList.remove('hidden');
  }

  function clearMessage() {
    if (!messageEl) return;
    messageEl.textContent = '';
    messageEl.classList.add('hidden');
  }

  function updateStats() {
    if (!statsEl) return;
    const candidateCount = statsEl.querySelector('#candidate-count');
    const turnCount = statsEl.querySelector('#turn-count');
    const topGuessEl = statsEl.querySelector('#top-guess');
    const infoGainEl = statsEl.querySelector('#info-gain');

    if (candidateCount) candidateCount.textContent = candidates.length.toLocaleString();
    if (turnCount) turnCount.textContent = `${currentRow} / 6`;

    if (candidates.length > 0 && candidates.length <= 500 && !gameOver) {
      const top = Solver.topGuesses(candidates, Solver.guesses, 1);
      if (top.length > 0) {
        if (topGuessEl) topGuessEl.textContent = top[0].word.toUpperCase();
        if (infoGainEl) infoGainEl.textContent = top[0].score.toFixed(2) + ' bits';
      }
    } else {
      if (topGuessEl) topGuessEl.textContent = gameOver ? '—' : 'Computing...';
      if (infoGainEl) infoGainEl.textContent = gameOver ? '—' : '...';
    }
  }

  // ── AI Solve Mode ──

  async function aiSolve(speed = null) {
    if (aiRunning || gameOver) return;
    mode = 'ai';
    aiRunning = true;
    const s = speed || aiSpeed;
    const { answers, guesses } = await Solver.loadWordLists();

    let aiCandidates = [...answers];

    for (let row = currentRow; row < 6; row++) {
      if (!aiRunning) break;

      const guess = Solver.bestGuess(aiCandidates, guesses);
      const pattern = Solver.scoreGuess(guess, answer);

      // Show the guess being typed
      Board.setRow(boardEl, row, guess.toUpperCase().split(''));
      await sleep(s * 0.5);

      // Reveal with flip animation
      await Board.revealRow(boardEl, row, guess, pattern, Math.max(100, s * 0.3));
      updateKeyboard(guess, pattern);

      aiCandidates = Solver.filterCandidates(aiCandidates, guess, pattern);
      candidates = aiCandidates;
      currentRow = row + 1;
      updateStats();

      if (Solver.isSolved(pattern)) {
        gameOver = true;
        aiRunning = false;
        showMessage(`🤖 AI solved in ${row + 1} guess${row > 0 ? 'es' : ''}!`, 'win');
        return;
      }

      await sleep(s);
    }

    if (!gameOver) {
      gameOver = true;
      aiRunning = false;
      showMessage(`🤖 AI failed. The word was ${answer.toUpperCase()}`, 'lose');
    }
  }

  function stopAI() {
    aiRunning = false;
  }

  function setAISpeed(speed) {
    aiSpeed = speed;
  }

  // ── Stats persistence (localStorage) ──

  function saveStats(won, guesses) {
    try {
      const stats = JSON.parse(localStorage.getItem('wordle-rl-stats') || '{}');
      stats.played = (stats.played || 0) + 1;
      stats.won = (stats.won || 0) + (won ? 1 : 0);
      if (!stats.distribution) stats.distribution = {};
      const key = won ? String(guesses) : 'X';
      stats.distribution[key] = (stats.distribution[key] || 0) + 1;
      localStorage.setItem('wordle-rl-stats', JSON.stringify(stats));
    } catch (e) { /* ignore storage errors */ }
  }

  function getStats() {
    try {
      return JSON.parse(localStorage.getItem('wordle-rl-stats') || '{"played":0,"won":0,"distribution":{}}');
    } catch (e) {
      return { played: 0, won: 0, distribution: {} };
    }
  }

  // ── Physical keyboard support ──

  function onKeyDown(e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === 'Enter') {
      e.preventDefault();
      handleKey('Enter');
    } else if (e.key === 'Backspace') {
      e.preventDefault();
      handleKey('Backspace');
    } else if (/^[a-zA-Z]$/.test(e.key)) {
      handleKey(e.key.toLowerCase());
    }
  }

  function enableKeyboard() {
    document.addEventListener('keydown', onKeyDown);
  }

  function disableKeyboard() {
    document.removeEventListener('keydown', onKeyDown);
  }

  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  return {
    init,
    newGame,
    handleKey,
    aiSolve,
    stopAI,
    setAISpeed,
    enableKeyboard,
    disableKeyboard,
    getStats,
    get answer() { return answer; },
    get isGameOver() { return gameOver; },
    get isAIRunning() { return aiRunning; },
    get currentMode() { return mode; }
  };
})();
