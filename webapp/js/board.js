/**
 * board.js — Animated Wordle board renderer
 *
 * Handles tile creation, flip animations, and keyboard state.
 */

const Board = (() => {
  const ROWS = 6;
  const COLS = 5;

  function create(boardEl) {
    boardEl.innerHTML = '';
    for (let r = 0; r < ROWS; r++) {
      const row = document.createElement('div');
      row.className = 'board-row';
      row.dataset.row = r;
      for (let c = 0; c < COLS; c++) {
        const tile = document.createElement('div');
        tile.className = 'board-tile';
        tile.dataset.row = r;
        tile.dataset.col = c;
        row.appendChild(tile);
      }
      boardEl.appendChild(row);
    }
  }

  function setLetter(boardEl, row, col, letter) {
    const tile = boardEl.querySelector(
      `.board-row[data-row="${row}"] .board-tile[data-col="${col}"]`
    );
    if (!tile) return;
    tile.textContent = letter || '';
    if (letter) {
      tile.classList.add('filled');
    } else {
      tile.classList.remove('filled');
    }
  }

  function setRow(boardEl, row, word) {
    for (let c = 0; c < COLS; c++) {
      setLetter(boardEl, row, c, word[c] || '');
    }
  }

  /**
   * Reveal a row with flip animation.
   * @param {Element} boardEl
   * @param {number} row
   * @param {string} word
   * @param {number[]} pattern - [0-2] per letter
   * @param {number} speed - ms delay between tiles
   * @returns {Promise} resolves when animation completes
   */
  function revealRow(boardEl, row, word, pattern, speed = 300) {
    return new Promise(resolve => {
      const rowEl = boardEl.querySelector(`.board-row[data-row="${row}"]`);
      if (!rowEl) { resolve(); return; }
      const tiles = rowEl.querySelectorAll('.board-tile');
      const classes = ['absent', 'present', 'correct'];

      let revealed = 0;
      tiles.forEach((tile, i) => {
        tile.textContent = word[i].toUpperCase();
        tile.classList.add('filled');

        setTimeout(() => {
          tile.classList.add('flip');
          // Apply color at midpoint of flip
          setTimeout(() => {
            tile.classList.add(classes[pattern[i]]);
          }, 250);

          revealed++;
          if (revealed === COLS) {
            setTimeout(resolve, 300);
          }
        }, i * speed);
      });
    });
  }

  /**
   * Clear the board.
   */
  function clear(boardEl) {
    const tiles = boardEl.querySelectorAll('.board-tile');
    tiles.forEach(tile => {
      tile.textContent = '';
      tile.className = 'board-tile';
    });
  }

  return { create, setLetter, setRow, revealRow, clear, ROWS, COLS };
})();
