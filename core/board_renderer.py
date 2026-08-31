"""
Shared, reusable animated Wordle board component.

Used by both train.py (to show periodic live demo games during training)
and play_live.py (to watch a single standalone game). Extracted here so
there's one source of truth for how the board looks and animates.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

COLORS = {
    2: "#6aaa64",  # green
    1: "#c9b458",  # yellow
    0: "#787c7e",  # gray
}
EMPTY_BORDER = "#d3d6da"
PENDING_BORDER = "#878a8c"


class WordleBoard:
    """An animated 6x5 Wordle grid, drawn onto a given matplotlib Axes.

    Call reveal_guess(row, guess, pattern) to animate one guess appearing
    and flipping to its colors. Call reset() to clear the board for a new
    game without recreating the figure/axes (important for reuse during
    training, where we don't want to pop up a new window every checkpoint).
    """

    def __init__(self, ax, max_guesses=6, speed=1.0):
        self.ax = ax
        self.max_guesses = max_guesses
        self.speed = speed

        self.ax.set_xlim(0, 5)
        self.ax.set_ylim(0, max_guesses)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        self.rects = {}
        self.texts = {}
        for row in range(max_guesses):
            for col in range(5):
                y = max_guesses - 1 - row
                rect = Rectangle((col, y), 0.92, 0.92, facecolor="white",
                                  edgecolor=EMPTY_BORDER, linewidth=2)
                self.ax.add_patch(rect)
                text = self.ax.text(col + 0.46, y + 0.46, "", ha="center", va="center",
                                     fontsize=18, fontweight="bold", color="black")
                self.rects[(row, col)] = rect
                self.texts[(row, col)] = text

    def reset(self):
        for row in range(self.max_guesses):
            for col in range(5):
                self.rects[(row, col)].set_facecolor("white")
                self.rects[(row, col)].set_edgecolor(EMPTY_BORDER)
                self.texts[(row, col)].set_text("")
                self.texts[(row, col)].set_color("black")

    def reveal_guess(self, canvas, row, guess, pattern, animate=True):
        """canvas = the figure's canvas, passed in so callers control when
        draw()/flush_events() happen (avoids each board fighting over the
        same figure when there are multiple subplots)."""
        pause = plt.pause if animate else (lambda _: None)

        for col in range(5):
            self.texts[(row, col)].set_text(guess[col].upper())
            self.rects[(row, col)].set_edgecolor(PENDING_BORDER)
            if animate:
                canvas.draw()
                canvas.flush_events()
                pause(self.speed * 0.12)

        for col in range(5):
            color = COLORS[pattern[col]]
            self.rects[(row, col)].set_facecolor(color)
            self.rects[(row, col)].set_edgecolor(color)
            self.texts[(row, col)].set_color("white")
            if animate:
                canvas.draw()
                canvas.flush_events()
                pause(self.speed * 0.2)

        if not animate:
            canvas.draw()
            canvas.flush_events()
