"""
Watch a single agent (DQN or entropy heuristic) play one Wordle game live,
animated tile-by-tile on the classic Wordle board. Uses the shared
WordleBoard renderer (board_renderer.py) - the same component train.py
uses for its live demo games during training.

Usage:
    python3 play_live.py --agent heuristic --answer knoll
    python3 play_live.py --agent dqn --model_path dqn_wordle.pt --n_words 200
    python3 play_live.py --agent dqn --model_path dqn_wordle.pt --n_words 200 --speed 0.5
"""
import argparse
import random
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.wordle_mdp import load_word_lists, score_guess, best_guess, filter_candidates
from core.board_renderer import WordleBoard


def play_heuristic(answer, answer_pool, guess_pool, max_guesses=6):
    """Generator yielding (guess, pattern) one at a time, as they happen."""
    candidates = list(answer_pool)
    for _ in range(max_guesses):
        guess = best_guess(candidates, guess_pool)
        pattern = score_guess(guess, answer)
        yield guess, pattern
        if pattern == (2, 2, 2, 2, 2):
            return
        candidates = filter_candidates(candidates, guess, pattern)


def play_dqn(answer, model_path, n_words, max_guesses=6):
    """Generator yielding (guess, pattern) one at a time, as they happen."""
    from core.wordle_env import WordleEnv, OBS_DIM
    from core.dqn_agent import DQNAgent

    answers, _ = load_word_lists()
    word_pool = answers[:n_words]

    env = WordleEnv(answer_pool=word_pool, action_words=word_pool, seed=0)
    agent = DQNAgent(obs_dim=OBS_DIM, n_actions=len(word_pool))
    agent.load(model_path)

    obs = env.reset(answer=answer)
    done = False
    while not done:
        mask = env.valid_action_mask()
        action = agent.act(obs, mask, epsilon=0.0)
        obs, reward, done, info = env.step(action)
        yield info["guess"], info["pattern"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["heuristic", "dqn"], default="heuristic")
    parser.add_argument("--answer", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=str(Path(__file__).parent.parent / "data" / "dqn_wordle.pt"))
    parser.add_argument("--n_words", type=int, default=200)
    parser.add_argument("--speed", type=float, default=1.0,
                         help="Speed multiplier for the animation. Lower = faster, higher = slower.")
    args = parser.parse_args()

    answers, guesses = load_word_lists()

    if args.agent == "heuristic":
        answer = args.answer or random.choice(answers)
        gen = play_heuristic(answer, answers, guesses)
        title = "Entropy Heuristic"
    else:
        word_pool = answers[:args.n_words]
        answer = args.answer or random.choice(word_pool)
        gen = play_dqn(answer, args.model_path, args.n_words)
        title = "DQN Agent"

    print(f"Answer: {answer}")

    plt.ion()
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    board = WordleBoard(ax, speed=args.speed)
    status_text = fig.text(0.5, 0.02, "", ha="center", fontsize=11)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.3)

    solved = False
    n_guesses = 0
    for row, (guess, pattern) in enumerate(gen):
        n_guesses += 1
        print(f"  {guess} -> {pattern}")
        board.reveal_guess(fig.canvas, row, guess, pattern, animate=True)
        if pattern == (2, 2, 2, 2, 2):
            solved = True
            break

    if solved:
        status_text.set_text(f"Solved in {n_guesses} guesses!")
        status_text.set_color("#2e7d32")
    else:
        status_text.set_text(f"Failed (answer: {answer.upper()})")
        status_text.set_color("#c62828")
    fig.canvas.draw()

    print("Close the window to exit.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
