"""
Train the DQN agent on WordleEnv, with a live combined visual:
  - LEFT:  an animated Wordle board that plays one live demo game (using
    the agent's CURRENT weights, greedy/no-exploration) every eval_every
    episodes - so you actually watch it get better over time.
  - RIGHT: win-rate and avg-guesses-to-solve curves, updated at the same
    checkpoints.

Usage:
    python3 train.py --n_words 200 --episodes 2000
    python3 train.py --n_words 200 --episodes 2000 --no_visual   # fast, text-only
"""
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.wordle_env import WordleEnv, OBS_DIM
from core.wordle_mdp import load_word_lists
from core.dqn_agent import DQNAgent
from core.board_renderer import WordleBoard


def evaluate(env, agent, n_eval=50, epsilon=0.0):
    wins, total_guesses = 0, 0
    for i in range(n_eval):
        answer = env.answer_pool[i % len(env.answer_pool)]
        obs = env.reset(answer=answer)
        done = False
        steps = 0
        while not done:
            mask = env.valid_action_mask()
            action = agent.act(obs, mask, epsilon=epsilon)
            obs, reward, done, info = env.step(action)
            steps += 1
        if info["solved"]:
            wins += 1
            total_guesses += steps
    win_rate = wins / n_eval
    avg_guesses = total_guesses / max(1, wins)
    return win_rate, avg_guesses


def play_demo_game(env, agent, board, fig, answer=None):
    """Plays ONE live-animated game with the agent's current weights
    (greedy, no exploration) on the shared board, then reports the result."""
    board.reset()
    obs = env.reset(answer=answer)
    done = False
    row = 0
    solved = False
    while not done:
        mask = env.valid_action_mask()
        action = agent.act(obs, mask, epsilon=0.0)
        obs, reward, done, info = env.step(action)
        board.reveal_guess(fig.canvas, row, info["guess"], info["pattern"], animate=True)
        row += 1
        solved = info["solved"]
    return solved, row


class TrainingDashboard:
    """Combined figure: live demo board (left) + win-rate/avg-guesses charts (right)."""

    def __init__(self, max_guesses=6, board_speed=1.0):
        plt.ion()
        self.fig = plt.figure(figsize=(11, 6))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1, 1.4])

        board_ax = self.fig.add_subplot(gs[:, 0])
        board_ax.set_title("Live demo game (current weights)", fontsize=11, fontweight="bold")
        self.board = WordleBoard(board_ax, max_guesses=max_guesses, speed=board_speed)

        self.ax1 = self.fig.add_subplot(gs[0, 1])
        self.ax2 = self.fig.add_subplot(gs[1, 1], sharex=self.ax1)

        (self.line1,) = self.ax1.plot([], [], "o-", color="#6aaa64")
        self.ax1.set_ylabel("Win rate")
        self.ax1.set_ylim(-0.05, 1.05)
        self.ax1.set_title("Training progress", fontsize=11, fontweight="bold")
        self.ax1.grid(alpha=0.3)

        (self.line2,) = self.ax2.plot([], [], "o-", color="#c9b458")
        self.ax2.set_ylabel("Avg guesses (wins)")
        self.ax2.set_xlabel("Episode")
        self.ax2.set_ylim(0, 6.5)
        self.ax2.grid(alpha=0.3)

        self.status = self.fig.text(0.25, 0.02, "", ha="center", fontsize=10)
        self.episodes, self.win_rates, self.avg_guesses = [], [], []

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.3)

    def checkpoint(self, env, agent, episode, win_rate, avg_guesses, demo_answer=None):
        # update charts
        self.episodes.append(episode)
        self.win_rates.append(win_rate)
        self.avg_guesses.append(avg_guesses)
        self.line1.set_data(self.episodes, self.win_rates)
        self.line2.set_data(self.episodes, self.avg_guesses)
        self.ax1.set_xlim(0, max(self.episodes) * 1.05)
        self.ax2.set_xlim(0, max(self.episodes) * 1.05)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # play one live demo game with current weights
        solved, n_guesses = play_demo_game(env, agent, self.board, self.fig, answer=demo_answer)
        msg = f"Ep {episode}: solved in {n_guesses}" if solved else f"Ep {episode}: failed"
        self.status.set_text(msg)
        self.status.set_color("#2e7d32" if solved else "#c62828")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def finish(self):
        plt.ioff()
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_words", type=int, default=200,
                         help="Restrict action/answer space to this many words (for tractable demo training)")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--target_update_every", type=int, default=20)
    parser.add_argument("--no_visual", action="store_true",
                         help="Disable the live board/chart window (faster, text-only, like the old train.py)")
    parser.add_argument("--board_speed", type=float, default=0.6,
                         help="Animation speed multiplier for the live demo board")
    args = parser.parse_args()

    answers, _ = load_word_lists()
    word_pool = answers[:args.n_words]
    print(f"Training on {len(word_pool)} words (action space size = {len(word_pool)})")

    env = WordleEnv(answer_pool=word_pool, action_words=word_pool, seed=0)
    agent = DQNAgent(obs_dim=OBS_DIM, n_actions=len(word_pool))

    dashboard = None if args.no_visual else TrainingDashboard(board_speed=args.board_speed)

    epsilon_start, epsilon_end, epsilon_decay_episodes = 1.0, 0.05, int(args.episodes * 0.7)

    start = time.time()
    for ep in range(1, args.episodes + 1):
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * ep / epsilon_decay_episodes)

        obs = env.reset()
        done = False
        while not done:
            mask = env.valid_action_mask()
            action = agent.act(obs, mask, epsilon=epsilon)
            next_obs, reward, done, info = env.step(action)
            next_mask = env.valid_action_mask() if not done else np.zeros(len(word_pool), dtype=bool)
            agent.remember(obs, action, reward, next_obs, done, next_mask)
            agent.train_step(batch_size=args.batch_size)
            obs = next_obs

        if ep % args.target_update_every == 0:
            agent.update_target()

        if ep % args.eval_every == 0:
            win_rate, avg_guesses = evaluate(env, agent, n_eval=min(50, len(word_pool)))
            elapsed = time.time() - start
            print(f"Episode {ep:5d}  eps={epsilon:.2f}  "
                  f"eval_win_rate={win_rate:.2f}  avg_guesses={avg_guesses:.2f}  "
                  f"({elapsed:.0f}s elapsed)")
            if dashboard:
                demo_word = word_pool[(ep // args.eval_every) % len(word_pool)]
                dashboard.checkpoint(env, agent, ep, win_rate, avg_guesses, demo_answer=demo_word)

    agent.save(str(Path(__file__).parent.parent / "data" / "dqn_wordle.pt"))
    print("\nSaved trained model to data/dqn_wordle.pt")

    win_rate, avg_guesses = evaluate(env, agent, n_eval=len(word_pool), epsilon=0.0)
    print(f"\nFinal eval over all {len(word_pool)} training words: "
          f"win_rate={win_rate:.2f}  avg_guesses={avg_guesses:.2f}")

    # Save training metrics for later plotting
    metrics = {
        "episodes": dashboard.episodes if dashboard else [],
        "win_rates": dashboard.win_rates if dashboard else [],
        "avg_guesses": dashboard.avg_guesses if dashboard else [],
        "final_win_rate": float(win_rate),
        "final_avg_guesses": float(avg_guesses),
        "n_words": len(word_pool),
        "total_episodes": args.episodes,
    }
    import json
    with open(str(Path(__file__).parent.parent / "data" / "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved training metrics to data/training_metrics.json")

    if dashboard:
        print("Close the window to exit.")
        dashboard.finish()


if __name__ == "__main__":
    main()