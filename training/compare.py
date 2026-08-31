"""
Head-to-head: trained DQN agent vs. the entropy-greedy heuristic policy,
on the same word pool.
"""
import argparse
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.wordle_env import WordleEnv, OBS_DIM
from core.wordle_mdp import load_word_lists, best_guess, filter_candidates
from core.dqn_agent import DQNAgent


def run_dqn(env, agent, word_pool):
    wins, total_guesses = 0, 0
    for answer in word_pool:
        obs = env.reset(answer=answer)
        done = False
        steps = 0
        while not done:
            mask = env.valid_action_mask()
            action = agent.act(obs, mask, epsilon=0.0)
            obs, reward, done, info = env.step(action)
            steps += 1
        if info["solved"]:
            wins += 1
            total_guesses += steps
    return wins / len(word_pool), total_guesses / max(1, wins)


def run_heuristic(word_pool, full_answer_pool, guess_pool):
    wins, total_guesses = 0, 0
    for answer in word_pool:
        candidates = list(full_answer_pool)
        for turn in range(1, 7):
            guess = best_guess(candidates, guess_pool)
            from core.wordle_mdp import score_guess
            pattern = score_guess(guess, answer)
            if pattern == (2, 2, 2, 2, 2):
                wins += 1
                total_guesses += turn
                break
            candidates = filter_candidates(candidates, guess, pattern)
        else:
            pass
    return wins / len(word_pool), total_guesses / max(1, wins)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_words", type=int, default=200)
    parser.add_argument("--model_path", type=str, default=str(Path(__file__).parent.parent / "data" / "dqn_wordle.pt"))
    args = parser.parse_args()

    answers, guesses = load_word_lists()
    word_pool = answers[:args.n_words]

    env = WordleEnv(answer_pool=word_pool, action_words=word_pool, seed=0)
    agent = DQNAgent(obs_dim=OBS_DIM, n_actions=len(word_pool))
    agent.load(args.model_path)

    dqn_win_rate, dqn_avg = run_dqn(env, agent, word_pool)
    print(f"DQN agent:        win_rate={dqn_win_rate:.2%}  avg_guesses={dqn_avg:.2f}")

    # heuristic restricted to the SAME word_pool as its answer/guess universe,
    # for a fair apples-to-apples comparison against the DQN's action space
    heur_win_rate, heur_avg = run_heuristic(word_pool, word_pool, word_pool)
    print(f"Entropy heuristic: win_rate={heur_win_rate:.2%}  avg_guesses={heur_avg:.2f}")


if __name__ == "__main__":
    main()
