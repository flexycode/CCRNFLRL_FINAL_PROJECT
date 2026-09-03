"""
Watch an agent play a single animated Wordle game.

Supports both the entropy heuristic and the trained DQN v2 agent.

Usage:
    python tools/play_live.py --agent heuristic --answer knoll
    python tools/play_live.py --agent dqn --model_path data/dqn_wordle_v2.pt --n_words 200
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from core.board_renderer import WordleBoard
from core.wordle_mdp import (load_word_lists, best_guess, filter_candidates,
                              score_guess, pattern_to_str)
from core.wordle_env import WordleEnv, OBS_DIM, WORD_FEAT_DIM


def play_heuristic(answer, board, fig, answers, guesses, speed):
    """Play a game using the entropy-greedy heuristic."""
    candidates = list(answers)
    for row in range(6):
        guess = best_guess(candidates, guesses)
        pattern = score_guess(guess, answer)
        board.reveal_guess(fig.canvas, row, guess, pattern, animate=True)
        print(f"  Turn {row+1}: {guess}  {pattern_to_str(pattern)}  ({len(candidates)} candidates)")
        if pattern == (2, 2, 2, 2, 2):
            print(f"  [+] Solved in {row+1} guesses!")
            return True
        candidates = filter_candidates(candidates, guess, pattern)
    print("  [x] Failed to solve in 6 guesses")
    return False


def play_dqn(answer, board, fig, word_pool, model_path, speed):
    """Play a game using the trained DQN v2 agent."""
    from core.dqn_agent import DQNAgent
    
    env = WordleEnv(answer_pool=word_pool, action_words=word_pool)
    agent = DQNAgent(obs_dim=OBS_DIM, word_dim=WORD_FEAT_DIM)
    agent.load(model_path)
    
    obs = env.reset(answer=answer)
    done = False
    row = 0
    
    while not done and row < 6:
        mask = env.valid_action_mask()
        word_feats, word_indices = env.valid_word_features(mask)
        action_local = agent.act(obs, word_feats, epsilon=0.0)
        action_idx = int(word_indices[action_local])
        
        obs, reward, done, info = env.step(action_idx)
        board.reveal_guess(fig.canvas, row, info["guess"], info["pattern"], animate=True)
        print(f"  Turn {row+1}: {info['guess']}  {pattern_to_str(info['pattern'])}  "
              f"({info['candidates_remaining']} candidates remaining)")
        
        if info["solved"]:
            print(f"  [+] Solved in {row+1} guesses!")
            return True
        row += 1
    
    if not info.get("solved"):
        print("  [x] Failed to solve in 6 guesses")
    return info.get("solved", False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["heuristic", "dqn"], default="heuristic")
    parser.add_argument("--answer", type=str, default=None)
    parser.add_argument("--model_path", type=str,
                         default=str(Path(__file__).parent.parent / "data" / "dqn_wordle_v2.pt"))
    parser.add_argument("--n_words", type=int, default=200)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    answers, guesses = load_word_lists()
    
    if args.answer:
        answer = args.answer.lower()
    else:
        import random
        answer = random.choice(answers[:args.n_words] if args.agent == "dqn" else answers)
    
    print(f"\n{'='*40}")
    print(f"Agent: {args.agent.upper()}")
    print(f"Answer: {answer}")
    print(f"{'='*40}\n")

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.set_title(f"{'DQN v2' if args.agent == 'dqn' else 'Entropy Heuristic'} — playing '{answer}'",
                 fontsize=12, fontweight='bold')
    board = WordleBoard(ax, speed=args.speed)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)

    if args.agent == "heuristic":
        play_heuristic(answer, board, fig, answers, guesses, args.speed)
    else:
        word_pool = answers[:args.n_words]
        play_dqn(answer, board, fig, word_pool, args.model_path, args.speed)

    plt.show()


if __name__ == "__main__":
    main()
