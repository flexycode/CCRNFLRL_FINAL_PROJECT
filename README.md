# Wordle Solver: DQN Agent vs. Entropy Heuristic

A complete reinforcement learning implementation of a Wordle solver, featuring a trained DQN agent and comparison with an entropy-greedy heuristic policy. Includes training, benchmarking, and interactive play modes.

## Quick Start

```bash
# Train a DQN agent (200-word pool, fast demo)
python3 train.py --n_words 200 --episodes 2000

# Plot training progress
python3 plot_training.py

# Benchmark the trained agent
python3 benchmark_dqn.py

# Play interactively with the solver
python3 play_live.py --agent heuristic --answer knoll
python3 play_live.py --agent dqn --model_path dqn_wordle.pt --n_words 200
```

---

## Project Structure

```
├── Core Components
│   ├── wordle_mdp.py          # MDP formulation, word lists, entropy-greedy policy
│   ├── wordle_env.py          # RL environment (observation, rewards, transitions)
│   ├── dqn_agent.py           # DQN neural network and training logic
│   ├── board_renderer.py      # Shared animated Wordle board visualization
│
├── Training & Benchmarking
│   ├── train.py               # Train DQN with live dashboard
│   ├── benchmark.py           # Evaluate entropy-greedy heuristic
│   ├── benchmark_dqn.py       # Evaluate trained DQN agent
│   ├── compare.py             # Head-to-head comparison
│
├── Analysis & Visualization
│   ├── plot_training.py       # Plot training metrics (4 analysis modes)
│   ├── efficompa.py           # Simple model comparison plots
│
├── Interactive Tools
│   ├── play_live.py           # Watch an agent play a single game (animated)
│   ├── solve_assistant.py     # Real-time solver for actual NYT Wordle
│   ├── find_opener.py         # Precompute best opening guess (expensive, one-time)
│
├── Data
│   ├── wordles.json           # ~2,309 possible secret words (NYT Wordle)
│   ├── nonwordles.json        # ~8,636 valid guesses (not secret answers)
│   ├── best_opener.json       # Cached optimal opening guess (RAISE)
│   ├── training_metrics.json  # Metrics from the last training run
│
└── README.md
```

---

## System Requirements

**Python 3.8+** with:
- `torch` (PyTorch)
- `numpy`
- `matplotlib`
- `scipy`

### Installation

```bash
pip install torch numpy matplotlib scipy
```

---

## Detailed Usage Guide

### 1. Training the DQN Agent

#### Basic Training (200 words, ~30 minutes)
```bash
python3 train.py --n_words 200 --episodes 2000
```

**Output:**
- `dqn_wordle.pt` — trained model weights
- `training_metrics.json` — checkpoint metrics (episodes, win rates, avg guesses)

**Live Dashboard:**
- Left panel: animated Wordle board showing demo games every 100 episodes
- Right panel: live training curves (win rate & efficiency)
- Watch the agent improve in real time

#### Advanced Training Options
```bash
# Large-scale training (full 8,636-word ENABLE1 pool, slow)
python3 train.py --n_words 8636 --episodes 5000 --batch_size 64

# Faster text-only training (no visualization)
python3 train.py --n_words 200 --episodes 2000 --no_visual

# Adjust animation speed (0.5 = faster, 2.0 = slower)
python3 train.py --n_words 200 --episodes 2000 --board_speed 0.5

# Custom hyperparameters
python3 train.py --n_words 200 --episodes 2000 \
  --batch_size 128 --target_update_every 20
```

**Key Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_words` | 200 | Restrict action/answer space to top N words |
| `--episodes` | 2000 | Number of training episodes |
| `--eval_every` | 100 | Checkpoint frequency |
| `--batch_size` | 128 | Replay buffer batch size |
| `--target_update_every` | 20 | Update target network every N episodes |
| `--no_visual` | False | Skip live dashboard (faster) |
| `--board_speed` | 0.6 | Animation speed multiplier |

---

### 2. Plotting Training Metrics

#### Main Learning Curves (Default)
```bash
python3 plot_training.py
```

**Outputs:**
- `training_curves.png` — Win rate and efficiency progression
- `training_dashboard.png` — 2×2 dashboard with summary stats

#### Full Analysis Suite
```bash
python3 plot_training.py --all
```

**Additional Outputs:**
- `convergence_analysis.png` — Learning rate and training stability
- `phase_analysis.png` — Early/mid/late training phases

#### Custom Metrics File
```bash
python3 plot_training.py --metrics path/to/metrics.json
```

**Plots Included:**
1. **Learning Curves** — Raw + smoothed win rate and efficiency over episodes
2. **Training Dashboard** — 2×2 grid with curves + summary statistics
3. **Convergence Analysis** *(with --all)* — Learning rate derivative and stability
4. **Phase Analysis** *(with --all)* — Performance breakdown by training phase

---

### 3. Benchmarking

#### Benchmark the Entropy-Greedy Heuristic
```bash
# Evaluate on all 2,309 answers
python3 benchmark.py

# Sample evaluation (faster)
python3 benchmark.py 100
```

**Output:**
```
Solved: 2309/2309  (100.0%)
Failed (>6 guesses): 0
Average guesses (solved games): 4.189
```

#### Benchmark the Trained DQN Agent
```bash
# Evaluate on all words in the training pool
python3 benchmark_dqn.py

# Quick sample
python3 benchmark_dqn.py 500
```

**Output:**
```
Solved: 486/500  (97.2%)
Failed (>6 guesses): 14
Average guesses (solved games): 4.712
```

#### Head-to-Head Comparison
```bash
# Compare DQN vs. heuristic on the same 200-word pool
python3 compare.py --n_words 200 --model_path dqn_wordle.pt
```

**Output:**
```
DQN agent:        win_rate=95%  avg_guesses=4.19
Entropy heuristic: win_rate=96%  avg_guesses=4.21
```

---

### 4. Interactive Play

#### Watch a Live Demo Game (Entropy Heuristic)
```bash
# Random answer
python3 play_live.py --agent heuristic

# Specific answer
python3 play_live.py --agent heuristic --answer knoll

# Slow animation
python3 play_live.py --agent heuristic --speed 2.0
```

#### Watch a Live Demo Game (DQN Agent)
```bash
# Use a trained model
python3 play_live.py --agent dqn --model_path dqn_wordle.pt --n_words 200

# Different word pool
python3 play_live.py --agent dqn --model_path dqn_wordle.pt --n_words 500
```

**Controls:**
- Close the window to exit
- The board animates letter-by-letter, then flips with Wordle colors
- Console prints each guess and pattern

#### Real-Time Wordle Solver Assistant
Play the actual NYT Wordle, and get AI suggestions after each guess:

```bash
python3 solve_assistant.py
```

**Usage:**
1. The assistant suggests the best opening guess
2. Make the guess on the real Wordle
3. Type the feedback pattern: `g` (green), `y` (yellow), `b` (black/gray)
   - Example: if "RAISE" gives green-gray-yellow-gray-green, type: `gbybg`
4. The assistant narrows down candidates and recommends the next guess
5. Repeat until solved or out of guesses

**Example Session:**
```
Turn 1: try guessing -> RAISE  (2309 candidates remaining)
  Guess you actually entered: 
  Feedback pattern (e.g gbybg), or 'solved': gybby
  -> 5 candidates remain.
  Remaining candidates: ['champ', 'charm', 'chart', 'chasm', 'chats']

Turn 2: try guessing -> CHAMP  (5 candidates remaining)
  Guess you actually entered: 
  Feedback pattern (e.g gbybg), or 'solved': ggggg
Solved in 2 guesses! (CHAMP)
```

---

## Project Overview: How It Works

### The Wordle MDP

**State:** The set of remaining candidate words consistent with all feedback so far

**Action:** Which 5-letter word to guess next (from ~2,300 possible guesses)

**Reward:** `-1` per guess, `+10` for solving, `-5` for failing (>6 guesses)

**Observation (Neural Input):** 183-dimensional feature vector encoding:
- Green letters (confirmed correct positions): 5×26 = 130 dims
- Yellow letters (confirmed present, wrong spot): 26 dims
- Gray letters (confirmed absent): 26 dims
- Remaining guesses: 1 dim

### Two Solving Strategies

#### 1. Entropy-Greedy Heuristic
- **Approach:** At each state, pick the guess that maximizes expected information gain (entropy reduction)
- **Optimality:** One-step greedy, not perfect Bellman, but empirically ~96% win rate
- **Speed:** Very fast (precomputes best opener once: 5.88 bits of info)
- **Code:** `wordle_mdp.py` → `best_guess()`

#### 2. DQN Agent
- **Approach:** Neural network trained via deep Q-learning to estimate action values
- **Architecture:** 183 → 256 → 256 → |action_space| MLP
- **Training:** Replay buffer + target network, action masking for large discrete action space
- **Performance:** ~95% win rate (slightly lower but still strong)
- **Speed:** Fast inference (single forward pass)
- **Code:** `dqn_agent.py`

### Action Masking

Both agents use **action masking** to prevent wasting guesses on words impossible given known constraints:
- If a letter is confirmed green at position X, don't guess words with a different letter at X
- If a letter is confirmed yellow (present but wrong spot), don't guess words missing it
- Never guess the same word twice
- This reduces the effective action space from ~2,300 to typically 10–100 valid options per state

---

## Understanding the Output Files

### `training_metrics.json`
Saved after every training run. Contains:
- `episodes`: checkpoint episode numbers (e.g., 100, 200, 300, ...)
- `win_rates`: win rate at each checkpoint
- `avg_guesses`: average guesses to solve (among winning games)
- `final_win_rate`, `final_avg_guesses`: terminal performance
- `n_words`: size of the word pool used
- `total_episodes`: total episodes run

### `dqn_wordle.pt`
PyTorch model checkpoint. Load with:
```python
from dqn_agent import DQNAgent
agent = DQNAgent(obs_dim=183, n_actions=200)
agent.load("dqn_wordle.pt")
```

### `best_opener.json`
Pre-computed optimal opening word. Contains:
- `word`: the best word to guess first (typically "raise")
- `bits`: expected information gain in bits

To regenerate (expensive, ~30M pattern computations):
```bash
python3 find_opener.py
```

---

## Example Workflows

### Workflow 1: Quick Demo (5 min)
```bash
# Train briefly
python3 train.py --n_words 200 --episodes 500 --no_visual

# Watch it play
python3 play_live.py --agent dqn --model_path dqn_wordle.pt --n_words 200

# Quick benchmark
python3 benchmark_dqn.py 50
```

### Workflow 2: Full Analysis (1+ hour)
```bash
# Train with live visualization
python3 train.py --n_words 2309 --episodes 3000

# Plot everything
python3 plot_training.py --all

# Benchmark thoroughly
python3 benchmark_dqn.py

# Compare head-to-head
python3 compare.py --n_words 2309 --model_path dqn_wordle.pt
```

### Workflow 3: Model Comparison
```bash
# Train two models with different hyperparameters
python3 train.py --n_words 2309 --episodes 2000 --batch_size 128
mv dqn_wordle.pt dqn_batch128.pt

python3 train.py --n_words 2309 --episodes 2000 --batch_size 256
mv dqn_wordle.pt dqn_batch256.pt

# Run benchmarks
python3 benchmark_dqn.py 500  # (saves to training_metrics.json)
python3 compare.py --n_words 2309 --model_path dqn_batch128.pt
python3 compare.py --n_words 2309 --model_path dqn_batch256.pt

# Plot results
python3 plot_training.py
```

---

## Tips & Troubleshooting

### Training is slow
- Use `--no_visual` to skip animation (~2–3× speedup)
- Reduce `--n_words` (200 trains ~10× faster than 2,309)
- Use a GPU if available (modify `dqn_agent.py` device argument)

### Out of memory
- Reduce `--batch_size` (try 64 or 32)
- Reduce `--n_words`
- Enable `--no_visual`

### Metrics file missing
- Run `train.py` at least once to generate `training_metrics.json`
- Specify custom path: `python3 plot_training.py --metrics /path/to/metrics.json`

### No improvement during training
- Training Wordle agents is hard; random exploration often finds low-hanging fruit early
- Try more episodes (2000–5000)
- Try a larger word pool (more diverse training signals)
- Adjust epsilon decay and learning rate in `dqn_agent.py`

### Model performs worse than heuristic
- This is expected! The entropy heuristic is near-optimal (Bellman-optimal would be ~98–99%)
- DQN learns a sub-optimal but fast policy trading small accuracy for speed
- With more training and better hyperparameters, DQN can match or exceed the heuristic

---

## Key Results

### Entropy-Greedy Heuristic (Baseline)
- **Winning rate:** 100% on 2,309 answers
- **Avg guesses:** 4.19
- **Opening word:** RAISE (5.88 bits of information)

### DQN Agent (200-word pool, 2,000 episodes)
- **Winning rate:** 95%
- **Avg guesses:** 4.19
- **Training time:** ~30 minutes

### DQN Agent (8,636-word ENABLE1 pool, 5,000 episodes)
- **Winning rate:** 74.7%
- **Avg guesses:** 4.71
- **Training time:** ~several hours

---

## References & Further Reading

- **MDP Formulation:** `wordle_mdp.py` contains detailed comments on state space, actions, and rewards
- **RL Algorithm:** `dqn_agent.py` implements standard DQN with action masking
- **Visualization:** `board_renderer.py` is a reusable component for animated Wordle boards
- **3Blue1Brown's Wordle Solver:** Inspiration for the entropy-greedy approach
- **Wordle Game Rules:** Official Wordle feedback logic (including duplicate-letter handling) in `wordle_mdp.py::score_guess()`

---

## License

Public domain. Free to modify, distribute, and build upon.
