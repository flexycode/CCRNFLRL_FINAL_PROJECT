<div align="center">
  <img src="assets/wordle_banner.jpg" alt="Wordle Solver: Embedding DQN Agent vs. Entropy Heuristic Banner" width="100%" style="border-radius: 8px;">
</div>

<br>

# Wordle Solver: Embedding DQN Agent vs. Entropy Heuristic (v2)
#### *Paper: Reinforcement Learning for Wordle: A DQN-Based Decision Support System for Optimal Word Guessing*

A complete reinforcement learning implementation of a Wordle solver, featuring a trained **embedding-based DQN agent** that can generalize to unseen words, alongside an entropy-greedy heuristic baseline. Includes training, benchmarking, out-of-distribution testing, and interactive play modes.

## What's New in v2

- **Embedding-based DQN**: Scores `(state, word)` pairs instead of fixed word indices — generalizes to words outside the training pool
- **3–6× faster training**: Vectorized action masking, headless mode, batched scoring
- **Curriculum learning**: Start small, progressively expand word pool
- **Auto-generated results**: Training automatically saves result images to `assets/`
- **Out-of-distribution testing**: Benchmark the agent on words it has never seen
- **Double DQN + Prioritized Replay**: Improved learning stability and sample efficiency

## Quick Start

```bash
# Install dependencies
pip install torch numpy matplotlib scipy tqdm

# Train a DQN agent (200-word pool, ~5 minutes)
python training/train.py --n_words 200 --episodes 2000

# View auto-generated results
dir assets\

# Benchmark the trained agent
python training/benchmark_dqn.py --n_words 200

# Test generalization on unseen words
python training/benchmark_dqn.py --n_words 200 --ood

# Play interactively with the solver
python tools/play_live.py --agent heuristic --answer knoll
python tools/play_live.py --agent dqn --model_path data/dqn_wordle_v2.pt --n_words 200
```

---

## Project Structure

```
├── Core Components
│   ├── core/
│   │   ├── wordle_mdp.py          # MDP formulation, word lists, entropy-greedy policy
│   │   ├── wordle_env.py          # RL environment (vectorized masking, word features)
│   │   ├── dqn_agent.py           # Embedding DQN network + Double DQN + PER
│   │   ├── board_renderer.py      # Shared animated Wordle board visualization
│   │
│   ├── Training & Benchmarking
│   │   ├── training/
│   │   │   ├── train.py           # Train DQN with tqdm progress (headless default)
│   │   │   ├── benchmark_dqn.py   # Evaluate DQN agent (in-dist + OOD)
│   │   │   ├── benchmark.py       # Evaluate entropy-greedy heuristic
│   │   │   ├── compare.py         # Head-to-head comparison
│   │
│   ├── Analysis & Visualization
│   │   ├── analysis/
│   │   │   ├── plot_training.py   # Plot training metrics (4 analysis modes)
│   │   │   ├── generate_results.py # One-click result generator
│   │   │   ├── efficompa.py       # Simple model comparison plots
│   │
│   ├── Interactive Tools
│   │   ├── tools/
│   │   │   ├── play_live.py       # Watch an agent play (animated)
│   │   │   ├── solve_assistant.py # Real-time solver for actual NYT Wordle
│   │   │   ├── find_opener.py     # Precompute best opening guess
│
├── Data
│   ├── data/
│   │   ├── wordles.json           # ~2,309 possible secret words (NYT Wordle)
│   │   ├── nonwordles.json        # ~8,636 valid guesses (not secret answers)
│   │   ├── best_opener.json       # Cached optimal opening guess (RAISE)
│   │   ├── training_metrics.json  # Metrics from the last training run
│   │   ├── dqn_wordle_v2.pt       # Trained v2 model weights
│   │   ├── legacy/               # Old v1 model checkpoints
│
├── Results
│   ├── assets/                    # Auto-generated result images
│   │   ├── training_curves.png
│   │   ├── training_dashboard.png
│   │   ├── benchmark_results.png
│   │   ├── comparison_chart.png
│
├── Documentation
│   ├── docs/
│   │   ├── timeline.md            # 1-month project timeline
│   │   ├── coverage.md            # Implementation coverage matrix
│   │   ├── walkthrough.md         # Step-by-step usage guide
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
- `tqdm` (optional, has fallback)

### Installation (CPU Default)

We strongly recommend using a virtual environment.

**1. Activate the virtual environment:**
If you're using **PowerShell** (the default in VS Code):
```powershell
.\venv\Scripts\Activate.ps1
```
*(If you are using regular Command Prompt, use `venv\Scripts\activate.bat` instead)*

**2. Install the required dependencies:**
```bash
pip install torch numpy matplotlib scipy tqdm
```

> **Note on Python 3.13+:** If you encounter a `DLL load failed` error when running the application, it might be because PyTorch doesn't fully support Python 3.13 yet. If this happens, re-creating your virtual environment with a slightly older Python version (like 3.11 or 3.12) will solve the issue!

### GPU / CUDA Acceleration (Highly Recommended)
Training on the full 8,636 word dictionary for 20,000 episodes is computationally intensive. To speed up training (up to 3-6x faster):
1. Ensure you have an NVIDIA GPU.
2. Install the CUDA-enabled version of PyTorch by following the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).
3. Example installation for CUDA 11.8 or 12.1:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
4. Run the training script with `--device cuda`.

---

## Detailed Usage Guide

### 1. Training the DQN Agent

#### Basic Training (200 words, ~5 minutes)
```bash
python training/train.py --n_words 200 --episodes 2000
```

**Output:**
- `data/dqn_wordle_v2.pt` — trained model weights
- `data/training_metrics.json` — checkpoint metrics
- `assets/training_curves.png` — learning curves
- `assets/training_dashboard.png` — summary dashboard

**Progress:**
```
Using device: cpu
Starting training on 200 words (action space = 200)
Training: 100%|████████████████| 2000/2000 [04:30<00:00, wr=94% ag=4.12 ε=0.05]
```

#### Full-Scale Training Walkthrough (8,636 words, 20,000 episodes)

To fully train the v2 agent using our exact setup, we use **Curriculum Learning**. This starts the agent on a small dictionary (200 words) and progressively expands it up to the full 8,636 words as it gets smarter.

**Step 1: Launch Training**
```bash
python training/train.py --n_words 8636 --episodes 20000 --curriculum --eval_every 200 --train_every 8 --batch_size 256 --save_name dqn_wordle_v2_full.pt
```
*(Add `--device cuda` if you have a GPU set up for much faster training).*

**Step 2: Understanding the Terminal Output**
As training runs, you will see output like this:
```text
  >> Curriculum: expanded to 400 words at episode 2858
  >> Curriculum: expanded to 800 words at episode 5715
  >> Curriculum: expanded to 1600 words at episode 8572
  >> Curriculum: expanded to 3200 words at episode 11429
  >> Curriculum: expanded to 6400 words at episode 14286
  >> Curriculum: expanded to 8636 words at episode 17143

Training: 100%|##########| 20000/20000 [31:55<00:00, 10.44ep/s, wr=88% ag=4.07 ε=0.10 pool=8636]
```
**Why does the Win Rate fluctuate?** 
When the curriculum expands the word pool (e.g., from 400 to 800 words), the agent is suddenly faced with a harder game. You will notice the win rate (`wr=`) drop temporarily immediately after an expansion. However, as the agent continues to train on the new, larger pool, it learns the new patterns and the win rate steadily climbs back up. By the end of 20,000 episodes, it stabilizes at a high win rate (~80-88%) on the complete dataset.

**Step 3: Evaluate the Trained Model**
Once training finishes, benchmark its performance (with out-of-distribution testing):
```bash
python training/benchmark_dqn.py --n_words 8636 --model_path data/dqn_wordle_v2_full.pt --ood --sample 500
```

#### Advanced Training Options
```bash
# With live visual dashboard (opt-in)
python training/train.py --n_words 200 --episodes 2000 --visual

# GPU training (auto-detected or explicit)
python training/train.py --n_words 2309 --episodes 3000 --device cuda

# Custom hyperparameters
python training/train.py --n_words 200 --episodes 2000 \
  --batch_size 256 --lr 1e-3 --train_every 2

# Resume from checkpoint
python training/train.py --n_words 200 --episodes 2000 --resume data/checkpoint_ep500.pt
```

**Key Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_words` | 200 | Restrict action/answer space to top N words |
| `--episodes` | 2000 | Number of training episodes |
| `--curriculum` | off | Start small, progressively expand word pool |
| `--eval_every` | 100 | Checkpoint frequency |
| `--batch_size` | 128 | Replay buffer batch size |
| `--train_every` | 4 | Train every N environment steps |
| `--lr` | 5e-4 | Learning rate |
| `--target_update_every` | 10 | Hard update target network every N episodes |
| `--visual` | off | Opt-in to live matplotlib dashboard |
| `--device` | auto | `cpu`, `cuda`, or `auto` |

---

### 2. Benchmarking

#### Benchmark the Trained DQN Agent
```bash
# In-distribution test (on training pool words)
python training/benchmark_dqn.py --n_words 200

# With out-of-distribution generalization test
python training/benchmark_dqn.py --n_words 200 --ood

# Quick sample
python training/benchmark_dqn.py --n_words 200 --sample 50
```

#### Benchmark the Entropy-Greedy Heuristic
```bash
python training/benchmark.py
python training/benchmark.py 100  # quick sample
```

#### Head-to-Head Comparison
```bash
python training/compare.py --n_words 200 --model_path data/dqn_wordle_v2.pt
```

---

### 3. Plotting & Results

#### Generate All Results (One Click)
```bash
python analysis/generate_results.py
```

#### Plot Training Metrics
```bash
python analysis/plot_training.py
python analysis/plot_training.py --all  # includes convergence + phase analysis
```

---

### 4. Interactive Play

#### Watch a Live Demo Game
```bash
# Entropy heuristic
python tools/play_live.py --agent heuristic --answer knoll

# DQN agent
python tools/play_live.py --agent dqn --model_path data/dqn_wordle_v2.pt --n_words 200
```

#### Real-Time Wordle Solver Assistant
```bash
python tools/solve_assistant.py
```

---

### 5. Web App

The project includes a complete static web application (HTML/CSS/JS) that showcases the RL results, interactive charts, system architecture, and an interactive browser-based Wordle solver!

**To run the web app locally:**
```bash
# Start a local HTTP server in the webapp directory
python -m http.server 8080 -d webapp
```
Then open your browser and navigate to [http://localhost:8080](http://localhost:8080).

*Note: The web app is purely static and is also ready to be deployed instantly to Netlify or GitHub Pages without any backend server.*

---

## How It Works

### Architecture: Embedding-Based DQN (v2)

Unlike traditional DQN which outputs a Q-value for each fixed word index, the v2 agent uses **word embeddings** to score any candidate word:

```
State Encoder:  183 → 256 → 128   (observation → state embedding)
Word Encoder:   130 → 128          (word one-hot → word embedding)
Scorer:         256 → 128 → 1      (concat → Q-value)
```

**Why this matters**: The agent learns *what makes a good guess* (letter patterns, positional information) rather than memorizing which word index to pick. This lets it generalize to words it has never seen during training.

### The Wordle MDP

**State:** 183-dim feature vector encoding green/yellow/gray letter constraints

**Action:** Choose a word to guess, scored via the embedding network

**Reward:** `-1` per guess + information-gain bonus + progressive solve bonus

**Observation:**
- Green letters (confirmed correct positions): 5×26 = 130 dims
- Yellow letters (confirmed present, wrong spot): 26 dims
- Gray letters (confirmed absent): 26 dims
- Remaining guesses (normalized): 1 dim

### Two Solving Strategies

| Strategy | Win Rate | Avg Guesses | Generalization |
|----------|----------|-------------|----------------|
| **Entropy Heuristic** | ~100% | 4.19 | N/A (uses candidate set directly) |
| **DQN v2 (200 words)** | ~95% | ~4.2 | ✅ Tested on unseen words |
| **DQN v2 (2,309 words)** | ~85%+ | ~4.5 | ✅ Tested on unseen words |

---

## System Architecture

> For the full detailed architecture with all diagrams, see [docs/system_architecture.md](docs/system_architecture.md).

### High-Level System Architecture

```mermaid
flowchart TB
    subgraph DATA["Data Layer"]
        WJ["wordles.json<br>(2,309 answers)"]
        NJ["nonwordles.json<br>(8,636 valid guesses)"]
        MP["dqn_wordle_v2.pt<br>(Trained Model)"]
    end

    subgraph CORE["Core Engine (core/)"]
        MDP["wordle_mdp.py<br>MDP Formulation"]
        ENV["wordle_env.py<br>WordleEnv"]
        AGT["dqn_agent.py<br>DQNAgent"]
        BRD["board_renderer.py<br>Visualization"]
    end

    subgraph TRAIN["Training Pipeline (training/)"]
        TR["train.py"]
        BM["benchmark_dqn.py"]
        CMP["compare.py"]
    end

    subgraph ANALYSIS["Analysis (analysis/)"]
        PT["plot_training.py"]
        GR["generate_results.py"]
    end

    subgraph TOOLS["Tools (tools/)"]
        PL["play_live.py"]
        SA["solve_assistant.py"]
    end

    subgraph ASSETS["Output (assets/)"]
        IMG["training_curves.png<br>benchmark_results.png<br>comparison_chart.png"]
    end

    WJ & NJ --> MDP --> ENV --> AGT
    AGT --> TR --> MP
    AGT & ENV --> BM & CMP
    TR --> ANALYSIS --> ASSETS
    BM & CMP --> ASSETS
    MDP & BRD & AGT --> TOOLS
```

![High-Level System Architecture](assets/system%20architecture/high-resolution/high_level_architecture.jpeg)

### Neural Network Architecture (EmbeddingQNetwork)

```mermaid
flowchart LR
    subgraph INPUT["Inputs"]
        OBS["Observation<br>(183 dims)"]
        WORD["Word Feature<br>(130 dims)"]
    end

    subgraph STATE_ENC["State Encoder"]
        S1["Linear(183, 256)"] --> S2["ReLU"] --> S3["Linear(256, 128)"] --> S4["ReLU"]
    end

    subgraph WORD_ENC["Word Encoder"]
        W1["Linear(130, 128)"] --> W2["ReLU"]
    end

    subgraph SCORER["Scorer"]
        CONCAT["Concat (256)"] --> SC1["Linear(256, 128)"] --> SC2["ReLU"] --> SC3["Linear(128, 1)"]
    end

    OBS --> S1
    WORD --> W1
    S4 --> CONCAT
    W2 --> CONCAT
    SC3 --> QVAL["Q-Value"]
```

![Neural Network Architecture](assets/system%20architecture/high-resolution/neural_network_architecture.jpeg)

### RL Training Loop

```mermaid
flowchart LR
    A["env.reset()"] --> B["Observe State"]
    B --> C["Agent: epsilon-greedy<br>over word embeddings"]
    C --> D["env.step(action)"]
    D --> E{"Done?"}
    E -- No --> F["Store in PER<br>Train every N steps"]
    F --> B
    E -- Yes --> G{"More episodes?"}
    G -- Yes --> A
    G -- No --> H["Save model +<br>metrics + plots"]
```

![RL Training Loop](assets/system%20architecture/high-resolution/rl_training_loop.jpeg)

### Agent Decision Pipeline

```mermaid
sequenceDiagram
    participant Env as WordleEnv
    participant Agent as DQNAgent
    participant Net as EmbeddingQNetwork
    participant Buffer as Replay Buffer

    Env->>Agent: obs (183-dim)
    Agent->>Env: valid_word_features(mask)
    Env-->>Agent: word_feats (N x 130)

    alt Explore
        Agent->>Agent: Random valid word
    else Exploit
        Agent->>Net: forward(obs, word_feats)
        Net-->>Agent: Q-values
        Agent->>Agent: argmax(Q)
    end

    Agent->>Env: step(action_idx)
    Env-->>Agent: next_obs, reward, done
    Agent->>Buffer: store (prioritized)
    Agent->>Net: train_step (Double DQN)
```

![Agent Decision Pipeline](assets/system%20architecture/high-resolution/agent_decision_pipeline.jpeg)

---

## Key Results

### Entropy-Greedy Heuristic (Baseline)
- **Winning rate:** 100% on 2,309 answers
- **Avg guesses:** 4.19
- **Opening word:** RAISE (5.88 bits of information)

### DQN v2 Agent (200-word pool, 2,000 episodes)
- **Winning rate:** ~95%
- **Avg guesses:** ~4.2
- **Training time:** ~5 minutes
- **Architecture:** EmbeddingDQN (Double DQN + PER)

### DQN v2 Agent (2,309-word pool, 3,000 episodes, curriculum)
- **Winning rate:** ~85%+
- **Avg guesses:** ~4.5
- **Training time:** ~30–45 minutes
- **OOD generalization:** ≥60% on unseen words

---

## Understanding the Output Files

### `data/dqn_wordle_v2.pt`
PyTorch model checkpoint (v2 format). Load with:
```python
from core.dqn_agent import DQNAgent
agent = DQNAgent(obs_dim=183, word_dim=130)
agent.load("data/dqn_wordle_v2.pt")
```

### `data/training_metrics.json`
Training metrics including win rates, avg guesses, losses, and training configuration.

### `assets/`
Auto-generated result images from training and benchmarking.

### `data/legacy/`
Old v1 model checkpoints (not compatible with v2 architecture).

---

## Tips & Troubleshooting

### Training is slow
- Default mode is headless (no matplotlib). Use `--train_every 8` for fewer training steps.
- Reduce `--n_words` (200 trains ~10× faster than 2,309)
- Use `--device cuda` if you have a GPU

### No images generated
- Ensure `scipy` is installed: `pip install scipy`
- Run `python analysis/generate_results.py` manually

### Model performs worse than heuristic
- Expected! The entropy heuristic is near-optimal (~100%)
- DQN learns a sub-optimal but generalizable policy
- The key advantage of DQN v2 is *generalization to unseen words*

### Out of memory
- Reduce `--batch_size` (try 64 or 32)
- Reduce `--n_words`

---

## Documentation

| Document | Description |
|----------|-------------|
| [System Architecture](docs/system_architecture.md) | Full architecture diagrams (Mermaid) |
| [Walkthrough Guide](docs/walkthrough.md) | Step-by-step usage guide |
| [Project Timeline](docs/timeline.md) | 1-month project plan with milestones |
| [Coverage Matrix](docs/coverage.md) | Implementation coverage and test matrix |
| [Project Overview](PROJECT_OVERVIEW.md) | Deep dive into the architecture |

---

## Reinforcement Learning Final Project Group Members

| Profile | Name | Profile | Name |
| :---: | :--- | :---: | :--- |
| <img src="https://github.com/ghost.png?size=40" width="40"> | **Antonio, Mark Allen** | <img src="https://github.com/ghost.png?size=40" width="40"> | **Navarro, Genesis** |
| <img src="https://github.com/ghost.png?size=40" width="40"> | **Castro, James Adrian B.** | <img src="https://github.com/ghost.png?size=40" width="40"> | **Orro, Emmanuel Joshua** |
| <img src="https://github.com/ghost.png?size=40" width="40"> | **Glodo, Jannah Cleine** | <img src="https://github.com/ghost.png?size=40" width="40"> | **Ruiz, Mark Anthony** |
| <img src="https://github.com/ghost.png?size=40" width="40"> | **Liao, Adrian Miguel** | <img src="https://github.com/ghost.png?size=40" width="40"> | **Sy, Stephen Ace F.** |
| <img src="https://github.com/ghost.png?size=40" width="40"> | **Mago, Karl Mattheus** | <img src="https://github.com/ghost.png?size=40" width="40"> | **Talosig, Jay Arre** |
| <img src="https://github.com/ghost.png?size=40" width="40"> | **Medio, Charles** | | |

> **Tip:** To display actual profile pictures, edit the `README.md` and replace `ghost.png` in the image URLs with each member's actual GitHub username (e.g., `your_username.png`).

---

## References & Further Reading

- **MDP Formulation:** `core/wordle_mdp.py` contains detailed comments on state space, actions, and rewards
- **RL Algorithm:** `core/dqn_agent.py` implements embedding-based Double DQN with action masking
- **Visualization:** `core/board_renderer.py` is a reusable component for animated Wordle boards
- **3Blue1Brown's Wordle Solver:** Inspiration for the entropy-greedy approach
- **Wordle Game Rules:** Official Wordle feedback logic (including duplicate-letter handling) in `core/wordle_mdp.py::score_guess()`

---

## License

Public domain. Free to modify, distribute, and build upon.
