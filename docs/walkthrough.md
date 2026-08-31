# Wordle DQN v2 — Walkthrough Guide

## Overview

This guide walks you through the revamped Wordle DQN solver from start to finish. The v2 architecture uses **embedding-based scoring** that can generalize to words outside its training data.

---

## Prerequisites

**Python 3.8+** with the following packages:

```bash
pip install torch numpy matplotlib scipy tqdm
```

---

## Step 1: Quick Training (5 minutes)

Train the agent on a small 200-word pool to verify everything works:

```bash
python training/train.py --n_words 200 --episodes 2000
```

**What happens:**
- A tqdm progress bar shows training progress, ETA, and current win rate
- No matplotlib window pops up (headless by default)
- Model is saved to `data/dqn_wordle_v2.pt`
- Training metrics saved to `data/training_metrics.json`
- Result images auto-generated in `assets/`

**Expected output:**
```
Using device: cpu
Starting training on 200 words (action space = 200)
Training: 100%|████████████████| 2000/2000 [04:32<00:00, 7.35ep/s, wr=94% ag=4.12 ε=0.05 pool=200 272s]

Training complete in 272s (4.5 min)
Final eval over all 200 words: win_rate=94.00%  avg_guesses=4.12
Saved trained model to data/dqn_wordle_v2.pt
Saved training metrics to data/training_metrics.json

Generating result images...
  ✓ Saved: assets/training_curves.png
  ✓ Saved: assets/training_dashboard.png
✓ All result images saved to assets/
```

---

## Step 2: View Training Results

Check the auto-generated images in `assets/`:

```bash
# List all generated images
dir assets\

# Or generate/regenerate all results
python analysis/generate_results.py
```

**Generated files:**
| File | Description |
|------|-------------|
| `assets/training_curves.png` | Win rate + efficiency learning curves |
| `assets/training_dashboard.png` | 2×2 dashboard with summary statistics |

---

## Step 3: Benchmark the Agent

### In-Distribution Test
Test on words the agent trained on:

```bash
python training/benchmark_dqn.py --n_words 200
```

### Out-of-Distribution Test
Test on words the agent has **never seen** — the real generalization test:

```bash
python training/benchmark_dqn.py --n_words 200 --ood
```

This tests the agent on words #201–#400 from the word list, which were NOT in the training pool.

---

## Step 4: Compare with Heuristic

See how the DQN stacks up against the entropy-greedy heuristic:

```bash
python training/compare.py --n_words 200
```

---

## Step 5: Full-Scale Training

### Standard Training (2,309 words, ~30 min)
```bash
python training/train.py --n_words 2309 --episodes 3000
```

### Curriculum Training (recommended for large pools)
Start with 200 words and progressively expand:

```bash
python training/train.py --n_words 2309 --episodes 3000 --curriculum
```

### Large-Scale Training (8,636 words, ~1–2 hours)
```bash
python training/train.py --n_words 8636 --episodes 5000 --curriculum
```

---

## Step 6: Watch the Agent Play

```bash
# Watch the DQN agent
python tools/play_live.py --agent dqn --model_path data/dqn_wordle_v2.pt --n_words 200

# Watch with a specific answer
python tools/play_live.py --agent dqn --model_path data/dqn_wordle_v2.pt --n_words 200 --answer crane

# Compare with the heuristic
python tools/play_live.py --agent heuristic --answer crane
```

---

## Step 7: Generate All Results

One command to generate everything:

```bash
python analysis/generate_results.py
```

---

## Advanced: Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--n_words` | 200 | Word pool size |
| `--episodes` | 2000 | Number of training episodes |
| `--curriculum` | off | Progressive word pool expansion |
| `--train_every` | 4 | Train every N environment steps |
| `--lr` | 5e-4 | Learning rate |
| `--batch_size` | 128 | Replay batch size |
| `--visual` | off | Opt-in to live matplotlib dashboard |
| `--device` | auto | `cpu`, `cuda`, or `auto` |
| `--resume` | None | Resume from a checkpoint |
| `--save_name` | `dqn_wordle_v2.pt` | Model save filename |

---

## Troubleshooting

### Training is slow
- Default is already headless (no matplotlib). Check `--train_every` value.
- Reduce `--n_words` for a faster test.
- Use `--device cuda` if you have a GPU.

### Import errors
- Make sure you run scripts from the project root: `python training/train.py`
- Or use the full path: `python c:\path\to\RL PROJECT\training\train.py`

### Missing tqdm
```bash
pip install tqdm
```
The training script has a built-in fallback and will still work without tqdm.

### No images generated
- Check that `scipy` is installed: `pip install scipy`
- Run `python analysis/generate_results.py` manually after training.

---

## Architecture Summary

### Before (v1): Fixed Word Index Output
```
State (183) → MLP → Q-value per word index (N outputs)
```
- Cannot generalize: output layer is tied to training word list
- 8,636 outputs from 183 inputs = too sparse to learn

### After (v2): Embedding-Based Scoring
```
State (183) → State Encoder → 128-dim embedding
Word  (130) → Word Encoder  → 128-dim embedding
Concat (256) → Scorer        → 1 Q-value
```
- Can evaluate ANY word: score = f(state_embedding, word_embedding)
- Generalizes to unseen words because it learns *what makes a good guess* rather than memorizing word indices
