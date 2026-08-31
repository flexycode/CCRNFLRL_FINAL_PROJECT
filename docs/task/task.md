# Wordle DQN Revamp — Task List

## Phase 1: Core Architecture Revamp
- [x] Create `data/legacy/` and move old model checkpoints
- [x] Add `encode_word()` utility to `wordle_mdp.py`
- [x] Rewrite `dqn_agent.py` with EmbeddingDQN architecture
- [x] Optimize `wordle_env.py` — vectorized masking, enhanced obs, word features
- [x] Unit test the new agent on a tiny word pool (smoke test) — 97% win in 46s!

## Phase 2: Training Pipeline Revamp
- [x] Rewrite `train.py` — headless default, tqdm progress, curriculum learning, checkpoints, auto-save results
- [x] Update `benchmark_dqn.py` — OOD testing, auto-save, progress bar
- [x] Update `compare.py` — enhanced comparison, auto-save chart
- [x] Fix Unicode characters for Windows cp1252 compatibility

## Phase 3: Visualization & Results
- [x] Create `assets/` directory
- [x] Update `plot_training.py` — save to `assets/`, headless support, `--save_only` flag
- [x] Create `analysis/generate_results.py` — one-click result generator
- [x] Update `play_live.py` for v2 agent

## Phase 4: Documentation
- [x] Create `docs/timeline.md` — 1-month project timeline
- [x] Create `docs/coverage.md` — implementation coverage matrix
- [x] Create `docs/walkthrough.md` — step-by-step guide
- [x] Update `README.md` — full refresh

## Phase 5: Verification
- [x] Run training on 200-word pool (smoke test) — 97% win, 3.87 avg, 46s
- [/] Run full 2000-episode training (in progress)
- [ ] Verify result images generated in `assets/`
- [ ] Run benchmark and OOD generalization test
- [ ] Final walkthrough artifact
