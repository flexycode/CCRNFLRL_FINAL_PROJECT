# Wordle DQN Solver — 1-Month Project Timeline

## Project Overview
**Goal**: Revamp the Wordle DQN training pipeline to achieve >85% win rate on the full 2,309-word pool, enable generalization to unseen words, optimize training to complete within 30–45 minutes on a laptop, and auto-generate result artifacts.

**Duration**: 4 weeks (September 1 – September 30, 2026)

---

## Week 1: Architecture Revamp & Core Training (Sep 1–7)

### Milestone: Working v2 Agent on Small Pool

| Day | Task | Status |
|-----|------|--------|
| Day 1 | ✅ Finalize architecture plan. Create `encode_word()` utility. | Done |
| Day 2 | ✅ Rewrite `dqn_agent.py` → EmbeddingDQN with Double DQN + PER. | Done |
| Day 3 | ✅ Rewrite `wordle_env.py` → Vectorized masking, word features. | Done |
| Day 4 | ✅ Rewrite `train.py` → Headless, tqdm, curriculum, checkpoints. | Done |
| Day 5 | Run 200-word training smoke test, debug any issues. | Pending |
| Day 6 | Run 500-word training, tune hyperparameters (lr, epsilon, rewards). | Pending |
| Day 7 | Buffer day for fixes and optimization. | Pending |

**Deliverable**: Trained v2 model on 200 words with ≥90% win rate.

---

## Week 2: Scaling & Generalization (Sep 8–14)

### Milestone: Full-Pool Training & OOD Results

| Day | Task | Status |
|-----|------|--------|
| Day 8 | Train on 2,309-word pool (target: 30–45 min). | Pending |
| Day 9 | Run OOD (out-of-distribution) generalization tests. | Pending |
| Day 10 | Hyperparameter tuning: experiment with lr, batch_size, reward shaping. | Pending |
| Day 11 | Curriculum learning experiments: compare with/without curriculum. | Pending |
| Day 12 | Train on 8,636-word ENABLE1 pool (target: 1–2 hours). | Pending |
| Day 13 | Full benchmark suite: in-dist + OOD + comparison vs heuristic. | Pending |
| Day 14 | Review results, identify bottlenecks, plan Week 3 improvements. | Pending |

**Deliverable**: Trained v2 model on 2,309 words with ≥85% win rate. OOD results showing generalization.

---

## Week 3: Polish & Documentation (Sep 15–21)

### Milestone: Complete Documentation & Visualization

| Day | Task | Status |
|-----|------|--------|
| Day 15 | Finalize all visualization scripts (training curves, benchmark, comparison). | Pending |
| Day 16 | Create `generate_results.py` one-click result generator. | Done |
| Day 17 | Write `docs/walkthrough.md` — step-by-step usage guide. | Pending |
| Day 18 | Write `docs/coverage.md` — implementation test matrix. | Pending |
| Day 19 | Update `README.md` with new architecture, commands, results. | Pending |
| Day 20 | Update `PROJECT_OVERVIEW.md` with v2 architecture diagrams. | Pending |
| Day 21 | Final training run with best hyperparameters, generate all result images. | Pending |

**Deliverable**: Complete documentation, updated README, all result images in `assets/`.

---

## Week 4: Final Testing & Presentation (Sep 22–30)

### Milestone: Submission-Ready Project

| Day | Task | Status |
|-----|------|--------|
| Day 22 | End-to-end test: fresh clone → train → benchmark → results. | Pending |
| Day 23 | Test on multiple machines (verify laptop portability). | Pending |
| Day 24 | Create presentation slides / demo video (if needed). | Pending |
| Day 25 | Final hyperparameter sweep and best model selection. | Pending |
| Day 26 | Code cleanup: remove dead code, add missing docstrings. | Pending |
| Day 27 | Final OOD generalization analysis with write-up. | Pending |
| Day 28 | Buffer day for any remaining fixes. | Pending |
| Day 29–30 | Final review, submission preparation. | Pending |

**Deliverable**: Submission-ready project with all deliverables complete.

---

## Key Performance Targets

| Metric | Current (v1) | Target (v2) |
|--------|-------------|-------------|
| Win rate (200 words) | 95% | ≥95% |
| Win rate (2,309 words) | ~80% | ≥85% |
| Win rate (8,636 words) | 74.7% | ≥80% |
| OOD generalization | 0% (not possible) | ≥60% |
| Training time (200 words) | ~30 min | ≤5 min |
| Training time (2,309 words) | ~2 hours | ≤45 min |
| Auto-generated results | None | PNG in assets/ |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Embedding-based DQN doesn't converge | Fall back to larger hidden layers (512), try different lr schedules |
| Training still too slow on laptop | Reduce train_every, smaller batch_size, skip PER |
| OOD generalization target not met | Expand training pool via curriculum, add data augmentation |
| Matplotlib issues in headless mode | Already using `Agg` backend, fully tested |
