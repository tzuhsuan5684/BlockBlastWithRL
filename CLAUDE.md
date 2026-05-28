# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

A multi-member RL coursework project training agents to play Block Blast (8×8 grid puzzle) using Gymnasium. The work is split across teammates by feature branch:

- `feature/dqn` (Member B) — DQN
- `feature/ppo` (Member C) — PPO (MaskablePPO or custom)
- `feature/reward` (Member D) — reward shaping
- `feature/baseline` (Member E) — random/greedy baselines

Member A owns and maintains the environment. The primary developer-facing reference is [README.md](README.md) (in Chinese) — treat it as the canonical spec for the env contract.

## Hard Rules

- **Never modify anything under [env/](env/).** That directory (the Gymnasium env and shape definitions) is Member A's territory. If a task seems to require changing it, surface the need rather than editing — the change has to go through Member A so all four downstream branches stay in sync.
- **Never modify [agents/network.py](agents/network.py) unilaterally.** It is the shared CNN backbone used by both the DQN and PPO members; a network change here invalidates the cross-method comparison that's the point of the project.
- After any env-adjacent change, [test_env.py](test_env.py) must pass before push. Treat its 4 tests as the env's acceptance criteria.

## Commands

Project uses **uv** for Python/package management (`.python-version` pins 3.13).

```bash
# Install deps (pyproject.toml is the source of truth; uv.lock pins exact versions)
uv sync

# Run the sanity test — required to pass before pushing env changes
uv run python test_env.py

# Train + eval + demo (convenience wrapper, uses dense reward by default)
uv run python run_ppo.py                          # train → eval → demo
uv run python run_ppo.py train --reward sparse
uv run python run_ppo.py train --reward both
uv run python run_ppo.py eval  --checkpoint path/to.pt

# Train with heuristic or afterstate variants (must call train_ppo.py directly —
# run_ppo.py does not forward these flags)
uv run python -m agents.ppo.train_ppo --reward dense --use-heuristics
uv run python -m agents.ppo.train_ppo --reward dense --use-afterstate

# Monitor training
uv run tensorboard --logdir runs/
```

Checkpoints land in `checkpoints/<reward>_seed<N>_<timestamp>/`; eval JSON in `results/`.

There is no lint/format/CI configured.

## Architecture

The codebase is small (env + agents/network.py + agents/ppo/ stack + a few helpers); the non-obvious parts are the contracts between them.

### Action space encoding — central to everything

`action ∈ Discrete(192)` where `action = piece_idx * 64 + row * 8 + col`. The flat-int encoding is exposed via [`encode_action` / `_decode_action`](env/block_blast_env.py) and is *the* coordinate any agent or evaluator speaks. Three observations follow from this:

1. **Action masking is mandatory.** Most of the 192 actions are illegal at any given state (out of bounds, cell already filled, slot already used). `env.action_masks()` returns a `(192,) bool` array and is also placed in `info["action_mask"]` on both `reset()` and `step()`. Unmasked agents will learn garbage — the env still accepts illegal actions and may crash. For SB3-contrib's `MaskablePPO` the mask is auto-pulled; custom DQN/PPO must apply `logits[~mask] = -inf` before argmax/Categorical.
2. **Three pieces are dealt per round.** Each `step()` consumes one of the three slots; only after all three are placed (`piece_shape_ids` all `-1`) does the env refill. This is why the action carries `piece_idx`.
3. **Termination check uses the mask.** `terminated = not np.any(mask)` — i.e. dead-end detection is exactly "no legal placement exists". Reward `-10` is added on the terminal step.

### Observation contract

`Dict` with four keys:

| key | shape | notes |
|---|---|---|
| `board` | `(8,8) float32` | 0 empty / 1 filled |
| `pieces` | `(3,5,5) float32` | each piece in a 5×5 binary grid; all-zeros when slot is used |
| `pieces_left` | `(3,) float32` | binary; 1 if slot still has a piece |
| `heuristics` | `(40,) float32` | Tier-1 hand-crafted features (all normalized to [0,1]): heights×8, holes×8, row_fill×8, col_fill×8, bumpiness×7, n_legal×1 |

DQN (Member B) can safely ignore `pieces_left` and `heuristics` — the env added them without breaking existing board/pieces readers. The [`obs_to_tensor`](agents/network.py) helper extracts `board`, `pieces`, and `pieces_left` for `BlockBlastNet`/`BlockBlastActorCritic`; use it instead of hand-rolling the conversion.

### Network sharing for fair comparison

Members B (DQN) and C (PPO) use [`BlockBlastNet`](agents/network.py) and [`BlockBlastActorCritic`](agents/network.py) respectively, both in the same file. The shared spatial-fusion backbone is: `board (1,8,8) → 2×Conv(3×3) → (64,8,8)` fused with `pieces (3,5,5) → piece_encoder (weight-shared) → pad → (48,8,8)` via a fusion conv → `FC(4096→128)`, then `cat(pieces_left) → FC(131→128) → heads`. Any performance gap between DQN and PPO is intended to be attributable to the algorithm, not the encoder.

**PPO-only network extensions** (do not touch `agents/network.py`):

- [`agents/ppo/network_heuristic.py`](agents/ppo/network_heuristic.py) — `BlockBlastActorCriticH`: subclasses `BlockBlastActorCritic`, replaces the final shared FC to additionally accept the 40-d heuristic vector via a small MLP. Activated with `--use-heuristics` in `train_ppo.py`.
- [`agents/ppo/network_afterstate.py`](agents/ppo/network_afterstate.py) — `BlockBlastAfterstateActorCritic`: ~2.7k params; actor scores each action's afterstate features with a single shared linear layer (Chen et al. 2026, arXiv:2603.26765 Fig. 11); critic uses the 40-d heuristic vector. Activated with `--use-afterstate`.

`--use-heuristics` and `--use-afterstate` are mutually exclusive. The checkpoint stores which mode was used so `evaluate.py` can reload the right model class.

### Reward shaping seam

[`reward_functions.py`](reward_functions.py) holds the constants (`SPARSE_LINE_REWARD`, `DEATH_PENALTY`, `HOLE_PENALTY`, `BUMPINESS_PENALTY`, `COMBO_STREAK_BONUS`); the env's `_dense_shaping()` reads `HOLE_PENALTY`, `BUMPINESS_PENALTY`, and `COMBO_STREAK_BONUS` from it via `from reward_functions import ...`. The wiring is complete — to tune dense rewards, only edit `reward_functions.py`. Current values: `HOLE_PENALTY = -0.02`, `BUMPINESS_PENALTY = -0.01`, `COMBO_STREAK_BONUS = 0.2` (reward multiplied by `1 + 0.2 × streak` for consecutive clears).

**Hard-won lesson (don't re-learn it)**: at the proposal-default magnitude `(-0.1, -0.05)` and especially at `(-0.3, -0.1)` that briefly lived in the repo, per-step shaping (−1 to −3 in typical mid-game states) overwhelms the +1 line-clear reward. The PPO critic's value loss explodes (~150), `approx_kl` rises above 0.05, and policy gradient gets clipped out — the agent ends up "learning" to die early to avoid accumulating negative reward. Reducing magnitudes 5–15× (current `−0.02 / −0.01`) keeps shaping as a gentle nudge rather than a dominant force, and recovers `value_loss < 1`. See [docs/ppo_journey.md](docs/ppo_journey.md) for the full diagnostic chain.

### Shape catalogue

[`env/shapes.py`](env/shapes.py) defines 35 shapes as lists of `(dr, dc)` offsets from a top-left anchor (all offsets ≥0, so placement at `(r,c)` occupies `(r+dr, c+dc)`). Rotations are pre-baked as separate entries (e.g. `L-0`, `L-90`, `L-180`, `L-270`) rather than computed — there is no rotation operator at runtime.

## Files Worth Knowing About

- [README.md](README.md) — the canonical contract and per-member usage examples (Chinese).
- [env/block_blast_env.py](env/block_blast_env.py) — env, action encoding, action masking, dense-shaping helpers, heuristic computation.
- [agents/network.py](agents/network.py) — shared CNN backbone (`BlockBlastNet`, `BlockBlastActorCritic`) and `obs_to_tensor`.
- [agents/ppo/ppo_agent.py](agents/ppo/ppo_agent.py) — `PPOAgent` dispatches across the three network modes (base / heuristic / afterstate).
- [agents/ppo/afterstate.py](agents/ppo/afterstate.py) — simulates each legal placement on a board copy and extracts 9 DT-style features; mirrors `env/block_blast_env.py`'s placement logic — **keep in sync if env rules change**.
- [agents/ppo/network_heuristic.py](agents/ppo/network_heuristic.py) — `BlockBlastActorCriticH`, extends base with 40-d heuristic MLP.
- [agents/ppo/network_afterstate.py](agents/ppo/network_afterstate.py) — `BlockBlastAfterstateActorCritic`, ~2.7k params afterstate scorer.
- [agents/greedy_agent.py](agents/greedy_agent.py) — one-step-lookahead baseline; useful reference for how to simulate an action without mutating env state.
- [reward_functions.py](reward_functions.py) — dense-shaping coefficients (env imports from here, see "Reward shaping seam" above).
- [run_ppo.py](run_ppo.py) — train/eval/demo automation wrapper; does **not** expose `--use-heuristics`/`--use-afterstate` (call `train_ppo.py` directly for those).
- [demo/play.py](demo/play.py) — pygame visualizer; supports `--agent random/greedy/ppo` with `--checkpoint` for the PPO case.
- [docs/ppo_journey.md](docs/ppo_journey.md) — narrative of Member C's PPO work, including the v1→v2 reward-coefficient diagnostic. Read this before re-tuning dense shaping.
- [docs/improvement_options.md](docs/improvement_options.md) — three options (BC warm-start / bigger network / handcrafted features) to push PPO past its current ~4 score plateau, with impact/effort/cross-member trade-offs.
- [test_env.py](test_env.py) — acceptance tests for the env contract.
- `proposal.docx` exists but is not part of the runtime; ignore unless asked.

## Empirical Snapshot (2026-05)

Reference numbers from Member C's 1M-step runs, useful as sanity checks if you re-train:

| Agent | reward_mode | network | ep_score_mean | ep_score_max | notes |
|---|---|---|---|---|---|
| Random | — | — | ~2 | — | uniform-from-legal baseline |
| PPO | sparse | base | 2.82 | ~14 | plateaus around 700k steps |
| PPO | dense | base | 4.22 | ~16 | still improving at 1M; ablation winner |
| PPO | dense | heuristic (`--use-heuristics`) | TBD | TBD | not yet benchmarked |
| PPO | dense | afterstate (`--use-afterstate`) | TBD | TBD | not yet benchmarked |

If a new training run is producing `ep_score_mean < 1` with `value_loss > 50`, suspect that someone bumped the shaping coefficients back up — check `reward_functions.py` first.
