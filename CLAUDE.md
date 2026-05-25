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
```

[run_experiments.py](run_experiments.py) is the unified entry point — `train`, `eval`, and `demo` subcommands wrap the per-agent scripts:

```bash
uv run python run_experiments.py train ppo --reward dense        # or: train dqn / add --use-afterstate
uv run python run_experiments.py eval                            # all 5 agents -> results/*.json + comparison_*.png
uv run python run_experiments.py demo                            # pygame visualizer (latest PPO ckpt by default)
```

There is no lint/format/CI configured.

## Architecture

The codebase is small (env + agents/network.py + agents/ppo/ stack + a few helpers); the non-obvious parts are the contracts between them.

### Action space encoding — central to everything

`action ∈ Discrete(192)` where `action = piece_idx * 64 + row * 8 + col`. The flat-int encoding is exposed via [`encode_action` / `_decode_action`](env/block_blast_env.py) and is *the* coordinate any agent or evaluator speaks. Three observations follow from this:

1. **Action masking is mandatory.** Most of the 192 actions are illegal at any given state (out of bounds, cell already filled, slot already used). `env.action_masks()` returns a `(192,) bool` array and is also placed in `info["action_mask"]` on both `reset()` and `step()`. Unmasked agents will learn garbage — the env still accepts illegal actions and may crash. For SB3-contrib's `MaskablePPO` the mask is auto-pulled; custom DQN/PPO must apply `logits[~mask] = -inf` before argmax/Categorical.
2. **Three pieces are dealt per round.** Each `step()` consumes one of the three slots; only after all three are placed (`piece_shape_ids` all `-1`) does the env refill. This is why the action carries `piece_idx`.
3. **Termination check uses the mask.** `terminated = not np.any(mask)` — i.e. dead-end detection is exactly "no legal placement exists". Reward `-10` is added on the terminal step.

### Observation contract

`Dict({"board": (8,8) float32, "pieces": (3,5,5) float32})`. Each piece is rendered into its own 5×5 binary grid (largest shape is 5-wide); used slots are all-zeros. The [`obs_to_tensor`](agents/network.py) helper handles batched-vs-single obs and adds the CNN channel dim — use it instead of hand-rolling the conversion.

### Network sharing for fair comparison

Members B (DQN) and C (PPO) are expected to use [`BlockBlastNet`](agents/network.py) and [`BlockBlastActorCritic`](agents/network.py) respectively from the same file. Both share the same dual-branch backbone: `board → 2× Conv(3×3) → FC(128)` concatenated with `pieces → FC(64)`, then a `FC(128) → output` head. The point is that any performance gap between DQN and PPO should be attributable to the algorithm, not the encoder.

### Reward shaping seam

[`reward_functions.py`](reward_functions.py) holds the constants (`SPARSE_LINE_REWARD`, `DEATH_PENALTY`, `HOLE_PENALTY`, `BUMPINESS_PENALTY`); the env's `_dense_shaping()` reads `HOLE_PENALTY` and `BUMPINESS_PENALTY` from it via `from reward_functions import ...`. The wiring is complete — to tune dense rewards, only edit `reward_functions.py`. Current values: `HOLE_PENALTY = -0.02`, `BUMPINESS_PENALTY = -0.01`.

**Hard-won lesson (don't re-learn it)**: at the proposal-default magnitude `(-0.1, -0.05)` and especially at `(-0.3, -0.1)` that briefly lived in the repo, per-step shaping (−1 to −3 in typical mid-game states) overwhelms the +1 line-clear reward. The PPO critic's value loss explodes (~150), `approx_kl` rises above 0.05, and policy gradient gets clipped out — the agent ends up "learning" to die early to avoid accumulating negative reward. Reducing magnitudes 5–15× (current `−0.02 / −0.01`) keeps shaping as a gentle nudge rather than a dominant force, and recovers `value_loss < 1`. See [docs/ppo_journey.md](docs/ppo_journey.md) for the full diagnostic chain.

### Shape catalogue

[`env/shapes.py`](env/shapes.py) defines 35 shapes as lists of `(dr, dc)` offsets from a top-left anchor (all offsets ≥0, so placement at `(r,c)` occupies `(r+dr, c+dc)`). Rotations are pre-baked as separate entries (e.g. `L-0`, `L-90`, `L-180`, `L-270`) rather than computed — there is no rotation operator at runtime.

## Files Worth Knowing About

- [README.md](README.md) — the canonical contract and per-member usage examples (Chinese).
- [env/block_blast_env.py](env/block_blast_env.py) — env, action encoding, action masking, dense-shaping helpers.
- [agents/network.py](agents/network.py) — shared CNN backbone and `obs_to_tensor`.
- [agents/ppo/](agents/ppo/) — Member C's custom PPO (rollout buffer, agent, train, evaluate, BC pretraining if added).
- [agents/greedy_agent.py](agents/greedy_agent.py) — one-step-lookahead baseline; useful reference for how to simulate an action without mutating env state.
- [reward_functions.py](reward_functions.py) — dense-shaping coefficients (env imports from here, see "Reward shaping seam" above).
- [demo/play.py](demo/play.py) — pygame visualizer; supports `--agent random/greedy/ppo` with `--checkpoint` for the PPO case.
- [docs/ppo_journey.md](docs/ppo_journey.md) — narrative of Member C's PPO work, including the v1→v2 reward-coefficient diagnostic. Read this before re-tuning dense shaping.
- [docs/improvement_options.md](docs/improvement_options.md) — three options (BC warm-start / bigger network / handcrafted features) to push PPO past its current ~4 score plateau, with impact/effort/cross-member trade-offs.
- [test_env.py](test_env.py) — acceptance tests for the env contract.
- [run_experiments.py](run_experiments.py) — unified `train` / `eval` / `demo` entry point (the only top-level CLI; per-agent `train_*.py` and `evaluate.py` modules are still callable but this script is the canonical wrapper).
- [evaluation/](evaluation/) — `metrics_schema.save_metrics` defines the 11-field JSON contract; `aggregate.py` + `plot_comparison.py` consume `results/*.json` (invoked automatically by `run_experiments.py eval`).
- `proposal.docx` exists but is not part of the runtime; ignore unless asked.

## Empirical Snapshot (2026-05)

Reference numbers from Member C's 1M-step runs, useful as sanity checks if you re-train:

| Agent | reward_mode | ep_score_mean | ep_score_max | notes |
|---|---|---|---|---|
| Random | — | ~2 | — | uniform-from-legal baseline |
| PPO | sparse | 2.82 | ~14 | plateaus around 700k steps |
| PPO | dense | 4.22 | ~16 | still improving at 1M; ablation winner |

If a new training run is producing `ep_score_mean < 1` with `value_loss > 50`, suspect that someone bumped the shaping coefficients back up — check `reward_functions.py` first.
