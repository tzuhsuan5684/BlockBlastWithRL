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
# Install deps (uv reads requirements.txt; pyproject lists no deps)
uv pip install -r requirements.txt

# Run the sanity test — required to pass before pushing env changes
uv run python test_env.py
```

There is no lint/format/CI configured.

## Architecture

The codebase is small (~5 source files); the non-obvious parts are the contracts between them.

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

[`reward_functions.py`](reward_functions.py) holds only constants (`SPARSE_LINE_REWARD`, `DEATH_PENALTY`, `HOLE_PENALTY`, `BUMPINESS_PENALTY`). The env's `_dense_shaping()` currently hardcodes `-0.1×holes − 0.05×bumpiness` rather than reading from `reward_functions.py` — Member D's job is to wire these constants in. If you're asked to tune dense rewards, this is the seam, not the env internals.

### Shape catalogue

[`env/shapes.py`](env/shapes.py) defines 35 shapes as lists of `(dr, dc)` offsets from a top-left anchor (all offsets ≥0, so placement at `(r,c)` occupies `(r+dr, c+dc)`). Rotations are pre-baked as separate entries (e.g. `L-0`, `L-90`, `L-180`, `L-270`) rather than computed — there is no rotation operator at runtime.

## Files Worth Knowing About

- [README.md](README.md) — the canonical contract and per-member usage examples (Chinese).
- [env/block_blast_env.py](env/block_blast_env.py) — env, action encoding, action masking, dense-shaping helpers.
- [agents/network.py](agents/network.py) — shared CNN backbone and `obs_to_tensor`.
- [agents/greedy_agent.py](agents/greedy_agent.py) — one-step-lookahead baseline; useful reference for how to simulate an action without mutating env state.
- [test_env.py](test_env.py) — acceptance tests for the env contract.
- `main.py` and `proposal.docx` exist but are not part of the runtime; ignore unless asked.
