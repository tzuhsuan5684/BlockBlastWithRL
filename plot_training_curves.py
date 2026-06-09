"""Extract training curves from TensorBoard logs and save as PNG."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUT_DIR = Path("results/curves")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
COLOURS = {
    "Random":   "#AAAAAA",
    "Greedy":   "#F5A623",
    "DQN":      "#E74C3C",
    "PPO":      "#3498DB",
    "PPO-AF":   "#2ECC71",
}
BASELINE = {"Random": 1.17, "Greedy": 3.97}

# ── run definitions ───────────────────────────────────────────────────────────
def _resolve(base: str) -> str:
    """Return the deepest event-containing subfolder under base."""
    import glob as _glob
    subdirs = sorted(_glob.glob(f"{base}/*"))
    return subdirs[-1] if subdirs else base

RUNS = [
    dict(label="DQN (sparse)",    agent="DQN",    color=COLOURS["DQN"],
         ls="--", path=_resolve("runs/dqn_sparse_seed0")),
    dict(label="DQN (dense)",     agent="DQN",    color=COLOURS["DQN"],
         ls=":",  path=_resolve("runs/dqn_dense_seed0")),
    dict(label="PPO (sparse)",    agent="PPO",    color=COLOURS["PPO"],
         ls="--", path=_resolve("runs/ppo_sparse_seed0")),
    dict(label="PPO (dense)",     agent="PPO",    color=COLOURS["PPO"],
         ls=":",  path=_resolve("runs/ppo_dense_seed0")),
    dict(label="PPO-AF (sparse)", agent="PPO-AF", color=COLOURS["PPO-AF"],
         ls="--", path=_resolve("runs/ppo_sparse_af_seed0")),
    dict(label="PPO-AF (dense)",  agent="PPO-AF", color=COLOURS["PPO-AF"],
         ls=":",  path=_resolve("runs/ppo_dense_af_seed0")),
]


def load_scalar(log_dir: str, tag: str):
    """Return (steps, values) arrays for one scalar tag from a TF event file."""
    acc = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return np.array([]), np.array([])
    events = acc.Scalars(tag)
    steps  = np.array([e.step  for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def smooth(values, window=10):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


# ═════════════════════════════════════════════════════════════════════════════
# Plot 1 — Episode Score vs Training Steps (all agents)
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.5))

for run in RUNS:
    steps, vals = load_scalar(run["path"], "rollout/ep_score_mean")
    if len(steps) == 0:
        # DQN logs per-episode; fall back to ep_score
        steps, vals = load_scalar(run["path"], "rollout/ep_score")
        # bin into 50k-step windows for a smoother curve
        if len(steps) > 0:
            bins   = np.arange(0, steps.max() + 50_001, 50_000)
            means  = [vals[(steps >= bins[i]) & (steps < bins[i+1])].mean()
                      for i in range(len(bins)-1)
                      if np.any((steps >= bins[i]) & (steps < bins[i+1]))]
            ctrs   = [(bins[i] + bins[i+1]) / 2
                      for i in range(len(bins)-1)
                      if np.any((steps >= bins[i]) & (steps < bins[i+1]))]
            steps, vals = np.array(ctrs), np.array(means)

    if len(steps) == 0:
        continue

    ax.plot(steps, smooth(vals, window=8),
            label=run["label"], color=run["color"],
            ls=run["ls"], lw=2.0, alpha=0.9)

# Baseline horizontal lines
for name, score in BASELINE.items():
    ax.axhline(score, color=COLOURS[name], lw=1.2, ls="-.",
               alpha=0.6, label=f"{name} baseline ({score:.1f})")

ax.set_xlabel("Training Steps", fontsize=13)
ax.set_ylabel("Mean Episode Score (lines cleared)", fontsize=13)
ax.set_title("Training Curve — Score vs. Steps", fontsize=15, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
ax.legend(fontsize=10, ncol=2, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
fig.tight_layout()
out = OUT_DIR / "training_score.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Plot 2 — Score vs Steps, PPO vs PPO-AF only (cleaner for presentation)
# ═════════════════════════════════════════════════════════════════════════════
PPO_RUNS = [r for r in RUNS if r["agent"] in ("PPO", "PPO-AF")]

fig, ax = plt.subplots(figsize=(9, 5))
for run in PPO_RUNS:
    steps, vals = load_scalar(run["path"], "rollout/ep_score_mean")
    if len(steps) == 0:
        continue
    ax.plot(steps, smooth(vals, window=8),
            label=run["label"], color=run["color"],
            ls=run["ls"], lw=2.2, alpha=0.95)

for name, score in BASELINE.items():
    ax.axhline(score, color=COLOURS[name], lw=1.2, ls="-.",
               alpha=0.55, label=f"{name} ({score:.1f})")

ax.set_xlabel("Training Steps", fontsize=13)
ax.set_ylabel("Mean Episode Score (lines cleared)", fontsize=13)
ax.set_title("PPO vs PPO-AF — Score during Training", fontsize=15, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
ax.legend(fontsize=11, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
fig.tight_layout()
out = OUT_DIR / "ppo_vs_ppoaf_score.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Plot 3 — Value Loss (PPO training stability)
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
for run in PPO_RUNS:
    steps, vals = load_scalar(run["path"], "train/value_loss")
    if len(steps) == 0:
        continue
    ax.plot(steps, smooth(vals, window=8),
            label=run["label"], color=run["color"],
            ls=run["ls"], lw=2.0, alpha=0.9)

ax.set_xlabel("Training Steps", fontsize=13)
ax.set_ylabel("Value Loss", fontsize=13)
ax.set_title("Value Loss during Training", fontsize=15, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
fig.tight_layout()
out = OUT_DIR / "value_loss.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Plot 4 — Entropy (exploration behaviour)
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
for run in PPO_RUNS:
    steps, vals = load_scalar(run["path"], "train/entropy")
    if len(steps) == 0:
        continue
    ax.plot(steps, smooth(vals, window=8),
            label=run["label"], color=run["color"],
            ls=run["ls"], lw=2.0, alpha=0.9)

ax.set_xlabel("Training Steps", fontsize=13)
ax.set_ylabel("Policy Entropy", fontsize=13)
ax.set_title("Policy Entropy during Training", fontsize=15, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
fig.tight_layout()
out = OUT_DIR / "entropy.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")

print("\nDone. All curves saved to results/curves/")
