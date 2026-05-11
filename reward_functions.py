"""
Reward function definitions.
Pass reward_mode='sparse' or 'dense' to BlockBlastEnv.__init__().

Sparse:  +n_lines cleared this step  –10 on death
Dense:   sparse  +HOLE_PENALTY×holes  +BUMPINESS_PENALTY×bumpiness

Coefficient history:
- proposal §3.1 default     : holes=-0.1,  bumpiness=-0.05
- earlier in-repo (too harsh): holes=-0.3,  bumpiness=-0.1
- current (tuned for PPO)   : holes=-0.02, bumpiness=-0.01
  Reason: at -0.3/-0.1, per-step shaping (-1 to -3) dwarfed the +1
  line-clear reward, causing critic value_loss to explode (~150) and
  PPO policy_loss to stall near 0. Reduced ~15x to keep shaping
  signal as a gentle nudge rather than a dominant force.
"""

SPARSE_LINE_REWARD = 1.0
DEATH_PENALTY      = -10.0
HOLE_PENALTY       = -0.02
BUMPINESS_PENALTY  = -0.01
