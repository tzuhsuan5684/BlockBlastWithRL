"""
Reward function definitions.
Pass reward_mode='sparse' or 'dense' to BlockBlastEnv.__init__().

Sparse:  +n_lines cleared this step  –10 on death
Dense:   sparse  +HOLE_PENALTY×holes  +BUMPINESS_PENALTY×bumpiness

Coefficient history:
- proposal §3.1 default     : holes=-0.1,  bumpiness=-0.05
- earlier in-repo (too harsh): holes=-0.3,  bumpiness=-0.1
- v2 (PPO 1M baseline)      : holes=-0.02, bumpiness=-0.01  → ep_score_mean ≈ 4.22
- current (v3, 2× v2)       : holes=-0.04, bumpiness=-0.02
  Reason: v2 was safe but dense PPO was still improving at 1M, suggesting
  shaping signal was on the weak side. Doubling keeps per-step shaping
  well below the +1 line-clear reward (~-0.6 to -1.2 in mid/late game)
  so value_loss should stay bounded — but the gradient pushing toward
  flatter, hole-free boards is twice as strong. If value_loss climbs
  past ~5 or ep_score_mean drops below v2's 4.22, revert.
"""

SPARSE_LINE_REWARD  = 1.0
DEATH_PENALTY       = -10.0
HOLE_PENALTY        = -0.04
BUMPINESS_PENALTY   = -0.02
COMBO_STREAK_BONUS  = 0.2   # streak 每多 1 層，reward 多 ×0.2
