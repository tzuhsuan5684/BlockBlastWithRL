"""
Reward function definitions for Member D.
Pass reward_mode='sparse' or 'dense' to BlockBlastEnv.__init__().

Sparse:  +n_lines cleared this step  –10 on death
Dense:   sparse  –0.1×holes  –0.05×bumpiness
"""

SPARSE_LINE_REWARD = 1.0
DEATH_PENALTY      = -10.0
HOLE_PENALTY       = -0.1
BUMPINESS_PENALTY  = -0.05
