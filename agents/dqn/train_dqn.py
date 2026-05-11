# 組員 B — DQN 訓練入口
# 在這裡實作 DQN 訓練邏輯
# 可以新增 dqn_agent.py、replay_buffer.py 等檔案

from env import BlockBlastEnv

def train():
    env = BlockBlastEnv(reward_mode="sparse")
    # TODO: 實作 DQN 訓練

if __name__ == "__main__":
    train()
