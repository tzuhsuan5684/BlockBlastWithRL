# 組員 C — PPO 訓練入口
# 在這裡實作 PPO 訓練邏輯

from env import BlockBlastEnv

def train():
    env = BlockBlastEnv(reward_mode="sparse")
    # TODO: 實作 PPO 訓練

if __name__ == "__main__":
    train()
