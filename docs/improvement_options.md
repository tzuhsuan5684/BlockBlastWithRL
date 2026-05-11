# 進一步突破 PPO 分數的方案討論

> 給組員 A/B/C/D/E 一起討論。**不是要全做**,選 0-2 個就夠了。
> 目的:把目前 PPO 4.22 分推到 10+ 分,讓 RL 方法在報告裡看起來「不只是會學,而是學得好」。

---

## 0. 我們在哪裡

| Agent | reward | steps | ep_score_mean |
|---|---|---|---|
| Random | — | — | ~2 |
| Greedy | — | — | **未測**(預估 10-30,組員 E 處理) |
| PPO + sparse v2 | sparse | 1M | 2.82 |
| **PPO + dense v2** | dense (-0.02/-0.01) | 1M | **4.22** |

**問題**: PPO 大概率還是輸給 Greedy。這對報告的故事不是死局(proposal §6 已允許「RL 沒贏 Greedy → 分析原因」),但能贏的話更亮眼。

### 為什麼分數低 — 結構性原因

1. Action space 192,絕大多數時刻只有 ~20-50 個合法 → 探索成本高
2. Line clear 是 sparse event(平均 5-15 步才出現一次)→ 信用分派(credit assignment)困難
3. Block Blast 不像 Atari,沒有 visual cue 暗示「往哪推一定贏」
4. Episode 短(~15-20 步),agent 沒太多步數規劃
5. 現在的 PPO 從隨機 policy 起步,要靠運氣撞到第一次消行才能 bootstrap → 訓練前期極度浪費

純調超參數的天花板大概是 **6-8 分**。要破 10 分得動 architecture 或 data。

---

## 1. 三個方案

### 🅰️ Behavior Cloning 預訓練(BC warm start)

**邏輯**: PPO 從隨機 policy 學最慢的就是「找到第一次消行」。Greedy 就是消行專家,讓 PPO 先用監督式學 Greedy 的 (obs → action),再接 PPO 微調。等於先讓 PPO 達到 Greedy 水準,再用長期規劃超越 Greedy。

**做法**:
```
1. 跑 Greedy agent 50k 步,存 (obs, action, mask) 對 → 訓練資料
2. 監督式訓練 BlockBlastActorCritic 的 actor head(cross-entropy on action prediction)
3. critic head 隨機初始化(或用收集到的 return 預測訓練)
4. 把 BC 好的 weights 餵進 PPO 當 init,接著正常跑 PPO 1M
```

**新增檔案**: `agents/ppo/bc_pretrain.py`(~80 行)
**修改檔案**: `agents/ppo/train_ppo.py` 加 `--init-checkpoint` 參數
**對其他組員影響**: **無**(不動 env、不動 shared network)

**預期**: 4.22 → **10-15**
**風險**: BC 可能讓 policy 過於模仿 Greedy 的短視行為,需要 entropy bonus 推它探索長期策略

---

### 🅱️ 升級網路架構

**邏輯**: 目前 backbone 是 32→64 channels 兩層 conv + FC(128),參數量小。對 192 action × 多 piece 配置的複雜決策可能容量不足。

**做法**:
```
agents/network.py 改成:
  Conv2d(1 → 64, 3x3) + ReLU
  Conv2d(64 → 128, 3x3) + ReLU
  Conv2d(128 → 128, 3x3) + ReLU
  FC(8192 → 256)
  + 加 residual connection 或 group norm
pieces head 也擴大:75 → 128
combined head:256+128 → 256 → output
```

**對其他組員影響**: **影響組員 B**(共用 network)
- B 的 DQN 訓練得跟著重跑
- B 寫好的 hyperparameter 可能要重調
- **必須先跟 B 同步,不然會破壞他的進度**

**預期**: 4.22 → **6-9**
**風險**: 網路變大,訓練時間變長(可能 1M 跑不完,要拉到 2M);參數量多但 8×8 棋盤本來資訊就少,邊際效益有限

---

### 🅲️ Hand-crafted features 進 obs

**邏輯**: 現在 agent 看的是 raw 8×8 binary board,要自己學「哪欄高、哪有洞、放下去清幾行」。直接把這些算好餵進去,等於送外掛。

**做法**: 在 [env/block_blast_env.py](../env/block_blast_env.py) `_get_obs()` 加額外 feature channels:

```python
# 現在
obs = {"board": (8,8), "pieces": (3,5,5)}

# 加完
obs = {
    "board":     (8,8),
    "pieces":    (3,5,5),
    "heights":   (8,),          # 每欄最高填到第幾列
    "holes":     (8,),          # 每欄有幾個洞
    "preview":   (3, 8, 8),     # 對每個 piece × 每個合法位置,放下去能消幾行(預先算好)
}
```

**對其他組員影響**: **影響全員**
- A: 動 env(現在解禁但要協調)
- B: obs space 改了,DQN 整個重訓,網路 input 也得改
- C: 自己的 checkpoint 全部作廢,要重訓
- D: reward 邏輯不受影響,可保留
- E: baseline 不受影響(random/greedy 不需這些 feature)

**預期**: 4.22 → **8-15**(線索豐富,學習效率大幅提升)
**風險**: 工程量最大,所有人都得配合;`preview` channel 算起來貴,可能拖慢訓練 fps

---

## 2. 方案比較

| 項目 | A: BC | B: 大網路 | C: features |
|---|---|---|---|
| 預估分數 | **10-15** | 6-9 | 8-15 |
| 新增 code | ~100 行 | ~30 行 | ~150 行 |
| 訓練時間影響 | +20%(多 BC 預訓練) | +50%(網路大) | +30%(算 feature) |
| 影響組員 | **無** | B(DQN) | A, B, D, E 全部 |
| 風險 | 低 | 中 | 高 |
| 額外報告材料 | **BC vs no-BC ablation**(超棒) | 網路大小 ablation | feature ablation |
| 學術新意 | 中 | 低 | 中-高 |

## 3. 組合方案

| 組合 | 預估 | 工程量 | 適合什麼情境 |
|---|---|---|---|
| 只做 A | 10-15 | 中 | **時間 / 人力有限,要安全收尾** |
| 只做 B | 6-9 | 低 | 不想動架構但想推一點 |
| 只做 C | 8-15 | 高 | 全員有時間配合 |
| **A + B** | **15-20** | 中-高 | 要拚高分,B 同意一起動網路 |
| A + C | 15-25 | 很高 | 全員配合 + 時間夠 |

---

## 4. 我(組員 C)的建議

**先做 A,看結果再決定要不要 + B**。

理由:
1. A 不影響任何人,可以**單兵推進**不卡其他人進度
2. A 預期效益最高(因為 PPO 真正的瓶頸是 cold start,不是網路容量)
3. **A 自己就是一個 ablation**: 「PPO」vs「PPO + BC」對比寫進報告,展示「初始化品質對 RL 收斂的關鍵性」,這是有質感的 finding
4. 如果 A 後 PPO ≈ 13 分還想推上去,再協調 B 一起改網路

---

## 5. 開放討論點

請大家就以下幾點表達意見後,我們敲定方案:

1. **時間預算**: 距離報告 deadline 還剩幾天?願意花在訓練 + tuning 的時間?
2. **B 同意動 `agents/network.py` 嗎?** 如果同意,A+B 組合是最強選項
3. **A 同意動 `env/_get_obs()` 嗎?** 如果同意,方案 C 才可行
4. **要不要分頭跑多個 seed?** 報告裡若能附上「3 seeds, mean ± std」,審稿觀感會好很多
5. **要不要為了「PPO 必須贏 Greedy」而 push?** proposal §6 允許 RL 沒贏的劇本,但拚一下分數會更亮眼

---

## 6. 我的 timeline 預估

如果定下做 A:
- D1: 寫 BC 訓練腳本 + 收 Greedy 資料(0.5 天)
- D2: BC pre-train + 接 PPO + 跑 1M(0.5 天,含等訓練)
- D3: 看結果 + 跑 evaluate.py 產 JSON(0.25 天)
- D4: 報告章節擴寫(0.5 天)

**總計 ~2 天**,可以接受。

如果定下做 A + B,**+1 天**(網路改完 + 重訓 + 跟 B 同步)。
