# 期末口頭報告分工 — 10 分鐘

## 時間配置

| 段落 | 主講 | 時間 | 投影片頁數 | 重點 |
|---|---|---:|---:|---|
| Opening + Motivation | E | 1:00 | 2 | 題目介紹、三大挑戰、兩個 RQ |
| Environment | A | 1:30 | 2 | 8×8 規則、obs/action/reward、action mask |
| DQN | B | 2:00 | 3 | 網路架構（CNN backbone）、Double DQN + replay、訓練曲線 |
| PPO | C | 2:00 | 3 | MaskablePPO 設定、entropy coef 調參、訓練曲線 |
| Reward Shaping | D | 1:00 | 1 | sparse vs dense 對比、係數設計理由 |
| Results & Discussion | E | 1:30 | 2 | comparison bar chart、回答 RQ1/RQ2、limitations |
| Q&A buffer | — | 1:00 | — | 預留時間 |
| **總計** | | **10:00** | **13 頁** | |

> 投影片頁數 ≈ 1 頁/分鐘（含開場結尾、過渡頁），視製作後微調。

## 投影片風格規範

- **Template**: Google Slides，預設 16:9
- **配色**：背景白、強調色 `#4C72B0`（sparse 對應色，與 `comparison_*.png` 一致），次要色 `#DD8452`（dense）
- **字體**：標題 28pt、內文 18pt（不要小於 16pt）
- **圖表**：直接貼 `report/figures/comparison_score.png` 與 `comparison_steps.png`，避免在投影片上重畫
- **每頁字數上限**：80 字（中英混排），用條列點而非段落
- **每頁一個重點**，標題就是該頁要傳達的單一結論

## 製作 + 預演時程

| 日期 | 事項 | 負責 |
|---|---|---|
| Week 3 Day 1 | 各人完成自己負責的投影片初稿 | 全員 |
| Week 3 Day 2 | E 統一整合風格、過渡頁、章節編號 | E |
| Week 3 Day 3 | 第一次預演 → 計時、抓問題 | 全員 |
| Week 3 Day 4 | 修改 + 第二次預演 | 全員 |
| Week 3 Day 5 | 報告當天 | 全員 |

## Q&A 預備（以 proposal §6 為基礎）

預期會被問的問題與回答方向：

1. **「為什麼 PPO 表現比 DQN 差/好？」** → entropy coef 設定、policy collapse 風險、sample efficiency 對比
2. **「為什麼不超過 Greedy？」** → sample efficiency 不足、reward shaping 設計、500k steps 是否夠
3. **「dense reward 的係數是怎麼選的？」** → D 解釋 −0.1×holes / −0.05×bumpiness 的尺度判斷
4. **「方塊形狀 35 種，是否包含旋轉？」** → A 解釋 shapes.py 結構
5. **「為什麼用 8×8 不是 9×9（Block Blast 原版）？」** → 與 Tetris 文獻對齊、降低訓練成本

## 檢查清單

- [ ] 投影片風格一致（color、字體、頁碼）
- [ ] 計時 ≤ 10 分鐘（每人嚴守時段）
- [ ] 圖表清晰、字大到後排看得見
- [ ] 每位組員至少預演過 2 次
- [ ] 投影片 + report.pdf 在發表前一天上傳到 Moodle
