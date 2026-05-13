# run_ppo.ps1 — 一鍵跑完 PPO train → evaluate → demo
#
# 用法：
#   .\run_ppo.ps1                          # 預設：sparse, seed=0, 500k steps
#   .\run_ppo.ps1 -Reward dense            # dense reward
#   .\run_ppo.ps1 -TotalSteps 1000000      # 訓練更多步
#   .\run_ppo.ps1 -SkipTrain               # 跳過訓練，直接 evaluate + demo
#   .\run_ppo.ps1 -SkipDemo                # 不開 demo 視窗
#   .\run_ppo.ps1 -Reward dense -TotalSteps 2000000 -Seed 1

param(
    [string] $Reward      = "dense",
    [int]    $Seed        = 0,
    [int]    $TotalSteps  = 500000,
    [int]    $NEpisodes   = 100,
    [float]  $Fps         = 4,
    [switch] $SkipTrain,
    [switch] $SkipEval,
    [switch] $SkipDemo
)

$Tag     = "ppo_${Reward}_seed${Seed}"
$CkptDir = "agents/ppo/checkpoints"
$LogDir  = "runs"
$OutDir  = "results"

# ── 1. Train ─────────────────────────────────────────────────────────────────
if (-not $SkipTrain) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host " [1/3] Training  ($Tag, ${TotalSteps} steps)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    uv run python -m agents.ppo.train_ppo `
        --reward      $Reward `
        --seed        $Seed `
        --total-steps $TotalSteps `
        --ckpt-dir    $CkptDir `
        --log-dir     $LogDir

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Training failed." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[skip] Training skipped." -ForegroundColor Yellow
}

# ── 找最新的 checkpoint ───────────────────────────────────────────────────────
$Ckpt = Get-ChildItem "$CkptDir/${Tag}_step*.pt" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 -ExpandProperty FullName

if (-not $Ckpt) {
    Write-Host "[ERROR] No checkpoint found in $CkptDir for tag '$Tag'." -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "Using checkpoint: $Ckpt" -ForegroundColor Green

# ── 2. Evaluate ───────────────────────────────────────────────────────────────
if (-not $SkipEval) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host " [2/3] Evaluating ($NEpisodes episodes)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $OutJson = "$OutDir/${Tag}.json"

    uv run python -m agents.ppo.evaluate `
        --checkpoint $Ckpt `
        --reward     $Reward `
        --episodes   $NEpisodes `
        --seed       42 `
        --out        $OutJson `
        --notes      "TotalSteps=$TotalSteps seed=$Seed"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Evaluation failed." -ForegroundColor Red
        exit 1
    }
    Write-Host "Result saved to: $OutJson" -ForegroundColor Green
} else {
    Write-Host "[skip] Evaluation skipped." -ForegroundColor Yellow
}

# ── 3. Demo ───────────────────────────────────────────────────────────────────
if (-not $SkipDemo) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host " [3/3] Demo (fps=$Fps, ESC to quit)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    uv run python demo/play.py `
        --agent      ppo `
        --checkpoint $Ckpt `
        --fps        $Fps
} else {
    Write-Host "[skip] Demo skipped." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
