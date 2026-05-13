# run_bc.ps1 — Behavior Cloning 完整流程：收集資料 → BC 預訓練 → PPO fine-tune → evaluate → demo
#
# 用法：
#   .\run_bc.ps1                                   # 全跑，預設參數
#   .\run_bc.ps1 -NSteps 100000                    # 收集 10 萬步示範
#   .\run_bc.ps1 -BCEpochs 50 -TrainCritic        # BC 訓練 50 epochs，同時初始化 critic
#   .\run_bc.ps1 -TotalSteps 1000000               # PPO fine-tune 跑 1M steps
#   .\run_bc.ps1 -SkipCollect -SkipBCTrain         # 跳過前兩步，直接用現有 BC 權重跑 PPO
#   .\run_bc.ps1 -SkipPPO                          # 只做 collect + BC train，不跑 PPO
#   .\run_bc.ps1 -SkipDemo                         # 不開 demo 視窗

param(
    [string] $Reward       = "dense",
    [int]    $Seed         = 0,

    # Step 1: collect
    [int]    $NSteps       = 50000,
    [string] $DataPath     = "agents/bc/data/greedy_50k.npz",

    # Step 2: BC train
    [int]    $BCEpochs     = 30,
    [int]    $BCBatch      = 256,
    [float]  $BCLr         = 1e-3,
    [bool]   $TrainCritic  = $true,
    [string] $BCWeights    = "agents/bc/bc_weights.pt",

    # Step 3: PPO fine-tune
    [int]    $TotalSteps   = 500000,
    [int]    $NEpisodes    = 100,
    [float]  $Fps          = 4,

    # skip flags
    [switch] $SkipCollect,
    [switch] $SkipBCTrain,
    [switch] $SkipPPO,
    [switch] $SkipEval,
    [switch] $SkipDemo
)

$Tag     = "bc_ppo_${Reward}_seed${Seed}"
$CkptDir = "agents/ppo/checkpoints"
$LogDir  = "runs"
$OutDir  = "results"

# ── 1. Collect Greedy demonstrations ─────────────────────────────────────────
if (-not $SkipCollect) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host " [1/4] Collecting $NSteps Greedy steps" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    uv run python -m agents.bc.collect_data `
        --n-steps $NSteps `
        --reward  $Reward `
        --seed    $Seed `
        --out     $DataPath

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Data collection failed." -ForegroundColor Red
        exit 1
    }
    Write-Host "Data saved to: $DataPath" -ForegroundColor Green
} else {
    Write-Host "[skip] Data collection skipped." -ForegroundColor Yellow
}

# ── 2. BC pretraining ─────────────────────────────────────────────────────────
if (-not $SkipBCTrain) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host " [2/4] BC pretraining ($BCEpochs epochs)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    $BCArgs = @(
        "--data",       $DataPath,
        "--epochs",     $BCEpochs,
        "--batch-size", $BCBatch,
        "--lr",         $BCLr,
        "--out",        $BCWeights,
        "--seed",       $Seed
    )
    if ($TrainCritic) { $BCArgs += "--train-critic" }

    uv run python -m agents.bc.train_bc @BCArgs

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] BC training failed." -ForegroundColor Red
        exit 1
    }
    Write-Host "BC weights saved to: $BCWeights" -ForegroundColor Green
} else {
    Write-Host "[skip] BC training skipped." -ForegroundColor Yellow
}

# ── 3. PPO fine-tune (warm-started from BC) ──────────────────────────────────
if (-not $SkipPPO) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host " [3/4] PPO fine-tune ($TotalSteps steps, BC init)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    if (-not (Test-Path $BCWeights)) {
        Write-Host "[ERROR] BC weights not found: $BCWeights" -ForegroundColor Red
        Write-Host "        Run without -SkipBCTrain first." -ForegroundColor Red
        exit 1
    }

    uv run python -m agents.ppo.train_ppo `
        --reward        $Reward `
        --seed          $Seed `
        --total-steps   $TotalSteps `
        --ckpt-dir      $CkptDir `
        --log-dir       $LogDir `
        --bc-checkpoint $BCWeights

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] PPO fine-tune failed." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[skip] PPO fine-tune skipped." -ForegroundColor Yellow
}

# ── 找最新 checkpoint ─────────────────────────────────────────────────────────
$Ckpt = Get-ChildItem "$CkptDir/ppo_${Reward}_seed${Seed}_step*.pt" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 -ExpandProperty FullName

if (-not $Ckpt) {
    Write-Host "[ERROR] No checkpoint found in $CkptDir." -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "Using checkpoint: $Ckpt" -ForegroundColor Green

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
if (-not $SkipEval) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host " [4/4] Evaluating ($NEpisodes episodes)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $OutJson = "$OutDir/${Tag}.json"

    uv run python -m agents.ppo.evaluate `
        --checkpoint $Ckpt `
        --reward     $Reward `
        --episodes   $NEpisodes `
        --seed       42 `
        --out        $OutJson `
        --notes      "BC_init NSteps=$NSteps BCEpochs=$BCEpochs TotalSteps=$TotalSteps seed=$Seed"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Evaluation failed." -ForegroundColor Red
        exit 1
    }
    Write-Host "Result saved to: $OutJson" -ForegroundColor Green
} else {
    Write-Host "[skip] Evaluation skipped." -ForegroundColor Yellow
}

# ── Demo ──────────────────────────────────────────────────────────────────────
if (-not $SkipDemo) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host " Demo (fps=$Fps, ESC to quit)" -ForegroundColor Cyan
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
