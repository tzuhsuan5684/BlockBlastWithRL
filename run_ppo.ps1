#!/usr/bin/env pwsh
<#
.SYNOPSIS
    PPO train / eval / demo automation script.

.DESCRIPTION
    Modes:
        train   -- train PPO (sparse / dense / both)
        eval    -- evaluate latest checkpoint
        demo    -- pygame visualizer
        all     -- train then eval (default)

.EXAMPLES
    .\run_ppo.ps1                              # default: all, dense reward
    .\run_ppo.ps1 -Mode train -Reward sparse
    .\run_ppo.ps1 -Mode train -Reward both
    .\run_ppo.ps1 -Mode eval  -Reward dense
    .\run_ppo.ps1 -Mode demo
    .\run_ppo.ps1 -Mode all   -TotalSteps 1000000
#>

param(
    [ValidateSet("train","eval","demo","all")]
    [string]$Mode        = "all",

    [ValidateSet("sparse","dense","both")]
    [string]$Reward      = "dense",

    [int]   $Seed        = 0,
    [int]   $TotalSteps  = 500000,
    [int]   $NEpisodes   = 100,
    [int]   $NEnvs       = 8,
    [string]$CkptDir     = "checkpoints",
    [string]$ResultsDir  = "results",
    [string]$Checkpoint  = "",
    [int]   $DemoFps     = 10
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------

function Write-Header([string]$msg) {
    Write-Host ""
    Write-Host "=======================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "=======================================" -ForegroundColor Cyan
}

function Get-LatestCheckpoint([string]$reward) {
    $files = Get-ChildItem -Path $CkptDir -Filter "ppo_${reward}_seed${Seed}_step*.pt" `
                           -ErrorAction SilentlyContinue |
             Sort-Object { [int]($_.BaseName -replace '.*_step','') } -Descending
    if (-not $files) { return $null }
    return $files[0].FullName
}

# ---------------------------------------------------------------------------

function Invoke-Train([string]$reward) {
    Write-Header "TRAIN  reward=$reward  steps=$TotalSteps  seed=$Seed"
    uv run python -m agents.ppo.train_ppo `
        --reward      $reward `
        --seed        $Seed `
        --total-steps $TotalSteps `
        --n-envs      $NEnvs `
        --ckpt-dir    $CkptDir
    if ($LASTEXITCODE -ne 0) { throw "train_ppo failed (reward=$reward)" }
}

function Invoke-Eval([string]$reward) {
    Write-Header "EVAL  reward=$reward  episodes=$NEpisodes"

    $ckpt = if ($Checkpoint) { $Checkpoint } else { Get-LatestCheckpoint $reward }
    if (-not $ckpt) {
        Write-Host "[skip] no checkpoint found for reward=$reward" -ForegroundColor Yellow
        return
    }
    Write-Host "  checkpoint: $ckpt"

    New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
    $ts      = Get-Date -Format "yyyyMMdd_HHmmss"
    $outFile = "$ResultsDir/ppo_${reward}_seed${Seed}_${ts}.json"

    uv run python -m agents.ppo.evaluate `
        --checkpoint $ckpt `
        --reward     $reward `
        --episodes   $NEpisodes `
        --seed       42 `
        --out        $outFile `
        --notes      "auto-eval seed=$Seed steps=$TotalSteps"
    if ($LASTEXITCODE -ne 0) { throw "evaluate failed (reward=$reward)" }
}

function Invoke-Demo {
    Write-Header "DEMO  (pygame)"

    $ckpt = $Checkpoint
    if (-not $ckpt) { $ckpt = Get-LatestCheckpoint "dense" }
    if (-not $ckpt) { $ckpt = Get-LatestCheckpoint "sparse" }

    if ($ckpt) {
        Write-Host "  agent=ppo  checkpoint=$ckpt"
        uv run python demo/play.py --agent ppo --checkpoint $ckpt --fps $DemoFps
    } else {
        Write-Host "  no checkpoint found -- running greedy agent" -ForegroundColor Yellow
        uv run python demo/play.py --agent greedy --fps $DemoFps
    }
}

# ---------------------------------------------------------------------------

$rewardModes = if ($Reward -eq "both") { @("sparse","dense") } else { @($Reward) }

switch ($Mode) {
    "train" { foreach ($r in $rewardModes) { Invoke-Train $r } }
    "eval"  { foreach ($r in $rewardModes) { Invoke-Eval  $r } }
    "demo"  { Invoke-Demo }
    "all"   {
        foreach ($r in $rewardModes) { Invoke-Train $r }
        foreach ($r in $rewardModes) { Invoke-Eval  $r }
    }
}

Write-Host ""
Write-Host "done." -ForegroundColor Green
