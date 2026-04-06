# --- 1. 初始化配置 ---
$Ticker = "512760"
$NTrials = 10000
$Timestamp = Get-Date -Format "yyyyMMdd_HHmm"
$LogDir = "logs/bombing_$Timestamp"

# 确保目录存在
if (!(Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" }
if (!(Test-Path "reports")) { New-Item -ItemType Directory -Path "reports" }
New-Item -ItemType Directory -Force -Path $LogDir

Write-Host "------------------------------------------------" 
Write-Host "Start Windows 16-Core Optimization Pipeline"
Write-Host "Target: $Ticker | Logs: $LogDir"
Write-Host "------------------------------------------------"

# --- 2. 第一阶段：变种子测试 ---
$Seeds = @(42, 123, 888)

foreach ($Seed in $Seeds) {
    Write-Host "Running Seed Test | Current Seed: $Seed ..." -ForegroundColor Yellow
    
    # 执行寻优指令 (注意：这里去掉了多余的管道，直接执行)
    # 确保 python 路径正确，如果不行请改用 python.exe
    python tools/optimizer/run_optimizer.py --ticker $Ticker --mode unwind --n_trials $NTrials --seed $Seed 

    # 归档该轮生成的报告
    $LatestReport = "reports/audit_report_$Ticker`_LATEST.md"
    if (Test-Path $LatestReport) {
        $DestReport = "$LogDir/audit_$Ticker`_seed$Seed.md"
        Copy-Item -Path $LatestReport -Destination $DestReport -Force
    }
    
    # 归档参数 JSON
    $ParamFile = "configs/$Ticker`_params.json"
    if (Test-Path $ParamFile) {
        Copy-Item -Path $ParamFile -Destination "$LogDir/params_seed$Seed.json" -Force
    }
}

# --- 3. 第二阶段：时间窗口专项测试 ---

Write-Host "Running 2022 Bear Market Test..." -ForegroundColor Red
python tools/optimizer/run_optimizer.py --ticker $Ticker --mode unwind --n_trials $NTrials --start_date "2022-01-01" --end_date "2022-12-31" --seed 42

if (Test-Path "reports/audit_report_$Ticker`_LATEST.md") {
    Copy-Item "reports/audit_report_$Ticker`_LATEST.md" "$LogDir/audit_$Ticker`_2022_bear.md" -Force
}

Write-Host "Optimization Task Completed!" -ForegroundColor Green