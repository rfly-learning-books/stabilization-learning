param(
    [ValidateSet('train','preflight','doctor')][string]$Mode='train',
    [string]$Config='configs/public/observer_cpu_smoke.json',
    [string]$Python='python',
    [Parameter(Mandatory=$true)][string]$Output
)
$ErrorActionPreference='Stop'
$projectPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pythonPath = $Python
$env:PYTHONPATH=$projectPath+';'+(Join-Path $projectPath 'stabilization-gym')+';'+(Join-Path $projectPath 'stabilization-learning-control')
$env:PYTHONHASHSEED='0'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUDA_VISIBLE_DEVICES='-1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
Push-Location $projectPath
try {
    & $pythonPath -X utf8 (Join-Path $PSScriptRoot 'run_observer_training_v5.py') $Mode --config $Config --output $Output
    if ($LASTEXITCODE -ne 0) { throw "Observer $Mode failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
