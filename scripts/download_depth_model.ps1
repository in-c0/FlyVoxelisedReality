$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$modelDir = Join-Path $repoRoot "models"
$modelPath = Join-Path $modelDir "model-small.onnx"
$modelUrl = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"

New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

if (Test-Path $modelPath) {
    $sizeMb = [math]::Round((Get-Item $modelPath).Length / 1MB, 1)
    Write-Host "Depth model already exists: $modelPath ($sizeMb MB)"
    exit 0
}

Write-Host "Downloading MiDaS v2.1 Small ONNX depth model..."
Write-Host $modelUrl

& curl.exe -L --fail --progress-bar $modelUrl -o $modelPath
if ($LASTEXITCODE -ne 0) {
    Remove-Item $modelPath -ErrorAction SilentlyContinue
    throw "Depth-model download failed (curl exit code $LASTEXITCODE)."
}

$sizeMb = [math]::Round((Get-Item $modelPath).Length / 1MB, 1)
if ((Get-Item $modelPath).Length -lt 1MB) {
    Remove-Item $modelPath -ErrorAction SilentlyContinue
    throw "Downloaded file is unexpectedly small; refusing to use it as a model."
}

Write-Host "Saved: $modelPath ($sizeMb MB)"
Write-Host "M1 can now run real relative monocular depth."
