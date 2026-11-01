$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$ModelsDir = Join-Path $ProjectRoot "assets\models"
$FontsDir = Join-Path $ProjectRoot "assets\fonts"
$DataDir = Join-Path $ProjectRoot "data"

New-Item -ItemType Directory -Force -Path $ModelsDir, $FontsDir, $DataDir | Out-Null

$PalmUrl = "https://huggingface.co/opencv/palm_detection_mediapipe/resolve/main/palm_detection_mediapipe_2023feb.onnx"
$HandUrl = "https://huggingface.co/opencv/handpose_estimation_mediapipe/resolve/main/handpose_estimation_mediapipe_2023feb.onnx"
$FontUrl = "https://raw.githubusercontent.com/JetBrains/JetBrainsMono/master/fonts/ttf/JetBrainsMono-Bold.ttf"

$PalmOut = Join-Path $ModelsDir "palm_detection_full.onnx"
$HandOut = Join-Path $ModelsDir "hand_landmark_full.onnx"
$FontOut = Join-Path $FontsDir "JetBrainsMono-Bold.ttf"

Write-Host "Downloading palm_detection_full.onnx..."
Invoke-WebRequest -Uri $PalmUrl -OutFile $PalmOut -UseBasicParsing

Write-Host "Downloading hand_landmark_full.onnx..."
Invoke-WebRequest -Uri $HandUrl -OutFile $HandOut -UseBasicParsing

Write-Host "Downloading JetBrainsMono-Bold.ttf..."
try {
    Invoke-WebRequest -Uri $FontUrl -OutFile $FontOut -UseBasicParsing
} catch {
    Write-Warning "Font download skipped; default font will be used."
}

Write-Host "Done."
