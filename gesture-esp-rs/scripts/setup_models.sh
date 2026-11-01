#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS_DIR="${PROJECT_ROOT}/assets/models"
FONTS_DIR="${PROJECT_ROOT}/assets/fonts"

mkdir -p "${MODELS_DIR}" "${FONTS_DIR}" "${PROJECT_ROOT}/data"

PALM_URL="https://huggingface.co/opencv/palm_detection_mediapipe/resolve/main/palm_detection_mediapipe_2023feb.onnx"
HAND_URL="https://huggingface.co/opencv/handpose_estimation_mediapipe/resolve/main/handpose_estimation_mediapipe_2023feb.onnx"
FONT_URL="https://raw.githubusercontent.com/JetBrains/JetBrainsMono/master/fonts/ttf/JetBrainsMono-Bold.ttf"

echo "Downloading palm_detection_full.onnx..."
curl -fL "${PALM_URL}" -o "${MODELS_DIR}/palm_detection_full.onnx"

echo "Downloading hand_landmark_full.onnx..."
curl -fL "${HAND_URL}" -o "${MODELS_DIR}/hand_landmark_full.onnx"

echo "Downloading JetBrainsMono-Bold.ttf..."
curl -fL "${FONT_URL}" -o "${FONTS_DIR}/JetBrainsMono-Bold.ttf" || echo "Font download skipped."

echo "Done."
