#!/bin/bash

# HiDream Example Usage Scripts
# This file contains example commands for using the HiDream models

echo "HiDream Example Usage"
echo "===================="

# Basic image generation with I1-Full model
echo "1. Basic image generation:"
echo "cargo run --example hidream --release -- --prompt \"A cat holding a sign that says 'Hi-Dreams.ai'\""
echo ""

# Fast generation with I1-Fast model
echo "2. Fast generation:"
echo "cargo run --example hidream --release -- --model i1-fast --prompt \"A robot in space\""
echo ""

# High quality generation with custom parameters
echo "3. High quality generation:"
echo "cargo run --example hidream --release -- \\"
echo "    --model i1-full \\"
echo "    --prompt \"A cyberpunk cityscape with neon lights\" \\"
echo "    --height 1360 \\"
echo "    --width 768 \\"
echo "    --guidance-scale 7.5 \\"
echo "    --num-inference-steps 50 \\"
echo "    --seed 123 \\"
echo "    --output cyberpunk_city.jpg"
echo ""

# Image editing with E1 model
echo "4. Image editing (requires input image):"
echo "cargo run --example hidream --release -- \\"
echo "    --model e1-full \\"
echo "    --prompt \"Editing Instruction: Convert to anime style. Target Image Description: An anime-style version with vibrant colors.\" \\"
echo "    --input-image input.jpg \\"
echo "    --guidance-scale 5.0 \\"
echo "    --image-guidance-scale 4.0 \\"
echo "    --output anime_style.jpg"
echo ""

# Style transfer example
echo "5. Style transfer:"
echo "cargo run --example hidream --release -- \\"
echo "    --model e1-full \\"
echo "    --prompt \"Editing Instruction: Convert to Ghibli style. Target Image Description: A Ghibli-style version with soft colors and artistic rendering.\" \\"
echo "    --input-image photo.jpg \\"
echo "    --negative-prompt \"low resolution, blur, artifacts\" \\"
echo "    --output ghibli_style.jpg"
echo ""

echo "Note: Make sure you have the required models downloaded and proper GPU setup for best performance."
echo "For CPU-only usage, add --cpu flag to any command."
