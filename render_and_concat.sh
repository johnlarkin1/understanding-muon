#!/bin/bash

# Simple script to render both scenes and concatenate them
set -e

SCRIPT="understanding_muon/viz/muon_overview_clean_v2.py"
QUALITY="${1:-p}"  # Default to production quality (1080p60)

echo "Rendering MuonOverview3D..."
uv run manim render -q${QUALITY} ${SCRIPT} MuonOverview3D

echo ""
echo "Rendering MuonGradient2D..."
uv run manim render -q${QUALITY} ${SCRIPT} MuonGradient2D

echo ""
echo "Concatenating videos..."

# Find the output directory (manim creates it based on quality)
VIDEO_DIR=$(find media/videos/muon_overview_clean_v2 -type d -mindepth 1 -maxdepth 1 | head -n 1)
VIDEO1="${VIDEO_DIR}/MuonOverview3D.mp4"
VIDEO2="${VIDEO_DIR}/MuonGradient2D.mp4"

# Create concat file for ffmpeg
echo "file '$(pwd)/${VIDEO1}'" > /tmp/concat.txt
echo "file '$(pwd)/${VIDEO2}'" >> /tmp/concat.txt

# Concatenate
OUTPUT="${VIDEO_DIR}/MuonOverview_Complete.mp4"
ffmpeg -f concat -safe 0 -i /tmp/concat.txt -c copy "${OUTPUT}" -y

rm /tmp/concat.txt

echo ""
echo "✓ Done! Output: ${OUTPUT}"
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${OUTPUT}" | awk '{printf "Duration: %.1f seconds\n", $1}'
