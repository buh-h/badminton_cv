#!/usr/bin/env bash

INPUT_DIR="$(pwd)"
OUTPUT_DIR="$(pwd)/resized"
WIDTH=512
HEIGHT=288
CODEC="libx264"
CRF=23
PRESET="medium"

while getopts "i:o:w:h:c" opt; do
  case $opt in
    i) INPUT_DIR="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    w) WIDTH="$OPTARG" ;;
    h) HEIGHT="$OPTARG" ;;
    c) CODEC="$OPTARG" ;;
    *) echo "Unknown flag"; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
for file in "$INPUT_DIR"/*.mp4; do
  filename=$(basename "$file")
  output="$OUTPUT_DIR/$filename"

  echo "Resizing $filename"

  SCALE="-vf scale=$WIDTH:$HEIGHT"
  
  ffmpeg -i "$file" \
    $SCALE \
    -c:v "$CODEC" \
    -preset "$PRESET" \
    -crf "$CRF" \
    -c:a copy \
    "$output"

done