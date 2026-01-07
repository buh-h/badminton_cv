#!/usr/bin/env bash

TRACKNET_DIR=""
VIDEO_DIR="$(pwd)"
OUTPUT_DIR="$(pwd)/prediction"
LARGE_VIDEO=""
BATCH_SIZE="1"
MAX_SAMPLE="500"


while [[ $# -gt 0 ]]; do
  case $1 in
    -t | --tracknet)
      TRACKNET_DIR="$2"
      shift 2
      ;;
    -v | --video_dir)
      VIDEO_DIR="$2"
      shift 2
      ;;
    -c | --tracknet_ckpt)
      TRACKNET_CKPT="$2"
      shift 2
      ;;
    -i | --inpaintnet_ckpt)
      INPAINTNET_CKPT="$2"
      shift 2
      ;;
    -o | --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -l | --large-video)
      LARGE_VIDEO="--large_video"
      shift 1
      ;;
    -b | --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    -s | --max-sample)
      MAX_SAMPLE="$2"
      shift 2
      ;;
  esac
done

if [ -z "$TRACKNET_DIR" ]; then
  echo "Usage: $0 --tracknet <tracknet_dir>"
  exit 1
fi

TRACKNET_CKPT="${TRACKNET_CKPT:-$TRACKNET_DIR/ckpts/TrackNet_best.pt}"
INPAINTNET_CKPT="${INPAINTNET_CKPT:-$TRACKNET_DIR/ckpts/InpaintNet_best.pt}"


mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
for file in "$VIDEO_DIR"/*.mp4; do
  python3 "$(TRACKNET_DIR)/predict.py" \
    --video_file "$file" \
    --tracknet_file "$TRACKNET_CKPT" \
    --inpaintnet_file "$INPAINTNET_CKPT" \
    --save_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --max_sample_num "$MAX_SAMPLE" \
    $LARGE_VIDEO

done