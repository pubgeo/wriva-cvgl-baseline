#!/usr/bin/env bash

set -euo pipefail

SITE="/WRIVA-CVGL-DEV-001"
OUTPUT_DIR="output_plots/$SITE"
BASE_DIR="/home/yejz1/wriva/CVGL/inference_outputs/test_full_pipeline/$SITE/evaluation_against_gt"
SAT_PATH="/home/yejz1/wriva/data/WRIVA-CVGL-Datasets-V3/WRIVA-CVGL-DEV-001/maxar/2025-01-23-48d16e49-49e7-3145-674a-091ebc87e671.tif"
# SAT_PATH="/home/yejz1/wriva/data/WRIVA-CVGL-Datasets-V3/$SITE/maxar/2024-09-14-ad2da7ce-1a2d-5b1b-3968-ffcb83b9ef8c.tif"

mkdir -p "$OUTPUT_DIR"

FILES=(
    neighbor_postsum_all_satellite_aggregate_eval
    neighbor_postsum_best_satellite_eval
    retrieval_all_satellite_aggregate_eval
    retrieval_best_satellite_eval
)

for name in "${FILES[@]}"; do
    python inference_utils/visualize_evaluations.py \
        --txt_path "$BASE_DIR/${name}.txt" \
        --satellite_path "$SAT_PATH" \
        --output_path "$OUTPUT_DIR/${name}.png"
done