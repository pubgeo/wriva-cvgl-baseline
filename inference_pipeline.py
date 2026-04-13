import argparse
import json
from pathlib import Path

from inference_utils.prepare_camera_clusters import (
    build_runtime_args as build_camera_cluster_args,
    calculate_camera_clusters
)

from inference_utils.infer_tiles import infer_tiles

from inference_utils.infer_tiles_neighbor_postsum import infer_tiles_neighbor_postsum

from inference_utils.evaluate_cluster_predictions_against_gt import evaluate_cluster_predictions_against_gt

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on tiled clusters")
    parser.add_argument(
        "--config",
        type=str,
        default="inference_configs/infer_pipeline.json",
        help="Path to inference config file",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    with open(Path(args.config), 'r') as f:
        full_cfg = json.load(f)

    global_cfg = full_cfg['global']
    cluster_prep_cfg = full_cfg["prepare_camera_clusters"]
    inference_cfg = full_cfg['infer_tiles']
    postsum_cfg = full_cfg["infer_tiles_neighbor_postsum"]
    gt_eval_cfg = full_cfg["evaluate_cluster_predictions_against_gt"]

    camera_cluster_args = build_camera_cluster_args(global_cfg, cluster_prep_cfg)
    calculate_camera_clusters(args=camera_cluster_args)

    dataset_path = Path(global_cfg['dataset_root'])
    dataset_name = dataset_path.name
    inference_cfg['data']['txt_file'] = Path(global_cfg['output_root']) / dataset_name / global_cfg['registration_subdir_name'] / global_cfg['camera_clusters_filename']
    infer_tiles(infer_cfg=inference_cfg, global_cfg=global_cfg)

    infer_tiles_neighbor_postsum(global_cfg, postsum_cfg)


    evaluate_cluster_predictions_against_gt(global_cfg, gt_eval_cfg)


if __name__=="__main__": 
    main()