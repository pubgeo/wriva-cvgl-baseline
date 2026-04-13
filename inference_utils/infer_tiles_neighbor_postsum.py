#!/usr/bin/env python3
"""
Read the dataset-level inference JSON plus camera_clusters.txt and write:

- retrieval_all_sat/best_satellite_summary.json
- retrieval_all_sat/best_satellite_predictions.txt
- retrieval_all_sat/all_satellite_aggregate_summary.json
- retrieval_all_sat/all_satellite_aggregate_predictions.txt
- neighbor_postsum_all_sat/best_satellite_summary.json
- neighbor_postsum_all_sat/best_satellite_predictions.txt
- neighbor_postsum_all_sat/all_satellite_aggregate_summary.json
- neighbor_postsum_all_sat/all_satellite_aggregate_predictions.txt

The inference JSON is expected at:
<output_root>/<dataset_name>/all_results.json

Legacy fallback:
<output_root>/<dataset_name>/<DATASET_NAME_AS_UNDERSCORES>_inference_pos.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from inference_utils.config_helpers import (
    add_config_argument,
    get_global_config,
    load_pipeline_config,
    require_config_section,
    require_config_value,
)


SAFE_RE = re.compile(r"[^A-Za-z0-9]+")
NEIGHBOR_MODE_CHOICES = ("local_xy_radius", "order_window")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read the dataset inference JSON plus registration/camera_clusters.txt and "
            "write per-cluster best-satellite and aggregate predictions before and after "
            "neighbor sum using infer_tiles.jsonc."
        )
    )
    add_config_argument(parser)
    return parser.parse_args(argv)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def dataset_token(name: str) -> str:
    return SAFE_RE.sub("_", name).strip("_")


def resolve_inference_json_path(dataset_out_root: Path, dataset_name: str) -> Path:
    candidates = [
        (dataset_out_root / "all_results.json").resolve(),
        (dataset_out_root / "inference" / "all_results.json").resolve(),
        (dataset_out_root / f"{dataset_token(dataset_name)}_inference_pos.json").resolve(),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    tried_paths = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find inference results JSON. Tried:\n"
        f"{tried_paths}"
    )


def cluster_label(local_cluster_id: int) -> str:
    return f"cluster_{int(local_cluster_id):03d}"


def load_camera_clusters(camera_clusters_txt: Path) -> Dict[str, Dict[str, Any]]:
    cluster_info_by_id: Dict[str, Dict[str, Any]] = {}
    with camera_clusters_txt.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            local_cluster_id = int(row["local_cluster_id"])
            global_cluster_id = int(row["global_cluster_id"])
            cluster_id = cluster_label(local_cluster_id)
            local_x = float(row["x"])
            local_y = float(row["y"])
            cluster_info = cluster_info_by_id.setdefault(
                cluster_id,
                {
                    "cluster_id": cluster_id,
                    "local_cluster_id": local_cluster_id,
                    "global_cluster_id": global_cluster_id,
                    "ground_filenames": [],
                    "local_xy_values": [],
                },
            )
            cluster_info["ground_filenames"].append(str(row["ground_filename"]))
            if not (math.isclose(local_x, -1.0) and math.isclose(local_y, -1.0)):
                cluster_info["local_xy_values"].append((float(local_x), float(local_y)))
    for cluster_info in cluster_info_by_id.values():
        cluster_info["ground_filenames"].sort()
        if cluster_info["local_xy_values"]:
            mean_x = sum(value[0] for value in cluster_info["local_xy_values"]) / float(len(cluster_info["local_xy_values"]))
            mean_y = sum(value[1] for value in cluster_info["local_xy_values"]) / float(len(cluster_info["local_xy_values"]))
            cluster_info["mean_local_xy"] = (float(mean_x), float(mean_y))
        else:
            cluster_info["mean_local_xy"] = None
    return cluster_info_by_id


def load_inference_records(inference_json: Path) -> Dict[str, Dict[str, Any]]:
    records = load_json(inference_json)
    if not isinstance(records, list):
        raise RuntimeError(f"Expected top-level list in {inference_json}, got {type(records)}")

    records_by_id: Dict[str, Dict[str, Any]] = {}
    for record in records:
        local_cluster_id = int(record["cluster_id"])
        cluster_id = cluster_label(local_cluster_id)
        satellite_details: Dict[int, Dict[str, Any]] = {}

        for sat_record in record["all_chips_per_satellite"]:
            sat_index = int(sat_record["sat_index"])
            sat_path = str(sat_record["sat_path"])
            chips: List[Dict[str, Any]] = []
            for chip in sat_record["chips"]:
                chip_box = [int(value) for value in chip["chip_box_xyxy"]]
                x1, y1, x2, y2 = chip_box
                chips.append(
                    {
                        "chip_index_local": int(chip["chip_index_local"]),
                        "score": float(chip["score"]),
                        "chip_box_xyxy": chip_box,
                        "x": int(x1),
                        "y": int(y1),
                        "chip_size": int(x2 - x1),
                        "tiled_area_lrtb": [float(value) for value in chip["tiled_area_lrtb"]],
                        "pos_xy_preds": [
                            [float(pair[0]), float(pair[1])]
                            for pair in chip["pos_xy_preds"]
                        ],
                    }
                )
            if not chips:
                continue
            satellite_details[sat_index] = {
                "sat_index": sat_index,
                "sat_path": sat_path,
                "chips": chips,
                "chips_by_box": {
                    tuple(int(value) for value in chip["chip_box_xyxy"]): chip
                    for chip in chips
                },
            }

        records_by_id[cluster_id] = {
            "cluster_id": cluster_id,
            "local_cluster_id": local_cluster_id,
            "site_id": str(record["site_id"]),
            "ground_paths": [str(path) for path in record["ground_paths"]],
            "ground_filenames": [Path(str(path)).name for path in record["ground_paths"]],
            "sat_paths": [str(path) for path in record["sat_paths"]],
            "all_scores": [float(score) for score in record["all_scores"]],
            "top_sat_indices": [int(value) for value in record["top_sat_indices"]],
            "top_sat_paths": [str(path) for path in record["top_sat_paths"]],
            "top_scores": [float(score) for score in record["top_scores"]],
            "n_ground": int(record["n_ground"]),
            "n_sat_candidates": int(record["n_sat_candidates"]),
            "satellite_details": satellite_details,
        }
    return records_by_id


def combine_chip_lists(chip_lists: Sequence[Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    score_by_box: Dict[Tuple[int, int, int, int], float] = {}
    tiled_area_by_box: Dict[Tuple[int, int, int, int], List[float]] = {}
    chip_size: Optional[int] = None

    for chip_list in chip_lists:
        for chip in chip_list:
            chip_box = tuple(int(value) for value in chip["chip_box_xyxy"])
            score_by_box[chip_box] = score_by_box.get(chip_box, 0.0) + float(chip["score"])
            tiled_area_by_box[chip_box] = [float(value) for value in chip["tiled_area_lrtb"]]
            if chip_size is None:
                chip_size = int(chip["chip_size"])

    if not score_by_box or chip_size is None:
        raise RuntimeError("No chip scores were available for aggregation")

    xs = sorted({chip_box[0] for chip_box in score_by_box})
    ys = sorted({chip_box[1] for chip_box in score_by_box})
    x_to_col = {value: index for index, value in enumerate(xs)}
    y_to_row = {value: index for index, value in enumerate(ys)}

    chips: List[Dict[str, Any]] = []
    best_chip: Optional[Dict[str, Any]] = None
    for chip_box, score in score_by_box.items():
        x1, y1, x2, y2 = chip_box
        chip_record = {
            "row": int(y_to_row[y1]),
            "col": int(x_to_col[x1]),
            "x": int(x1),
            "y": int(y1),
            "score": float(score),
            "chip_box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "chip_size": int(chip_size),
            "center_xy": [float(x1) + 0.5 * float(chip_size), float(y1) + 0.5 * float(chip_size)],
            "tiled_area_lrtb": list(tiled_area_by_box[chip_box]),
        }
        chips.append(chip_record)
        if best_chip is None or float(chip_record["score"]) > float(best_chip["score"]):
            best_chip = dict(chip_record)

    return {
        "chip_size": int(chip_size),
        "chips": sorted(chips, key=lambda item: (int(item["row"]), int(item["col"]))),
        "best_chip": best_chip,
        "n_chips": int(len(chips)),
        "peak_score": float(best_chip["score"]),
    }


def softmax_weights(values: Sequence[float]) -> List[float]:
    values = [float(value) for value in values]
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / float(len(values)) for _ in values]
    return [value / total for value in exps]


def local_xy_to_global_xy(
    local_xy: Sequence[float],
    chip_box_xyxy: Sequence[int],
) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in chip_box_xyxy]
    chip_w = x2 - x1
    chip_h = y2 - y1
    return (
        float(x1 + (float(local_xy[0]) + 1.0) * 0.5 * chip_w),
        float(y1 + (float(local_xy[1]) + 1.0) * 0.5 * chip_h),
    )


def mean_xy(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    if not points:
        raise RuntimeError("No positions were available")
    return (
        float(sum(point[0] for point in points) / float(len(points))),
        float(sum(point[1] for point in points) / float(len(points))),
    )


def write_prediction_rows_txt(path: Path, cluster_records: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "cluster_id",
                "global_cluster_id",
                "image_filename",
                "pred_retrieval_x_px",
                "pred_retrieval_y_px",
                "pred_position_head_x_px",
                "pred_position_head_y_px",
            ]
        )
        for cluster_record in cluster_records:
            for image_record in cluster_record["per_image_position_head_correction"]:
                pred_before = image_record["pred_xy_before_pos_head"]
                pred_after = image_record["pred_xy_after_pos_head"]
                writer.writerow(
                    [
                        str(cluster_record["cluster_id"]),
                        int(cluster_record["global_cluster_id"]),
                        Path(str(image_record["image_path"])).name,
                        "" if pred_before is None else float(pred_before[0]),
                        "" if pred_before is None else float(pred_before[1]),
                        "" if pred_after is None else float(pred_after[0]),
                        "" if pred_after is None else float(pred_after[1]),
                    ]
                )


def infer_tiles_neighbor_postsum(
    global_cfg,
    script_cfg
) -> int:
    dataset_root = Path(
        str(require_config_value(global_cfg, "dataset_root"))
    ).expanduser().resolve()
    output_root = Path(
        str(require_config_value(global_cfg, "output_root"))
    ).expanduser().resolve()
    retrieval_dir_name = str(require_config_value(global_cfg, "retrieval_dir_name"))
    postsum_dir_name = str(require_config_value(global_cfg, "postsum_dir_name"))
    registration_dir_name = str(require_config_value(global_cfg, "registration_subdir_name"))
    camera_clusters_filename = str(require_config_value(global_cfg, "camera_clusters_filename"))
    neighbor_mode = str(require_config_value(script_cfg, "neighbor_mode")).strip().lower()
    neighbor_search_radius = float(require_config_value(script_cfg, "neighbor_search_radius"))
    order_neighbor_before = int(require_config_value(script_cfg, "order_neighbor_before"))
    order_neighbor_after = int(require_config_value(script_cfg, "order_neighbor_after"))
    if neighbor_mode not in NEIGHBOR_MODE_CHOICES:
        raise RuntimeError(
            f"Unsupported neighbor_mode={neighbor_mode!r}; use one of {list(NEIGHBOR_MODE_CHOICES)}"
        )

    dataset_name = dataset_root.name
    dataset_out_root = (output_root / dataset_name).resolve()
    inference_json = resolve_inference_json_path(
        dataset_out_root=dataset_out_root,
        dataset_name=dataset_name,
    )
    camera_clusters_txt = (dataset_out_root / registration_dir_name / camera_clusters_filename).resolve()
    retrieval_root = (dataset_out_root / retrieval_dir_name).resolve()
    postsum_root = (dataset_out_root / postsum_dir_name).resolve()
    retrieval_root.mkdir(parents=True, exist_ok=True)
    postsum_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Using inference results: {inference_json}", flush=True)

    cluster_info_by_id = load_camera_clusters(camera_clusters_txt)
    inference_by_id = load_inference_records(inference_json)

    cluster_ids_in_order = sorted(inference_by_id.keys(), key=lambda value: inference_by_id[value]["local_cluster_id"])
    neighbors_by_cluster_id: Dict[str, List[str]] = {}
    if neighbor_mode == "local_xy_radius":
        if not all(cluster_info_by_id[cluster_id]["mean_local_xy"] is not None for cluster_id in cluster_ids_in_order):
            raise RuntimeError(
                "NEIGHBOR_MODE=local_xy_radius requires real x/y values in camera_clusters.txt for every cluster"
            )
        for cluster_id in cluster_ids_in_order:
            cluster_meta = cluster_info_by_id[cluster_id]
            cluster_xy = cluster_meta["mean_local_xy"]
            neighbors: List[str] = []
            for other_cluster_id in cluster_ids_in_order:
                other_meta = cluster_info_by_id[other_cluster_id]
                if int(other_meta["global_cluster_id"]) != int(cluster_meta["global_cluster_id"]):
                    continue
                other_xy = other_meta["mean_local_xy"]
                distance = math.hypot(
                    float(cluster_xy[0]) - float(other_xy[0]),
                    float(cluster_xy[1]) - float(other_xy[1]),
                )
                if distance <= float(neighbor_search_radius):
                    neighbors.append(other_cluster_id)
            neighbors_by_cluster_id[cluster_id] = neighbors
        print(
            f"[INFO] Neighbor mode=local_xy_radius radius={float(neighbor_search_radius):.1f}",
            flush=True,
        )
    elif neighbor_mode == "order_window":
        for cluster_index, cluster_id in enumerate(cluster_ids_in_order):
            start = max(0, cluster_index - int(order_neighbor_before))
            end = min(len(cluster_ids_in_order), cluster_index + int(order_neighbor_after) + 1)
            neighbors_by_cluster_id[cluster_id] = list(cluster_ids_in_order[start:end])
        print(
            f"[INFO] Neighbor mode=order_window before={int(order_neighbor_before)} after={int(order_neighbor_after)}",
            flush=True,
        )
    else:
        raise RuntimeError(
            f"Unsupported NEIGHBOR_MODE={neighbor_mode!r}; use 'local_xy_radius' or 'order_window'"
        )

    pre_best_clusters: List[Dict[str, Any]] = []
    pre_all_aggregate_clusters: List[Dict[str, Any]] = []
    post_best_clusters: List[Dict[str, Any]] = []
    post_all_aggregate_clusters: List[Dict[str, Any]] = []

    for cluster_id in cluster_ids_in_order:
        if cluster_id not in cluster_info_by_id:
            raise RuntimeError(f"{cluster_id} exists in inference JSON but not in {camera_clusters_txt}")

        cluster_meta = cluster_info_by_id[cluster_id]
        inference_record = inference_by_id[cluster_id]
        satellite_details = inference_record["satellite_details"]
        available_sat_indices = sorted(satellite_details.keys())
        if not available_sat_indices:
            raise RuntimeError(f"{cluster_id} has no satellite chip scores in {inference_json}")

        best_sat_index_pre = max(
            available_sat_indices,
            key=lambda sat_index: (
                float(inference_record["all_scores"][sat_index]),
                -float(sat_index),
            ),
        )
        best_satellite_pre = satellite_details[best_sat_index_pre]
        best_chip_pre = dict(combine_chip_lists([best_satellite_pre["chips"]])["best_chip"])
        best_chip_box_pre = tuple(int(value) for value in best_chip_pre["chip_box_xyxy"])
        best_chip_for_pos_pre = best_satellite_pre["chips_by_box"][best_chip_box_pre]
        pre_best_retrieval_xy = tuple(float(value) for value in best_chip_pre["center_xy"])
        pre_best_position_xy_per_image: List[Tuple[float, float]] = []
        pre_best_per_image: List[Dict[str, Any]] = []
        for image_index, image_path in enumerate(inference_record["ground_paths"]):
            pred_after = local_xy_to_global_xy(
                best_chip_for_pos_pre["pos_xy_preds"][image_index],
                best_chip_pre["chip_box_xyxy"],
            )
            pre_best_position_xy_per_image.append(pred_after)
            pre_best_per_image.append(
                {
                    "index": int(image_index),
                    "image_path": str(image_path),
                    "pred_xy_before_pos_head": [float(pre_best_retrieval_xy[0]), float(pre_best_retrieval_xy[1])],
                    "pred_xy_after_pos_head": [float(pred_after[0]), float(pred_after[1])],
                }
            )
        pre_best_position_xy = mean_xy(pre_best_position_xy_per_image)
        pre_best_clusters.append(
            {
                "cluster_id": cluster_id,
                "global_cluster_id": int(cluster_meta["global_cluster_id"]),
                "selected_sat_index": int(best_sat_index_pre),
                "selected_sat_path": str(inference_record["sat_paths"][best_sat_index_pre]),
                "best_chip": dict(best_chip_pre),
                "selected_peak_score": float(inference_record["all_scores"][best_sat_index_pre]),
                "pred_global_xy": [float(pre_best_position_xy[0]), float(pre_best_position_xy[1])],
                "per_image_position_head_correction": pre_best_per_image,
            }
        )

        pre_aggregate_chip_map = combine_chip_lists([satellite_details[sat_index]["chips"] for sat_index in available_sat_indices])
        pre_aggregate_best_chip = dict(pre_aggregate_chip_map["best_chip"])
        pre_aggregate_best_chip_box = tuple(int(value) for value in pre_aggregate_best_chip["chip_box_xyxy"])
        pre_aggregate_scores = [float(inference_record["all_scores"][sat_index]) for sat_index in available_sat_indices]
        pre_aggregate_weights = softmax_weights(pre_aggregate_scores)
        pre_aggregate_retrieval_xy = tuple(float(value) for value in pre_aggregate_best_chip["center_xy"])
        pre_aggregate_position_xy_per_image: List[Tuple[float, float]] = []
        pre_aggregate_per_image: List[Dict[str, Any]] = []
        pre_aggregate_chips = [
            satellite_details[sat_index]["chips_by_box"][pre_aggregate_best_chip_box]
            for sat_index in available_sat_indices
        ]
        for image_index, image_path in enumerate(inference_record["ground_paths"]):
            sum_x = 0.0
            sum_y = 0.0
            for chip, weight in zip(pre_aggregate_chips, pre_aggregate_weights):
                pred_after = local_xy_to_global_xy(
                    chip["pos_xy_preds"][image_index],
                    pre_aggregate_best_chip["chip_box_xyxy"],
                )
                sum_x += float(weight) * float(pred_after[0])
                sum_y += float(weight) * float(pred_after[1])
            pre_aggregate_position_xy_per_image.append((float(sum_x), float(sum_y)))
            pre_aggregate_per_image.append(
                {
                    "index": int(image_index),
                    "image_path": str(image_path),
                    "pred_xy_before_pos_head": [float(pre_aggregate_retrieval_xy[0]), float(pre_aggregate_retrieval_xy[1])],
                    "pred_xy_after_pos_head": [float(sum_x), float(sum_y)],
                }
            )
        pre_aggregate_position_xy = mean_xy(pre_aggregate_position_xy_per_image)
        pre_all_aggregate_clusters.append(
            {
                "cluster_id": cluster_id,
                "global_cluster_id": int(cluster_meta["global_cluster_id"]),
                "satellite_indices_used": [int(value) for value in available_sat_indices],
                "satellite_paths_used": [str(inference_record["sat_paths"][sat_index]) for sat_index in available_sat_indices],
                "satellite_weights": [
                    {
                        "sat_index": int(sat_index),
                        "sat_path": str(inference_record["sat_paths"][sat_index]),
                        "peak_score": float(inference_record["all_scores"][sat_index]),
                        "weight": float(weight),
                    }
                    for sat_index, weight in zip(available_sat_indices, pre_aggregate_weights)
                ],
                "best_chip": dict(pre_aggregate_best_chip),
                "pred_global_xy": [float(pre_aggregate_position_xy[0]), float(pre_aggregate_position_xy[1])],
                "per_image_position_head_correction": pre_aggregate_per_image,
            }
        )

        neighbor_cluster_ids = neighbors_by_cluster_id[cluster_id]
        post_satellite_maps: Dict[int, Dict[str, Any]] = {}
        for sat_index in available_sat_indices:
            chip_lists = []
            for neighbor_cluster_id in neighbor_cluster_ids:
                neighbor_details = inference_by_id[neighbor_cluster_id]["satellite_details"]
                if sat_index in neighbor_details:
                    chip_lists.append(neighbor_details[sat_index]["chips"])
            if chip_lists:
                post_satellite_maps[sat_index] = combine_chip_lists(chip_lists)

        if not post_satellite_maps:
            raise RuntimeError(f"{cluster_id} has no neighbor-summed satellite scores")

        best_sat_index_post = max(
            post_satellite_maps.keys(),
            key=lambda sat_index: (
                float(post_satellite_maps[sat_index]["peak_score"]),
                -float(sat_index),
            ),
        )
        post_best_chip = dict(post_satellite_maps[best_sat_index_post]["best_chip"])
        post_best_chip_box = tuple(int(value) for value in post_best_chip["chip_box_xyxy"])
        post_best_chip_for_pos = satellite_details[best_sat_index_post]["chips_by_box"][post_best_chip_box]
        post_best_retrieval_xy = tuple(float(value) for value in post_best_chip["center_xy"])
        post_best_position_xy_per_image: List[Tuple[float, float]] = []
        post_best_per_image: List[Dict[str, Any]] = []
        for image_index, image_path in enumerate(inference_record["ground_paths"]):
            pred_after = local_xy_to_global_xy(
                post_best_chip_for_pos["pos_xy_preds"][image_index],
                post_best_chip["chip_box_xyxy"],
            )
            post_best_position_xy_per_image.append(pred_after)
            post_best_per_image.append(
                {
                    "index": int(image_index),
                    "image_path": str(image_path),
                    "pred_xy_before_pos_head": [float(post_best_retrieval_xy[0]), float(post_best_retrieval_xy[1])],
                    "pred_xy_after_pos_head": [float(pred_after[0]), float(pred_after[1])],
                }
            )
        post_best_position_xy = mean_xy(post_best_position_xy_per_image)
        post_best_clusters.append(
            {
                "cluster_id": cluster_id,
                "global_cluster_id": int(cluster_meta["global_cluster_id"]),
                "selected_sat_index": int(best_sat_index_post),
                "selected_sat_path": str(inference_record["sat_paths"][best_sat_index_post]),
                "neighbor_cluster_ids": list(neighbor_cluster_ids),
                "best_chip": dict(post_best_chip),
                "selected_peak_score": float(post_satellite_maps[best_sat_index_post]["peak_score"]),
                "pred_global_xy": [float(post_best_position_xy[0]), float(post_best_position_xy[1])],
                "per_image_position_head_correction": post_best_per_image,
            }
        )

        post_aggregate_sat_indices = sorted(post_satellite_maps.keys())
        post_aggregate_chip_map = combine_chip_lists(
            [post_satellite_maps[sat_index]["chips"] for sat_index in post_aggregate_sat_indices]
        )
        post_aggregate_best_chip = dict(post_aggregate_chip_map["best_chip"])
        post_aggregate_best_chip_box = tuple(int(value) for value in post_aggregate_best_chip["chip_box_xyxy"])
        post_aggregate_scores = [float(post_satellite_maps[sat_index]["peak_score"]) for sat_index in post_aggregate_sat_indices]
        post_aggregate_weights = softmax_weights(post_aggregate_scores)
        post_aggregate_retrieval_xy = tuple(float(value) for value in post_aggregate_best_chip["center_xy"])
        post_aggregate_position_xy_per_image: List[Tuple[float, float]] = []
        post_aggregate_per_image: List[Dict[str, Any]] = []
        post_aggregate_chips = [
            satellite_details[sat_index]["chips_by_box"][post_aggregate_best_chip_box]
            for sat_index in post_aggregate_sat_indices
        ]
        for image_index, image_path in enumerate(inference_record["ground_paths"]):
            sum_x = 0.0
            sum_y = 0.0
            for chip, weight in zip(post_aggregate_chips, post_aggregate_weights):
                pred_after = local_xy_to_global_xy(
                    chip["pos_xy_preds"][image_index],
                    post_aggregate_best_chip["chip_box_xyxy"],
                )
                sum_x += float(weight) * float(pred_after[0])
                sum_y += float(weight) * float(pred_after[1])
            post_aggregate_position_xy_per_image.append((float(sum_x), float(sum_y)))
            post_aggregate_per_image.append(
                {
                    "index": int(image_index),
                    "image_path": str(image_path),
                    "pred_xy_before_pos_head": [float(post_aggregate_retrieval_xy[0]), float(post_aggregate_retrieval_xy[1])],
                    "pred_xy_after_pos_head": [float(sum_x), float(sum_y)],
                }
            )
        post_aggregate_position_xy = mean_xy(post_aggregate_position_xy_per_image)
        post_all_aggregate_clusters.append(
            {
                "cluster_id": cluster_id,
                "global_cluster_id": int(cluster_meta["global_cluster_id"]),
                "neighbor_cluster_ids": list(neighbor_cluster_ids),
                "satellite_indices_used": [int(value) for value in post_aggregate_sat_indices],
                "satellite_paths_used": [str(inference_record["sat_paths"][sat_index]) for sat_index in post_aggregate_sat_indices],
                "satellite_weights": [
                    {
                        "sat_index": int(sat_index),
                        "sat_path": str(inference_record["sat_paths"][sat_index]),
                        "peak_score_after_neighbor_sum": float(post_satellite_maps[sat_index]["peak_score"]),
                        "weight": float(weight),
                    }
                    for sat_index, weight in zip(post_aggregate_sat_indices, post_aggregate_weights)
                ],
                "best_chip": dict(post_aggregate_best_chip),
                "pred_global_xy": [float(post_aggregate_position_xy[0]), float(post_aggregate_position_xy[1])],
                "per_image_position_head_correction": post_aggregate_per_image,
            }
        )

    retrieval_best_summary = {
        "dataset_root": str(dataset_root),
        "inference_json": str(inference_json),
        "camera_clusters_txt": str(camera_clusters_txt),
        "selection_metric": "Per cluster, choose the satellite with the highest peak chip score.",
        "clusters": pre_best_clusters,
    }
    retrieval_best_txt = (retrieval_root / "best_satellite_predictions.txt").resolve()
    write_json((retrieval_root / "best_satellite_summary.json").resolve(), retrieval_best_summary)
    write_prediction_rows_txt(retrieval_best_txt, pre_best_clusters)

    retrieval_all_summary = {
        "dataset_root": str(dataset_root),
        "inference_json": str(inference_json),
        "camera_clusters_txt": str(camera_clusters_txt),
        "aggregation": (
            "Per cluster, sum the available chip-score grids across satellites, "
            "choose the peak chip, then decode a weighted local 10x10 position heatmap."
        ),
        "clusters": pre_all_aggregate_clusters,
    }
    retrieval_all_txt = (retrieval_root / "all_satellite_aggregate_predictions.txt").resolve()
    write_json((retrieval_root / "all_satellite_aggregate_summary.json").resolve(), retrieval_all_summary)
    write_prediction_rows_txt(retrieval_all_txt, pre_all_aggregate_clusters)

    postsum_best_summary = {
        "dataset_root": str(dataset_root),
        "inference_json": str(inference_json),
        "camera_clusters_txt": str(camera_clusters_txt),
        "neighbor_mode": neighbor_mode,
        "neighbor_search_radius": None if neighbor_mode != "local_xy_radius" else float(neighbor_search_radius),
        "neighbor_before": None if neighbor_mode != "order_window" else int(order_neighbor_before),
        "neighbor_after": None if neighbor_mode != "order_window" else int(order_neighbor_after),
        "selection_metric": (
            "Per cluster, for each available satellite, sum chip-score grids from neighboring "
            "clusters and choose the satellite with the highest post-sum peak chip score."
        ),
        "clusters": post_best_clusters,
    }
    postsum_best_txt = (postsum_root / "best_satellite_predictions.txt").resolve()
    write_json((postsum_root / "best_satellite_summary.json").resolve(), postsum_best_summary)
    write_prediction_rows_txt(postsum_best_txt, post_best_clusters)

    postsum_all_summary = {
        "dataset_root": str(dataset_root),
        "inference_json": str(inference_json),
        "camera_clusters_txt": str(camera_clusters_txt),
        "neighbor_mode": neighbor_mode,
        "neighbor_search_radius": None if neighbor_mode != "local_xy_radius" else float(neighbor_search_radius),
        "neighbor_before": None if neighbor_mode != "order_window" else int(order_neighbor_before),
        "neighbor_after": None if neighbor_mode != "order_window" else int(order_neighbor_after),
        "aggregation": (
            "Per cluster, first neighbor-sum the available chip-score grids satellite-by-satellite, "
            "then sum those post-sum grids across satellites, choose the peak chip, "
            "and decode a weighted local 10x10 position heatmap."
        ),
        "clusters": post_all_aggregate_clusters,
    }
    postsum_all_txt = (postsum_root / "all_satellite_aggregate_predictions.txt").resolve()
    write_json((postsum_root / "all_satellite_aggregate_summary.json").resolve(), postsum_all_summary)
    write_prediction_rows_txt(postsum_all_txt, post_all_aggregate_clusters)

    write_json(
        (retrieval_root / "run_summary.json").resolve(),
        {
            "dataset_root": str(dataset_root),
            "inference_json": str(inference_json),
            "camera_clusters_txt": str(camera_clusters_txt),
            "best_satellite_summary_json": str((retrieval_root / "best_satellite_summary.json").resolve()),
            "best_satellite_predictions_txt": str(retrieval_best_txt),
            "all_satellite_aggregate_summary_json": str((retrieval_root / "all_satellite_aggregate_summary.json").resolve()),
            "all_satellite_aggregate_predictions_txt": str(retrieval_all_txt),
        },
    )
    write_json(
        (postsum_root / "run_summary.json").resolve(),
        {
            "dataset_root": str(dataset_root),
            "inference_json": str(inference_json),
            "camera_clusters_txt": str(camera_clusters_txt),
            "neighbor_mode": neighbor_mode,
            "neighbor_search_radius": None if neighbor_mode != "local_xy_radius" else float(neighbor_search_radius),
            "neighbor_before": None if neighbor_mode != "order_window" else int(order_neighbor_before),
            "neighbor_after": None if neighbor_mode != "order_window" else int(order_neighbor_after),
            "best_satellite_summary_json": str((postsum_root / "best_satellite_summary.json").resolve()),
            "best_satellite_predictions_txt": str(postsum_best_txt),
            "all_satellite_aggregate_summary_json": str((postsum_root / "all_satellite_aggregate_summary.json").resolve()),
            "all_satellite_aggregate_predictions_txt": str(postsum_all_txt),
        },
    )

    print(f"[DONE] dataset={dataset_name}", flush=True)
    print(f"[DONE] inference_json={inference_json}", flush=True)
    print(f"[DONE] retrieval_root={retrieval_root}", flush=True)
    print(f"[DONE] postsum_root={postsum_root}", flush=True)
    return 0

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_pipeline_config(args.config)
    global_cfg = get_global_config(config)
    script_cfg = require_config_section(config, "infer_tiles_neighbor_postsum")
    infer_tiles_neighbor_postsum(global_cfg=global_cfg, script_cfg=script_cfg)

if __name__ == "__main__":
    raise SystemExit(main())
