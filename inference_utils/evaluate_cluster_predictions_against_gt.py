#!/usr/bin/env python3
"""
Evaluate the four clustered-inference result variants against GT from reference/.

Results:
1. retrieval_best_satellite
2. retrieval_all_satellite_aggregate
3. neighbor_postsum_best_satellite
4. neighbor_postsum_all_satellite_aggregate
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from inference_utils.config_helpers import (
    add_config_argument,
    get_global_config,
    load_pipeline_config,
    require_config_section,
    require_config_value,
)

try:
    import numpy as np
    import rasterio
    from rasterio.warp import transform as warp_transform
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This script requires numpy and rasterio in the active Python environment. "
        f"Missing module: {exc.name}"
    ) from exc

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate clustered retrieval/postsum outputs against GT from reference/ "
            "for the four result variants using infer_tiles.jsonc."
        )
    )
    add_config_argument(parser)
    return parser.parse_args(argv)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cluster_label(local_cluster_id: int) -> str:
    return f"cluster_{int(local_cluster_id):03d}"


def _find_lat_lon_in_dict(data: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    for lat_key, lon_key in (
        ("lat", "lon"),
        ("latitude", "longitude"),
        ("Lat", "Lon"),
        ("LAT", "LON"),
    ):
        if lat_key in data and lon_key in data:
            try:
                return float(data[lat_key]), float(data[lon_key])
            except Exception:
                return None
    return None


def _walk_find_lat_lon(obj: Any) -> Optional[Tuple[float, float]]:
    if isinstance(obj, dict):
        lat_lon = _find_lat_lon_in_dict(obj)
        if lat_lon is not None:
            return lat_lon
        for value in obj.values():
            lat_lon = _walk_find_lat_lon(value)
            if lat_lon is not None:
                return lat_lon
    elif isinstance(obj, list):
        for value in obj:
            lat_lon = _walk_find_lat_lon(value)
            if lat_lon is not None:
                return lat_lon
    return None


def extract_lat_lon(json_path: Path) -> Optional[Tuple[float, float]]:
    try:
        data = load_json(json_path)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    extrinsics = data.get("extrinsics")
    if isinstance(extrinsics, dict):
        lat_lon = _find_lat_lon_in_dict(extrinsics)
        if lat_lon is not None:
            return lat_lon

    for container in (data.get("metadata"), data):
        if isinstance(container, dict):
            lat_lon = _find_lat_lon_in_dict(container)
            if lat_lon is not None:
                return lat_lon
    return _walk_find_lat_lon(data)


def latlon_to_pixel(dataset: Any, lat: float, lon: float) -> Optional[Tuple[float, float]]:
    try:
        if dataset.crs is None:
            return None
        xs, ys = warp_transform("EPSG:4326", dataset.crs, [float(lon)], [float(lat)])
        if not xs or not ys:
            return None
        inv = ~dataset.transform
        col, row = inv * (float(xs[0]), float(ys[0]))
        if not (math.isfinite(col) and math.isfinite(row)):
            return None
        return float(col), float(row)
    except Exception:
        return None


def is_in_bounds(pixel_xy: Optional[Tuple[float, float]], width: int, height: int) -> bool:
    if pixel_xy is None:
        return False
    x, y = float(pixel_xy[0]), float(pixel_xy[1])
    return 0.0 <= x < float(width) and 0.0 <= y < float(height)


def topk_hit_for_gt_point(
    scores: np.ndarray,
    xs: Sequence[int],
    ys: Sequence[int],
    chip_size: int,
    gt_xy: Tuple[float, float],
    ks: Sequence[int],
) -> Dict[int, int]:
    out = {int(k): 0 for k in ks}
    flat = scores.reshape(-1)
    finite_mask = np.isfinite(flat)
    if not np.any(finite_mask):
        return out

    finite_idx = np.where(finite_mask)[0]
    ranked = finite_idx[np.argsort(-flat[finite_idx])]
    gx, gy = float(gt_xy[0]), float(gt_xy[1])
    width = int(scores.shape[1])
    hits_by_rank = np.zeros((ranked.size,), dtype=np.uint8)
    for rank_index, flat_index in enumerate(ranked):
        row = int(flat_index) // width
        col = int(flat_index) % width
        x0 = float(xs[col])
        y0 = float(ys[row])
        if x0 <= gx < (x0 + float(chip_size)) and y0 <= gy < (y0 + float(chip_size)):
            hits_by_rank[rank_index] = 1
    for k in ks:
        out[int(k)] = int(np.any(hits_by_rank[: max(int(k), 1)]))
    return out


def summarize_errors(values: Sequence[float]) -> Dict[str, Any]:
    valid = [float(value) for value in values if math.isfinite(float(value))]
    if not valid:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "max": None,
        }
    arr = np.asarray(valid, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def build_overall_topk(
    hits: Dict[int, int],
    total: int,
    eval_ks: Sequence[int],
) -> Dict[str, Any]:
    return {
        "n_eval_images": int(total),
        "hits": {f"top{int(k)}": int(hits[int(k)]) for k in eval_ks},
        "rates": {
            f"top{int(k)}": None if int(total) <= 0 else float(hits[int(k)]) / float(total)
            for k in eval_ks
        },
    }


def write_rows_txt(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    eval_ks: Sequence[int],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        header = [
            "cluster_id",
            "global_cluster_id",
            "image_filename",
            "gt_x_px",
            "gt_y_px",
            "pred_retrieval_x_px",
            "pred_retrieval_y_px",
            "pred_position_head_x_px",
            "pred_position_head_y_px",
            "retrieval_error_px",
            "position_head_error_px",
        ]
        header.extend(f"top{int(k)}" for k in eval_ks)
        writer.writerow(header)
        for row in rows:
            values = [
                str(row["cluster_id"]),
                int(row["global_cluster_id"]),
                str(row["image_filename"]),
                "" if row["gt_xy"] is None else float(row["gt_xy"][0]),
                "" if row["gt_xy"] is None else float(row["gt_xy"][1]),
                "" if row["pred_retrieval_xy"] is None else float(row["pred_retrieval_xy"][0]),
                "" if row["pred_retrieval_xy"] is None else float(row["pred_retrieval_xy"][1]),
                "" if row["pred_position_head_xy"] is None else float(row["pred_position_head_xy"][0]),
                "" if row["pred_position_head_xy"] is None else float(row["pred_position_head_xy"][1]),
                "" if row["retrieval_error_px"] is None else float(row["retrieval_error_px"]),
                "" if row["position_head_error_px"] is None else float(row["position_head_error_px"]),
            ]
            values.extend(int(row["topk_hits"][f"top{int(k)}"]) for k in eval_ks)
            writer.writerow(values)


def load_reference_latlons(reference_dir: Path) -> Dict[str, Tuple[float, float]]:
    latlons: Dict[str, Tuple[float, float]] = {}
    for json_path in sorted(reference_dir.glob("*.json")):
        lat_lon = extract_lat_lon(json_path)
        if lat_lon is not None:
            latlons[json_path.stem] = (float(lat_lon[0]), float(lat_lon[1]))
    return latlons


def resolve_image_path(
    dataset_root: Path,
    saved_image_path: str,
    ground_dir_candidates: Sequence[str],
) -> Path:
    path = Path(str(saved_image_path)).expanduser()
    if path.exists():
        return path.resolve()

    image_name = path.name
    for dir_name in ground_dir_candidates:
        candidate = (dataset_root / dir_name / image_name).resolve()
        if candidate.exists():
            return candidate
    return (dataset_root / str(ground_dir_candidates[0]) / image_name).resolve()


def resolve_satellite_path(
    dataset_root: Path,
    saved_satellite_path: str,
    satellite_dir_candidates: Sequence[str],
) -> Path:
    path = Path(str(saved_satellite_path)).expanduser()
    if path.exists():
        return path.resolve()

    image_name = path.name
    for dir_name in satellite_dir_candidates:
        candidate = (dataset_root / dir_name / image_name).resolve()
        if candidate.exists():
            return candidate
    return (dataset_root / str(satellite_dir_candidates[0]) / image_name).resolve()


def build_cluster_predictions_from_summary_clusters(
    cluster_records: Sequence[Dict[str, Any]],
    dataset_root: Path,
    ground_dir_candidates: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for cluster_record in cluster_records:
        cluster_id = str(cluster_record["cluster_id"])
        per_image_records = []
        for image_record in cluster_record.get("per_image_position_head_correction") or []:
            image_path = resolve_image_path(
                dataset_root,
                str(image_record["image_path"]),
                ground_dir_candidates,
            )
            pred_before = image_record.get("pred_xy_before_pos_head")
            pred_after = image_record.get("pred_xy_after_pos_head")
            per_image_records.append(
                {
                    "image_path": image_path,
                    "image_filename": image_path.name,
                    "pred_retrieval_xy": None
                    if pred_before is None
                    else (float(pred_before[0]), float(pred_before[1])),
                    "pred_position_head_xy": None
                    if pred_after is None
                    else (float(pred_after[0]), float(pred_after[1])),
                }
            )
        out[cluster_id] = {
            "cluster_id": cluster_id,
            "global_cluster_id": int(cluster_record["global_cluster_id"]),
            "per_image_records": per_image_records,
        }
    return out


def load_inference_records(inference_json: Path) -> Dict[str, Dict[str, Any]]:
    records = load_json(inference_json)
    if not isinstance(records, list):
        raise RuntimeError(f"Expected top-level list in {inference_json}, got {type(records)}")

    out: Dict[str, Dict[str, Any]] = {}
    for record in records:
        cluster_id = cluster_label(int(record["cluster_id"]))
        satellite_details: Dict[int, Dict[str, Any]] = {}
        for sat_record in record["all_chips_per_satellite"]:
            sat_index = int(sat_record["sat_index"])
            chips: List[Dict[str, Any]] = []
            for chip in sat_record["chips"]:
                chip_box = [int(value) for value in chip["chip_box_xyxy"]]
                x1, y1, x2, y2 = chip_box
                chips.append(
                    {
                        "score": float(chip["score"]),
                        "chip_box_xyxy": chip_box,
                        "chip_size": int(x2 - x1),
                    }
                )
            satellite_details[sat_index] = {
                "sat_index": sat_index,
                "sat_path": str(sat_record["sat_path"]),
                "chips": chips,
            }
        out[cluster_id] = {
            "cluster_id": cluster_id,
            "ground_paths": [str(path) for path in record["ground_paths"]],
            "satellite_details": satellite_details,
        }
    return out


def build_score_grid_from_chip_lists(chip_lists: Sequence[Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    score_by_box: Dict[Tuple[int, int, int, int], float] = {}
    chip_size: Optional[int] = None
    for chip_list in chip_lists:
        for chip in chip_list:
            chip_box = tuple(int(value) for value in chip["chip_box_xyxy"])
            score_by_box[chip_box] = score_by_box.get(chip_box, 0.0) + float(chip["score"])
            if chip_size is None:
                chip_size = int(chip["chip_size"])

    if not score_by_box or chip_size is None:
        raise RuntimeError("No chip scores were available to build a score grid")

    xs = sorted({chip_box[0] for chip_box in score_by_box})
    ys = sorted({chip_box[1] for chip_box in score_by_box})
    x_to_col = {value: index for index, value in enumerate(xs)}
    y_to_row = {value: index for index, value in enumerate(ys)}
    scores = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
    for chip_box, score in score_by_box.items():
        row = int(y_to_row[int(chip_box[1])])
        col = int(x_to_col[int(chip_box[0])])
        scores[row, col] = float(score)
    return {
        "scores": scores,
        "xs": xs,
        "ys": ys,
        "chip_size": int(chip_size),
    }


def build_retrieval_best_score_grids(
    cluster_records: Sequence[Dict[str, Any]],
    inference_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for cluster_record in cluster_records:
        cluster_id = str(cluster_record["cluster_id"])
        sat_index = int(cluster_record["selected_sat_index"])
        chips = inference_by_id[cluster_id]["satellite_details"][sat_index]["chips"]
        out[cluster_id] = build_score_grid_from_chip_lists([chips])
    return out


def build_retrieval_aggregate_score_grids(
    cluster_records: Sequence[Dict[str, Any]],
    inference_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for cluster_record in cluster_records:
        cluster_id = str(cluster_record["cluster_id"])
        sat_indices = [int(value) for value in cluster_record["satellite_indices_used"]]
        chip_lists = [inference_by_id[cluster_id]["satellite_details"][sat_index]["chips"] for sat_index in sat_indices]
        out[cluster_id] = build_score_grid_from_chip_lists(chip_lists)
    return out


def build_postsum_best_score_grids(
    cluster_records: Sequence[Dict[str, Any]],
    inference_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for cluster_record in cluster_records:
        cluster_id = str(cluster_record["cluster_id"])
        sat_index = int(cluster_record["selected_sat_index"])
        chip_lists: List[Sequence[Dict[str, Any]]] = []
        for neighbor_cluster_id in cluster_record["neighbor_cluster_ids"]:
            neighbor_details = inference_by_id[str(neighbor_cluster_id)]["satellite_details"]
            if sat_index in neighbor_details:
                chip_lists.append(neighbor_details[sat_index]["chips"])
        out[cluster_id] = build_score_grid_from_chip_lists(chip_lists)
    return out


def build_postsum_aggregate_score_grids(
    cluster_records: Sequence[Dict[str, Any]],
    inference_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for cluster_record in cluster_records:
        cluster_id = str(cluster_record["cluster_id"])
        sat_indices = [int(value) for value in cluster_record["satellite_indices_used"]]
        chip_lists: List[Sequence[Dict[str, Any]]] = []
        for sat_index in sat_indices:
            for neighbor_cluster_id in cluster_record["neighbor_cluster_ids"]:
                neighbor_details = inference_by_id[str(neighbor_cluster_id)]["satellite_details"]
                if sat_index in neighbor_details:
                    chip_lists.append(neighbor_details[sat_index]["chips"])
        out[cluster_id] = build_score_grid_from_chip_lists(chip_lists)
    return out


def evaluate_case(
    *,
    case_name: str,
    cluster_predictions: Dict[str, Dict[str, Any]],
    score_grid_by_cluster: Dict[str, Dict[str, Any]],
    gt_pixel_by_image: Dict[str, Optional[Tuple[float, float]]],
    eval_ks: Sequence[int],
    note: str,
    out_json: Path,
    out_txt: Path,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    hits = {int(k): 0 for k in eval_ks}
    topk_total = 0
    retrieval_errors: List[float] = []
    position_head_errors: List[float] = []

    for cluster_id, cluster_prediction in cluster_predictions.items():
        score_info = score_grid_by_cluster[cluster_id]
        for image_record in cluster_prediction["per_image_records"]:
            image_path = Path(str(image_record["image_path"])).resolve()
            gt_xy = gt_pixel_by_image.get(image_path.name)
            topk_hits = {f"top{int(k)}": 0 for k in eval_ks}
            if gt_xy is not None:
                topk_total += 1
                raw_hits = topk_hit_for_gt_point(
                    scores=score_info["scores"],
                    xs=score_info["xs"],
                    ys=score_info["ys"],
                    chip_size=int(score_info["chip_size"]),
                    gt_xy=gt_xy,
                    ks=eval_ks,
                )
                for k in eval_ks:
                    hits[int(k)] += int(raw_hits[int(k)])
                    topk_hits[f"top{int(k)}"] = int(raw_hits[int(k)])

            pred_retrieval_xy = image_record["pred_retrieval_xy"]
            pred_position_head_xy = image_record["pred_position_head_xy"]
            retrieval_error_px = None
            position_head_error_px = None
            if gt_xy is not None and pred_retrieval_xy is not None:
                retrieval_error_px = float(
                    math.hypot(
                        float(pred_retrieval_xy[0]) - float(gt_xy[0]),
                        float(pred_retrieval_xy[1]) - float(gt_xy[1]),
                    )
                )
                retrieval_errors.append(retrieval_error_px)
            if gt_xy is not None and pred_position_head_xy is not None:
                position_head_error_px = float(
                    math.hypot(
                        float(pred_position_head_xy[0]) - float(gt_xy[0]),
                        float(pred_position_head_xy[1]) - float(gt_xy[1]),
                    )
                )
                position_head_errors.append(position_head_error_px)

            rows.append(
                {
                    "cluster_id": str(cluster_id),
                    "global_cluster_id": int(cluster_prediction["global_cluster_id"]),
                    "image_filename": image_path.name,
                    "gt_xy": gt_xy,
                    "pred_retrieval_xy": pred_retrieval_xy,
                    "pred_position_head_xy": pred_position_head_xy,
                    "retrieval_error_px": retrieval_error_px,
                    "position_head_error_px": position_head_error_px,
                    "topk_hits": topk_hits,
                }
            )

    summary = {
        "case_name": str(case_name),
        "note": str(note),
        "n_rows": int(len(rows)),
        "retrieval_error_px": summarize_errors(retrieval_errors),
        "position_head_error_px": summarize_errors(position_head_errors),
        "topk": build_overall_topk(hits, topk_total, eval_ks),
        "rows": [
            {
                **row,
                "gt_xy": None if row["gt_xy"] is None else [float(row["gt_xy"][0]), float(row["gt_xy"][1])],
                "pred_retrieval_xy": None
                if row["pred_retrieval_xy"] is None
                else [float(row["pred_retrieval_xy"][0]), float(row["pred_retrieval_xy"][1])],
                "pred_position_head_xy": None
                if row["pred_position_head_xy"] is None
                else [float(row["pred_position_head_xy"][0]), float(row["pred_position_head_xy"][1])],
            }
            for row in rows
        ],
    }
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_rows_txt(out_txt, rows, eval_ks)
    return {
        "case_name": str(case_name),
        "summary_json": str(out_json.resolve()),
        "summary_txt": str(out_txt.resolve()),
        "topk": summary["topk"],
        "retrieval_error_px": summary["retrieval_error_px"],
        "position_head_error_px": summary["position_head_error_px"],
        "n_rows": int(len(rows)),
    }


def project_gt_pixels_for_selected_satellites(
    reference_latlon_by_stem: Dict[str, Tuple[float, float]],
    cluster_records: Sequence[Dict[str, Any]],
    dataset_root: Path,
    satellite_dir_candidates: Sequence[str],
) -> Dict[str, Optional[Tuple[float, float]]]:
    datasets: Dict[Path, Any] = {}
    try:
        out: Dict[str, Optional[Tuple[float, float]]] = {}
        for cluster_record in cluster_records:
            sat_path = resolve_satellite_path(
                dataset_root,
                str(cluster_record["selected_sat_path"]),
                satellite_dir_candidates,
            )
            dataset = datasets.get(sat_path)
            if dataset is None:
                dataset = rasterio.open(sat_path)
                datasets[sat_path] = dataset
            width = int(dataset.width)
            height = int(dataset.height)
            for image_record in cluster_record.get("per_image_position_head_correction") or []:
                image_filename = Path(str(image_record["image_path"])).name
                stem = Path(image_filename).stem
                lat_lon = reference_latlon_by_stem.get(stem)
                if lat_lon is None:
                    out[image_filename] = None
                    continue
                pixel_xy = latlon_to_pixel(dataset, float(lat_lon[0]), float(lat_lon[1]))
                out[image_filename] = pixel_xy if is_in_bounds(pixel_xy, width, height) else None
        return out
    finally:
        for dataset in datasets.values():
            dataset.close()


def project_gt_pixels_for_cluster_satellite_sets(
    reference_latlon_by_stem: Dict[str, Tuple[float, float]],
    cluster_records: Sequence[Dict[str, Any]],
    dataset_root: Path,
    satellite_dir_candidates: Sequence[str],
    *,
    sat_paths_key: str,
) -> Dict[str, Optional[Tuple[float, float]]]:
    datasets: Dict[Path, Any] = {}
    try:
        out: Dict[str, Optional[Tuple[float, float]]] = {}
        for cluster_record in cluster_records:
            sat_paths = [
                resolve_satellite_path(dataset_root, str(path), satellite_dir_candidates)
                for path in cluster_record[sat_paths_key]
            ]
            cluster_datasets = []
            for sat_path in sat_paths:
                dataset = datasets.get(sat_path)
                if dataset is None:
                    dataset = rasterio.open(sat_path)
                    datasets[sat_path] = dataset
                cluster_datasets.append(dataset)
            for image_record in cluster_record.get("per_image_position_head_correction") or []:
                image_filename = Path(str(image_record["image_path"])).name
                stem = Path(image_filename).stem
                lat_lon = reference_latlon_by_stem.get(stem)
                if lat_lon is None:
                    out[image_filename] = None
                    continue
                pixels: List[Tuple[float, float]] = []
                for dataset in cluster_datasets:
                    pixel_xy = latlon_to_pixel(dataset, float(lat_lon[0]), float(lat_lon[1]))
                    if is_in_bounds(pixel_xy, int(dataset.width), int(dataset.height)):
                        pixels.append((float(pixel_xy[0]), float(pixel_xy[1])))
                if not pixels:
                    out[image_filename] = None
                else:
                    arr = np.asarray(pixels, dtype=np.float64)
                    out[image_filename] = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
        return out
    finally:
        for dataset in datasets.values():
            dataset.close()


def evaluate_cluster_predictions_against_gt(global_cfg, script_cfg) -> int:
    dataset_root = Path(
        str(require_config_value(global_cfg, "dataset_root"))
    ).expanduser().resolve()
    output_root = Path(
        str(require_config_value(global_cfg, "output_root"))
    ).expanduser().resolve()
    reference_dir_name = str(require_config_value(global_cfg, "reference_dir_name"))
    retrieval_dir_name = str(require_config_value(global_cfg, "retrieval_dir_name"))
    postsum_dir_name = str(require_config_value(global_cfg, "postsum_dir_name"))
    eval_dir_name = str(require_config_value(global_cfg, "eval_dir_name"))

    ground_dir_candidates = require_config_value(global_cfg, "ground_dir_candidates")
    if not isinstance(ground_dir_candidates, (list, tuple)) or not ground_dir_candidates:
        raise ValueError("global.ground_dir_candidates must be a non-empty list or tuple")
    ground_dir_candidates = tuple(str(value) for value in ground_dir_candidates)

    satellite_dir_candidates = require_config_value(global_cfg, "satellite_dir_candidates")
    if not isinstance(satellite_dir_candidates, (list, tuple)) or not satellite_dir_candidates:
        raise ValueError("global.satellite_dir_candidates must be a non-empty list or tuple")
    satellite_dir_candidates = tuple(str(value) for value in satellite_dir_candidates)

    eval_ks = require_config_value(script_cfg, "eval_ks")
    if not isinstance(eval_ks, (list, tuple)) or not eval_ks:
        raise ValueError("evaluate_cluster_predictions_against_gt.eval_ks must be a non-empty list or tuple")
    eval_ks = tuple(int(value) for value in eval_ks)

    dataset_output_root = (output_root / dataset_root.name).resolve()
    retrieval_root = (dataset_output_root / retrieval_dir_name).resolve()
    postsum_root = (dataset_output_root / postsum_dir_name).resolve()
    eval_root = (dataset_output_root / eval_dir_name).resolve()
    eval_root.mkdir(parents=True, exist_ok=True)

    reference_latlon_by_stem = load_reference_latlons((dataset_root / reference_dir_name).resolve())
    retrieval_best_summary = load_json(retrieval_root / "best_satellite_summary.json")
    inference_by_id = load_inference_records(Path(str(retrieval_best_summary["inference_json"])).resolve())
    retrieval_best_predictions = build_cluster_predictions_from_summary_clusters(
        retrieval_best_summary["clusters"],
        dataset_root,
        ground_dir_candidates,
    )
    retrieval_best_scores = build_retrieval_best_score_grids(retrieval_best_summary["clusters"], inference_by_id)
    retrieval_best_gt = project_gt_pixels_for_selected_satellites(
        reference_latlon_by_stem,
        retrieval_best_summary["clusters"],
        dataset_root,
        satellite_dir_candidates,
    )

    retrieval_aggregate_summary_path = (retrieval_root / "all_satellite_aggregate_summary.json").resolve()
    retrieval_aggregate_summary = load_json(retrieval_aggregate_summary_path)
    retrieval_aggregate_predictions = build_cluster_predictions_from_summary_clusters(
        retrieval_aggregate_summary["clusters"],
        dataset_root,
        ground_dir_candidates,
    )
    retrieval_aggregate_scores = build_retrieval_aggregate_score_grids(
        retrieval_aggregate_summary["clusters"],
        inference_by_id,
    )
    retrieval_aggregate_gt = project_gt_pixels_for_cluster_satellite_sets(
        reference_latlon_by_stem,
        retrieval_aggregate_summary["clusters"],
        dataset_root,
        satellite_dir_candidates,
        sat_paths_key="satellite_paths_used",
    )

    postsum_best_summary = load_json(postsum_root / "best_satellite_summary.json")
    postsum_best_predictions = build_cluster_predictions_from_summary_clusters(
        postsum_best_summary["clusters"],
        dataset_root,
        ground_dir_candidates,
    )
    postsum_best_scores = build_postsum_best_score_grids(postsum_best_summary["clusters"], inference_by_id)
    postsum_best_gt = project_gt_pixels_for_selected_satellites(
        reference_latlon_by_stem,
        postsum_best_summary["clusters"],
        dataset_root,
        satellite_dir_candidates,
    )

    postsum_aggregate_summary_path = (postsum_root / "all_satellite_aggregate_summary.json").resolve()
    postsum_aggregate_summary = load_json(postsum_aggregate_summary_path)
    postsum_aggregate_predictions = build_cluster_predictions_from_summary_clusters(
        postsum_aggregate_summary["clusters"],
        dataset_root,
        ground_dir_candidates,
    )
    postsum_aggregate_scores = build_postsum_aggregate_score_grids(
        postsum_aggregate_summary["clusters"],
        inference_by_id,
    )
    postsum_aggregate_gt = project_gt_pixels_for_cluster_satellite_sets(
        reference_latlon_by_stem,
        postsum_aggregate_summary["clusters"],
        dataset_root,
        satellite_dir_candidates,
        sat_paths_key="satellite_paths_used",
    )

    case_summaries = [
        evaluate_case(
            case_name="retrieval_best_satellite",
            cluster_predictions=retrieval_best_predictions,
            score_grid_by_cluster=retrieval_best_scores,
            gt_pixel_by_image=retrieval_best_gt,
            eval_ks=eval_ks,
            note="Before neighbor aggregation, no satellite aggregation. Uses the selected best single satellite.",
            out_json=(eval_root / "retrieval_best_satellite_eval.json").resolve(),
            out_txt=(eval_root / "retrieval_best_satellite_eval.txt").resolve(),
        ),
        evaluate_case(
            case_name="retrieval_all_satellite_aggregate",
            cluster_predictions=retrieval_aggregate_predictions,
            score_grid_by_cluster=retrieval_aggregate_scores,
            gt_pixel_by_image=retrieval_aggregate_gt,
            eval_ks=eval_ks,
            note="Before neighbor aggregation, with satellite aggregation. GT pixel uses the mean projection across all satellites.",
            out_json=(eval_root / "retrieval_all_satellite_aggregate_eval.json").resolve(),
            out_txt=(eval_root / "retrieval_all_satellite_aggregate_eval.txt").resolve(),
        ),
        evaluate_case(
            case_name="neighbor_postsum_best_satellite",
            cluster_predictions=postsum_best_predictions,
            score_grid_by_cluster=postsum_best_scores,
            gt_pixel_by_image=postsum_best_gt,
            eval_ks=eval_ks,
            note="After neighbor aggregation, no satellite aggregation. Uses the selected best single satellite.",
            out_json=(eval_root / "neighbor_postsum_best_satellite_eval.json").resolve(),
            out_txt=(eval_root / "neighbor_postsum_best_satellite_eval.txt").resolve(),
        ),
        evaluate_case(
            case_name="neighbor_postsum_all_satellite_aggregate",
            cluster_predictions=postsum_aggregate_predictions,
            score_grid_by_cluster=postsum_aggregate_scores,
            gt_pixel_by_image=postsum_aggregate_gt,
            eval_ks=eval_ks,
            note="After neighbor aggregation, with satellite aggregation. GT pixel uses the mean projection across all satellites.",
            out_json=(eval_root / "neighbor_postsum_all_satellite_aggregate_eval.json").resolve(),
            out_txt=(eval_root / "neighbor_postsum_all_satellite_aggregate_eval.txt").resolve(),
        ),
    ]

    with (eval_root / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_root": str(dataset_root),
                "output_root": str(output_root),
                "evaluation_root": str(eval_root),
                "cases": case_summaries,
            },
            handle,
            indent=2,
        )

    print(f"[EVAL-DONE] dataset={dataset_root.name} out={eval_root}", flush=True)
    return 0

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_pipeline_config(args.config)
    global_cfg = get_global_config(config)
    script_cfg = require_config_section(config, "evaluate_cluster_predictions_against_gt")

    evaluate_cluster_predictions_against_gt(global_cfg=global_cfg, script_cfg=script_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
