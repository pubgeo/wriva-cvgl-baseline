#!/usr/bin/env python3
"""
Run HLoc/COLMAP reconstruction for ground images and export camera clusters.

Default expected layout:
    <dataset_root>/
      ground/ or image/
        *.jpg / *.png / ...

Outputs:
    <output_root>/<dataset_name>/registration/
      camera_clusters.txt
      run/
        arb_colmap/
          colmap/
            sparse/0/
              cameras.txt
              images.txt
              points3D.txt
              models/<component_id>/
                cameras.txt
                images.txt
                points3D.txt
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from inference_utils.config_helpers import (
    add_config_argument,
    get_global_config,
    load_pipeline_config,
    require_config_section,
    require_config_value,
)

THIS_DIR = Path(__file__).resolve().parent
CVGL_ROOT = THIS_DIR.parent
DIRECT_MATCH_DIR = CVGL_ROOT / "direct_match"
BASELINE_UTILS_DIR = CVGL_ROOT / "baseline_utils"
for import_root in (THIS_DIR, CVGL_ROOT, DIRECT_MATCH_DIR, BASELINE_UTILS_DIR):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

# from read_write_model import read_model, write_model
# from hloc import extract_features, match_features, pairs_from_retrieval, reconstruction


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
CLUSTER_SIZE = 8
CLUSTER_SOURCE_COLMAP = "colmap"
CLUSTER_SOURCE_FILENAME_ORDER = "filename_order"
CLUSTER_SOURCE_CHOICES = (
    CLUSTER_SOURCE_COLMAP,
    CLUSTER_SOURCE_FILENAME_ORDER,
)
# the ANGLE_DEG parameters are for colmap clustering to make sure the images have enough overlap


@dataclass(frozen=True)
class CameraFrame:
    """One usable camera sample after loading either COLMAP poses or filename-order placeholders."""
    image_path: Path
    global_cluster_id: int
    coord_xyz: Tuple[float, float, float]
    qvec: Tuple[float, float, float, float]
    forward: Tuple[float, float, float]
    heading_xy: Optional[Tuple[float, float]]


@dataclass(frozen=True)
class LocalCameraCluster:
    """One output cluster written to camera_clusters.txt with a local 0-based cluster id."""
    local_cluster_id: int
    global_cluster_id: int
    frames: Tuple[CameraFrame, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run HLoc/COLMAP reconstruction and export camera_clusters.txt using "
            "settings from infer_tiles.jsonc."
        )
    )
    add_config_argument(parser)
    return parser.parse_args()


def build_runtime_args(
    global_cfg: Dict[str, Any],
    config: Dict[str, Any],
) -> argparse.Namespace:
    ground_dir_value = config.get("ground_dir")
    output_dir_value = config.get("output_dir")
    cluster_source = str(require_config_value(config, "cluster_source"))
    if cluster_source not in CLUSTER_SOURCE_CHOICES:
        raise ValueError(
            f"Unsupported cluster_source={cluster_source!r}. "
            f"Expected one of: {CLUSTER_SOURCE_CHOICES}"
        )

    ground_dir_candidates = require_config_value(global_cfg, "ground_dir_candidates")
    if not isinstance(ground_dir_candidates, (list, tuple)) or not ground_dir_candidates:
        raise ValueError("global.ground_dir_candidates must be a non-empty list or tuple")

    return argparse.Namespace(
        dataset_root=Path(str(require_config_value(global_cfg, "dataset_root"))).expanduser(),
        ground_dir=None if ground_dir_value is None else Path(str(ground_dir_value)).expanduser(),
        output_dir=None if output_dir_value is None else Path(str(output_dir_value)).expanduser(),
        output_root=Path(str(require_config_value(global_cfg, "output_root"))).expanduser(),
        registration_subdir_name=str(require_config_value(global_cfg, "registration_subdir_name")),
        camera_clusters_filename=str(require_config_value(global_cfg, "camera_clusters_filename")),
        ground_dir_candidates=tuple(str(value) for value in ground_dir_candidates),
        force=bool(require_config_value(config, "force")),
        num_matched=int(require_config_value(config, "num_matched")),
        match_threshold=float(require_config_value(config, "match_threshold")),
        min_match_score=float(require_config_value(config, "min_match_score")),
        skip_geometric_verification=bool(require_config_value(config, "skip_geometric_verification")),
        cluster_source=cluster_source,
        cluster_size=int(require_config_value(config, "cluster_size")),
        cluster_view_soft_angle_deg=float(require_config_value(config, "cluster_view_soft_angle_deg")),
        cluster_view_hard_angle_deg=float(require_config_value(config, "cluster_view_hard_angle_deg")),
    )


def first_existing_subdir(root_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        candidate = (root_dir / str(name)).resolve()
        if candidate.is_dir():
            return candidate
    return None


def resolve_dataset_ground_dir(
    dataset_root: Path,
    ground_dir_candidates: tuple[str, ...],
) -> Path:
    dataset_root = dataset_root.resolve()
    ground_dir = first_existing_subdir(dataset_root, ground_dir_candidates)
    if ground_dir is None:
        raise FileNotFoundError(
            f"No ground/image directory found under {dataset_root}. "
            f"Tried: {list(ground_dir_candidates)}"
        )
    return ground_dir.resolve()


def default_output_dir_for_dataset(
    dataset_root: Path,
    output_root: Path,
    registration_subdir_name: str,
) -> Path:
    return (output_root / dataset_root.name / registration_subdir_name).resolve()


def list_ground_images(ground_dir: Path) -> list[Path]:
    ground_dir = ground_dir.resolve()
    if not ground_dir.is_dir():
        raise FileNotFoundError(f"Missing ground directory: {ground_dir}")

    image_paths: list[Path] = []
    for path in ground_dir.iterdir():
        if not path.name or path.name.startswith("."):
            continue
        try:
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(path.resolve())
        except OSError as exc:
            print(f"[WARN] Skipping ground image due to I/O error: {path} ({exc})", flush=True)
    image_paths = sorted(image_paths)
    if not image_paths:
        raise FileNotFoundError(f"No ground images found in: {ground_dir}")

    return image_paths


def write_text_models(model_root: Path) -> None:
    cameras, images, points3d = read_model(model_root, ext=".bin")
    write_model(cameras, images, points3d, model_root, ext=".txt")

    models_root = model_root / "models"
    if not models_root.exists():
        return

    for subdir in sorted(models_root.iterdir()):
        if not subdir.is_dir():
            continue
        try:
            cameras, images, points3d = read_model(subdir, ext=".bin")
            write_model(cameras, images, points3d, subdir, ext=".txt")
        except Exception:
            continue


def run_hloc_reconstruction(
    ground_dir: Path,
    image_paths: list[Path],
    arb_colmap_dir: Path,
    *,
    force: bool,
    num_matched: int,
    match_threshold: float,
    min_match_score: float,
    skip_geometric_verification: bool,
) -> Path:
    if force and arb_colmap_dir.exists():
        shutil.rmtree(arb_colmap_dir)

    model_root = arb_colmap_dir / "colmap" / "sparse" / "0"
    if (model_root / "images.bin").is_file() and (model_root / "cameras.bin").is_file():
        write_text_models(model_root)
        return model_root

    outputs = arb_colmap_dir / "colmap"
    sfm_pairs = outputs / "pairs_retrieval.txt"
    outputs.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    global_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superglue"].copy()
    matcher_conf["model"]["match_threshold"] = float(match_threshold)
    image_list = [path.relative_to(ground_dir).as_posix() for path in image_paths]

    global_path = extract_features.main(global_conf, ground_dir, outputs, image_list=image_list)
    pairs_from_retrieval.main(global_path, sfm_pairs, num_matched=int(num_matched))

    feature_path = extract_features.main(feature_conf, ground_dir, outputs, image_list=image_list)
    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf["output"], outputs)

    reconstruction.main(
        model_root,
        ground_dir,
        sfm_pairs,
        feature_path,
        match_path,
        verbose=True,
        camera_mode="PER_IMAGE",
        image_list=image_list,
        image_options={},
        mapper_options={},
        min_match_score=float(min_match_score),
        skip_geometric_verification=bool(skip_geometric_verification),
    )

    write_text_models(model_root)
    return model_root


def load_colmap_components(model_root: Path) -> list[tuple[int, Path, tuple[dict, dict, dict]]]:
    models_root = model_root / "models"
    components: list[tuple[int, Path, tuple[dict, dict, dict]]] = []
    covered_names: set[str] = set()

    if models_root.exists():
        for component_dir in sorted(path for path in models_root.iterdir() if path.is_dir()):
            try:
                model = read_model(component_dir, ext=".bin")
            except Exception:
                continue
            _cameras, images, _points3d = model
            covered_names.update(str(image.name) for image in images.values())
            component_index = len(components)
            components.append((component_index, component_dir, model))

    try:
        root_model = read_model(model_root, ext=".bin")
    except Exception:
        root_model = None

    if root_model is not None:
        cameras, images, _points3d = root_model
        if components:
            leftover_images = {
                image_id: image
                for image_id, image in images.items()
                if str(image.name) not in covered_names
            }
            if leftover_images:
                component_index = len(components)
                components.append((component_index, model_root, (cameras, leftover_images, {})))
        else:
            component_index = len(components)
            components.append((component_index, model_root, root_model))

    if not components:
        raise FileNotFoundError(f"No readable COLMAP models found under: {model_root}")

    return components


def colmap_camera_center(colmap_image: Any) -> np.ndarray:
    rotation = R.from_quat(np.roll(np.asarray(colmap_image.qvec, dtype=np.float64), -1))
    return -rotation.inv().apply(np.asarray(colmap_image.tvec, dtype=np.float64))


def qvec_to_forward_world(qvec: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    rot_wc = R.from_quat(np.roll(np.asarray(qvec, dtype=np.float64), -1))
    forward = rot_wc.inv().apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    norm = float(np.linalg.norm(forward))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError(f"invalid qvec for forward direction: {qvec}")
    forward /= norm
    return float(forward[0]), float(forward[1]), float(forward[2])


def forward_to_heading_xy(
    forward: Tuple[float, float, float] | np.ndarray,
) -> Optional[Tuple[float, float]]:
    vec = np.asarray(forward, dtype=np.float64).reshape(-1)
    if vec.shape[0] < 2:
        return None
    heading = vec[:2].astype(np.float64, copy=True)
    norm = float(np.linalg.norm(heading))
    if not math.isfinite(norm) or norm <= 1e-8:
        return None
    heading /= norm
    return float(heading[0]), float(heading[1])


def cluster_mean_forward(frames: Sequence[CameraFrame]) -> Optional[np.ndarray]:
    if not frames:
        return None
    arr = np.asarray([fr.forward for fr in frames], dtype=np.float64)
    mean_forward = np.mean(arr, axis=0)
    norm = float(np.linalg.norm(mean_forward))
    if not math.isfinite(norm) or norm <= 1e-8:
        return None
    return mean_forward / norm


def cluster_mean_heading_xy(frames: Sequence[CameraFrame]) -> Optional[np.ndarray]:
    headings = [fr.heading_xy for fr in frames if fr.heading_xy is not None]
    if not headings:
        return None
    arr = np.asarray(headings, dtype=np.float64)
    mean_heading = np.mean(arr, axis=0)
    norm = float(np.linalg.norm(mean_heading))
    if not math.isfinite(norm) or norm <= 1e-8:
        return None
    return mean_heading / norm


def angular_distance_deg(
    a: Sequence[float] | np.ndarray | None,
    b: Sequence[float] | np.ndarray | None,
) -> Optional[float]:
    if a is None or b is None:
        return None
    a_vec = np.asarray(a, dtype=np.float64)
    b_vec = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
    if not math.isfinite(denom) or denom <= 1e-8:
        return None
    cosine = float(np.clip(np.dot(a_vec, b_vec) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def pairwise_view_angle_deg(a: CameraFrame, b: CameraFrame) -> Optional[float]:
    heading_angle = angular_distance_deg(a.heading_xy, b.heading_xy)
    if heading_angle is not None:
        return heading_angle
    return angular_distance_deg(a.forward, b.forward)


def cluster_view_angle_deg(
    frame: CameraFrame,
    mean_heading_xy: np.ndarray | None,
    mean_forward: np.ndarray | None,
) -> Optional[float]:
    heading_angle = angular_distance_deg(frame.heading_xy, mean_heading_xy)
    if heading_angle is not None:
        return heading_angle
    return angular_distance_deg(frame.forward, mean_forward)


def read_colmap_registered_frames(
    ground_dir: Path,
    model_root: Path,
) -> tuple[list[CameraFrame], dict[str, Any]]:
    components = load_colmap_components(model_root)
    frames: list[CameraFrame] = []
    skipped: Counter[str] = Counter()
    seen_image_names: set[str] = set()
    component_sizes: Counter[int] = Counter()

    for global_cluster_id, _component_dir, model in components:
        _cameras, images, _points3d = model
        for image in images.values():
            image_name = str(image.name)
            if image_name in seen_image_names:
                skipped["duplicate_image_name"] += 1
                continue
            seen_image_names.add(image_name)

            image_path = (ground_dir / image_name).resolve()
            try:
                image_exists = image_path.is_file()
            except OSError:
                skipped["image_io_error"] += 1
                continue
            if not image_exists:
                skipped["missing_image"] += 1
                continue

            qvec = tuple(float(v) for v in np.asarray(image.qvec, dtype=np.float64).reshape(4))
            try:
                forward = qvec_to_forward_world(qvec)
            except Exception:
                skipped["invalid_qvec"] += 1
                continue
            heading_xy = forward_to_heading_xy(forward)
            center = colmap_camera_center(image).reshape(3)

            frames.append(
                CameraFrame(
                    image_path=image_path,
                    global_cluster_id=int(global_cluster_id),
                    coord_xyz=(float(center[0]), float(center[1]), float(center[2])),
                    qvec=qvec,
                    forward=forward,
                    heading_xy=heading_xy,
                )
            )
            component_sizes[int(global_cluster_id)] += 1

    frames.sort(key=lambda fr: (int(fr.global_cluster_id), fr.image_path.name))
    stats = {
        "rows_usable": int(len(frames)),
        "skipped": {str(k): int(v) for k, v in sorted(skipped.items())},
        "registration_component_sizes": {str(k): int(v) for k, v in sorted(component_sizes.items())},
    }
    return frames, stats


def build_filename_order_frames(ground_dir: Path) -> tuple[list[CameraFrame], dict[str, Any]]:
    image_paths = list_ground_images(ground_dir)
    qvec = (1.0, 0.0, 0.0, 0.0)
    forward = (0.0, 0.0, 1.0)

    frames = [
        CameraFrame(
            image_path=image_path.resolve(),
            global_cluster_id=0,
            coord_xyz=(-1.0, -1.0, -1.0),
            qvec=qvec,
            forward=forward,
            heading_xy=forward_to_heading_xy(forward),
        )
        for image_path in sorted(image_paths, key=lambda path: path.name)
    ]
    stats = {
        "rows_total": int(len(image_paths)),
        "rows_usable": int(len(frames)),
        "coordinate_mode": "placeholder_minus_one",
    }
    return frames, stats


def build_filename_chunk_groups(
    ordered_frames: Sequence[CameraFrame],
    *,
    cluster_size: int,
) -> list[list[CameraFrame]]:
    groups: list[list[CameraFrame]] = []
    current_group: list[CameraFrame] = []

    for frame in ordered_frames:
        if not current_group:
            current_group.append(frame)
            continue

        should_break_on_size = len(current_group) >= int(cluster_size)
        if should_break_on_size:
            groups.append(current_group)
            current_group = [frame]
            continue

        current_group.append(frame)

    if current_group:
        groups.append(current_group)
    return groups


def build_registration_local_cluster_groups(
    frames: Sequence[CameraFrame],
    *,
    cluster_size: int,
    soft_angle_deg: float,
    hard_angle_deg: float,
) -> list[tuple[int, list[CameraFrame]]]:
    groups: Dict[int, List[CameraFrame]] = defaultdict(list)
    for frame in frames:
        groups[int(frame.global_cluster_id)].append(frame)

    out: list[tuple[int, list[CameraFrame]]] = []
    for global_cluster_id in sorted(groups.keys()):
        remaining = sorted(groups[global_cluster_id], key=lambda fr: fr.image_path.name)
        while remaining:
            chosen = [remaining.pop(0)]
            while remaining and len(chosen) < int(cluster_size):
                mean_xyz = np.mean(np.asarray([fr.coord_xyz for fr in chosen], dtype=np.float64), axis=0)
                mean_heading_xy = cluster_mean_heading_xy(chosen)
                mean_forward = cluster_mean_forward(chosen)

                valid_candidates: list[tuple[tuple[int, int, float, float, float, str], int]] = []
                for idx, frame in enumerate(remaining):
                    dist_m = float(np.linalg.norm(np.asarray(frame.coord_xyz, dtype=np.float64) - mean_xyz))
                    ang_deg = cluster_view_angle_deg(frame, mean_heading_xy, mean_forward)
                    pairwise_angles = [pairwise_view_angle_deg(frame, other) for other in chosen]
                    hard_ok = all(angle is None or angle <= float(hard_angle_deg) for angle in pairwise_angles)
                    if not hard_ok:
                        continue
                    soft_bad = 0 if all(
                        angle is None or angle <= float(soft_angle_deg)
                        for angle in pairwise_angles
                    ) else 1
                    ang_key = 0.0 if ang_deg is None else float(ang_deg)
                    worst_pairwise = 0.0 if not pairwise_angles else max(
                        0.0 if angle is None else float(angle)
                        for angle in pairwise_angles
                    )
                    valid_candidates.append(
                        ((soft_bad, 0, dist_m, ang_key, worst_pairwise, frame.image_path.name), idx)
                    )

                if not valid_candidates:
                    break
                best_index = min(valid_candidates, key=lambda item: item[0])[1]
                chosen.append(remaining.pop(best_index))

            out.append((int(global_cluster_id), sorted(chosen, key=lambda fr: fr.image_path.name)))
    return out


def build_camera_clusters(
    frames: Sequence[CameraFrame],
    *,
    cluster_source: str,
    cluster_size: int,
    soft_angle_deg: float,
    hard_angle_deg: float,
) -> tuple[list[LocalCameraCluster], dict[str, Any]]:
    if cluster_size <= 0:
        raise ValueError(f"cluster_size must be > 0, got {cluster_size}")

    group_specs: list[tuple[int, list[CameraFrame]]] = []
    meta: dict[str, Any] = {
        "cluster_source": cluster_source,
        "cluster_size_target": int(cluster_size),
    }

    if cluster_source == CLUSTER_SOURCE_FILENAME_ORDER:
        ordered_frames = sorted(frames, key=lambda frame: frame.image_path.name)
        groups = build_filename_chunk_groups(
            ordered_frames,
            cluster_size=int(cluster_size),
        )
        group_specs = [(0, group) for group in groups]
        meta.update(
            {
                "mode": "filename_order_chunks",
                "frame_order": "ascending_filename",
                "coordinate_mode": "placeholder_minus_one",
            }
        )
    else:
        group_specs = build_registration_local_cluster_groups(
            frames,
            cluster_size=int(cluster_size),
            soft_angle_deg=float(soft_angle_deg),
            hard_angle_deg=float(hard_angle_deg),
        )
        component_sizes: Dict[str, int] = {}
        for frame in frames:
            key = str(int(frame.global_cluster_id))
            component_sizes[key] = component_sizes.get(key, 0) + 1
        meta.update(
            {
                "mode": "registration_local_xyz_heading_groups",
                "registration_component_sizes": component_sizes,
                "view_angle_soft_deg": float(soft_angle_deg),
                "view_angle_hard_deg": float(hard_angle_deg),
            }
        )

    clusters: list[LocalCameraCluster] = []
    for local_cluster_id, (global_cluster_id, group_frames) in enumerate(group_specs):
        clusters.append(
            LocalCameraCluster(
                local_cluster_id=int(local_cluster_id),
                global_cluster_id=int(global_cluster_id),
                frames=tuple(group_frames),
            )
        )

    meta["n_clusters"] = int(len(clusters))
    meta["cluster_sizes"] = [int(len(cluster.frames)) for cluster in clusters]
    return clusters, meta


def write_camera_clusters_txt(
    clusters: Sequence[LocalCameraCluster],
    output_dir: Path,
    camera_clusters_filename: str,
) -> Path:
    output_path = (output_dir / camera_clusters_filename).resolve()
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "ground_filename",
                "local_cluster_id",
                "x",
                "y",
                "global_cluster_id",
            ]
        )
        for cluster in sorted(clusters, key=lambda item: item.local_cluster_id):
            for frame in sorted(cluster.frames, key=lambda item: item.image_path.name):
                writer.writerow(
                    [
                        frame.image_path.name,
                        int(cluster.local_cluster_id),
                        float(frame.coord_xyz[0]),
                        float(frame.coord_xyz[1]),
                        int(cluster.global_cluster_id),
                    ]
                )
    return output_path


def calibrate_dataset(
    dataset_root: Path,
    ground_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    cluster_source = str(args.cluster_source)

    print(f"[INFO] calibrating {dataset_root.name}")
    print(f"[INFO] dataset_root={dataset_root}")
    print(f"[INFO] ground_dir={ground_dir}")
    print(f"[INFO] cluster_source={cluster_source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_ground_images(ground_dir)

    model_root: Optional[Path] = None
    cameras_txt: Optional[Path] = None
    images_txt: Optional[Path] = None
    cluster_frame_stats: dict[str, Any]
    if cluster_source == CLUSTER_SOURCE_FILENAME_ORDER:
        cluster_frames, cluster_frame_stats = build_filename_order_frames(ground_dir)
        status = "generated"
    else:
        model_root = (output_dir / "run" / "arb_colmap" / "colmap" / "sparse" / "0").resolve()
        had_existing_model = (
            (model_root / "images.bin").is_file()
            and (model_root / "cameras.bin").is_file()
            and not bool(args.force)
        )
        run_dir = output_dir / "run"
        arb_colmap_dir = run_dir / "arb_colmap"
        run_dir.mkdir(parents=True, exist_ok=True)

        model_root = run_hloc_reconstruction(
            ground_dir,
            image_paths,
            arb_colmap_dir,
            force=bool(args.force),
            num_matched=args.num_matched,
            match_threshold=args.match_threshold,
            min_match_score=args.min_match_score,
            skip_geometric_verification=bool(args.skip_geometric_verification),
        )
        cameras_txt = (model_root / "cameras.txt").resolve()
        images_txt = (model_root / "images.txt").resolve()
        cluster_frames, cluster_frame_stats = read_colmap_registered_frames(ground_dir, model_root)
        status = "existing" if had_existing_model else "generated"

    if not cluster_frames:
        raise RuntimeError(
            f"No usable frames found for cluster_source={cluster_source} under {dataset_root}"
        )

    clusters, clustering_meta = build_camera_clusters(
        cluster_frames,
        cluster_source=cluster_source,
        cluster_size=int(args.cluster_size),
        soft_angle_deg=float(args.cluster_view_soft_angle_deg),
        hard_angle_deg=float(args.cluster_view_hard_angle_deg),
    )
    camera_clusters_txt = write_camera_clusters_txt(
        clusters,
        output_dir,
        args.camera_clusters_filename,
    )

    cluster_row_count = int(sum(len(cluster.frames) for cluster in clusters))

    if cameras_txt is not None and images_txt is not None:
        print(f"[OK] {dataset_root.name}: wrote {cameras_txt}")
        print(f"[OK] {dataset_root.name}: wrote {images_txt}")
    print(f"[OK] {dataset_root.name}: wrote {camera_clusters_txt}")
    print(
        f"[INFO] {dataset_root.name}: input_images={len(image_paths)} "
        f"usable_cluster_frames={len(cluster_frames)} local_clusters={len(clusters)}"
    )

    summary: dict[str, object] = {
        "dataset": dataset_root.name,
        "dataset_root": str(dataset_root),
        "ground_dir": str(ground_dir),
        "image_count": int(len(image_paths)),
        "cluster_frame_count": int(len(cluster_frames)),
        "camera_cluster_rows": cluster_row_count,
        "local_cluster_count": int(len(clusters)),
        "output_dir": str(output_dir),
        "model_root": None if model_root is None else str(model_root),
        "cameras_txt": None if cameras_txt is None else str(cameras_txt),
        "images_txt": None if images_txt is None else str(images_txt),
        "camera_clusters_txt": str(camera_clusters_txt),
        "cluster_source": cluster_source,
        "status": status,
        "cluster_frame_stats": cluster_frame_stats,
        "clustering": clustering_meta,
    }
    return summary


def calculate_camera_clusters(
    args
) -> None:
    dataset_root = args.dataset_root.resolve()
    ground_dir = None if args.ground_dir is None else args.ground_dir.resolve()
    output_dir = None if args.output_dir is None else args.output_dir.resolve()

    if ground_dir is None or not ground_dir.is_dir():
        ground_dir = resolve_dataset_ground_dir(dataset_root, args.ground_dir_candidates)

    output_dir = (
        output_dir
        or default_output_dir_for_dataset(
            dataset_root,
            args.output_root.resolve(),
            args.registration_subdir_name,
        )
    ).resolve()
    calibrate_dataset(
        dataset_root,
        ground_dir,
        output_dir,
        args,
    )

def main():
    cli_args = parse_args()
    config = load_pipeline_config(cli_args.config)
    global_cfg = get_global_config(config)
    prepare_cfg = require_config_section(config, "prepare_camera_clusters")
    args = build_runtime_args(global_cfg, prepare_cfg)

    calculate_camera_clusters(
        args=args,
    )

if __name__ == "__main__":
    main()
