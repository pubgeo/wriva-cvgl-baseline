#!/usr/bin/env python3
from __future__ import annotations

import re
import math
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset

import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RASTER_EXTS = {".tif", ".tiff", ".jp2"}
MANIFEST_VERSION = 1


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    arr_f = arr.astype(np.float32)
    maxv = float(arr_f.max()) if arr_f.size > 0 else 0.0
    if maxv <= 1.0:
        arr_f *= 255.0
    elif maxv > 255.0:
        arr_f = arr_f / maxv * 255.0
    arr_f = np.clip(arr_f, 0.0, 255.0)
    return arr_f.astype(np.uint8)


class ImagePairDataset(Dataset):
    """
    Manifest-based image/satellite pair dataset.

    Satellite chip modes:
      1) "positive_center":
         Training-style mode. Uses n_sat chips total:
           - 1 positive chip centered near GT
           - n_sat - 1 negative chips
      2) "random":
         Returns random_n_sat random chips.
      3) "tiled":
         Returns as many chips as needed to tile the raster.

    Important:
      - n_sat is only a training-style parameter for "positive_center".
      - In inference:
          * "random" returns random_n_sat chips
          * "tiled" returns all tiles needed for coverage
      - Batch collation pads satellite chips to the max count in the batch.

    GT metadata:
      - If GT lat/lon maps into the raster, ground_pixel is returned.
      - In "positive_center", positive_center is the designated positive chip center.
      - In "random"/"tiled", gt_chip_index indicates which returned chip contains GT
        (or -1 if none / unknown).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        n_ground: int,
        n_sat: int,
        sat_chip_size: int,
        positive_center_jitter_px: float,
        image_size: int,
        train: bool,
        normalize: bool,
        negative_min_distance_px: Optional[float],
        max_negative_tries: int,
        max_retry: int,
        channel_last: bool,
        pos_grid: int,
        keep_sat_open: bool,
        retrieval_target_mode: str = "gaussian",
        retrieval_target_sigma_scale: float = 0.25,
        retrieval_target_min_weight: float = 1e-3,
        sat_image_size: Optional[int] = None,
        negative_local_window_px: Optional[float] = None,
        sat_chip_sizes: Optional[Sequence[int]] = None,
        site_id: Optional[str] = None,
        sat_sampling_mode: str = "positive_center",   # "positive_center", "random", "tiled"
        random_n_sat: Optional[int] = None,           # used only for random mode
        use_ground_truth_center: bool = True,
        tile_stride_px: Optional[int] = None,         # used only for tiled mode
        sat_sampling_window_px: Optional[int | Tuple[int, int] | Sequence[int]] = None,       # Used for sampling random and/or tiled mode
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.n_ground = int(n_ground)
        self.n_sat = int(n_sat)  # training-time count for positive_center mode
        self.positive_center_jitter_px = max(float(positive_center_jitter_px), 0.0)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.normalize = bool(normalize)
        self.max_negative_tries = int(max_negative_tries)
        self.max_retry = int(max_retry)
        self.channel_last = bool(channel_last)
        self.pos_grid = max(int(pos_grid), 1)
        self.keep_sat_open = bool(keep_sat_open)
        self.negative_local_window_px = (
            float(negative_local_window_px) if negative_local_window_px is not None else None
        )
        self.site_id = str(site_id) if site_id is not None else None

        self.sat_sampling_mode = str(sat_sampling_mode).strip().lower()
        if self.sat_sampling_mode not in {"positive_center", "random", "tiled"}:
            raise ValueError(
                "sat_sampling_mode must be 'positive_center', 'random', or 'tiled', "
                f"got {sat_sampling_mode!r}"
            )

        self.random_n_sat = int(random_n_sat) if random_n_sat is not None else int(n_sat)
        if self.random_n_sat <= 0:
            raise ValueError(f"random_n_sat must be > 0, got {self.random_n_sat}")

        self.use_ground_truth_center = bool(use_ground_truth_center)

        base_sat_chip = int(sat_chip_size)
        parsed_chip_sizes: List[int] = []
        if sat_chip_sizes is not None:
            for v in sat_chip_sizes:
                try:
                    vi = int(v)
                except Exception:
                    continue
                if vi > 0:
                    parsed_chip_sizes.append(vi)
        if not parsed_chip_sizes:
            parsed_chip_sizes = [base_sat_chip]
        parsed_chip_sizes = sorted(set(parsed_chip_sizes))
        self.sat_chip_sizes = parsed_chip_sizes
        self.sat_chip_size = int(parsed_chip_sizes[0])
        self.eval_sat_chip_size = int(parsed_chip_sizes[0])

        self.sat_image_size = int(sat_image_size) if sat_image_size is not None else int(image_size)
        self.negative_min_distance_px = (
            float(negative_min_distance_px) if negative_min_distance_px is not None else float(self.sat_chip_size)
        )

        self.tile_stride_px = int(tile_stride_px) if tile_stride_px is not None else int(self.sat_chip_size)
        if self.tile_stride_px <= 0:
            raise ValueError(f"tile_stride_px must be > 0, got {self.tile_stride_px}")

        import torchvision.transforms as T

        ground_ops: List[Any] = [T.Resize((self.image_size, self.image_size))]
        if self.train:
            ground_ops.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
        ground_ops.append(T.ToTensor())

        sat_ops: List[Any] = []
        if self.sat_image_size > 0:
            sat_ops.append(T.Resize((self.sat_image_size, self.sat_image_size)))
        if self.train:
            sat_ops.append(T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03))
        sat_ops.append(T.ToTensor())

        if self.normalize:
            norm = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ground_ops.append(norm)
            sat_ops.append(norm)

        self.sat_sampling_window_px = self._parse_valid_window_px(sat_sampling_window_px)

        self.t_ground = T.Compose(ground_ops)
        self.t_sat = T.Compose(sat_ops)

        self.retrieval_target_mode = str(retrieval_target_mode).strip().lower()
        self.retrieval_target_sigma_scale = max(float(retrieval_target_sigma_scale), 1e-6)
        self.retrieval_target_min_weight = max(float(retrieval_target_min_weight), 0.0)

        if self.retrieval_target_mode not in {"hard", "gaussian"}:
            raise ValueError(
                f"Unsupported retrieval_target_mode={retrieval_target_mode!r}; expected one of ['hard', 'gaussian']"
            )

        self._sat_ds_cache: Dict[str, Any] = {}
        self._load_manifest()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_sat_ds_cache"] = {}
        return state

    def __del__(self):
        self._close_sat_datasets()

    def _load_manifest(self) -> None:
        with self.manifest_path.open("rb") as f:
            data = pickle.load(f)

        version = int(data.get("version", -1))
        if version != MANIFEST_VERSION:
            raise RuntimeError(
                f"Manifest version mismatch: got {version}, expected {MANIFEST_VERSION}"
            )

        image_paths: List[str] = list(data["image_path"])
        json_paths: List[str] = list(data["json_path"])
        sat_paths: List[str] = list(data["sat_path"])
        site_ids: List[str] = list(data["site_id"])
        pair_ids: List[str] = list(data["pair_id"])
        lats: np.ndarray = np.asarray(data["lat"], dtype=np.float32)
        lons: np.ndarray = np.asarray(data["lon"], dtype=np.float32)

        n = len(image_paths)
        if not (
            len(json_paths) == len(sat_paths) == len(site_ids) == len(pair_ids) == n
            and len(lats) == len(lons) == n
        ):
            raise RuntimeError("Manifest columns have mismatched lengths")

        if self.site_id is not None:
            keep_idx = [i for i, sid in enumerate(site_ids) if sid == self.site_id]
            if not keep_idx:
                raise RuntimeError(f"No samples found for site_id={self.site_id!r}")

            self.image_paths = [image_paths[i] for i in keep_idx]
            self.json_paths = [json_paths[i] for i in keep_idx]
            self.sat_paths = [sat_paths[i] for i in keep_idx]
            self.site_ids = [site_ids[i] for i in keep_idx]
            self.pair_ids = [pair_ids[i] for i in keep_idx]
            self.lats = lats[keep_idx]
            self.lons = lons[keep_idx]
        else:
            self.image_paths = image_paths
            self.json_paths = json_paths
            self.sat_paths = sat_paths
            self.site_ids = site_ids
            self.pair_ids = pair_ids
            self.lats = lats
            self.lons = lons
            
        self._build_timestamp_index()

    def _get_sat_dataset(self, sat_path: str):
        if not self.keep_sat_open:
            return None
        ds = self._sat_ds_cache.get(sat_path)
        if ds is not None:
            return ds
        ds = rasterio.open(sat_path)
        self._sat_ds_cache[sat_path] = ds
        return ds

    def _close_sat_datasets(self) -> None:
        if not hasattr(self, "_sat_ds_cache"):
            return
        for ds in self._sat_ds_cache.values():
            try:
                ds.close()
            except Exception:
                pass
        self._sat_ds_cache.clear()

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_rgb(self, path: str) -> Optional[Image.Image]:
        try:
            return Image.open(path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            return None

    ################# START: n_ground selection helpers #####################
    _TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")

    def _extract_timestamp_from_path(self, path: str) -> str:
        stem = Path(path).stem
        m = self._TIMESTAMP_RE.search(stem)
        if not m:
            raise RuntimeError(
                f"Could not extract timestamp from image filename: {path!r}. "
                "Expected a substring like YYYY-MM-DD_HH-MM-SS-ffffff."
            )
        return m.group(1)


    def _build_timestamp_index(self) -> None:
        self.image_timestamps: List[str] = []
        self.timestamp_to_indices: Dict[str, List[int]] = {}

        for i, path in enumerate(self.image_paths):
            ts = self._extract_timestamp_from_path(path)
            self.image_timestamps.append(ts)
            self.timestamp_to_indices.setdefault(ts, []).append(i)
    ################# END: n_ground selection helpers #####################

    def _latlon_to_pixel(
        self,
        ds: rasterio.io.DatasetReader,
        lat: float,
        lon: float,
    ) -> Optional[Tuple[float, float]]:
        try:
            xs, ys = warp_transform("EPSG:4326", ds.crs, [float(lon)], [float(lat)])
            if len(xs) == 0 or len(ys) == 0:
                return None
            x_map = float(xs[0])
            y_map = float(ys[0])
            if not math.isfinite(x_map) or not math.isfinite(y_map):
                return None
            inv = ~ds.transform
            col_f, row_f = inv * (x_map, y_map)
            if not math.isfinite(col_f) or not math.isfinite(row_f):
                return None
            return float(col_f), float(row_f)
        except Exception:
            return None

    def _is_in_bounds(self, px: Optional[Tuple[float, float]], width: int, height: int) -> bool:
        if px is None:
            return False
        x, y = px
        return 0.0 <= float(x) < float(width) and 0.0 <= float(y) < float(height)

    def _sample_sat_chip_size(self, rng: Any = None) -> int:
        if len(self.sat_chip_sizes) <= 1:
            return int(self.sat_chip_sizes[0])
        if self.train and self.sat_sampling_mode == "positive_center":
            rand = rng if rng is not None else random
            return int(rand.choice(self.sat_chip_sizes))
        return int(self.eval_sat_chip_size)

    def _valid_center_bounds(
        self,
        width: int,
        height: int,
        chip_size: Optional[int] = None,
    ) -> Tuple[float, float, float, float]:
        chip = int(self.sat_chip_size if chip_size is None else chip_size)
        half = chip / 2.0

        left, right, top, bottom = self._sampling_rect(width=width, height=height)

        region_w = max(right - left, 0.0)
        region_h = max(bottom - top, 0.0)

        if region_w >= chip:
            min_x = left + half
            max_x = right - half
        else:
            mid_x = 0.5 * (left + right)
            min_x = mid_x
            max_x = mid_x

        if region_h >= chip:
            min_y = top + half
            max_y = bottom - half
        else:
            mid_y = 0.5 * (top + bottom)
            min_y = mid_y
            max_y = mid_y

        return min_x, max_x, min_y, max_y

    def _clamp_center(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
        chip: int,
    ) -> Tuple[float, float]:
        min_x, max_x, min_y, max_y = self._valid_center_bounds(width=width, height=height, chip_size=chip)
        x = float(min(max(x, min_x), max_x))
        y = float(min(max(y, min_y), max_y))
        return x, y

    def _center_to_topleft(
        self,
        cx: float,
        cy: float,
        chip: int,
        width: int,
        height: int,
    ) -> Tuple[int, int]:
        chip_w = min(int(chip), int(width))
        chip_h = min(int(chip), int(height))
        half_w = chip_w / 2.0
        half_h = chip_h / 2.0

        cx, cy = self._clamp_center(cx, cy, width=width, height=height, chip=chip)

        x0 = int(round(cx - half_w))
        y0 = int(round(cy - half_h))
        max_x0 = max(int(width) - chip_w, 0)
        max_y0 = max(int(height) - chip_h, 0)
        x0 = min(max(x0, 0), max_x0)
        y0 = min(max(y0, 0), max_y0)
        return x0, y0

    def _topleft_to_center(
        self,
        x0: int,
        y0: int,
        chip: int,
        width: int,
        height: int,
    ) -> Tuple[float, float]:
        chip_w = min(int(chip), int(width))
        chip_h = min(int(chip), int(height))
        return float(x0) + chip_w / 2.0, float(y0) + chip_h / 2.0

    def _axis_tiling_positions_in_region(
        self,
        region_start: float,
        region_end: float,
        full_length: int,
        chip: int,
        stride: int,
    ) -> List[int]:
        """
        Returns valid top-left positions for tiling inside [region_start, region_end),
        while keeping the chip inside both the region and the image.
        """
        if full_length <= 0:
            return [0]

        chip_eff = min(int(chip), int(full_length))
        max_global_start = max(int(full_length) - chip_eff, 0)

        min_start = max(int(math.ceil(region_start)), 0)
        max_start = min(int(math.floor(region_end - chip_eff)), max_global_start)

        if max_start < min_start:
            # Region is smaller than chip; collapse to one centered tile.
            centered = int(round(0.5 * (region_start + region_end - chip_eff)))
            centered = min(max(centered, 0), max_global_start)
            return [centered]

        xs = list(range(min_start, max_start + 1, int(stride)))
        if not xs or xs[-1] != max_start:
            xs.append(max_start)
        return xs


    def _read_chip(
        self,
        ds: rasterio.io.DatasetReader,
        center_x: float,
        center_y: float,
        chip_size: Optional[int] = None,
    ) -> Image.Image:
        chip_px = int(self.sat_chip_size if chip_size is None else chip_size)
        width = int(ds.width)
        height = int(ds.height)
        chip_w = min(chip_px, width)
        chip_h = min(chip_px, height)
        half_w = chip_w / 2.0
        half_h = chip_h / 2.0

        center_x, center_y = self._clamp_center(center_x, center_y, width=width, height=height, chip=chip_px)
        col0 = int(round(center_x - half_w))
        row0 = int(round(center_y - half_h))
        max_col0 = max(width - chip_w, 0)
        max_row0 = max(height - chip_h, 0)
        col0 = min(max(col0, 0), max_col0)
        row0 = min(max(row0, 0), max_row0)
        window = Window(col_off=col0, row_off=row0, width=chip_w, height=chip_h)

        if ds.count >= 3:
            arr = ds.read(indexes=[1, 2, 3], window=window, boundless=False)
        elif ds.count == 2:
            arr2 = ds.read(indexes=[1, 2], window=window, boundless=False)
            arr = np.concatenate([arr2, arr2[1:2]], axis=0)
        else:
            arr1 = ds.read(indexes=[1], window=window, boundless=False)
            arr = np.repeat(arr1, repeats=3, axis=0)

        arr = np.transpose(arr, (1, 2, 0))
        arr = normalize_to_uint8(arr)
        chip_img = Image.fromarray(arr, mode="RGB")
        if chip_img.size != (chip_px, chip_px):
            chip_img = chip_img.resize((chip_px, chip_px), resample=Image.BILINEAR)
        return chip_img

    def _distance_to_anchors(self, x: float, y: float, anchors: Sequence[Tuple[float, float]]) -> float:
        if not anchors:
            return float("inf")
        return min(math.hypot(x - ax, y - ay) for ax, ay in anchors)

    def _sample_negative_centers(
        self,
        width: int,
        height: int,
        anchors: Sequence[Tuple[float, float]],
        num_negatives: int,
        positive_center: Optional[Tuple[float, float]] = None,
        chip_size: Optional[int] = None,
        rng: Any = None,
    ) -> List[Tuple[float, float]]:
        if num_negatives <= 0:
            return []
        if width <= 0 or height <= 0:
            return [(0.0, 0.0)] * num_negatives

        rand = rng if rng is not None else random
        out: List[Tuple[float, float]] = []
        threshold = float(self.negative_min_distance_px)

        min_x, max_x, min_y, max_y = self._valid_center_bounds(width=width, height=height, chip_size=chip_size)
        if self.negative_local_window_px is not None and positive_center is not None:
            local = float(max(self.negative_local_window_px, 0.0))
            cand_min_x = max(min_x, float(positive_center[0]) - local)
            cand_max_x = min(max_x, float(positive_center[0]) + local)
            cand_min_y = max(min_y, float(positive_center[1]) - local)
            cand_max_y = min(max_y, float(positive_center[1]) + local)
            if cand_min_x <= cand_max_x and cand_min_y <= cand_max_y:
                min_x, max_x, min_y, max_y = cand_min_x, cand_max_x, cand_min_y, cand_max_y

        tries = 0
        while len(out) < num_negatives and tries < self.max_negative_tries:
            tries += 1
            x = rand.uniform(min_x, max_x) if max_x > min_x else min_x
            y = rand.uniform(min_y, max_y) if max_y > min_y else min_y

            d_anchor = self._distance_to_anchors(x, y, anchors)
            d_prev = self._distance_to_anchors(x, y, out)
            if d_anchor >= threshold and d_prev >= threshold * 0.5:
                out.append((x, y))

        while len(out) < num_negatives:
            x = rand.uniform(min_x, max_x) if max_x > min_x else min_x
            y = rand.uniform(min_y, max_y) if max_y > min_y else min_y
            out.append((x, y))
        return out

    def _enumerate_tiled_centers(
        self,
        width: int,
        height: int,
        chip_size: int,
        stride_px: Optional[int] = None,
    ) -> List[Tuple[float, float]]:
        stride = int(self.tile_stride_px if stride_px is None else stride_px)
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")

        left, right, top, bottom = self._sampling_rect(width=width, height=height)

        xs = self._axis_tiling_positions_in_region(
            region_start=left,
            region_end=right,
            full_length=width,
            chip=chip_size,
            stride=stride,
        )
        ys = self._axis_tiling_positions_in_region(
            region_start=top,
            region_end=bottom,
            full_length=height,
            chip=chip_size,
            stride=stride,
        )

        centers: List[Tuple[float, float]] = []
        for y0 in ys:
            for x0 in xs:
                centers.append(
                    self._topleft_to_center(
                        x0=x0,
                        y0=y0,
                        chip=chip_size,
                        width=width,
                        height=height,
                    )
                )
        return centers
    
    def _sample_random_centers(
        self,
        width: int,
        height: int,
        num_samples: int,
        chip_size: Optional[int] = None,
        rng: Any = None,
    ) -> List[Tuple[float, float]]:
        if num_samples <= 0:
            return []
        if width <= 0 or height <= 0:
            return [(0.0, 0.0)] * num_samples

        rand = rng if rng is not None else random
        min_x, max_x, min_y, max_y = self._valid_center_bounds(
            width=width,
            height=height,
            chip_size=chip_size,
        )

        out: List[Tuple[float, float]] = []
        for _ in range(num_samples):
            x = rand.uniform(min_x, max_x) if max_x > min_x else min_x
            y = rand.uniform(min_y, max_y) if max_y > min_y else min_y
            out.append((x, y))
        return out

    def _find_chip_for_pixel(
        self,
        ground_pixel: Optional[Tuple[float, float]],
        chip_bounds: Sequence[Tuple[int, int, int, int]],
    ) -> Tuple[int, Optional[Tuple[float, float]]]:
        """
        Returns:
          gt_chip_index: first chip containing the pixel, else -1
          gt_xy_in_chip: normalized [-1, 1] local coord inside that chip, else None
        """
        if ground_pixel is None:
            return -1, None

        gx, gy = float(ground_pixel[0]), float(ground_pixel[1])

        for i, (x0, y0, x1, y1) in enumerate(chip_bounds):
            if x0 <= gx < x1 and y0 <= gy < y1:
                w = max(float(x1 - x0), 1.0)
                h = max(float(y1 - y0), 1.0)
                u = (gx - float(x0)) / w
                v = (gy - float(y0)) / h
                xy = (float(u * 2.0 - 1.0), float(v * 2.0 - 1.0))
                return i, xy
        return -1, None
    
    def _chip_window_from_center(
        self,
        center: Tuple[float, float],
        chip_size: Optional[int] = None,
    ) -> Tuple[float, float, float, float]:
        chip = int(self.sat_chip_size if chip_size is None else chip_size)
        half = float(chip) * 0.5
        col0 = int(round(float(center[0]) - half))
        row0 = int(round(float(center[1]) - half))
        return float(col0), float(row0), float(col0 + chip), float(row0 + chip)
    
    def _build_retrieval_targets(
        self,
        chip_centers: Sequence[Optional[Tuple[float, float]]],
        positive_center: Tuple[float, float],
        chip_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target = torch.zeros((self.n_sat,), dtype=torch.float32)
        pos_mask = torch.zeros((self.n_sat,), dtype=torch.float32)
        valid: List[Tuple[int, Tuple[float, float]]] = []
        px = float(positive_center[0])
        py = float(positive_center[1])
        sigma = max(float(chip_size) * self.retrieval_target_sigma_scale, 1.0)

        for idx, center in enumerate(chip_centers[: self.n_sat]):
            if center is None:
                continue
            cx = float(center[0])
            cy = float(center[1])
            valid.append((idx, (cx, cy)))
            x0, y0, x1, y1 = self._chip_window_from_center((cx, cy), chip_size=chip_size)
            if x0 <= px < x1 and y0 <= py < y1:
                pos_mask[idx] = 1.0
            if self.retrieval_target_mode == "hard":
                continue
            dist_sq = (cx - px) * (cx - px) + (cy - py) * (cy - py)
            w = math.exp(-0.5 * dist_sq / (sigma * sigma))
            if w >= self.retrieval_target_min_weight:
                target[idx] = float(w)

        if not valid:
            target[0] = 1.0
            pos_mask[0] = 1.0
            return target, pos_mask

        if self.retrieval_target_mode == "hard":
            first_idx = int(valid[0][0])
            target[first_idx] = 1.0
            pos_mask[first_idx] = 1.0
            return target, pos_mask

        if float(target.sum().item()) <= 0.0:
            best_idx = min(valid, key=lambda item: (item[1][0] - px) ** 2 + (item[1][1] - py) ** 2)[0]
            target[int(best_idx)] = 1.0
        else:
            target /= target.sum().clamp_min(1e-12)

        if float(pos_mask.sum().item()) <= 0.0:
            pos_mask[int(torch.argmax(target).item())] = 1.0
        return target, pos_mask

    def _sample_satellite_chips(
        self,
        ds,
        lat: float,
        lon: float,
        rng: Any = None,
    ) -> Tuple[
        Optional[Tuple[float, float]],   # ground_pixel
        Optional[Tuple[float, float]],   # positive_center
        List[Tuple[float, float]],       # chip_centers
        List[Tuple[int, int, int, int]], # chip_bounds
        List[torch.Tensor],              # sat_imgs
        int,                             # chip_size_cur
        int,                             # gt_chip_index
        Optional[Tuple[float, float]],   # gt_xy_in_chip
    ]:
        if ds.crs is None:
            raise ValueError("Satellite raster has no CRS")
        if ds.width <= 0 or ds.height <= 0:
            raise ValueError(f"Invalid satellite raster size: {ds.width}x{ds.height}")

        rand = rng if rng is not None else random
        chip_size_cur = self._sample_sat_chip_size(rng=rand)

        ground_pixel: Optional[Tuple[float, float]] = None
        positive_center: Optional[Tuple[float, float]] = None

        if self.use_ground_truth_center:
            gp = self._latlon_to_pixel(ds, lat, lon)
            if self._is_in_bounds(gp, int(ds.width), int(ds.height)):
                ground_pixel = gp
                pos_x, pos_y = float(gp[0]), float(gp[1])

                if self.train and self.sat_sampling_mode == "positive_center" and self.positive_center_jitter_px > 0.0:
                    jitter = float(self.positive_center_jitter_px)
                    pos_x += rand.uniform(-jitter, jitter)
                    pos_y += rand.uniform(-jitter, jitter)

                pos_x, pos_y = self._clamp_center(
                    pos_x,
                    pos_y,
                    width=int(ds.width),
                    height=int(ds.height),
                    chip=chip_size_cur,
                )
                positive_center = (pos_x, pos_y)

        if self.sat_sampling_mode == "positive_center":
            if positive_center is None:
                raise ValueError(
                    "sat_sampling_mode='positive_center' requires a valid ground-truth center. "
                    "Use 'random' or 'tiled' when GT may be unavailable."
                )

            anchors: List[Tuple[float, float]] = [positive_center]
            if ground_pixel is not None:
                anchors.append((float(ground_pixel[0]), float(ground_pixel[1])))

            negatives = self._sample_negative_centers(
                width=int(ds.width),
                height=int(ds.height),
                anchors=anchors,
                num_negatives=max(0, self.n_sat - 1),
                positive_center=positive_center,
                chip_size=chip_size_cur,
                rng=rand,
            )
            chip_centers = [positive_center] + negatives

        elif self.sat_sampling_mode == "random":
            chip_centers = self._sample_random_centers(
                width=int(ds.width),
                height=int(ds.height),
                num_samples=self.random_n_sat,
                chip_size=chip_size_cur,
                rng=rand,
            )

        elif self.sat_sampling_mode == "tiled":
            chip_centers = self._enumerate_tiled_centers(
                width=int(ds.width),
                height=int(ds.height),
                chip_size=chip_size_cur,
                stride_px=self.tile_stride_px,
            )

        else:
            raise RuntimeError(f"Unsupported sat_sampling_mode={self.sat_sampling_mode!r}")

        sat_imgs: List[torch.Tensor] = []
        chip_bounds: List[Tuple[int, int, int, int]] = []
        sat_w = int(ds.width)
        sat_h = int(ds.height)

        for cx, cy in chip_centers:
            x0, y0 = self._center_to_topleft(cx, cy, chip=chip_size_cur, width=sat_w, height=sat_h)
            chip_w = min(int(chip_size_cur), sat_w)
            chip_h = min(int(chip_size_cur), sat_h)
            x1 = x0 + chip_w
            y1 = y0 + chip_h

            chip_img = self._read_chip(ds, cx, cy, chip_size=chip_size_cur)
            chip_bounds.append((int(x0), int(y0), int(x1), int(y1)))
            sat_imgs.append(self.t_sat(chip_img))

        gt_chip_index, gt_xy_in_chip = self._find_chip_for_pixel(ground_pixel, chip_bounds)

        return (
            ground_pixel,
            positive_center if self.sat_sampling_mode == "positive_center" else None,
            chip_centers,
            chip_bounds,
            sat_imgs,
            int(chip_size_cur),
            int(gt_chip_index),
            gt_xy_in_chip,
        )

    # def _sample_ground_indices_for_timestamp(
    #     self,
    #     idx: int,
    #     rng: Any = None,
    # ) -> List[int]:
    #     rand = rng if rng is not None else random

    #     ts = self.image_timestamps[idx]
    #     candidates = self.timestamp_to_indices.get(ts, [])
    #     if not candidates:
    #         raise RuntimeError(
    #             f"No images found for timestamp {ts!r} at idx={idx}"
    #         )

    #     if self.n_ground <= 1:
    #         return [idx]

    #     if len(candidates) < self.n_ground:
    #         raise RuntimeError(
    #             f"Requested n_ground={self.n_ground}, but only found "
    #             f"{len(candidates)} image(s) with timestamp {ts!r}."
    #         )

    #     # Prefer including the anchor/current image in the sampled set.
    #     remaining = [i for i in candidates if i != idx]
    #     if self.n_ground == 1:
    #         return [idx]

    #     sampled = [idx] + rand.sample(remaining, self.n_ground - 1)
    #     rand.shuffle(sampled)
    #     return sampled

    def _sample_ground_indices_for_timestamp(
        self,
        idx: int,
        rng: Any = None,
    ) -> List[int]:
        rand = rng if rng is not None else random

        ts = self.image_timestamps[idx]
        candidates = self.timestamp_to_indices.get(ts, [])

        if not candidates:
            raise RuntimeError(
                f"No images found for timestamp {ts!r} at idx={idx}"
            )

        if self.n_ground <= 1:
            return [idx]

        if len(candidates) < self.n_ground:
            warnings.warn(
                f"Requested n_ground={self.n_ground}, but only found "
                f"{len(candidates)} image(s) with timestamp {ts!r}. "
                f"Sampling with replacement.",
                RuntimeWarning,
            )

            # Ensure anchor is included if possible
            sampled = [idx]

            # sample remaining WITH replacement (excluding idx if possible)
            remaining = [i for i in candidates if i != idx]
            pool = remaining if remaining else candidates

            for _ in range(self.n_ground - 1):
                sampled.append(rand.choice(pool))

            rand.shuffle(sampled)
            return sampled

        # Normal case: enough candidates (sample without replacement)
        remaining = [i for i in candidates if i != idx]
        sampled = [idx] + rand.sample(remaining, self.n_ground - 1)
        rand.shuffle(sampled)
        return sampled

    def _load_and_pack_ground_group(
        self,
        indices: Sequence[int],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        List[str],
        List[Tuple[float, float]],
        List[int],
    ]:
        imgs: List[torch.Tensor] = []
        paths: List[str] = []
        latlons: List[Tuple[float, float]] = []
        out_indices: List[int] = []

        for gi in indices:
            path = self.image_paths[gi]
            img = self._load_rgb(path)
            if img is None:
                raise ValueError(f"Failed to load ground image: {path}")
            imgs.append(self.t_ground(img))
            paths.append(path)
            latlons.append((float(self.lats[gi]), float(self.lons[gi])))
            out_indices.append(int(gi))

        c, h, w = imgs[0].shape
        ground = torch.zeros((len(imgs), c, h, w), dtype=torch.float32)
        mask = torch.ones((len(imgs),), dtype=torch.float32)

        for i, x in enumerate(imgs):
            ground[i] = x

        return ground, mask, paths, latlons, out_indices
    
    def _ground_pixels_for_indices(
        self,
        ds: rasterio.io.DatasetReader,
        indices: Sequence[int],
    ) -> List[Optional[Tuple[float, float]]]:
        out: List[Optional[Tuple[float, float]]] = []

        for gi in indices:
            gp: Optional[Tuple[float, float]] = None
            if self.use_ground_truth_center:
                gp_try = self._latlon_to_pixel(
                    ds,
                    float(self.lats[gi]),
                    float(self.lons[gi]),
                )
                if self._is_in_bounds(gp_try, int(ds.width), int(ds.height)):
                    gp = gp_try
            out.append(gp)

        return out

    def _pack_sat(self, imgs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not imgs:
            raise ValueError("imgs must be non-empty")

        c, h, w = imgs[0].shape
        n_cur = len(imgs)
        sat = torch.zeros((n_cur, c, h, w), dtype=torch.float32)
        sat_mask = torch.ones((n_cur,), dtype=torch.float32)
        for i, x in enumerate(imgs):
            sat[i] = x
        return sat, sat_mask

    def _build_pos_targets(
        self,
        positive_center: Tuple[float, float],
        ground_pixels: Sequence[Optional[Tuple[float, float]]],
        chip_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = len(ground_pixels)
        pos_xy = torch.zeros((n, 2), dtype=torch.float32)
        pos_mask = torch.zeros((n,), dtype=torch.float32)

        chip = float(self.sat_chip_size if chip_size is None else chip_size)
        half = chip / 2.0
        x0 = positive_center[0] - half
        y0 = positive_center[1] - half

        for i, px in enumerate(ground_pixels):
            if px is None:
                continue
            gx, gy = px
            local_x = (gx - x0) / chip
            local_y = (gy - y0) / chip
            if 0.0 <= local_x < 1.0 and 0.0 <= local_y < 1.0:
                pos_xy[i, 0] = float(local_x * 2.0 - 1.0)
                pos_xy[i, 1] = float(local_y * 2.0 - 1.0)
                pos_mask[i] = 1.0

        u = torch.clamp((pos_xy[:, 0] + 1.0) * 0.5, 0.0, 1.0 - 1e-6)
        v = torch.clamp((pos_xy[:, 1] + 1.0) * 0.5, 0.0, 1.0 - 1e-6)
        ix = torch.floor(u * self.pos_grid).long()
        iy = torch.floor(v * self.pos_grid).long()
        pos_label = (iy * self.pos_grid + ix) * (pos_mask > 0).long()
        return pos_xy, pos_mask, pos_label
    
    def _parse_valid_window_px(
        self,
        valid_window_px: Optional[int | Tuple[int, int] | Sequence[int]],
    ) -> Optional[Tuple[int, int]]:
        if valid_window_px is None:
            return None

        if isinstance(valid_window_px, (int, float)):
            v = int(valid_window_px)
            if v <= 0:
                raise ValueError(f"valid_window_px must be > 0, got {valid_window_px}")
            return (v, v)

        if isinstance(valid_window_px, Sequence) and len(valid_window_px) == 2:
            w = int(valid_window_px[0])
            h = int(valid_window_px[1])
            if w <= 0 or h <= 0:
                raise ValueError(f"valid_window_px values must be > 0, got {valid_window_px}")
            return (w, h)

        raise ValueError(
            "valid_window_px must be None, a positive int, or a 2-tuple/list like (width_px, height_px)"
        )
    
    def _sampling_rect(
        self,
        width: int,
        height: int,
    ) -> Tuple[float, float, float, float]:
        """
        Returns a centered rectangle (left, right, top, bottom) in pixel-edge coordinates
        that defines the allowed sampling area.

        If valid_window_px is None, this is the full image.
        """
        if self.sat_sampling_window_px is None:
            return 0.0, float(width), 0.0, float(height)

        win_w = min(int(self.sat_sampling_window_px[0]), int(width))
        win_h = min(int(self.sat_sampling_window_px[1]), int(height))

        left = (float(width) - float(win_w)) * 0.5
        right = left + float(win_w)

        top = (float(height) - float(win_h)) * 0.5
        bottom = top + float(win_h)

        return left, right, top, bottom

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import traceback

        attempt = 0
        cur_idx = int(idx)
        last_exc = None

        while attempt < self.max_retry:
            rand = random if self.train else random.Random(int(idx))
            attempt += 1

            image_path = self.image_paths[cur_idx]
            json_path = self.json_paths[cur_idx]
            sat_path = self.sat_paths[cur_idx]
            site_id = self.site_ids[cur_idx]
            pair_id = self.pair_ids[cur_idx]
            lat = float(self.lats[cur_idx])
            lon = float(self.lons[cur_idx])

            try:
                img = self._load_rgb(image_path)
                if img is None:
                    cur_idx = rand.randrange(len(self))
                    continue

                ground_indices = self._sample_ground_indices_for_timestamp(cur_idx, rng=rand)
                ground, mask, ground_paths, ground_latlon, ground_sample_indices = (
                    self._load_and_pack_ground_group(ground_indices)
                )

                try:
                    if self.keep_sat_open:
                        ds = self._get_sat_dataset(sat_path)
                        (
                            ground_pixel,
                            positive_center,
                            chip_centers,
                            chip_bounds,
                            sat_imgs,
                            sat_chip_size_used,
                            gt_chip_index,
                            gt_xy_in_chip,
                        ) = self._sample_satellite_chips(ds=ds, lat=lat, lon=lon, rng=rand)
                    else:
                        with rasterio.open(sat_path) as ds:
                            (
                                ground_pixel,
                                positive_center,
                                chip_centers,
                                chip_bounds,
                                sat_imgs,
                                sat_chip_size_used,
                                gt_chip_index,
                                gt_xy_in_chip,
                            ) = self._sample_satellite_chips(ds=ds, lat=lat, lon=lon, rng=rand)
                except Exception:
                    cur_idx = rand.randrange(len(self))
                    continue

                if not sat_imgs:
                    cur_idx = rand.randrange(len(self))
                    continue

                sat, sat_mask = self._pack_sat(sat_imgs)

                n_ground_out = int(ground.shape[0])
                ground_pixels = self._ground_pixels_for_indices(ds, ground_sample_indices)

                if len(ground_pixels) != n_ground_out:
                    raise RuntimeError(
                        f"ground_pixels length mismatch: got {len(ground_pixels)} expected {n_ground_out}"
                    )
                
                retrieval_target = None
                retrieval_pos_mask = None
                if positive_center is not None:
                    pos_xy, pos_mask, pos_label = self._build_pos_targets(
                        positive_center=positive_center,
                        ground_pixels=ground_pixels,
                        chip_size=sat_chip_size_used,
                    )
                    retrieval_target, retrieval_pos_mask = self._build_retrieval_targets(
                        chip_centers=chip_centers,
                        positive_center=positive_center,
                        chip_size=sat_chip_size_used,
                    )
                else:
                    pos_xy = torch.zeros((n_ground_out, 2), dtype=torch.float32)
                    pos_mask = torch.zeros((n_ground_out,), dtype=torch.float32)
                    pos_label = torch.zeros((n_ground_out,), dtype=torch.long)

                orient_label = torch.zeros((n_ground_out,), dtype=torch.long)
                orient_mask = torch.zeros((n_ground_out,), dtype=torch.float32)

                if self.channel_last:
                    ground = ground.permute(0, 2, 3, 1).contiguous()
                    sat = sat.permute(0, 2, 3, 1).contiguous()

                sat_stem = Path(sat_path).stem
                return {
                    "scene_id": f"{site_id}/{pair_id}::{sat_stem}",
                    "site_id": site_id,
                    "pair_id": pair_id,
                    "image_path": image_path,
                    "json_path": json_path,
                    "sat_path": sat_path,
                    "ground_paths": ground_paths,
                    "ground_latlon": ground_latlon,
                    "ground_pixels": ground_pixels,
                    "has_ground_truth_center": ground_pixel is not None,
                    "positive_center": positive_center,   # only meaningful in positive_center mode
                    "chip_centers": chip_centers,
                    "chip_bounds": chip_bounds,
                    "sat_chip_size_used": int(sat_chip_size_used),
                    "sat_sampling_mode": self.sat_sampling_mode,
                    "random_n_sat": int(self.random_n_sat),
                    "tile_stride_px": int(self.tile_stride_px),
                    "num_sat": int(sat.shape[0]),
                    "sat_sampling_window_px": self.sat_sampling_window_px,
                    "gt_chip_index": int(gt_chip_index),
                    "gt_xy_in_chip": gt_xy_in_chip,
                    "ground": ground,
                    "mask": mask,
                    "sat": sat,
                    "sat_mask": sat_mask,
                    "retrieval_target": retrieval_target,
                    "retrieval_pos_mask": retrieval_pos_mask,
                    "pos_xy": pos_xy,
                    "pos_mask": pos_mask,
                    "pos_label": pos_label,
                    "orient_label": orient_label,
                    "orient_mask": orient_mask,
                }

            except (rasterio.errors.RasterioError, ValueError, OSError) as e:
                last_exc = e
                print(
                    f"[GETITEM RETRY] idx={cur_idx} attempt={attempt}/{self.max_retry} "
                    f"scene={site_id}/{pair_id} error={type(e).__name__}: {e}",
                    flush=True,
                )
                cur_idx = rand.randrange(len(self))
                continue

            except Exception:
                print(
                    f"[GETITEM FATAL] idx={cur_idx} attempt={attempt}/{self.max_retry} "
                    f"scene={site_id}/{pair_id}",
                    flush=True,
                )
                traceback.print_exc()
                raise

        raise RuntimeError(
            f"Failed to sample a valid item after {self.max_retry} retries "
            f"(start_idx={idx}, last_idx={cur_idx}, "
            f"last_error={type(last_exc).__name__ if last_exc else None}: {last_exc})"
        ) from last_exc


def collate_image_pair(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # Fixed-shape tensors
    fixed_tensor_keys = [
        "ground",
        "mask",
        "pos_xy",
        "pos_mask",
        "pos_label",
        "retrieval_target",
        "retrieval_pos_mask",
        "orient_label",
        "orient_mask",
    ]
    for k in fixed_tensor_keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)

    # Variable-length satellite chips: pad to batch max
    sat_list = [b["sat"] for b in batch]
    sat_mask_list = [b["sat_mask"] for b in batch]

    max_n_sat = max(int(x.shape[0]) for x in sat_list)
    sat_shape0 = sat_list[0].shape

    if len(sat_shape0) != 4:
        raise ValueError(f"Expected sat tensor rank 4, got shape={sat_shape0}")

    padded_sat = []
    padded_sat_mask = []

    for sat, sat_mask in zip(sat_list, sat_mask_list):
        n_cur = int(sat.shape[0])
        if n_cur < max_n_sat:
            pad_shape = (max_n_sat - n_cur, *sat.shape[1:])
            sat_pad = torch.zeros(pad_shape, dtype=sat.dtype)
            sat = torch.cat([sat, sat_pad], dim=0)

            mask_pad = torch.zeros((max_n_sat - n_cur,), dtype=sat_mask.dtype)
            sat_mask = torch.cat([sat_mask, mask_pad], dim=0)

        padded_sat.append(sat)
        padded_sat_mask.append(sat_mask)

    out["sat"] = torch.stack(padded_sat, dim=0)
    out["sat_mask"] = torch.stack(padded_sat_mask, dim=0)

    meta_keys = [
        "scene_id",
        "site_id",
        "pair_id",
        "image_path",
        "json_path",
        "sat_path",
        "ground_paths",
        "ground_latlon",
        "ground_pixels",
        "has_ground_truth_center",
        "positive_center",
        "chip_centers",
        "chip_bounds",
        "sat_chip_size_used",
        "sat_sampling_mode",
        "random_n_sat",
        "tile_stride_px",
        "sat_sampling_window_px",
        "num_sat",
        "gt_chip_index",
        "gt_xy_in_chip",
    ]
    for k in meta_keys:
        out[k] = [b[k] for b in batch]

    return out