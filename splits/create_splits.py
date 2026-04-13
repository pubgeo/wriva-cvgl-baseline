#!/usr/bin/env python3

import argparse
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import rasterio
from rasterio.warp import transform


IMAGE_DIR_CANDIDATES = ("image", "ground")
SAT_DIR_CANDIDATES = ("maxar", "maxar-optical")
REF_DIR_NAME = "reference"

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
SAT_EXTS = {".tif", ".tiff"}


@dataclass
class SiteData:
    name: str
    root: Path
    image_dir: Path
    sat_dir: Path
    ref_dir: Optional[Path]
    images: List[Path]
    satellites: List[Path]
    image_to_ref: Dict[Path, Optional[Path]]


def find_first_existing_dir(parent: Path, candidates: Sequence[str]) -> Optional[Path]:
    for c in candidates:
        p = parent / c
        if p.is_dir():
            return p
    return None


def collect_sites(root_dir: Path) -> List[SiteData]:
    sites: List[SiteData] = []

    for site_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        image_dir = find_first_existing_dir(site_dir, IMAGE_DIR_CANDIDATES)
        sat_dir = find_first_existing_dir(site_dir, SAT_DIR_CANDIDATES)
        ref_dir = site_dir / REF_DIR_NAME if (site_dir / REF_DIR_NAME).is_dir() else None

        if image_dir is None or sat_dir is None:
            warnings.warn(
                f"Skipping site '{site_dir.name}' because it does not contain "
                f"one of {IMAGE_DIR_CANDIDATES} and one of {SAT_DIR_CANDIDATES}."
            )
            continue

        images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        satellites = sorted([p for p in sat_dir.iterdir() if p.suffix.lower() in SAT_EXTS])

        image_to_ref: Dict[Path, Optional[Path]] = {}

        for img in images:
            ref_path = None

            # 1) First: look in the SAME folder as the image
            same_dir_candidate = img.parent / f"{img.stem}.json"
            if same_dir_candidate.exists():
                ref_path = same_dir_candidate

            # 2) Fallback: look in the reference folder
            elif ref_dir is not None:
                ref_candidate = ref_dir / f"{img.stem}.json"
                if ref_candidate.exists():
                    ref_path = ref_candidate

            image_to_ref[img] = ref_path

        if not images:
            warnings.warn(f"Skipping site '{site_dir.name}' because it has no images.")
            continue
        if not satellites:
            warnings.warn(f"Skipping site '{site_dir.name}' because it has no satellites.")
            continue

        sites.append(
            SiteData(
                name=site_dir.name,
                root=site_dir,
                image_dir=image_dir,
                sat_dir=sat_dir,
                ref_dir=ref_dir,
                images=images,
                satellites=satellites,
                image_to_ref=image_to_ref,
            )
        )

    return sites


def split_sites(
    sites: List[SiteData],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[SiteData], List[SiteData], List[SiteData]]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    sites = list(sites)
    rng = random.Random(seed)
    rng.shuffle(sites)

    n = len(sites)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    # Adjust to ensure total matches n.
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = n - n_train - n_val

    train_sites = sites[:n_train]
    val_sites = sites[n_train:n_train + n_val]
    test_sites = sites[n_train + n_val:]

    # Safety check
    assert len(train_sites) + len(val_sites) + len(test_sites) == n
    assert set(s.name for s in train_sites).isdisjoint(set(s.name for s in val_sites))
    assert set(s.name for s in train_sites).isdisjoint(set(s.name for s in test_sites))
    assert set(s.name for s in val_sites).isdisjoint(set(s.name for s in test_sites))

    return train_sites, val_sites, test_sites


def read_satellite_rgb_preview(
    sat_path: Path,
    max_dim: int = 1024
) -> np.ndarray:
    """
    Read a preview RGB array from a GeoTIFF for heuristic filtering.
    Returns uint8/float ndarray with shape (H, W, C).
    """
    with rasterio.open(sat_path) as src:
        count = src.count
        if count < 1:
            raise ValueError(f"No bands found in {sat_path}")

        out_h = src.height
        out_w = src.width
        scale = max(out_h / max_dim, out_w / max_dim, 1.0)
        preview_h = max(1, int(round(out_h / scale)))
        preview_w = max(1, int(round(out_w / scale)))

        if count >= 3:
            bands = [1, 2, 3]
        else:
            bands = [1]

        arr = src.read(
            bands,
            out_shape=(len(bands), preview_h, preview_w),
            resampling=rasterio.enums.Resampling.bilinear,
        )

        arr = np.moveaxis(arr, 0, -1)  # (H, W, C)

        # Normalize to 0..255 for heuristic checks.
        arr = arr.astype(np.float32)
        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            return np.zeros((preview_h, preview_w, arr.shape[-1]), dtype=np.uint8)

        vals = arr[finite_mask]
        vmin = np.percentile(vals, 1)
        vmax = np.percentile(vals, 99)

        if vmax <= vmin:
            vmax = vmin + 1.0

        arr = (arr - vmin) / (vmax - vmin)
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
        return arr


def compute_whitespace_fraction(rgb: np.ndarray) -> float:
    """
    Heuristic whitespace fraction:
    - pixels that are very bright in all channels
    - OR nearly black in all channels (common nodata padding)
    """
    if rgb.ndim == 2:
        rgb = rgb[..., None]

    if rgb.shape[-1] == 1:
        gray = rgb[..., 0]
        white = gray >= 250
        black = gray <= 5
    else:
        white = np.all(rgb >= 250, axis=-1)
        black = np.all(rgb <= 5, axis=-1)

    whitespace = white | black
    return float(whitespace.mean())


def rgb_to_hsv_np(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Vectorized RGB->HSV, returns H,S,V in [0,1].
    """
    rgb = rgb_uint8.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    s = np.zeros_like(cmax)
    v = cmax

    nonzero = cmax > 0
    s[nonzero] = delta[nonzero] / cmax[nonzero]

    mask = delta > 0
    rmask = mask & (cmax == r)
    gmask = mask & (cmax == g)
    bmask = mask & (cmax == b)

    h[rmask] = ((g[rmask] - b[rmask]) / delta[rmask]) % 6
    h[gmask] = ((b[gmask] - r[gmask]) / delta[gmask]) + 2
    h[bmask] = ((r[bmask] - g[bmask]) / delta[bmask]) + 4
    h = h / 6.0

    hsv = np.stack([h, s, v], axis=-1)
    return hsv


def compute_cloud_fraction(rgb: np.ndarray) -> float:
    """
    Simple heuristic cloud detector:
    - bright
    - low saturation
    - not counted as whitespace

    This is a heuristic, not a physical cloud mask.
    """
    if rgb.ndim == 2 or rgb.shape[-1] == 1:
        gray = rgb[..., 0] if rgb.ndim == 3 else rgb
        cloud = gray >= 220
        whitespace = (gray >= 250) | (gray <= 5)
        valid_cloud = cloud & (~whitespace)
        return float(valid_cloud.mean())

    hsv = rgb_to_hsv_np(rgb[..., :3])
    s = hsv[..., 1]
    v = hsv[..., 2]

    whitespace = np.all(rgb[..., :3] >= 250, axis=-1) | np.all(rgb[..., :3] <= 5, axis=-1)
    cloud = (v >= 0.82) & (s <= 0.22) & (~whitespace)
    return float(cloud.mean())


def satellite_is_valid(
    sat_path: Path,
    whitespace_threshold: float = 0.5,
    cloud_threshold: float = 0.10,
) -> Tuple[bool, Dict[str, float]]:
    try:
        rgb = read_satellite_rgb_preview(sat_path)
        whitespace_fraction = compute_whitespace_fraction(rgb)
        cloud_fraction = compute_cloud_fraction(rgb)

        valid = (
            whitespace_fraction < whitespace_threshold
            and cloud_fraction < cloud_threshold
        )

        return valid, {
            "whitespace_fraction": whitespace_fraction,
            "cloud_fraction": cloud_fraction,
        }
    except Exception as e:
        warnings.warn(f"Failed to evaluate satellite '{sat_path}': {e}")
        return False, {"whitespace_fraction": 1.0, "cloud_fraction": 1.0}


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def extract_lonlat_points_from_reference(ref_data: dict) -> Optional[List[Tuple[float, float]]]:
    """
    Extract reference coordinates from the provided JSON schema.

    Expected format:
    {
        ...
        "extrinsics": {
            "lat": ...,
            "lon": ...
        },
        ...
    }

    Returns a list of (lon, lat) points.
    For this schema, we only have a single camera location point.
    """
    if not isinstance(ref_data, dict):
        return None

    extrinsics = ref_data.get("extrinsics")
    if not isinstance(extrinsics, dict):
        return None

    lat = extrinsics.get("lat")
    lon = extrinsics.get("lon")

    if lat is None or lon is None:
        return None

    try:
        return [(float(lon), float(lat))]
    except (TypeError, ValueError):
        return None

def points_within_satellite(ref_json_path: Path, sat_path: Path) -> bool:
    """
    Returns True if all extracted reference points lie within the satellite bounds.

    Assumptions:
    - reference coordinates are lon/lat in EPSG:4326
    - satellite has valid CRS and bounds
    """
    ref_data = load_json(ref_json_path)
    lonlat_points = extract_lonlat_points_from_reference(ref_data)

    if not lonlat_points:
        warnings.warn(
            f"Could not parse geospatial reference from '{ref_json_path}'. "
            f"Excluding pair because validation is enabled."
        )
        return False

    with rasterio.open(sat_path) as src:
        if src.crs is None:
            warnings.warn(
                f"Satellite '{sat_path}' has no CRS. "
                f"Excluding pair because validation is enabled."
            )
            return False

        lons = [p[0] for p in lonlat_points]
        lats = [p[1] for p in lonlat_points]

        try:
            xs, ys = transform("EPSG:4326", src.crs, lons, lats)
        except Exception as e:
            warnings.warn(
                f"Failed to transform coordinates for '{ref_json_path}' against '{sat_path}': {e}"
            )
            return False

        bounds = src.bounds
        for x, y in zip(xs, ys):
            if not (bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top):
                return False

    return True


def build_pairs_for_sites(
    sites: List[SiteData],
    validate_with_reference: bool,
    relative_to: Optional[Path] = None,
    label: Optional[str] = ""
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []

    satellite_validity_cache: Dict[Path, bool] = {}

    for site in tqdm(sites, desc=f"Building {label} pairs"):
        valid_sats: List[Path] = []

        for sat in site.satellites:
            if sat not in satellite_validity_cache:
                ok, stats = satellite_is_valid(sat)
                satellite_validity_cache[sat] = ok
                if not ok:
                    warnings.warn(
                        f"Rejected satellite '{sat}' "
                        f"(whitespace={stats['whitespace_fraction']:.3f}, "
                        f"cloud={stats['cloud_fraction']:.3f})"
                    )
            if satellite_validity_cache[sat]:
                valid_sats.append(sat)

        if not valid_sats:
            warnings.warn(f"No valid satellites left for site '{site.name}'.")
            continue

        for img in site.images:
            ref_json = site.image_to_ref.get(img)

            for sat in valid_sats:
                keep_pair = True

                if validate_with_reference:
                    if ref_json is None or not ref_json.exists():
                        keep_pair = False
                    else:
                        keep_pair = points_within_satellite(ref_json, sat)

                if keep_pair:
                    if relative_to is not None:
                        img_str = str(img.relative_to(relative_to))
                        sat_str = str(sat.relative_to(relative_to))
                    else:
                        img_str = str(img.resolve())
                        sat_str = str(sat.resolve())
                    pairs.append((img_str, sat_str))

    return pairs


def write_pairs_txt(pairs: List[Tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for img_path, sat_path in pairs:
            f.write(f"{img_path} {sat_path}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/val/test site-level splits with image-satellite pair files."
    )
    parser.add_argument("--root_dir", type=Path, required=True, help="Root directory containing site folders.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for train.txt, val.txt, test.txt.")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--validate_with_reference",
        action="store_true",
        help="If set, require the reference JSON footprint/point to lie within the satellite extent.",
    )

    parser.add_argument(
        "--output_relative_paths",
        action="store_true",
        help="If set, write paths relative to root_dir instead of absolute paths.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root_dir = args.root_dir.resolve()
    out_dir = args.out_dir.resolve()

    sites = collect_sites(root_dir)
    if not sites:
        raise RuntimeError(f"No valid sites found under {root_dir}")

    train_sites, val_sites, test_sites = split_sites(
        sites,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"Found {len(sites)} valid sites total")
    print(f"Train sites: {[s.name for s in train_sites]}")
    print(f"Val sites:   {[s.name for s in val_sites]}")
    print(f"Test sites:  {[s.name for s in test_sites]}")

    relative_to = root_dir if args.output_relative_paths else None

    train_pairs = build_pairs_for_sites(
        train_sites,
        validate_with_reference=args.validate_with_reference,
        relative_to=relative_to,
        label="train",
    )
    val_pairs = build_pairs_for_sites(
        val_sites,
        validate_with_reference=args.validate_with_reference,
        relative_to=relative_to,
        label="val",
    )
    test_pairs = build_pairs_for_sites(
        test_sites,
        validate_with_reference=args.validate_with_reference,
        relative_to=relative_to,
        label="test",
    )

    if len(train_pairs)>0:
        write_pairs_txt(train_pairs, out_dir / "train_split.txt")
        print(f"Wrote {len(train_pairs)} pairs to {out_dir / 'train_split.txt'}")
    if len(val_pairs)>0:
        write_pairs_txt(val_pairs, out_dir / "val_split.txt")
        print(f"Wrote {len(val_pairs)} pairs to {out_dir / 'val_split.txt'}")
    if len(test_pairs)>0:
        write_pairs_txt(test_pairs, out_dir / "test_split.txt")
        print(f"Wrote {len(test_pairs)} pairs to {out_dir / 'test_split.txt'}")



if __name__ == "__main__":
    main()