#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RASTER_EXTS = {".tif", ".tiff", ".jp2"}
MANIFEST_VERSION = 1

IMAGE_DIR_NAMES = {"image", "ground"}
SAT_DIR_NAMES = {"maxar", "maxar-optical"}


def _find_lat_lon_in_dict(d: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    key_pairs = [
        ("lat", "lon"),
        ("latitude", "longitude"),
        ("Lat", "Lon"),
        ("LAT", "LON"),
    ]
    for k_lat, k_lon in key_pairs:
        if k_lat in d and k_lon in d:
            try:
                return float(d[k_lat]), float(d[k_lon])
            except Exception:
                return None
    return None


def _walk_find_lat_lon(obj: Any) -> Optional[Tuple[float, float]]:
    if isinstance(obj, dict):
        ll = _find_lat_lon_in_dict(obj)
        if ll is not None:
            return ll
        for value in obj.values():
            ll = _walk_find_lat_lon(value)
            if ll is not None:
                return ll
    elif isinstance(obj, list):
        for value in obj:
            ll = _walk_find_lat_lon(value)
            if ll is not None:
                return ll
    return None


def extract_lat_lon(json_path: Path) -> Optional[Tuple[float, float]]:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    for container in (data.get("extrinsics"), data.get("metadata"), data):
        if isinstance(container, dict):
            ll = _find_lat_lon_in_dict(container)
            if ll is not None:
                return ll
    return _walk_find_lat_lon(data)


def resolve_pair_path(path_text: str, pairs_txt: Path, dataset_root: Optional[Path]) -> Path:
    p = Path(path_text).expanduser()
    if p.is_absolute():
        return p.resolve()
    base = dataset_root if dataset_root is not None else pairs_txt.parent
    return (base / p).resolve()


def parse_pair_line(
    line: str,
    line_no: int,
    pairs_txt: Path,
    dataset_root: Optional[Path],
) -> Optional[Tuple[Path, Path]]:
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None
    if "\t" in raw:
        parts = raw.split("\t")
    else:
        parts = raw.split(maxsplit=1)
    if len(parts) < 2:
        raise ValueError(f"Invalid pair line at {pairs_txt}:{line_no}: {raw}")
    image_path = resolve_pair_path(parts[0], pairs_txt, dataset_root)
    sat_path = resolve_pair_path(parts[1], pairs_txt, dataset_root)
    return image_path, sat_path


def _find_site_root_from_image(image_path: Path) -> Optional[Path]:
    """
    Given something like:
      <root>/<site>/image/foo.jpg
      <root>/<site>/ground/foo.jpg
    return:
      <root>/<site>
    """
    for parent in image_path.parents:
        if parent.name in IMAGE_DIR_NAMES:
            return parent.parent
    return None


def infer_site_id(image_path: Path) -> str:
    site_root = _find_site_root_from_image(image_path)
    if site_root is not None:
        return site_root.name
    return "unknown_site"


def reference_json_for_image(image_path: Path) -> Optional[Path]:
    """
    Look for reference json in either:
      1. same folder as image:      <site>/image/foo.json
      2. reference folder:          <site>/reference/foo.json

    Returns the preferred existing path if found, otherwise returns the
    most likely reference-folder path.
    """
    stem = image_path.stem

    # Option 1: same directory as image
    same_dir_json = image_path.with_suffix(".json")
    if same_dir_json.is_file():
        return same_dir_json

    # Option 2: <site>/reference/<stem>.json
    site_root = _find_site_root_from_image(image_path)
    if site_root is not None:
        ref_json = site_root / "reference" / f"{stem}.json"
        if ref_json.is_file():
            return ref_json
        return ref_json

    # Fallback if structure is unexpected
    return same_dir_json


def build_manifest(
    pairs_txt: Path,
    dataset_root: Optional[Path],
    site_filter: Optional[Sequence[str]],
    validate_paths: bool,
) -> Dict[str, Any]:
    site_filter_set = set(site_filter) if site_filter else None

    image_paths: List[str] = []
    json_paths: List[str] = []
    sat_paths: List[str] = []
    site_ids: List[str] = []
    pair_ids: List[str] = []
    lats: List[float] = []
    lons: List[float] = []

    num_lines = 0
    num_kept = 0
    num_skipped = 0

    total_lines = sum(1 for _ in pairs_txt.open("r", encoding="utf-8"))

    with pairs_txt.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(
            tqdm(f, total=total_lines, desc="Processing pairs"),
            start=1,
        ):
            num_lines += 1
            parsed = parse_pair_line(line, line_no, pairs_txt, dataset_root)
            if parsed is None:
                continue

            image_path, sat_path = parsed
            site_id = infer_site_id(image_path)
            if site_filter_set is not None and site_id not in site_filter_set:
                continue

            json_path = reference_json_for_image(image_path)
            if json_path is None:
                num_skipped += 1
                continue

            if validate_paths:
                if not image_path.is_file():
                    num_skipped += 1
                    continue
                if image_path.suffix.lower() not in IMAGE_EXTS:
                    num_skipped += 1
                    continue
                if not sat_path.is_file():
                    num_skipped += 1
                    continue
                if sat_path.suffix.lower() not in RASTER_EXTS:
                    num_skipped += 1
                    continue
                if not json_path.is_file():
                    num_skipped += 1
                    continue

                # Optional sanity checks for expected folder names
                if image_path.parent.name not in IMAGE_DIR_NAMES:
                    num_skipped += 1
                    continue
                if sat_path.parent.name not in SAT_DIR_NAMES:
                    num_skipped += 1
                    continue

            ll = extract_lat_lon(json_path)
            if ll is None:
                num_skipped += 1
                continue

            image_paths.append(str(image_path))
            json_paths.append(str(json_path))
            sat_paths.append(str(sat_path))
            site_ids.append(site_id)
            pair_ids.append(image_path.stem)
            lats.append(float(ll[0]))
            lons.append(float(ll[1]))
            num_kept += 1

            if line_no % 100000 == 0:
                print(
                    f"[BUILD] lines={line_no} kept={num_kept} skipped={num_skipped}",
                    flush=True,
                )

    manifest = {
        "version": MANIFEST_VERSION,
        "pairs_txt": str(pairs_txt.resolve()),
        "dataset_root": None if dataset_root is None else str(dataset_root),
        "count": num_kept,
        "image_path": image_paths,
        "json_path": json_paths,
        "sat_path": sat_paths,
        "site_id": site_ids,
        "pair_id": pair_ids,
        "lat": np.asarray(lats, dtype=np.float32),
        "lon": np.asarray(lons, dtype=np.float32),
    }
    print(
        f"[DONE] lines={num_lines} kept={num_kept} skipped={num_skipped}",
        flush=True,
    )
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-txt", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--site", action="append", default=None, help="Repeatable site filter")
    ap.add_argument(
        "--no-validate-paths",
        action="store_true",
        help="Skip is_file()/suffix validation while building",
    )
    args = ap.parse_args()

    pairs_txt = args.pairs_txt.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve() if args.dataset_root is not None else None
    output = args.output.expanduser().resolve()

    manifest = build_manifest(
        pairs_txt=pairs_txt,
        dataset_root=dataset_root,
        site_filter=args.site,
        validate_paths=not args.no_validate_paths,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(manifest, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[WROTE] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())