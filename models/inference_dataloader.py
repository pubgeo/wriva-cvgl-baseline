from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Any, Iterable, Tuple

import math
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def load_rgb_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")

def parse_sampling_window_px(
    sat_sampling_window_px,
) -> Optional[Tuple[int, int]]:
    if sat_sampling_window_px is None:
        return None

    if isinstance(sat_sampling_window_px, int):
        if sat_sampling_window_px <= 0:
            raise ValueError(f"sat_sampling_window_px must be > 0, got {sat_sampling_window_px}")
        return (sat_sampling_window_px, sat_sampling_window_px)

    if isinstance(sat_sampling_window_px, (tuple, list)) and len(sat_sampling_window_px) == 2:
        w, h = int(sat_sampling_window_px[0]), int(sat_sampling_window_px[1])
        if w <= 0 or h <= 0:
            raise ValueError(
                f"sat_sampling_window_px values must be > 0, got {sat_sampling_window_px}"
            )
        return (w, h)

    raise ValueError(
        "sat_sampling_window_px must be None, an int, or a 2-element list/tuple like [width_px, height_px]"
    )


def sampling_rect(
    width: int,
    height: int,
    sat_sampling_window_px: Optional[Tuple[int, int]],
) -> Tuple[float, float, float, float]:
    if sat_sampling_window_px is None:
        return 0.0, float(width), 0.0, float(height)

    win_w = min(int(sat_sampling_window_px[0]), int(width))
    win_h = min(int(sat_sampling_window_px[1]), int(height))

    left = (float(width) - float(win_w)) * 0.5
    right = left + float(win_w)
    top = (float(height) - float(win_h)) * 0.5
    bottom = top + float(win_h)

    return left, right, top, bottom


def axis_tiling_positions_in_region(
    region_start: float,
    region_end: float,
    full_length: int,
    chip: int,
    stride: int,
) -> List[int]:
    if full_length <= 0:
        return [0]

    chip_eff = min(int(chip), int(full_length))
    max_global_start = max(int(full_length) - chip_eff, 0)

    min_start = max(int(math.ceil(region_start)), 0)
    max_start = min(int(math.floor(region_end - chip_eff)), max_global_start)

    if max_start < min_start:
        centered = int(round(0.5 * (region_start + region_end - chip_eff)))
        centered = min(max(centered, 0), max_global_start)
        return [centered]

    xs = list(range(min_start, max_start + 1, int(stride)))
    if not xs or xs[-1] != max_start:
        xs.append(max_start)
    return xs


def tile_pil_image_with_boxes(
    img: Image.Image,
    chip_size: int,
    tile_stride_px: int,
    sat_sampling_window_px: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Image.Image], List[Tuple[int, int, int, int]]]:
    """
    Returns:
      chips: list of PIL chip crops
      boxes: list of (x1, y1, x2, y2) boxes in the possibly padded image coordinates
    """
    w, h = img.size

    if w < chip_size or h < chip_size:
        pad_w = max(0, chip_size - w)
        pad_h = max(0, chip_size - h)
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top
        img = T.Pad((pad_left, pad_top, pad_right, pad_bottom))(img)
        w, h = img.size

    left, right, top, bottom = sampling_rect(
        width=w,
        height=h,
        sat_sampling_window_px=sat_sampling_window_px,
    )

    xs = axis_tiling_positions_in_region(
        region_start=left,
        region_end=right,
        full_length=w,
        chip=chip_size,
        stride=tile_stride_px,
    )
    ys = axis_tiling_positions_in_region(
        region_start=top,
        region_end=bottom,
        full_length=h,
        chip=chip_size,
        stride=tile_stride_px,
    )

    chips: List[Image.Image] = []
    boxes: List[Tuple[int, int, int, int]] = []

    for y in ys:
        for x in xs:
            chips.append(img.crop((x, y, x + chip_size, y + chip_size)))
            boxes.append((x, y, x + chip_size, y + chip_size))

    return chips, boxes, [left, right, top, bottom]


class ClusterInferenceDataset(Dataset):
    """
    Dataset that yields one cluster at a time.

    Expected directory structure:
      image_base_dir/
        <site_id>/
          ground/
            ...
          maxar/
            ...

    Output dict:
      {
        "cluster_id": int,
        "site_id": str,
        "ground_imgs": Tensor[N, 3, ground_image_size, ground_image_size],
        "ground_mask": Tensor[N],
        "sat_imgs": Tensor[M, 3, sat_chip_size, sat_chip_size],   # flat across all satellite images
        "ground_paths": List[str],
        "sat_paths": List[str],
        "chip_counts": List[int],                         # len == number of satellite images
        "chip_metadata": List[dict],                      # len == M
      }

    Notes:
    - N = number of ground images in the cluster
    - M = total number of chips across all site-level maxar images
    """

    def __init__(
        self,
        txt_file: str | Path,
        sat_chip_size: int,
        tile_stride_px: int,
        image_base_dir: str | Path,
        site_ids: Optional[Iterable[str]] = None,
        sat_sampling_window_px: Optional[int | Tuple[int, int]] = None,
        ground_image_size:int = 224,
        ground_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        sat_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        sort_paths: bool = True,
        has_header: bool = True,
        delimiter: str = "\t",
    ):
        self.txt_file = Path(txt_file)
        self.sat_chip_size = int(sat_chip_size)
        self.tile_stride_px = int(tile_stride_px)
        self.image_base_dir = Path(image_base_dir)

        if self.tile_stride_px <= 0:
            raise ValueError(f"tile_stride_px must be > 0, got {self.tile_stride_px}")

        if not self.image_base_dir.exists():
            raise FileNotFoundError(f"image_base_dir does not exist: {self.image_base_dir}")

        all_site_ids = self._discover_valid_site_ids()

        if site_ids is None:
            self.site_ids = all_site_ids
        else:
            site_ids = set(site_ids)
            missing = site_ids - set(all_site_ids)
            if missing:
                raise ValueError(
                    f"Requested site_ids not found or invalid (missing ground/maxar): {sorted(missing)}"
                )
            self.site_ids = sorted(site_ids)

        self.sat_sampling_window_px = parse_sampling_window_px(sat_sampling_window_px)
        self.sort_paths = sort_paths
        self.has_header = has_header
        self.delimiter = delimiter
        self.ground_image_size = ground_image_size

        self.ground_transform = T.Compose([
            T.Resize((self.ground_image_size, self.ground_image_size)),
            T.ToTensor(),
        ])
        self.sat_transform = sat_transform or T.ToTensor()

        self._ground_index = self._build_ground_index()
        self.samples = self._parse_txt()

    def _discover_valid_site_ids(self) -> List[str]:
        site_ids = []

        for p in self.image_base_dir.iterdir():
            if not p.is_dir():
                continue

            ground_dir = p / "ground"
            maxar_dir = p / "maxar"

            if ground_dir.is_dir() and maxar_dir.is_dir():
                site_ids.append(p.name)

        if not site_ids:
            raise RuntimeError(
                f"No valid site_ids found under {self.image_base_dir}. "
                f"Expected structure: <site_id>/ground and <site_id>/maxar"
            )

        return sorted(site_ids)

    def _get_site_ground_dir(self, site_id: str) -> Path:
        return self.image_base_dir / site_id / "ground"

    def _get_site_maxar_dir(self, site_id: str) -> Path:
        return self.image_base_dir / site_id / "maxar"

    def _build_ground_index(self) -> Dict[str, Optional[str]]:
        """
        Maps:
        - relative path under ground dir -> site_id
        - basename -> site_id if unique, else None
        """
        index: Dict[str, Optional[str]] = {}

        for site_id in self.site_ids:
            ground_dir = self._get_site_ground_dir(site_id)

            for p in ground_dir.rglob("*"):
                if not p.is_file():
                    continue

                rel = p.relative_to(ground_dir)
                index[str(rel)] = site_id

                if p.name not in index:
                    index[p.name] = site_id
                else:
                    index[p.name] = None

        return index

    def _find_site_id_for_ground_file(self, ground_filename_from_txt: str, line_num: int) -> str:
        """
        Resolve site_id using:
        1. txt_file path substring match against available site_ids
        2. fallback to ground filename / relative path lookup in the ground index
        """

        # ---- First: preserve original behavior using txt_file path ----
        txt_path_str = str(self.txt_file)
        txt_matches = [site_id for site_id in self.site_ids if site_id in txt_path_str]

        if len(txt_matches) == 1:
            return txt_matches[0]

        if len(txt_matches) > 1:
            raise ValueError(
                f"Line {line_num}: multiple site_id matches for txt path '{self.txt_file}': "
                f"{txt_matches}"
            )

        # ---- Fallback: resolve from ground filename/path ----
        rel_key = str(Path(ground_filename_from_txt))
        base_key = Path(ground_filename_from_txt).name

        rel_match = self._ground_index.get(rel_key, None)
        if rel_match is not None:
            return rel_match

        base_match = self._ground_index.get(base_key, None)
        if base_match is not None:
            return base_match

        raise FileNotFoundError(
            f"Line {line_num}: could not resolve site_id for ground file "
            f"'{ground_filename_from_txt}'. "
            f"No unique match found from txt path '{self.txt_file}' or ground index."
        )

    def _resolve_ground_path(self, ground_filename_from_txt: str, site_id: str, line_num: int) -> str:
        ground_dir = self._get_site_ground_dir(site_id)

        if not ground_dir.exists():
            raise FileNotFoundError(
                f"Line {line_num}: ground dir does not exist for site_id='{site_id}': {ground_dir}"
            )

        path_from_txt = Path(ground_filename_from_txt)

        # Try exact relative path under ground/
        candidate_rel = ground_dir / path_from_txt
        if candidate_rel.exists():
            return str(candidate_rel)

        # Fallback to basename directly under ground/
        candidate_base = ground_dir / path_from_txt.name
        if candidate_base.exists():
            return str(candidate_base)

        raise FileNotFoundError(
            f"Line {line_num}: expected ground file '{ground_filename_from_txt}' "
            f"under '{ground_dir}', but it was not found."
        )

    def _parse_txt(self) -> List[Dict[str, Any]]:
        clusters = defaultdict(lambda: {"ground_paths": [], "site_id": None})

        with open(self.txt_file, "r") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                if self.has_header and line_num == 1:
                    continue

                parts = line.split(self.delimiter)
                if len(parts) < 2:
                    parts = line.split()
                    if len(parts) < 2:
                        raise ValueError(
                            f"Line {line_num}: expected at least 2 columns, got: {line}"
                        )

                ground_filename = parts[0]
                cluster_id = int(parts[1])

                site_id = self._find_site_id_for_ground_file(
                    ground_filename_from_txt=ground_filename,
                    line_num=line_num,
                )

                if self.site_ids is not None and site_id not in self.site_ids:
                    continue

                ground_path = self._resolve_ground_path(
                    ground_filename_from_txt=ground_filename,
                    site_id=site_id,
                    line_num=line_num,
                )

                existing_site_id = clusters[cluster_id]["site_id"]
                if existing_site_id is None:
                    clusters[cluster_id]["site_id"] = site_id
                elif existing_site_id != site_id:
                    raise ValueError(
                        f"Cluster {cluster_id} contains mixed site_ids: "
                        f"'{existing_site_id}' and '{site_id}'. "
                        f"Each cluster must belong to exactly one site."
                    )

                clusters[cluster_id]["ground_paths"].append(ground_path)

        samples = []
        for cluster_id, data in clusters.items():
            ground_paths = data["ground_paths"]
            site_id = data["site_id"]

            if site_id is None:
                raise ValueError(f"Cluster {cluster_id} is missing site_id")

            if len(ground_paths) == 0:
                continue

            if self.sort_paths:
                ground_paths = sorted(ground_paths)

            maxar_dir = self._get_site_maxar_dir(site_id)
            if not maxar_dir.exists():
                raise FileNotFoundError(
                    f"Expected maxar dir does not exist for site_id='{site_id}': {maxar_dir}"
                )

            samples.append(
                {
                    "cluster_id": cluster_id,
                    "ground_paths": ground_paths,
                    "maxar_dir": str(maxar_dir),
                    "site_id": site_id,
                }
            )

        samples = sorted(samples, key=lambda x: x["cluster_id"])
        return samples

    def _get_satellite_paths_for_cluster(self, maxar_dir: str | Path) -> List[str]:
        maxar_dir = Path(maxar_dir)
        if not maxar_dir.exists():
            raise FileNotFoundError(f"Maxar directory does not exist: {maxar_dir}")

        sat_paths = [str(p) for p in maxar_dir.iterdir() if p.is_file() and is_image_file(p)]
        if self.sort_paths:
            sat_paths = sorted(sat_paths)

        if len(sat_paths) == 0:
            raise RuntimeError(f"No satellite images found in: {maxar_dir}")

        return sat_paths

    def _load_ground_stack(self, ground_paths: List[str]) -> torch.Tensor:
        ground_tensors = []
        for p in ground_paths:
            img = load_rgb_image(p)
            tensor = self.ground_transform(img)
            ground_tensors.append(tensor)
        return torch.stack(ground_tensors, dim=0)

    def _load_satellite_chip_groups(
        self,
        sat_paths: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[List[Dict[str, Any]]]]:
        """
        Returns:
        sat_imgs: Tensor[B_sat, M_max, C, H, W]
        sat_mask: Tensor[B_sat, M_max]   # True where chip is valid
        chip_counts: List[int]           # len == B_sat
        chip_metadata: List[List[dict]]  # one list per satellite image
        """
        sat_chip_groups: List[torch.Tensor] = []
        chip_counts: List[int] = []
        chip_metadata: List[List[Dict[str, Any]]] = []

        for sat_idx, sat_path in enumerate(sat_paths):
            sat_img = load_rgb_image(sat_path)

            chips, boxes, sampled_rect = tile_pil_image_with_boxes(
                sat_img,
                chip_size=self.sat_chip_size,
                tile_stride_px=self.tile_stride_px,
                sat_sampling_window_px=self.sat_sampling_window_px,
            )

            if len(chips) == 0:
                continue

            chip_tensor = torch.stack(
                [self.sat_transform(chip) for chip in chips],
                dim=0,
            )  # [M_i, C, H, W]

            sat_chip_groups.append(chip_tensor)
            chip_counts.append(len(chips))

            per_sat_metadata = []
            for local_chip_idx, box in enumerate(boxes):
                per_sat_metadata.append(
                    {
                        "sat_index": int(sat_idx),
                        "sat_path": str(sat_path),
                        "tiled_area_lrtb": sampled_rect,
                        "chip_index_local": int(local_chip_idx),
                        "chip_box_xyxy": [int(v) for v in box],
                    }
                )
            chip_metadata.append(per_sat_metadata)

        if not sat_chip_groups:
            raise RuntimeError("No satellite chips were generated.")

        b_sat = len(sat_chip_groups)
        m_max = max(t.shape[0] for t in sat_chip_groups)
        c, h, w = sat_chip_groups[0].shape[1:]

        sat_imgs = torch.zeros(
            b_sat, m_max, c, h, w,
            dtype=sat_chip_groups[0].dtype,
        )
        sat_mask = torch.zeros(b_sat, m_max, dtype=torch.bool)

        for i, chips_i in enumerate(sat_chip_groups):
            m_i = chips_i.shape[0]
            sat_imgs[i, :m_i] = chips_i
            sat_mask[i, :m_i] = True

        return sat_imgs, sat_mask, chip_counts, chip_metadata

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        cluster_id = sample["cluster_id"]
        site_id = sample["site_id"]
        ground_paths = sample["ground_paths"]
        maxar_dir = sample["maxar_dir"]

        sat_paths = self._get_satellite_paths_for_cluster(maxar_dir)

        # [N, 3, H, W]
        ground_imgs_single = self._load_ground_stack(ground_paths)
        ground_mask_single = torch.ones(ground_imgs_single.shape[0], dtype=torch.bool)

        # [B_sat, M_max, 3, Hs, Ws], [B_sat, M_max]
        sat_imgs, sat_mask, chip_counts, chip_metadata = self._load_satellite_chip_groups(sat_paths)

        b_sat = sat_imgs.shape[0]
        n = ground_imgs_single.shape[0]

        # Repeat the same ground cluster once per satellite image
        # ground_imgs: [B_sat, N, 3, H, W]
        ground_imgs = ground_imgs_single.unsqueeze(0).repeat(b_sat, 1, 1, 1, 1)

        # ground_mask: [B_sat, N]
        ground_mask = ground_mask_single.unsqueeze(0).repeat(b_sat, 1)

        return {
            "cluster_id": cluster_id,
            "site_id": site_id,
            "ground_imgs": ground_imgs,    # [B_sat, N, 3, H, W]
            "ground_mask": ground_mask,    # [B_sat, N]
            "sat_imgs": sat_imgs,          # [B_sat, M_max, 3, Hs, Ws]
            "sat_mask": sat_mask,          # [B_sat, M_max]
            "ground_paths": ground_paths,
            "sat_paths": sat_paths,
            "chip_counts": chip_counts,
            "chip_metadata": chip_metadata,
        }
    
def cluster_inference_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) != 1:
        raise ValueError(
            "This collate_fn expects batch_size=1 because each dataset item "
            "already contains the model batch dimension over satellite candidates."
        )
    return batch[0]

# def cluster_inference_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """
#     Pads variable-length dimensions across a batch.

#     Output:
#       ground_imgs: [B, N_max, 3, H, W]
#       ground_mask: [B, N_max]
#       sat_imgs:    [B, M_max, 3, Hs, Ws]
#       sat_mask:    [B, M_max]
#     """
#     batch_size = len(batch)

#     max_n = max(item["ground_imgs"].shape[0] for item in batch)
#     max_m = max(item["sat_imgs"].shape[0] for item in batch)

#     _, c_g, h_g, w_g = batch[0]["ground_imgs"].shape
#     _, c_s, h_s, w_s = batch[0]["sat_imgs"].shape

#     ground_imgs = torch.zeros(batch_size, max_n, c_g, h_g, w_g, dtype=batch[0]["ground_imgs"].dtype)
#     ground_mask = torch.zeros(batch_size, max_n, dtype=torch.bool)

#     sat_imgs = torch.zeros(batch_size, max_m, c_s, h_s, w_s, dtype=batch[0]["sat_imgs"].dtype)
#     sat_mask = torch.zeros(batch_size, max_m, dtype=torch.bool)

#     cluster_ids = []
#     site_ids = []
#     ground_paths = []
#     sat_paths = []
#     chip_counts_batch = []
#     chip_metadata_batch = []

#     for i, item in enumerate(batch):
#         n = item["ground_imgs"].shape[0]
#         m = item["sat_imgs"].shape[0]

#         ground_imgs[i, :n] = item["ground_imgs"]
#         ground_mask[i, :n] = item["ground_mask"]

#         sat_imgs[i, :m] = item["sat_imgs"]
#         sat_mask[i, :m] = True

#         cluster_ids.append(item["cluster_id"])
#         site_ids.append(item["site_id"])
#         ground_paths.append(item["ground_paths"])
#         sat_paths.append(item["sat_paths"])
#         chip_counts_batch.append(item["chip_counts"])
#         chip_metadata_batch.append(item["chip_metadata"])

#     return {
#         "cluster_id": torch.tensor(cluster_ids, dtype=torch.long),
#         "site_id": site_ids,
#         "ground_imgs": ground_imgs,
#         "ground_mask": ground_mask,
#         "sat_imgs": sat_imgs,
#         "sat_mask": sat_mask,
#         "ground_paths": ground_paths,
#         "sat_paths": sat_paths,
#         "chip_counts": chip_counts_batch,
#         "chip_metadata": chip_metadata_batch,
#     }