import argparse
import json
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Type, Optional, Callable, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.inference_dataloader import (
    ClusterInferenceDataset,
    cluster_inference_collate_fn,
)
from models.helpers import unwrap_model

from models.flex_geo_match import (
    FlexGeoApprox as FlexGeoApprox_flex_geo,
    pos_logits_to_heatmap as pos_logits_to_heatmap_flex_geo,
    in_sample_contrastive as in_sample_contrastive_flex_geo,
)

from models.flex_geo_match_dinov3 import (
    FlexGeoApprox as FlexGeoApprox_flex_geo_dinov3,
    pos_logits_to_heatmap as pos_logits_to_heatmap_flex_geo_dinov3,
    in_sample_contrastive as in_sample_contrastive_flex_geo_dinov3,
)

from models.flex_geo_match_dinov3_posloss import (
    FlexGeoApprox as FlexGeoApprox_flex_geo_dinov3_posloss,
    pos_logits_to_heatmap as pos_logits_to_heatmap_flex_geo_dinov3_posloss,
    in_sample_contrastive as in_sample_contrastive_flex_geo_dinov3_posloss,
)

from models.flex_geo_match_dinov3_posloss_v2 import (
    FlexGeoApprox as FlexGeoApprox_flex_geo_dinov3_posloss_v2,
    pos_logits_to_heatmap as pos_logits_to_heatmap_flex_geo_dinov3_posloss_v2,
    in_sample_contrastive as in_sample_contrastive_flex_geo_dinov3_posloss_v2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on tiled clusters")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/yejz1/wriva/CVGL/src/inference_configs/infer_tiles.json",
        help="Path to inference config file",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def get_device(device_str: str) -> str:
    if device_str == "cpu":
        return "cpu"
    return device_str if torch.cuda.is_available() else "cpu"


def resolve_model_paths(model_cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    model_root = Path(model_cfg["model_root"])
    checkpoint_path = model_root / model_cfg["checkpoint_name"]
    run_cfg_path = model_root / f'{model_cfg["config_filename"]}.json'

    return checkpoint_path, run_cfg_path


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "flex_geo": {
        "model_cls": FlexGeoApprox_flex_geo,
        "pos_logits_to_heatmap": pos_logits_to_heatmap_flex_geo,
        "in_sample_contrastive": in_sample_contrastive_flex_geo,
    },
    "flex_geo_dinov3": {
        "model_cls": FlexGeoApprox_flex_geo_dinov3,
        "pos_logits_to_heatmap": pos_logits_to_heatmap_flex_geo_dinov3,
        "in_sample_contrastive": in_sample_contrastive_flex_geo_dinov3,
    },
    "flex_geo_dinov3_posloss": {
        "model_cls": FlexGeoApprox_flex_geo_dinov3_posloss,
        "pos_logits_to_heatmap": pos_logits_to_heatmap_flex_geo_dinov3_posloss,
        "in_sample_contrastive": in_sample_contrastive_flex_geo_dinov3_posloss,
    },
    "flex_geo_dinov3_posloss_v2": {
        "model_cls": FlexGeoApprox_flex_geo_dinov3_posloss_v2,
        "pos_logits_to_heatmap": pos_logits_to_heatmap_flex_geo_dinov3_posloss_v2,
        "in_sample_contrastive": in_sample_contrastive_flex_geo_dinov3_posloss_v2,
    },
}


def get_model_family_fns(model_type) -> Tuple[Type, Callable, Callable]:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model_type={model_type!r}. "
            f"Expected one of: {list(MODEL_REGISTRY.keys())}"
        )

    entry = MODEL_REGISTRY[model_type]
    return (
        entry["model_cls"],
        entry["pos_logits_to_heatmap"],
        entry["in_sample_contrastive"],
    )


def build_model_kwargs(
    model_type: str,
    run_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    p = run_cfg

    common_kwargs = {
        "embed_dim": int(p.get("embed_dim", 1024)),
        "enable_pos": bool(p.get("pos_mode", True)),
        "pos_mode": p.get("pos_mode", "grid"),
        "pos_grid": int(p.get("pos_grid", 4)),
        "pretrained": False, # Always false during inference
        "enable_ial": p.get("enable_ial", True),
        "ial_num_classes": int(p.get("ial_num_classes", 4)),
    }

    if model_type == "flex_geo":
        return common_kwargs

    if model_type == "flex_geo_dinov3":
        return {
            **common_kwargs,
            "pos_loss_type": p.get("pos_loss_type", "reg"),
            "pos_reg_beta": float(p.get("pos_reg_beta", 0.1)),
            "backbone_model_id": p.get(
                "backbone_model_id",
                "facebook/dinov3-vitb16-pretrain-lvd1689m",
            ),
            "sff_scale": float(p.get("sff_scale", 2.0)),
            "share_backbone": bool(p.get("share_backbone", True)),
        }

    if model_type == "flex_geo_dinov3_posloss":
        return {
            **common_kwargs,
            "pos_loss_type": p.get("pos_loss_type", "reg"),
            "pos_reg_beta": float(p.get("pos_reg_beta", 0.1)),
            "pos_head_variant": p.get("pos_head_variant", "pairwise_residual"),
            "pos_head_hidden_dim": int(p.get("pos_head_hidden_dim", 1024)),
            "pos_head_depth": int(p.get("pos_head_depth", 2)),
            "separate_pos_neck": bool(p.get("separate_pos_neck", True)),
            "backbone_model_id": p.get(
                "backbone_model_id",
                "facebook/dinov3-vitb16-pretrain-lvd1689m",
            ),
            "sff_scale": float(p.get("sff_scale", 2.0)),
            "share_backbone": bool(p.get("share_backbone", True)),
        }

    if model_type == "flex_geo_dinov3_posloss_v2":
        return {
            **common_kwargs,
            "pos_loss_type": p.get("pos_loss_type", "reg"),
            "pos_reg_beta": float(p.get("pos_reg_beta", 0.1)),
            "pos_head_variant": p.get("pos_head_variant", "pairwise_residual"),
            "pos_head_hidden_dim": int(p.get("pos_head_hidden_dim", 1024)),
            "pos_head_depth": int(p.get("pos_head_depth", 2)),
            "pos_heatmap_loss": p.get("pos_heatmap_loss", "soft_ce"),
            "pos_heatmap_sigma": float(p.get("pos_heatmap_sigma", 1.0)),
            "pos_heatmap_xy_weight": float(p.get("pos_heatmap_xy_weight", 0.25)),
            "separate_pos_neck": bool(p.get("separate_pos_neck", True)),
            "backbone_model_id": p.get(
                "backbone_model_id",
                "facebook/dinov3-vitb16-pretrain-lvd1689m",
            ),
            "sff_scale": float(p.get("sff_scale", 2.0)),
            "share_backbone": bool(p.get("share_backbone", True)),
        }
    
    raise ValueError(f"Model type {model_type} not supported for parameter loading")


def load_model(
    ckpt_path: Path,
    model_class,
    run_cfg_path: Path,
    model_type: str,
    device: str,
) -> torch.nn.Module:
    raw_run_cfg = load_json(run_cfg_path)
    run_cfg = raw_run_cfg.get("config", raw_run_cfg)

    model_kwargs = build_model_kwargs(
        model_type=model_type,
        run_cfg=run_cfg,
    )

    model = model_class(**model_kwargs)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model


def build_dataset(data_cfg: Dict[str, Any]) -> ClusterInferenceDataset:
    sat_chip_size = int(data_cfg.get("sat_chip_size", 256))
    overlap_ratio = float(data_cfg.get("overlap_ratio", 0.5))
    tile_stride_px = int((1.0 - overlap_ratio) * sat_chip_size)
    tile_stride_px = max(tile_stride_px, 1)

    return ClusterInferenceDataset(
        txt_file=data_cfg["txt_file"],
        sat_chip_size=sat_chip_size,
        tile_stride_px=tile_stride_px,
        image_base_dir=data_cfg["image_base_dir"],
        site_ids=data_cfg.get("site_ids", None),
        ground_image_size=data_cfg["ground_image_size"],
        sat_sampling_window_px=data_cfg.get("sat_sampling_window_px", None),
        has_header=data_cfg.get("has_header", True),
        delimiter=data_cfg.get("delimiter", "\t"),
    )


def build_dataloader(
    dataset: ClusterInferenceDataset,
    runtime_cfg: Dict[str, Any],
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=runtime_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=cluster_inference_collate_fn,
    )


def _compute_logits_from_outputs(
    outputs: Dict[str, Any],
    sat_mask: Optional[torch.Tensor],
    in_sample_contrastive_fn,
) -> Tuple[torch.Tensor, bool]:
    """
    Returns:
      logits: [B_sat, M]
      use_in_sample: bool

    Assumes:
      outputs["g_set"]  -> [B_sat, D]
      outputs["s_all"]  -> [B_sat, M, D]  (preferred)
      outputs["s"]      -> [B_sat, D] or [B_sat, M, D]
      sat_mask          -> [B_sat, M] or None
    """
    g_set = outputs["g_set"]   # [B_sat, D]
    temp = outputs["temp"]

    use_in_sample = "s_all" in outputs
    if use_in_sample and sat_mask is not None and sat_mask.sum(dim=1).max().item() <= 1:
        use_in_sample = False

    if use_in_sample:
        s_all = outputs["s_all"]   # [B_sat, M, D]
        _, logits = in_sample_contrastive_fn(g_set, s_all, sat_mask, temp)
    else:
        s = outputs["s"]
        if s.ndim == 3:
            logits = torch.einsum("bd,bmd->bm", g_set, s)
        elif s.ndim == 2:
            # one embedding per satellite, shape [B_sat, D]
            logits = torch.sum(g_set * s, dim=1, keepdim=True)   # [B_sat, 1]
        else:
            raise ValueError(f"Unexpected satellite embedding shape: {tuple(s.shape)}")

    if sat_mask is not None:
        if logits.ndim != 2:
            raise ValueError(f"Expected logits to be 2D [B_sat, M], got {tuple(logits.shape)}")
        if logits.shape != sat_mask.shape:
            # allow [B_sat, 1] when only one chip is represented
            if logits.shape[0] != sat_mask.shape[0] or logits.shape[1] != 1:
                raise ValueError(
                    f"logits/sat_mask shape mismatch: logits={tuple(logits.shape)} "
                    f"sat_mask={tuple(sat_mask.shape)}"
                )
        if logits.shape == sat_mask.shape:
            logits = logits.masked_fill(~sat_mask.bool(), float("-inf"))

    return logits, bool(use_in_sample)


def compute_logits_and_heatmaps(
    model: torch.nn.Module,
    model_type: str,
    batch: Dict[str, Any],
    device: str,
    in_sample_contrastive_fn,
    pos_logits_to_heatmap_fn,
) -> Dict[str, Any]:
    model.eval()
    core_model = unwrap_model(model)

    ground_imgs = batch["ground_imgs"].to(device, non_blocking=True)   # [B_sat, N, 3, H, W]
    ground_mask = batch["ground_mask"].to(device, non_blocking=True)   # [B_sat, N]
    sat_imgs = batch["sat_imgs"].to(device, non_blocking=True)         # [B_sat, M, 3, H, W]

    sat_mask = batch.get("sat_mask")
    if sat_mask is not None:
        sat_mask = sat_mask.to(device, non_blocking=True)              # [B_sat, M]

    amp_device = "cuda" if torch.device(device).type == "cuda" else "cpu"

    with torch.no_grad():
        with torch.amp.autocast(device_type=amp_device, enabled=False):
            outputs = core_model(
                ground_imgs=ground_imgs,
                ground_mask=ground_mask,
                sat_imgs=sat_imgs,
            )

            logits, use_in_sample = _compute_logits_from_outputs(
                outputs=outputs,
                sat_mask=sat_mask,
                in_sample_contrastive_fn=in_sample_contrastive_fn,
            )

            # ------------------------------------------------------------------
            # Extra: compute position predictions for ALL M satellite candidates
            # sat_imgs: [B, M, 3, H, W]
            # ------------------------------------------------------------------
            model_for_pos = core_model.module if hasattr(core_model, "module") else core_model

            if model_for_pos.enable_pos and sat_imgs.dim() == 5:
                B, M, C, H, W = sat_imgs.shape

                # Reuse ground positional embeddings already produced by forward:
                # outputs["g_emb"] is retrieval embedding, not what pos head needs.
                # So recompute the ground positional embedding exactly as in forward.
                g = ground_imgs.view(B * ground_imgs.shape[1], ground_imgs.shape[2], ground_imgs.shape[3], ground_imgs.shape[4])
                ground_views = model_for_pos.encode_ground_views(g)
                g_pos_emb = ground_views.pos.view(B, ground_imgs.shape[1], -1)   # [B, N, D]

                # Flatten satellite candidates to [B*M, 3, H, W]
                sat_flat = sat_imgs.view(B * M, C, H, W)

                # Repeat ground positional embeddings so each satellite candidate
                # gets paired with its corresponding ground set.
                g_pos_flat = g_pos_emb.repeat_interleave(M, dim=0)  # [B*M, N, D]

                # Chunk to control memory
                pos_chunk_size = 64  # adjust if needed
                pos_all_parts = {}

                for start in range(0, B * M, pos_chunk_size):
                    end = min(start + pos_chunk_size, B * M)

                    sat_chunk = sat_flat[start:end]
                    g_pos_chunk = g_pos_flat[start:end]

                    if model_type=="flex_geo_dinov3_posloss":
                        pos_inference_kwargs=dict(
                            ground_pos_emb=g_pos_chunk,
                            pos_mode=model_for_pos.pos_mode,
                            pos_grid=model_for_pos.pos_grid,
                        )
                    elif model_type=="flex_geo_dinov3_posloss_v2":
                        pos_inference_kwargs=dict(
                            ground_pos_emb=g_pos_chunk,
                        )
                    else:
                        raise ValueError(f"Unsupported model_class for position inference: {model_type}")

                    if model_for_pos.uses_token_pos_head:
                        sat_pos_token_grid = model_for_pos.encode_sat_pos_token_grid(sat_chunk)
                        pos_chunk_out = model_for_pos.predict_position_outputs(
                            sat_pos_token_grid=sat_pos_token_grid,
                            **pos_inference_kwargs
                        )
                    else:
                        sat_views_chunk = model_for_pos.encode_sat_views(sat_chunk)
                        pos_chunk_out = model_for_pos.predict_position_outputs(
                            sat_pos_emb=sat_views_chunk.pos,
                            **pos_inference_kwargs
                        )

                    # pos_chunk_out.keys(): dict_keys(['pos_logits', 'pos_xy_pred'])
                    for k, v in pos_chunk_out.items():
                        pos_all_parts.setdefault(k, []).append(v)

                # Reassemble and reshape back to [B, M, ...]
                for k, parts in pos_all_parts.items():
                    v = torch.cat(parts, dim=0)          # [B*M, ...]
                    outputs[f"{k}_all"] = v.view(B, M, *v.shape[1:])

    return {
        "logits": logits,                        # [B_sat, M]
        "use_in_sample": use_in_sample,
        "pos_logits": outputs["pos_logits_all"], # -> [B, M, ...]
        "pos_xy_pred": outputs["pos_xy_pred_all"],  # -> [B, M, ...]
        # "pos_heatmap": pos_heatmap,       # one heatmap per satellite
    }

def aggregate_chip_scores_to_satellite_scores(
    chip_scores: np.ndarray,
    reduction: str = "max",
) -> np.ndarray:
    """
    Args:
      chip_scores: [B_sat, M] array, with invalid entries already masked to -inf.

    Returns:
      satellite_scores: [B_sat]
    """
    if chip_scores.ndim != 2:
        raise ValueError(f"Expected chip_scores with shape [B_sat, M], got {chip_scores.shape}")

    if reduction == "max":
        return np.max(chip_scores, axis=1).astype(np.float32)
    elif reduction == "mean":
        finite_mask = np.isfinite(chip_scores)
        sums = np.where(finite_mask, chip_scores, 0.0).sum(axis=1)
        counts = finite_mask.sum(axis=1)
        out = np.full(chip_scores.shape[0], -np.inf, dtype=np.float32)
        valid = counts > 0
        out[valid] = (sums[valid] / counts[valid]).astype(np.float32)
        return out
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

## Reference code to output only top k satellite chips, rather than all chips
## In our inference workflow, we need all chips to be output from inference
# def build_topk_chips_per_satellite(
#     chip_scores: np.ndarray,
#     sat_paths: List[str],
#     chip_counts: List[int],
#     chip_metadata: List[List[Dict[str, Any]]],
#     top_k_chips: int = 10,
#     restrict_to_sat_indices: Optional[set[int]] = None,
# ) -> List[Dict[str, Any]]:
#     """
#     Args:
#       chip_scores:   [B_sat, M]
#       chip_metadata: length B_sat; each entry is a list for that satellite only
#     """
#     if chip_scores.ndim != 2:
#         raise ValueError(f"Expected chip_scores with shape [B_sat, M], got {chip_scores.shape}")

#     b_sat = chip_scores.shape[0]

#     if len(sat_paths) != b_sat:
#         raise ValueError(f"sat_paths length mismatch: {len(sat_paths)} vs {b_sat}")
#     if len(chip_counts) != b_sat:
#         raise ValueError(f"chip_counts length mismatch: {len(chip_counts)} vs {b_sat}")
#     if len(chip_metadata) != b_sat:
#         raise ValueError(f"chip_metadata length mismatch: {len(chip_metadata)} vs {b_sat}")

#     per_sat_results: List[Dict[str, Any]] = []

#     for sat_idx, sat_path in enumerate(sat_paths):
#         if restrict_to_sat_indices is not None and sat_idx not in restrict_to_sat_indices:
#             continue

#         count = int(chip_counts[sat_idx])
#         per_sat_chip_scores = chip_scores[sat_idx, :count]
#         per_sat_chip_meta = chip_metadata[sat_idx]

#         if count == 0:
#             per_sat_results.append({
#                 "sat_index": int(sat_idx),
#                 "sat_path": sat_path,
#                 "n_chips": 0,
#                 "top_chips": [],
#             })
#             continue

#         if len(per_sat_chip_meta) < count:
#             raise ValueError(
#                 f"chip_metadata[{sat_idx}] has length {len(per_sat_chip_meta)} but expected at least {count}"
#             )

#         ranked_local = np.argsort(per_sat_chip_scores)[::-1]
#         top_local = ranked_local[:top_k_chips]

#         top_chips = []
#         for local_idx in top_local:
#             meta = per_sat_chip_meta[int(local_idx)]
#             top_chips.append({
#                 "chip_index_local": int(local_idx),
#                 "score": float(per_sat_chip_scores[int(local_idx)]),
#                 "tiled_area_lrtb": meta["tiled_area_lrtb"],
#                 "chip_box_xyxy": meta["chip_box_xyxy"],
#             })

#         per_sat_results.append({
#             "sat_index": int(sat_idx),
#             "sat_path": sat_path,
#             "n_chips": int(count),
#             "top_chips": top_chips,
#         })

#     return per_sat_results

# def build_chips_per_satellite(
#     chip_scores: np.ndarray,
#     sat_paths: List[str],
#     chip_counts: List[int],
#     chip_metadata: List[List[Dict[str, Any]]],
#     restrict_to_sat_indices: Optional[set[int]] = None,
# ) -> List[Dict[str, Any]]:
#     """
#     Convert per-satellite chip inputs into a structured dictionary format.

#     Args:
#       chip_scores:   [B_sat, M]
#       chip_metadata: length B_sat; each entry is a list for that satellite only
#     """
#     if chip_scores.ndim != 2:
#         raise ValueError(f"Expected chip_scores with shape [B_sat, M], got {chip_scores.shape}")

#     b_sat = chip_scores.shape[0]

#     if len(sat_paths) != b_sat:
#         raise ValueError(f"sat_paths length mismatch: {len(sat_paths)} vs {b_sat}")
#     if len(chip_counts) != b_sat:
#         raise ValueError(f"chip_counts length mismatch: {len(chip_counts)} vs {b_sat}")
#     if len(chip_metadata) != b_sat:
#         raise ValueError(f"chip_metadata length mismatch: {len(chip_metadata)} vs {b_sat}")

#     per_sat_results: List[Dict[str, Any]] = []

#     for sat_idx, sat_path in enumerate(sat_paths):
#         if restrict_to_sat_indices is not None and sat_idx not in restrict_to_sat_indices:
#             continue

#         count = int(chip_counts[sat_idx])
#         per_sat_chip_scores = chip_scores[sat_idx, :count]
#         per_sat_chip_meta = chip_metadata[sat_idx]

#         if len(per_sat_chip_meta) < count:
#             raise ValueError(
#                 f"chip_metadata[{sat_idx}] has length {len(per_sat_chip_meta)} but expected at least {count}"
#             )

#         chips = []
#         for local_idx in range(count):
#             meta = per_sat_chip_meta[local_idx]
#             chips.append({
#                 "chip_index_local": int(local_idx),
#                 "score": float(per_sat_chip_scores[local_idx]),
#                 "tiled_area_lrtb": meta["tiled_area_lrtb"],
#                 "chip_box_xyxy": meta["chip_box_xyxy"],
#             })

#         per_sat_results.append({
#             "sat_index": int(sat_idx),
#             "sat_path": sat_path,
#             "n_chips": int(count),
#             "chips": chips,
#         })

#     return per_sat_results

def build_chips_per_satellite(
    chip_scores: np.ndarray,
    pos_xy_pred_allchips: np.ndarray,
    sat_paths: List[str],
    chip_counts: List[int],
    chip_metadata: List[List[Dict[str, Any]]],
    restrict_to_sat_indices: Optional[set[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Convert per-satellite chip inputs into a structured dictionary format.

    Args:
      chip_scores:            [B_sat, M]
      pos_xy_pred_allchips:   [B_sat, M, 8, 2]
      chip_metadata:          length B_sat; each entry is a list for that satellite only
    """
    if chip_scores.ndim != 2:
        raise ValueError(f"Expected chip_scores with shape [B_sat, M], got {chip_scores.shape}")

    if pos_xy_pred_allchips.ndim != 4 or pos_xy_pred_allchips.shape[-2:] != (8, 2):
        raise ValueError(
            f"Expected pos_xy_pred_allchips with shape [B_sat, M, 8, 2], got {pos_xy_pred_allchips.shape}"
        )

    b_sat, M = chip_scores.shape

    if pos_xy_pred_allchips.shape[0] != b_sat or pos_xy_pred_allchips.shape[1] != M:
        raise ValueError(
            f"pos_xy_pred_allchips shape mismatch: {pos_xy_pred_allchips.shape} vs chip_scores {chip_scores.shape}"
        )

    if len(sat_paths) != b_sat:
        raise ValueError(f"sat_paths length mismatch: {len(sat_paths)} vs {b_sat}")
    if len(chip_counts) != b_sat:
        raise ValueError(f"chip_counts length mismatch: {len(chip_counts)} vs {b_sat}")
    if len(chip_metadata) != b_sat:
        raise ValueError(f"chip_metadata length mismatch: {len(chip_metadata)} vs {b_sat}")

    per_sat_results: List[Dict[str, Any]] = []

    for sat_idx, sat_path in enumerate(sat_paths):
        if restrict_to_sat_indices is not None and sat_idx not in restrict_to_sat_indices:
            continue

        count = int(chip_counts[sat_idx])
        per_sat_chip_scores = chip_scores[sat_idx, :count]
        per_sat_chip_meta = chip_metadata[sat_idx]
        per_sat_pos_preds = pos_xy_pred_allchips[sat_idx, :count]  # [count, 8, 2]

        if len(per_sat_chip_meta) < count:
            raise ValueError(
                f"chip_metadata[{sat_idx}] has length {len(per_sat_chip_meta)} but expected at least {count}"
            )

        chips = []
        for local_idx in range(count):
            meta = per_sat_chip_meta[local_idx]

            chips.append({
                "chip_index_local": int(local_idx),
                "score": float(per_sat_chip_scores[local_idx]),
                "tiled_area_lrtb": meta["tiled_area_lrtb"],
                "chip_box_xyxy": meta["chip_box_xyxy"],
                "pos_xy_preds": per_sat_pos_preds[local_idx].tolist(),  # serialized
            })

        per_sat_results.append({
            "sat_index": int(sat_idx),
            "sat_path": sat_path,
            "n_chips": int(count),
            "chips": chips,
        })

    return per_sat_results

def build_result(
    cluster_id: int,
    site_id: str,
    satellite_scores: np.ndarray,
    ground_mask_row: torch.Tensor,
    ground_paths: List[str],
    sat_paths: List[str],
    chip_counts: List[int],
    top_k_satellites: int = 10,
    all_chips_per_satellite: Optional[List[Dict[str, Any]]] = None,
    pos_heatmap: Optional[Any] = None,
    use_in_sample: bool = False,
) -> Dict[str, Any]:
    ranked_indices = np.argsort(satellite_scores)[::-1]
    top_indices = ranked_indices[:top_k_satellites]

    return {
        "cluster_id": int(cluster_id),
        "site_id": site_id,
        "n_ground": int(ground_mask_row.sum().item()),
        "n_sat_candidates": int(len(sat_paths)),
        "n_sat_chips_total": int(sum(chip_counts)),
        "chip_counts_per_sat": [int(x) for x in chip_counts],
        "top_sat_indices": [int(i) for i in top_indices],
        "top_sat_paths": [sat_paths[int(i)] for i in top_indices],
        "top_scores": [float(satellite_scores[i]) for i in top_indices],
        "all_scores": [float(score) for score in satellite_scores],
        "ground_paths": ground_paths,
        "sat_paths": sat_paths,
        "all_chips_per_satellite": all_chips_per_satellite,
        "use_in_sample": bool(use_in_sample),
        # "pos_heatmap": pos_heatmap,
    }

def save_cluster_result(output_dir: Path, result: Dict[str, Any]) -> None:
    cluster_id = result["cluster_id"]
    site_id = result["site_id"]
    cluster_out_dir = output_dir / str(site_id) 
    cluster_out_dir.mkdir(parents=True, exist_ok=True)

    with open(cluster_out_dir / f"cluster_{cluster_id:04d}_results.json", "w") as f:
        json.dump(result, f, indent=2)

def run_cluster_inference(
    model,
    model_type,
    loader,
    output_dir: Path,
    device,
    in_sample_contrastive_fn,
    pos_logits_to_heatmap_fn,
    score_reduction: str = "max",
    top_k_sat_chips: int = 10,
    top_k_satellites: int = 3,
    save_topk_chips_for_top_satellites_only: bool = True,
):
    model.eval()
    results = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(loader):
        infer_out = compute_logits_and_heatmaps(
            model=model,
            model_type=model_type,
            batch=batch,
            device=device,
            in_sample_contrastive_fn=in_sample_contrastive_fn,
            pos_logits_to_heatmap_fn=pos_logits_to_heatmap_fn,
        )

        logits = infer_out["logits"]                  # [B_sat, M]
        # pos_heatmap = infer_out["pos_heatmap"]        # expected aligned with B_sat
        use_in_sample = infer_out["use_in_sample"]
        pos_xy_pred_allchips = infer_out['pos_xy_pred']

        chip_scores_np = logits.detach().cpu().float().numpy()   # [B_sat, M]
        satellite_scores = aggregate_chip_scores_to_satellite_scores(
            chip_scores=chip_scores_np,
            reduction=score_reduction,
        )

        ranked_indices = np.argsort(satellite_scores)[::-1]
        top_indices = ranked_indices[:top_k_satellites]

        restrict_to_sat_indices = (
            set(int(i) for i in top_indices)
            if save_topk_chips_for_top_satellites_only
            else None
        )

        all_chips_per_satellite = build_chips_per_satellite(
            chip_scores=chip_scores_np,
            pos_xy_pred_allchips=pos_xy_pred_allchips,
            sat_paths=batch["sat_paths"],
            chip_counts=batch["chip_counts"],
            chip_metadata=batch["chip_metadata"],
            restrict_to_sat_indices=restrict_to_sat_indices,
        )

        cluster_id = int(batch["cluster_id"])
        site_id = batch["site_id"]

        # ground_mask rows are repeated copies for this cluster, so row 0 is enough
        ground_mask_row = batch["ground_mask"][0]

        # # optional: serialize heatmap to plain python if needed
        # pos_heatmap_serializable = None
        # if pos_heatmap is not None:
        #     if torch.is_tensor(pos_heatmap):
        #         pos_heatmap_serializable = pos_heatmap.detach().cpu().tolist()
        #     else:
        #         pos_heatmap_serializable = pos_heatmap

        result = build_result(
            cluster_id=cluster_id,
            site_id=site_id,
            satellite_scores=satellite_scores,
            ground_mask_row=ground_mask_row,
            ground_paths=batch["ground_paths"],
            sat_paths=batch["sat_paths"],
            chip_counts=batch["chip_counts"],
            top_k_satellites=top_k_satellites,
            all_chips_per_satellite=all_chips_per_satellite,
            # pos_heatmap=pos_heatmap_serializable,
            use_in_sample=use_in_sample,
        )

        # save_cluster_result(output_dir=output_dir, result=result)
        with open(output_dir / f"cluster_{cluster_id:04d}_results.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Processed cluster {i}")
        results.append(result)

    with open(output_dir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Inference completed. Results saved to {output_dir}")

    return results

def infer_tiles(
    infer_cfg,
    global_cfg=None,
) -> None:
    data_cfg = infer_cfg["data"]
    runtime_cfg = infer_cfg["runtime"]
    model_cfg = infer_cfg["model"]
    output_cfg = infer_cfg["output"]

    device = get_device(runtime_cfg.get("device", "cuda"))

    if "base_dir" in output_cfg:
        output_dir = Path(output_cfg["base_dir"]) / model_cfg['model_type']
    elif global_cfg is not None and "output_root" in global_cfg:
        dataset_root = Path(
            str(global_cfg.get("dataset_root"))
        ).expanduser().resolve()
        dataset_name = dataset_root.name
        output_dir = Path(global_cfg["output_root"]) / dataset_name/ "inference"
    else:
        raise ValueError("Inference output root directory not found.")

    model_class, pos_logits_to_heatmap_fn, in_sample_contrastive_fn = get_model_family_fns(
        model_cfg["model_type"]
    )
    checkpoint_path, run_cfg_path = resolve_model_paths(model_cfg)
    model = load_model(
        ckpt_path=checkpoint_path,
        model_class=model_class,
        run_cfg_path=run_cfg_path,
        model_type=model_cfg["model_type"],
        device=device,
    )

    dataset = build_dataset(data_cfg)
    loader = build_dataloader(dataset, runtime_cfg)

    results = run_cluster_inference(
        model=model,
        model_type=model_cfg["model_type"],
        loader=loader,
        output_dir=output_dir,
        device=device,
        in_sample_contrastive_fn=in_sample_contrastive_fn,
        pos_logits_to_heatmap_fn=pos_logits_to_heatmap_fn,
        score_reduction=output_cfg.get("score_reduction", "max"),
        top_k_sat_chips=output_cfg.get("top_k_sat_chips", 10),
        top_k_satellites=output_cfg.get("top_k_satellites", 3),
        save_topk_chips_for_top_satellites_only=output_cfg.get(
            "save_topk_chips_for_top_satellites_only", True
        ),
    )

def main():
    args = parse_args()
    infer_cfg = load_json(Path(args.config))
    infer_tiles(infer_cfg=infer_cfg)

if __name__ == "__main__":
    main()