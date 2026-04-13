"""
FlexGeo style baseline (approximate) for Set CVGL (arXiv:2412.18852)

What this implements
1) DINOv3 backbone for ground and satellite
2) Unordered set fusion via a similarity guided weighting (SFF like)
3) Contrastive loss between fused ground set embedding and satellite embedding

What this does not implement precisely
IAL details are not fully specified in the public paper text to a level that guarantees exact reproduction.
A placeholder hook is included if you want to add per image auxiliary heads.

Paper training note (from the paper HTML)
They train with ConvNeXt Base, AdamW, 80 epochs, batch size 64, 4 query images per set. (SetVL 480K) :contentReference[oaicite:4]{index=4}
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Set
import os
import random
import math
import re
import time
import sys
import json
import shutil
from datetime import datetime, timedelta
from tqdm import tqdm
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.distributed as dist
from PIL import Image, UnidentifiedImageError

try:
    from transformers import AutoConfig, AutoModel  # type: ignore
except Exception:
    AutoConfig = None
    AutoModel = None


from models.helpers import (
    l2n,
    is_dist_initialized,
    unwrap_model,
    pos_xy_to_label,
    Sample,
    recall_at_k,
    _dist_is_on,
    _all_reduce_sum_scalar,
)

def register_grad_layout_hooks(module: nn.Module):
    """
    DDP may warn for tensors with singleton channel dims (e.g., depthwise conv
    weights shaped [C,1,K,K]) when grad strides differ from bucket view strides.
    Normalize those grads to contiguous layout to avoid reducer warnings.
    """
    hooks = []
    for p in module.parameters():
        if not p.requires_grad:
            continue
        if p.dim() != 4 or p.shape[1] != 1:
            continue
        target_stride = p.stride()

        def _hook(grad, target_stride=target_stride):
            if grad is None:
                return grad
            if grad.stride() != target_stride:
                return grad.contiguous()
            return grad

        hooks.append(p.register_hook(_hook))
    return hooks


def cosine_with_warmup_factor(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    min_factor: float,
) -> float:
    total_epochs = max(int(total_epochs), 1)
    warmup_epochs = max(int(warmup_epochs), 0)
    min_factor = float(min(max(min_factor, 0.0), 1.0))
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    cosine_span = max(total_epochs - warmup_epochs, 1)
    if cosine_span <= 1:
        progress = 1.0
    else:
        progress = float(epoch - warmup_epochs) / float(cosine_span - 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_factor + (1.0 - min_factor) * cosine


def aux_ramp_factor(epoch: int, warmup_epochs: int) -> float:
    warmup_epochs = max(int(warmup_epochs), 0)
    if warmup_epochs <= 0:
        return 1.0
    return min(float(epoch + 1) / float(warmup_epochs), 1.0)


# -----------------
# Position settings
# -----------------
POS_LOSS_WEIGHT = 0.2
POS_LOSS_TYPE = "reg"  # "ce" (grid/quadrant classification), "reg" (x,y regression), or "heatmap" (native token-grid supervision)
POS_REG_BETA = 0.1    # SmoothL1 beta for regression (<=0 falls back to L1)
POS_REG_LOSS = "l2_sum"  # "smooth_l1" | "l1" | "l2_mean" | "l2_sum"
POS_HEATMAP_SIGMA = 1.0  # Gaussian sigma in token-cell units for pos_loss_type="heatmap"
POS_HEATMAP_LOSS = "soft_ce"  # "soft_ce" | "kl"
POS_HEATMAP_XY_WEIGHT = 0.25  # auxiliary xy regression weight for pos_loss_type="heatmap"
POS_MODE = "grid"  # "grid" for heatmap, "quadrant" to match paper
POS_GRID = 2       # grid size (POS_GRID x POS_GRID) when POS_MODE="grid"
POS_HEAD_VARIANT = "pairwise_residual"  # "legacy_mlp" | "pairwise_residual" | "sat_token_heatmap"
POS_HEAD_HIDDEN_DIM = 1024
POS_HEAD_DEPTH = 2
SEPARATE_POS_NECK = True



def pos_xy_to_label(pos_xy: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
    """
    pos_xy: [N,2] in normalized coords (x,y) in [-1,1]
    pos_mask: [N] 1 for valid positions, 0 for invalid
    """
    if POS_MODE == "quadrant":
        x = pos_xy[:, 0]
        y = pos_xy[:, 1]
        # 0: x<0,y<0; 1: x>=0,y<0; 2: x<0,y>=0; 3: x>=0,y>=0
        label = (x >= 0).long() + 2 * (y >= 0).long()
        return label

    # grid mode
    grid = int(POS_GRID)
    u = (pos_xy[:, 0] + 1.0) * 0.5
    v = (pos_xy[:, 1] + 1.0) * 0.5
    u = torch.clamp(u, 0.0, 1.0 - 1e-6)
    v = torch.clamp(v, 0.0, 1.0 - 1e-6)
    ix = torch.floor(u * grid).long()
    iy = torch.floor(v * grid).long()
    label = iy * grid + ix
    # ensure invalid positions are assigned a dummy label (will be masked)
    label = label * (pos_mask > 0).long()
    return label


def infer_pos_heatmap_hw(num_bins: int) -> Tuple[int, int]:
    bins = int(num_bins)
    if bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")
    if bins == 4 and POS_MODE == "quadrant":
        return 2, 2
    side = int(round(math.sqrt(bins)))
    if side * side == bins:
        return side, side
    expected = int(POS_GRID)
    if expected > 0 and bins == expected * expected:
        return expected, expected
    raise ValueError(f"Cannot infer heatmap shape from num_bins={bins}")


def pos_logits_to_heatmap(pos_logits: torch.Tensor) -> torch.Tensor:
    """
    Convert position predictions to a normalized heatmap.
    pos_logits: [B, K] or [B, N, K]
      - K=square spatial bins (grid classification or token-heatmap logits)
      - K=2 (regression output interpreted as normalized x,y)
    returns: [B, H, W] where H=W=POS_GRID (grid mode) or 2 (quadrant mode)
    """
    if pos_logits.dim() == 3:
        pos_logits = pos_logits.mean(dim=1)

    if pos_logits.dim() != 2:
        raise ValueError(f"pos_logits must be [B,K] or [B,N,K], got shape={tuple(pos_logits.shape)}")

    # Regression mode compatibility: convert predicted (x,y) into a small
    # Gaussian heatmap so existing visualization code can still run.
    if pos_logits.shape[-1] == 2:
        xy = torch.clamp(pos_logits, -1.0, 1.0)
        H, W = infer_pos_heatmap_hw(4 if POS_MODE == "quadrant" else max(int(POS_GRID), 1) ** 2)
        ys = torch.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, device=xy.device, dtype=xy.dtype)
        xs = torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device=xy.device, dtype=xy.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        sigma = max(1.0 / float(max(H, W)), 1e-3)
        dx2 = (xx.unsqueeze(0) - xy[:, 0].view(-1, 1, 1)) ** 2
        dy2 = (yy.unsqueeze(0) - xy[:, 1].view(-1, 1, 1)) ** 2
        heat = torch.exp(-(dx2 + dy2) / (2.0 * sigma * sigma))
        return heat / (heat.sum(dim=(1, 2), keepdim=True) + 1e-6)

    probs = torch.softmax(pos_logits, dim=-1)
    H, W = infer_pos_heatmap_hw(int(pos_logits.shape[-1]))
    return probs.view(-1, H, W)


def heatmap_to_xy(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Convert heatmap to expected (x,y) in normalized coords [-1,1].
    heatmap: [B, H, W]
    returns: [B, 2] with x,y in [-1,1]
    """
    if heatmap.dim() != 3:
        raise ValueError("heatmap must be [B,H,W]")
    B, H, W = heatmap.shape
    ys = torch.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, device=heatmap.device)
    xs = torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device=heatmap.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    w = heatmap / (heatmap.sum(dim=(1, 2), keepdim=True) + 1e-6)
    x = (w * xx).sum(dim=(1, 2))
    y = (w * yy).sum(dim=(1, 2))
    return torch.stack([x, y], dim=-1)


def heatmap_to_pixel(heatmap: torch.Tensor, chip_size: int) -> torch.Tensor:
    """
    Convert heatmap to expected pixel coordinate within chip.
    returns: [B, 2] (x_px, y_px)
    """
    xy = heatmap_to_xy(heatmap)
    half = chip_size / 2.0
    x_px = (xy[:, 0] + 1.0) * half
    y_px = (xy[:, 1] + 1.0) * half
    return torch.stack([x_px, y_px], dim=-1)


def compute_xy_reg_loss(
    pos_xy_pred: torch.Tensor,
    target_xy: torch.Tensor,
    mode: str = POS_REG_LOSS,
    smooth_l1_beta: float = POS_REG_BETA,
) -> torch.Tensor:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "smooth_l1":
        beta = float(smooth_l1_beta)
        if beta > 0.0:
            return F.smooth_l1_loss(pos_xy_pred, target_xy, beta=beta)
        return F.l1_loss(pos_xy_pred, target_xy)
    if mode_norm == "l1":
        return F.l1_loss(pos_xy_pred, target_xy)

    sq_dist = torch.sum((pos_xy_pred - target_xy) ** 2, dim=-1)
    if mode_norm == "l2_mean":
        return sq_dist.mean()
    if mode_norm == "l2_sum":
        return sq_dist.sum()
    raise ValueError(f"Unsupported pos_reg_loss={mode!r}")


def _normalized_xy_to_token_xy(target_xy: torch.Tensor, heatmap_hw: Tuple[int, int]) -> torch.Tensor:
    if target_xy.shape[-1] != 2:
        raise ValueError(f"target_xy must end with size 2, got shape={tuple(target_xy.shape)}")
    H, W = int(heatmap_hw[0]), int(heatmap_hw[1])
    xy = torch.clamp(target_xy, -1.0, 1.0)
    x = ((xy[..., 0] + 1.0) * 0.5 * float(W)) - 0.5
    y = ((xy[..., 1] + 1.0) * 0.5 * float(H)) - 0.5
    x = torch.clamp(x, -0.5, float(W) - 0.5)
    y = torch.clamp(y, -0.5, float(H) - 0.5)
    return torch.stack([x, y], dim=-1)


def build_gaussian_heatmap_target(
    target_xy: torch.Tensor,
    heatmap_hw: Tuple[int, int],
    sigma: float = POS_HEATMAP_SIGMA,
) -> torch.Tensor:
    if target_xy.shape[-1] != 2:
        raise ValueError(f"target_xy must end with size 2, got shape={tuple(target_xy.shape)}")
    H, W = int(heatmap_hw[0]), int(heatmap_hw[1])
    prefix = target_xy.shape[:-1]
    flat_xy = _normalized_xy_to_token_xy(target_xy.reshape(-1, 2), (H, W))
    if flat_xy.shape[0] == 0:
        return target_xy.new_zeros(*prefix, H, W)

    ys = torch.arange(H, device=flat_xy.device, dtype=flat_xy.dtype)
    xs = torch.arange(W, device=flat_xy.device, dtype=flat_xy.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    yy = yy.unsqueeze(0)
    xx = xx.unsqueeze(0)

    sigma_val = float(sigma)
    if sigma_val <= 0.0:
        ix = flat_xy[:, 0].round().clamp(0, W - 1).long()
        iy = flat_xy[:, 1].round().clamp(0, H - 1).long()
        target = torch.zeros(flat_xy.shape[0], H, W, device=flat_xy.device, dtype=flat_xy.dtype)
        target[torch.arange(flat_xy.shape[0], device=flat_xy.device), iy, ix] = 1.0
        return target.view(*prefix, H, W)

    dx2 = (xx - flat_xy[:, 0].view(-1, 1, 1)) ** 2
    dy2 = (yy - flat_xy[:, 1].view(-1, 1, 1)) ** 2
    target = torch.exp(-(dx2 + dy2) / (2.0 * sigma_val * sigma_val))
    target = target / (target.sum(dim=(1, 2), keepdim=True) + 1e-6)
    return target.view(*prefix, H, W)


def compute_pos_heatmap_loss(
    pos_logits: torch.Tensor,
    target_xy: torch.Tensor,
    valid_mask: torch.Tensor,
    sigma: float = POS_HEATMAP_SIGMA,
    loss_type: str = POS_HEATMAP_LOSS,
    aux_xy_weight: float = POS_HEATMAP_XY_WEIGHT,
    aux_xy_loss: str = POS_REG_LOSS,
    aux_xy_beta: float = POS_REG_BETA,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    valid = valid_mask > 0.5
    if not valid.any():
        return None, None, None

    if pos_logits.dim() != 3:
        raise ValueError(f"pos_logits must be [B,N,K], got shape={tuple(pos_logits.shape)}")
    if target_xy.shape[:2] != pos_logits.shape[:2]:
        raise ValueError(
            f"target_xy batch/query dims must match pos_logits; got {tuple(target_xy.shape)} vs {tuple(pos_logits.shape)}"
        )

    K = int(pos_logits.shape[-1])
    heatmap_hw = infer_pos_heatmap_hw(K)
    flat_logits = pos_logits[valid]
    flat_target_xy = torch.clamp(target_xy[valid], -1.0, 1.0)
    target_heat = build_gaussian_heatmap_target(flat_target_xy, heatmap_hw, sigma=sigma).reshape(-1, K)
    log_probs = F.log_softmax(flat_logits, dim=-1)

    loss_type_norm = str(loss_type).strip().lower()
    if loss_type_norm == "soft_ce":
        heatmap_loss = -(target_heat * log_probs).sum(dim=-1).mean()
    elif loss_type_norm == "kl":
        heatmap_loss = F.kl_div(log_probs, target_heat, reduction="batchmean")
    else:
        raise ValueError(f"Unsupported pos_heatmap_loss={loss_type!r}")

    total_loss = heatmap_loss
    aux_loss = None
    aux_weight = float(aux_xy_weight)
    if aux_weight > 0.0:
        pred_xy = heatmap_to_xy(log_probs.exp().view(-1, heatmap_hw[0], heatmap_hw[1]))
        aux_loss = compute_xy_reg_loss(
            pos_xy_pred=pred_xy,
            target_xy=flat_target_xy,
            mode=aux_xy_loss,
            smooth_l1_beta=aux_xy_beta,
        )
        total_loss = total_loss + (aux_weight * aux_loss)
    return total_loss, heatmap_loss, aux_loss


def compute_pos_reg_loss(
    pos_xy_pred: torch.Tensor,
    target_xy: torch.Tensor,
    valid_mask: torch.Tensor,
    mode: str = POS_REG_LOSS,
    smooth_l1_beta: float = POS_REG_BETA,
) -> Optional[torch.Tensor]:
    """
    Position regression loss on [B,N,2].
    - l2_sum: sum squared distance over images per sample, then mean over batch.
    - l2_mean: mean squared distance over all valid images.
    - smooth_l1 / l1: per-coordinate losses over valid entries.
    """
    valid = valid_mask > 0.5
    if not valid.any():
        return None

    mode_norm = str(mode).strip().lower()
    if mode_norm == "l2_sum":
        sq_dist = torch.sum((pos_xy_pred - target_xy) ** 2, dim=-1)  # [B,N]
        per_sample = (sq_dist * valid.float()).sum(dim=1)  # [B]
        return per_sample.mean()
    return compute_xy_reg_loss(
        pos_xy_pred=pos_xy_pred[valid],
        target_xy=target_xy[valid],
        mode=mode,
        smooth_l1_beta=smooth_l1_beta,
    )


def compute_position_loss(
    core_model: nn.Module,
    out: Dict[str, torch.Tensor],
    pos_xy: Optional[torch.Tensor],
    pos_label: Optional[torch.Tensor],
    pos_mask: Optional[torch.Tensor],
    pos_reg_loss: str = POS_REG_LOSS,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    pos_loss_type = str(getattr(core_model, "pos_loss_type", "ce")).strip().lower()

    if pos_loss_type == "reg":
        pos_xy_pred = out.get("pos_xy_pred")
        if pos_xy_pred is None:
            return None, None
        if pos_xy is None or pos_mask is None:
            return None, pos_xy_pred
        target_xy = torch.clamp(pos_xy, -1.0, 1.0)
        pos_loss = compute_pos_reg_loss(
            pos_xy_pred=pos_xy_pred,
            target_xy=target_xy,
            valid_mask=pos_mask,
            mode=pos_reg_loss,
            smooth_l1_beta=float(getattr(core_model, "pos_reg_beta", POS_REG_BETA)),
        )
        return pos_loss, pos_xy_pred

    pos_logits = out.get("pos_logits")
    if pos_logits is None:
        return None, None

    if pos_loss_type == "heatmap":
        if pos_xy is None or pos_mask is None:
            return None, pos_logits
        target_xy = torch.clamp(pos_xy, -1.0, 1.0)
        pos_loss, _, _ = compute_pos_heatmap_loss(
            pos_logits=pos_logits,
            target_xy=target_xy,
            valid_mask=pos_mask,
            sigma=float(getattr(core_model, "pos_heatmap_sigma", POS_HEATMAP_SIGMA)),
            loss_type=str(getattr(core_model, "pos_heatmap_loss", POS_HEATMAP_LOSS)),
            aux_xy_weight=float(getattr(core_model, "pos_heatmap_xy_weight", POS_HEATMAP_XY_WEIGHT)),
            aux_xy_loss=pos_reg_loss,
            aux_xy_beta=float(getattr(core_model, "pos_reg_beta", POS_REG_BETA)),
        )
        return pos_loss, pos_logits

    if pos_label is None or pos_mask is None:
        return None, pos_logits
    valid = pos_mask > 0.5
    if not valid.any():
        return None, pos_logits
    return F.cross_entropy(pos_logits[valid], pos_label[valid]), pos_logits


class DINOv3Encoder(nn.Module):
    def __init__(
        self,
        model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        out_dim: int = 1024,
        pretrained: bool = True,
    ):
        super().__init__()
        if AutoModel is None:
            raise RuntimeError(
                "transformers is required for DINOv3 backbone. "
                "Install with `pip install transformers`."
            )

        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_id)
        else:
            if AutoConfig is None:
                raise RuntimeError("transformers AutoConfig is unavailable.")
            cfg = AutoConfig.from_pretrained(model_id)
            self.backbone = AutoModel.from_config(cfg)

        feat_dim = int(getattr(self.backbone.config, "hidden_size", 0))
        if feat_dim <= 0:
            raise RuntimeError(
                f"Could not resolve hidden_size from backbone config for model_id={model_id!r}"
            )
        self.num_hidden_layers = int(getattr(self.backbone.config, "num_hidden_layers", 0))
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            hs = getattr(out, "last_hidden_state", None)
            if hs is None or hs.dim() != 3:
                raise RuntimeError("DINOv3 backbone did not return last_hidden_state [B,T,C].")
            pooled = hs.mean(dim=1)
        return pooled

    def project(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(pooled)

    def _extract_patch_grid(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        if last_hidden_state.dim() != 3:
            raise RuntimeError(
                f"Expected last_hidden_state [B,T,C] for patch extraction, got {tuple(last_hidden_state.shape)}"
            )
        B, T, C = last_hidden_state.shape
        candidate_skips = [1, 0, 2, 3, 4]
        max_extra = min(T, 8)
        for skip in candidate_skips + [k for k in range(5, max_extra) if k not in candidate_skips]:
            patch_count = T - int(skip)
            if patch_count <= 0:
                continue
            side = int(round(math.sqrt(patch_count)))
            if side * side == patch_count:
                return last_hidden_state[:, skip:, :].reshape(B, side, side, C)
        raise RuntimeError(f"Could not infer square patch grid from last_hidden_state shape={tuple(last_hidden_state.shape)}")

    def forward_token_grid(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        hs = getattr(out, "last_hidden_state", None)
        if hs is None:
            raise RuntimeError("DINOv3 backbone did not return last_hidden_state needed for token-grid localization.")
        return self._extract_patch_grid(hs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.forward_features(x))


class SimilarityGuidedSetFuser(nn.Module):
    """
    Paper-style SFF weighting.
    Given per-image embeddings e_i, compute pairwise cosine similarities
    and derive inverse-similarity weights (Eq. 2/3).
    """
    def __init__(self, dim: int, scale: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.scale = float(scale)
        self.eps = float(eps)

    def forward(self, e: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        e: [B, N, D]
        mask: [B, N]  1 for valid, 0 for padded
        returns
        fused: [B, D]
        weights: [B, N]
        """
        B, N, D = e.shape
        e_n = l2n(e)

        # Paper-style SFF: pairwise cosine similarity -> normalized adjacency
        # -> inverse-similarity weighting with exponent "scale" (Eq. 2/3).
        pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        sim = torch.matmul(e_n, e_n.transpose(1, 2))
        adj = 0.5 * (sim + 1.0)  # normalize cosine from [-1,1] to [0,1]
        adj = adj * pair_mask

        a_sum = adj.sum(dim=2).clamp_min(self.eps)
        w = torch.pow(a_sum, -self.scale) * mask
        w = w / w.sum(dim=1, keepdim=True).clamp_min(self.eps)

        fused = torch.sum(e * w.unsqueeze(-1), dim=1)
        return fused, w


class PositionResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.norm(x)
        r = F.gelu(self.fc1(r))
        r = self.fc2(r)
        return x + r


class LegacyPositionHead(nn.Sequential):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, apply_tanh: bool):
        layers: List[nn.Module] = [
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        ]
        if apply_tanh:
            layers.append(nn.Tanh())
        super().__init__(*layers)


class PairwiseResidualPositionHead(nn.Module):
    """
    Position head that starts from the same concat(g, s) interface but expands it
    into richer pairwise features inside the module. This keeps inference call
    sites simple while giving the frozen backbone a more expressive decoder.
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        apply_tanh: bool,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.apply_tanh = bool(apply_tanh)
        pair_dim = int(embed_dim) * 4
        self.input_proj = nn.Linear(pair_dim, int(hidden_dim))
        self.blocks = nn.ModuleList(
            [PositionResidualBlock(int(hidden_dim)) for _ in range(max(int(depth), 1))]
        )
        self.out_norm = nn.LayerNorm(int(hidden_dim))
        self.out_proj = nn.Linear(int(hidden_dim), int(out_dim))

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        if pair.shape[-1] != self.embed_dim * 2:
            raise ValueError(
                f"Expected pair features with last dim {self.embed_dim * 2}, got {tuple(pair.shape)}"
            )
        g, s = torch.split(pair, self.embed_dim, dim=-1)
        g_n = l2n(g)
        s_n = l2n(s)
        feat = torch.cat([g_n, s_n, g_n * s_n, torch.abs(g_n - s_n)], dim=-1)
        x = self.input_proj(feat)
        for block in self.blocks:
            x = block(x)
        x = self.out_proj(self.out_norm(x))
        if self.apply_tanh:
            x = torch.tanh(x)
        return x


class SatelliteTokenPositionHead(nn.Module):
    """
    Query-vs-token heatmap head for within-tile localization.
    Ground embeddings act as queries; satellite patch tokens preserve spatial layout.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        inner_dim = int(hidden_dim) if int(hidden_dim) > 0 else int(embed_dim)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.query_proj = nn.Linear(embed_dim, inner_dim)
        self.token_proj = nn.Linear(embed_dim, inner_dim)
        self.query_blocks = nn.ModuleList(
            [PositionResidualBlock(inner_dim) for _ in range(max(int(depth), 1))]
        )
        self.token_blocks = nn.ModuleList(
            [PositionResidualBlock(inner_dim) for _ in range(max(int(depth) - 1, 0))]
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))

    def forward(self, query: torch.Tensor, sat_tokens: torch.Tensor) -> torch.Tensor:
        if query.dim() != 3:
            raise ValueError(f"Expected query [B,N,D], got {tuple(query.shape)}")
        if sat_tokens.dim() != 4:
            raise ValueError(f"Expected sat_tokens [B,H,W,D], got {tuple(sat_tokens.shape)}")
        B, H, W, _ = sat_tokens.shape
        q = self.query_proj(self.query_norm(query))
        for block in self.query_blocks:
            q = block(q)
        t = self.token_proj(self.token_norm(sat_tokens))
        t = t.reshape(B, H * W, -1)
        for block in self.token_blocks:
            t = block(t)
        q = l2n(q)
        t = l2n(t)
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        return torch.einsum("bnd,btd->bnt", q, t).view(B, query.shape[1], H, W) * scale


@dataclass
class BranchEmbeddings:
    retr: torch.Tensor
    pos: torch.Tensor


def infer_pos_head_variant_from_state(state: Dict[str, Any], default: str = POS_HEAD_VARIANT) -> str:
    if not isinstance(state, dict):
        return str(default)
    keys = set(state.keys())
    if any(k.startswith("pos_head.query_proj.") for k in keys):
        return "sat_token_heatmap"
    if any(k.startswith("pos_head.input_proj.") for k in keys):
        return "pairwise_residual"
    if "pos_head.0.weight" in keys or "pos_head.2.weight" in keys:
        return "legacy_mlp"
    return str(default)


def infer_separate_pos_neck_from_state(state: Dict[str, Any], default: bool = SEPARATE_POS_NECK) -> bool:
    if not isinstance(state, dict):
        return bool(default)
    keys = set(state.keys())
    if any(k.startswith("ground_pos_proj.") for k in keys):
        return True
    if any(k.startswith("sat_pos_proj.") for k in keys):
        return True
    return False


def model_kwargs_from_ckpt(ckpt_obj: Any, module: Optional[Any] = None) -> Dict[str, Any]:
    """
    Reconstruct FlexGeoApprox kwargs from checkpoint cfg/state.
    `module` can be a loaded checkpoint-local flex_geo_match module.
    """
    mod = module if module is not None else sys.modules[__name__]
    cfg = {}
    if isinstance(ckpt_obj, dict) and isinstance(ckpt_obj.get("cfg"), dict):
        cfg = dict(ckpt_obj["cfg"])
    state = ckpt_obj.get("model") if isinstance(ckpt_obj, dict) and isinstance(ckpt_obj.get("model"), dict) else ckpt_obj
    state = state if isinstance(state, dict) else {}
    has_attr_head = ("attr_head.weight" in state) or ("attr_head.bias" in state)
    if "enable_ial" in cfg:
        enable_ial = bool(cfg["enable_ial"])
    elif "ial" in cfg:
        enable_ial = bool(cfg["ial"])
    else:
        enable_ial = bool(has_attr_head)

    pos_head_variant_default = getattr(mod, "POS_HEAD_VARIANT", POS_HEAD_VARIANT)
    if isinstance(cfg.get("pos_head_variant"), str) and cfg["pos_head_variant"].strip():
        pos_head_variant = cfg["pos_head_variant"].strip()
    else:
        pos_head_variant = infer_pos_head_variant_from_state(state, default=pos_head_variant_default)
    if "separate_pos_neck" in cfg:
        separate_pos_neck = bool(cfg["separate_pos_neck"])
    else:
        separate_pos_neck = infer_separate_pos_neck_from_state(
            state,
            default=getattr(mod, "SEPARATE_POS_NECK", SEPARATE_POS_NECK),
        )

    candidate: Dict[str, Any] = {
        "embed_dim": 1024,
        "enable_pos": True,
        "pos_mode": str(cfg.get("pos_mode", getattr(mod, "POS_MODE", "grid"))),
        "pos_grid": int(cfg.get("pos_grid", getattr(mod, "POS_GRID", 2))),
        "pos_loss_type": str(cfg.get("pos_loss_type", getattr(mod, "POS_LOSS_TYPE", "ce"))),
        "pretrained": False,
        "enable_ial": enable_ial,
        "pos_head_variant": str(pos_head_variant),
        "pos_head_hidden_dim": int(cfg.get("pos_head_hidden_dim", getattr(mod, "POS_HEAD_HIDDEN_DIM", 1024))),
        "pos_head_depth": int(cfg.get("pos_head_depth", getattr(mod, "POS_HEAD_DEPTH", 2))),
        "separate_pos_neck": bool(separate_pos_neck),
    }
    if enable_ial:
        if "ial_classes" in cfg:
            try:
                candidate["ial_num_classes"] = int(cfg["ial_classes"])
            except Exception:
                pass
        elif "ial_num_classes" in cfg:
            try:
                candidate["ial_num_classes"] = int(cfg["ial_num_classes"])
            except Exception:
                pass
        elif isinstance(state.get("attr_head.weight"), torch.Tensor) and state["attr_head.weight"].ndim == 2:
            candidate["ial_num_classes"] = int(state["attr_head.weight"].shape[0])
    if "pos_reg_beta" in cfg:
        try:
            candidate["pos_reg_beta"] = float(cfg["pos_reg_beta"])
        except Exception:
            pass
    if "pos_heatmap_sigma" in cfg:
        try:
            candidate["pos_heatmap_sigma"] = float(cfg["pos_heatmap_sigma"])
        except Exception:
            pass
    if isinstance(cfg.get("pos_heatmap_loss"), str) and cfg["pos_heatmap_loss"].strip():
        candidate["pos_heatmap_loss"] = cfg["pos_heatmap_loss"].strip()
    if "pos_heatmap_xy_weight" in cfg:
        try:
            candidate["pos_heatmap_xy_weight"] = float(cfg["pos_heatmap_xy_weight"])
        except Exception:
            pass
    if isinstance(cfg.get("backbone_model_id"), str) and cfg["backbone_model_id"].strip():
        candidate["backbone_model_id"] = cfg["backbone_model_id"].strip()
    if "sff_scale" in cfg:
        try:
            candidate["sff_scale"] = float(cfg["sff_scale"])
        except Exception:
            pass
    if "share_backbone" in cfg:
        candidate["share_backbone"] = bool(cfg["share_backbone"])

    sig = inspect.signature(mod.FlexGeoApprox.__init__)
    allowed = {k for k in sig.parameters.keys() if k != "self"}
    return {k: v for k, v in candidate.items() if k in allowed}


class FlexGeoApprox(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        enable_pos: bool = True,
        pos_mode: str = "grid",
        pos_grid: int = 4,
        pos_loss_type: str = POS_LOSS_TYPE,
        pos_reg_beta: float = POS_REG_BETA,
        pos_heatmap_sigma: float = POS_HEATMAP_SIGMA,
        pos_heatmap_loss: str = POS_HEATMAP_LOSS,
        pos_heatmap_xy_weight: float = POS_HEATMAP_XY_WEIGHT,
        pos_head_variant: str = POS_HEAD_VARIANT,
        pos_head_hidden_dim: int = POS_HEAD_HIDDEN_DIM,
        pos_head_depth: int = POS_HEAD_DEPTH,
        separate_pos_neck: bool = SEPARATE_POS_NECK,
        pretrained: bool = True,
        backbone_model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        enable_ial: bool = True,
        ial_num_classes: int = 4,
        sff_scale: float = 2.0,
        share_backbone: bool = True,
    ):
        super().__init__()
        self.ground_enc = DINOv3Encoder(model_id=backbone_model_id, out_dim=embed_dim, pretrained=pretrained)
        self.share_backbone = bool(share_backbone)
        if self.share_backbone:
            self.sat_enc = self.ground_enc
        else:
            self.sat_enc = DINOv3Encoder(model_id=backbone_model_id, out_dim=embed_dim, pretrained=pretrained)
        self.fuser = SimilarityGuidedSetFuser(embed_dim, scale=sff_scale)

        # Keep temperature fixed for DDP stability.
        self.register_buffer("temp", torch.tensor(0.07, dtype=torch.float32))

        # Optional IAL head (use orient_label by default)
        self.enable_ial = enable_ial
        if self.enable_ial:
            self.attr_head = nn.Linear(embed_dim, int(ial_num_classes))

        # Position head (for heatmap-style localization)
        self.enable_pos = enable_pos
        self.pos_mode = pos_mode
        self.pos_grid = pos_grid
        self.pos_loss_type = str(pos_loss_type).strip().lower()
        self.pos_reg_beta = float(pos_reg_beta)
        self.pos_heatmap_sigma = float(pos_heatmap_sigma)
        self.pos_heatmap_loss = str(pos_heatmap_loss).strip().lower()
        self.pos_heatmap_xy_weight = float(pos_heatmap_xy_weight)
        self.pos_head_variant = str(pos_head_variant).strip().lower()
        self.pos_head_hidden_dim = int(pos_head_hidden_dim)
        self.pos_head_depth = max(int(pos_head_depth), 1)
        self.separate_pos_neck = bool(separate_pos_neck)
        if self.pos_loss_type not in {"ce", "reg", "heatmap"}:
            raise ValueError(f"Unsupported pos_loss_type={pos_loss_type!r}; expected one of ['ce','reg','heatmap']")
        if self.pos_heatmap_loss not in {"soft_ce", "kl"}:
            raise ValueError(
                f"Unsupported pos_heatmap_loss={pos_heatmap_loss!r}; expected one of ['soft_ce','kl']"
            )
        if self.pos_head_variant not in {"legacy_mlp", "pairwise_residual", "sat_token_heatmap"}:
            raise ValueError(
                f"Unsupported pos_head_variant={pos_head_variant!r}; expected one of "
                "['legacy_mlp','pairwise_residual','sat_token_heatmap']"
            )
        if self.pos_loss_type == "heatmap" and self.pos_head_variant != "sat_token_heatmap":
            raise ValueError("pos_loss_type='heatmap' requires pos_head_variant='sat_token_heatmap'")
        if self.enable_pos:
            if self.uses_separate_pos_neck:
                self.ground_pos_proj = self._make_branch_proj(self.ground_enc, embed_dim)
                self.sat_pos_proj = self._make_branch_proj(self.sat_enc, embed_dim)
            self.pos_head = self._make_position_head(embed_dim)

    @property
    def uses_separate_pos_neck(self) -> bool:
        return bool(self.enable_pos and self.separate_pos_neck)

    @property
    def uses_token_pos_head(self) -> bool:
        return bool(self.enable_pos and self.pos_head_variant == "sat_token_heatmap")

    def _make_branch_proj(self, encoder: DINOv3Encoder, out_dim: int) -> nn.Sequential:
        in_dim = int(encoder.proj[0].in_features)
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def _make_position_head(self, embed_dim: int) -> nn.Module:
        if self.pos_loss_type == "reg":
            out_dim = 2
            apply_tanh = True
        else:
            out_dim = 4 if self.pos_mode == "quadrant" else int(self.pos_grid) * int(self.pos_grid)
            apply_tanh = False
        if self.pos_head_variant == "sat_token_heatmap":
            return SatelliteTokenPositionHead(
                embed_dim=embed_dim,
                hidden_dim=self.pos_head_hidden_dim,
                depth=self.pos_head_depth,
            )
        if self.pos_head_variant == "legacy_mlp":
            return LegacyPositionHead(
                in_dim=embed_dim * 2,
                hidden_dim=self.pos_head_hidden_dim,
                out_dim=out_dim,
                apply_tanh=apply_tanh,
            )
        return PairwiseResidualPositionHead(
            embed_dim=embed_dim,
            hidden_dim=self.pos_head_hidden_dim,
            out_dim=out_dim,
            depth=self.pos_head_depth,
            apply_tanh=apply_tanh,
        )

    def _ground_retr_from_features(self, feat: torch.Tensor) -> torch.Tensor:
        return self.ground_enc.project(feat)

    def _ground_pos_from_features(self, feat: torch.Tensor) -> torch.Tensor:
        if not self.uses_separate_pos_neck:
            return self._ground_retr_from_features(feat)
        return l2n(self.ground_pos_proj(feat))

    def _sat_retr_from_features(self, feat: torch.Tensor) -> torch.Tensor:
        return l2n(self.sat_enc.project(feat))

    def _sat_pos_from_features(self, feat: torch.Tensor) -> torch.Tensor:
        if not self.uses_separate_pos_neck:
            return self._sat_retr_from_features(feat)
        return l2n(self.sat_pos_proj(feat))

    def _encode_ground_from_features(self, feat: torch.Tensor) -> BranchEmbeddings:
        return BranchEmbeddings(
            retr=self._ground_retr_from_features(feat),
            pos=self._ground_pos_from_features(feat),
        )

    def _encode_sat_from_features(self, feat: torch.Tensor) -> BranchEmbeddings:
        return BranchEmbeddings(
            retr=self._sat_retr_from_features(feat),
            pos=self._sat_pos_from_features(feat),
        )

    def encode_ground_views(self, x: torch.Tensor) -> BranchEmbeddings:
        feat = self.ground_enc.forward_features(x)
        return self._encode_ground_from_features(feat)

    def encode_sat_views(self, x: torch.Tensor) -> BranchEmbeddings:
        feat = self.sat_enc.forward_features(x)
        return self._encode_sat_from_features(feat)

    def encode_ground_retr(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.ground_enc.forward_features(x)
        return self._ground_retr_from_features(feat)

    def encode_sat_retr(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.sat_enc.forward_features(x)
        return self._sat_retr_from_features(feat)

    def encode_ground_pos(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.ground_enc.forward_features(x)
        return self._ground_pos_from_features(feat)

    def encode_sat_pos(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.sat_enc.forward_features(x)
        return self._sat_pos_from_features(feat)

    def encode_sat_pos_token_grid(self, x: torch.Tensor) -> torch.Tensor:
        token_grid = self.sat_enc.forward_token_grid(x)
        if self.uses_separate_pos_neck:
            token_grid = self.sat_pos_proj(token_grid)
        else:
            token_grid = self.sat_enc.project(token_grid)
        return l2n(token_grid)

    def _position_logits_from_token_grid(
        self,
        ground_pos_emb: torch.Tensor,
        sat_pos_token_grid: torch.Tensor,
    ) -> torch.Tensor:
        logits_hw = self.pos_head(ground_pos_emb, sat_pos_token_grid)
        if logits_hw.dim() != 4:
            raise ValueError(
                f"Token position head must return [B,N,H,W], got {tuple(logits_hw.shape)}"
            )
        return logits_hw.flatten(start_dim=2)

    def predict_position_outputs(
        self,
        ground_pos_emb: torch.Tensor,
        sat_pos_emb: Optional[torch.Tensor] = None,
        sat_pos_token_grid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.enable_pos:
            return {}

        out: Dict[str, torch.Tensor]
        if self.uses_token_pos_head:
            if sat_pos_token_grid is None:
                raise ValueError("sat_pos_token_grid is required for pos_head_variant='sat_token_heatmap'")
            pos_logits = self._position_logits_from_token_grid(ground_pos_emb, sat_pos_token_grid)
            out = {"pos_logits": pos_logits}
            if self.pos_loss_type in {"reg", "heatmap"}:
                B, N, K = pos_logits.shape
                heat = pos_logits_to_heatmap(pos_logits.reshape(B * N, K))
                out["pos_xy_pred"] = heatmap_to_xy(heat).view(B, N, 2)
            else:
                H_tok, W_tok = sat_pos_token_grid.shape[1:3]
                target_hw = infer_pos_heatmap_hw(4 if self.pos_mode == "quadrant" else int(self.pos_grid) ** 2)
                if (H_tok, W_tok) != target_hw:
                    B, N, _ = pos_logits.shape
                    logits_hw = pos_logits.view(B * N, 1, H_tok, W_tok)
                    pooled = F.adaptive_avg_pool2d(logits_hw, target_hw)
                    out["pos_logits"] = pooled.view(B, N, target_hw[0] * target_hw[1])
            return out

        if sat_pos_emb is None:
            raise ValueError("sat_pos_emb is required for pooled position heads")
        s_rep = sat_pos_emb.unsqueeze(1).expand_as(ground_pos_emb)
        fcat = torch.cat([ground_pos_emb, s_rep], dim=-1)
        pos_out = self.pos_head(fcat)
        out = {"pos_logits": pos_out}
        if self.pos_loss_type == "reg":
            out["pos_xy_pred"] = pos_out
        return out

    def predict_position_from_satellite(
        self,
        ground_pos_emb: torch.Tensor,
        sat_imgs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.uses_token_pos_head:
            sat_pos_token_grid = self.encode_sat_pos_token_grid(sat_imgs)
            return self.predict_position_outputs(
                ground_pos_emb=ground_pos_emb,
                sat_pos_token_grid=sat_pos_token_grid,
            )
        sat_pos_emb = self.encode_sat_pos(sat_imgs)
        return self.predict_position_outputs(
            ground_pos_emb=ground_pos_emb,
            sat_pos_emb=sat_pos_emb,
        )

    def position_finetune_modules(self) -> List[nn.Module]:
        modules: List[nn.Module] = []
        if not self.enable_pos:
            return modules
        if self.uses_separate_pos_neck:
            modules.extend([self.ground_pos_proj, self.sat_pos_proj])
        modules.append(self.pos_head)
        return modules

    def forward(
        self,
        ground_imgs: torch.Tensor,
        ground_mask: torch.Tensor,
        sat_imgs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        ground_imgs: [B, N, 3, H, W]
        ground_mask: [B, N]
        sat_imgs: [B, 3, H, W] or [B, M, 3, H, W]
        """
        B, N, C, H, W = ground_imgs.shape
        g = ground_imgs.view(B * N, C, H, W)
        ground_views = self.encode_ground_views(g)
        g_emb = ground_views.retr.view(B, N, -1)
        g_pos_emb = ground_views.pos.view(B, N, -1)

        g_set, w = self.fuser(g_emb, ground_mask)
        g_set = l2n(g_set)

        s_emb_all = None
        s_pos_all = None
        if sat_imgs.dim() == 5:
            B2, M, C, H, W = sat_imgs.shape
            if B2 != B:
                raise ValueError(f"sat batch mismatch: ground B={B} sat B={B2}")
            s = sat_imgs.view(B * M, C, H, W)
            sat_views = self.encode_sat_views(s)
            s_emb_all = sat_views.retr.view(B, M, -1)
            s_pos_all = sat_views.pos.view(B, M, -1)
            s_emb = s_emb_all[:, 0]
            s_pos = s_pos_all[:, 0]
        elif sat_imgs.dim() == 4:
            sat_views = self.encode_sat_views(sat_imgs)
            s_emb = sat_views.retr
            s_pos = sat_views.pos
        else:
            raise ValueError("sat_imgs must be [B,3,H,W] or [B,M,3,H,W]")

        out = {"g_set": g_set, "s": s_emb, "w": w, "g_emb": g_emb}
        # Keep temperature in forward graph so DDP does not treat it as "unused".
        out["temp"] = self.temp
        if s_emb_all is not None:
            out["s_all"] = s_emb_all
        if self.enable_pos:
            if self.uses_token_pos_head:
                pos_sat_imgs = sat_imgs[:, 0] if sat_imgs.dim() == 5 else sat_imgs
                pos_token_grid = self.encode_sat_pos_token_grid(pos_sat_imgs)
                out.update(
                    self.predict_position_outputs(
                        ground_pos_emb=g_pos_emb,
                        sat_pos_token_grid=pos_token_grid,
                    )
                )
            else:
                out.update(
                    self.predict_position_outputs(
                        ground_pos_emb=g_pos_emb,
                        sat_pos_emb=s_pos,
                    )
                )
        if self.enable_ial:
            out["attr_logits"] = self.attr_head(g_emb)
        return out


def _dinov3_stage_index(param_name: str) -> int:
    """
    Stage index for Hugging Face DINOv3-like ViT backbones.
    1=embeddings, 2=encoder block 0, 3=encoder block 1, ...
    99=other/non-stage.
    """
    if (
        param_name.startswith("embeddings.")
        or ".embeddings." in param_name
        or param_name.startswith("patch_embed.")
        or ".patch_embed." in param_name
        or ".patch_embeddings." in param_name
    ):
        return 1
    m = re.search(r"(?:^|\.)(?:encoder\.layer|encoder\.layers)\.(\d+)(?:\.|$)", param_name)
    if m:
        return int(m.group(1)) + 2
    m = re.search(r"(?:^|\.)(?:vit\.encoder\.layer|vit\.encoder\.layers)\.(\d+)(?:\.|$)", param_name)
    if m:
        return int(m.group(1)) + 2
    # Some backbones expose transformer blocks as blocks.<idx> / layers.<idx>.
    m = re.search(r"(?:^|\.)(?:blocks|layers)\.(\d+)(?:\.|$)", param_name)
    if m:
        return int(m.group(1)) + 2
    # Fallback: use first numeric path token as block index.
    # This catches backbones whose transformer blocks are named in
    # non-standard ways (e.g., model.<k>..., layer.<k>..., etc.).
    for tok in param_name.split("."):
        if tok.isdigit():
            return int(tok) + 2
    return 99


def set_dinov3_stage_trainable(encoder: DINOv3Encoder, freeze_stages: int, freeze: bool) -> Dict[str, int]:
    configured_max = int(getattr(encoder, "num_hidden_layers", 0)) + 1
    observed_max = 1
    for name, _ in encoder.backbone.named_parameters():
        idx = _dinov3_stage_index(name)
        if idx < 99:
            observed_max = max(observed_max, int(idx))
    max_stages = max(int(configured_max), int(observed_max), 1)
    freeze_stages = max(0, min(int(freeze_stages), int(max_stages)))
    changed = 0
    affected = 0
    for name, p in encoder.backbone.named_parameters():
        if _dinov3_stage_index(name) > freeze_stages:
            continue
        affected += int(p.numel())
        want_grad = not bool(freeze)
        if p.requires_grad != want_grad:
            p.requires_grad = want_grad
            changed += int(p.numel())
    return {
        "affected": affected,
        "changed": changed,
        "resolved_freeze_stages": int(freeze_stages),
        "max_stages": int(max_stages),
    }


def apply_backbone_freeze(core_model: FlexGeoApprox, freeze_stages: int, freeze: bool) -> Dict[str, int]:
    g = set_dinov3_stage_trainable(core_model.ground_enc, freeze_stages=freeze_stages, freeze=freeze)
    if core_model.ground_enc is core_model.sat_enc:
        s = {"affected": 0, "changed": 0}
    else:
        s = set_dinov3_stage_trainable(core_model.sat_enc, freeze_stages=freeze_stages, freeze=freeze)
    return {
        "affected": int(g["affected"] + s["affected"]),
        "changed": int(g["changed"] + s["changed"]),
        "resolved_freeze_stages": int(g.get("resolved_freeze_stages", freeze_stages)),
        "max_stages": int(g.get("max_stages", 1)),
    }


def count_params(module: nn.Module) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for p in module.parameters():
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def set_trainable_subset(module: nn.Module, trainable_modules: List[nn.Module]) -> Dict[str, int]:
    trainable_param_ids = {id(p) for m in trainable_modules for p in m.parameters()}
    seen: Set[int] = set()
    affected = 0
    changed = 0
    enabled = 0
    for p in module.parameters():
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        n = int(p.numel())
        affected += n
        want_grad = pid in trainable_param_ids
        if want_grad:
            enabled += n
        if p.requires_grad != want_grad:
            p.requires_grad = want_grad
            changed += n
    return {
        "affected": affected,
        "changed": changed,
        "enabled": enabled,
    }


def apply_position_only_training(core_model: FlexGeoApprox) -> Dict[str, int]:
    trainable_modules = core_model.position_finetune_modules()
    if not trainable_modules:
        raise ValueError("train_pos_only requested but model has no position modules to optimize.")
    return set_trainable_subset(core_model, trainable_modules)


def clip_style_contrastive(g: torch.Tensor, s: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """
    Symmetric contrastive loss
    """
    t = temperature.clamp(0.01, 0.2)
    logits = torch.matmul(g, s.transpose(0, 1)) / t
    labels = torch.arange(g.shape[0], device=g.device)
    loss_g2s = F.cross_entropy(logits, labels)
    loss_s2g = F.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (loss_g2s + loss_s2g)


def soft_target_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if valid_mask is not None:
        logits = logits.masked_fill(valid_mask <= 0.5, float("-inf"))
        target = target.masked_fill(valid_mask <= 0.5, 0.0)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    log_probs = F.log_softmax(logits, dim=-1)
    safe_log_probs = torch.where(target > 0, log_probs, torch.zeros_like(log_probs))
    return -(target * safe_log_probs).sum(dim=-1).mean()


def in_sample_contrastive(
    g_set: torch.Tensor,
    s_emb: torch.Tensor,
    sat_mask: Optional[torch.Tensor],
    temperature: torch.Tensor,
    retrieval_target: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Contrastive loss within each sample (no in-batch negatives).
    g_set: [B, D]
    s_emb: [B, M, D]
    sat_mask: [B, M] or None
    returns: (loss, logits)
    """
    t = temperature.clamp(0.01, 0.2)
    logits = torch.einsum("bd,bmd->bm", g_set, s_emb) / t
    if retrieval_target is not None:
        loss = soft_target_cross_entropy(logits, retrieval_target, valid_mask=sat_mask)
        if sat_mask is not None:
            logits = logits.masked_fill(sat_mask <= 0.5, float("-inf"))
    else:
        if sat_mask is not None:
            logits = logits.masked_fill(sat_mask <= 0.5, float("-inf"))
        labels = torch.zeros(g_set.shape[0], dtype=torch.long, device=g_set.device)
        loss = F.cross_entropy(logits, labels)
    return loss, logits


def in_sample_single_contrastive(
    g_emb: torch.Tensor,
    s_emb: torch.Tensor,
    ground_mask: Optional[torch.Tensor],
    sat_mask: Optional[torch.Tensor],
    temperature: torch.Tensor,
    retrieval_target: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Per-ground-image InfoNCE within each sample (paper Lsingle style).
    g_emb: [B, N, D]
    s_emb: [B, M, D]
    ground_mask: [B, N] or None
    sat_mask: [B, M] or None
    """
    t = temperature.clamp(0.01, 0.2)
    g_n = l2n(g_emb)
    s_n = l2n(s_emb)
    logits = torch.einsum("bnd,bmd->bnm", g_n, s_n) / t
    sat_mask_expanded = sat_mask[:, None, :] if sat_mask is not None else None
    if retrieval_target is not None:
        target = retrieval_target[:, None, :].expand(-1, g_emb.shape[1], -1)
        if sat_mask_expanded is not None:
            logits = logits.masked_fill(sat_mask_expanded <= 0.5, float("-inf"))
            target = target.masked_fill(sat_mask_expanded <= 0.5, 0.0)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        log_probs = F.log_softmax(logits, dim=-1)
        safe_log_probs = torch.where(target > 0, log_probs, torch.zeros_like(log_probs))
        per_query = -(target * safe_log_probs).sum(dim=-1)
        if ground_mask is not None:
            valid = ground_mask > 0.5
            if valid.any():
                return per_query[valid].mean()
            return logits.sum() * 0.0
        return per_query.mean()
    if sat_mask_expanded is not None:
        logits = logits.masked_fill(sat_mask_expanded <= 0.5, float("-inf"))
    labels = torch.zeros((g_emb.shape[0], g_emb.shape[1]), dtype=torch.long, device=g_emb.device)
    if ground_mask is not None:
        valid = ground_mask > 0.5
        if valid.any():
            return F.cross_entropy(logits[valid], labels[valid])
        return logits.sum() * 0.0
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def batch_single_contrastive(
    g_emb: torch.Tensor,
    s_emb: torch.Tensor,
    ground_mask: Optional[torch.Tensor],
    temperature: torch.Tensor,
) -> torch.Tensor:
    """
    Per-ground-image InfoNCE with in-batch satellite negatives.
    g_emb: [B, N, D]
    s_emb: [B, D]
    ground_mask: [B, N] or None
    """
    t = temperature.clamp(0.01, 0.2)
    g_n = l2n(g_emb)
    s_n = l2n(s_emb)
    logits = torch.einsum("bnd,md->bnm", g_n, s_n) / t
    B, N, _ = logits.shape
    labels = torch.arange(B, device=g_emb.device).unsqueeze(1).expand(B, N)
    if ground_mask is not None:
        valid = ground_mask > 0.5
        if valid.any():
            return F.cross_entropy(logits[valid], labels[valid])
        return logits.sum() * 0.0
    return F.cross_entropy(logits.reshape(B * N, B), labels.reshape(B * N))


@dataclass
class Sample:
    sat_paths: List[str]
    ground_paths: List[str]
    scene_id: str
    ground_pos: Optional[List[Tuple[float, float]]] = None
    ground_orient: Optional[List[Optional[int]]] = None
    # Kept as optional metadata; city classification head/loss is disabled.
    ground_city: Optional[List[Optional[str]]] = None


class SetCvgDataset(Dataset):
    """
    You need to create an index file mapping each satellite image to a list of ground images.

    Expected index format (jsonl suggested)
    {"scene_id":"xxx","sat":"path/to/sat.jpg","ground":["g1.jpg","g2.jpg", ...]}

    This loader will sample N images per set at train time.
    """
    def __init__(
        self,
        items: List[Sample],
        n_query: int = 4,
        n_sat: int = 1,
        train: bool = True,
        image_size: int = 224,
    ):
        super().__init__()
        self.items = items
        self.n_query = n_query
        self.n_sat = n_sat
        self.train = train
        self.image_size = image_size
        self.max_retry = 10

        
        self.Image = Image

        import torchvision.transforms as T
        if train:
            self.t_g = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(0.2, 0.2, 0.2, 0.05),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            self.t_g = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

        self.t_s = self.t_g

    def _load(self, path: str):
        try:
            img = self.Image.open(path).convert("RGB")
            return img
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            return None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tries = 0
        while True:
            it = self.items[idx]
            sat_paths = it.sat_paths if hasattr(it, "sat_paths") else []
            if not sat_paths:
                if not self.train:
                    raise FileNotFoundError(f"Missing satellite paths for scene={it.scene_id}")
                sat_paths = []

            # Positive satellite must be first
            pos_path = sat_paths[0] if sat_paths else None
            if pos_path is not None:
                # Enforce positives/negatives from the same cluster folder
                pos_dir = os.path.dirname(pos_path)
                same_dir = [p for p in sat_paths if os.path.dirname(p) == pos_dir]
                if same_dir:
                    if same_dir[0] != pos_path:
                        same_dir = [pos_path] + [p for p in same_dir if p != pos_path]
                    sat_paths = same_dir
            if pos_path is None:
                if not self.train:
                    raise FileNotFoundError(f"Missing positive satellite for scene={it.scene_id}")
                pos_img = None
            else:
                pos_img = self._load(pos_path)
                if pos_img is None and not self.train:
                    raise FileNotFoundError(f"Missing or invalid satellite image: {pos_path}")

            gs = it.ground_paths
            gp = it.ground_pos if hasattr(it, "ground_pos") else None
            go = it.ground_orient if hasattr(it, "ground_orient") else None
            if self.train:
                chosen_idx = random.sample(range(len(gs)), k=min(self.n_query, len(gs)))
            else:
                chosen_idx = list(range(min(self.n_query, len(gs))))

            imgs = []
            chosen_pos = []
            chosen_orient = []
            for i in chosen_idx:
                p = gs[i]
                img = self._load(p)
                if img is None:
                    continue
                imgs.append(self.t_g(img))
                chosen_pos.append(gp[i] if gp is not None and i < len(gp) else None)
                chosen_orient.append(go[i] if go is not None and i < len(go) else None)

            # Build satellite tensor (M)
            sat_tensor = None
            sat_mask = None
            if pos_img is not None:
                sat_imgs = [self.t_s(pos_img)]
                neg_paths = sat_paths[1:] if len(sat_paths) > 1 else []
                if self.n_sat > 1 and neg_paths:
                    if self.train:
                        neg_k = min(self.n_sat - 1, len(neg_paths))
                        neg_sel = random.sample(neg_paths, k=neg_k)
                    else:
                        neg_sel = neg_paths[: max(0, self.n_sat - 1)]
                    for p in neg_sel:
                        nimg = self._load(p)
                        if nimg is None:
                            continue
                        sat_imgs.append(self.t_s(nimg))
                M = max(self.n_sat, 1)
                C, H, W = sat_imgs[0].shape
                sat_tensor = torch.zeros((M, C, H, W), dtype=torch.float32)
                sat_mask = torch.zeros((M,), dtype=torch.float32)
                for i, x in enumerate(sat_imgs[:M]):
                    sat_tensor[i] = x
                    sat_mask[i] = 1.0

            if pos_img is not None and len(imgs) > 0 and sat_tensor is not None:
                break

            tries += 1
            if not self.train:
                raise FileNotFoundError(
                    f"Missing or invalid images for sample idx={idx} scene={it.scene_id}"
                )
            if tries >= self.max_retry:
                raise RuntimeError(
                    f"Too many invalid images while sampling (idx={idx}, scene={it.scene_id})"
                )
            idx = random.randrange(len(self.items))

        N = self.n_query
        C, H, W = imgs[0].shape
        ground = torch.zeros((N, C, H, W), dtype=torch.float32)
        mask = torch.zeros((N,), dtype=torch.float32)
        # Normalized center-offset target inside positive satellite chip:
        # x,y in [-1,1], where (0,0) is chip center.
        pos_xy = torch.zeros((N, 2), dtype=torch.float32)
        pos_mask = torch.zeros((N,), dtype=torch.float32)
        orient_label = torch.zeros((N,), dtype=torch.long)
        orient_mask = torch.zeros((N,), dtype=torch.float32)

        for i, x in enumerate(imgs[:N]):
            ground[i] = x
            mask[i] = 1.0
            if i < len(chosen_pos) and chosen_pos[i] is not None:
                # Treat metadata as center-offset; clamp to valid normalized range.
                pos_xy[i, 0] = float(max(min(float(chosen_pos[i][0]), 1.0), -1.0))
                pos_xy[i, 1] = float(max(min(float(chosen_pos[i][1]), 1.0), -1.0))
                pos_mask[i] = 1.0
            if i < len(chosen_orient) and chosen_orient[i] is not None:
                orient_label[i] = int(chosen_orient[i])
                orient_mask[i] = 1.0

        pos_label = pos_xy_to_label(pos_xy, pos_mask)
        return {
            "sat": sat_tensor,
            "sat_mask": sat_mask,
            "ground": ground,
            "mask": mask,
            "pos_xy": pos_xy,
            "pos_mask": pos_mask,
            "pos_label": pos_label,
            "orient_label": orient_label,
            "orient_mask": orient_mask,
        }


def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    sat = torch.stack([b["sat"] for b in batch], dim=0)
    sat_mask = torch.stack([b["sat_mask"] for b in batch], dim=0)
    ground = torch.stack([b["ground"] for b in batch], dim=0)
    mask = torch.stack([b["mask"] for b in batch], dim=0)
    pos_xy = torch.stack([b["pos_xy"] for b in batch], dim=0)
    pos_mask = torch.stack([b["pos_mask"] for b in batch], dim=0)
    pos_label = torch.stack([b["pos_label"] for b in batch], dim=0)
    orient_label = torch.stack([b["orient_label"] for b in batch], dim=0)
    orient_mask = torch.stack([b["orient_mask"] for b in batch], dim=0)
    return {
        "sat": sat,
        "sat_mask": sat_mask,
        "ground": ground,
        "mask": mask,
        "pos_xy": pos_xy,
        "pos_mask": pos_mask,
        "pos_label": pos_label,
        "orient_label": orient_label,
        "orient_mask": orient_mask,
    }


def train_one_epoch(
    model,
    loader,
    optim,
    device,
    use_amp: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    pos_weight: float = POS_LOSS_WEIGHT,
    pos_reg_loss: str = POS_REG_LOSS,
    single_weight: float = 1.0,
    ial_weight: float = 0.2,
    accum_steps: int = 1,
    epoch: int = 0,
    log_every: int = 50,
    global_step_base: int = 0,
    log_fn=None,
    main_process: bool = True,
    train_pos_only: bool = False,
):
    model.train()
    core_model = unwrap_model(model)
    total = 0.0
    n = 0
    total_retr = 0.0
    total_pos = 0.0
    total_single = 0.0
    total_ial = 0.0
    retr_batches = 0
    pos_batches = 0
    single_batches = 0
    ial_batches = 0
    pos_valid_sum = 0.0
    pos_valid_total = 0.0
    orient_valid_sum = 0.0
    orient_valid_total = 0.0
    accum_steps = max(int(accum_steps), 1)
    optim.zero_grad(set_to_none=True)
    last_step = -1
    last_grad_norm: Optional[float] = None
    start = time.time()
    seen = 0
    for step, b in enumerate(loader):
        last_step = step
        sat = b["sat"].to(device)
        sat_mask = b.get("sat_mask")
        if sat_mask is not None:
            sat_mask = sat_mask.to(device)
        retrieval_target = b.get("retrieval_target")
        if retrieval_target is not None:
            retrieval_target = retrieval_target.to(device)
        ground = b["ground"].to(device)
        mask = b["mask"].to(device)
        pos_xy = b.get("pos_xy")
        pos_label = b.get("pos_label")
        pos_mask = b.get("pos_mask")
        orient_label = b.get("orient_label")
        orient_mask = b.get("orient_mask")
        if pos_xy is not None:
            pos_xy = pos_xy.to(device)
        if pos_label is not None:
            pos_label = pos_label.to(device)
        if pos_mask is not None:
            pos_mask = pos_mask.to(device)
            pos_valid_sum += float((pos_mask > 0.5).sum().item())
            pos_valid_total += float(pos_mask.numel())
        if orient_label is not None:
            orient_label = orient_label.to(device)
        if orient_mask is not None:
            orient_mask = orient_mask.to(device)
            orient_valid_sum += float((orient_mask > 0.5).sum().item())
            orient_valid_total += float(orient_mask.numel())
        batch_pos_valid = int((pos_mask > 0.5).sum().item()) if pos_mask is not None else 0
        batch_pos_total = int(pos_mask.numel()) if pos_mask is not None else 0
        batch_orient_valid = int((orient_mask > 0.5).sum().item()) if orient_mask is not None else 0
        batch_orient_total = int(orient_mask.numel()) if orient_mask is not None else 0
        batch_sat_valid = int((sat_mask > 0.5).sum().item()) if sat_mask is not None else int(sat.shape[0] * sat.shape[1])
        batch_sat_total = int(sat_mask.numel()) if sat_mask is not None else int(sat.shape[0] * sat.shape[1])

        amp_device = "cuda" if str(device).startswith("cuda") else "cpu"
        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            out = model(ground, mask, sat)
            g_emb = out["g_emb"]
            g_set = out["g_set"]
            temp = out["temp"]

            pos_loss = None
            single_loss = None
            ial_loss = None

            use_in_sample = "s_all" in out and sat.shape[1] > 1
            if use_in_sample and sat_mask is not None:
                if sat_mask.sum(dim=1).max().item() <= 1:
                    use_in_sample = False
            if use_in_sample:
                s_emb_all = out["s_all"]
                loss, _ = in_sample_contrastive(
                    g_set,
                    s_emb_all,
                    sat_mask,
                    temp,
                    retrieval_target=retrieval_target,
                )
            else:
                s_emb = out["s"]
                loss = clip_style_contrastive(g_set, s_emb, temp)
            retr_loss = loss
            if train_pos_only:
                loss = retr_loss * 0.0

            if core_model.enable_ial and not train_pos_only:
                if use_in_sample:
                    single_loss = in_sample_single_contrastive(
                        g_emb=g_emb,
                        s_emb=s_emb_all,
                        ground_mask=mask,
                        sat_mask=sat_mask,
                        temperature=temp,
                        retrieval_target=retrieval_target,
                    )
                else:
                    single_loss = batch_single_contrastive(
                        g_emb=g_emb,
                        s_emb=s_emb,
                        ground_mask=mask,
                        temperature=temp,
                    )
                loss = loss + single_weight * single_loss

            if core_model.enable_pos:
                pos_loss, pos_graph_anchor = compute_position_loss(
                    core_model=core_model,
                    out=out,
                    pos_xy=pos_xy,
                    pos_label=pos_label,
                    pos_mask=pos_mask,
                    pos_reg_loss=pos_reg_loss,
                )
                if pos_loss is not None:
                    loss = loss + pos_weight * pos_loss
                elif pos_graph_anchor is not None:
                    # Keep DDP graph consistent even when a local rank has no valid labels.
                    loss = loss + (0.0 * pos_graph_anchor.sum())

            if core_model.enable_ial and "attr_logits" in out:
                attr_logits = out["attr_logits"]
                if orient_label is not None and orient_mask is not None:
                    valid = orient_mask > 0.5
                    if valid.any():
                        ial_loss = F.cross_entropy(attr_logits[valid], orient_label[valid])
                        loss = loss + ial_weight * ial_loss
                    else:
                        loss = loss + (0.0 * attr_logits.sum())
                else:
                    loss = loss + (0.0 * attr_logits.sum())

        raw_loss = loss
        loss = loss / float(accum_steps)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_steps == 0:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optim)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                last_grad_norm = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
                scaler.step(optim)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                last_grad_norm = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
                optim.step()
            optim.zero_grad(set_to_none=True)

        raw_loss_value = float(raw_loss.detach().item())
        retr_loss_value = float(retr_loss.detach().item())
        total += raw_loss_value
        n += 1
        total_retr += retr_loss_value
        retr_batches += 1
        if pos_loss is not None:
            total_pos += float(pos_loss.detach().item())
            pos_batches += 1
        if single_loss is not None:
            total_single += float(single_loss.detach().item())
            single_batches += 1
        if ial_loss is not None:
            total_ial += float(ial_loss.detach().item())
            ial_batches += 1
        seen += int(ground.shape[0])
        if main_process and step % max(int(log_every), 1) == 0:
            lr = optim.param_groups[0]["lr"]
            elapsed = max(time.time() - start, 1e-6)
            ips = seen / elapsed
            pos_val = float(pos_loss.detach().item()) if pos_loss is not None else None
            single_val = float(single_loss.detach().item()) if single_loss is not None else None
            ial_val = float(ial_loss.detach().item()) if ial_loss is not None else None
            pos_avg = (total_pos / max(pos_batches, 1)) if pos_batches > 0 else None
            single_avg = (total_single / max(single_batches, 1)) if single_batches > 0 else None
            ial_avg = (total_ial / max(ial_batches, 1)) if ial_batches > 0 else None
            msg = (
                f"epoch {epoch} step {step+1}/{len(loader)} "
                f"loss {raw_loss_value:.4f} avg {total / max(n, 1):.4f} "
                f"retr {retr_loss_value:.4f} avg_retr {total_retr / max(retr_batches, 1):.4f} "
                f"lr {lr:.3e} ips {ips:.1f}"
            )
            if pos_val is not None:
                msg += f" pos {pos_val:.4f}"
                if pos_avg is not None:
                    msg += f" avg_pos {pos_avg:.4f}"
            if single_val is not None:
                msg += f" single {single_val:.4f}"
                if single_avg is not None:
                    msg += f" avg_single {single_avg:.4f}"
            if ial_val is not None:
                msg += f" ita {ial_val:.4f}"
                if ial_avg is not None:
                    msg += f" avg_ita {ial_avg:.4f}"
            msg += (
                f" w(pos/single/ita)="
                f"{float(pos_weight):.3f}/{float(single_weight):.3f}/{float(ial_weight):.3f}"
            )
            msg += (
                f" valid(pos/ita/sat)="
                f"{batch_pos_valid}/{batch_pos_total},"
                f"{batch_orient_valid}/{batch_orient_total},"
                f"{batch_sat_valid}/{batch_sat_total}"
            )
            if last_grad_norm is not None:
                msg += f" gnorm {last_grad_norm:.3f}"
            if str(device).startswith("cuda"):
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
                msg += f" mem {mem_gb:.2f}GB"
            print(msg)

            if log_fn is not None:
                log = {
                    "train/loss": raw_loss_value,
                    "train/loss_avg": float(total / max(n, 1)),
                    "train/retr_loss": retr_loss_value,
                    "train/retr_loss_avg": float(total_retr / max(retr_batches, 1)),
                    "train/lr": float(lr),
                    "train/ips": float(ips),
                    "train/pos_valid_batch": float(batch_pos_valid / max(batch_pos_total, 1)),
                    "train/ita_valid_batch": float(batch_orient_valid / max(batch_orient_total, 1)),
                    "train/sat_valid_batch": float(batch_sat_valid / max(batch_sat_total, 1)),
                    "train/pos_weight_active": float(pos_weight),
                    "train/single_weight_active": float(single_weight),
                    "train/ial_weight_active": float(ial_weight),
                }
                if last_grad_norm is not None:
                    log["train/grad_norm"] = float(last_grad_norm)
                if pos_val is not None:
                    log["train/pos_loss"] = pos_val
                if pos_avg is not None:
                    log["train/pos_loss_avg"] = float(pos_avg)
                if single_val is not None:
                    log["train/single_loss"] = single_val
                if single_avg is not None:
                    log["train/single_loss_avg"] = float(single_avg)
                if ial_val is not None:
                    log["train/ita_loss"] = ial_val
                if ial_avg is not None:
                    log["train/ita_loss_avg"] = float(ial_avg)
                log_fn(log, step=global_step_base + step)
    # handle leftover grads when step count isn't divisible by accum_steps
    if last_step >= 0 and (last_step + 1) % accum_steps != 0:
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optim)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            last_grad_norm = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
            scaler.step(optim)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            last_grad_norm = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
            optim.step()
        optim.zero_grad(set_to_none=True)
    if is_dist_initialized():
        stats = torch.tensor(
            [
                total,
                float(n),
                total_retr,
                float(retr_batches),
                total_pos,
                float(pos_batches),
                total_single,
                float(single_batches),
                total_ial,
                float(ial_batches),
                pos_valid_sum,
                pos_valid_total,
                orient_valid_sum,
                orient_valid_total,
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total = float(stats[0].item())
        n = int(stats[1].item())
        total_retr = float(stats[2].item())
        retr_batches = int(stats[3].item())
        total_pos = float(stats[4].item())
        pos_batches = int(stats[5].item())
        total_single = float(stats[6].item())
        single_batches = int(stats[7].item())
        total_ial = float(stats[8].item())
        ial_batches = int(stats[9].item())
        pos_valid_sum = float(stats[10].item())
        pos_valid_total = float(stats[11].item())
        orient_valid_sum = float(stats[12].item())
        orient_valid_total = float(stats[13].item())
    out_stats = {
        "loss": total / max(n, 1),
        "retr_loss": total_retr / max(retr_batches, 1),
        "pos_valid_frac": (pos_valid_sum / max(pos_valid_total, 1.0)),
        "orient_valid_frac": (orient_valid_sum / max(orient_valid_total, 1.0)),
    }
    if pos_batches > 0:
        out_stats["pos_loss"] = total_pos / max(pos_batches, 1)
    if single_batches > 0:
        out_stats["single_loss"] = total_single / max(single_batches, 1)
    if ial_batches > 0:
        out_stats["ita_loss"] = total_ial / max(ial_batches, 1)
    return out_stats


def _code_snapshot_sources() -> List[str]:
    this_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(this_file)
    return [
        this_file,
        os.path.join(src_dir, "visym_cluster_dataloader.py"),
    ]


def _copy_code_sources_to_dir(target_dir: str, sources: List[str], tag: str) -> Dict[str, str]:
    os.makedirs(target_dir, exist_ok=True)
    copied: Dict[str, str] = {}
    for src in sources:
        if not os.path.isfile(src):
            print(f"[WARN] {tag} source missing: {src}")
            continue
        dst = os.path.join(target_dir, os.path.basename(src))
        try:
            shutil.copy2(src, dst)
            copied[os.path.basename(src)] = dst
        except Exception as e:
            print(f"[WARN] Failed to copy {tag.lower()} file {src} -> {dst}: {e}")
    return copied


def save_ckpt(path: str, model, optim, scaler, epoch: int, metrics: Dict[str, float], cfg: Dict[str, Any]):
    ckpt_dir = os.path.dirname(path)
    os.makedirs(ckpt_dir, exist_ok=True)
    # Keep key source files next to checkpoint .pt for portability.
    _copy_code_sources_to_dir(
        target_dir=ckpt_dir,
        sources=_code_snapshot_sources(),
        tag="Checkpoint code copy",
    )
    model_to_save = unwrap_model(model)
    torch.save(
        {
            "epoch": epoch,
            "model": model_to_save.state_dict(),
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "metrics": metrics,
            "cfg": cfg,
        },
        path,
    )


def snapshot_training_code(output_dir: str) -> Dict[str, str]:
    """
    Copy key training scripts into output_dir so checkpoints can be loaded
    later with the exact code version used for training.
    """
    snap_dir = os.path.join(output_dir, "code_snapshot")
    return _copy_code_sources_to_dir(
        target_dir=snap_dir,
        sources=_code_snapshot_sources(),
        tag="Code snapshot",
    )


def load_init_ckpt(path: str, model: nn.Module, strict: bool = True) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model") if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict) else ckpt
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint does not contain a valid state dict: {path}")
    incompatible = model.load_state_dict(state, strict=strict)
    return {
        "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "metrics": ckpt.get("metrics") if isinstance(ckpt, dict) else None,
        "missing_keys": list(getattr(incompatible, "missing_keys", [])),
        "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
    }


def resolve_monitor_metric_name(name: str, train_pos_only: bool) -> str:
    metric = str(name).strip().lower()
    if metric == "auto":
        return "pos_loss" if train_pos_only else "r1"
    return metric


def monitor_metric_value(metrics: Dict[Any, float], metric_name: str) -> float:
    if metric_name == "r1":
        return float(metrics[1])
    if metric_name not in metrics:
        raise KeyError(f"Requested monitor metric {metric_name!r} not present in metrics: {sorted(metrics.keys())}")
    return float(metrics[metric_name])


def monitor_metric_is_better(current: float, best: float, metric_name: str, min_delta: float) -> bool:
    if metric_name == "r1":
        return current > (best + float(min_delta))
    return current < (best - float(min_delta))


def monitor_metric_default_best(metric_name: str) -> float:
    return float("-inf") if metric_name == "r1" else float("inf")

######### MODEL EVALUATION ##############
@torch.no_grad()
def _all_reduce_sum_int(x: int, device: torch.device) -> int:
    if not _dist_is_on():
        return int(x)
    t = torch.tensor([x], device=device, dtype=torch.long)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


@torch.no_grad()
def _all_gather_variable_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Gather a [N, D] tensor from all ranks when N may differ across ranks.
    Returns concatenated [sum_i N_i, D].
    """
    if not _dist_is_on():
        return x

    world_size = dist.get_world_size()
    device = x.device
    assert x.dim() == 2, f"Expected 2D tensor, got shape={tuple(x.shape)}"

    local_n = torch.tensor([x.shape[0]], device=device, dtype=torch.long)
    sizes = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(sizes, local_n)
    sizes = [int(s.item()) for s in sizes]
    max_n = max(sizes)

    if x.shape[0] < max_n:
        pad = torch.zeros(
            (max_n - x.shape[0], x.shape[1]),
            device=device,
            dtype=x.dtype,
        )
        x_padded = torch.cat([x, pad], dim=0)
    else:
        x_padded = x

    gathered = [torch.zeros_like(x_padded) for _ in range(world_size)]
    dist.all_gather(gathered, x_padded)

    chunks = [g[:n] for g, n in zip(gathered, sizes)]
    return torch.cat(chunks, dim=0)

@torch.no_grad()
def eval_model(
    model,
    loader,
    device,
    pos_weight: float = POS_LOSS_WEIGHT,
    pos_reg_loss: str = POS_REG_LOSS,
    single_weight: float = 1.0,
    ial_weight: float = 0.2,
    use_amp: bool = False,
):
    model.eval()
    core_model = unwrap_model(model)
    distributed = _dist_is_on()

    hits = {1: 0, 5: 0, 10: 0}
    total = 0

    G: List[torch.Tensor] = []
    S: List[torch.Tensor] = []

    total_loss = 0.0
    total_retr_loss = 0.0
    total_pos_loss = 0.0
    total_single_loss = 0.0
    total_ial_loss = 0.0

    loss_batches = 0
    pos_batches = 0
    single_batches = 0
    ial_batches = 0

    pos_valid_sum = 0.0
    pos_valid_total = 0.0
    orient_valid_sum = 0.0
    orient_valid_total = 0.0

    is_main = (not distributed) or dist.get_rank() == 0
    pbar = tqdm(loader, desc="Validation", disable=not is_main)

    for b in pbar:
        sat = b["sat"].to(device, non_blocking=True)
        sat_mask = b.get("sat_mask")
        if sat_mask is not None:
            sat_mask = sat_mask.to(device, non_blocking=True)

        retrieval_target = b.get("retrieval_target")
        if retrieval_target is not None:
            retrieval_target = retrieval_target.to(device, non_blocking=True)

        retrieval_pos_mask = b.get("retrieval_pos_mask")
        if retrieval_pos_mask is not None:
            retrieval_pos_mask = retrieval_pos_mask.to(device, non_blocking=True)

        ground = b["ground"].to(device, non_blocking=True)
        mask = b["mask"].to(device, non_blocking=True)

        pos_xy = b.get("pos_xy")
        pos_label = b.get("pos_label")
        pos_mask = b.get("pos_mask")
        orient_label = b.get("orient_label")
        orient_mask = b.get("orient_mask")

        if pos_xy is not None:
            pos_xy = pos_xy.to(device, non_blocking=True)
        if pos_label is not None:
            pos_label = pos_label.to(device, non_blocking=True)
        if pos_mask is not None:
            pos_mask = pos_mask.to(device, non_blocking=True)
            pos_valid_sum += float((pos_mask > 0.5).sum().item())
            pos_valid_total += float(pos_mask.numel())
        if orient_label is not None:
            orient_label = orient_label.to(device, non_blocking=True)
        if orient_mask is not None:
            orient_mask = orient_mask.to(device, non_blocking=True)
            orient_valid_sum += float((orient_mask > 0.5).sum().item())
            orient_valid_total += float(orient_mask.numel())

        amp_device = "cuda" if str(device).startswith("cuda") else "cpu"
        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            out = model(ground, mask, sat)
            g_set = out["g_set"]
            temp = out["temp"]

            use_in_sample = "s_all" in out and sat.shape[1] > 1
            if use_in_sample and sat_mask is not None:
                if sat_mask.sum(dim=1).max().item() <= 1:
                    use_in_sample = False

            if use_in_sample:
                s_emb_all = out["s_all"]
                loss, logits = in_sample_contrastive(
                    g_set,
                    s_emb_all,
                    sat_mask,
                    temp,
                    retrieval_target=retrieval_target,
                )

                rank = torch.argsort(logits, dim=1, descending=True)
                for k in hits:
                    if retrieval_pos_mask is not None:
                        hits[k] += int(
                            (retrieval_pos_mask.gather(1, rank[:, :k]) > 0.5).any(dim=1).sum().item()
                        )
                    else:
                        hits[k] += int((rank[:, :k] == 0).any(dim=1).sum().item())
                total += int(g_set.shape[0])
            else:
                s_emb = out["s"]
                loss = clip_style_contrastive(g_set, s_emb, temp)
                G.append(g_set.detach())
                S.append(s_emb.detach())

            retr_loss = loss

            pos_loss = None
            single_loss = None
            ial_loss = None

            if core_model.enable_ial:
                if use_in_sample:
                    single_loss = in_sample_single_contrastive(
                        g_emb=out["g_emb"],
                        s_emb=s_emb_all,
                        ground_mask=mask,
                        sat_mask=sat_mask,
                        temperature=temp,
                        retrieval_target=retrieval_target,
                    )
                else:
                    single_loss = batch_single_contrastive(
                        g_emb=out["g_emb"],
                        s_emb=s_emb,
                        ground_mask=mask,
                        temperature=temp,
                    )
                loss = loss + single_weight * single_loss

            if core_model.enable_pos:
                pos_loss, pos_graph_anchor = compute_position_loss(
                    core_model=core_model,
                    out=out,
                    pos_xy=pos_xy,
                    pos_label=pos_label,
                    pos_mask=pos_mask,
                    pos_reg_loss=pos_reg_loss,
                )
                if pos_loss is not None:
                    loss = loss + pos_weight * pos_loss
                elif pos_graph_anchor is not None:
                    loss = loss + (0.0 * pos_graph_anchor.sum())

            if (
                orient_label is not None
                and orient_mask is not None
                and core_model.enable_ial
                and "attr_logits" in out
            ):
                attr_logits = out["attr_logits"]
                valid = orient_mask > 0.5
                if valid.any():
                    ial_loss = F.cross_entropy(attr_logits[valid], orient_label[valid])
                    loss = loss + ial_weight * ial_loss

        total_loss += float(loss.item())
        total_retr_loss += float(retr_loss.item())
        loss_batches += 1

        if pos_loss is not None:
            total_pos_loss += float(pos_loss.item())
            pos_batches += 1
        if single_loss is not None:
            total_single_loss += float(single_loss.item())
            single_batches += 1
        if ial_loss is not None:
            total_ial_loss += float(ial_loss.item())
            ial_batches += 1

    # Reduce scalar accumulators across ranks
    if distributed:
        for k in hits:
            hits[k] = _all_reduce_sum_int(hits[k], device)
        total = _all_reduce_sum_int(total, device)

        total_loss = _all_reduce_sum_scalar(total_loss, device)
        total_retr_loss = _all_reduce_sum_scalar(total_retr_loss, device)
        total_pos_loss = _all_reduce_sum_scalar(total_pos_loss, device)
        total_single_loss = _all_reduce_sum_scalar(total_single_loss, device)
        total_ial_loss = _all_reduce_sum_scalar(total_ial_loss, device)

        loss_batches = _all_reduce_sum_int(loss_batches, device)
        pos_batches = _all_reduce_sum_int(pos_batches, device)
        single_batches = _all_reduce_sum_int(single_batches, device)
        ial_batches = _all_reduce_sum_int(ial_batches, device)

        pos_valid_sum = _all_reduce_sum_scalar(pos_valid_sum, device)
        pos_valid_total = _all_reduce_sum_scalar(pos_valid_total, device)
        orient_valid_sum = _all_reduce_sum_scalar(orient_valid_sum, device)
        orient_valid_total = _all_reduce_sum_scalar(orient_valid_total, device)

    metrics: Dict[Any, float]
    if total > 0:
        # in-sample branch: hits accumulated directly
        metrics = {k: hits[k] / max(total, 1) for k in hits}
    elif G and S:
        G_local = torch.cat(G, dim=0)
        S_local = torch.cat(S, dim=0)

        if distributed:
            G_all = _all_gather_variable_2d(G_local)
            S_all = _all_gather_variable_2d(S_local)
        else:
            G_all = G_local
            S_all = S_local

        metrics = recall_at_k(G_all, S_all, ks=(1, 5, 10))
    else:
        metrics = {1: 0.0, 5: 0.0, 10: 0.0}

    metrics["loss"] = total_loss / max(loss_batches, 1)
    metrics["retr_loss"] = total_retr_loss / max(loss_batches, 1)

    if pos_batches > 0:
        metrics["pos_loss"] = total_pos_loss / pos_batches
    if single_batches > 0:
        metrics["single_loss"] = total_single_loss / single_batches
    if ial_batches > 0:
        metrics["ita_loss"] = total_ial_loss / ial_batches

    metrics["pos_valid_frac"] = pos_valid_sum / max(pos_valid_total, 1.0)
    metrics["ita_valid_frac"] = orient_valid_sum / max(orient_valid_total, 1.0)

    return metrics

def filter_items_by_site(items: List[Sample], site_id: Optional[str]) -> List[Sample]:
    if not site_id:
        return items
    prefix = site_id if site_id.startswith("site") else f"site{site_id}"
    prefix = prefix.rstrip("/") + "/"
    return [it for it in items if (it.scene_id or "").startswith(prefix)]


def parse_sat_chip_sizes(text: str, fallback_chip: int) -> List[int]:
    vals: List[int] = []
    src = str(text).replace(";", ",")
    for tok in src.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except Exception:
            continue
        if v > 0:
            vals.append(v)
    if not vals and int(fallback_chip) > 0:
        vals.append(int(fallback_chip))
    vals = sorted(set(vals))
    if not vals:
        raise ValueError(f"Invalid sat chip size config: text={text!r} fallback={fallback_chip}")
    return vals