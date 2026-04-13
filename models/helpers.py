import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from PIL import Image, UnidentifiedImageError

import os
import math
import shutil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class Sample:
    sat_paths: List[str]
    ground_paths: List[str]
    scene_id: str
    ground_pos: Optional[List[Tuple[float, float]]] = None
    ground_orient: Optional[List[Optional[int]]] = None
    # Kept as optional metadata; city classification head/loss is disabled.
    ground_city: Optional[List[Optional[str]]] = None


def l2n(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if is_dist_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    return get_rank() == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model

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

def pos_xy_to_label(
    pos_xy: torch.Tensor, 
    pos_mask: torch.Tensor,
    pos_mode: str,
    pos_grid: int
) -> torch.Tensor:
    """
    pos_xy: [N,2] in normalized coords (x,y) in [-1,1]
    pos_mask: [N] 1 for valid positions, 0 for invalid
    """
    if pos_mode == "quadrant":
        x = pos_xy[:, 0]
        y = pos_xy[:, 1]
        # 0: x<0,y<0; 1: x>=0,y<0; 2: x<0,y>=0; 3: x>=0,y>=0
        label = (x >= 0).long() + 2 * (y >= 0).long()
        return label

    # grid mode
    grid = int(pos_grid)
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

def save_ckpt(path: str, model, optim, scaler, epoch: int, metrics: Dict[str, float], cfg: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    os.makedirs(snap_dir, exist_ok=True)

    this_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(this_file)
    candidates = [
        this_file,
        os.path.join(src_dir, "visym_cluster_dataloader.py"),
    ]

    copied: Dict[str, str] = {}
    for src in candidates:
        if not os.path.isfile(src):
            print(f"[WARN] Code snapshot source missing: {src}")
            continue
        dst = os.path.join(snap_dir, os.path.basename(src))
        try:
            shutil.copy2(src, dst)
            copied[os.path.basename(src)] = dst
        except Exception as e:
            print(f"[WARN] Failed to snapshot code file {src} -> {dst}: {e}")
    return copied



def load_items_from_jsonl(jsonl_path: str, root: str = "") -> List[Sample]:
    import json
    if not root:
        root = os.path.dirname(os.path.abspath(jsonl_path))
    else:
        root = os.path.abspath(root)
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            sat_field = d["sat"]
            if isinstance(sat_field, list):
                sat_list = [os.path.join(root, p) for p in sat_field]
            else:
                sat_list = [os.path.join(root, sat_field)]
            g = [os.path.join(root, p) for p in d["ground"]]
            gp = d.get("ground_pos")
            go = d.get("ground_orient")
            gc = d.get("ground_city")
            items.append(Sample(
                sat_paths=sat_list,
                ground_paths=g,
                scene_id=d.get("scene_id", ""),
                ground_pos=gp,
                ground_orient=go,
                ground_city=gc,
            ))
    return items


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

@torch.no_grad()
def recall_at_k(g: torch.Tensor, s: torch.Tensor, ks=(1, 5, 10)) -> Dict[int, float]:
    sim = torch.matmul(g, s.transpose(0, 1))
    gt = torch.arange(g.shape[0], device=g.device)
    rank = torch.argsort(sim, dim=1, descending=True)
    out = {}
    for k in ks:
        hit = (rank[:, :k] == gt.unsqueeze(1)).any(dim=1).float().mean().item()
        out[k] = hit
    return out

def _dist_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()

@torch.no_grad()
def _all_reduce_sum_scalar(x: float, device: torch.device) -> float:
    if not _dist_is_on():
        return float(x)
    t = torch.tensor([x], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())
