"""
FlexGeo style baseline (approximate) for Set CVGL (arXiv:2412.18852)

What this implements
1) ConvNeXt backbone for ground and satellite
2) Unordered set fusion via a similarity guided weighting (SFF like)
3) Contrastive loss between fused ground set embedding and satellite embedding

What this does not implement precisely
IAL details are not fully specified in the public paper text to a level that guarantees exact reproduction.
A placeholder hook is included if you want to add per image auxiliary heads.

Paper training note (from the paper HTML)
They train with ConvNeXt Base, AdamW, 80 epochs, batch size 64, 4 query images per set. (SetVL 480K) :contentReference[oaicite:4]{index=4}
"""
from typing import List, Tuple, Dict, Any, Optional
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset
import timm
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from models.helpers import (
    l2n,
    is_dist_initialized,
    unwrap_model,
    pos_xy_to_label,
    Sample,
    is_main_process,
    recall_at_k,
    _dist_is_on,
    _all_reduce_sum_scalar,
)

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

# Model-speicifc function for inference?
def pos_logits_to_heatmap(pos_logits: torch.Tensor, pos_mode:str, pos_grid:int) -> torch.Tensor:
    """
    Convert position logits to a normalized heatmap.
    pos_logits: [B, K] or [B, N, K]
    returns: [B, H, W] where H=W=POS_GRID (grid mode) or 2 (quadrant mode)
    """
    if pos_logits.dim() == 3:
        pos_logits = pos_logits.mean(dim=1)
    probs = torch.softmax(pos_logits, dim=-1)
    if pos_mode == "quadrant":
        return probs.view(-1, 2, 2)
    grid = int(pos_grid)
    return probs.view(-1, grid, grid)


class ConvNeXtEncoder(nn.Module):
    def __init__(self, name: str = "convnext_base", out_dim: int = 1024, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        z = self.proj(f)
        return z


class SimilarityGuidedSetFuser(nn.Module):
    """
    SFF like idea
    Given per image embeddings e_i, compute pairwise similarity matrix, then derive per image weights.
    Weighting is content adaptive but order free.

    This is an approximation that behaves well for partial overlap sets.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.post = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

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

        sim = torch.matmul(e_n, e_n.transpose(1, 2))
        sim = sim * mask.unsqueeze(1) * mask.unsqueeze(2)

        denom = (mask.sum(dim=1, keepdim=True).clamp_min(1.0))
        mean_sim = sim.sum(dim=2) / denom

        g = self.gate(mean_sim.unsqueeze(-1)).squeeze(-1)
        g = g.masked_fill(mask == 0, float("-inf"))
        w = torch.softmax(g, dim=1)

        fused = torch.sum(e * w.unsqueeze(-1), dim=1)
        fused = fused + self.post(fused)
        return fused, w


class FlexGeoApprox(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        enable_pos: bool = True,
        pos_mode: str = "grid",
        pos_grid: int = 4,
        pretrained: bool = True,
        enable_ial: bool = True,
        ial_num_classes: int = 4,
    ):
        super().__init__()
        self.ground_enc = ConvNeXtEncoder("convnext_base", out_dim=embed_dim, pretrained=pretrained)
        self.sat_enc = ConvNeXtEncoder("convnext_base", out_dim=embed_dim, pretrained=pretrained)
        self.fuser = SimilarityGuidedSetFuser(embed_dim)

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
        if self.enable_pos:
            if self.pos_mode == "quadrant":
                pos_out_dim = 4
            else:
                pos_out_dim = int(self.pos_grid) * int(self.pos_grid)
            self.pos_head = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, pos_out_dim),
            )

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
        g_emb = self.ground_enc(g).view(B, N, -1)

        g_set, w = self.fuser(g_emb, ground_mask)
        g_set = l2n(g_set)

        s_emb_all = None
        if sat_imgs.dim() == 5:
            B2, M, C, H, W = sat_imgs.shape
            if B2 != B:
                raise ValueError(f"sat batch mismatch: ground B={B} sat B={B2}")
            s = sat_imgs.view(B * M, C, H, W)
            s_emb_all = l2n(self.sat_enc(s)).view(B, M, -1)
            s_emb = s_emb_all[:, 0]
        elif sat_imgs.dim() == 4:
            s_emb = l2n(self.sat_enc(sat_imgs))
        else:
            raise ValueError("sat_imgs must be [B,3,H,W] or [B,M,3,H,W]")

        out = {"g_set": g_set, "s": s_emb, "w": w, "g_emb": g_emb}
        # Keep temperature in forward graph so DDP does not treat it as "unused".
        out["temp"] = self.temp
        if s_emb_all is not None:
            out["s_all"] = s_emb_all
        if self.enable_pos:
            s_rep = s_emb.unsqueeze(1).expand_as(g_emb)
            fcat = torch.cat([g_emb, s_rep], dim=-1)
            out["pos_logits"] = self.pos_head(fcat)
        if self.enable_ial:
            out["attr_logits"] = self.attr_head(g_emb)
        return out


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


def in_sample_contrastive(
    g_set: torch.Tensor,
    s_emb: torch.Tensor,
    sat_mask: Optional[torch.Tensor],
    temperature: torch.Tensor,
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
    if sat_mask is not None:
        logits = logits.masked_fill(sat_mask <= 0.5, float("-inf"))
    labels = torch.zeros(g_set.shape[0], dtype=torch.long, device=g_set.device)
    loss = F.cross_entropy(logits, labels)
    return loss, logits


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
        max_retry: int = 10,
        pos_mode: str = "grid",
        pos_grid: int = 2,
    ):
        super().__init__()
        self.items = items
        self.n_query = n_query
        self.n_sat = n_sat
        self.train = train
        self.image_size = image_size
        self.max_retry = max_retry,
        self.pos_mode = pos_mode
        self.pos_grid = pos_grid

        
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
        pos_xy = torch.zeros((N, 2), dtype=torch.float32)
        pos_mask = torch.zeros((N,), dtype=torch.float32)
        orient_label = torch.zeros((N,), dtype=torch.long)
        orient_mask = torch.zeros((N,), dtype=torch.float32)

        for i, x in enumerate(imgs[:N]):
            ground[i] = x
            mask[i] = 1.0
            if i < len(chosen_pos) and chosen_pos[i] is not None:
                pos_xy[i, 0] = float(chosen_pos[i][0])
                pos_xy[i, 1] = float(chosen_pos[i][1])
                pos_mask[i] = 1.0
            if i < len(chosen_orient) and chosen_orient[i] is not None:
                orient_label[i] = int(chosen_orient[i])
                orient_mask[i] = 1.0

        pos_label = pos_xy_to_label(
            pos_xy=pos_xy, 
            pos_mask=pos_mask,
            pos_mode=self.pos_mode,
            pos_grid=self.pos_grid
        )

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

##########################
##### MODEL TRAINING #####
##########################
def train_one_epoch(
    model,
    loader,
    optim,
    device,
    use_amp: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    pos_weight: float = 0.1,
    ial_weight: float = 0.2,
    accum_steps: int = 1,
    epoch: int = 0,
    log_every: int = 50,
    global_step_base: int = 0,
    log_fn=None,
    main_process: bool = True,
    single_weight=None,  # For compatibility with other models
):
    model.train()
    core_model = unwrap_model(model)

    total = 0.0
    n = 0

    total_retr = 0.0
    retr_batches = 0

    total_pos = 0.0
    pos_batches = 0

    total_ial = 0.0
    ial_batches = 0

    accum_steps = max(int(accum_steps), 1)
    optim.zero_grad(set_to_none=True)
    last_step = -1
    start = time.time()
    seen = 0

    for step, b in enumerate(loader):    
        last_step = step
        sat = b["sat"].to(device)
        sat_mask = b.get("sat_mask")
        if sat_mask is not None:
            sat_mask = sat_mask.to(device)
        ground = b["ground"].to(device)
        mask = b["mask"].to(device)
        pos_label = b.get("pos_label")
        pos_mask = b.get("pos_mask")
        orient_label = b.get("orient_label")
        orient_mask = b.get("orient_mask")

        if pos_label is not None:
            pos_label = pos_label.to(device)
        if pos_mask is not None:
            pos_mask = pos_mask.to(device)
        if orient_label is not None:
            orient_label = orient_label.to(device)
        if orient_mask is not None:
            orient_mask = orient_mask.to(device)

        amp_device = "cuda" if str(device).startswith("cuda") else "cpu"
        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            out = model(ground, mask, sat)
            g_emb = out["g_emb"]
            g_set = out["g_set"]
            temp = out["temp"]

            pos_loss = None
            ial_loss = None

            use_in_sample = "s_all" in out and sat.shape[1] > 1
            if use_in_sample and sat_mask is not None:
                if sat_mask.sum(dim=1).max().item() <= 1:
                    use_in_sample = False

            if use_in_sample:
                s_emb_all = out["s_all"]
                retr_loss, _ = in_sample_contrastive(g_set, s_emb_all, sat_mask, temp)
                s_pos = s_emb_all[:, 0]
            else:
                s_emb = out["s"]
                retr_loss = clip_style_contrastive(g_set, s_emb, temp)
                s_pos = s_emb

            loss = retr_loss

            if core_model.enable_pos and "pos_logits" in out:
                pos_logits = out["pos_logits"]
                if pos_label is not None and pos_mask is not None:
                    valid = pos_mask > 0.5
                    if valid.any():
                        pos_loss = F.cross_entropy(pos_logits[valid], pos_label[valid])
                        loss = loss + pos_weight * pos_loss
                    else:
                        # Keep DDP graph consistent even when a local rank has no valid labels.
                        loss = loss + (0.0 * pos_logits.sum())
                else:
                    loss = loss + (0.0 * pos_logits.sum())

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

        if ial_loss is not None:
            total_ial += float(ial_loss.detach().item())
            ial_batches += 1

        seen += int(ground.shape[0])

        if main_process and step % max(int(log_every), 1) == 0:
            lr = optim.param_groups[0]["lr"]
            elapsed = max(time.time() - start, 1e-6)
            ips = seen / elapsed
            pos_val = float(pos_loss.detach().item()) if pos_loss is not None else None
            ial_val = float(ial_loss.detach().item()) if ial_loss is not None else None

            msg = (
                f"epoch {epoch} step {step+1}/{len(loader)} "
                f"loss {raw_loss_value:.4f} avg {total / max(n, 1):.4f} "
                f"retr {retr_loss_value:.4f} avg_retr {total_retr / max(retr_batches, 1):.4f} "
                f"lr {lr:.3e} ips {ips:.1f}"
            )
            if pos_val is not None:
                msg += f" pos {pos_val:.4f}"
            if ial_val is not None:
                msg += f" ita {ial_val:.4f}"
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
                }
                if pos_val is not None:
                    log["train/pos_loss"] = pos_val
                if ial_val is not None:
                    log["train/ita_loss"] = ial_val
                log_fn(log, step=global_step_base + step)

    # This ensures all ranks report the same number of iterations
    if is_dist_initialized():
        # Each rank reports its step count
        local_steps = torch.tensor([last_step + 1], device=device, dtype=torch.int64)
        all_steps = [torch.zeros_like(local_steps) for _ in range(dist.get_world_size())]
        dist.all_gather(all_steps, local_steps)
        max_steps = max(t.item() for t in all_steps)
        
        # If ranks have different step counts, pad with a no-op barrier step
        # This shouldn't normally happen but provides safety
        if last_step + 1 < max_steps:
            if is_main_process():
                print(f"[WARN] Rank imbalance: local_steps={last_step + 1}, max_steps={max_steps}")

    # handle leftover grads when step count isn't divisible by accum_steps
    if last_step >= 0 and (last_step + 1) % accum_steps != 0:
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                total_ial,
                float(ial_batches),
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
        total_ial = float(stats[6].item())
        ial_batches = int(stats[7].item())

    out_stats = {
        "loss": total / max(n, 1),
        "retr_loss": total_retr / max(retr_batches, 1),
    }
    if pos_batches > 0:
        out_stats["pos_loss"] = total_pos / max(pos_batches, 1)
    if ial_batches > 0:
        out_stats["ita_loss"] = total_ial / max(ial_batches, 1)

    if is_dist_initialized():
        dist.barrier()

    return out_stats


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


############################
##### MODEL EVALUATION #####
############################
def _get_world_size() -> int:
    return dist.get_world_size() if _dist_is_on() else 1


def _get_rank() -> int:
    return dist.get_rank() if _dist_is_on() else 0


def _reduce_sum_tensor(t: torch.Tensor) -> torch.Tensor:
    if _dist_is_on():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def _all_gather_variable_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors with variable size in dim 0 across ranks.
    Returns concatenated tensor on every rank.
    """
    if not _dist_is_on():
        return x

    world_size = dist.get_world_size()
    device = x.device

    local_n = torch.tensor([x.shape[0]], device=device, dtype=torch.long)
    sizes = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(sizes, local_n)
    sizes = torch.cat(sizes, dim=0)
    max_n = int(sizes.max().item())

    if x.shape[0] < max_n:
        pad_shape = (max_n - x.shape[0],) + x.shape[1:]
        pad = torch.zeros(pad_shape, device=device, dtype=x.dtype)
        x_pad = torch.cat([x, pad], dim=0)
    else:
        x_pad = x

    gathered = [torch.zeros_like(x_pad) for _ in range(world_size)]
    dist.all_gather(gathered, x_pad)

    chunks = []
    for g, n in zip(gathered, sizes.tolist()):
        if n > 0:
            chunks.append(g[:n])
    if not chunks:
        return x[:0]
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def eval_model(
    model,
    loader,
    device,
    pos_weight: float = 0.1,
    ial_weight: float = 0.2,
    use_amp: bool = False,
    single_weight: float = None,  # compatibility
):
    model.eval()
    core_model = unwrap_model(model)

    # local accumulators
    local_hits = {1: 0.0, 5: 0.0, 10: 0.0}
    local_total = 0.0

    G_local = []
    S_local = []

    total_loss_sum = 0.0
    total_retr_loss_sum = 0.0
    total_loss_count = 0.0

    total_pos_loss_sum = 0.0
    total_pos_count = 0.0

    total_ial_loss_sum = 0.0
    total_ial_count = 0.0

    amp_device = "cuda" if str(device).startswith("cuda") else "cpu"
    rank = _get_rank()

    pbar = tqdm(loader, desc="Validation", disable=(_dist_is_on() and rank != 0))

    for b in pbar:
        sat = b["sat"].to(device, non_blocking=True)
        sat_mask = b.get("sat_mask")
        if sat_mask is not None:
            sat_mask = sat_mask.to(device, non_blocking=True)

        ground = b["ground"].to(device, non_blocking=True)
        mask = b["mask"].to(device, non_blocking=True)

        pos_label = b.get("pos_label")
        pos_mask = b.get("pos_mask")
        orient_label = b.get("orient_label")
        orient_mask = b.get("orient_mask")

        if pos_label is not None:
            pos_label = pos_label.to(device, non_blocking=True)
        if pos_mask is not None:
            pos_mask = pos_mask.to(device, non_blocking=True)
        if orient_label is not None:
            orient_label = orient_label.to(device, non_blocking=True)
        if orient_mask is not None:
            orient_mask = orient_mask.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            out = model(ground, mask, sat)
            g_set = out["g_set"]
            temp = out["temp"]

            use_in_sample = "s_all" in out and sat.shape[1] > 1
            if use_in_sample and sat_mask is not None and sat_mask.sum(dim=1).max().item() <= 1:
                use_in_sample = False

            retr_loss_for_metrics = None

            if use_in_sample:
                s_all = out["s_all"]
                retr_loss, logits = in_sample_contrastive(g_set, s_all, sat_mask, temp)
                retr_loss_for_metrics = retr_loss

                topk = torch.topk(logits, k=max(local_hits), dim=1).indices
                for k in local_hits:
                    local_hits[k] += float((topk[:, :k] == 0).any(dim=1).sum().item())
                local_total += float(g_set.shape[0])

            else:
                s = out["s"]
                retr_loss = clip_style_contrastive(g_set, s, temp)
                retr_loss_for_metrics = retr_loss

                # store features on GPU until end so we can all_gather them
                G_local.append(g_set.detach().float())
                S_local.append(s.detach().float())

            loss = retr_loss

            if core_model.enable_pos and "pos_logits" in out and pos_label is not None and pos_mask is not None:
                valid = pos_mask > 0.5
                if valid.any():
                    pos_loss = F.cross_entropy(out["pos_logits"][valid], pos_label[valid])
                    loss = loss + pos_weight * pos_loss
                    total_pos_loss_sum += float(pos_loss.item())
                    total_pos_count += 1.0

            if core_model.enable_ial and "attr_logits" in out and orient_label is not None and orient_mask is not None:
                valid = orient_mask > 0.5
                if valid.any():
                    ial_loss = F.cross_entropy(out["attr_logits"][valid], orient_label[valid])
                    loss = loss + ial_weight * ial_loss
                    total_ial_loss_sum += float(ial_loss.item())
                    total_ial_count += 1.0

        total_loss_sum += float(loss.detach().item())
        total_retr_loss_sum += float(retr_loss_for_metrics.detach().item())
        total_loss_count += 1.0

    # ------------------------------------------------------------
    # Reduce scalar metrics across ranks
    # ------------------------------------------------------------
    device_for_reduce = device if isinstance(device, torch.device) else torch.device(device)

    total_loss_sum = _all_reduce_sum_scalar(total_loss_sum, device_for_reduce)
    total_retr_loss_sum = _all_reduce_sum_scalar(total_retr_loss_sum, device_for_reduce)
    total_loss_count = _all_reduce_sum_scalar(total_loss_count, device_for_reduce)

    total_pos_loss_sum = _all_reduce_sum_scalar(total_pos_loss_sum, device_for_reduce)
    total_pos_count = _all_reduce_sum_scalar(total_pos_count, device_for_reduce)

    total_ial_loss_sum = _all_reduce_sum_scalar(total_ial_loss_sum, device_for_reduce)
    total_ial_count = _all_reduce_sum_scalar(total_ial_count, device_for_reduce)

    # ------------------------------------------------------------
    # Compute retrieval metrics
    # ------------------------------------------------------------
    if local_total > 0:
        # in-sample retrieval case: just reduce hits/total
        hit_tensor = torch.tensor(
            [local_hits[1], local_hits[5], local_hits[10], local_total],
            device=device_for_reduce,
            dtype=torch.float64,
        )
        hit_tensor = _reduce_sum_tensor(hit_tensor)

        global_h1, global_h5, global_h10, global_total = hit_tensor.tolist()
        denom = max(global_total, 1.0)
        metrics = {
            1: global_h1 / denom,
            5: global_h5 / denom,
            10: global_h10 / denom,
        }

    elif G_local and S_local:
        # clip-style global retrieval: gather feature banks across ranks
        G_local = torch.cat(G_local, dim=0)
        S_local = torch.cat(S_local, dim=0)

        G_all = _all_gather_variable_tensor(G_local)
        S_all = _all_gather_variable_tensor(S_local)

        metrics = recall_at_k(G_all, S_all, ks=(1, 5, 10))

    else:
        metrics = {1: 0.0, 5: 0.0, 10: 0.0}

    metrics["loss"] = total_loss_sum / max(total_loss_count, 1.0)
    metrics["retr_loss"] = total_retr_loss_sum / max(total_loss_count, 1.0)

    if total_pos_count > 0:
        metrics["pos_loss"] = total_pos_loss_sum / total_pos_count
    if total_ial_count > 0:
        metrics["ita_loss"] = total_ial_loss_sum / total_ial_count

    return metrics