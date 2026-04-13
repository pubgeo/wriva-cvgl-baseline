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
from typing import List, Tuple, Dict, Any, Optional
import os
import random
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from models.helpers import (
    l2n,
    is_dist_initialized,
    unwrap_model,
    pos_xy_to_label,
    Sample,
    recall_at_k,
    _dist_is_on,
    _all_reduce_sum_scalar
)

try:
    from transformers import AutoConfig, AutoModel  # type: ignore
except Exception:
    AutoConfig = None
    AutoModel = None

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


# Model-specific inference function?
def pos_logits_to_heatmap(
    pos_logits: torch.Tensor,
    pos_mode: str = "grid",
    pos_grid: int = 2
) -> torch.Tensor:
    """
    Convert position predictions to a normalized heatmap.
    pos_logits: [B, K] or [B, N, K]
      - K=grid^2 / 4 (classification logits)
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
        if pos_mode == "quadrant":
            H = W = 2
        else:
            H = W = max(int(pos_grid), 1)
        ys = torch.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, device=xy.device, dtype=xy.dtype)
        xs = torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device=xy.device, dtype=xy.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        sigma = max(1.0 / float(max(H, W)), 1e-3)
        dx2 = (xx.unsqueeze(0) - xy[:, 0].view(-1, 1, 1)) ** 2
        dy2 = (yy.unsqueeze(0) - xy[:, 1].view(-1, 1, 1)) ** 2
        heat = torch.exp(-(dx2 + dy2) / (2.0 * sigma * sigma))
        return heat / (heat.sum(dim=(1, 2), keepdim=True) + 1e-6)

    probs = torch.softmax(pos_logits, dim=-1)
    if pos_mode == "quadrant":
        return probs.view(-1, 2, 2)
    grid = int(pos_grid)
    return probs.view(-1, grid, grid)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            hs = getattr(out, "last_hidden_state", None)
            if hs is None or hs.dim() != 3:
                raise RuntimeError("DINOv3 backbone did not return last_hidden_state [B,T,C].")
            pooled = hs.mean(dim=1)
        z = self.proj(pooled)
        return z


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


class FlexGeoApprox(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        enable_pos: bool = True,
        pos_mode: str = "grid",
        pos_grid: int = 4,
        pos_loss_type: str = "reg", # or "ce"
        pos_reg_beta: float = 0.1,
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
        if self.pos_loss_type not in {"ce", "reg"}:
            raise ValueError(f"Unsupported pos_loss_type={pos_loss_type!r}; expected one of ['ce','reg']")
        if self.enable_pos:
            if self.pos_loss_type == "reg":
                pos_out_dim = 2
                self.pos_head = nn.Sequential(
                    nn.Linear(embed_dim * 2, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, pos_out_dim),
                    nn.Tanh(),
                )
            else:
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
            pos_out = self.pos_head(fcat)
            if self.pos_loss_type == "reg":
                out["pos_xy_pred"] = pos_out
                # Compatibility key used by existing inference/visualization code.
                out["pos_logits"] = pos_out
            else:
                out["pos_logits"] = pos_out
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


def in_sample_single_contrastive(
    g_emb: torch.Tensor,
    s_emb: torch.Tensor,
    ground_mask: Optional[torch.Tensor],
    sat_mask: Optional[torch.Tensor],
    temperature: torch.Tensor,
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
    if sat_mask is not None:
        logits = logits.masked_fill(sat_mask[:, None, :] <= 0.5, float("-inf"))
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
        self.max_retry = max_retry
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
    pos_weight:float=0.1,
    single_weight: float=1.0,
    ial_weight: float=0.2,
    accum_steps: int = 1,
    epoch: int = 0,
    log_every: int = 50,
    global_step_base: int = 0,
    log_fn=None,
    main_process: bool = True,
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
                loss, _ = in_sample_contrastive(g_set, s_emb_all, sat_mask, temp)
                s_pos = s_emb_all[:, 0]
            else:
                s_emb = out["s"]
                loss = clip_style_contrastive(g_set, s_emb, temp)
                s_pos = s_emb
            retr_loss = loss

            if core_model.enable_ial:
                if use_in_sample:
                    single_loss = in_sample_single_contrastive(
                        g_emb=g_emb,
                        s_emb=s_emb_all,
                        ground_mask=mask,
                        sat_mask=sat_mask,
                        temperature=temp,
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
                if getattr(core_model, "pos_loss_type", "ce") == "reg" and "pos_xy_pred" in out:
                    pos_xy_pred = out["pos_xy_pred"]
                    if pos_xy is not None and pos_mask is not None:
                        valid = pos_mask > 0.5
                        if valid.any():
                            target_xy = torch.clamp(pos_xy, -1.0, 1.0)
                            beta = float(getattr(core_model, "pos_reg_beta", 0.1))
                            if beta > 0.0:
                                pos_loss = F.smooth_l1_loss(pos_xy_pred[valid], target_xy[valid], beta=beta)
                            else:
                                pos_loss = F.l1_loss(pos_xy_pred[valid], target_xy[valid])
                            loss = loss + pos_weight * pos_loss
                        else:
                            # Keep DDP graph consistent even when a local rank has no valid labels.
                            loss = loss + (0.0 * pos_xy_pred.sum())
                    else:
                        loss = loss + (0.0 * pos_xy_pred.sum())
                elif "pos_logits" in out:
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

############################
##### MODEL EVALUATION #####
############################
@torch.no_grad()
def _all_gather_cat_first_dim(x: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors with possibly different size in dim 0 across ranks, then
    concatenate along dim 0. Other dims must match.
    """
    if not _dist_is_on():
        return x

    world_size = dist.get_world_size()
    device = x.device

    local_n = torch.tensor([x.shape[0]], device=device, dtype=torch.long)
    sizes = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(sizes, local_n)
    sizes = [int(s.item()) for s in sizes]
    max_n = max(sizes)

    if x.shape[0] != max_n:
        pad_shape = (max_n - x.shape[0],) + x.shape[1:]
        pad = torch.zeros(pad_shape, device=device, dtype=x.dtype)
        x_pad = torch.cat([x, pad], dim=0)
    else:
        x_pad = x

    gathered = [torch.zeros_like(x_pad) for _ in range(world_size)]
    dist.all_gather(gathered, x_pad)

    out = []
    for t, n in zip(gathered, sizes):
        out.append(t[:n])
    return torch.cat(out, dim=0)


@torch.no_grad()
def eval_model(
    model,
    loader,
    device,
    pos_weight: float = 0.1,
    single_weight: float = 1.0,
    ial_weight: float = 0.2,
    use_amp: bool = False,
):
    model.eval()
    core_model = unwrap_model(model)

    hits = {1: 0.0, 5: 0.0, 10: 0.0}
    total = 0.0

    G = []
    S = []

    total_loss = 0.0
    total_retr_loss = 0.0
    total_pos_loss = 0.0
    total_single_loss = 0.0
    total_ial_loss = 0.0

    loss_batches = 0.0
    pos_batches = 0.0
    single_batches = 0.0
    ial_batches = 0.0

    pos_valid_sum = 0.0
    pos_valid_total = 0.0
    orient_valid_sum = 0.0
    orient_valid_total = 0.0

    for b in tqdm(loader, desc="Validation", disable=(_dist_is_on() and dist.get_rank() != 0)):
        sat = b["sat"].to(device, non_blocking=True)
        sat_mask = b.get("sat_mask")
        if sat_mask is not None:
            sat_mask = sat_mask.to(device, non_blocking=True)

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
                loss, logits = in_sample_contrastive(g_set, s_emb_all, sat_mask, temp)

                rank = torch.argsort(logits, dim=1, descending=True)
                for k in hits:
                    hits[k] += float((rank[:, :k] == 0).any(dim=1).sum().item())
                total += float(g_set.shape[0])
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
                if (
                    getattr(core_model, "pos_loss_type", "ce") == "reg"
                    and pos_xy is not None
                    and pos_mask is not None
                    and "pos_xy_pred" in out
                ):
                    pos_xy_pred = out["pos_xy_pred"]
                    valid = pos_mask > 0.5
                    if valid.any():
                        target_xy = torch.clamp(pos_xy, -1.0, 1.0)
                        beta = float(getattr(core_model, "pos_reg_beta", 0.1))
                        if beta > 0.0:
                            pos_loss = F.smooth_l1_loss(
                                pos_xy_pred[valid], target_xy[valid], beta=beta
                            )
                        else:
                            pos_loss = F.l1_loss(pos_xy_pred[valid], target_xy[valid])
                        loss = loss + pos_weight * pos_loss

                elif pos_label is not None and pos_mask is not None and "pos_logits" in out:
                    pos_logits = out["pos_logits"]
                    valid = pos_mask > 0.5
                    if valid.any():
                        pos_loss = F.cross_entropy(pos_logits[valid], pos_label[valid])
                        loss = loss + pos_weight * pos_loss

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
        loss_batches += 1.0

        if pos_loss is not None:
            total_pos_loss += float(pos_loss.item())
            pos_batches += 1.0

        if single_loss is not None:
            total_single_loss += float(single_loss.item())
            single_batches += 1.0

        if ial_loss is not None:
            total_ial_loss += float(ial_loss.item())
            ial_batches += 1.0

    # Aggregate scalar accumulators across ranks
    if _dist_is_on():
        for k in hits:
            hits[k] = _all_reduce_sum_scalar(hits[k], device)
        total = _all_reduce_sum_scalar(total, device)

        total_loss = _all_reduce_sum_scalar(total_loss, device)
        total_retr_loss = _all_reduce_sum_scalar(total_retr_loss, device)
        total_pos_loss = _all_reduce_sum_scalar(total_pos_loss, device)
        total_single_loss = _all_reduce_sum_scalar(total_single_loss, device)
        total_ial_loss = _all_reduce_sum_scalar(total_ial_loss, device)

        loss_batches = _all_reduce_sum_scalar(loss_batches, device)
        pos_batches = _all_reduce_sum_scalar(pos_batches, device)
        single_batches = _all_reduce_sum_scalar(single_batches, device)
        ial_batches = _all_reduce_sum_scalar(ial_batches, device)

        pos_valid_sum = _all_reduce_sum_scalar(pos_valid_sum, device)
        pos_valid_total = _all_reduce_sum_scalar(pos_valid_total, device)
        orient_valid_sum = _all_reduce_sum_scalar(orient_valid_sum, device)
        orient_valid_total = _all_reduce_sum_scalar(orient_valid_total, device)

    metrics: Dict[Any, float]

    if total > 0:
        metrics = {k: hits[k] / max(total, 1.0) for k in hits}
    elif G and S:
        G = torch.cat(G, dim=0)
        S = torch.cat(S, dim=0)

        # Gather embeddings from all ranks before global retrieval evaluation
        G = _all_gather_cat_first_dim(G)
        S = _all_gather_cat_first_dim(S)

        metrics = recall_at_k(G, S, ks=(1, 5, 10))
    else:
        metrics = {1: 0.0, 5: 0.0, 10: 0.0}

    metrics["loss"] = total_loss / max(loss_batches, 1.0)
    metrics["retr_loss"] = total_retr_loss / max(loss_batches, 1.0)

    if pos_batches > 0:
        metrics["pos_loss"] = total_pos_loss / pos_batches
    if single_batches > 0:
        metrics["single_loss"] = total_single_loss / single_batches
    if ial_batches > 0:
        metrics["ita_loss"] = total_ial_loss / ial_batches

    metrics["pos_valid_frac"] = pos_valid_sum / max(pos_valid_total, 1.0)
    metrics["ita_valid_frac"] = orient_valid_sum / max(orient_valid_total, 1.0)

    return metrics