import argparse
import json
import os
import sys
import inspect
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from PIL import Image, UnidentifiedImageError

from models.visym_cluster_dataloader import VisymClusterDataset, collate_visym_cluster
from models.image_pair_dataloader import ImagePairDataset, collate_image_pair
from models.helpers import (
    is_main_process,
    unwrap_model,
    register_grad_layout_hooks,
    cosine_with_warmup_factor,
    aux_ramp_factor,
    save_ckpt,
    load_init_ckpt,
    snapshot_training_code,
    parse_sat_chip_sizes,
    resolve_monitor_metric_name,
    monitor_metric_value,
    monitor_metric_is_better,
    monitor_metric_default_best, 
    is_dist_initialized,
)

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

MODEL_TYPE = "flex_geo_dinov3" 
DATA_MODE = "manifest"
# -----------------
# Position settings
# -----------------
POS_LOSS_WEIGHT = 0.2
POS_LOSS_TYPE = "reg"  # "ce" (grid/quadrant classification) or "reg" (x,y regression)
POS_REG_BETA = 0.1    # SmoothL1 beta for regression (<=0 falls back to L1)
POS_MODE = "grid"  # "grid" for heatmap, "quadrant" to match paper
POS_GRID = 2       # grid size (POS_GRID x POS_GRID) when POS_MODE="grid"
POS_REG_LOSS = "l2_sum"  # "smooth_l1" | "l1" | "l2_mean" | "l2_sum"
POS_HEAD_VARIANT = "pairwise_residual"  # "legacy_mlp" | "pairwise_residual" | "sat_token_heatmap"
POS_HEAD_HIDDEN_DIM = 1024
POS_HEAD_DEPTH = 2
SEPARATE_POS_NECK = True


# -----------------
# IAL (attribute) settings
# -----------------
IAL_LOSS_WEIGHT = 0.2
SINGLE_LOSS_WEIGHT = 1.0  # paper Lsingle term inside IAL
IAL_NUM_CLASSES = 4  # match orient_label bins from dataset (default 4)

# -----------------
# Hard-coded debug config
# -----------------
TRAIN_MANIFEST = "/home/yejz1/wriva/CVGL/src/splits/splits/visym/train_manifest.pkl"
VAL_MANIFEST = "/home/yejz1/wriva/CVGL/src/splits/splits/visym/val_manifest.pkl"
VISYM_DATASET_ROOT = "/home/apluser/wriva/data/visym-cvgl"
PAIR_SPLIT_ROOT = "/home/apluser/CVGL/train_cfg"
TRAIN_PAIRS = os.path.join(PAIR_SPLIT_ROOT, "train_pairs_site_disjoint.txt")
VAL_PAIRS = os.path.join(PAIR_SPLIT_ROOT, "val_pairs_site_disjoint.txt")
SITE_ID = None  # e.g. "STR0001" or "siteSTR0001"; None uses all sites
CACHE_ROOT = os.path.join(VISYM_DATASET_ROOT, ".cvgl_loader_cache")
OUTPUT_DIR = "/home/apluser/CVGL_train_output/260305_CVGL_model_ssf_pos"

EPOCHS = 120
BATCH_SIZE = 20
IMAGE_SIZE = 224
# Keep direct resize warp for ground images during training.
# (No aspect-preserving crop/pad in train pipeline.)
GROUND_RESIZE_MODE = "direct_resize_square"
LR = 1e-4
WD = 0.05
PRETRAINED = True
BACKBONE_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
RETRIEVAL_TARGET_MODE = "gaussian"
RETRIEVAL_TARGET_SIGMA_SCALE = 0.25
RETRIEVAL_TARGET_MIN_WEIGHT = 1e-3
ENABLE_IAL = True  # ITA/IAL loss (orientation)
SINGLE_WEIGHT = SINGLE_LOSS_WEIGHT
IAL_WEIGHT = IAL_LOSS_WEIGHT
IAL_CLASSES = IAL_NUM_CLASSES

SAVE_EVERY_EPOCHS = 0
LOG_EVERY = 20

WANDB_ENABLE = True
WANDB_PROJECT = "cvgl_direct_match"
WANDB_RUN_NAME = None  # set string to override
WANDB_DIR = OUTPUT_DIR

# -----------------
# Training runtime settings (hard-coded)
# -----------------
USE_AMP = True
GRAD_ACCUM_STEPS = 4  # effective batch = batch_size * GRAD_ACCUM_STEPS
LR_WARMUP_EPOCHS = 5
MIN_LR = 1e-6
AUX_WARMUP_EPOCHS = 10
VAL_BATCH_SIZE = 0  # <=0 uses train batch size
FREEZE_BACKBONE_STAGES = 0  # 0=none, 1=embeddings, 2=embeddings+block0, ...
FREEZE_BACKBONE_EPOCHS = 0  # <=0 keeps selected stages frozen for full training
EARLY_STOP_PATIENCE = 0  # <=0 disables early stopping
EARLY_STOP_MIN_DELTA = 0.0
DDP_TIMEOUT_MINUTES = 120
DDP_FIND_UNUSED_PARAMETERS = True

# -----------------
# Hard-coded set sizes (edit here)
# -----------------
N_QUERY = 1
N_SAT = 16
SAT_CHIP_SIZE = 120
SAT_CHIP_SIZES = (120, 140, 160)  # multi-scale chip sampling range
POS_CENTER_JITTER_PX = 12.0
SAT_IMAGE_SIZE = 160  # keep fixed tensor size when training with multi-scale chips
NEGATIVE_MIN_DISTANCE_PX = 150.0
NEGATIVE_LOCAL_WINDOW_PX = 800.0
RETRIEVAL_ONLY_WARMUP_EPOCHS = 8  # train retrieval-only first, then ramp auxiliary losses
SFF_SCALE = 2.0  # paper Eq.(2)
LOADER_NUM_WORKERS = 8
CACHE_ENABLED = True
KEEP_SAT_OPEN = False
LOADER_PERSISTENT_WORKERS = False
LOADER_PREFETCH_FACTOR = 2
MP_SHARING_STRATEGY = "file_system"  # mitigates "Too many open files" from shared-memory FDs
LOADER_PROGRESS = True
LOADER_PROGRESS_EVERY = 1000

class TeeStream:
    """Mirror stdout to an on-disk log file."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str):
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        return False
    
def build_dataset(args, train: bool, sat_chip_sizes, site_filter=None):
    if args.data_mode == "manifest":
        manifest_path = args.train_manifest if train else args.val_manifest
        return ImagePairDataset(
            manifest_path=manifest_path,
            n_ground=args.n_query,
            n_sat=args.n_sat,
            sat_chip_size=args.sat_chip_size,
            sat_chip_sizes=sat_chip_sizes,
            positive_center_jitter_px=args.pos_center_jitter_px if train else 0.0,
            image_size=args.image_size,
            sat_image_size=args.sat_image_size,
            train=train,
            normalize=True,
            negative_min_distance_px=args.negative_min_distance_px,
            negative_local_window_px=args.negative_local_window_px,
            max_negative_tries=400,
            max_retry=10,
            channel_last=False,
            pos_grid=args.pos_grid,
            keep_sat_open=args.keep_sat_open,
        )

    elif args.data_mode == "cluster":
        pairs_txt = args.train_pairs if train else args.val_pairs
        return VisymClusterDataset(
            pairs_txt=pairs_txt,
            dataset_root=args.dataset_root,
            n_ground=args.n_query,
            n_sat=args.n_sat,
            sat_chip_size=args.sat_chip_size,
            image_size=args.image_size,
            train=train,
            normalize=True,
            site_filter=site_filter,
            cache_enabled=args.cache,
            cache_root=args.cache_root,
            keep_sat_open=args.keep_sat_open,
            show_progress=args.loader_progress,
            progress_every=args.loader_progress_every,
        )

    else:
        raise ValueError(f"Unsupported data_mode: {args.data_mode}")
    
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model_type", 
        type=str, 
        default=MODEL_TYPE, 
        choices=["flex_geo", "flex_geo_dinov3", "flex_geo_dinov3_posloss"]
    )

    p.add_argument(
        "--data_mode",
        type=str,
        default=DATA_MODE,
        choices=["manifest", "cluster"],
        help="manifest = train from manifest files, cluster = train from cluster/pair files",
    )

    p.add_argument("--train_pairs", type=str, default=TRAIN_PAIRS)
    p.add_argument("--val_pairs", type=str, default=VAL_PAIRS)
    p.add_argument("--dataset_root", type=str, default=VISYM_DATASET_ROOT)
    p.add_argument("--cache_root", type=str, default=CACHE_ROOT)
    p.add_argument("--train_manifest", type=str, default=TRAIN_MANIFEST)
    p.add_argument("--val_manifest", type=str, default=VAL_MANIFEST)
    p.add_argument("--site_id", type=str, default=SITE_ID)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--val_batch_size", type=int, default=VAL_BATCH_SIZE, help="<=0 uses train batch_size")
    p.add_argument("--n_query", type=int, default=N_QUERY)
    p.add_argument("--n_sat", type=int, default=N_SAT)
    p.add_argument("--sat_chip_size", type=int, default=SAT_CHIP_SIZE)
    p.add_argument(
        "--sat_chip_sizes",
        type=str,
        default=",".join(str(v) for v in SAT_CHIP_SIZES),
        help="comma-separated satellite chip sizes for multi-scale training (e.g. 120,140,160)",
    )
    p.add_argument("--sat_image_size", type=int, default=SAT_IMAGE_SIZE, help="<=0 keeps native sat chip size")
    p.add_argument("--negative_min_distance_px", type=float, default=NEGATIVE_MIN_DISTANCE_PX)
    p.add_argument("--negative_local_window_px", type=float, default=NEGATIVE_LOCAL_WINDOW_PX)
    p.add_argument(
        "--pos_center_jitter_px",
        type=float,
        default=POS_CENTER_JITTER_PX,
        help="max training-time deviation from the cluster-mean positive chip center, in pixels",
    )
    p.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    p.add_argument("--num_workers", type=int, default=LOADER_NUM_WORKERS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--min_lr", type=float, default=MIN_LR)
    p.add_argument("--warmup_epochs", type=int, default=LR_WARMUP_EPOCHS)
    p.add_argument("--wd", type=float, default=WD)
    p.add_argument(
        "--backbone_model_id",
        type=str,
        default=BACKBONE_MODEL_ID,
        help="Hugging Face model id for DINOv3 backbone (e.g. facebook/dinov3-vitb16-pretrain-lvd1689m)",
    )
    p.add_argument("--sff_scale", type=float, default=SFF_SCALE, help="SFF inverse-similarity exponent (paper uses 2)")
    p.add_argument("--share_backbone", action="store_true", default=True, help="share ground/sat encoder weights")
    p.add_argument("--no-share_backbone", dest="share_backbone", action="store_false", help="use separate ground/sat encoders")
    p.add_argument(
        "--ddp_timeout_minutes",
        type=int,
        default=DDP_TIMEOUT_MINUTES,
        help="process-group timeout in minutes (important when only rank0 runs long validation)",
    )
    p.add_argument(
        "--ddp_find_unused_parameters",
        action="store_true",
        default=DDP_FIND_UNUSED_PARAMETERS,
        help="enable DDP unused-parameter detection (safer across backbone variants)",
    )
    p.add_argument(
        "--no-ddp_find_unused_parameters",
        dest="ddp_find_unused_parameters",
        action="store_false",
        help="disable DDP unused-parameter detection (faster, but may error if params are conditionally unused)",
    )
    p.add_argument(
        "--freeze_backbone_stages",
        type=int,
        default=FREEZE_BACKBONE_STAGES,
        help="0=none, 1=embeddings, 2=embeddings+block0, 3=embeddings+block0..1, ... for both ground/sat DINOv3 encoders",
    )
    p.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=FREEZE_BACKBONE_EPOCHS,
        help="if >0, apply stage freeze only for first N epochs; <=0 keeps selected stages frozen for full run",
    )
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=EARLY_STOP_PATIENCE,
        help="stop when the monitored validation metric has no significant improvement for N epochs; <=0 disables",
    )
    p.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=EARLY_STOP_MIN_DELTA,
        help="minimum monitored-metric improvement required to reset early-stop patience",
    )
    p.add_argument(
        "--monitor_metric",
        type=str,
        default="auto",
        choices=["auto", "r1", "loss", "retr_loss", "pos_loss", "single_loss", "ita_loss"],
        help="metric used for best checkpoint selection and early stopping",
    )
    p.add_argument(
        "--init_ckpt",
        type=str,
        default="",
        help="optional checkpoint to load before training starts",
    )
    p.add_argument(
        "--init_ckpt_strict",
        action="store_true",
        default=True,
        help="require an exact state-dict match when loading --init_ckpt",
    )
    p.add_argument(
        "--no-init_ckpt_strict",
        dest="init_ckpt_strict",
        action="store_false",
        help="allow missing or unexpected keys when loading --init_ckpt",
    )
    p.add_argument(
        "--train_pos_only",
        action="store_true",
        default=False,
        help="freeze the retrieval branch and optimize only the position neck/head",
    )
    p.add_argument("--pretrained", action="store_true", default=PRETRAINED, help="use pretrained DINOv3 weights")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="do not use pretrained weights")
    p.add_argument("--amp", action="store_true", default=USE_AMP, help="use mixed precision training (CUDA only)")
    p.add_argument("--ial", action="store_true", default=ENABLE_IAL, help="enable ITA/IAL loss (uses orient_label)")
    p.add_argument("--no-ial", dest="ial", action="store_false", help="disable ITA/IAL loss")
    p.add_argument("--pos_weight", type=float, default=POS_LOSS_WEIGHT)
    p.add_argument("--pos_mode", type=str, default=POS_MODE)
    p.add_argument("--pos_grid", type=int, default=POS_GRID)
    p.add_argument(
        "--pos_loss_type",
        type=str,
        default=POS_LOSS_TYPE,
        choices=["ce", "reg"],
        help="position loss type: ce=grid/quadrant classification, reg=coordinate regression",
    )
    p.add_argument(
        "--pos_reg_beta",
        type=float,
        default=POS_REG_BETA,
        help="SmoothL1 beta for --pos_loss_type=reg (<=0 uses L1)",
    )
    p.add_argument(
        "--pos_reg_loss",
        type=str,
        default=POS_REG_LOSS,
        choices=["smooth_l1", "l1", "l2_mean", "l2_sum"],
        help="regression loss for --pos_loss_type=reg",
    )
    p.add_argument(
        "--pos_head_variant",
        type=str,
        default=POS_HEAD_VARIANT,
        choices=["legacy_mlp", "pairwise_residual", "sat_token_heatmap"],
        help="position head architecture",
    )
    p.add_argument(
        "--pos_head_hidden_dim",
        type=int,
        default=POS_HEAD_HIDDEN_DIM,
        help="hidden dim for the position head",
    )
    p.add_argument(
        "--pos_head_depth",
        type=int,
        default=POS_HEAD_DEPTH,
        help="depth parameter for the selected position head",
    )
    p.add_argument(
        "--separate_pos_neck",
        action="store_true",
        default=SEPARATE_POS_NECK,
        help="use a separate projection neck for the position branch while sharing the backbone",
    )
    p.add_argument(
        "--no-separate_pos_neck",
        dest="separate_pos_neck",
        action="store_false",
        help="reuse retrieval embeddings for the position branch (legacy behavior)",
    )
    p.add_argument("--single_weight", type=float, default=SINGLE_WEIGHT)
    p.add_argument("--ial_weight", type=float, default=IAL_WEIGHT)
    p.add_argument("--aux_warmup_epochs", type=int, default=AUX_WARMUP_EPOCHS)
    p.add_argument("--retrieval_only_warmup_epochs", type=int, default=RETRIEVAL_ONLY_WARMUP_EPOCHS)
    p.add_argument(
        "--retrieval_target_mode",
        type=str,
        default=RETRIEVAL_TARGET_MODE,
        choices=["gaussian", "hard"],
        help="how targets are assigned accross chips",
    )
    p.add_argument(
        "--retrieval_target_sigma_scale",
        type=float,
        default=RETRIEVAL_TARGET_SIGMA_SCALE,
        help="target weighting",
    )
    p.add_argument(
        "--retrieval_target_min_weight",
        type=float,
        default=RETRIEVAL_TARGET_MIN_WEIGHT,
        help="minimum threshold for including a chip in the soft target distribution",
    )

    p.add_argument("--ial_classes", type=int, default=IAL_CLASSES)
    p.add_argument("--save_every", type=int, default=SAVE_EVERY_EPOCHS)
    p.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--log_every", type=int, default=LOG_EVERY)
    p.add_argument(
        "--grad_accum_steps",
        type=int,
        default=GRAD_ACCUM_STEPS,
        help="gradient accumulation steps; effective batch = batch_size * world_size * grad_accum_steps",
    )
    p.add_argument("--cache", action="store_true", default=CACHE_ENABLED, help="enable record cache for Visym loader")
    p.add_argument("--no-cache", dest="cache", action="store_false", help="disable record cache")
    p.add_argument("--keep_sat_open", action="store_true", default=KEEP_SAT_OPEN, help="reuse open raster handles per worker")
    p.add_argument("--no-keep_sat_open", dest="keep_sat_open", action="store_false", help="open raster per sample")
    p.add_argument(
        "--persistent-workers",
        action="store_true",
        default=LOADER_PERSISTENT_WORKERS,
        help="keep dataloader workers alive between epochs",
    )
    p.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="restart workers each epoch (lower file-descriptor pressure)",
    )
    p.add_argument("--prefetch_factor", type=int, default=LOADER_PREFETCH_FACTOR)
    p.add_argument(
        "--mp_sharing_strategy",
        type=str,
        default=MP_SHARING_STRATEGY,
        choices=["file_system", "file_descriptor", "none"],
        help="torch multiprocessing tensor sharing strategy",
    )
    p.add_argument("--loader_progress", action="store_true", default=LOADER_PROGRESS, help="show dataset build/cache progress")
    p.add_argument("--no-loader_progress", dest="loader_progress", action="store_false", help="silence dataset build progress")
    p.add_argument("--loader_progress_every", type=int, default=LOADER_PROGRESS_EVERY)

    args = p.parse_args()

    return args


def main(args):
    # ------------------------------------------------------------------
    # Model-specific imports
    # ------------------------------------------------------------------
    apply_backbone_freeze = None
    count_params = None
    apply_position_only_training = None
    
    if args.model_type == "flex_geo":
        from models.flex_geo_match import (
            FlexGeoApprox,
            train_one_epoch,
            eval_model,
        )
    elif args.model_type == "flex_geo_dinov3":
        from models.flex_geo_match_dinov3 import (
            FlexGeoApprox,
            train_one_epoch,
            eval_model,
            apply_backbone_freeze,
            count_params
        )
    elif args.model_type == "flex_geo_dinov3_posloss":
        from models.flex_geo_match_dinov3_posloss import (
            FlexGeoApprox,
            train_one_epoch,
            eval_model,
            apply_backbone_freeze,
            count_params,
            apply_position_only_training,
        )
    else:
        raise ValueError(
            f"Incorrect model type selected.\n"
            f"{args.model_type} is not one of "
            f"['flex_geo', 'flex_geo_dinov3', 'flex_geo_dinov3_posloss']"
        )
    
    # ------------------------------------------------------------------
    # Args normalization
    # ------------------------------------------------------------------
    args.pos_loss_type = str(args.pos_loss_type).strip().lower()

    if hasattr(args, "monitor_metric"):
        args.monitor_metric = resolve_monitor_metric_name(
            args.monitor_metric,
            train_pos_only=bool(getattr(args, "train_pos_only", False)),
        )
    else:
        args.monitor_metric = "r1"

    args.init_ckpt = str(getattr(args, "init_ckpt", "")).strip()

    val_batch_size = int(args.batch_size) if int(args.val_batch_size) <= 0 else int(args.val_batch_size)
    sat_chip_sizes = parse_sat_chip_sizes(args.sat_chip_sizes, args.sat_chip_size)

    if args.sat_image_size <= 0 and len(sat_chip_sizes) > 1:
        args.sat_image_size = int(args.image_size)
        if is_main_process():
            print(
                "[INFO] Multi-scale sat chips requested with sat_image_size<=0. "
                f"Auto-setting sat_image_size={args.sat_image_size} for consistent batch tensor shapes."
            )

    # ------------------------------------------------------------------
    # Distributed setup
    # ------------------------------------------------------------------
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    ddp_timeout = timedelta(minutes=max(int(args.ddp_timeout_minutes), 1))

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA. Launch with torchrun on GPU nodes.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=ddp_timeout,
        )
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------------
    log_file_handle = None
    metrics_file_handle = None
    stdout_backup = None

    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        train_log_path = os.path.join(args.output_dir, "train.log")
        metrics_log_path = os.path.join(args.output_dir, "metrics.jsonl")

        stdout_backup = sys.stdout
        log_file_handle = open(train_log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = TeeStream(stdout_backup, log_file_handle)

        metrics_file_handle = open(metrics_log_path, "a", encoding="utf-8", buffering=1)

        print(f"[INFO] Logging stdout to {train_log_path}")
        print(f"[INFO] Writing metrics to {metrics_log_path}")

    if args.mp_sharing_strategy != "none":
        try:
            torch.multiprocessing.set_sharing_strategy(args.mp_sharing_strategy)
            if is_main_process():
                print(f"[INFO] torch sharing_strategy={args.mp_sharing_strategy}")
        except Exception as e:
            if is_main_process():
                print(f"[WARN] Failed to set sharing strategy to {args.mp_sharing_strategy}: {e}")

    if is_main_process() and distributed:
        print(
            f"[INFO] Distributed mode enabled: world_size={world_size} "
            f"ddp_timeout_minutes={int(args.ddp_timeout_minutes)} "
            # f"broadcast_buffers={int(bool(args.ddp_broadcast_buffers))}"
        )

    if args.data_mode == "manifest":
        if not os.path.isfile(args.train_manifest):
            raise FileNotFoundError(f"Missing train manifest file: {args.train_manifest}")
        if not os.path.isfile(args.val_manifest):
            raise FileNotFoundError(f"Missing val manifest file: {args.val_manifest}")
    elif args.data_mode == "cluster":
        if not args.train_pairs or not os.path.isfile(args.train_pairs):
            raise FileNotFoundError(f"Missing train pairs file: {args.train_pairs}")
        if not args.val_pairs or not os.path.isfile(args.val_pairs):
            raise FileNotFoundError(f"Missing val pairs file: {args.val_pairs}")
        if not args.dataset_root or not os.path.isdir(args.dataset_root):
            raise FileNotFoundError(f"Missing dataset_root directory: {args.dataset_root}")

    # ------------------------------------------------------------------
    # Dataset setup
    # ------------------------------------------------------------------
    if args.data_mode == "manifest":
        collate_fn = collate_image_pair
    elif args.data_mode == "cluster":
        collate_fn = collate_visym_cluster
    else:
        raise ValueError(f"Unsupported data_mode: {args.data_mode}")

    site_filter = None
    if args.site_id:
        site = args.site_id if args.site_id.startswith("site") else f"site{args.site_id}"
        site_filter = [site]

    train_ds = build_dataset(
        args=args,
        train=True,
        sat_chip_sizes=sat_chip_sizes,
        site_filter=site_filter,
    )

    # val_ds = None
    # if is_main_process():
    val_ds = build_dataset(
        args=args,
        train=False,
        sat_chip_sizes=sat_chip_sizes,
        site_filter=site_filter,
    )

    if is_main_process():
        print(
            f"train items: {len(train_ds)} val items: {len(val_ds) if val_ds is not None else 0} "
            f"batch_size(train/val)={args.batch_size}/{val_batch_size} "
            f"site={args.site_id} cache={args.cache} keep_sat_open={args.keep_sat_open} "
            # f"dataset_root={args.dataset_root} "
            f"pos_center_jitter_px={args.pos_center_jitter_px:.1f} "
            f"ground_resize_mode={GROUND_RESIZE_MODE} "
            f"sat_chip_sizes={sat_chip_sizes} sat_image_size={args.sat_image_size} "
            f"neg_min_dist={args.negative_min_distance_px:.1f} "
            f"neg_local_window={args.negative_local_window_px:.1f}"
        )
        if sat_chip_sizes:
            min_chip = float(min(sat_chip_sizes))
            if float(args.pos_center_jitter_px) < 0.25 * min_chip:
                print(
                    "[WARN] pos_center_jitter_px is small relative to sat chip size. "
                    "Position targets will stay center-biased and the model can collapse toward near-zero outputs."
                )

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    ) if distributed else None

    train_loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    ) if distributed else None

    val_loader_kwargs = dict(
        batch_size=val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    if args.num_workers > 0:
        prefetch = max(int(args.prefetch_factor), 1)
        train_loader_kwargs["prefetch_factor"] = prefetch
        val_loader_kwargs["prefetch_factor"] = prefetch
        if args.persistent_workers:
            train_loader_kwargs["persistent_workers"] = True
            val_loader_kwargs["persistent_workers"] = True

    train_ld = DataLoader(train_ds, **train_loader_kwargs)
    val_ld = DataLoader(val_ds, **val_loader_kwargs) if val_ds is not None else None

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if args.model_type == "flex_geo":
        model = FlexGeoApprox(
            embed_dim=1024,
            enable_pos=True,
            pos_mode=args.pos_mode,
            pos_grid=args.pos_grid,
            pretrained=args.pretrained,
            enable_ial=args.ial,
            ial_num_classes=args.ial_classes,
        ).to(device)
    elif args.model_type == "flex_geo_dinov3":
        model = FlexGeoApprox(
            embed_dim=1024,
            enable_pos=True,
            pos_mode=args.pos_mode,
            pos_grid=args.pos_grid,
            pos_loss_type=args.pos_loss_type,
            pos_reg_beta=args.pos_reg_beta,
            pretrained=args.pretrained,
            backbone_model_id=args.backbone_model_id,
            enable_ial=args.ial,
            ial_num_classes=args.ial_classes,
            sff_scale=args.sff_scale,
            share_backbone=args.share_backbone,
        ).to(device)
    elif args.model_type=="flex_geo_dinov3_posloss":
        model = FlexGeoApprox(
            embed_dim=1024,
            enable_pos=True,
            pos_mode=args.pos_mode,
            pos_grid=args.pos_grid,
            pos_loss_type=args.pos_loss_type,
            pos_reg_beta=args.pos_reg_beta,
            pos_head_variant=args.pos_head_variant,
            pos_head_hidden_dim=args.pos_head_hidden_dim,
            pos_head_depth=args.pos_head_depth,
            separate_pos_neck=args.separate_pos_neck,
            pretrained=args.pretrained,
            backbone_model_id=args.backbone_model_id,
            enable_ial=args.ial,
            ial_num_classes=args.ial_classes,
            sff_scale=args.sff_scale,
            share_backbone=args.share_backbone,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # ------------------------------------------------------------------
    # Optional init checkpoint
    # ------------------------------------------------------------------
    if args.init_ckpt:
        if load_init_ckpt is None:
            raise ValueError(
                f"--init_ckpt was provided, but model_type={args.model_type} "
                "does not support load_init_ckpt."
            )
        if not os.path.isfile(args.init_ckpt):
            raise FileNotFoundError(f"Missing init checkpoint: {args.init_ckpt}")

        init_info = load_init_ckpt(args.init_ckpt, model, strict=bool(args.init_ckpt_strict))
        if is_main_process():
            print(
                f"[INFO] Loaded init checkpoint: path={args.init_ckpt} "
                f"strict={int(bool(args.init_ckpt_strict))} "
                f"epoch={init_info.get('epoch')}"
            )
            missing_keys = init_info.get("missing_keys") or []
            unexpected_keys = init_info.get("unexpected_keys") or []
            if missing_keys:
                print(f"[WARN] init_ckpt missing_keys={len(missing_keys)}")
            if unexpected_keys:
                print(f"[WARN] init_ckpt unexpected_keys={len(unexpected_keys)}")

    # ------------------------------------------------------------------
    # Optional position-only training
    # ------------------------------------------------------------------
    if bool(getattr(args, "train_pos_only", False)):
        if apply_position_only_training is None:
            raise ValueError(
                f"--train_pos_only was provided, but model_type={args.model_type} "
                "does not support apply_position_only_training."
            )
        pos_only_stats = apply_position_only_training(model)
        if is_main_process():
            if count_params is not None:
                trainable_params, total_params = count_params(model)
                print(
                    f"[INFO] Position-only mode: enabled={pos_only_stats['enabled']} "
                    f"changed={pos_only_stats['changed']} "
                    f"trainable={trainable_params}/{total_params}"
                )
            else:
                print(
                    f"[INFO] Position-only mode: enabled={pos_only_stats['enabled']} "
                    f"changed={pos_only_stats['changed']}"
                )

    grad_layout_hooks = register_grad_layout_hooks(model)

    ddp_find_unused = bool(args.ddp_find_unused_parameters)
    if is_main_process() and distributed and ddp_find_unused:
        print(
            "[INFO] Enabling DDP find_unused_parameters=True to tolerate "
            "conditionally/partially unused parameters across model variants."
        )

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=ddp_find_unused,
            gradient_as_bucket_view=False,
            # broadcast_buffers=bool(args.ddp_broadcast_buffers),
        )

    if bool(getattr(args, "train_pos_only", False)):
        optim_params = [p for p in model.parameters() if p.requires_grad]
        if not optim_params:
            raise RuntimeError("No trainable parameters remain after applying --train_pos_only.")
    else:
        optim_params = model.parameters()

    optim = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # ------------------------------------------------------------------
    # Schedule / config
    # ------------------------------------------------------------------
    min_lr_factor = float(args.min_lr) / max(float(args.lr), 1e-12)
    min_lr_factor = min(max(min_lr_factor, 0.0), 1.0)

    if is_main_process():
        print(
            f"[INFO] LR schedule: warmup_epochs={max(int(args.warmup_epochs), 0)} "
            f"base_lr={args.lr:.3e} min_lr={args.min_lr:.3e} min_factor={min_lr_factor:.4f}"
        )
        print(
            f"[INFO] Batching: batch_size={int(args.batch_size)} world_size={int(world_size)} "
            f"grad_accum_steps={max(int(args.grad_accum_steps), 1)} "
            f"effective_batch={int(args.batch_size) * int(world_size) * max(int(args.grad_accum_steps), 1)}"
        )
        print(f"[INFO] Backbone: model_id={args.backbone_model_id} pretrained={int(bool(args.pretrained))}")
        print(f"[INFO] SFF: scale={float(args.sff_scale):.3f}")
        print(f"[INFO] Shared backbone: {int(bool(args.share_backbone))}")
        print(
            f"[INFO] Position head: mode={args.pos_mode} grid={int(args.pos_grid)} "
            f"loss_type={args.pos_loss_type} "
            f"reg_loss={getattr(args, 'pos_reg_loss', '<n/a>')} "
            f"reg_beta={float(args.pos_reg_beta):.4f}"
        )
        if hasattr(args, "pos_head_variant"):
            print(
                f"[INFO] Position head variant={args.pos_head_variant} "
                f"hidden_dim={int(args.pos_head_hidden_dim)} "
                f"depth={int(args.pos_head_depth)} "
                f"separate_pos_neck={int(bool(args.separate_pos_neck))}"
            )
        print(
            f"[INFO] Aux ramp: pos_weight={args.pos_weight:.4f} "
            f"single_weight={args.single_weight:.4f} "
            f"ial_weight={args.ial_weight:.4f} "
            f"warmup_epochs={max(int(args.aux_warmup_epochs), 0)} "
            f"retrieval_only_warmup_epochs={max(int(args.retrieval_only_warmup_epochs), 0)}"
        )
        print(
            f"[INFO] Backbone freeze: freeze_backbone_stages={int(args.freeze_backbone_stages)} "
            f"freeze_backbone_epochs={int(args.freeze_backbone_epochs)}"
        )
        print(
            f"[INFO] Early stop: patience={int(args.early_stop_patience)} "
            f"min_delta={float(args.early_stop_min_delta):.6f} "
            f"monitor_metric={args.monitor_metric}"
        )
        print(
            f"[INFO] Init/Fine-tune: init_ckpt={args.init_ckpt if args.init_ckpt else '<none>'} "
            f"train_pos_only={int(bool(getattr(args, 'train_pos_only', False)))}"
        )

    # code_snapshot_files: Dict[str, str] = {}
    # if is_main_process():
    #     code_snapshot_files = snapshot_training_code(args.output_dir)
    #     if code_snapshot_files:
    #         print("[INFO] Saved code snapshot files:")
    #         for k in sorted(code_snapshot_files.keys()):
    #             print(f"  - {k}: {code_snapshot_files[k]}")

    cfg: Dict[str, Any] = vars(args).copy()
    cfg["sat_chip_sizes"] = sat_chip_sizes
    cfg["val_batch_size"] = val_batch_size
    cfg["distributed"] = distributed
    cfg["world_size"] = world_size
    # cfg["code_snapshot_files"] = code_snapshot_files

    if is_main_process():
        cfg_path = os.path.join(args.output_dir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(
                cfg,
                f,
                indent=2,
                sort_keys=True,
            )
        print(f"[INFO] Saved run config: {cfg_path}")

    # ------------------------------------------------------------------
    # wandb
    # ------------------------------------------------------------------
    wb_run = None
    if is_main_process() and WANDB_ENABLE and wandb is not None:
        run_name = WANDB_RUN_NAME
        if not run_name:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            site = args.site_id if args.site_id else "all_sites"
            run_name = f"{site}_{stamp}"
        wb_run = wandb.init(project=WANDB_PROJECT, name=run_name, config=cfg, dir=args.output_dir)
    elif is_main_process() and WANDB_ENABLE and wandb is None:
        print("[WARN] wandb not installed; skipping wandb logging.")

    def log_fn(metrics: Dict[str, float], step: int):
        if not is_main_process():
            return
        if wb_run is not None:
            wandb.log(metrics, step=step)
        if metrics_file_handle is not None:
            rec: Dict[str, Any] = {
                "ts": datetime.now().isoformat(),
                "step": int(step),
            }
            rec.update(metrics)
            metrics_file_handle.write(json.dumps(rec, sort_keys=True) + "\n")

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------
    core_model = unwrap_model(model)
    best_r1 = float("-inf")
    best_monitor = monitor_metric_default_best(args.monitor_metric)
    early_best_monitor = monitor_metric_default_best(args.monitor_metric)
    early_bad_epochs = 0
    freeze_state_prev: Optional[bool] = None

    try:
        for epoch in range(args.epochs):
            # ----------------------------------------------------------
            # Freeze / unfreeze backbone
            # ----------------------------------------------------------
            if not bool(getattr(args, "train_pos_only", False)):
                freeze_enabled = int(args.freeze_backbone_stages) > 0
                if freeze_enabled:
                    if int(args.freeze_backbone_epochs) > 0:
                        freeze_active = epoch < int(args.freeze_backbone_epochs)
                    else:
                        freeze_active = True
                else:
                    freeze_active = False

                if (
                    apply_backbone_freeze is not None
                    and (freeze_state_prev is None or freeze_active != freeze_state_prev)
                ):
                    freeze_stats = apply_backbone_freeze(
                        core_model,
                        freeze_stages=int(args.freeze_backbone_stages),
                        freeze=freeze_active,
                    )
                    freeze_state_prev = freeze_active
                    if is_main_process() and count_params is not None:
                        trainable_params, total_params = count_params(core_model)
                        state = "frozen" if freeze_active else "unfrozen"
                        print(
                            f"[INFO] Backbone stage-freeze update: state={state} "
                            f"freeze_stages={int(args.freeze_backbone_stages)} "
                            f"resolved={int(freeze_stats.get('resolved_freeze_stages', int(args.freeze_backbone_stages)))} "
                            f"max={int(freeze_stats.get('max_stages', 1))} "
                            f"affected={freeze_stats['affected']} changed={freeze_stats['changed']} "
                            f"trainable={trainable_params}/{total_params}"
                        )

            # ----------------------------------------------------------
            # LR schedule
            # ----------------------------------------------------------
            lr_factor = cosine_with_warmup_factor(
                epoch=epoch,
                total_epochs=args.epochs,
                warmup_epochs=args.warmup_epochs,
                min_factor=min_lr_factor,
            )
            cur_lr = float(args.lr) * lr_factor
            for pg in optim.param_groups:
                pg["lr"] = cur_lr

            # ----------------------------------------------------------
            # Aux schedule
            # ----------------------------------------------------------
            if bool(getattr(args, "train_pos_only", False)):
                retrieval_only = False
                aux_factor = 1.0
                train_pos_weight = float(args.pos_weight)
                train_single_weight = 0.0
                train_ial_weight = 0.0
            else:
                retrieval_only = epoch < max(int(args.retrieval_only_warmup_epochs), 0)
                aux_epoch = max(epoch - max(int(args.retrieval_only_warmup_epochs), 0), 0)
                aux_factor = aux_ramp_factor(epoch=aux_epoch, warmup_epochs=args.aux_warmup_epochs)

                if retrieval_only:
                    train_pos_weight = 0.0
                    train_single_weight = 0.0
                    train_ial_weight = 0.0
                else:
                    train_pos_weight = float(args.pos_weight) * aux_factor
                    train_single_weight = float(args.single_weight) * aux_factor
                    train_ial_weight = float(args.ial_weight) * aux_factor

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_kwargs = dict(
                model=model,
                loader=train_ld,
                optim=optim,
                device=device,
                use_amp=args.amp and device.type == "cuda",
                scaler=scaler,
                pos_weight=train_pos_weight,
                single_weight=train_single_weight,
                ial_weight=train_ial_weight,
                accum_steps=max(int(args.grad_accum_steps), 1),
                epoch=epoch,
                log_every=args.log_every,
                global_step_base=epoch * len(train_ld),
                log_fn=log_fn,
                main_process=is_main_process(),
            )

            # only pass these if your train/eval functions support them
            if "pos_reg_loss" in inspect.signature(train_one_epoch).parameters:
                train_kwargs["pos_reg_loss"] = args.pos_reg_loss
            if "train_pos_only" in inspect.signature(train_one_epoch).parameters:
                train_kwargs["train_pos_only"] = bool(getattr(args, "train_pos_only", False))

            train_stats = train_one_epoch(**train_kwargs)

            loss = float(train_stats.get("loss", 0.0))
            train_retr_loss = float(train_stats.get("retr_loss", 0.0))
            stop_training = False

             # Sync all ranks after training epoch
            if distributed:
                try:
                    dist.barrier()
                except RuntimeError as e:
                    if "timeout" in str(e).lower():
                        if is_main_process():
                            print(f"[ERROR] DDP barrier timeout at epoch {epoch}. This may indicate:")
                            print("  - DataLoader workers hanging")
                            print("  - Different batch counts across ranks")
                            print("  - GPU/CUDA errors on non-main ranks")
                        # Try to continue or raise
                        raise RuntimeError("DDP barrier timeout - see logs") from e
                    raise

            # if is_main_process():
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

            if val_ld is None:
                raise RuntimeError("Validation loader is missing on main process.")

            eval_kwargs = dict(
                model=core_model,
                loader=val_ld,
                device=device,
                pos_weight=train_pos_weight,
                single_weight=train_single_weight,
                ial_weight=train_ial_weight,
                use_amp=args.amp and device.type == "cuda",
            )

            if "pos_reg_loss" in inspect.signature(eval_model).parameters:
                eval_kwargs["pos_reg_loss"] = args.pos_reg_loss

            metrics = eval_model(**eval_kwargs)

            if is_main_process():
                r1 = metrics[1]
                val_loss = float(metrics.get("loss", 0.0))
                val_retr_loss = float(metrics.get("retr_loss", 0.0))
                monitor_value = monitor_metric_value(metrics, args.monitor_metric)

                msg = (
                    f"epoch {epoch} train_loss {loss:.4f} train_retr {train_retr_loss:.4f} "
                    f"val_loss {val_loss:.4f} "
                    f"lr {cur_lr:.3e} aux_scale {aux_factor:.3f} "
                    f"retr_only {int(retrieval_only)} "
                    f"val_r1 {r1:.4f} val_r5 {metrics[5]:.4f} val_r10 {metrics[10]:.4f} "
                    f"val_retr {val_retr_loss:.4f} "
                    f"monitor({args.monitor_metric}) {monitor_value:.4f}"
                )
                if "pos_loss" in train_stats:
                    msg += f" train_pos {float(train_stats['pos_loss']):.4f}"
                if "single_loss" in train_stats:
                    msg += f" train_single {float(train_stats['single_loss']):.4f}"
                if "ita_loss" in train_stats:
                    msg += f" train_ita {float(train_stats['ita_loss']):.4f}"
                if "pos_loss" in metrics:
                    msg += f" val_pos {metrics['pos_loss']:.4f}"
                if "single_loss" in metrics:
                    msg += f" val_single {metrics['single_loss']:.4f}"
                if "ita_loss" in metrics:
                    msg += f" val_ita {metrics['ita_loss']:.4f}"
                msg += (
                    f" train_pos_valid {float(train_stats.get('pos_valid_frac', 0.0)):.3f}"
                    f" val_pos_valid {float(metrics.get('pos_valid_frac', 0.0)):.3f}"
                    f" train_ita_valid {float(train_stats.get('orient_valid_frac', 0.0)):.3f}"
                    f" val_ita_valid {float(metrics.get('ita_valid_frac', 0.0)):.3f}"
                )
                print(msg)

                val_log = {
                    "epoch": float(epoch),
                    "train/loss_epoch": float(loss),
                    "train/retr_loss_epoch": float(train_retr_loss),
                    "train/pos_valid_frac": float(train_stats.get("pos_valid_frac", 0.0)),
                    "train/ita_valid_frac": float(train_stats.get("orient_valid_frac", 0.0)),
                    "val/loss": val_loss,
                    "val/retr_loss": val_retr_loss,
                    "val/r1": float(r1),
                    "val/r5": float(metrics[5]),
                    "val/r10": float(metrics[10]),
                    "train/lr_epoch": cur_lr,
                    "train/aux_scale": float(aux_factor),
                    "train/retrieval_only": float(int(retrieval_only)),
                    "val/monitor_metric": float(monitor_value),
                    "train/pos_weight_epoch": train_pos_weight,
                    "train/single_weight_epoch": train_single_weight,
                    "train/ial_weight_epoch": train_ial_weight,
                    "val/pos_valid_frac": float(metrics.get("pos_valid_frac", 0.0)),
                    "val/ita_valid_frac": float(metrics.get("ita_valid_frac", 0.0)),
                }

                if "pos_loss" in train_stats:
                    val_log["train/pos_loss_epoch"] = float(train_stats["pos_loss"])
                if "single_loss" in train_stats:
                    val_log["train/single_loss_epoch"] = float(train_stats["single_loss"])
                if "ita_loss" in train_stats:
                    val_log["train/ita_loss_epoch"] = float(train_stats["ita_loss"])
                if "pos_loss" in metrics:
                    val_log["val/pos_loss"] = float(metrics["pos_loss"])
                if "single_loss" in metrics:
                    val_log["val/single_loss"] = float(metrics["single_loss"])
                if "ita_loss" in metrics:
                    val_log["val/ita_loss"] = float(metrics["ita_loss"])

                log_fn(val_log, step=(epoch + 1) * len(train_ld))

                # Periodic checkpoint
                if (epoch + 1) % max(int(args.save_every), 1) == 0:
                    ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch+1:03d}.pt")
                    save_ckpt(ckpt_path, model, optim, scaler, epoch, metrics, cfg)
                    print(f"saved {ckpt_path}")

                # Keep tracking best r1
                if r1 > best_r1:
                    best_r1 = r1

                # Best checkpoint based on monitor metric
                if monitor_metric_is_better(monitor_value, best_monitor, args.monitor_metric, 0.0):
                    best_monitor = monitor_value
                    best_path = os.path.join(args.output_dir, "best.pt")
                    save_ckpt(best_path, model, optim, scaler, epoch, metrics, cfg)
                    print(f"saved {best_path}")

                # Early stopping
                if int(args.early_stop_patience) > 0:
                    if monitor_metric_is_better(
                        monitor_value,
                        early_best_monitor,
                        args.monitor_metric,
                        float(args.early_stop_min_delta),
                    ):
                        early_best_monitor = monitor_value
                        early_bad_epochs = 0
                    else:
                        early_bad_epochs += 1
                        if early_bad_epochs >= int(args.early_stop_patience):
                            stop_training = True
                            print(
                                f"[INFO] Early stopping at epoch {epoch}: "
                                f"{args.monitor_metric}={monitor_value:.4f} "
                                f"best_{args.monitor_metric}={early_best_monitor:.4f} "
                                f"min_delta={float(args.early_stop_min_delta):.6f} "
                                f"patience={int(args.early_stop_patience)}"
                            )

            if distributed:
                stop_tensor = torch.zeros(1, device=device, dtype=torch.int32)
                if is_main_process() and stop_training:
                    stop_tensor[0] = 1
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())

            if stop_training:
                break

    finally:
        if is_main_process() and wb_run is not None:
            wb_run.finish()
        if distributed:
            dist.destroy_process_group()
        if is_main_process() and metrics_file_handle is not None:
            metrics_file_handle.close()
        if is_main_process() and log_file_handle is not None:
            sys.stdout.flush()
            if stdout_backup is not None:
                sys.stdout = stdout_backup
            log_file_handle.close()

if __name__=="__main__":
    args = parse_args()
    main(args)