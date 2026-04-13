# JHU APL WRIVA CVGL Baseline

This repository contains the baseline code for the **WRIVA-CVGL Challenge 2026**.

Competition page and dataset download:

- https://ieee-dataport.org/competitions/wriva-cvgl-challenge-2026

This repository is actively being updated. Future releases will include:
- Updated baseline models
- Updated evaluation data and metrics

## 1. Background

The objective of this baseline is to predict camera locations within a satellite image, where each camera corresponds to a ground-view perspective image. The predicted results are reported in **pixel coordinates**.

To simplify the initial competition setting, the current baseline focuses on prediction inside a **420m x 420m** search region. With a **0.5m ground sample distance (GSD)** on the satellite imagery, this corresponds to an **840 x 840 pixel** search window.

The baseline predicts:

- a retrieval score over candidate satellite tiles
- a finer in-tile position estimate for the ground image camera location

Those pixel predictions can then be aligned with the GeoTIFF coordinate system of the satellite raster.

This codebase is still under development. We will continue updating the baseline and the documentation in this repository during the competition.

## 2. Datasets

This repository was designed to be compatible with two datsets: `Visym-CVGL` and `WRIVA-CVGL-DEV`. Currently the satellite images are centered relative to the camera locations to ensure coverage. However, **test images may not always be centered when presented**. 

The datasets are described in more detail below.

### 2.1 Visym-CVGL

Visym-CVGL is used for **training** the baseline model.

It contains:

- perspective-view ground images
- ground-truth ground image camera locations and georegistration metadata
- satellite imagery and georegistration metadata

This makes it suitable for:

- training
- validation
- supervised evaluation during development

### 2.2 WRIVA-CVGL-DEV

WRIVA-CVGL-DEV is the **first released testing batch** for baseline evaluation and challenge development.

It is used for:

- testing the inference pipeline
- benchmarking retrieval and localization behavior
- validating post-processing and evaluation code

We will release additional CVGL evaluation data later.

## 3. Dataset Structure

### 3.1 Common Site Layout

Most scripts expect a dataset root containing site folders like this:

```text
<dataset_root>/
  <site_id>/
     ground/
      <image_id>.jpg
    maxar/
      <satellite_image>.tif
    reference/
      <image_id>.json
```

Notes:

- `ground/` containes only perspective-view ground images corresponding to camera locations..
- `maxar/` stores the site-level satellite GeoTIFFs.
- `reference/` stores the per-image camera location metadata for each ground image.

### 3.2 Visym-CVGL Layout

For training, Visym-CVGL follows the same site-oriented layout:

```text
Visym-CVGL/
  <site_id>/
    ground/
      *.jpg
    maxar/
      *.tif
    reference/
      *.json
```

The `reference/*.json` files provide the metadata needed for training labels and validation.

### 3.3 WRIVA-CVGL-DEV Layout

For testing, WRIVA-CVGL-DEV is organized the same way:

```text
WRIVA-CVGL-DEV/
  <site_id>/
    ground/
      *.jpg
    maxar/
      *.tif
    reference/
      *.json
```

In the competition setting, WRIVA data is used as the held-out test-style dataset for inference and evaluation workflows.

## 4. Installation

From the repository root, install the Python dependencies used by the `src/` workflow:

```bash
pip install -r requirements.txt
```

You will also need:

- a PyTorch environment
- GPU support for training and most inference runs
- rasterio-compatible GDAL installation for GeoTIFF reading

## 5. Baseline Model

Our baseline model was trained using the equivalent of 2 A100 GPUs.

The baseline implements part of the **Set-CVGL** idea from:

- https://arxiv.org/abs/2412.18852

This repository does **not** try to reproduce that paper exactly. We use part of the idea and architecture direction, but the implementation here contains significant modifications.

### 5.1 Model Summary

The current baseline model used in `train.py` supports these variants:

- `flex_geo`: Initial implementation of ideas from the Set-CVGL paper.
- `flex_geo_dinov3`: Using the DinoV3 model as the backbone for the shared encoder layers.
- `flex_geo_dinov3_posloss`: Initial implementation of position loss.
- `flex_geo_dinov3_posloss_v2`: Current implementation of position loss.

The main baseline training launcher currently uses:

- `flex_geo_dinov3_posloss_v2`

### 5.2 Backbone

The current training code uses a **DINOv3** visual backbone for both the ground and satellite branches. In the default training configuration, the backbone model id is:

- `facebook/dinov3-vitb16-pretrain-lvd1689m`

### 5.3 Retrieval Branch

The retrieval branch:

- encodes the ground images
- fuses one or more ground images into a joint representation
- encodes the candidate satellite chips
- predicts similarity scores between the ground representation and the satellite chips

The retrieval objective is to select the correct satellite tile or chip that contains the target ground-image location. **The model makes the assumption that the target camera location is close to the target ground-image location.**

### 5.4 Position Prediction Branch

The position branch predicts the ground-image location **inside the selected satellite chip**.

Depending on the model configuration, this branch can output either:

- a heatmap-like prediction over satellite tokens
- or normalized in-chip `(x, y)` coordinates

Those predictions are then converted to:

- chip-local pixel coordinates
- full satellite-image pixel coordinates
- GeoTIFF-aligned positions

The final objective is to estimate the ground-image location in satellite-image pixels and align that prediction with the GeoTIFF coordinates.

### 5.5 Single-Image and Multi-Image Ground Input

The baseline is designed to be robust to either:

- a single ground image
- multiple neighboring ground images

This is controlled in training by parameters such as:

- `--n_query`

We are actively investigating both single-image and multi-image input settings. The multi-image clustered input is motivated by the set-based CVGL concept in the Set-CVGL paper, but the implementation here has been significantly adapted for this baseline.

## 6. Data Loaders

The baseline currently uses two main data-loading modes for the core workflow.

### 6.1 Training and Validation Loader

Training and validation primarily use:

- `models/image_pair_dataloader.py`
- class: `ImagePairDataset`

This loader:

- reads ground images and their associated satellite raster
- samples satellite sub-chips of predefined size
- identifies positive and negative chips using the ground-truth projected location
- supports multiple sampling modes including positive-centered chips, random chips, and tiled coverage

For training, the important idea is:

- one or more chips are treated as positives based on overlap with the ground-image camera location
- the remaining chips are treated as negatives
- the model learns both retrieval and in-chip position prediction

Key parameters used in code include:

- `--n_query`: number of ground images in one training sample
- `--n_sat`: number of satellite chips sampled per sample
- `--sat_chip_size`: base chip size in pixels
- `--sat_chip_sizes`: optional multi-scale chip sizes
- `--sat_image_size`: network input size for satellite chips
- `--negative_min_distance_px`
- `--negative_local_window_px`
- `--pos_center_jitter_px`

### 6.2 Tile-Based Inference Loader

For inference on WRIVA-style evaluation data, the baseline uses:

- `models/inference_dataloader.py`
- class: `ClusterInferenceDataset`

This loader:

- reads `camera_clusters.txt`
- groups neighboring ground images into clusters
- tiles satellite rasters into overlapping chips using a fixed stride
- returns all tiled chips needed for inference over the requested search region

This tile-based loader is the basis for the inference section below. More detail is described in the inference workflow.

## 7. Data Preparation and Splitting

Due to data size, the training code does not train directly from raw data alone. Instead, the data is first preprocessed in to splits that are stored in a `.txt` file, then converted to manifests (`.pkl`) that are compatible with the training dataloader.

### 7.1 Split Generation

Split generation is handled in:

- `splits/create_splits.py`

This script creates site-disjoint train/val/test split files. To avoid data leakage between sites, each site can only be in a single split.

### 7.2 Manifest Generation

Manifest generation is handled in:

- `splits/build_image_pair_manifest.py`

Example usage can be found in:

- `splits/make_splits.sh`
- `splits/build_manifests.sh`
- `splits/make_splits_and_build_manifest.sh`

These scripts still use hard-coded paths, so read and edit them before running.

### 7.3 Example Split and Manifest Workflow

Create splits:

```bash
cd splits

python create_splits.py \
  --root_dir <VISYM_DATASET_ROOT> \
  --out_dir splits/visym \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42 \
  --output_relative_paths \
  --validate_with_reference
```

Build manifests:

```bash
python build_image_pair_manifest.py \
  --pairs-txt splits/visym/train_split.txt \
  --dataset-root <VISYM_DATASET_ROOT> \
  --output splits/visym/train_manifest.pkl \
  --no-validate-paths

python build_image_pair_manifest.py \
  --pairs-txt splits/visym/val_split.txt \
  --dataset-root <VISYM_DATASET_ROOT> \
  --output splits/visym/val_manifest.pkl \
  --no-validate-paths
```

## 8. Training the Baseline Model

This section is based on the current code in `train.py` and `train.sh`.

### 8.1 Main Training Entry Point

Primary training entrypoint:

- `train.py`

Simplest launcher:

- `train.sh`

### 8.2 Current Default Launcher

The current `train.sh` launches:

- `model_type=flex_geo_dinov3_posloss_v2`
- `data_mode=manifest`
- distributed training with `torchrun`
- two GPUs by default

Example:

```bash
bash train.sh
```

### 8.3 What the Training Script Does

At a high level, `train.py`:

1. loads the chosen model variant
2. builds the training and validation datasets
3. creates data loaders
4. trains the retrieval branch and the position branch jointly
5. writes checkpoints, logs, and the run config

The current code supports:

- retrieval loss
- position loss
- optional IAL / attribute loss
- optional retrieval-only warmup
- optional position-only training
- DDP training
- checkpoint loading and resume-style initialization

### 8.4 Important Training Parameters

The following parameters are the most important ones for users:

- `--model_type`: model family
- `--data_mode`: `manifest` or `cluster`
- `--train_manifest`
- `--val_manifest`
- `--output_dir`
- `--n_query`
- `--n_sat`
- `--sat_chip_size`
- `--sat_chip_sizes`
- `--sat_image_size`
- `--batch_size`
- `--epochs`
- `--lr`
- `--min_lr`
- `--pos_weight`
- `--pos_loss_type`
- `--pos_reg_loss`
- `--pos_head_variant`
- `--single_weight`
- `--ial_weight`
- `--monitor_metric`

### 8.5 Recommended Training Notes

- Use **Visym-CVGL** for training and validation.
- Keep site-level separation between train and validation splits.
- If file paths change, regenerate manifests.
- If you use the provided shell scripts, verify the hard-coded paths first.

## 9. Inference

The intended inference flow has four stages:

  1. Prepare camera clusters via filename parsing OR colmap.
      - The example provided is configured to use filename parsing
  2. Tile-based inferencing
      - Unlike in training, there is no ground truth or positive chip during inference. As a result, the inference dataloader will tile the satellite into chips, and then perform inference on each chip. 
  3. Post processing and aggregation
      - The results for each chip after inference are aggregated 
  4. Comparison with ground truth

The full four-step inference workflow can be run with 
```
python inference_pipeline.py
```
which is automatically configured to read inference configs from `inference_configs/infer_pipeline.json`.

The pipeline can also be run with 
```
python inference.py --config /path/to/config
```

After running the inference workflow, results can be plotted by using the following command:
```
python inference_utils/visualize_evaluations.py \
    --txt_path /path/to/eval/txt/file \
    --satellite_path /path/to/corresponding/satellite/tiff \
    --output_path /path/to/output
```
Examples are also shown in visualize_inference.sh

Below, each step is described in more detail.

### 9.1 Step 1: Prepare Camera Clusters

The model can take multiple neighboring ground images and jointly predict:

- a retrieval location over satellite tiles
- a position estimate for each image

To do this, ground images need to be grouped into clusters.

Current script:

- `prepare_camera_clusters.py`

This script supports two clustering strategies:

- `filename_order`
- `colmap`

#### Filename-Order Clustering

The current simplified approach can cluster images by filename order. In the current datasets, filenames encode timestamps, so this gives a practical approximation to temporal neighborhoods.

#### COLMAP + HLoc Clustering

For future cases where filenames are shuffled or no longer preserve time order, the repository also provides camera registration using:

- HLoc
- COLMAP

That path is available in the same preparation script and is intended for more geometry-aware clustering.

Example:

```bash
python prepare_camera_clusters.py \
  --dataset_root <SITE_ROOT> \
  --cluster_source filename_order \
  --cluster_size 8
```

Main parameters:

- `--dataset_root`
- `--ground_dir`
- `--cluster_source`
- `--cluster_size`
- `--cluster_view_soft_angle_deg`
- `--cluster_view_hard_angle_deg`

Output:

```text
<output_root>/<dataset_name>/registration/camera_clusters.txt
```

This file is generated by `prepare_camera_clusters.py`. It is not part of the originally provided dataset.

Current format:

```text
ground_filename    local_cluster_id    x    y    global_cluster_id
```

Field meanings:

- `ground_filename`: image filename in the site `ground/` folder
- `local_cluster_id`: local 0-based cluster id used by inference
- `x`, `y`: cluster location values used for downstream clustering logic
- `global_cluster_id`: larger registration component id

### 9.2 Step 2: Run Tile-Based Inference

Main script:

- `infer_tiles.py`

This script:

- loads clustered ground-image inputs
- tiles the satellite imagery with a fixed chip size and stride
- scores every candidate tile with the retrieval branch
- predicts per-image in-chip positions for the selected tiles

The tiled inference configuration is controlled by:

- `inference_configs/infer_tiles.json`

Main configuration groups:

#### Data

- `txt_file`: path to `camera_clusters.txt`
- `image_base_dir`: dataset or site root
- `sat_chip_size`: chip size in pixels
- `ground_image_size`: resized input size for ground images
- `overlap_ratio`: determines stride between adjacent tiles
- `sat_sampling_window_px`: optional spatial crop window inside the satellite image
- `has_header`
- `delimiter`

#### Runtime

- `device`
- `batch_size`
- `num_workers`

#### Model

- `model_type`
- `model_root`
- `config_filename`
- `checkpoint_name`

#### Output

- `base_dir`
- `top_k_sat_chips`
- `top_k_satellites`
- `score_reduction`

Run:

```bash
bash inference.sh
```

which currently runs:

```bash
python infer_tiles.py --config inference_configs/infer_tiles.json
```

Outputs currently include:

- per-cluster `ranking.json`
- per-site `all_results.json`

These JSON outputs store the retrieval scores for the tiled chips and the per-chip position predictions.

### 9.3 Step 3: Post-Processing and Aggregation

After inference, the baseline supports an optimization and aggregation step.

Main script:

- `infer_tiles_neighbor_postsum.py`

Conceptually, the process is:

1. build a score map over all candidate tiles
2. apply neighborhood summation across nearby ground-image clusters
3. aggregate across satellites when needed
4. select the strongest final prediction

This stage produces:

- best-satellite predictions
- all-satellite aggregate predictions
- versions before and after post-sum aggregation

Important note:

- this part of the baseline is still under active development
- the repository currently uses a dataset-level inference JSON handoff for this stage
- we will continue updating this flow in the repo

### 9.4 Step 4: Compare with Ground Truth

Evaluation script:

- `evaluate_cluster_predictions_against_gt.py`

This script compares predictions against ground truth and reports:

- top-K retrieval metrics
- retrieval-head pixel error
- position-head pixel error

The current evaluator reports four result variants:

- retrieval best satellite
- retrieval all-satellite aggregate
- neighbor-postsum best satellite
- neighbor-postsum all-satellite aggregate

## 10. Notes on Current Development Status

- The baseline is still evolving.
- Some scripts still use hard-coded local paths.
- The tile inference and post-processing pipeline will continue to be cleaned up during the competition.
- Additional CVGL evaluation data will be released later.

## 11. Credit

This baseline borrows part of its architectural motivation from the following work:

Wu, Qiong, et al. "Cross-view image set geo-localization." arXiv preprint arXiv:2412.18852 (2024).

We implement only partial concepts from that work, with significant modifications for the WRIVA-CVGL baseline setting.
