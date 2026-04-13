#!/usr/bin/env python3
"""
Overlay GT and pred_position_head points on a satellite image.

Usage:
    python overlay_clusters.py \
        --txt_path /path/to/example.txt \
        --satellite_path /path/to/satellite_image.png \
        --output_path /path/to/overlay.png

Notes:
- Assumes the txt file is tab-separated.
- Ground truth is drawn as a crosshair (+) in neon green.
- Prediction is drawn as an X in red.
- Dotted grey lines are drawn between GT and predictions.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

GT_COLOR = "#39FF14"      
PRED_COLOR = "#FF0000"   
PAIR_LINE_COLOR = "#FFFFFF" 


def load_data(txt_path: str) -> pd.DataFrame:
    """
    Load the tab-separated text file into a DataFrame.
    """
    df = pd.read_csv(txt_path, sep="\t")

    required_cols = [
        "cluster_id",
        "gt_x_px",
        "gt_y_px",
        "pred_position_head_x_px",
        "pred_position_head_y_px",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def load_image(image_path: str) -> np.ndarray:
    """
    Load a GeoTIFF image as an RGB NumPy array with correct colors.
    """
    with rasterio.open(image_path) as src:
        # Read all bands
        img = src.read()  # shape: (bands, height, width)

        # If at least 3 bands, assume RGB
        if img.shape[0] >= 3:
            img = img[:3]  # take first 3 bands (R, G, B)
        else:
            # Single band → replicate to RGB
            img = np.repeat(img, 3, axis=0)

        # Convert to (H, W, C)
        img = np.transpose(img, (1, 2, 0))

        # Normalize if needed (e.g., 16-bit → 8-bit)
        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype(np.uint8)

    return img


def draw_crosshair(
    ax: plt.Axes,
    x: float,
    y: float,
    color,
    size: float = 4,
    linewidth: float = 1,
    zorder: int = 2,
) -> None:
    """
    Draw a crosshair (+) centered at (x, y).
    """
    ax.plot(
        [x - size, x + size],
        [y, y],
        color=color,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.plot(
        [x, x],
        [y - size, y + size],
        color=color,
        linewidth=linewidth,
        zorder=zorder,
    )


def plot_overlay(
    image: np.ndarray,
    df: pd.DataFrame,
    output_path: str | None = None,
    gt_crosshair_size: float = 7,
    pred_marker_size: float = 18,
    figsize: Tuple[int, int] = (12, 12),
    hide_labels: bool = False,
) -> None:
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.set_title("Satellite Overlay: Ground Truth (+) vs Prediction (x)")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    gt_x = df["gt_x_px"].to_numpy(dtype=float)
    gt_y = df["gt_y_px"].to_numpy(dtype=float)
    pred_x = df["pred_position_head_x_px"].to_numpy(dtype=float)
    pred_y = df["pred_position_head_y_px"].to_numpy(dtype=float)

    # Dotted grey lines between GT and prediction pairs
    for xg, yg, xp, yp in zip(gt_x, gt_y, pred_x, pred_y):
        ax.plot(
            [xg, xp],
            [yg, yp],
            color=PAIR_LINE_COLOR,
            linestyle=":",
            linewidth=0.5,
            alpha=0.7,
            zorder=1,
        )

    # GT crosshairs
    for xg, yg in zip(gt_x, gt_y):
        draw_crosshair(
            ax,
            xg,
            yg,
            color=GT_COLOR,
            size=gt_crosshair_size,
            linewidth=1.5,
            zorder=2,
        )

    # Predictions plotted on top of GT
    ax.scatter(
        pred_x,
        pred_y,
        s=pred_marker_size,
        c=PRED_COLOR,
        marker="x",
        linewidths=0.8,
        label="Prediction",
        zorder=3,
    )

    marker_legend = [
        Line2D(
            [0], [0],
            marker="+",
            color=GT_COLOR,
            linestyle="None",
            markersize=8,
            markeredgewidth=1.5,
            label="Ground Truth (+)",
        ),
        Line2D(
            [0], [0],
            marker="x",
            color=PRED_COLOR,
            linestyle="None",
            markersize=6,
            markeredgewidth=1.2,
            label="Prediction (x)",
        ),
        Line2D(
            [0], [0],
            color=PAIR_LINE_COLOR,
            linestyle=":",
            linewidth=1.0,
            label="GT-Prediction Pair",
        ),
    ]

    if not hide_labels:
        ax.legend(
            handles=marker_legend,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            title="Markers",
        )

    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_aspect("equal")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay GT/predictions on a satellite image.")
    parser.add_argument("--txt_path", required=True, help="Path to tab-separated txt file.")
    parser.add_argument("--satellite_path", required=True, help="Path to satellite image.")
    parser.add_argument("--output_path", default=None, help="Optional path to save output image.")
    parser.add_argument(
        "--hide_labels",
        action="store_true",
        help="Hide marker legend.",
    )
    return parser.parse_args()


def visualize_evaluations(args):
    df = load_data(args.txt_path)
    image = load_image(args.satellite_path)

    plot_overlay(
        image=image,
        df=df,
        output_path=args.output_path,
        hide_labels=args.hide_labels,
    )


def main() -> None:
    args = parse_args()
    visualize_evaluations(args)


if __name__ == "__main__":
    main()