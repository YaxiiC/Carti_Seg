# -*- coding: utf-8 -*-
"""
MedSAM inference on prepared .npy slices using POINT prompts + visualization + Dice.

Assumes your data prep saved:
  imgs/<prefix>case-###.npy  # (1024,1024,3) float32 in [0,1]
  gts/<prefix>case-###.npy   # (1024,1024)   uint8 labels (0..5)
"""

import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch
from segment_anything import sam_model_registry, SamPredictor

# ---------- CONFIG DEFAULTS (override via CLI) ----------
DEFAULT_IMGS_DIR = r"C:\Users\chris\MICCAI2026\MRI_Knee_OAIZIBCM\imgs"
DEFAULT_GTS_DIR  = r"C:\Users\chris\MICCAI2026\MRI_Knee_OAIZIBCM\gts"
DEFAULT_CKPT     = r"C:\Users\chris\MICCAI2026\MedSAM\work_dir\MedSAM\medsam_vit_b.pth"

# OAIZIB-CM label ids -> names
LABELS = {
    1: "Femur",
    2: "Femoral Cartilage",
    3: "Tibia",
    4: "Medial Tibial Cartilage",
    5: "Lateral Tibial Cartilage",
}

# semi-transparent overlay colors for labels 0..5 (0 is background)
LABEL_COLORS = [
    (0.0, 0.0, 0.0, 0.0),   # 0 background (transparent)
    (1.0, 0.0, 0.0, 0.55),  # 1 red
    (0.0, 1.0, 0.0, 0.55),  # 2 green
    (0.0, 0.0, 1.0, 0.55),  # 3 blue
    (1.0, 1.0, 0.0, 0.55),  # 4 yellow
    (1.0, 0.0, 1.0, 0.55),  # 5 magenta
]
CMAP = ListedColormap(LABEL_COLORS)

# ---------- UTILS ----------
def pick_slices(img_dir, gt_dir, num_slices=5, seed=0):
    rng = random.Random(seed)
    candidates = [f[:-4] for f in os.listdir(img_dir) if f.endswith(".npy")]
    candidates = [b for b in candidates if os.path.exists(os.path.join(gt_dir, b + ".npy"))]
    if not candidates:
        raise FileNotFoundError(f"No matching (img, gt) .npy pairs found under {img_dir} and {gt_dir}")
    rng.shuffle(candidates)
    return candidates[:num_slices]

def to_uint8_rgb(img_float01):
    """(H,W,3) float [0,1] -> uint8 RGB [0,255]"""
    x = np.clip(img_float01, 0, 1) * 255.0
    return x.astype(np.uint8)

def dice_score(gt, pr, label_id):
    gt_bin = (gt == label_id)
    pr_bin = (pr == label_id)
    inter = (gt_bin & pr_bin).sum()
    s = gt_bin.sum() + pr_bin.sum()
    return (2.0 * inter / s) if s > 0 else np.nan  # NaN if class absent in both

def sample_points_for_label(gt, lid, k_pos=1, k_bg=0, rng=None):
    """
    Sample k_pos foreground points from label 'lid' and k_bg background points elsewhere.
    Returns:
        point_coords: (N,2) float32 in (x,y)
        point_labels: (N,) 1 for FG, 0 for BG
        vis: dict with coords separated for plotting
    """
    if rng is None:
        rng = np.random.default_rng()

    H, W = gt.shape
    ys, xs = np.where(gt == lid)
    if ys.size == 0:
        return None, None, {"fg": np.zeros((0,2)), "bg": np.zeros((0,2))}

    # Foreground picks
    sel = rng.choice(ys.size, size=min(k_pos, ys.size), replace=False)
    fg_points = np.stack([xs[sel], ys[sel]], axis=1).astype(np.float32)  # (x,y)

    # Background picks: anywhere gt==0 (simple), random
    bg_points = np.zeros((0,2), dtype=np.float32)
    if k_bg > 0:
        by, bx = np.where(gt == 0)
        if by.size > 0:
            bsel = rng.choice(by.size, size=min(k_bg, by.size), replace=False)
            bg_points = np.stack([bx[bsel], by[bsel]], axis=1).astype(np.float32)

    # Combine
    if bg_points.shape[0] > 0:
        point_coords = np.concatenate([fg_points, bg_points], axis=0)
        point_labels = np.concatenate([np.ones((fg_points.shape[0],), dtype=np.int32),
                                       np.zeros((bg_points.shape[0],), dtype=np.int32)], axis=0)
    else:
        point_coords = fg_points
        point_labels = np.ones((fg_points.shape[0],), dtype=np.int32)

    vis = {"fg": fg_points, "bg": bg_points}
    return point_coords, point_labels, vis

def visualize(img, gt, pred, clicks_per_label, title=""):
    """
    clicks_per_label: dict lid -> {'fg': (n,2), 'bg': (m,2)} in (x,y)
    """
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    axs[0].imshow(img)
    axs[0].set_title("Image"); axs[0].axis("off")

    axs[1].imshow(gt, cmap=CMAP, vmin=0, vmax=len(LABEL_COLORS)-1)
    axs[1].set_title("GT"); axs[1].axis("off")

    axs[2].imshow(pred, cmap=CMAP, vmin=0, vmax=len(LABEL_COLORS)-1)
    axs[2].set_title("Pred (points)"); axs[2].axis("off")

    axs[3].imshow(img)
    axs[3].imshow(pred, cmap=CMAP, vmin=0, vmax=len(LABEL_COLORS)-1)
    axs[3].set_title("Overlay + Clicks"); axs[3].axis("off")

    # Draw clicks
    for lid, pts in clicks_per_label.items():
        fg = pts.get("fg", np.zeros((0,2)))
        bg = pts.get("bg", np.zeros((0,2)))
        # plot on panel 3
        if fg.shape[0] > 0:
            axs[3].scatter(fg[:,0], fg[:,1], s=30, marker='o', edgecolors='white', facecolors='none', linewidths=1.5)
        if bg.shape[0] > 0:
            axs[3].scatter(bg[:,0], bg[:,1], s=30, marker='x', c='white', linewidths=1.5)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# ---------- MAIN PIPE ----------
def main():
    ap = argparse.ArgumentParser(description="MedSAM point-prompt inference on a few slices with Dice + viz.")
    ap.add_argument("--imgs_dir", default=DEFAULT_IMGS_DIR, type=str)
    ap.add_argument("--gts_dir",  default=DEFAULT_GTS_DIR,  type=str)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT, type=str)
    ap.add_argument("--model", default="vit_b", choices=["vit_b","vit_l","vit_h"])
    ap.add_argument("--num_slices", default=5, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    ap.add_argument("--show", action="store_true", help="Show matplotlib windows.")
    # point sampling controls
    ap.add_argument("--fg_points", default=1, type=int, help="Foreground points per label (per slice).")
    ap.add_argument("--bg_points", default=0, type=int, help="Background points per label (per slice).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load MedSAM
    print(f"Loading MedSAM ({args.model}) from: {args.checkpoint}")
    sam = sam_model_registry[args.model](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    # Pick slices
    slice_basenames = pick_slices(args.imgs_dir, args.gts_dir, args.num_slices, args.seed)
    print(f"Evaluating {len(slice_basenames)} slices with {args.fg_points} FG and {args.bg_points} BG point(s) per class.")

    # Aggregate stats
    overall_per_class = {lid: [] for lid in LABELS.keys()}

    for base in slice_basenames:
        img = np.load(os.path.join(args.imgs_dir, base + ".npy"))      # (H,W,3) float [0,1]
        gt  = np.load(os.path.join(args.gts_dir,  base + ".npy"))      # (H,W)   uint8

        # Prepare image for predictor
        rgb_u8 = to_uint8_rgb(img)
        predictor.set_image(rgb_u8)

        H, W, _ = rgb_u8.shape
        pred = np.zeros((H, W), dtype=np.uint8)
        clicks_vis = {}

        # For each label, sample points and predict; merge into a single multiclass mask
        for lid in LABELS.keys():
            pts, labels, vis = sample_points_for_label(gt, lid, k_pos=args.fg_points, k_bg=args.bg_points, rng=rng)
            clicks_vis[lid] = vis
            if pts is None or pts.shape[0] == 0:
                continue

            # SamPredictor expects (N,2) coords in (x,y) and (N,) labels in {1 (FG), 0 (BG)}
            masks, scores, _ = predictor.predict(
                point_coords=pts.astype(np.float32),      # (N,2)
                point_labels=labels.astype(np.int32),     # (N,)
                multimask_output=False
            )
            # masks shape: (1,H,W) bool
            pred[masks[0]] = lid

        # Dice per class & macro
        dice_per_class = {}
        valid = []
        for lid in LABELS.keys():
            d = dice_score(gt, pred, lid)
            dice_per_class[lid] = d
            if not np.isnan(d):
                valid.append(d)
                overall_per_class[lid].append(d)
        macro = float(np.mean(valid)) if valid else float("nan")

        # Log
        parts = []
        for lid in LABELS:
            name_short = LABELS[lid].split()[0]
            val = dice_per_class[lid]
            parts.append(f"{name_short}={(val if not np.isnan(val) else float('nan')):.3f}")
        print(f"{base}: macroDice={macro:.3f} " + " ".join(parts))

        # Viz
        if args.show:
            title = f"{base} | macroDice={macro:.3f} | points FG={args.fg_points}, BG={args.bg_points}"
            visualize(img, gt, pred, clicks_vis, title=title)

    # Overall summary
    print("\n=== Overall (point prompts) ===")
    overall_macro = []
    for lid, vals in overall_per_class.items():
        if len(vals) > 0:
            m = float(np.mean(vals))
            print(f"  {lid}: {LABELS[lid]}  Dice={m:.3f} (n={len(vals)})")
            overall_macro.append(m)
        else:
            print(f"  {lid}: {LABELS[lid]}  Dice=nan (n=0)")
    if overall_macro:
        print(f"  Macro over present classes: {np.mean(overall_macro):.3f}")
    else:
        print("  Macro over present classes: nan")

if __name__ == "__main__":
    main()
