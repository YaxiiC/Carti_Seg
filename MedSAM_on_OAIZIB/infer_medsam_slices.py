# -*- coding: utf-8 -*-
"""
MedSAM inference on prepared .npy slices + visualization + Dice.

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
    (1.0, 0.0, 0.0, 0.45),  # 1 red
    (0.0, 1.0, 0.0, 0.45),  # 2 green
    (0.0, 0.0, 1.0, 0.45),  # 3 blue
    (1.0, 1.0, 0.0, 0.45),  # 4 yellow
    (1.0, 0.0, 1.0, 0.45),  # 5 magenta
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

def bbox_from_label(mask2d, label_id, pad=5):
    ys, xs = np.where(mask2d == label_id)
    if xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    h, w = mask2d.shape
    x0 = int(max(0, x0 - pad))
    y0 = int(max(0, y0 - pad))
    x1 = int(min(w - 1, x1 + pad))
    y1 = int(min(h - 1, y1 + pad))
    return np.array([x0, y0, x1, y1], dtype=np.int32)

def dice_score(gt, pr, label_id):
    gt_bin = (gt == label_id)
    pr_bin = (pr == label_id)
    inter = (gt_bin & pr_bin).sum()
    s = gt_bin.sum() + pr_bin.sum()
    return (2.0 * inter / s) if s > 0 else np.nan  # NaN if class absent in both

def visualize(img, gt, pred, title=""):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(img)
    axs[0].set_title("Image"); axs[0].axis("off")
    axs[1].imshow(gt, cmap=CMAP, vmin=0, vmax=len(LABEL_COLORS)-1)
    axs[1].set_title("GT"); axs[1].axis("off")
    axs[2].imshow(pred, cmap=CMAP, vmin=0, vmax=len(LABEL_COLORS)-1)
    axs[2].set_title("Pred"); axs[2].axis("off")
    axs[3].imshow(img)
    axs[3].imshow(gt, cmap=CMAP, vmin=0, vmax=len(LABEL_COLORS)-1)
    axs[3].set_title("Overlay (GT)"); axs[3].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# ---------- MAIN PIPE ----------
def main():
    ap = argparse.ArgumentParser(description="MedSAM inference on a few slices with Dice and visualization.")
    ap.add_argument("--imgs_dir", default=DEFAULT_IMGS_DIR, type=str)
    ap.add_argument("--gts_dir",  default=DEFAULT_GTS_DIR,  type=str)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT, type=str)
    ap.add_argument("--model", default="vit_b", choices=["vit_b","vit_l","vit_h"])
    ap.add_argument("--num_slices", default=5, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    ap.add_argument("--show", action="store_true", help="Show matplotlib windows.")
    args = ap.parse_args()

    # Load MedSAM
    print(f"Loading MedSAM ({args.model}) from: {args.checkpoint}")
    sam = sam_model_registry[args.model](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    # Pick slices
    slice_basenames = pick_slices(args.imgs_dir, args.gts_dir, args.num_slices, args.seed)
    print(f"Evaluating {len(slice_basenames)} slices.")

    # Aggregate stats
    per_slice_stats = []
    overall_per_class = {lid: [] for lid in LABELS.keys()}

    for base in slice_basenames:
        img = np.load(os.path.join(args.imgs_dir, base + ".npy"))      # (H,W,3) float [0,1]
        gt  = np.load(os.path.join(args.gts_dir,  base + ".npy"))      # (H,W)   uint8

        # Prepare image for predictor
        rgb_u8 = to_uint8_rgb(img)
        predictor.set_image(rgb_u8)

        # Predict per class via GT-derived bbox and merge
        H, W, _ = rgb_u8.shape
        pred = np.zeros((H, W), dtype=np.uint8)

        for lid in LABELS.keys():
            box = bbox_from_label(gt, lid, pad=5)
            if box is None:
                continue
            masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=False)
            # masks[0] -> boolean (H,W); assign label id where True
            pred[masks[0]] = lid

        # Dice per class & macro (mean over classes present in GT or Pred)
        dice_per_class = {}
        valid_dices = []
        for lid, name in LABELS.items():
            d = dice_score(gt, pred, lid)
            dice_per_class[lid] = d
            if not np.isnan(d):
                valid_dices.append(d)
                overall_per_class[lid].append(d)

        macro = float(np.mean(valid_dices)) if valid_dices else float("nan")
        per_slice_stats.append((base, macro, dice_per_class))

        print(f"{base}: macroDice={macro:.3f} " +
              " ".join([f"{LABELS[lid].split()[0]}={dice_per_class[lid]:.3f}" if not np.isnan(dice_per_class[lid]) else f"{LABELS[lid].split()[0]}=nan"
                        for lid in LABELS]))

        if args.show:
            title = f"{base}  |  macroDice={macro:.3f}"
            visualize(img, gt, pred, title=title)

    # Overall summary
    print("\n=== Overall ===")
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
