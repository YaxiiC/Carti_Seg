# -*- coding: utf-8 -*-
"""
MedSAM inference on prepared .npy slices with MIXED prompts:
  - Boxes for ALL labels (1..5)
  - PLUS a PARTIAL MASK prompt for cartilage labels (2, 4, 5)

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
from skimage import transform

# ---------- CONFIG DEFAULTS ----------
DEFAULT_IMGS_DIR = r"C:\Users\chris\MICCAI2026\MRI_Knee_OAIZIBCM\imgs"
DEFAULT_GTS_DIR  = r"C:\Users\chris\MICCAI2026\MRI_Knee_OAIZIBCM\gts"
DEFAULT_CKPT     = r"C:\Users\chris\MICCAI2026\MedSAM\work_dir\MedSAM\medsam_vit_b.pth"

# OAIZIB-CM labels
LABELS = {
    1: "Femur",
    2: "Femoral Cartilage",
    3: "Tibia",
    4: "Medial Tibial Cartilage",
    5: "Lateral Tibial Cartilage",
}
CARTILAGE_LABELS = [2, 4, 5]

# Colors for viz (0..5)
LABEL_COLORS = [
    (0.0, 0.0, 0.0, 0.0),   # 0 background
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
    cands = [f[:-4] for f in os.listdir(img_dir) if f.endswith(".npy")]
    cands = [b for b in cands if os.path.exists(os.path.join(gt_dir, b + ".npy"))]
    if not cands:
        raise FileNotFoundError(f"No matching (img, gt) .npy pairs under {img_dir} and {gt_dir}")
    rng.shuffle(cands)
    return cands[:num_slices]

def to_uint8_rgb(img_float01):
    x = np.clip(img_float01, 0, 1) * 255.0
    return x.astype(np.uint8)

def bbox_from_label(mask2d, label_id, pad=5):
    ys, xs = np.where(mask2d == label_id)
    if xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    h, w = mask2d.shape
    x0 = int(max(0, x0 - pad)); y0 = int(max(0, y0 - pad))
    x1 = int(min(w - 1, x1 + pad)); y1 = int(min(h - 1, y1 + pad))
    return np.array([x0, y0, x1, y1], dtype=np.int32)

def partial_mask_from_gt(gt, lid, frac=0.3, seed=None):
    """
    Create a partial mask for a given label by randomly keeping a fraction of its pixels.
    Returns a binary (H,W) uint8 mask with 1s on the kept pixels.
    """
    rng = np.random.default_rng(seed)
    H, W = gt.shape
    ys, xs = np.where(gt == lid)
    pmask = np.zeros((H, W), dtype=np.uint8)
    if ys.size == 0:
        return pmask
    k = max(1, int(frac * ys.size))
    sel = rng.choice(ys.size, size=k, replace=False)
    pmask[ys[sel], xs[sel]] = 1
    return pmask

def lowres_mask_logits_from_binary(mask2d, target=256, logit_scale=10.0):
    """
    Return mask_input as (B, H, W) = (1, 256, 256) float32 logits.
    The predictor will add the channel dim internally.
    """
    low = transform.resize(
        mask2d.astype(np.uint8),
        (target, target),
        order=0, preserve_range=True, mode="constant", anti_aliasing=False
    ).astype(np.float32)
    logits = (low * 2.0 - 1.0) * logit_scale      # (256,256)
    logits = torch.from_numpy(logits[None, ...])  # (1,256,256)  <-- no channel dim
    return logits


def dice_score(gt, pr, label_id):
    gt_bin = (gt == label_id); pr_bin = (pr == label_id)
    inter = (gt_bin & pr_bin).sum()
    denom = gt_bin.sum() + pr_bin.sum()
    return (2.0 * inter / denom) if denom > 0 else np.nan

def visualize(img, gt, pred, boxes_per_label, partial_masks, title=""):
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    axs[0].imshow(img)
    axs[0].set_title("Image"); axs[0].axis("off")

    axs[1].imshow(gt, cmap=CMAP, vmin=0, vmax=len(LABEL_COLORS)-1)
    axs[1].set_title("GT"); axs[1].axis("off")

    axs[2].imshow(pred, cmap=CMAP, vmin=0, vmax=len(LABEL_COLORS)-1)
    axs[2].set_title("Pred"); axs[2].axis("off")

    axs[3].imshow(img)
    axs[3].imshow(pred, cmap=CMAP, vmin=0, vmax=len(LABEL_COLORS)-1)
    axs[3].set_title("Overlay + Prompts"); axs[3].axis("off")

    # ---- draw boxes (for bones) ----
    for lid, box in boxes_per_label.items():
        if box is None:
            continue
        x0, y0, x1, y1 = box
        axs[3].plot([x0, x1, x1, x0, x0],
                    [y0, y0, y1, y1, y0],
                    'w-', linewidth=1.2)

    # ---- overlay partial mask regions (for cartilage) ----
    for lid, pm in partial_masks.items():
        if pm is None:
            continue
        if np.any(pm > 0):
            # light pink overlay for mask area
            axs[3].imshow(
                np.ma.masked_where(pm == 0, pm),
                cmap=ListedColormap([(1.0, 0.75, 0.8, 0.4)]),  # RGBA: lightpink with transparency
                interpolation='none'
            )
            # optional contour outline for visibility
            axs[3].contour(pm, levels=[0.5], colors='white', linewidths=0.8)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser(description="MedSAM: boxes for all, +partial mask prompt for cartilage.")
    ap.add_argument("--imgs_dir", default=DEFAULT_IMGS_DIR, type=str)
    ap.add_argument("--gts_dir",  default=DEFAULT_GTS_DIR,  type=str)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT, type=str)
    ap.add_argument("--model", default="vit_b", choices=["vit_b","vit_l","vit_h"])
    ap.add_argument("--num_slices", default=5, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    ap.add_argument("--show", action="store_true")
    # prompts
    ap.add_argument("--bbox_pad", default=5, type=int, help="Padding pixels around bbox for all labels.")
    ap.add_argument("--mask_frac", default=0.3, type=float, help="Fraction of cartilage pixels to keep as partial mask.")
    ap.add_argument("--mask_logit_scale", default=10.0, type=float, help="Logit magnitude for mask_input (+/- scale).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load MedSAM
    print(f"Loading MedSAM ({args.model}) from: {args.checkpoint}")
    sam = sam_model_registry[args.model](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    # Pick slices
    bases = pick_slices(args.imgs_dir, args.gts_dir, args.num_slices, args.seed)
    print(f"Evaluating {len(bases)} slice(s). Boxes for all labels; partial-mask prompts for cartilage {CARTILAGE_LABELS}.")

    overall_per_class = {lid: [] for lid in LABELS.keys()}

    for base in bases:
        img = np.load(os.path.join(args.imgs_dir, base + ".npy"))  # (H,W,3) float [0,1]
        gt  = np.load(os.path.join(args.gts_dir,  base + ".npy"))  # (H,W)   uint8

        rgb_u8 = to_uint8_rgb(img)
        predictor.set_image(rgb_u8)

        H, W, _ = rgb_u8.shape
        pred = np.zeros((H, W), dtype=np.uint8)

        boxes_vis = {}
        partial_masks_vis = {}

        for lid in LABELS.keys():
            # Build bbox for every label
            box = bbox_from_label(gt, lid, pad=args.bbox_pad)
            boxes_vis[lid] = box
            if box is None:
                continue

            # For cartilage: build a partial mask prompt
            mask_input = None
            if lid in CARTILAGE_LABELS:
                pmask = partial_mask_from_gt(gt, lid, frac=args.mask_frac, seed=int(rng.integers(1<<31)))
                partial_masks_vis[lid] = pmask
                if pmask.sum() > 0:
                    logits = lowres_mask_logits_from_binary(pmask, target=256, logit_scale=args.mask_logit_scale)
                    mask_input = logits.to(device=args.device, dtype=torch.float32)  # (1,256,256)

            else:
                partial_masks_vis[lid] = None

            # Build kwargs and predict: (one box + optional mask_input)
            kwargs = {"box": box[None, :], "multimask_output": False}
            if mask_input is not None:
                kwargs["mask_input"] = mask_input  # (1,1,256,256) float logits

            masks, scores, _ = predictor.predict(**kwargs)
            pred[masks[0]] = lid

        # Dice per class & macro
        per_class = {}
        valid = []
        for lid in LABELS.keys():
            d = dice_score(gt, pred, lid)
            per_class[lid] = d
            if not np.isnan(d):
                valid.append(d)
                overall_per_class[lid].append(d)
        macro = float(np.mean(valid)) if valid else float("nan")

        # Log
        brief = lambda s: s.split()[0]
        parts = [f"{brief(LABELS[lid])}={(per_class[lid] if not np.isnan(per_class[lid]) else float('nan')):.3f}" for lid in LABELS]
        print(f"{base}: macroDice={macro:.3f} " + " ".join(parts))

        if args.show:
            title = f"{base} | macroDice={macro:.3f} | partial-mask frac={args.mask_frac}"
            visualize(img, gt, pred, boxes_vis, partial_masks_vis, title=title)

    # Overall summary
    print("\n=== Overall (boxes for all + partial mask for cartilage) ===")
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
