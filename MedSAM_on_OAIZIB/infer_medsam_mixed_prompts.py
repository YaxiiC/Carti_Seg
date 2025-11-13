# -*- coding: utf-8 -*-
"""
MedSAM inference on prepared .npy slices with MIXED prompts:
  - Boxes for ALL labels (1..5)
  - PLUS points for cartilage labels (2, 4, 5)

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

def sample_points_for_label(gt, lid, k_pos=1, k_bg=0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    ys, xs = np.where(gt == lid)
    if ys.size == 0:
        return None, None, {"fg": np.zeros((0,2)), "bg": np.zeros((0,2))}
    sel = rng.choice(ys.size, size=min(k_pos, ys.size), replace=False)
    fg = np.stack([xs[sel], ys[sel]], axis=1).astype(np.float32)  # (x,y)

    bg = np.zeros((0,2), dtype=np.float32)
    if k_bg > 0:
        by, bx = np.where(gt == 0)
        if by.size > 0:
            bsel = rng.choice(by.size, size=min(k_bg, by.size), replace=False)
            bg = np.stack([bx[bsel], by[bsel]], axis=1).astype(np.float32)

    if bg.shape[0] > 0:
        coords = np.concatenate([fg, bg], axis=0)
        labels = np.concatenate([np.ones((fg.shape[0],), dtype=np.int32),
                                 np.zeros((bg.shape[0],), dtype=np.int32)], axis=0)
    else:
        coords = fg
        labels = np.ones((fg.shape[0],), dtype=np.int32)

    return coords, labels, {"fg": fg, "bg": bg}

def dice_score(gt, pr, label_id):
    gt_bin = (gt == label_id); pr_bin = (pr == label_id)
    inter = (gt_bin & pr_bin).sum()
    denom = gt_bin.sum() + pr_bin.sum()
    return (2.0 * inter / denom) if denom > 0 else np.nan

def visualize(img, gt, pred, clicks_per_label, boxes_per_label, title=""):
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    axs[0].imshow(img); axs[0].set_title("Image"); axs[0].axis("off")
    axs[1].imshow(gt, cmap=CMAP, vmin=0, vmax=len(LABEL_COLORS)-1); axs[1].set_title("GT"); axs[1].axis("off")
    axs[2].imshow(pred, cmap=CMAP, vmin=0, vmax=len(LABEL_COLORS)-1); axs[2].set_title("Pred"); axs[2].axis("off")
    axs[3].imshow(img); axs[3].imshow(pred, cmap=CMAP, vmin=0, vmax=len(LABEL_COLORS)-1); axs[3].set_title("Overlay + Prompts"); axs[3].axis("off")

    # draw points
    for lid, pts in clicks_per_label.items():
        fg = pts.get("fg", np.zeros((0,2))); bg = pts.get("bg", np.zeros((0,2)))
        if fg.shape[0] > 0:
            axs[3].scatter(fg[:,0], fg[:,1], s=28, marker='o', edgecolors='white', facecolors='none', linewidths=1.5)
        if bg.shape[0] > 0:
            axs[3].scatter(bg[:,0], bg[:,1], s=28, marker='x', c='white', linewidths=1.5)

    # draw boxes
    for lid, box in boxes_per_label.items():
        if box is None: continue
        x0,y0,x1,y1 = box
        axs[3].plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],'w-', linewidth=1.2)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def box_to_mask(h, w, box):
    """box = [x0,y0,x1,y1]; returns boolean mask inside the box."""
    if box is None:
        return np.ones((h, w), dtype=bool)
    x0, y0, x1, y1 = map(int, box)
    m = np.zeros((h, w), dtype=bool)
    m[max(0,y0):min(h,y1+1), max(0,x0):min(w,x1+1)] = True
    return m


# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser(description="MedSAM: boxes for all, +points for cartilage.")
    ap.add_argument("--imgs_dir", default=DEFAULT_IMGS_DIR, type=str)
    ap.add_argument("--gts_dir",  default=DEFAULT_GTS_DIR,  type=str)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT, type=str)
    ap.add_argument("--model", default="vit_b", choices=["vit_b","vit_l","vit_h"])
    ap.add_argument("--num_slices", default=5, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    ap.add_argument("--show", action="store_true")
    # prompts
    ap.add_argument("--fg_points", default=1, type=int, help="Foreground points per cartilage label.")
    ap.add_argument("--bg_points", default=0, type=int, help="Background points per cartilage label.")
    ap.add_argument("--bbox_pad", default=5, type=int, help="Padding pixels around bbox for all labels.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load MedSAM
    print(f"Loading MedSAM ({args.model}) from: {args.checkpoint}")
    sam = sam_model_registry[args.model](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    # Pick slices
    bases = pick_slices(args.imgs_dir, args.gts_dir, args.num_slices, args.seed)
    print(f"Evaluating {len(bases)} slice(s). Boxes for all labels; points added for {CARTILAGE_LABELS}.")

    overall_per_class = {lid: [] for lid in LABELS.keys()}

    for base in bases:
        img = np.load(os.path.join(args.imgs_dir, base + ".npy"))  # (H,W,3) float [0,1]
        gt  = np.load(os.path.join(args.gts_dir,  base + ".npy"))  # (H,W)   uint8

        rgb_u8 = to_uint8_rgb(img)
        predictor.set_image(rgb_u8)

        H, W, _ = rgb_u8.shape
        pred = np.zeros((H, W), dtype=np.uint8)

        clicks_vis = {}
        boxes_vis = {}

        # For ALL labels: build a bbox
        for lid in LABELS.keys():
            box = bbox_from_label(gt, lid, pad=args.bbox_pad)
            boxes_vis[lid] = box
            if box is None:
                continue

            # For cartilage labels, ALSO sample points and do a combined prompt
            point_coords = None
            point_labels = None
            if lid in CARTILAGE_LABELS:
                pts, labels, vis = sample_points_for_label(gt, lid, k_pos=args.fg_points, k_bg=args.bg_points, rng=rng)
                clicks_vis[lid] = vis
                if pts is not None and pts.shape[0] > 0:
                    point_coords = np.ascontiguousarray(pts.astype(np.float32))   # (N,2)
                    point_labels = np.ascontiguousarray(labels.astype(np.int32)) # (N,)
            else:
                # for non-cartilage, record empty clicks for visualization consistency
                clicks_vis[lid] = {"fg": np.zeros((0,2)), "bg": np.zeros((0,2))}

            # Build kwargs: box always, points only if present
            kwargs = {"box": box[None, :], "multimask_output": False}
            if point_coords is not None:
                kwargs["point_coords"] = point_coords
            if point_labels is not None:
                kwargs["point_labels"] = point_labels

            masks, scores, _ = predictor.predict(**kwargs)
            m = masks[0]                          # (H,W) bool
            m &= box_to_mask(H, W, box)           # <-- hard-clip to the bbox
            pred[m] = lid

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
            title = f"{base} | macroDice={macro:.3f} | FG={args.fg_points} BG={args.bg_points} (+box for all)"
            visualize(img, gt, pred, clicks_vis, boxes_vis, title=title)

    # Overall summary
    print("\n=== Overall (boxes for all + points for cartilage) ===")
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
