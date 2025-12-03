"""Evaluation script for the NURBS cartilage model.

This utility mirrors the training pipeline in ``nurbs_training.py``:
- loads the best checkpoint saved during training,
- runs inference on a test split,
- voxelizes the predicted NURBS surface back into a binary mask,
- compares the mask with the ground-truth segmentation, and
- reports common segmentation metrics (Dice and Hausdorff distance).

The implementation intentionally avoids external geometry packages so that
it can run in restricted environments. Voxelization is approximated by
projecting predicted surface samples onto the voxel grid and applying a
small morphological closing to build a solid mask.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from nurbs_training import (
    CartilageDataset,
    CartilageLoss,
    CartilageUNet,
    MultiPatchNURBSTemplate,
    NURBSTemplate,
    default_template_paths,
    parse_roi,
)


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state", ckpt)
    # Handle DataParallel checkpoints
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return ckpt


def _points_to_mask(
    points: np.ndarray,
    volume_shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float],
    dilation_iters: int = 5,
    #erosion_iters: int = 5,
    fill_solid: bool = False,
) -> np.ndarray:
    """Rasterize a point cloud onto a voxel grid and apply light smoothing."""

    mask = np.zeros(volume_shape, dtype=np.uint8)
    if points.size == 0:
        return mask

    # Convert physical coordinates to voxel indices (z, y, x)
    coords = np.round(points / np.asarray(spacing)).astype(int)
    z_idx, y_idx, x_idx = coords.T
    valid = (
        (z_idx >= 0)
        & (z_idx < volume_shape[0])
        & (y_idx >= 0)
        & (y_idx < volume_shape[1])
        & (x_idx >= 0)
        & (x_idx < volume_shape[2])
    )
    mask[z_idx[valid], y_idx[valid], x_idx[valid]] = 1

    # Morphological closing implemented with torch convolutions to avoid SciPy
    tensor = torch.from_numpy(mask[None, None].astype(np.float32))
    kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32)
    for _ in range(dilation_iters):
        tensor = (F.conv3d(tensor, kernel, padding=1) > 0).float()
    for _ in range(dilation_iters):
        tensor = (F.conv3d(tensor, kernel, padding=1) >= kernel.numel()).float()
    shell = tensor.squeeze().numpy().astype(np.uint8)
    if fill_solid:
        return _fill_solid(shell)
    return shell


def _fill_solid(mask: np.ndarray) -> np.ndarray:
    """Fill the interior of a closed surface mask using a 3D flood fill."""

    depth, height, width = mask.shape
    outside = np.zeros_like(mask, dtype=bool)
    q: deque[Tuple[int, int, int]] = deque()

    def enqueue_if_background(z: int, y: int, x: int) -> None:
        if mask[z, y, x] == 0 and not outside[z, y, x]:
            outside[z, y, x] = True
            q.append((z, y, x))

    for z in (0, depth - 1):
        for y in range(height):
            for x in range(width):
                enqueue_if_background(z, y, x)
    for y in (0, height - 1):
        for z in range(depth):
            for x in range(width):
                enqueue_if_background(z, y, x)
    for x in (0, width - 1):
        for z in range(depth):
            for y in range(height):
                enqueue_if_background(z, y, x)

    neighbors = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ]

    while q:
        z, y, x = q.popleft()
        for dz, dy, dx in neighbors:
            nz, ny, nx = z + dz, y + dy, x + dx
            if 0 <= nz < depth and 0 <= ny < height and 0 <= nx < width:
                if mask[nz, ny, nx] == 0 and not outside[nz, ny, nx]:
                    outside[nz, ny, nx] = True
                    q.append((nz, ny, nx))

    filled = mask.copy()
    filled[(mask == 0) & (~outside)] = 1
    return filled.astype(np.uint8)


def _compute_bbox(mask: np.ndarray, margin: int) -> Tuple[int, int, int, int, int, int]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return (
            0,
            mask.shape[0] - 1,
            0,
            mask.shape[1] - 1,
            0,
            mask.shape[2] - 1,
        )

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    z_min = max(z_min - margin, 0)
    y_min = max(y_min - margin, 0)
    x_min = max(x_min - margin, 0)

    z_max = min(z_max + margin, mask.shape[0] - 1)
    y_max = min(y_max + margin, mask.shape[1] - 1)
    x_max = min(x_max + margin, mask.shape[2] - 1)

    return z_min, z_max, y_min, y_max, x_min, x_max


def _crop_with_bbox(array: np.ndarray, bbox: Tuple[int, int, int, int, int, int]) -> np.ndarray:
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    return array[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    return 2.0 * intersection / denom if denom > 0 else 0.0


def _chunked_min_dist(
    src: torch.Tensor, dst: torch.Tensor, chunk_size: int = 4096
) -> torch.Tensor:
    """Return per-point min distances from ``src`` to ``dst`` using chunked cdist.

    Computing a full pairwise distance matrix is prohibitively memory intensive for
    dense masks (hundreds of thousands of voxels). Chunking both dimensions keeps
    intermediate tensors small while producing the same result as a full cdist.
    """

    min_dists: torch.Tensor | None = None
    for dst_start in range(0, dst.shape[0], chunk_size):
        dst_chunk = dst[dst_start : dst_start + chunk_size]
        dist_chunk = torch.cdist(src, dst_chunk)
        min_chunk = dist_chunk.min(dim=1).values
        if min_dists is None:
            min_dists = min_chunk
        else:
            min_dists = torch.minimum(min_dists, min_chunk)
    return min_dists if min_dists is not None else torch.empty(0)


def _directed_hausdorff(pred_t: torch.Tensor, tgt_t: torch.Tensor) -> float:
    """Compute directed Hausdorff distance from ``pred_t`` to ``tgt_t``."""

    if tgt_t.numel() == 0:
        return float("inf")

    min_dists = _chunked_min_dist(pred_t, tgt_t)
    return float(min_dists.max().item()) if min_dists.numel() > 0 else float("inf")


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute the symmetric Hausdorff distance between two binary masks."""

    pred_pts = np.argwhere(pred > 0)
    tgt_pts = np.argwhere(target > 0)
    if pred_pts.size == 0 or tgt_pts.size == 0:
        return float("inf")

    pred_t = torch.from_numpy(pred_pts.astype(np.float32))
    tgt_t = torch.from_numpy(tgt_pts.astype(np.float32))

    forward = _directed_hausdorff(pred_t, tgt_t)
    backward = _directed_hausdorff(tgt_t, pred_t)
    return max(forward, backward)


def _build_model(template: NURBSTemplate | MultiPatchNURBSTemplate, predict_weights: bool, device: torch.device) -> CartilageUNet:
    model = CartilageUNet(
        in_channels=1,
        n_ctrl=template.n_ctrl,
        predict_weights=predict_weights,
        base_channels=16,
    ).to(device)
    return model


def _evaluate_single(
    model: CartilageUNet,
    template: NURBSTemplate | MultiPatchNURBSTemplate,
    volume_path: Path,
    seg_path: Path,
    roi_label: int,
    roi_name: str,
    margin: int,
    device: torch.device,
    vis_out_dir: Path | None = None,
    vis_num_slices: int = 3,
    vis_full_volume: bool = False,
    vis_case_idx: int | None = None,
) -> Tuple[float, float]:
    vol_img = nib.load(str(volume_path))
    seg_img = nib.load(str(seg_path))

    volume = vol_img.get_fdata().astype(np.float32)
    seg = seg_img.get_fdata().astype(np.int64)
    mask = (seg == roi_label).astype(np.uint8)
    spacing = vol_img.header.get_zooms()

    bbox = _compute_bbox(mask, margin)
    volume_crop = _crop_with_bbox(volume, bbox)
    mask_crop = _crop_with_bbox(mask, bbox)
    volume_tensor = torch.from_numpy(volume_crop)[None, None].to(device)

    model.eval()
    with torch.no_grad():
        delta_p, delta_w = model(volume_tensor)
        pred_points, _ = template.evaluate(delta_p, delta_w)

    # Move to CPU and drop batch dimension
    pred_points_np = pred_points.squeeze(0).cpu().numpy()

    fill_solid = roi_name in ("femur", "tibia")
    pred_mask = _points_to_mask(
        pred_points_np, volume_crop.shape, spacing, dilation_iters=13, fill_solid=fill_solid
    )

    dsc = dice_score(pred_mask, mask_crop)
    hd = hausdorff_distance(pred_mask, mask_crop)

    if vis_out_dir is not None:
        _save_case_visualizations(
            idx=vis_case_idx,
            vol_path=volume_path,
            vis_out_dir=vis_out_dir,
            volume=volume,
            volume_crop=volume_crop,
            mask_crop=mask_crop,
            pred_mask=pred_mask,
            affine=vol_img.affine,
            bbox=bbox,
            vis_num_slices=vis_num_slices,
            vis_full_volume=vis_full_volume,
        )

    return dsc, hd


def _save_case_visualizations(
    idx: int | None,
    vol_path: Path,
    vis_out_dir: Path,
    volume: np.ndarray,
    volume_crop: np.ndarray,
    mask_crop: np.ndarray,
    pred_mask: np.ndarray,
    affine: np.ndarray,
    bbox: Tuple[int, int, int, int, int, int],
    vis_num_slices: int,
    vis_full_volume: bool,
) -> None:
    vis_out_dir.mkdir(parents=True, exist_ok=True)
    case_prefix = f"vis_case_{idx:03d}" if idx is not None else "vis_case"
    case_vis_dir = vis_out_dir / f"{case_prefix}_{vol_path.stem}"
    case_vis_dir.mkdir(exist_ok=True)

    nib.save(nib.Nifti1Image(mask_crop.astype(np.uint8), affine), case_vis_dir / "gt_mask_crop.nii.gz")
    nib.save(nib.Nifti1Image(pred_mask.astype(np.uint8), affine), case_vis_dir / "pred_mask_crop.nii.gz")

    if vis_full_volume:
        z_min, z_max, y_min, y_max, x_min, x_max = bbox
        pred_full = np.zeros_like(volume, dtype=np.uint8)
        gt_full = np.zeros_like(volume, dtype=np.uint8)
        pred_full[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1] = pred_mask
        gt_full[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1] = mask_crop
        nib.save(nib.Nifti1Image(pred_full, affine), case_vis_dir / "pred_mask_full.nii.gz")
        nib.save(nib.Nifti1Image(gt_full, affine), case_vis_dir / "gt_mask_full.nii.gz")

    depth = volume_crop.shape[0]
    slices = np.linspace(0, depth - 1, num=max(vis_num_slices, 1), dtype=int)
    for k, z in enumerate(np.unique(slices)):
        slice_img = volume_crop[z]
        plt.figure(figsize=(6, 6))
        plt.imshow(slice_img, cmap="gray")
        plt.imshow(np.ma.masked_where(pred_mask[z] == 0, pred_mask[z]), cmap="Greens", alpha=0.4)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(case_vis_dir / f"slice_{k:02d}.png", dpi=150)
        plt.close()

    if idx is not None:
        print(f"Saved visualizations for case {idx:03d} to {case_vis_dir}")
    else:
        print(f"Saved visualizations to {case_vis_dir}")


def evaluate_model(
    checkpoint_path: Path,
    template_paths: Sequence[Path],
    test_pairs: Iterable[Tuple[Path, Path]],
    roi_name: str,
    roi_label: int = 2,
    margin: int = 8,
    predict_weights: bool = False,
    device: torch.device | None = None,
    vis_out_dir: Path | None = None,
    vis_num_slices: int = 3,
    vis_full_volume: bool = False,
) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(template_paths) == 1:
        template: NURBSTemplate | MultiPatchNURBSTemplate = NURBSTemplate.from_npz(template_paths[0], device)
    else:
        template = MultiPatchNURBSTemplate.from_paths(template_paths, device)

    model = _build_model(template, predict_weights, device)
    ckpt = _load_checkpoint(model, checkpoint_path, device)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} with val metrics: {ckpt.get('val_metrics', {})}")

    dices: List[float] = []
    hds: List[float] = []
    test_pairs = list(test_pairs)

    for idx, (vol_path, seg_path) in enumerate(test_pairs):
        dsc, hd = _evaluate_single(
            model,
            template,
            vol_path,
            seg_path,
            roi_label,
            roi_name,
            margin,
            device,
            vis_out_dir=vis_out_dir,
            vis_num_slices=vis_num_slices,
            vis_full_volume=vis_full_volume,
            vis_case_idx=idx,
        )
        dices.append(dsc)
        hds.append(hd)
        print(f"Case {idx:03d} ({vol_path.stem}): Dice={dsc:.4f}, HD={hd:.2f} vox")

    if dices:
        print("\nAverage metrics across test set:")
        print(f"Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}")
        print(f"Hausdorff: {np.mean(hds):.2f} ± {np.std(hds):.2f} vox")

        if vis_out_dir is not None:
            vis_out_dir.mkdir(parents=True, exist_ok=True)

            plt.figure()
            plt.hist(dices, bins=min(len(dices), 20))
            plt.title("Dice Score Distribution")
            plt.xlabel("Dice score")
            plt.ylabel("Count")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(vis_out_dir / "dice_hist.png", dpi=150)
            plt.close()

            plt.figure()
            plt.scatter(dices, hds)
            plt.title("Dice vs. Hausdorff Distance")
            plt.xlabel("Dice")
            plt.ylabel("HD (vox)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(vis_out_dir / "dice_vs_hd.png", dpi=150)
            plt.close()
    else:
        print("No test volumes found—nothing to evaluate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NURBS cartilage/bone model on a held-out test set.")
    parser.add_argument(
        "--roi",
        type=str,
        default="2",
        help="ROI id or anatomical name (must match the trained model).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the trained checkpoint (defaults to <roi_name>_best_model.pth).",
    )
    parser.add_argument(
        "--templates",
        type=Path,
        nargs="+",
        default=None,
        help="NURBS template npz file(s); derived from ROI name if omitted.",
    )
    parser.add_argument(
        "--n-patches",
        type=int,
        default=2,
        help="Number of longitudinal patches when using default template paths.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "OAI-ZIB-CM-ICP" / "aligned",
        help="Root folder containing imagesTs/ and labelsTs/ splits.",
    )
    parser.add_argument("--roi-label", type=int, default=None, help="Segmentation label to evaluate.")
    parser.add_argument("--margin", type=int, default=8, help="Padding around the ROI bounding box.")
    parser.add_argument(
        "--predict-weights", action="store_true", help="Set if the checkpoint was trained with weight offsets."
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id to use (e.g. 0, 1, ...).",
    )
    parser.add_argument(
        "--vis-out-dir",
        type=Path,
        default=None,
        help="Directory to write visualization artifacts (one subfolder per case).",
    )
    parser.add_argument(
        "--vis-num-slices",
        type=int,
        default=3,
        help="Number of axial slices through the cropped volume to visualize per case.",
    )
    parser.add_argument(
        "--vis-full-volume",
        action="store_true",
        help="Reconstruct and save full-volume NIfTI masks alongside cropped masks.",
    )
    args = parser.parse_args()

    # create device from gpu id
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    roi_id, roi_name = parse_roi(args.roi)

    images_dir = args.data_root / "imagesTs"
    labels_dir = args.data_root / "labelsTs"
    volume_paths = sorted(images_dir.glob("*.nii.gz"))
    seg_paths = sorted(labels_dir.glob("*.nii.gz"))
    pairs = list(zip(volume_paths, seg_paths))

    pairs = pairs[:5]

    template_paths = args.templates or default_template_paths(roi_name, n_patches=args.n_patches)
    checkpoint_path = args.checkpoint or Path(f"{roi_name}_best_model.pth")
    roi_label = args.roi_label if args.roi_label is not None else roi_id

    evaluate_model(
        checkpoint_path=checkpoint_path,
        template_paths=template_paths,
        test_pairs=pairs,
        roi_name=roi_name,
        roi_label=roi_label,
        margin=args.margin,
        predict_weights=args.predict_weights,
        device=device,
        vis_out_dir=args.vis_out_dir,
        vis_num_slices=args.vis_num_slices,
        vis_full_volume=args.vis_full_volume,
    )
