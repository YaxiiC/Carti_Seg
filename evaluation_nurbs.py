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
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from nurbs_training import (
    CartilageDataset,
    CartilageLoss,
    CartilageUNet,
    MultiPatchNURBSTemplate,
    NURBSTemplate,
    crop_to_mask_bbox,
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
    dilation_iters: int = 2,
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
    return tensor.squeeze().numpy().astype(np.uint8)


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    return 2.0 * intersection / denom if denom > 0 else 0.0


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute the symmetric Hausdorff distance between two binary masks."""

    pred_pts = np.argwhere(pred > 0)
    tgt_pts = np.argwhere(target > 0)
    if pred_pts.size == 0 or tgt_pts.size == 0:
        return float("inf")

    pred_t = torch.from_numpy(pred_pts.astype(np.float32))
    tgt_t = torch.from_numpy(tgt_pts.astype(np.float32))
    dist = torch.cdist(pred_t, tgt_t)
    forward = dist.min(dim=1).values.max().item()
    backward = dist.min(dim=0).values.max().item()
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
    margin: int,
    device: torch.device,
) -> Tuple[float, float]:
    vol_img = nib.load(str(volume_path))
    seg_img = nib.load(str(seg_path))

    volume = vol_img.get_fdata().astype(np.float32)
    seg = seg_img.get_fdata().astype(np.int64)
    mask = (seg == roi_label).astype(np.uint8)
    spacing = vol_img.header.get_zooms()

    volume_crop, mask_crop = crop_to_mask_bbox(volume, mask, margin=margin)
    volume_tensor = torch.from_numpy(volume_crop)[None, None].to(device)

    model.eval()
    with torch.no_grad():
        delta_p, delta_w = model(volume_tensor)
        pred_points, _ = template.evaluate(delta_p, delta_w)

    # Move to CPU and drop batch dimension
    pred_points_np = pred_points.squeeze(0).cpu().numpy()

    pred_mask = _points_to_mask(pred_points_np, volume_crop.shape, spacing)

    dsc = dice_score(pred_mask, mask_crop)
    hd = hausdorff_distance(pred_mask, mask_crop)
    return dsc, hd


def evaluate_model(
    checkpoint_path: Path,
    template_paths: Sequence[Path],
    test_pairs: Iterable[Tuple[Path, Path]],
    roi_label: int = 2,
    margin: int = 8,
    predict_weights: bool = False,
) -> None:
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
    for idx, (vol_path, seg_path) in enumerate(test_pairs):
        dsc, hd = _evaluate_single(model, template, vol_path, seg_path, roi_label, margin, device)
        dices.append(dsc)
        hds.append(hd)
        print(f"Case {idx:03d} ({vol_path.stem}): Dice={dsc:.4f}, HD={hd:.2f} vox")

    if dices:
        print("\nAverage metrics across test set:")
        print(f"Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}")
        print(f"Hausdorff: {np.mean(hds):.2f} ± {np.std(hds):.2f} vox")
    else:
        print("No test volumes found—nothing to evaluate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NURBS cartilage model on a held-out test set.")
    parser.add_argument("--checkpoint", type=Path, default=Path("best_model.pth"), help="Path to the trained checkpoint.")
    parser.add_argument(
        "--templates",
        type=Path,
        nargs="+",
        default=[
            Path("femoral_template_surf_patch0.npz"),
            Path("femoral_template_surf_patch1.npz"),
        ],
        help="NURBS template npz file(s).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "OAI-ZIB-CM-ICP" / "aligned",
        help="Root folder containing imagesTs/ and labelsTs/ splits.",
    )
    parser.add_argument("--roi-label", type=int, default=2, help="Segmentation label to evaluate.")
    parser.add_argument("--margin", type=int, default=8, help="Padding around the ROI bounding box.")
    parser.add_argument("--predict-weights", action="store_true", help="Set if the checkpoint was trained with weight offsets.")
    args = parser.parse_args()

    images_dir = args.data_root / "imagesTs"
    labels_dir = args.data_root / "labelsTs"
    volume_paths = sorted(images_dir.glob("*.nii.gz"))
    seg_paths = sorted(labels_dir.glob("*.nii.gz"))
    pairs = list(zip(volume_paths, seg_paths))

    evaluate_model(
        checkpoint_path=args.checkpoint,
        template_paths=args.templates,
        test_pairs=pairs,
        roi_label=args.roi_label,
        margin=args.margin,
        predict_weights=args.predict_weights,
    )
