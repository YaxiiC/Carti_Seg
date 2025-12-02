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

import matplotlib.pyplot as plt
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
    default_template_paths,
    parse_roi,
    sample_surface_from_mask,
)


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state", ckpt)
    # Handle DataParallel checkpoints
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return ckpt


# NOTE: Predicted NURBS points live in the same physical coordinate system as the
# cropped volume used during training/evaluation. Because ``crop_to_mask_bbox``
# shifts the origin away from the full-volume (0, 0, 0), we must subtract the
# crop origin in millimetres before converting to voxel indices.
def _points_to_mask(
    points: np.ndarray,
    volume_shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float],
    *,
    origin_phys: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    dilation_iters: int = 5,
    fill_solid: bool = False,
) -> np.ndarray:
    """Rasterize a point cloud onto a voxel grid and apply light smoothing.

    ``points`` are expected to live in the same physical coordinate system as the
    cropped volume on which metrics are computed. When the volume is extracted
    via ``crop_to_mask_bbox``, its origin shifts; ``origin_phys`` should encode
    that shift in millimetres so we can convert physical coordinates back to
    voxel indices inside the crop (``(points - origin_phys) / spacing``).
    """

    mask = np.zeros(volume_shape, dtype=np.uint8)
    if points.size == 0:
        return mask

    # Convert physical coordinates to voxel indices (z, y, x)
    coords = np.round((points - np.asarray(origin_phys)) / np.asarray(spacing)).astype(int)
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


def _choose_slices(depth: int, vis_num_slices: int) -> List[int]:
    vis_num_slices = max(1, vis_num_slices)
    if depth == 0:
        return [0]
    candidates = np.linspace(0, depth - 1, num=vis_num_slices).astype(int)
    return sorted(np.unique(candidates).tolist())


def _save_slice_overlays(
    case_dir: Path,
    volume_crop: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    vis_num_slices: int,
) -> None:
    slices = _choose_slices(volume_crop.shape[0], vis_num_slices)
    for idx, z in enumerate(slices):
        plt.figure(figsize=(6, 6))
        plt.imshow(volume_crop[z], cmap="gray")
        plt.imshow(np.ma.masked_where(gt_mask[z] == 0, gt_mask[z]), cmap="Greens", alpha=0.4)
        plt.imshow(np.ma.masked_where(pred_mask[z] == 0, pred_mask[z]), cmap="Reds", alpha=0.4)
        plt.title(f"Slice z={z}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(case_dir / f"slice_{idx:02d}.png", dpi=150)
        plt.close()


def _save_nifti_masks(
    case_dir: Path,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    affine: np.ndarray,
    *,
    pred_name: str = "pred_mask_crop.nii.gz",
    gt_name: str = "gt_mask_crop.nii.gz",
) -> None:
    nib.save(nib.Nifti1Image(pred_mask.astype(np.uint8), affine), case_dir / pred_name)
    nib.save(nib.Nifti1Image(gt_mask.astype(np.uint8), affine), case_dir / gt_name)


def _embed_crop_to_full(
    crop_mask: np.ndarray,
    full_shape: Tuple[int, int, int],
    bbox: Tuple[int, int, int, int, int, int],
) -> np.ndarray:
    full = np.zeros(full_shape, dtype=crop_mask.dtype)
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    full[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1] = crop_mask
    return full


def _save_summary_plots(vis_out_dir: Path, dices: List[float], hds: List[float]) -> None:
    if dices:
        plt.figure()
        plt.hist(dices, bins=10, color="steelblue", edgecolor="black")
        plt.title("Dice score distribution")
        plt.xlabel("Dice")
        plt.ylabel("Count")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(vis_out_dir / "dice_hist.png", dpi=150)
        plt.close()

    if dices and hds and len(dices) == len(hds):
        plt.figure()
        plt.scatter(dices, hds, c="darkred")
        plt.title("Dice vs Hausdorff")
        plt.xlabel("Dice")
        plt.ylabel("Hausdorff (vox)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(vis_out_dir / "dice_vs_hd.png", dpi=150)
        plt.close()


def _run_voxelization_sanity_check() -> None:
    """Simple synthetic check to ensure voxelization is consistent."""

    spacing = (1.0, 1.0, 1.0)
    mask = np.zeros((32, 32, 32), dtype=np.uint8)
    mask[10:22, 12:24, 9:21] = 1
    pts, _ = sample_surface_from_mask(mask, spacing=spacing, num_samples=4096)
    recon = _points_to_mask(pts, mask.shape, spacing, origin_phys=(0.0, 0.0, 0.0), dilation_iters=6, fill_solid=True)
    dsc = dice_score(recon, mask)
    hd = hausdorff_distance(recon, mask)
    assert dsc > 0.95 and hd < 2.0, f"Sanity check failed: Dice={dsc:.3f}, HD={hd:.3f}"


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
    debug: bool = False,
) -> Tuple[
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Tuple[int, int, int, int, int, int],
    Tuple[float, float, float],
    Tuple[int, int, int],
    np.ndarray,
]:
    vol_img = nib.load(str(volume_path))
    seg_img = nib.load(str(seg_path))

    volume = vol_img.get_fdata().astype(np.float32)
    seg = seg_img.get_fdata().astype(np.int64)
    mask = (seg == roi_label).astype(np.uint8)
    spacing = vol_img.header.get_zooms()

    volume_crop, mask_crop, bbox = crop_to_mask_bbox(volume, mask, margin=margin, return_bbox=True)
    volume_tensor = torch.from_numpy(volume_crop)[None, None].to(device)

    model.eval()
    with torch.no_grad():
        delta_p, delta_w = model(volume_tensor)
        pred_points, _ = template.evaluate(delta_p, delta_w)

    # Move to CPU and drop batch dimension
    pred_points_np = pred_points.squeeze(0).cpu().numpy()

    # Bounding box origin in physical units (z, y, x)
    origin_phys = (
        float(bbox[0] * spacing[0]),
        float(bbox[2] * spacing[1]),
        float(bbox[4] * spacing[2]),
    )

    if debug:
        gt_coords = np.argwhere(mask_crop > 0)
        gt_min = gt_coords.min(axis=0) if gt_coords.size else np.array([0, 0, 0])
        gt_max = gt_coords.max(axis=0) if gt_coords.size else np.array(volume_crop.shape) - 1
        print(
            f"[DEBUG] volume_crop shape: {volume_crop.shape}, spacing: {spacing}",
            f"\n[DEBUG] bbox (z_min,z_max,y_min,y_max,x_min,x_max): {bbox}",
            f"\n[DEBUG] pred_points range (min->max per axis, phys): {pred_points_np.min(axis=0)} -> {pred_points_np.max(axis=0)}",
            f"\n[DEBUG] gt mask voxel range (crop idx): {gt_min} -> {gt_max}",
        )

    fill_solid = roi_name in ("femur", "tibia")
    pred_mask = _points_to_mask(
        pred_points_np,
        volume_crop.shape,
        spacing,
        origin_phys=origin_phys,
        dilation_iters=12,
        fill_solid=fill_solid,
    )

    dsc = dice_score(pred_mask, mask_crop)
    hd = hausdorff_distance(pred_mask, mask_crop)
    return dsc, hd, pred_mask, mask_crop, volume_crop, bbox, spacing, volume.shape, vol_img.affine


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
    run_sanity_check: bool = False,
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

    if run_sanity_check:
        _run_voxelization_sanity_check()
        print("Voxelization sanity check passed.")

    if vis_out_dir is not None:
        vis_out_dir.mkdir(parents=True, exist_ok=True)

    dices: List[float] = []
    hds: List[float] = []
    for idx, (vol_path, seg_path) in enumerate(test_pairs):
        (
            dsc,
            hd,
            pred_mask,
            mask_crop,
            volume_crop,
            bbox,
            spacing,
            full_shape,
            affine,
        ) = _evaluate_single(
            model,
            template,
            vol_path,
            seg_path,
            roi_label,
            roi_name,
            margin,
            device,
            debug=(idx == 0),
        )
        dices.append(dsc)
        hds.append(hd)
        print(f"Case {idx:03d} ({vol_path.stem}): Dice={dsc:.4f}, HD={hd:.2f} vox")

        if vis_out_dir is not None:
            case_dir = vis_out_dir / f"vis_case_{idx:03d}_{vol_path.stem}"
            case_dir.mkdir(parents=True, exist_ok=True)
            _save_nifti_masks(case_dir, pred_mask, mask_crop, affine)
            _save_slice_overlays(case_dir, volume_crop, pred_mask, mask_crop, vis_num_slices)

            if vis_full_volume:
                pred_full = _embed_crop_to_full(pred_mask, full_shape, bbox)
                gt_full = _embed_crop_to_full(mask_crop, full_shape, bbox)
                _save_nifti_masks(
                    case_dir,
                    pred_full,
                    gt_full,
                    affine,
                    pred_name="pred_mask_full.nii.gz",
                    gt_name="gt_mask_full.nii.gz",
                )

            print(f"Saved visualizations for case {idx:03d} to {case_dir}")

    if dices:
        print("\nAverage metrics across test set:")
        print(f"Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}")
        print(f"Hausdorff: {np.mean(hds):.2f} ± {np.std(hds):.2f} vox")
    else:
        print("No test volumes found—nothing to evaluate.")

    if vis_out_dir is not None:
        _save_summary_plots(vis_out_dir, dices, hds)


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
        help="If set, save NIfTI masks and slice overlays for each case to this directory.",
    )
    parser.add_argument(
        "--vis-num-slices",
        type=int,
        default=3,
        help="Number of axial slices to visualize per case when saving overlays.",
    )
    parser.add_argument(
        "--vis-full-volume",
        action="store_true",
        help="Also save predicted/GT masks embedded back into the full volume space.",
    )
    parser.add_argument(
        "--run-sanity-check",
        action="store_true",
        help="Run a small voxelization sanity test before evaluating the dataset.",
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
        run_sanity_check=args.run_sanity_check,
    )
