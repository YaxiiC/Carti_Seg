r"""raining utilities for femoral cartilage NURBS prediction with PyTorch.

This module contains:
- CartilageUNet: a lightweight 3D U-Net producing control-point displacements.
- NURBSTemplate / MultiPatchNURBSTemplate: helpers to load template NURBS parameters
  and evaluate surfaces differentiably on a fixed (u, v) grid.
- Geometry losses: chamfer distance, normal consistency, Laplacian regularization,
  and optional thickness constraints for double-surface models.
- Dataset and training skeleton for pairing MRI volumes with NURBS supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure


# 固定的长轴方向（与 fit_nurbs_from_central_template_2patch.py 保持一致），用于两 patch 模板的排列
PC_LONG = np.array([0.0, 1.0, 0.0], dtype=np.float64)


# -----------------------------
# 3D U-Net backbone
# -----------------------------


def conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    """Two-layer 3D convolutional block with InstanceNorm and ReLU."""

    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.InstanceNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = conv_block(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad in case of odd input dimensions
        diff_d = skip.size(2) - x.size(2)
        diff_h = skip.size(3) - x.size(3)
        diff_w = skip.size(4) - x.size(4)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2, diff_d // 2, diff_d - diff_d // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class CartilageUNet(nn.Module):
    """Simple 3D U-Net producing control-point displacements (and optional weight updates)."""

    def __init__(
        self,
        in_channels: int,
        n_ctrl: int,
        predict_weights: bool = False,
        base_channels: int = 32,
    ):
        super().__init__()
        self.predict_weights = predict_weights
        self.inc = conv_block(in_channels, base_channels)
        self.down1 = conv_block(base_channels, base_channels * 2, stride=2)
        self.down2 = conv_block(base_channels * 2, base_channels * 4, stride=2)
        self.down3 = conv_block(base_channels * 4, base_channels * 8, stride=2)

        self.bottleneck = conv_block(base_channels * 8, base_channels * 16, stride=2)

        self.up3 = UpBlock(base_channels * 16, base_channels * 8)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4)
        self.up1 = UpBlock(base_channels * 4, base_channels * 2)
        self.up0 = UpBlock(base_channels * 2, base_channels)

        #ut_dim = n_ctrl * 3 + (n_ctrl if predict_weights else 0)
        #elf.head = nn.Conv3d(base_channels, out_dim, kernel_size=1)
 
        out_dim = n_ctrl * 3 + (n_ctrl if predict_weights else 0)
        self.pool = nn.AdaptiveAvgPool3d(1)   # 全局平均池化到 (B, C, 1,1,1)
        self.head = nn.Linear(base_channels, out_dim)

        self.n_ctrl = n_ctrl

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        xb = self.bottleneck(x3)

        x = self.up3(xb, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)

        x = self.pool(x)            # (B, C, 1,1,1)
        x = x.view(x.size(0), -1)   # (B, C)
        out = self.head(x)   

        #ut = self.head(x)
        #ut = out.mean(dim=[2, 3, 4])  # global average pooling over spatial dims
        delta_p = out[:, : self.n_ctrl * 3].view(out.size(0), self.n_ctrl, 3)
        if self.predict_weights:
            delta_w = out[:, self.n_ctrl * 3 :].view(out.size(0), self.n_ctrl)
        else:
            delta_w = None
        return delta_p, delta_w


# -----------------------------
# NURBS utilities
# -----------------------------


def bspline_basis_one_dim(knots: torch.Tensor, degree: int, samples: torch.Tensor) -> torch.Tensor:
    """Compute Cox–de Boor B-spline basis functions for one parametric dimension.

    Args:
        knots: Tensor of shape (n_knots,).
        degree: B-spline degree.
        samples: Parameter samples in the knot domain, shape (n_samples,).

    Returns:
        Basis matrix of shape (n_basis, n_samples).
    """

    device = samples.device
    n_basis = knots.numel() - degree - 1
    # Zeroth-degree basis
    basis = []
    for i in range(n_basis):
        left = knots[i]
        right = knots[i + 1]
        basis.append(((samples >= left) & (samples < right)).to(torch.float32))
    basis[-1] = torch.maximum(basis[-1], (samples == knots[-1]).to(torch.float32))
    basis = torch.stack(basis, dim=0)

    for k in range(1, degree + 1):
        next_basis = torch.zeros_like(basis)
        for i in range(n_basis):
            denom1 = knots[i + k] - knots[i]
            denom2 = knots[i + k + 1] - knots[i + 1]
            term1 = 0.0
            term2 = 0.0
            if denom1 > 0:
                term1 = (samples - knots[i]) / denom1 * basis[i]
            if denom2 > 0 and i + 1 < n_basis:
                term2 = (knots[i + k + 1] - samples) / denom2 * basis[i + 1]
            next_basis[i] = term1 + term2
        basis = next_basis
    return basis.to(device)


@dataclass
class NURBSTemplate:
    ctrlpts: torch.Tensor  # (nu, nv, 3)
    knots_u: torch.Tensor
    knots_v: torch.Tensor
    degree_u: int
    degree_v: int
    weights: torch.Tensor  # (nu, nv)
    uv_grid: torch.Tensor  # (num_samples, 2)

    @staticmethod
    def _canonical_uv_grid(
        knots_u: np.ndarray,
        knots_v: np.ndarray,
        degree_u: int,
        degree_v: int,
        uv_grid: Optional[np.ndarray] = None,
        samples_u: int = 48,
        samples_v: int = 48,
    ) -> np.ndarray:
        """Return a meshgrid-flattened UV grid even if the npz lacks it."""

        def _default_samples(knots: np.ndarray, degree: int, n_samples: int) -> np.ndarray:
            u_min = float(knots[degree])
            u_max = float(knots[-degree - 1])
            return np.linspace(u_min, u_max, n_samples, dtype=np.float32)

        if uv_grid is None:
            u = _default_samples(knots_u, degree_u, samples_u)
            v = _default_samples(knots_v, degree_v, samples_v)
        else:
            uv_grid = np.asarray(uv_grid, dtype=np.float32)
            u = np.unique(uv_grid[:, 0])
            v = np.unique(uv_grid[:, 1])

        uu, vv = np.meshgrid(u, v, indexing="ij")
        return np.stack([uu.reshape(-1), vv.reshape(-1)], axis=1)

    @classmethod
    def from_npz(cls, path: Path, device: torch.device) -> "NURBSTemplate":
        data = np.load(path)
        ctrlpts = torch.from_numpy(data["ctrlpts"]).to(torch.float32).to(device)
        weights = torch.from_numpy(data.get("weights", np.ones(ctrlpts.shape[:2], dtype=np.float32))).to(device)
        knots_u_np = data["knots_u"]
        knots_v_np = data["knots_v"]
        knots_u = torch.from_numpy(knots_u_np).to(torch.float32).to(device)
        knots_v = torch.from_numpy(knots_v_np).to(torch.float32).to(device)
        degree_u = int(data["degree_u"])
        degree_v = int(data["degree_v"])
        uv_grid_np = cls._canonical_uv_grid(knots_u_np, knots_v_np, degree_u, degree_v, data.get("uv_grid"))
        uv_grid = torch.from_numpy(uv_grid_np).to(torch.float32).to(device)
        return cls(ctrlpts=ctrlpts, knots_u=knots_u, knots_v=knots_v, degree_u=degree_u, degree_v=degree_v, weights=weights, uv_grid=uv_grid)

    @property
    def n_ctrl(self) -> int:
        return self.ctrlpts.shape[0] * self.ctrlpts.shape[1]

    def laplacian_matrix(self) -> torch.Tensor:
        """Construct a grid Laplacian for control-point regularization."""

        nu, nv = self.ctrlpts.shape[:2]
        n = nu * nv
        L = torch.zeros((n, n), dtype=torch.float32, device=self.ctrlpts.device)
        for i in range(nu):
            for j in range(nv):
                idx = i * nv + j
                neighbors = []
                if i > 0:
                    neighbors.append((i - 1) * nv + j)
                if i < nu - 1:
                    neighbors.append((i + 1) * nv + j)
                if j > 0:
                    neighbors.append(i * nv + j - 1)
                if j < nv - 1:
                    neighbors.append(i * nv + j + 1)
                L[idx, idx] = len(neighbors)
                for n_idx in neighbors:
                    L[idx, n_idx] = -1
        return L

    def _uv_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return sorted unique u/v samples implied by the stored UV grid."""

        u = torch.unique(self.uv_grid[:, 0], sorted=True)
        v = torch.unique(self.uv_grid[:, 1], sorted=True)
        return u, v

    def basis_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pre-compute basis matrices for the stored UV samples."""

        u, v = self._uv_samples()
        basis_u = bspline_basis_one_dim(self.knots_u, self.degree_u, u)
        basis_v = bspline_basis_one_dim(self.knots_v, self.degree_v, v)
        return basis_u, basis_v

    def evaluate(self, delta_p: torch.Tensor, delta_w: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the NURBS surface on the template UV grid.

        Args:
            delta_p: Control-point displacement, shape (B, n_ctrl, 3).
            delta_w: Optional weight corrections, shape (B, n_ctrl).
        Returns:
            points: Evaluated surface points, shape (B, N, 3), where N = len(uv_grid).
            normals: Approximate normals from finite differences, shape (B, N, 3).
        """

        B = delta_p.shape[0]
        nu, nv = self.ctrlpts.shape[:2]
        ctrl = self.ctrlpts.view(1, nu * nv, 3) + delta_p
        ctrl = ctrl.view(B, nu, nv, 3)
        if delta_w is not None:
            w = F.softplus(self.weights.view(1, nu * nv) + delta_w).view(B, nu, nv)
        else:
            w = self.weights.view(1, nu, nv).expand(B, -1, -1)

        basis_u, basis_v = self.basis_matrices()
        basis_u = basis_u.to(ctrl.device)
        basis_v = basis_v.to(ctrl.device)
        su = basis_u.shape[1]
        sv = basis_v.shape[1]

        # Compute tensor-product basis weights for all samples
        bu = basis_u.transpose(0, 1)  # (su, nu)
        bv = basis_v.transpose(0, 1)  # (sv, nv)
        # outer product -> (su, sv, nu, nv)
        outer = bu[:, None, :, None] * bv[None, :, None, :]
        ctrl_w = w.unsqueeze(-1) * ctrl  # (B, nu, nv, 3)

        numerator = torch.einsum("uvij,bijc->buvc", outer, ctrl_w)
        denominator = torch.einsum("uvij,bij->buv", outer, w)
        points = numerator / denominator.unsqueeze(-1)
        points_flat = points.view(B, su * sv, 3)

        # Finite differences for normals on the (su, sv) grid
        pts_grid = points
        du = F.pad(pts_grid[:, 1:, :, :] - pts_grid[:, :-1, :, :], (0, 0, 0, 0, 0, 1))
        dv = F.pad(pts_grid[:, :, 1:, :] - pts_grid[:, :, :-1, :], (0, 0, 0, 1, 0, 0))
        normals = torch.cross(du, dv, dim=-1)
        normals = F.normalize(normals + 1e-8, dim=-1)
        normals_flat = normals.view(B, su * sv, 3)
        return points_flat, normals_flat


class MultiPatchNURBSTemplate:
    def __init__(self, templates: Sequence[NURBSTemplate]):
        self.templates = templates
        # 保持与模板生成时相同的长轴方向，便于后续需要按照 patch 顺序做后处理（如厚度或可视化）
        self.pc_long = torch.from_numpy(PC_LONG).to(torch.float32)

    @classmethod
    def from_paths(cls, paths: Sequence[Path], device: torch.device) -> "MultiPatchNURBSTemplate":
        # 默认按文件名排序，确保 patch0 / patch1 的顺序与固定长轴分割一致
        sorted_paths = sorted(paths)
        templates = [NURBSTemplate.from_npz(p, device) for p in sorted_paths]
        return cls(templates)

    @property
    def n_ctrl(self) -> int:
        return sum(t.n_ctrl for t in self.templates)

    def laplacian_matrix(self) -> torch.Tensor:
        blocks = [t.laplacian_matrix() for t in self.templates]
        return torch.block_diag(*blocks)

    def evaluate(self, delta_p: torch.Tensor, delta_w: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        normals = []
        cursor = 0
        for template in self.templates:
            n = template.n_ctrl
            dp_slice = delta_p[:, cursor : cursor + n, :]
            dw_slice = delta_w[:, cursor : cursor + n] if delta_w is not None else None
            pts, nrm = template.evaluate(dp_slice, dw_slice)
            outputs.append(pts)
            normals.append(nrm)
            cursor += n
        return torch.cat(outputs, dim=1), torch.cat(normals, dim=1)


# -----------------------------
# Geometry losses
# -----------------------------


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer distance between two point clouds.

    Args:
        pred: (B, Np, 3)
        target: (B, Nt, 3)
    Returns:
        Scalar loss.
    """

    dist1 = torch.cdist(pred, target)  # (B, Np, Nt)
    min1, _ = torch.min(dist1, dim=2)
    min2, _ = torch.min(dist1, dim=1)
    loss = min1.mean() + min2.mean()
    return loss


def normal_consistency(pred_normals: torch.Tensor, target_points: torch.Tensor, target_normals: torch.Tensor, pred_points: torch.Tensor) -> torch.Tensor:
    """Normal alignment using nearest neighbors from predicted to target."""

    dist = torch.cdist(pred_points, target_points)
    idx = torch.argmin(dist, dim=2)  # (B, Np)
    gathered = torch.gather(target_normals, 1, idx[..., None].expand(-1, -1, 3))
    alignment = 1.0 - torch.abs((pred_normals * gathered).sum(dim=2))
    return alignment.mean()


def laplacian_regularizer(delta_p: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Discrete Laplacian on control-point displacements."""

    B = delta_p.shape[0]
    n_ctrl = delta_p.shape[1]
    L_expanded = L.view(1, n_ctrl, n_ctrl)
    lap = torch.matmul(L_expanded, delta_p)
    return (lap ** 2).mean()


def thickness_loss(inner_pts: torch.Tensor, outer_pts: torch.Tensor, max_thick: float = 6.0) -> torch.Tensor:
    """Penalize non-positive or overly thick regions between paired surfaces."""

    distances = torch.norm(outer_pts - inner_pts, dim=2)
    penalty = F.relu(-distances) + F.relu(distances - max_thick)
    return penalty.mean()


@dataclass
class CartilageLoss:
    w_chamfer: float = 1.0
    w_normals: float = 0.1
    w_lap: float = 0.01
    w_thick: float = 0.0

    def __call__(
        self,
        pred_points: torch.Tensor,
        pred_normals: torch.Tensor,
        gt_points: torch.Tensor,
        gt_normals: torch.Tensor,
        delta_p: torch.Tensor,
        L: torch.Tensor,
        inner_points: Optional[torch.Tensor] = None,
        outer_points: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        losses["chamfer"] = chamfer_distance(pred_points, gt_points) * self.w_chamfer
        losses["normals"] = normal_consistency(pred_normals, gt_points, gt_normals, pred_points) * self.w_normals
        losses["laplacian"] = laplacian_regularizer(delta_p, L) * self.w_lap
        if self.w_thick > 0 and inner_points is not None and outer_points is not None:
            losses["thickness"] = thickness_loss(inner_points, outer_points) * self.w_thick
        total = sum(losses.values())
        losses["total"] = total
        return losses


# -----------------------------
# Dataset and utilities
# -----------------------------

def sample_surface_from_mask(mask: np.ndarray,
                             spacing: Tuple[float, float, float],
                             num_samples: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a surface point cloud and normals from a binary mask."""
    verts, faces, normals, _ = measure.marching_cubes(
        mask.astype(np.float32), level=0.5, spacing=spacing
    )
    face_normals = normals[faces].mean(axis=1)
    pts = verts
    if pts.shape[0] > num_samples:
        idx = np.random.choice(pts.shape[0], size=num_samples, replace=False)
    else:
        idx = np.random.choice(pts.shape[0], size=num_samples, replace=True)
    sampled_pts = pts[idx]
    sampled_normals = face_normals[idx % face_normals.shape[0]]
    return sampled_pts, sampled_normals


def crop_to_mask_bbox(volume: np.ndarray,
                      mask: np.ndarray,
                      margin: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据 mask 的非零体素做 3D bbox 裁剪，并在三维方向各加 margin 体素。
    返回裁剪后的 (volume_crop, mask_crop)。

    volume, mask 形状: (D, H, W)
    """
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        # 没有 ROI，直接返回原图（不会用于训练）
        return volume, mask

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    z_min = max(z_min - margin, 0)
    y_min = max(y_min - margin, 0)
    x_min = max(x_min - margin, 0)

    z_max = min(z_max + margin, volume.shape[0] - 1)
    y_max = min(y_max + margin, volume.shape[1] - 1)
    x_max = min(x_max + margin, volume.shape[2] - 1)

    volume_crop = volume[z_min : z_max + 1,
                         y_min : y_max + 1,
                         x_min : x_max + 1]
    mask_crop = mask[z_min : z_max + 1,
                     y_min : y_max + 1,
                     x_min : x_max + 1]
    return volume_crop, mask_crop


class CartilageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 pairs: Sequence[Tuple[Path, Path]],
                 roi_label: int = 2,
                 num_samples: int = 4096,
                 margin: int = 8):
        self.pairs = pairs
        self.roi_label = roi_label
        self.num_samples = num_samples
        self.margin = margin

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vol_path, seg_path = self.pairs[idx]
        vol_img = nib.load(str(vol_path))
        seg_img = nib.load(str(seg_path))

        volume = vol_img.get_fdata().astype(np.float32)   # (D, H, W)
        seg = seg_img.get_fdata().astype(np.int64)        # (D, H, W)
        mask = (seg == self.roi_label).astype(np.uint8)

        # ① 先用 mask 做 bbox 裁剪，减小 3D 体的尺寸
        volume_crop, mask_crop = crop_to_mask_bbox(volume, mask, margin=self.margin)

        # ② 从裁剪后的 mask 提取 surface supervision
        pts, nrm = sample_surface_from_mask(
            mask_crop, spacing=vol_img.header.get_zooms(), num_samples=self.num_samples
        )

        # ③ 转 tensor（UNet 输入形状: (C, D, H, W)）
        volume_tensor = torch.from_numpy(volume_crop)[None]  # (1, D, H, W)
        pts_tensor = torch.from_numpy(pts).to(torch.float32)
        nrm_tensor = torch.from_numpy(nrm).to(torch.float32)

        return {
            "volume": volume_tensor,
            "points": pts_tensor,
            "normals": nrm_tensor,
        }



# -----------------------------
# Training skeleton
# -----------------------------


def train_epoch(
    model: CartilageUNet,
    template: NURBSTemplate | MultiPatchNURBSTemplate,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: CartilageLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    metrics: Dict[str, float] = {}
    L = template.laplacian_matrix().to(device)
    for batch in dataloader:
        volume = batch["volume"].to(device)
        gt_points = batch["points"].to(device)
        gt_normals = batch["normals"].to(device)

        optimizer.zero_grad()
        delta_p, delta_w = model(volume)
        pred_points, pred_normals = template.evaluate(delta_p, delta_w)
        losses = loss_fn(pred_points, pred_normals, gt_points, gt_normals, delta_p, L)
        losses["total"].backward()
        optimizer.step()

        for k, v in losses.items():
            metrics.setdefault(k, 0.0)
            metrics[k] += float(v.detach().cpu())
    for k in metrics:
        metrics[k] /= len(dataloader)
    return metrics


@torch.no_grad()
def evaluate_epoch(
    model: CartilageUNet,
    template: NURBSTemplate | MultiPatchNURBSTemplate,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: CartilageLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    metrics: Dict[str, float] = {}
    L = template.laplacian_matrix().to(device)
    for batch in dataloader:
        volume = batch["volume"].to(device)
        gt_points = batch["points"].to(device)
        gt_normals = batch["normals"].to(device)

        delta_p, delta_w = model(volume)
        pred_points, pred_normals = template.evaluate(delta_p, delta_w)
        losses = loss_fn(pred_points, pred_normals, gt_points, gt_normals, delta_p, L)

        for k, v in losses.items():
            metrics.setdefault(k, 0.0)
            metrics[k] += float(v.detach().cpu())
    for k in metrics:
        metrics[k] /= len(dataloader)
    return metrics


@torch.no_grad()
def evaluate_epoch(
    model: CartilageUNet,
    template: NURBSTemplate | MultiPatchNURBSTemplate,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: CartilageLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    metrics: Dict[str, float] = {}
    L = template.laplacian_matrix().to(device)
    for batch in dataloader:
        volume = batch["volume"].to(device)
        gt_points = batch["points"].to(device)
        gt_normals = batch["normals"].to(device)

        delta_p, delta_w = model(volume)
        pred_points, pred_normals = template.evaluate(delta_p, delta_w)
        losses = loss_fn(pred_points, pred_normals, gt_points, gt_normals, delta_p, L)

        for k, v in losses.items():
            metrics.setdefault(k, 0.0)
            metrics[k] += float(v.detach().cpu())
    for k in metrics:
        metrics[k] /= len(dataloader)
    return metrics


def example_training_loop(data_pairs: Iterable[Tuple[Path, Path]], template_paths: Sequence[Path], predict_weights: bool = False, epochs: int = 10) -> None:
    """Minimal runnable training skeleton.

    Args:
        data_pairs: iterable of (volume_path, seg_path) pairs aligned to template space.
        template_paths: list of npz files (single path for single-patch, multiple for multi-patch).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(template_paths) == 1:
        template: NURBSTemplate | MultiPatchNURBSTemplate = NURBSTemplate.from_npz(template_paths[0], device)
    else:
        template = MultiPatchNURBSTemplate.from_paths(template_paths, device)

    model = CartilageUNet(
        in_channels=1,
        n_ctrl=template.n_ctrl,
        predict_weights=predict_weights,
        base_channels=16 # 原来是 32，先减半
    )
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = CartilageLoss()

    dataset = CartilageDataset(list(data_pairs))
    val_size = max(1, int(0.2 * len(dataset))) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = (
        torch.utils.data.random_split(dataset, [train_size, val_size]) if val_size > 0 else (dataset, None)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = (
        torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
        if val_dataset is not None
        else None
    )

    def _format_metrics(metrics: Dict[str, float]) -> str:
        return " | ".join(f"{k}: {v:.4f}" for k, v in sorted(metrics.items())) if metrics else "(no metrics)"

    best_val = float("inf")
    checkpoint_path = Path("best_model_new.pth")

    for epoch in range(epochs):
        train_metrics = train_epoch(model, template, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch:03d} | Train -> {_format_metrics(train_metrics)}")

        if val_loader is not None:
            val_metrics = evaluate_epoch(model, template, val_loader, loss_fn, device)
            val_chamfer = val_metrics.get("chamfer", float("inf"))
            print(f"Epoch {epoch:03d} | Val   -> {_format_metrics(val_metrics)}")

            if val_chamfer < best_val:
                best_val = val_chamfer
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_metrics": val_metrics,
                    },
                    checkpoint_path,
                )
                print(f"Saved new best checkpoint to {checkpoint_path} (val chamfer={val_chamfer:.4f})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train cartilage NURBS model")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of image–label pairs to use (default: all)",
    )
    args = parser.parse_args()

    base = Path.home() / "OAI-ZIB-CM-ICP" / "aligned"
    volume_dir = base / "imagesTr"
    seg_dir    = base / "labelsTr"

    volume_paths = sorted(volume_dir.glob("*.nii.gz"))
    seg_paths    = sorted(seg_dir.glob("*.nii.gz"))
    pairs = list(zip(volume_paths, seg_paths))

    if args.max_samples is not None:
        pairs = pairs[: 100]

    # 默认使用由 fit_nurbs_from_central_template_2patch.py 生成的双 patch 模板
    default_templates = [
        Path("femoral_template_surf_patch0.npz"),
        Path("femoral_template_surf_patch1.npz"),
    ]

    if pairs:
        example_training_loop(
            pairs,
            default_templates,
            predict_weights=True,
            epochs=400,
        )
    else:
        print("No training data found. Populate the aligned/volumes and aligned/labels directories to run training.")

