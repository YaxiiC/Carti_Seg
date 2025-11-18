#!/usr/bin/env python3
"""
ffd_dataloader.py  (debuggable)

Run a quick step-by-step test:
  python ffd_dataloader.py --data_root C:/Users/chris/MICCAI2026/OAI-ZIB-CM --split Tr --roi_id 2 --index 0 --max_vertices 5000 --verbose
  python ffd_dataloader.py --data_root C:/Users/chris/MICCAI2026/OAI-ZIB-CM --split Tr --roi_id 2 --index 0 --max_vertices 5000 --verbose
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage.measure import marching_cubes, label as cc_label
import random
import argparse

# ---------------------------
# IO & basic utils
# ---------------------------

def nii_spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    return spacing

def zscore(x: np.ndarray) -> np.ndarray:
    m = float(np.mean(x))
    s = float(np.std(x)) + 1e-8
    return (x - m) / s

def canonical_stem(p: Path) -> str:
    name = p.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    parts = name.split("_")
    if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) == 4:
        name = "_".join(parts[:-1])
    return name

def find_pairs(data_root: Path, split: str = "Tr", verbose: bool = False) -> List[Tuple[Path, Path]]:
    img_dir = data_root / f"images{split}"
    lab_dir = data_root / f"labels{split}"
    imgs = sorted(list(img_dir.glob("*.nii")) + list(img_dir.glob("*.nii.gz")))
    labs = sorted(list(lab_dir.glob("*.nii")) + list(lab_dir.glob("*.nii.gz")))
    labs_by = {canonical_stem(p): p for p in labs}

    pairs = []
    chosen = set()

    # First pass: prefer _0000 if present
    for ip in imgs:
        if ip.name.endswith("_0000.nii") or ip.name.endswith("_0000.nii.gz"):
            stem = canonical_stem(ip)
            lp = labs_by.get(stem)
            if lp is not None:
                pairs.append((ip, lp))
                chosen.add(stem)

    # Second pass: any remaining
    if not pairs:
        for ip in imgs:
            stem = canonical_stem(ip)
            if stem in chosen:
                continue
            lp = labs_by.get(stem)
            if lp is not None:
                pairs.append((ip, lp))

    if verbose:
        print(f"[find_pairs] images dir: {img_dir}")
        print(f"[find_pairs] labels dir: {lab_dir}")
        print(f"[find_pairs] found images={len(imgs)}, labels={len(labs)}, matched pairs={len(pairs)}")
        if pairs[:3]:
            print(f"[find_pairs] sample pairs:")
            for a, b in pairs[:3]:
                print(f"  - {a.name}  <->  {b.name}")

    return pairs

# ---------------------------
# Marching Cubes to surface
# ---------------------------

def roi_mask_from_label(label_np: np.ndarray, roi_id: int, keep_largest_cc: bool = True, verbose: bool = False) -> np.ndarray:
    lab_int = np.round(label_np).astype(np.int32)
    mask = (lab_int == int(roi_id)).astype(np.uint8)
    if verbose:
        vox = int(mask.sum())
        print(f"[roi_mask] roi_id={roi_id}, voxels={vox}, keep_largest_cc={keep_largest_cc}")
    if keep_largest_cc and mask.max() > 0:
        lab = cc_label(mask, connectivity=1)
        if lab.max() > 0:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            k = int(np.argmax(sizes))
            mask = (lab == k).astype(np.uint8)
            if verbose:
                print(f"[roi_mask] largest CC label={k}, size={int(sizes[k])}")
    return mask

def mesh_from_mask(mask: np.ndarray, spacing: np.ndarray, level: float = 0.5, verbose: bool = False):
    if mask.max() == 0:
        if verbose:
            print("[mesh_from_mask] empty mask -> None")
        return None
    if verbose:
        print(f"[mesh_from_mask] marching_cubes level={level}, spacing={spacing.tolist()}")
    v, f, n, _ = marching_cubes(mask, level=level, spacing=spacing)
    if verbose:
        print(f"[mesh_from_mask] raw verts={v.shape[0]}, faces={f.shape[0]}")
    if n is None or len(n) != len(v):
        # Fallback normals
        tri = v[f]
        face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        face_n = face_n / (np.linalg.norm(face_n, axis=1, keepdims=True) + 1e-8)
        n = np.zeros_like(v)
        for i, (a, b, c) in enumerate(f):
            n[a] += face_n[i]; n[b] += face_n[i]; n[c] += face_n[i]
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)
        if verbose:
            print("[mesh_from_mask] normals computed from faces")
    return v.astype(np.float32), f.astype(np.int64), n.astype(np.float32)

def subsample_vertices(verts: np.ndarray, faces: np.ndarray, normals: np.ndarray,
                       max_vertices: Optional[int], seed: int = 0, verbose: bool = False):
    V = verts.shape[0]
    if max_vertices is None or max_vertices >= V:
        if verbose:
            print(f"[subsample] skip (V={V}, max_vertices={max_vertices})")
        return verts, faces, normals
    rng = np.random.default_rng(seed)
    idx_keep = np.sort(rng.choice(V, size=max_vertices, replace=False))
    map_old2new = -np.ones(V, dtype=np.int64)
    map_old2new[idx_keep] = np.arange(idx_keep.size, dtype=np.int64)
    f_kept = []
    drop_faces = 0
    for (a, b, c) in faces:
        na, nb, nc = map_old2new[a], map_old2new[b], map_old2new[c]
        if (na >= 0) and (nb >= 0) and (nc >= 0):
            f_kept.append([na, nb, nc])
        else:
            drop_faces += 1
    f_kept = np.asarray(f_kept, dtype=np.int64) if f_kept else np.zeros((0, 3), dtype=np.int64)
    if verbose:
        print(f"[subsample] V {V} -> {idx_keep.size}, faces {faces.shape[0]} -> {f_kept.shape[0]} (dropped {drop_faces})")
    return verts[idx_keep], f_kept, normals[idx_keep]

# ---------------------------
# Differentiable cubic B-spline FFD
# ---------------------------

def bspline_basis(u: torch.Tensor) -> torch.Tensor:
    u2 = u * u
    u3 = u2 * u
    B0 = (1 - 3 * u + 3 * u2 - u3) / 6.0
    B1 = (4 - 6 * u2 + 3 * u3) / 6.0
    B2 = (1 + 3 * u + 3 * u2 - 3 * u3) / 6.0
    B3 = u3 / 6.0
    return torch.stack([B0, B1, B2, B3], dim=-1)

@dataclass
class LatticeSpec:
    origin: torch.Tensor     # (3,) in mm
    spacing: torch.Tensor    # (3,) control spacing in mm
    size: Tuple[int, int, int]  # (nx, ny, nz)

class BSplineFFD(nn.Module):
    def __init__(self, spec: LatticeSpec, verbose: bool = False):
        super().__init__()
        self.register_buffer("origin", spec.origin.float())
        self.register_buffer("spacing", spec.spacing.float())
        self.nx, self.ny, self.nz = spec.size
        self.verbose = verbose

    def forward(self, verts: torch.Tensor, deltaG: torch.Tensor) -> torch.Tensor:
        if self.verbose:
            print(f"[FFD] verts={tuple(verts.shape)}, deltaG={(self.nx,self.ny,self.nz,3)} origin={self.origin.tolist()} spacing={self.spacing.tolist()}")
        rel = (verts - self.origin[None, :]) / self.spacing[None, :]
        base = torch.floor(rel)
        idx0 = base - 1.0
        u = rel - base
        Bu = bspline_basis(u[:, 0].clamp(0, 1))
        Bv = bspline_basis(u[:, 1].clamp(0, 1))
        Bw = bspline_basis(u[:, 2].clamp(0, 1))
        out = torch.zeros_like(verts)
        for a in range(4):
            ia = (idx0[:, 0] + a).long()
            wa = Bu[:, a][:, None]
            for b in range(4):
                ib = (idx0[:, 1] + b).long()
                wb = Bv[:, b][:, None]
                for c in range(4):
                    ic = (idx0[:, 2] + c).long()
                    wc = Bw[:, c][:, None]
                    w = wa * wb * wc
                    mask = (ia >= 0) & (ia < self.nx) & (ib >= 0) & (ib < self.ny) & (ic >= 0) & (ic < self.nz)
                    if not torch.any(mask):
                        continue
                    dg = deltaG[ia[mask], ib[mask], ic[mask], :]
                    out[mask] += w[mask] * dg
        return verts + out

# ---------------------------
# Losses
# ---------------------------

def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x ** 2).sum(dim=1, keepdim=True)
    y2 = (y ** 2).sum(dim=1, keepdim=True).T
    d2 = x2 + y2 - 2 * (x @ y.T)
    d2 = d2.clamp_min(0.0)
    return d2.min(dim=1)[0].mean() + d2.min(dim=0)[0].mean()

def uniform_laplacian_loss(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    V = verts.shape[0]
    device = verts.device
    adj = [[] for _ in range(V)]
    for a, b, c in faces.tolist():
        adj[a].extend([b, c]); adj[b].extend([a, c]); adj[c].extend([a, b])
    loss = 0.0; cnt = 0
    for i in range(V):
        nbs = list(set(adj[i]))
        if not nbs:
            continue
        nbv = verts[torch.tensor(nbs, device=device)]
        loss += ((verts[i] - nbv.mean(dim=0)) ** 2).sum()
        cnt += 1
    return loss / max(cnt, 1)

def normal_consistency(pred_v: torch.Tensor, pred_f: torch.Tensor,
                       gt_v: torch.Tensor, gt_n: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        d2 = ((pred_v[:, None, :] - gt_v[None, :, :]) ** 2).sum(dim=2)
        idx = torch.argmin(d2, dim=1)
    sel_gt_n = gt_n[idx]
    tri = pred_v[pred_f]
    fn = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    fn = fn / (fn.norm(dim=1, keepdim=True) + 1e-8)
    V = pred_v.shape[0]
    vnorm = torch.zeros_like(pred_v)
    counts = torch.zeros((V, 1), device=pred_v.device)
    for i in range(pred_f.shape[0]):
        a, b, c = pred_f[i]
        n = fn[i]
        vnorm[a] += n; vnorm[b] += n; vnorm[c] += n
        counts[a] += 1; counts[b] += 1; counts[c] += 1
    vnorm = vnorm / (counts + 1e-8)
    vnorm = vnorm / (vnorm.norm(dim=1, keepdim=True) + 1e-8)
    cos = (vnorm * sel_gt_n).sum(dim=1).abs()
    return (1.0 - cos).mean()

def thickness_regularization(inner_v: torch.Tensor, outer_v: torch.Tensor,
                             inner_f: torch.Tensor,
                             tmin: float = 0.0, tmax: float = 6.0) -> torch.Tensor:
    tri = inner_v[inner_f]
    fn = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    fn = fn / (fn.norm(dim=1, keepdim=True) + 1e-8)
    V = inner_v.shape[0]
    vnorm = torch.zeros_like(inner_v)
    counts = torch.zeros((V, 1), device=inner_v.device)
    for i in range(inner_f.shape[0]):
        a, b, c = inner_f[i]
        n = fn[i]
        vnorm[a] += n; vnorm[b] += n; vnorm[c] += n
        counts[a] += 1; counts[b] += 1; counts[c] += 1
    vnorm = vnorm / (counts + 1e-8)
    vnorm = vnorm / (vnorm.norm(dim=1, keepdim=True) + 1e-8)
    d2 = ((inner_v[:, None, :] - outer_v[None, :, :]) ** 2).sum(dim=2)
    j = torch.argmin(d2, dim=1)
    vec = outer_v[j] - inner_v
    dist = (vec * vnorm).sum(dim=1)
    loss_pos = torch.relu(tmin - dist).mean()
    loss_max = torch.relu(dist - tmax).mean()
    return loss_pos + loss_max

# ---------------------------
# Dataset
# ---------------------------

class OAIFFDTemplateDataset(Dataset):
    def __init__(self,
                 data_root: str,
                 split: str,
                 roi_id: int = 2,
                 n_cases: Optional[int] = None,
                 seed: int = 0,
                 marching_level: float = 0.5,
                 keep_largest_cc: bool = True,
                 max_vertices: Optional[int] = 5000,
                 target_shape: Optional[Tuple[int, int, int]] = None,
                 verbose: bool = False):
        self.data_root = Path(data_root)
        self.roi_id = int(roi_id)
        self.marching_level = marching_level
        self.keep_largest_cc = keep_largest_cc
        self.max_vertices = max_vertices
        self.target_shape = tuple(target_shape) if target_shape is not None else None
        self.verbose = verbose
        self.pairs = find_pairs(self.data_root, split=split, verbose=verbose)
        if n_cases is not None:
            random.seed(seed)
            self.pairs = random.sample(self.pairs, min(n_cases, len(self.pairs)))
        self.rng_seed = seed
        if self.verbose:
            resize_str = f", target_shape={self.target_shape}" if self.target_shape is not None else ""
            print(f"[Dataset] init -> n_pairs={len(self.pairs)}, roi_id={self.roi_id}, level={self.marching_level}, max_vertices={self.max_vertices}{resize_str}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_p, lab_p = self.pairs[idx]
        if self.verbose:
            print(f"\n[Sample {idx}] img={img_p.name}, lab={lab_p.name}")

        img = nib.load(str(img_p))
        lab = nib.load(str(lab_p))
        vol = img.get_fdata().astype(np.float32)
        lab_np = lab.get_fdata()
        spacing = nii_spacing_from_affine(img.affine)
        orig_shape = vol.shape

        if self.verbose:
            print(f"[Sample {idx}] volume shape={vol.shape}, spacing(mm)={spacing.tolist()}")

        vol = zscore(vol)
        if self.verbose:
            print(f"[Sample {idx}] volume z-scored: mean~{float(vol.mean()):.3f}, std~{float(vol.std()):.3f}")

        if self.target_shape is not None:
            vol_t = torch.from_numpy(vol[None, None, ...])
            vol_t = F.interpolate(vol_t, size=self.target_shape, mode="trilinear", align_corners=False)
            vol = vol_t.squeeze(0).squeeze(0).numpy()
            scale = np.array(orig_shape, dtype=np.float32) / np.array(self.target_shape, dtype=np.float32)
            spacing = spacing * scale
            if self.verbose:
                print(f"[Sample {idx}] resized volume to {self.target_shape}, new spacing(mm)={spacing.tolist()}")

        mask = roi_mask_from_label(lab_np, self.roi_id, keep_largest_cc=self.keep_largest_cc, verbose=self.verbose)
        m = mesh_from_mask(mask, spacing, level=self.marching_level, verbose=self.verbose)

        empty = False
        if m is None:
            empty = True
            verts = np.zeros((4, 3), np.float32)
            faces = np.array([[0, 1, 2]], np.int64)
            normals = np.zeros_like(verts)
            if self.verbose:
                print(f"[Sample {idx}] ROI empty -> returning dummy mesh")
        else:
            verts, faces, normals = m
            verts, faces, normals = subsample_vertices(
                verts, faces, normals, max_vertices=self.max_vertices,
                seed=self.rng_seed + idx, verbose=self.verbose
            )
            if self.verbose:
                print(f"[Sample {idx}] final verts={verts.shape[0]}, faces={faces.shape[0]}")

        sample = {
            "image": torch.from_numpy(vol[None, ...]),
            "gt_verts": torch.from_numpy(verts),
            "gt_normals": torch.from_numpy(normals),
            "gt_faces": torch.from_numpy(faces),
            "spacing": torch.tensor(spacing, dtype=torch.float32),
            "id": img_p.stem,
            "empty": empty
        }
        return sample

# ---------------------------
# Collate for variable-size meshes
# ---------------------------

def collate_fn_variable_mesh(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = [b for b in batch if not b.get("empty", False)]
    if len(batch) == 0:
        return {
            "image": torch.empty(0),
            "gt_verts": [],
            "gt_normals": [],
            "gt_faces": [],
            "spacing": [],
            "id": [],
            "empty": True
        }
    images = torch.stack([b["image"] for b in batch], dim=0)
    out = {
        "image": images,
        "gt_verts": [b["gt_verts"] for b in batch],
        "gt_normals": [b["gt_normals"] for b in batch],
        "gt_faces": [b["gt_faces"] for b in batch],
        "spacing": [b["spacing"] for b in batch],
        "id": [b["id"] for b in batch],
        "empty": False
    }
    print(f"[collate] B={images.shape[0]}, image={tuple(images.shape)}, verts={[tuple(v.shape) for v in out['gt_verts']]} faces={[tuple(f.shape) for f in out['gt_faces']]}")
    return out

# ---------------------------
# CLI smoke test
# ---------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Debug OAI FFD dataloader")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="Tr", choices=["Tr", "Ts"])
    ap.add_argument("--roi_id", type=int, default=2)
    ap.add_argument("--index", type=int, default=0, help="sample index to fetch")
    ap.add_argument("--max_vertices", type=int, default=5000)
    ap.add_argument("--max_vertices", type=int, default=5000)
    ap.add_argument("--marching_level", type=float, default=0.5)
    ap.add_argument("--keep_largest_cc", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ds = OAIFFDTemplateDataset(
        data_root=args.data_root,
        split=args.split,
        roi_id=args.roi_id,
        n_cases=None,
        seed=0,
        marching_level=args.marching_level,
        keep_largest_cc=args.keep_largest_cc,
        max_vertices=args.max_vertices,
        verbose=args.verbose
    )

    print(f"\n[main] dataset length = {len(ds)}")
    i = min(max(args.index, 0), max(len(ds)-1, 0))
    sample = ds[i]
    print(f"[main] got sample[{i}] id={sample['id']}")
    print(f"[main] image: {tuple(sample['image'].shape)}  (dtype={sample['image'].dtype})")
    print(f"[main] verts: {tuple(sample['gt_verts'].shape)}  faces: {tuple(sample['gt_faces'].shape)}  empty={sample['empty']}")
    print(f"[main] spacing: {sample['spacing'].tolist()}")
