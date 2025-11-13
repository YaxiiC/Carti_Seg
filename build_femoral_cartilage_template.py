#!/usr/bin/env python3
"""
build_femoral_cartilage_template.py

Single-file template builder restricted to ROI=2 (Femoral Cartilage) from OAI-ZIB-CM.

Workflow:
  multi-label NIfTI -> (mask==2) -> marching cubes -> smoothing -> rigid ICP -> mean surface
  -> optional outer surface by normal offset (~2 mm)

Run (Windows CMD, one line):
  python build_femoral_cartilage_template.py --data_root C:\path\to\OAI-ZIB-CM --out_dir C:\path\to\Carti_Seg --split Tr --n_cases 25 --roi_id 2 --keep_largest_cc --offset_outer_mm 2.0

Dependencies:
  pip install numpy nibabel scikit-image open3d tqdm scipy
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes, label as cc_label
from tqdm import tqdm

LABELS_MAP = {
    1: "Femur",
    2: "Femoral Cartilage",
    3: "Tibia",
    4: "Medial Tibial Cartilage",
    5: "Lateral Tibial Cartilage",
}

# ----------------------------
# Utilities
# ----------------------------

def nii_spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    """Derive voxel spacing from a NIfTI affine (assumes orthogonal axes)."""
    return np.linalg.norm(affine[:3, :3], axis=0)

def load_roi_binary(mask_path: Path, roi_id: int = 2, keep_largest_cc: bool = True):
    """Load multi-label mask, extract ROI==roi_id as binary float32 array, optionally keep largest CC."""
    img = nib.load(str(mask_path))
    data = img.get_fdata()
    spacing = nii_spacing_from_affine(img.affine)

    roi_mask = (np.round(data).astype(np.int32) == int(roi_id)).astype(np.uint8)

    if roi_mask.max() == 0:
        return roi_mask.astype(np.float32), spacing  # empty

    if keep_largest_cc:
        lab = cc_label(roi_mask, connectivity=1)
        if lab.max() > 0:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            roi_mask = (lab == np.argmax(sizes)).astype(np.uint8)

    return roi_mask.astype(np.float32), spacing

def mesh_from_mask(mask: np.ndarray, spacing, level: float = 0.5) -> o3d.geometry.TriangleMesh:
    """Marching Cubes -> Open3D TriangleMesh in physical space (uses NIfTI spacing)."""
    if mask.max() <= 0:
        raise ValueError("ROI mask is empty (no voxels for the requested label).")
    verts, faces, norms, _ = marching_cubes(mask, level=level, spacing=spacing)
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces.astype(np.int32))
    )
    if norms is None or len(norms) != len(verts):
        mesh.compute_vertex_normals()
    else:
        mesh.vertex_normals = o3d.utility.Vector3dVector(norms)
    return mesh

def smooth_mesh(mesh: o3d.geometry.TriangleMesh, method: str = "taubin",
                iterations: int = 30, lambda_: float = 0.5, mu: float = -0.53):
    """Taubin (default) or Laplacian smoothing; cleans degeneracies and recomputes normals."""
    if method == "taubin":
        mesh = mesh.filter_smooth_taubin(number_of_iterations=iterations, lambda_filter=lambda_, mu=mu)
    else:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=iterations)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh

def to_pointcloud(mesh: o3d.geometry.TriangleMesh, n_samples: int = 50000):
    """Uniformly sample mesh surface points for ICP."""
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    return mesh.sample_points_uniformly(number_of_points=n_samples)

def rigid_icp(source_mesh: o3d.geometry.TriangleMesh,
              target_mesh: o3d.geometry.TriangleMesh,
              voxel_down: float = 0.8,
              max_iters: int = 100,
              threshold: float = 5.0):
    """Rigid point-to-point ICP (source->target). Returns transformed mesh and 4x4 transform."""
    src = to_pointcloud(source_mesh)
    tgt = to_pointcloud(target_mesh)
    if voxel_down and voxel_down > 0:
        src = src.voxel_down_sample(voxel_down)
        tgt = tgt.voxel_down_sample(voxel_down)
    init = np.eye(4)
    init[:3, 3] = np.array(tgt.get_center()) - np.array(src.get_center())
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, threshold, init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iters),
    )
    T = reg.transformation
    out = o3d.geometry.TriangleMesh(source_mesh)
    out.transform(T)
    out.compute_vertex_normals()
    return out, T

def average_on_reference_topology(ref_mesh: o3d.geometry.TriangleMesh,
                                  aligned_meshes,
                                  k: int = 1) -> o3d.geometry.TriangleMesh:
    """Mean surface on the reference topology via nearest-neighbor vertex correspondence."""
    ref_v = np.asarray(ref_mesh.vertices)
    accum = np.zeros_like(ref_v)
    counts = np.zeros((ref_v.shape[0], 1))
    for m in aligned_meshes:
        mv = np.asarray(m.vertices)
        tree = cKDTree(mv)
        _, idx = tree.query(ref_v, k=k)
        if k == 1:
            matched = mv[idx]
        else:
            matched = mv[idx].mean(axis=1)
        accum += matched
        counts += 1.0
    mean_v = accum / np.clip(counts, 1.0, None)
    mean_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mean_v),
        triangles=o3d.utility.Vector3iVector(np.asarray(ref_mesh.triangles)),
    )
    mean_mesh.remove_degenerate_triangles()
    mean_mesh.compute_vertex_normals()
    return mean_mesh

def offset_surface_along_normals(mesh: o3d.geometry.TriangleMesh, offset_mm: float = 2.0):
    """Create an offset surface by displacing vertices along vertex normals."""
    m = o3d.geometry.TriangleMesh(mesh)
    m.compute_vertex_normals()
    v = np.asarray(m.vertices)
    n = np.asarray(m.vertex_normals)
    m.vertices = o3d.utility.Vector3dVector(v + offset_mm * n)
    m.compute_vertex_normals()
    return m

# ------- filename pairing (handles _0000 vs no-suffix) -------

def _canonical_stem(p: Path) -> str:
    """
    Returns a stem that ignores channel suffixes like _0000 and the extension.
    Examples:
      oaizib_001_0000.nii.gz -> oaizib_001
      oaizib_001_0001.nii    -> oaizib_001
      oaizib_001.nii         -> oaizib_001
    """
    name = p.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    parts = name.split("_")
    if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) == 4:
        name = "_".join(parts[:-1])
    return name

def gather_pairs(oai_root: Path, split: str = "Tr"):
    """
    Build (image,label) pairs by canonical stem:
      imagesTr/oaizib_001_0000.nii  <-> labelsTr/oaizib_001.nii
    If multiple image channels exist (e.g., _0000, _0001), it picks _0000 by default.
    """
    img_dir = oai_root / f"images{split}"
    lab_dir = oai_root / f"labels{split}"
    if not img_dir.exists() or not lab_dir.exists():
        raise FileNotFoundError(f"Expected {img_dir} and {lab_dir}")

    img_files = sorted(list(img_dir.glob("*.nii")) + list(img_dir.glob("*.nii.gz")))
    lab_files = sorted(list(lab_dir.glob("*.nii")) + list(lab_dir.glob("*.nii.gz")))

    imgs_by_stem = defaultdict(list)
    for ip in img_files:
        imgs_by_stem[_canonical_stem(ip)].append(ip)

    labs_by_stem = {}
    for lp in lab_files:
        labs_by_stem[_canonical_stem(lp)] = lp

    pairs = []
    missing_labels = []
    for stem, candidates in imgs_by_stem.items():
        lp = labs_by_stem.get(stem, None)
        if lp is None:
            missing_labels.append(stem)
            continue
        preferred = None
        for c in candidates:
            if c.name.endswith("_0000.nii") or c.name.endswith("_0000.nii.gz"):
                preferred = c
                break
        if preferred is None:
            preferred = candidates[0]
        pairs.append((preferred, lp))

    if len(pairs) == 0:
        raise RuntimeError(
            "No (image,label) pairs found after stem matching.\n"
            f"Found {len(img_files)} images, {len(lab_files)} labels.\n"
            f"Example images: {[p.name for p in img_files[:3]]}\n"
            f"Example labels: {[p.name for p in lab_files[:3]]}\n"
        )

    print(f"Matched {len(pairs)} image/label pairs by stem. Missing labels for {len(missing_labels)} stems.")
    if missing_labels[:5]:
        print("First missing-label stems:", missing_labels[:5])
    return pairs

# ----------------------------
# Main pipeline
# ----------------------------

def build_template_for_roi(
    data_root,
    out_dir,
    roi_id: int = 2,                    # Femoral Cartilage
    n_cases: int = 25,
    split: str = "Tr",
    seed: int = 0,
    keep_largest_cc: bool = True,
    mc_level: float = 0.5,
    smooth_method: str = "taubin",
    smooth_iters: int = 30,
    taubin_lambda: float = 0.5,
    taubin_mu: float = -0.53,
    icp_down_voxel: float = 0.8,
    icp_max_iters: int = 100,
    icp_threshold: float = 5.0,
    offset_outer_mm: float = 2.0,
    save_aligned: bool = True
):
    random.seed(seed)
    np.random.seed(seed)

    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aligned").mkdir(exist_ok=True)

    label_name = LABELS_MAP.get(int(roi_id), f"ROI_{roi_id}")
    pairs = gather_pairs(data_root, split=split)
    if not pairs:
        raise RuntimeError("No (image,label) pairs found.")

    sampled = random.sample(pairs, min(n_cases, len(pairs)))
    meshes = []
    names = []

    print(f"Extracting ROI={roi_id} ({label_name}) from {len(sampled)} masks...")
    for img_path, lab_path in tqdm(sampled):
        roi_mask, spacing = load_roi_binary(lab_path, roi_id=roi_id, keep_largest_cc=keep_largest_cc)
        if roi_mask.max() == 0:
            # Skip empty ROI (some cases may lack the label)
            continue
        m = mesh_from_mask(roi_mask, spacing, level=mc_level)
        m = smooth_mesh(m, method=smooth_method, iterations=smooth_iters,
                        lambda_=taubin_lambda, mu=taubin_mu)
        meshes.append(m)
        names.append(img_path.name)

    if len(meshes) == 0:
        raise RuntimeError(f"No non-empty meshes for ROI={roi_id}. Check labels and ROI id.")

    # Reference topology: first mesh
    ref_mesh = meshes[0]
    aligned = [ref_mesh]
    transforms = [np.eye(4)]

    print("Rigid ICP alignment to reference...")
    for m in tqdm(meshes[1:]):
        am, T = rigid_icp(m, ref_mesh, voxel_down=icp_down_voxel,
                          max_iters=icp_max_iters, threshold=icp_threshold)
        aligned.append(am)
        transforms.append(T)

    if save_aligned:
        for nm, m in zip(names, aligned):
            outp = out_dir / "aligned" / (nm.replace(".nii", "").replace(".gz", "") + f"_roi{roi_id}_aligned.ply")
            o3d.io.write_triangle_mesh(str(outp), m)

    print("Computing mean template on reference topology...")
    mean_mesh = average_on_reference_topology(ref_mesh, aligned, k=1)

    # Inner (template) and outer (offset)
    inner_mesh = smooth_mesh(mean_mesh, method=smooth_method,
                             iterations=max(1, smooth_iters // 2),
                             lambda_=taubin_lambda, mu=taubin_mu)

    outer_mesh = None
    if offset_outer_mm and offset_outer_mm > 0:
        outer_mesh = offset_surface_along_normals(inner_mesh, offset_mm=offset_outer_mm)
        outer_mesh = smooth_mesh(outer_mesh, method=smooth_method,
                                 iterations=max(1, smooth_iters // 3),
                                 lambda_=taubin_lambda, mu=taubin_mu)

    # Save
    base = f"{label_name.lower().replace(' ', '_')}_template"
    inner_path = out_dir / f"{base}_inner.ply"
    o3d.io.write_triangle_mesh(str(inner_path), inner_mesh)

    manifest = {
        "data_root": str(data_root),
        "split": split,
        "roi_id": int(roi_id),
        "roi_name": label_name,
        "n_used": len(aligned),
        "sampled_cases": names,
        "preproc": {
            "keep_largest_cc": keep_largest_cc,
            "mc_level": mc_level
        },
        "smoothing": {
            "method": smooth_method,
            "iterations": smooth_iters,
            "taubin_lambda": taubin_lambda,
            "taubin_mu": taubin_mu
        },
        "icp": {
            "voxel_down": icp_down_voxel,
            "max_iters": icp_max_iters,
            "threshold": icp_threshold
        },
        "offset_outer_mm": float(offset_outer_mm),
        "outputs": {
            "inner_mesh": str(inner_path)
        }
    }

    if outer_mesh is not None:
        outer_path = out_dir / f"{base}_outer_offset{int(round(offset_outer_mm))}mm.ply"
        o3d.io.write_triangle_mesh(str(outer_path), outer_mesh)
        manifest["outputs"]["outer_mesh"] = str(outer_path)

    with open(out_dir / f"{base}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nDone.")
    print(f"Saved inner template: {inner_path}")
    if outer_mesh is not None:
        print(f"Saved outer template: {manifest['outputs']['outer_mesh']}")
    print(f"Manifest: {out_dir / (base + '_manifest.json')}")

# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build femoral cartilage (ROI=2) template from OAI-ZIB-CM")
    p.add_argument("--data_root", type=str, required=True, help="Path to OAI-ZIB-CM")
    p.add_argument("--out_dir", type=str, default="outputs/template_femoral_cartilage")
    p.add_argument("--split", type=str, default="Tr", choices=["Tr", "Ts"])
    p.add_argument("--n_cases", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--roi_id", type=int, default=2, help="Use 2 for Femoral Cartilage")
    p.add_argument("--keep_largest_cc", action="store_true", help="Keep only largest connected component of ROI")
    p.add_argument("--mc_level", type=float, default=0.5, help="Marching Cubes isovalue for binary mask")
    p.add_argument("--smooth_method", type=str, default="taubin", choices=["taubin", "laplacian"])
    p.add_argument("--smooth_iters", type=int, default=30)
    p.add_argument("--taubin_lambda", type=float, default=0.5)
    p.add_argument("--taubin_mu", type=float, default=-0.53)
    p.add_argument("--icp_down_voxel", type=float, default=0.8, help="Voxel size for ICP downsample (mm)")
    p.add_argument("--icp_max_iters", type=int, default=100)
    p.add_argument("--icp_threshold", type=float, default=5.0, help="ICP correspondence threshold (mm)")
    p.add_argument("--offset_outer_mm", type=float, default=2.0, help="Normal offset to create outer surface (mm)")
    p.add_argument("--no_save_aligned", action="store_true", help="Skip saving per-case aligned meshes")
    return p.parse_args()

if __name__ == "__main__":
    a = parse_args()
    build_template_for_roi(
        data_root=a.data_root,
        out_dir=a.out_dir,
        roi_id=a.roi_id,
        n_cases=a.n_cases,
        split=a.split,
        seed=a.seed,
        keep_largest_cc=a.keep_largest_cc,
        mc_level=a.mc_level,
        smooth_method=a.smooth_method,
        smooth_iters=a.smooth_iters,
        taubin_lambda=a.taubin_lambda,
        taubin_mu=a.taubin_mu,
        icp_down_voxel=a.icp_down_voxel,
        icp_max_iters=a.icp_max_iters,
        icp_threshold=a.icp_threshold,
        offset_outer_mm=a.offset_outer_mm,
        save_aligned=not a.no_save_aligned,
    )
