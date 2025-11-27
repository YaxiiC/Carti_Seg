#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
align_to_template_icp.py

使用软骨 ROI（例如 ROI=2：Femoral Cartilage）做 ICP 对齐到模板网格，
并将对应的 MRI 体数据和 label 体数据用相同刚体变换重采样，
输出为新的 .nii.gz，全部统一到“模板空间”。

目录结构假定为 OAI-ZIB-CM：

OAI-ZIB-CM/
  imagesTr/
    oaizib_001_0000.nii.gz
    ...
  labelsTr/
    oaizib_001.nii.gz
    ...
  imagesTs/
  labelsTs/

示例调用（Windows CMD，一行）：

python align_to_template_icp.py ^
  --data_root C:\Users\chris\MICCAI2026\OAI-ZIB-CM ^
  --out_root C:\Users\chris\MICCAI2026\Carti_Seg ^
  --split Tr ^
  --roi_id 2 ^
  --ref_mesh C:\Users\chris\MICCAI2026\Carti_Seg\femoral_cartilage_template_central.ply ^
  --keep_largest_cc

  python align_to_template_icp.py ^
  --data_root C:\Users\chris\MICCAI2026\OAI-ZIB-CM ^
  --out_root C:\Users\chris\MICCAI2026\Carti_Seg ^
  --split Ts ^
  --roi_id 2 ^
  --ref_mesh C:\Users\chris\MICCAI2026\Carti_Seg\average_mesh.ply ^
  --keep_largest_cc

依赖：
  pip install numpy nibabel scikit-image open3d scipy tqdm
"""

import argparse
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import open3d as o3d
from scipy.ndimage import affine_transform
from skimage.measure import marching_cubes, label as cc_label
from tqdm import tqdm


# ----------------------------
# 工具函数
# ----------------------------

def nii_spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    """从 NIfTI affine 中提取 voxel spacing（假设轴正交）。"""
    return np.linalg.norm(affine[:3, :3], axis=0)


def load_roi_binary(mask_path: Path, roi_id: int = 2, keep_largest_cc: bool = True):
    """加载 label NIfTI，提取 ROI==roi_id 的二值掩膜和 spacing。"""
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
    """Marching Cubes -> Open3D TriangleMesh（物理坐标，单位 mm）。"""
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


def smooth_mesh(mesh: o3d.geometry.TriangleMesh,
                method: str = "taubin",
                iterations: int = 30,
                lambda_: float = 0.5,
                mu: float = -0.53):
    """Taubin（默认）或 Laplacian 平滑，清理退化三角形并重算法向。"""
    if method == "taubin":
        mesh = mesh.filter_smooth_taubin(number_of_iterations=iterations,
                                         lambda_filter=lambda_,
                                         mu=mu)
    else:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=iterations)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def to_pointcloud(mesh: o3d.geometry.TriangleMesh, n_samples: int = 50000):
    """均匀采样网格表面的点用于 ICP。"""
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    return mesh.sample_points_uniformly(number_of_points=n_samples)


def rigid_icp(source_mesh: o3d.geometry.TriangleMesh,
              target_mesh: o3d.geometry.TriangleMesh,
              voxel_down: float = 0.8,
              max_iters: int = 100,
              threshold: float = 5.0):
    """刚体点到点 ICP（source->target）。返回 transformed mesh 和 4x4 变换矩阵 T。"""
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


def _canonical_stem(p: Path) -> str:
    """
    去掉 _0000 通道后缀，用于匹配 image / label:
      oaizib_001_0000.nii.gz -> oaizib_001
      oaizib_001.nii.gz      -> oaizib_001
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
    构建 (image, label) 对：
      imagesTr/oaizib_001_0000.nii  <-> labelsTr/oaizib_001.nii
    多通道时优先选 _0000。
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
    for stem, candidates in imgs_by_stem.items():
        lp = labs_by_stem.get(stem, None)
        if lp is None:
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

    print(f"Matched {len(pairs)} image/label pairs by stem.")
    return pairs


def compute_affine_for_resample(spacing: np.ndarray,
                                T_world: np.ndarray) -> np.ndarray:
    """
    给定 voxel spacing (sx,sy,sz) 和 4x4 刚体变换 T_world（在 mm 空间），
    构造 scipy.ndimage.affine_transform 使用的 3x3 matrix + 3x1 offset。

    这里假设：
      world_mm = S @ index,  S = diag(sx, sy, sz, 1)
    我们要在“输出空间”的 index' 上重采样得到 world' = T_world @ world，
    于是 index_in = S^-1 @ T_world^-1 @ S @ index_out

    返回：A (4x4)，其中 index_in = A @ [i', j', k', 1]
    """
    sx, sy, sz = spacing
    S = np.eye(4, dtype=np.float64)
    S[0, 0] = sx
    S[1, 1] = sy
    S[2, 2] = sz
    S_inv = np.linalg.inv(S)
    T_inv = np.linalg.inv(T_world)

    A = S_inv @ T_inv @ S  # 4x4
    return A


def resample_nifti_with_icp_transform(
    img: nib.Nifti1Image,
    T_world: np.ndarray,
    order: int = 1,
) -> nib.Nifti1Image:
    """
    使用 ICP 得到的 4x4 刚体变换 T_world (在 mm 空间)，
    对 NIfTI 体数据做重采样，返回新的 NIfTI。
    - order=1 线性插值（MRI）
    - order=0 最近邻（label）
    """
    data = img.get_fdata()
    affine = img.affine
    spacing = nii_spacing_from_affine(affine)

    A = compute_affine_for_resample(spacing, T_world)  # 4x4
    matrix = A[:3, :3]
    offset = A[:3, 3]

    # scipy 的 affine_transform 是从 output index -> input index 的映射
    resampled = affine_transform(
        data,
        matrix=matrix,
        offset=offset,
        output_shape=data.shape,
        order=order,
        mode="constant",
        cval=0.0,
    )

    # 新 affine 可以简单设为对角 spacing（你后续训练大多只看 array，不太依赖 affine）
    new_affine = np.eye(4, dtype=np.float64)
    new_affine[0, 0] = spacing[0]
    new_affine[1, 1] = spacing[1]
    new_affine[2, 2] = spacing[2]

    out_img = nib.Nifti1Image(resampled.astype(img.get_data_dtype()), new_affine)
    return out_img


# ----------------------------
# 主流程：对所有病例做 ICP + 重采样
# ----------------------------

def align_dataset_to_template(
    data_root: Path,
    out_root: Path,
    ref_mesh_path: Path,
    split: str = "Tr",
    roi_id: int = 2,
    keep_largest_cc: bool = True,
    mc_level: float = 0.5,
    smooth_method: str = "taubin",
    smooth_iters: int = 20,
    taubin_lambda: float = 0.5,
    taubin_mu: float = -0.53,
    icp_down_voxel: float = 0.8,
    icp_max_iters: int = 100,
    icp_threshold: float = 5.0,
):
    data_root = Path(data_root)
    out_root = Path(out_root)
    out_img_dir = out_root / "aligned" / f"images{split}"
    out_lab_dir = out_root / "aligned" / f"labels{split}"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lab_dir.mkdir(parents=True, exist_ok=True)

    # 读取参考模板 mesh（例如 femoral_cartilage_template_central.ply）
    print(f"Loading reference mesh: {ref_mesh_path}")
    ref_mesh = o3d.io.read_triangle_mesh(str(ref_mesh_path))
    if not ref_mesh.has_vertices():
        raise RuntimeError(f"Reference mesh has no vertices: {ref_mesh_path}")
    ref_mesh.compute_vertex_normals()

    # 匹配 image/label 对
    pairs = gather_pairs(data_root, split=split)

    print(f"\nAligning {len(pairs)} cases to template via ICP (ROI={roi_id})...")
    for img_path, lab_path in tqdm(pairs):
        stem = _canonical_stem(img_path)
        try:
            # 1) 提取 ROI mask 并转成 mesh
            roi_mask, spacing = load_roi_binary(lab_path, roi_id=roi_id, keep_largest_cc=keep_largest_cc)
            if roi_mask.max() == 0:
                print(f"[WARN] {stem}: ROI={roi_id} is empty, skip.")
                continue

            mesh = mesh_from_mask(roi_mask, spacing, level=mc_level)
            mesh = smooth_mesh(mesh, method=smooth_method, iterations=smooth_iters,
                               lambda_=taubin_lambda, mu=taubin_mu)

            # 2) ICP: case mesh -> ref_mesh，得到刚体变换 T
            aligned_mesh, T = rigid_icp(mesh, ref_mesh,
                                        voxel_down=icp_down_voxel,
                                        max_iters=icp_max_iters,
                                        threshold=icp_threshold)

            # 3) 用 T 重采样 MRI 和 label 到模板空间
            img_nii = nib.load(str(img_path))
            lab_nii = nib.load(str(lab_path))

            aligned_img = resample_nifti_with_icp_transform(img_nii, T, order=1)
            aligned_lab = resample_nifti_with_icp_transform(lab_nii, T, order=0)

            # 4) 保存结果
            out_img_path = out_img_dir / (stem + "_aligned.nii.gz")
            out_lab_path = out_lab_dir / (stem + "_aligned.nii.gz")
            nib.save(aligned_img, str(out_img_path))
            nib.save(aligned_lab, str(out_lab_path))

        except Exception as e:
            print(f"[ERROR] Failed to align {stem}: {e}")

    print("\nDone.")
    print(f"Aligned images saved under: {out_img_dir}")
    print(f"Aligned labels saved under: {out_lab_dir}")


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="ICP-align OAI-ZIB-CM MRI+labels to a femoral cartilage template mesh and resample to .nii.gz"
    )
    p.add_argument("--data_root", type=str, required=True,
                   help="Path to OAI-ZIB-CM root (with imagesTr, labelsTr, imagesTs, labelsTs)")
    p.add_argument("--out_root", type=str, required=True,
                   help="Output root, will create out_root/aligned/imagesTr, labelsTr, ...")
    p.add_argument("--ref_mesh_path", type=str, required=True,
                   help="Reference template mesh (.ply), e.g. femoral_cartilage_template_central.ply or average_mesh.ply")
    p.add_argument("--split", type=str, default="Tr", choices=["Tr", "Ts"],
                   help="Which split to align (Tr or Ts)")
    p.add_argument("--roi_id", type=int, default=2,
                   help="Label value for cartilage ROI (2 for femoral cartilage in OAI-ZIB-CM)")
    p.add_argument("--keep_largest_cc", action="store_true",
                   help="Keep only largest connected component in ROI mask")
    p.add_argument("--mc_level", type=float, default=0.5,
                   help="Marching Cubes isovalue for binary mask")
    p.add_argument("--smooth_method", type=str, default="taubin", choices=["taubin", "laplacian"],
                   help="Mesh smoothing method")
    p.add_argument("--smooth_iters", type=int, default=20,
                   help="Number of smoothing iterations for ROI mesh")
    p.add_argument("--taubin_lambda", type=float, default=0.5)
    p.add_argument("--taubin_mu", type=float, default=-0.53)
    p.add_argument("--icp_down_voxel", type=float, default=0.8,
                   help="Voxel size (mm) for downsampling ICP point clouds")
    p.add_argument("--icp_max_iters", type=int, default=100,
                   help="Max ICP iterations")
    p.add_argument("--icp_threshold", type=float, default=5.0,
                   help="ICP correspondence distance threshold (mm)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    align_dataset_to_template(
        data_root=Path(args.data_root),
        out_root=Path(args.out_root),
        ref_mesh_path=Path(args.ref_mesh_path),
        split=args.split,
        roi_id=args.roi_id,
        keep_largest_cc=args.keep_largest_cc,
        mc_level=args.mc_level,
        smooth_method=args.smooth_method,
        smooth_iters=args.smooth_iters,
        taubin_lambda=args.taubin_lambda,
        taubin_mu=args.taubin_mu,
        icp_down_voxel=args.icp_down_voxel,
        icp_max_iters=args.icp_max_iters,
        icp_threshold=args.icp_threshold,
    )
