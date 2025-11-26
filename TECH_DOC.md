# Femoral Cartilage NURBS Workflow

This document summarizes the three-stage workflow used to generate a femoral cartilage template, fit multi-patch NURBS surfaces, and train a neural network for template-based prediction.

## 1. Template generation (`build_femoral_template.py`)
- Purpose: build a femoral cartilage template (ROI=2) from OAI-ZIB-CM masks with cubic tensor-product B-spline fitting and an aligned thickness map.【F:build_femoral_template.py†L4-L20】
- Pipeline highlights: random sampling of 20–30 masks, ROI extraction, marching cubes to meshes with smoothing, rigid ICP alignment, average surface construction, NURBS surface fitting on a fixed (u, v) grid, and constant-thickness initialization.【F:build_femoral_template.py†L7-L15】
- Key CLI options: dataset root, number of cases, output directory, smoothing iterations, sampling density for ICP, and control-grid resolution (`--uv_size_u`, `--uv_size_v`).【F:build_femoral_template.py†L49-L117】
- Primary outputs: `average_mesh.ply`, `femoral_template_surf.npz` (control points, knots, degrees), and `femoral_template_thickness.npy` (thickness map).【F:build_femoral_template.py†L17-L20】

## 2. Multi-patch NURBS fitting (`fit_nurbs_from_central_template_2patch.py`)
- Purpose: fit two- or three-patch cubic B-spline surfaces to the central template mesh and evaluate Chamfer distance against the original surface.【F:fit_nurbs_from_central_template_2patch.py†L4-L25】
- Patch construction: split the template along the fixed world X-axis into 2 or 3 longitudinal patches, then within each patch perform PCA to define local axes, project vertices to a normalized [0,1]^2 grid, and pick nearest vertices as control points on a regular quad grid (no additional fitting).【F:fit_nurbs_from_central_template_2patch.py†L8-L25】【F:fit_nurbs_from_central_template_2patch.py†L119-L195】
- Important parameters: patch count (`--n_patches`), per-patch control lattice size (`--size_u`, `--size_v`), spline degrees, and sampling densities for Chamfer evaluation; outputs include per-patch `.npz` files, a merged `femoral_template_nurbs_mesh_multi.ply`, and `femoral_template_chamfer_stats.txt`.【F:fit_nurbs_from_central_template_2patch.py†L27-L112】

## 3. Training (`nurbs_training.py`)
- Purpose: provide PyTorch utilities to predict femoral cartilage NURBS surfaces from MRI volumes via a lightweight 3D U-Net and differentiable NURBS evaluation.【F:nurbs_training.py†L1-L10】
- Network backbone: `CartilageUNet` with encoder-decoder stages and global pooling head that outputs control-point displacements (and optional weight updates).【F:nurbs_training.py†L35-L121】
- NURBS support: `NURBSTemplate` and `MultiPatchNURBSTemplate` helpers (not shown here) load template control points/knots, build canonical UV grids, and enable differentiable surface sampling for loss computation.【F:nurbs_training.py†L168-L200】
- Shared alignment: uses the same fixed long-axis vector as the fitting stage (`PC_LONG`) to keep patch ordering consistent during training.【F:nurbs_training.py†L26-L28】
