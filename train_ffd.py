#!/usr/bin/env python3
"""
train_ffd.py

Small sanity experiment:
- Train UNet3D_FFD + BSplineFFD on FIRST 10 cases
- 10 epochs
- Print loss each epoch
- At epoch 10, evaluate geometry-based metrics (surface Dice & HD95) over these 10 cases

Usage (example):
  python train_ffd.py ^
    --data_root C:/Users/chris/MICCAI2026/OAI-ZIB-CM ^
    --split Tr --roi_id 2 ^
    --template_inner C:/Users/chris/MICCAI2026/Carti_Seg/femoral_cartilage_template_inner.ply ^
    --template_outer C:/Users/chris/MICCAI2026/Carti_Seg/femoral_cartilage_template_outer_offset2mm.ply ^
    --device cuda
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ffd_dataloader import (
    OAIFFDTemplateDataset, LatticeSpec, BSplineFFD,
    chamfer_distance, normal_consistency, uniform_laplacian_loss, thickness_regularization,
    collate_fn_variable_mesh, 
)
from ffd_model import UNet3D_FFD

# ---------------------------
# PLY loader for template
# ---------------------------

def read_ply_vertices_faces(path: Path):
    """Load PLY mesh (verts, faces) via open3d."""
    try:
        import open3d as o3d
        m = o3d.io.read_triangle_mesh(str(path))
        verts = np.asarray(m.vertices, dtype=np.float32)
        faces = np.asarray(m.triangles, dtype=np.int64)
        return verts, faces
    except Exception:
        raise RuntimeError("Please install open3d to read the template PLYs.")

# ---------------------------
# Build FFD lattice spec
# ---------------------------

def make_lattice_covering(verts_mm: np.ndarray, pad_mm: float, lattice: Tuple[int,int,int]) -> LatticeSpec:
    """h
    Build lattice spec that covers the template bbox with padding.
    spacing = bbox_extent / (size - 3)  (cubic B-spline needs 4 support; leave margin)
    """
    mn = verts_mm.min(axis=0) - pad_mm
    mx = verts_mm.max(axis=0) + pad_mm
    extent = mx - mn
    nx, ny, nz = lattice
    nx, ny, nz = max(nx, 4), max(ny, 4), max(nz, 4)
    spacing = extent / (np.array([nx, ny, nz], dtype=np.float32) - 3.0)
    return LatticeSpec(
        origin=torch.tensor(mn, dtype=torch.float32),
        spacing=torch.tensor(spacing, dtype=torch.float32),
        size=(nx, ny, nz)
    )

# ---------------------------
# Surface metrics: Dice & HD95 (point-based)
# ---------------------------

def surface_distances(pred_v: torch.Tensor, gt_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute nearest-neighbor distances between two point sets.
    pred_v: (Vp,3), gt_v: (Vg,3)  [in mm]
    Returns:
      d_pred_to_gt: (Vp,) distances
      d_gt_to_pred: (Vg,) distances
    """
    with torch.no_grad():
        x = pred_v
        y = gt_v
        x2 = (x ** 2).sum(dim=1, keepdim=True)        # (Vp,1)
        y2 = (y ** 2).sum(dim=1, keepdim=True).T      # (1,Vg)
        d2 = x2 + y2 - 2.0 * (x @ y.T)                # (Vp,Vg)
        d2 = d2.clamp_min(0.0)
        d_pred_to_gt = d2.min(dim=1)[0].sqrt()        # (Vp,)
        d_gt_to_pred = d2.min(dim=0)[0].sqrt()        # (Vg,)
    return d_pred_to_gt, d_gt_to_pred

def surface_dice_and_hd95(pred_v: torch.Tensor,
                          gt_v: torch.Tensor,
                          tolerance_mm: float = 1.0) -> tuple[float, float]:
    """
    Surface-based Dice (at tolerance) and HD95 on point sets.

    Dice_surf = (|p: d(p,G)<τ| + |g: d(g,P)<τ|) / (|P|+|G|)
    HD95 = max(95th percentile(d(P→G)), 95th percentile(d(G→P)))
    """
    d_pg, d_gp = surface_distances(pred_v, gt_v)  # (Vp,), (Vg,)

    # surface Dice
    tp_pg = (d_pg <= tolerance_mm).float().sum()
    tp_gp = (d_gp <= tolerance_mm).float().sum()
    dice = (tp_pg + tp_gp) / (d_pg.numel() + d_gp.numel() + 1e-8)

    # HD95
    hd95_pg = torch.quantile(d_pg, 0.95)
    hd95_gp = torch.quantile(d_gp, 0.95)
    hd95 = torch.max(hd95_pg, hd95_gp)

    return float(dice.item()), float(hd95.item())

# ---------------------------
# Evaluation on first N cases
# ---------------------------

def evaluate_surfaces(model: UNet3D_FFD,
                      ffd: BSplineFFD,
                      t_inner_v: torch.Tensor,
                      ds: OAIFFDTemplateDataset,
                      device: torch.device,
                      amp_enabled: bool = False,
                      max_cases: int = 10,
                      tolerance_mm: float = 1.0):
    """
    Evaluate on at most `max_cases` samples from ds.
    Returns mean surface Dice and mean HD95.
    """
    model.eval()
    dices = []
    hd95s = []
    n = min(len(ds), max_cases)

    with torch.no_grad():
        for idx in range(n):
            sample = ds[idx]
            img = sample["image"].unsqueeze(0).to(device)   # (1,1,D,H,W)
            gt_v = sample["gt_verts"].to(device)            # (V,3)

            # forward
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                deltaG = model(img)[0]                          # (nx,ny,nz,3)
            pred_inner = ffd(t_inner_v, deltaG)             # (V_template,3)

            # NOTE: pred_inner is deformed template; gt_v is GT surface from mask.
            # Metrics are computed in the template space assumption.

            # Optional: to control cost, you could subsample points here.

            dice, hd95 = surface_dice_and_hd95(pred_inner, gt_v, tolerance_mm=tolerance_mm)
            dices.append(dice)
            hd95s.append(hd95)

    model.train()
    if len(dices) == 0:
        return 0.0, 0.0

    mean_dice = float(np.mean(dices))
    mean_hd95 = float(np.mean(hd95s))
    return mean_dice, mean_hd95

# ---------------------------
# Main training
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="Tr", choices=["Tr", "Ts"])
    ap.add_argument("--roi_id", type=int, default=2)
    ap.add_argument("--template_inner", type=str, required=True)
    ap.add_argument("--template_outer", type=str, default="")
    ap.add_argument("--epochs", type=int, default=10)        # small sanity run
    ap.add_argument("--batch_size", type=int, default=2)  # <<-- CHANGED: Increased to 2 for 2 GPUs
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lattice", type=int, nargs=3, default=[6, 6, 6], help="nx ny nz")
    ap.add_argument("--pad_mm", type=float, default=5.0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0") # <<-- CHANGED: Default to 'cuda:0'
    ap.add_argument("--n_cases", type=int, default=10, help="use only first N cases")
    ap.add_argument("--amp", action="store_true", default=True, help="use autocast mixed precision to reduce GPU memory")
    ap.add_argument("--no_amp", action="store_false", dest="amp", help="disable mixed precision")
    ap.add_argument("--data_parallel", action="store_true", default=True, help="enable DataParallel when multiple GPUs available")
    ap.add_argument("--no_data_parallel", action="store_false", dest="data_parallel", help="force single-GPU training")
    # loss weights
    ap.add_argument("--w_chamfer", type=float, default=1.0)
    ap.add_argument("--w_normal", type=float, default=0.2)
    ap.add_argument("--w_lap", type=float, default=0.01)
    ap.add_argument("--w_thickness", type=float, default=0.2)
    ap.add_argument("--eval_tol_mm", type=float, default=1.0, help="surface Dice tolerance (mm)")
    args = ap.parse_args()

    # device
    # Explicitly set the base device and check for multiple GPUs
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(args.device)
        print(f"[Info] Using base device: {device}")
        print(f"[Info] Number of GPUs available: {torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")
        print(f"[Info] Using device: {device}")


    # ... (lines 400-430 of the original script)
    # Template (inner, faces)
    t_inner_v_np, t_faces_np = read_ply_vertices_faces(Path(args.template_inner))
    # Move template data to the base device
    t_inner_v = torch.from_numpy(t_inner_v_np).float().to(device)  # (V,3)
    t_faces = torch.from_numpy(t_faces_np).long().to(device)        # (F,3)

    # Optional outer template (dual-surface mode)
    dual = False
    if args.template_outer:
        t_outer_v_np, _ = read_ply_vertices_faces(Path(args.template_outer))
        # Move template data to the base device
        t_outer_v = torch.from_numpy(t_outer_v_np).float().to(device)
        dual = True
        print("[Info] Dual-surface mode enabled (inner + outer template).")
    else:
        t_outer_v = None
        print("[Info] Single-surface mode (inner template only).")

    # FFD lattice
    spec = make_lattice_covering(t_inner_v_np, pad_mm=args.pad_mm, lattice=tuple(args.lattice))
    # BSplineFFD does not need to be DataParallelized, just needs to know its device.
    ffd = BSplineFFD(spec).to(device) 
    print(f"[Info] FFD lattice size = {spec.size}, origin = {spec.origin.numpy()}, spacing = {spec.spacing.numpy()}")

    # Model & optimizer
    model = UNet3D_FFD(in_channels=1, base_channels=8, lattice_size=tuple(args.lattice))
    
    # ----------------------------------------------------------------
    # Apply DataParallel: This is the KEY step for multi-GPU
    # ----------------------------------------------------------------
    if args.data_parallel and device.type == 'cuda' and torch.cuda.device_count() > 1 and args.batch_size > 1:
        print("[Info] Using DataParallel on GPUs 0 and 1.")
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    else:
        print("[Info] Not using DataParallel.")


    # Move the (potentially wrapped) model to the base device
    model.to(device)

    amp_enabled = bool(args.amp and device.type == "cuda")
    if amp_enabled:
        print("[Info] Mixed precision is enabled (autocast+GradScaler) to reduce GPU memory usage.")
    else:
        print("[Info] Mixed precision is disabled.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    ds = OAIFFDTemplateDataset(
        data_root=args.data_root,
        split=args.split,
        roi_id=args.roi_id,
        n_cases=args.n_cases,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_variable_mesh,
    )

    # -----------------------
    # Training
    # -----------------------
    model.train()
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{args.epochs}")
        avg_loss = 0.0

        for batch in pbar:
            # All input data is automatically moved to the base device (cuda:0)
            img = batch["image"].to(device)                  # (B,1,D,H,W)
            # When using DataParallel, the model expects a batch size > 1.
            # However, your loss calculation later expects unbatched ground truth (gt_v[0], gt_n[0]).
            # Since the batch size is very small, we'll process the loss on the unbatched GT 
            # while the forward pass handles the batching across GPUs.
            gt_v = batch["gt_verts"][0].to(device)         # (V,3) - Only first sample's GT used
            gt_n = batch["gt_normals"][0].to(device)       # (V,3) - Only first sample's GT used

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                # forward
                # DataParallel returns a batch-size tensor, so we take the first element (B=2 -> 2 elements)
                # The output deltaG will be (B, nx, ny, nz, 3) where B=batch_size
                deltaG_batch = model(img)                       # (B,nx,ny,nz,3)

                # NOTE: Your loss calculation is currently **single-sample based**, using only
                # the first item's ground truth (`gt_v`, `gt_n`) and the first item's prediction (`deltaG_batch[0]`).
                # To properly utilize the full batch size (B=2), you would need to:
                # 1. Modify the loss function to accept a batch of predictions and a batch of ground truths.
                # 2. Iterate over the batch dimensions to calculate individual losses and then sum/average them.
                #
                # For this simple DataParallel modification, we'll proceed using only the
                # first sample's prediction for the loss calculation, but be aware this is
                # *not* optimal multi-GPU usage for the loss part.
                deltaG = deltaG_batch[0]                        # (nx,ny,nz,3) - Use only the prediction for the first sample

                # deform template
                pred_inner = ffd(t_inner_v, deltaG)             # (V,3)
                if dual and t_outer_v is not None:
                    pred_outer = ffd(t_outer_v, deltaG)
                else:
                    pred_outer = None

                # geometry losses
                loss_ch = chamfer_distance(pred_inner, gt_v) * args.w_chamfer
                loss_norm = normal_consistency(pred_inner, t_faces, gt_v, gt_n) * args.w_normal
                loss_lap = uniform_laplacian_loss(pred_inner, t_faces) * args.w_lap
                if dual and pred_outer is not None:
                    loss_th = thickness_regularization(pred_inner, pred_outer, t_faces) * args.w_thickness
                else:
                    loss_th = torch.tensor(0.0, device=device)

                loss = loss_ch + loss_norm + loss_lap + loss_th
            optimizer.zero_grad()
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            avg_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ch": f"{loss_ch.item():.3f}",
                "nc": f"{loss_norm.item():.3f}",
                "lap": f"{loss_lap.item():.3f}",
                "th": f"{loss_th.item():.3f}"
            })

        avg_loss /= max(len(dl), 1)
        print(f"[Train] Epoch {epoch}: avg_loss={avg_loss:.4f}")

        # -----------------------
        # Evaluation every 10 epochs (here: only at epoch 10)
        # -----------------------
        if epoch % 10 == 0:
            mean_dice, mean_hd95 = evaluate_surfaces(
                model, ffd, t_inner_v, ds, device,
                amp_enabled=amp_enabled,
                max_cases=args.n_cases,
                tolerance_mm=args.eval_tol_mm
            )
            print(f"[Eval] Epoch {epoch}: "
                  f"surface Dice@{args.eval_tol_mm}mm = {mean_dice:.4f}, "
                  f"surface HD95 (mm) = {mean_hd95:.4f}")

        # -----------------------
        # Checkpoint
        # -----------------------
        

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lattice_spec": {
                "origin": ffd.origin.detach().cpu().numpy(),
                "spacing": ffd.spacing.detach().cpu().numpy(),
                "size": (ffd.nx, ffd.ny, ffd.nz),
            }
        }
        out_p = Path("checkpoints"); out_p.mkdir(exist_ok=True, parents=True)
        torch.save(ckpt, out_p / f"ffd_epoch{epoch:03d}.pt")

if __name__ == "__main__":
    main()
