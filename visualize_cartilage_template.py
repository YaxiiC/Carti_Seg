#!/usr/bin/env python3
r"""
visualize_cartilage_template.py

Viewer for NURBS-based cartilage template (central surface + thickness).

- Reads a manifest JSON produced by the central-surface template builder
  (NURBS_template_central_surface_generation.py).
- Loads:
    - central_mesh (.ply)
    - thickness_default (.npy) if available
- Reconstructs inner/outer surfaces as:
      S_inner = S_c - 0.5 * T * n
      S_outer = S_c + 0.5 * T * n

Supports:
  - Modern Open3D draw() (no GUI init)
  - O3DVisualizer GUI fallback
  - Legacy draw_geometries() fallback
  - Optional offscreen PNG rendering

Usage (Windows CMD, one line, from the folder containing the manifest):
  python visualize_cartilage_template.py ^
    --manifest C:/Users/chris/MICCAI2026/Carti_Seg/femoral_cartilage_template_manifest.json ^
    --mode central_inner_outer --alpha_outer 0.45 --bg dark

Save PNG:
  python visualize_cartilage_template.py ^
    --manifest C:/Users/chris/MICCAI2026/Carti_Seg/femoral_cartilage_template_manifest.json ^
    --mode central_inner_outer ^
    --save_png C:/path/to/out.png
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d

# ----------------------------
# Helpers: geometry + colors
# ----------------------------

def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    """Load a PLY mesh and ensure normals exist."""
    m = o3d.io.read_triangle_mesh(str(path))
    if not m.has_vertex_normals():
        m.compute_vertex_normals()
    if not m.has_triangle_normals():
        m.compute_triangle_normals()
    return m

def colorize(mesh: o3d.geometry.TriangleMesh, rgb=(1.0, 1.0, 1.0)):
    """Return a copy of mesh painted with a uniform RGB color."""
    out = o3d.geometry.TriangleMesh(mesh)
    out.paint_uniform_color(rgb)
    return out

def make_material(rgba, transparent: bool):
    """Create a MaterialRecord with (possibly) transparent shader."""
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency" if transparent else "defaultLit"
    mat.base_color = tuple(float(c) for c in rgba)  # (r,g,b,a)
    mat.point_size = 2.0
    return mat

def build_inner_outer_from_central_and_thickness(
    central_mesh: o3d.geometry.TriangleMesh,
    thickness: np.ndarray,
) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh]:
    """
    Build inner and outer surfaces from:
      - central surface mesh (with vertex normals)
      - thickness field T(u,v) on the same (u,v) sampling grid

    Assumes:
      - central_mesh vertices are ordered as row-major [u,v] grid,
      - thickness.shape == (res_u, res_v),
      - num_vertices == res_u * res_v.
    """
    central_mesh.compute_vertex_normals()
    verts_c = np.asarray(central_mesh.vertices)           # (N,3)
    normals = np.asarray(central_mesh.vertex_normals)     # (N,3)

    res_u, res_v = thickness.shape
    N = verts_c.shape[0]
    if res_u * res_v != N:
        raise RuntimeError(
            f"Thickness grid {thickness.shape} does not match vertex count {N} "
            f"(expected res_u * res_v == N)."
        )

    T_flat = thickness.reshape(-1)                        # (N,)
    offset = 0.5 * T_flat[:, None] * normals             # (N,3)

    verts_inner = verts_c - offset
    verts_outer = verts_c + offset

    inner_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts_inner),
        triangles=central_mesh.triangles,
    )
    inner_mesh.remove_degenerate_triangles()
    inner_mesh.compute_vertex_normals()

    outer_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts_outer),
        triangles=central_mesh.triangles,
    )
    outer_mesh.remove_degenerate_triangles()
    outer_mesh.compute_vertex_normals()

    return inner_mesh, outer_mesh

def color_mesh(mesh: o3d.geometry.TriangleMesh, rgb: Tuple[float, float, float]):
    """In-place uniform coloring."""
    if mesh is None:
        return
    mesh.paint_uniform_color(rgb)

# ----------------------------
# Build geometries from manifest
# ----------------------------

def load_template_from_manifest(
    manifest_path: Path,
    mode: str,
) -> Tuple[Optional[o3d.geometry.TriangleMesh],
           Optional[o3d.geometry.TriangleMesh],
           Optional[o3d.geometry.TriangleMesh]]:
    """
    From the manifest JSON, load:
      - central_mesh (.ply)
      - thickness_default (.npy) if available
    And optionally reconstruct inner+outer.

    Returns (central_mesh, inner_mesh, outer_mesh). Some may be None depending on mode.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    out_dir = manifest_path.parent
    outputs = manifest.get("outputs", {})

    central_mesh_path = outputs.get("central_mesh", None)
    thickness_path = outputs.get("thickness_default", None)

    if central_mesh_path is None:
        raise RuntimeError("Manifest does not contain 'central_mesh' in 'outputs'.")

    # Resolve paths (relative to manifest dir if not absolute)
    central_mesh_path = Path(central_mesh_path)
    if not central_mesh_path.is_absolute():
        central_mesh_path = out_dir / central_mesh_path

    central_mesh = load_mesh(central_mesh_path)

    inner_mesh = None
    outer_mesh = None

    if mode in ("inner_outer", "central_inner_outer") and thickness_path is not None:
        thickness_path = Path(thickness_path)
        if not thickness_path.is_absolute():
            thickness_path = out_dir / thickness_path
        if not thickness_path.exists():
            raise FileNotFoundError(f"Thickness file not found: {thickness_path}")
        thickness = np.load(thickness_path)  # shape (res_u, res_v)
        inner_mesh, outer_mesh = build_inner_outer_from_central_and_thickness(
            central_mesh,
            thickness,
        )

    return central_mesh, inner_mesh, outer_mesh

# ----------------------------
# Modern interactive draw()
# ----------------------------

def draw_modern(
    central_mesh: Optional[o3d.geometry.TriangleMesh],
    inner_mesh: Optional[o3d.geometry.TriangleMesh],
    outer_mesh: Optional[o3d.geometry.TriangleMesh],
    alpha_outer: float,
    bg_color_rgb: Tuple[float, float, float],
):
    geoms = []

    # Axes
    geoms.append({
        "name": "axes",
        "geometry": o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0),
        "material": make_material((1.0, 1.0, 1.0, 1.0), transparent=False),
    })

    if central_mesh is not None:
        cm = colorize(central_mesh, rgb=(0.95, 0.95, 0.95))
        geoms.append({
            "name": "central",
            "geometry": cm,
            "material": make_material((0.95, 0.95, 0.95, 1.0), transparent=False),
        })

    if inner_mesh is not None:
        im = colorize(inner_mesh, rgb=(0.10, 0.60, 1.00))
        geoms.append({
            "name": "inner",
            "geometry": im,
            "material": make_material(
                (0.10, 0.60, 1.00, 0.9),  # almost opaque
                transparent=True
            ),
        })

    if outer_mesh is not None:
        om = colorize(outer_mesh, rgb=(1.00, 0.30, 0.10))
        geoms.append({
            "name": "outer",
            "geometry": om,
            "material": make_material(
                (1.00, 0.30, 0.10, float(np.clip(alpha_outer, 0.0, 1.0))),
                transparent=True
            ),
        })

    o3d.visualization.draw(
        geoms,
        title="Cartilage NURBS Template Viewer",
        bg_color=bg_color_rgb,
        show_skybox=False,
    )

# ----------------------------
# O3DVisualizer interactive (GUI)
# ----------------------------

def draw_gui(
    central_mesh: Optional[o3d.geometry.TriangleMesh],
    inner_mesh: Optional[o3d.geometry.TriangleMesh],
    outer_mesh: Optional[o3d.geometry.TriangleMesh],
    alpha_outer: float,
    bg_color_rgb: Tuple[float, float, float],
):
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    win = o3d.visualization.O3DVisualizer("Cartilage NURBS Template Viewer", 1280, 900)
    win.show_settings = True

    # BG requires float32 RGBA on some builds
    r, g, b = bg_color_rgb
    bg_rgba = np.array([r, g, b, 1.0], dtype=np.float32)
    try:
        win.set_background(bg_rgba)  # type: ignore
    except Exception:
        try:
            win.set_background_color(bg_rgba)  # type: ignore
        except Exception:
            pass

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
    win.add_geometry("axes", axis, o3d.visualization.rendering.MaterialRecord())

    if central_mesh is not None:
        cm = colorize(central_mesh, rgb=(0.95, 0.95, 0.95))
        mat_c = make_material((0.95, 0.95, 0.95, 1.0), transparent=False)
        win.add_geometry("central", cm, mat_c)

    if inner_mesh is not None:
        im = colorize(inner_mesh, rgb=(0.10, 0.60, 1.00))
        mat_in = make_material((0.10, 0.60, 1.00, 0.9), transparent=True)
        win.add_geometry("inner", im, mat_in)

    if outer_mesh is not None:
        om = colorize(outer_mesh, rgb=(1.00, 0.30, 0.10))
        mat_out = make_material(
            (1.00, 0.30, 0.10, float(np.clip(alpha_outer, 0.0, 1.0))),
            transparent=True
        )
        win.add_geometry("outer", om, mat_out)

    # Fit camera to all geometries
    bbox = None
    for name in ("central", "inner", "outer"):
        if win.has_geometry(name):
            g_bbox = win.get_geometry_bbox(name)
            bbox = g_bbox if bbox is None else bbox + g_bbox
    if bbox is not None:
        win.setup_camera(60.0, bbox, bbox.get_center())

    app.add_window(win)
    app.run()

# ----------------------------
# Legacy fallback (no material control)
# ----------------------------

def draw_legacy(
    central_mesh: Optional[o3d.geometry.TriangleMesh],
    inner_mesh: Optional[o3d.geometry.TriangleMesh],
    outer_mesh: Optional[o3d.geometry.TriangleMesh],
):
    geoms = []
    if central_mesh is not None:
        geoms.append(colorize(central_mesh, rgb=(0.85, 0.85, 0.85)))
    if inner_mesh is not None:
        geoms.append(colorize(inner_mesh, rgb=(0.10, 0.60, 1.00)))
    if outer_mesh is not None:
        geoms.append(colorize(outer_mesh, rgb=(1.00, 0.30, 0.10)))
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0))
    o3d.visualization.draw_geometries(geoms, window_name="Cartilage NURBS Template Viewer (legacy)")

# ----------------------------
# Offscreen PNG
# ----------------------------

def save_png_offscreen(
    central_mesh: Optional[o3d.geometry.TriangleMesh],
    inner_mesh: Optional[o3d.geometry.TriangleMesh],
    outer_mesh: Optional[o3d.geometry.TriangleMesh],
    png_path: Path,
    alpha_outer: float = 0.45,
    width: int = 1600,
    height: int = 1200,
    bg_color_rgb: Tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background(bg_color_rgb)

    # Simple lighting
    scene.scene.set_sun_light([-1, -1, -1], [1.0, 1.0, 1.0], 45000)
    scene.scene.enable_sun_light(True)

    if central_mesh is not None:
        cm = colorize(central_mesh, rgb=(0.90, 0.90, 0.90))
        scene.add_geometry(
            "central",
            cm,
            make_material((0.90, 0.90, 0.90, 1.0), transparent=False),
        )

    if inner_mesh is not None:
        im = colorize(inner_mesh, rgb=(0.10, 0.60, 1.00))
        scene.add_geometry(
            "inner",
            im,
            make_material((0.10, 0.60, 1.00, 0.9), transparent=True),
        )

    if outer_mesh is not None:
        om = colorize(outer_mesh, rgb=(1.00, 0.30, 0.10))
        scene.add_geometry(
            "outer",
            om,
            make_material(
                (1.00, 0.30, 0.10, float(np.clip(alpha_outer, 0.0, 1.0))),
                transparent=True,
            ),
        )

    # Compute bounding box over all geometries
    bbox = None
    for name in ("central", "inner", "outer"):
        if scene.has_geometry(name):
            g_bbox = scene.bounding_box(name)
            bbox = g_bbox if bbox is None else bbox + g_bbox
    if bbox is None:
        raise ValueError("No geometry loaded to render.")

    center = bbox.get_center()
    extent = np.linalg.norm(bbox.get_extent())
    cam = scene.camera
    cam_pos = center + np.array([0, 0, extent * 1.8])
    cam.look_at(center, cam_pos, np.array([0, 1, 0]))
    cam.set_projection(60.0, float(width) / float(height), 0.1, extent * 10.0)

    img = renderer.render_to_image()
    o3d.io.write_image(str(png_path), img)
    print(f"Saved PNG: {png_path}")

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Visualize NURBS cartilage template (central + thickness).")
    ap.add_argument("--manifest", type=str, default="", help="Path to template manifest JSON")
    ap.add_argument("--mode", type=str, default="central_inner_outer",
                    choices=["central", "inner_outer", "central_inner_outer"],
                    help="Which surfaces to show")
    ap.add_argument("--alpha_outer", type=float, default=0.45, help="Outer surface transparency (0..1)")
    ap.add_argument("--bg", type=str, default="dark", choices=["dark", "light"],
                    help="Background theme")
    ap.add_argument("--save_png", type=str, default="",
                    help="If set, save a PNG to this path (offscreen, no interactive window)")
    args = ap.parse_args()

    if not args.manifest:
        # Fallback default if not provided
        args.manifest = "femoral_cartilage_template_manifest.json"

    manifest_path = Path(args.manifest)
    bg_rgb = (0.0, 0.0, 0.0) if args.bg == "dark" else (1.0, 1.0, 1.0)

    # Load template geometry
    central_mesh, inner_mesh, outer_mesh = load_template_from_manifest(manifest_path, args.mode)

    if args.save_png:
        save_png_offscreen(
            central_mesh,
            inner_mesh,
            outer_mesh,
            Path(args.save_png),
            alpha_outer=args.alpha_outer,
            bg_color_rgb=bg_rgb,
        )
        return

    # Try modern draw(); if not available, try GUI O3DVisualizer; then legacy.
    try:
        _ = o3d.visualization.draw
        draw_modern(central_mesh, inner_mesh, outer_mesh, args.alpha_outer, bg_rgb)
    except Exception as e1:
        print(f"[Info] draw() path failed, trying GUI visualizer: {e1}")
        try:
            draw_gui(central_mesh, inner_mesh, outer_mesh, args.alpha_outer, bg_rgb)
        except Exception as e2:
            print(f"[Info] Falling back to legacy viewer: {e2}")
            draw_legacy(central_mesh, inner_mesh, outer_mesh)

if __name__ == "__main__":
    main()
