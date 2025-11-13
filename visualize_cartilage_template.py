#!/usr/bin/env python3
r"""
visualize_cartilage_template.py

Viewer for inner/outer cartilage template meshes with transparency and optional PNG export.
- Tries modern draw() first (no GUI init needed).
- If unavailable, uses O3DVisualizer with proper initialization and RGBA background.
- Falls back to legacy draw_geometries() if needed.

Usage (Windows CMD, one line):
  python visualize_cartilage_template.py ^
    --inner C:/Users/chris/MICCAI2026/Carti_Seg/femoral_cartilage_template_inner.ply ^
    --outer C:/Users/chris/MICCAI2026/Carti_Seg/femoral_cartilage_template_outer_offset2mm.ply ^
    --alpha_outer 0.45 --bg dark

Save PNG:
  python visualize_cartilage_template.py --inner ... --outer ... --save_png C:/path/to/out.png
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import open3d as o3d

# ----------------------------
# Helpers
# ----------------------------

def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    m = o3d.io.read_triangle_mesh(str(path))
    if not m.has_vertex_normals():
        m.compute_vertex_normals()
    if not m.has_triangle_normals():
        m.compute_triangle_normals()
    return m

def colorize(mesh: o3d.geometry.TriangleMesh, rgb=(1.0, 1.0, 1.0)):
    out = o3d.geometry.TriangleMesh(mesh)
    out.paint_uniform_color(rgb)
    return out

def make_material(rgba, transparent: bool):
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency" if transparent else "defaultLit"
    mat.base_color = tuple(float(c) for c in rgba)  # (r,g,b,a)
    mat.point_size = 2.0
    return mat

# ----------------------------
# Modern interactive draw()
# ----------------------------

def draw_modern(inner_path: Optional[Path], outer_path: Optional[Path],
                alpha_outer: float, bg_color_rgb: Tuple[float, float, float]):
    geoms = []
    if inner_path:
        inner = colorize(load_mesh(inner_path), rgb=(0.95, 0.95, 0.95))
        geoms.append({"name": "inner", "geometry": inner,
                      "material": make_material((0.95, 0.95, 0.95, 1.0), transparent=False)})
    if outer_path:
        outer = colorize(load_mesh(outer_path), rgb=(0.10, 0.60, 1.00))
        geoms.append({"name": "outer", "geometry": outer,
                      "material": make_material((0.10, 0.60, 1.00, float(np.clip(alpha_outer, 0.0, 1.0))), transparent=True)})
    geoms.append({"name": "axes", "geometry": o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)})

    o3d.visualization.draw(
        geoms,
        title="Cartilage Template Viewer",
        bg_color=bg_color_rgb,
        show_skybox=False,
    )

# ----------------------------
# O3DVisualizer interactive (needs GUI init, RGBA bg)
# ----------------------------

def draw_gui(inner_path: Optional[Path], outer_path: Optional[Path],
             alpha_outer: float, bg_color_rgb: Tuple[float, float, float]):
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    win = o3d.visualization.O3DVisualizer("Cartilage Template Viewer", 1280, 900)
    win.show_settings = True

    # BG requires float32 RGBA on some builds
    r, g, b = bg_color_rgb
    bg_rgba = np.array([r, g, b, 1.0], dtype=np.float32)
    try:
        # Newer API
        win.set_background(bg_rgba)  # type: ignore
    except Exception:
        try:
            # Older API variant
            win.set_background_color(bg_rgba)  # type: ignore
        except Exception:
            pass  # If both fail, keep default

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
    win.add_geometry("axes", axis, o3d.visualization.rendering.MaterialRecord())

    if inner_path:
        inner = colorize(load_mesh(inner_path), rgb=(0.95, 0.95, 0.95))
        mat_in = make_material((0.95, 0.95, 0.95, 1.0), transparent=False)
        win.add_geometry("inner", inner, mat_in)

    if outer_path:
        outer = colorize(load_mesh(outer_path), rgb=(0.10, 0.60, 1.00))
        mat_out = make_material((0.10, 0.60, 1.00, float(np.clip(alpha_outer, 0.0, 1.0))), transparent=True)
        win.add_geometry("outer", outer, mat_out)

    # Fit camera
    bbox = None
    for name in ("inner", "outer"):
        if win.has_geometry(name):
            g_bbox = win.get_geometry_bbox(name)
            bbox = g_bbox if bbox is None else bbox + g_bbox
    if bbox is not None:
        win.setup_camera(60.0, bbox, bbox.get_center())

    app.add_window(win)
    app.run()

# ----------------------------
# Legacy fallback (no transparency)
# ----------------------------

def draw_legacy(inner_path: Optional[Path], outer_path: Optional[Path]):
    geoms = []
    if inner_path:
        geoms.append(colorize(load_mesh(inner_path), rgb=(0.8, 0.8, 0.8)))
    if outer_path:
        geoms.append(colorize(load_mesh(outer_path), rgb=(0.1, 0.6, 1.0)))
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0))
    o3d.visualization.draw_geometries(geoms, window_name="Cartilage Template Viewer (legacy)")

# ----------------------------
# Offscreen PNG
# ----------------------------

def save_png_offscreen(inner_path: Optional[Path], outer_path: Optional[Path],
                       png_path: Path,
                       alpha_outer: float = 0.45,
                       width: int = 1600, height: int = 1200,
                       bg_color_rgb: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background(bg_color_rgb)

    scene.scene.set_sun_light([-1, -1, -1], [1.0, 1.0, 1.0], 45000)
    scene.scene.enable_sun_light(True)

    if inner_path:
        inner = colorize(load_mesh(inner_path), rgb=(0.9, 0.9, 0.9))
        scene.add_geometry("inner", inner, make_material((0.9, 0.9, 0.9, 1.0), transparent=False))

    if outer_path:
        outer = colorize(load_mesh(outer_path), rgb=(0.1, 0.6, 1.0))
        scene.add_geometry("outer", outer, make_material((0.1, 0.6, 1.0, float(np.clip(alpha_outer, 0.0, 1.0))), transparent=True))

    bbox = None
    for name in ("inner", "outer"):
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
    ap = argparse.ArgumentParser(description="Visualize cartilage template meshes (Open3D).")
    ap.add_argument("--inner", type=str, default="", help="Path to inner PLY mesh")
    ap.add_argument("--outer", type=str, default="", help="Path to outer PLY mesh")
    ap.add_argument("--alpha_outer", type=float, default=0.45, help="Outer transparency (0..1)")
    ap.add_argument("--bg", type=str, default="dark", choices=["dark", "light"], help="Background theme")
    ap.add_argument("--save_png", type=str, default="", help="If set, save a PNG to this path (offscreen)")
    args = ap.parse_args()

    bg_rgb = (0, 0, 0) if args.bg == "dark" else (1, 1, 1)
    inner_path = Path(args.inner) if args.inner else None
    outer_path = Path(args.outer) if args.outer else None

    if inner_path is None and outer_path is None:
        inner_path = Path("C:/Users/chris/MICCAI2026/Carti_Seg/femoral_cartilage_template_inner.ply")
        outer_path = Path("C:/Users/chris/MICCAI2026/Carti_Seg/femoral_cartilage_template_outer_offset2mm.ply")

    if args.save_png:
        save_png_offscreen(inner_path, outer_path, Path(args.save_png),
                           alpha_outer=args.alpha_outer, bg_color_rgb=bg_rgb)
        return

    # Try modern draw(); if not available, try GUI O3DVisualizer; then legacy.
    try:
        _ = o3d.visualization.draw
        draw_modern(inner_path, outer_path, args.alpha_outer, bg_rgb)
    except Exception as e1:
        print(f"[Info] draw() path failed, trying GUI visualizer: {e1}")
        try:
            draw_gui(inner_path, outer_path, args.alpha_outer, bg_rgb)
        except Exception as e2:
            print(f"[Info] Falling back to legacy viewer: {e2}")
            draw_legacy(inner_path, outer_path)

if __name__ == "__main__":
    main()
