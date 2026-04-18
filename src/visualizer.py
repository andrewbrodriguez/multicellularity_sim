"""
Napari visualizer for the multicellularity simulation.

Layers
------
  "environment"  — (T, RES, RES, 3) Gaussian density heatmap (RGB).
                   Red = defectors, Blue = cooperators, Gray = lone cells.
  "Lone cells"   — point cloud, gray
  "Cooperators"  — point cloud, blue
  "Defectors"    — point cloud, red

All point coordinates are scaled to match the heatmap image pixels so the
two layers align perfectly.  The time-axis slider scrubs through frames.
"""

import numpy as np
from typing import Dict, List, Tuple

VIS_RES = 400   # pixel resolution of the rendered heatmap image

# RGBA for point layers
_PT_RGBA = {
    0: [0.60, 0.60, 0.60, 0.90],   # lone   – gray
    1: [0.15, 0.45, 1.00, 0.95],   # coop   – blue
    2: [1.00, 0.18, 0.18, 0.95],   # defect – red
}

# Linear-RGB colour used in the additive heatmap blend
_HMAP_RGB = {
    0: np.array([0.55, 0.55, 0.55]),
    1: np.array([0.10, 0.35, 1.00]),
    2: np.array([1.00, 0.15, 0.15]),
}

_LAYER_META = [
    (0, "Lone cells",  _PT_RGBA[0]),
    (1, "Cooperators", _PT_RGBA[1]),
    (2, "Defectors",   _PT_RGBA[2]),
]


# ── heatmap renderer ──────────────────────────────────────────────────────────

def _render_heatmap(
    records: List[Dict],
    world_w: float,
    world_h: float,
    res: int,
) -> np.ndarray:
    """
    Returns (res, res, 3) uint8 RGB heatmap for one time step.
    Each cell contributes a Gaussian blob at its position; types are
    blended additively in their respective colours.
    """
    from scipy.ndimage import gaussian_filter

    sx = res / world_w
    sy = res / world_h
    sigma = res / world_w * 1.1          # ~1.1 world-unit blur radius

    density: Dict[int, np.ndarray] = {t: np.zeros((res, res), np.float32) for t in range(3)}

    for rec in records:
        x, y  = rec["pos"]
        r = int(x * sx) % res
        c = int(y * sy) % res
        density[rec["type"]][r, c] += 1.0

    blurred = {t: gaussian_filter(d, sigma=sigma) for t, d in density.items()}

    glob_max = max(b.max() for b in blurred.values()) + 1e-10

    img = np.zeros((res, res, 3), np.float32)
    for t, color in _HMAP_RGB.items():
        img += (blurred[t] / glob_max)[:, :, np.newaxis] * color

    img = np.clip(img + 0.04, 0.0, 1.0)          # faint dark background
    return (img * 255).astype(np.uint8)


# ── data builder ──────────────────────────────────────────────────────────────

def build_napari_data(
    snapshots: List[Dict],
    grid_width: float,
    grid_height: float,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Returns
    -------
    image_stack : (T, VIS_RES, VIS_RES, 3) uint8
    points_by_type : {type_code → (N, 3) float32}  coords = [t_idx, px, py]
        px, py are already scaled to [0, VIS_RES) to match image pixels.
    """
    T  = len(snapshots)
    sx = VIS_RES / grid_width
    sy = VIS_RES / grid_height

    image_stack = np.zeros((T, VIS_RES, VIS_RES, 3), dtype=np.uint8)
    pts: Dict[int, List] = {0: [], 1: [], 2: []}

    for t_idx, snap in enumerate(snapshots):
        records = snap.get("cell_records", [])
        if records:
            image_stack[t_idx] = _render_heatmap(records, grid_width, grid_height, VIS_RES)
            for rec in records:
                x, y = rec["pos"]
                pts[rec["type"]].append([float(t_idx), x * sx, y * sy])

    points_by_type = {
        k: (np.array(v, dtype=np.float32) if v else np.empty((0, 3), dtype=np.float32))
        for k, v in pts.items()
    }
    return image_stack, points_by_type


# ── viewer ────────────────────────────────────────────────────────────────────

def launch_viewer(
    snapshots: List[Dict],
    grid_width: float,
    grid_height: float,
) -> None:
    try:
        import napari
    except ImportError:
        raise ImportError("napari is required: pip install 'napari[all]'")

    if not snapshots:
        print("No snapshots to display.")
        return

    ticks = [s["tick"] for s in snapshots]
    T     = len(ticks)
    print(f"Building Napari data for {T} frames (ticks {ticks[0]}–{ticks[-1]})…")

    image_stack, points_by_type = build_napari_data(snapshots, grid_width, grid_height)

    viewer = napari.Viewer(title="Multicellularity Simulation")

    # background heatmap
    viewer.add_image(
        image_stack,
        name="environment",
        rgb=True,
        opacity=1.0,
        blending="opaque",
    )

    # one points layer per cell type → automatic legend + individual toggles
    pt_size = VIS_RES / grid_width * 0.55   # ~0.55 world-unit radius in pixels
    for ctype, layer_name, color in _LAYER_META:
        pts = points_by_type[ctype]
        if len(pts) == 0:
            continue
        viewer.add_points(
            pts,
            ndim=3,
            name=layer_name,
            face_color=color,
            edge_color=[0.0, 0.0, 0.0, 0.0],
            size=pt_size,
            symbol="disc",
            blending="translucent",
        )

    viewer.dims.axis_labels = ("time", "x", "y")

    def _title(t_idx: int) -> str:
        s = snapshots[t_idx]
        return (
            f"Tick {ticks[t_idx]:5d}  |  "
            f"Cells: {s.get('total_cells','?'):4}  |  "
            f"Clusters: {s.get('num_clusters','?'):3}  |  "
            f"Coop: {s.get('cooperator_pct', 0.0):5.1f}%  |  "
            f"Def: {s.get('defector_pct', 0.0):5.1f}%"
        )

    viewer.title = _title(0)

    @viewer.dims.events.current_step.connect
    def _on_slider(_event) -> None:
        t = viewer.dims.current_step[0]
        if 0 <= t < T:
            viewer.title = _title(t)

    napari.run()
