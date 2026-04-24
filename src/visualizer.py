"""
Napari visualizer for the multicellularity simulation.

Layers
------
  "environment"    — (T, RES, RES, 3) Gaussian density heatmap on agarose background.
                     Red = defectors, Blue = cooperators, Dark gray = lone cells.
  "Adhesion bonds" — time-resolved line segments between nearest cluster-mates only.
  "Lone cells"     — point cloud, gray
  "Cooperators"    — point cloud, blue
  "Defectors"      — point cloud, red

Dock widget (right panel) shows reward system constants and the current task,
updating live as the time slider is scrubbed.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

VIS_RES = 400   # pixel resolution of the rendered heatmap image

# Warm pale yellow — colour of agarose gel under bright-field illumination
AGAROSE_BG = np.array([1.0, 0.97, 0.82], dtype=np.float32)

# RGBA for point layers (shown on top of heatmap)
_PT_RGBA = {
    0: [0.50, 0.50, 0.50, 0.90],   # lone   – gray
    1: [0.15, 0.45, 1.00, 0.95],   # coop   – blue
    2: [1.00, 0.18, 0.18, 0.95],   # defect – red
}

# Linear-RGB colours for the Gaussian density heatmap
_HMAP_RGB = {
    0: np.array([0.35, 0.35, 0.35], dtype=np.float32),  # lone   – dark gray
    1: np.array([0.10, 0.35, 1.00], dtype=np.float32),  # coop   – blue
    2: np.array([1.00, 0.15, 0.15], dtype=np.float32),  # defect – red
}

_LAYER_META = [
    (0, "Lone cells",  _PT_RGBA[0]),
    (1, "Cooperators", _PT_RGBA[1]),
    (2, "Defectors",   _PT_RGBA[2]),
]

# Edge colour encodes the logical operation each cell carries in its genome
_OP_EDGE_RGBA: Dict[str, List[float]] = {
    "AND":  [0.05, 0.88, 0.35, 1.0],  # green
    "OR":   [1.00, 0.55, 0.00, 1.0],  # orange
    "XOR":  [0.72, 0.10, 1.00, 1.0],  # purple
    "NAND": [0.00, 0.88, 0.95, 1.0],  # cyan
}
_OP_HEX = {"AND": "#0EE059", "OR": "#FF8C00", "XOR": "#B81AFF", "NAND": "#00E0F2"}


# RGBA fill colours for each operation (semi-transparent region tiles)
_OP_REGION_RGBA = {
    "AND":  np.array([0.05, 0.88, 0.35, 0.28], dtype=np.float32),
    "OR":   np.array([1.00, 0.55, 0.00, 0.28], dtype=np.float32),
    "XOR":  np.array([0.72, 0.10, 1.00, 0.28], dtype=np.float32),
    "NAND": np.array([0.00, 0.88, 0.95, 0.28], dtype=np.float32),
}
# Slightly brighter for multi-step tasks (more alpha)
_STEP_ALPHA = {1: 0.20, 2: 0.32, 3: 0.45}


_ALL_TASKS_VIS = [
    ("AND",), ("OR",), ("XOR",), ("NAND",),
    ("AND", "XOR"), ("NAND", "OR"), ("OR", "AND"), ("XOR", "NAND"),
    ("AND", "OR", "XOR"), ("OR", "NAND", "AND"),
    ("XOR", "AND", "OR"),  ("NAND", "XOR", "AND"),
]
_OP_RGB = {
    "AND":  np.array([0.05, 0.88, 0.35], dtype=np.float32),
    "OR":   np.array([1.00, 0.55, 0.00], dtype=np.float32),
    "XOR":  np.array([0.72, 0.10, 1.00], dtype=np.float32),
    "NAND": np.array([0.00, 0.88, 0.95], dtype=np.float32),
}


def _render_task_map(
    regional_rewards: np.ndarray,  # (n_rows, n_cols, 12) float32, each row sums to 15
    grid_width: float,
    grid_height: float,
    res: int,
) -> np.ndarray:
    """
    Returns (res, res, 4) RGBA uint8 for the reward-vector landscape.

    Hue   = weighted blend of op colours, weighted by total reward attributed
            to each op across all tasks in the vector.
    Alpha = 0.15 + 0.40 * (dominant_op_share)  — brighter = more specialised.
    """
    n_rows, n_cols, _ = regional_rewards.shape
    img  = np.zeros((res, res, 4), dtype=np.float32)
    px_r = res / n_rows
    px_c = res / n_cols

    # Precompute per-task → op index mapping (first op of each task)
    ops = ["AND", "OR", "XOR", "NAND"]
    task_op_idx = np.array([ops.index(t[0]) for t in _ALL_TASKS_VIS], dtype=np.int32)

    for r in range(n_rows):
        for c in range(n_cols):
            vec = regional_rewards[r, c]          # (12,)
            op_weights = np.zeros(4, dtype=np.float32)
            for i, oi in enumerate(task_op_idx):
                op_weights[oi] += vec[i]

            total = op_weights.sum()
            if total == 0:
                continue

            # Blend RGB weighted by op contribution
            rgb = np.zeros(3, dtype=np.float32)
            for oi, op in enumerate(ops):
                rgb += (op_weights[oi] / total) * _OP_RGB[op]

            dominant_share = op_weights.max() / total   # 0.25 (uniform) → 1.0 (pure)
            alpha = 0.15 + 0.45 * dominant_share

            r0, r1 = int(r * px_r), int((r + 1) * px_r)
            c0, c1 = int(c * px_c), int((c + 1) * px_c)
            img[r0:r1, c0:c1, :3] = rgb
            img[r0:r1, c0:c1,  3] = alpha

            # 1-px tile border
            img[r0,   c0:c1] = [0.08, 0.08, 0.08, 0.55]
            img[r1-1, c0:c1] = [0.08, 0.08, 0.08, 0.55]
            img[r0:r1, c0]   = [0.08, 0.08, 0.08, 0.55]
            img[r0:r1, c1-1] = [0.08, 0.08, 0.08, 0.55]

    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


# ── heatmap renderer ──────────────────────────────────────────────────────────

def _render_heatmap(
    records: List[Dict],
    world_w: float,
    world_h: float,
    res: int,
) -> np.ndarray:
    """
    Returns (res, res, 3) uint8 RGB heatmap for one time step.
    Cells are Gaussian blobs alpha-composited over the agarose background.
    """
    from scipy.ndimage import gaussian_filter

    sx = res / world_w
    sy = res / world_h
    sigma = res / world_w * 0.3

    density: Dict[int, np.ndarray] = {t: np.zeros((res, res), np.float32) for t in range(3)}
    for rec in records:
        x, y = rec["pos"]
        r = int(x * sx) % res
        c = int(y * sy) % res
        density[rec["type"]][r, c] += 1.0

    blurred = {t: gaussian_filter(d, sigma=sigma) for t, d in density.items()}
    glob_max = max(b.max() for b in blurred.values()) + 1e-10

    # Additive cell-colour layer
    cell_img = np.zeros((res, res, 3), np.float32)
    for t, color in _HMAP_RGB.items():
        cell_img += (blurred[t] / glob_max)[:, :, np.newaxis] * color

    # Alpha: how much cell colour displaces the agarose background
    alpha = np.clip(cell_img.max(axis=2, keepdims=True) * 1.8, 0.0, 1.0)

    # Composite cell layer over agarose
    img = AGAROSE_BG * (1.0 - alpha) + np.clip(cell_img, 0.0, 1.0) * alpha
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


# ── data builder ──────────────────────────────────────────────────────────────

def build_napari_data(
    snapshots: List[Dict],
    grid_width: float,
    grid_height: float,
) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray], List[np.ndarray]]:
    """
    Returns
    -------
    image_stack       : (T, VIS_RES, VIS_RES, 3) uint8 heatmap frames
    points_by_type    : {type_code → (N, 3) float32}  coords = [t_idx, px, py]
    edge_colors_by_type: {type_code → (N, 4) float32} RGBA edge colour per point,
                         encoding the cell's logical operation (AND/OR/XOR/NAND).
    bond_lines        : list of (2, 3) float32 arrays  [[t,x1,y1],[t,x2,y2]]
    """
    T  = len(snapshots)
    # sx/sy only used for heatmap pixel rendering; points and bonds use world (µm) coords directly
    sx = VIS_RES / grid_width
    sy = VIS_RES / grid_height

    image_stack = np.zeros((T, VIS_RES, VIS_RES, 3), dtype=np.uint8)
    pts:   Dict[int, List] = {0: [], 1: [], 2: []}
    ecols: Dict[int, List] = {0: [], 1: [], 2: []}
    sizes: Dict[int, List] = {0: [], 1: [], 2: []}
    bond_lines: List[np.ndarray] = []

    # Sizes in µm — range from small (newborn) to larger (near division).
    # 25% of the cell's physical 10 µm diameter gives ~2.5 µm, clearly visible
    # without crowding at the 250 µm world scale.
    _PT_SIZE_MIN = 0.75
    _PT_SIZE_MAX = 2.25

    for t_idx, snap in enumerate(snapshots):
        records = snap.get("cell_records", [])
        if records:
            image_stack[t_idx] = _render_heatmap(records, grid_width, grid_height, VIS_RES)
            for rec in records:
                x, y = rec["pos"]
                t = rec["type"]
                # Store in world (µm) coords — matches the image layers which use scale=(µm/px)
                pts[t].append([float(t_idx), x, y])
                ecols[t].append(_OP_EDGE_RGBA.get(rec.get("op", "AND"), [1, 1, 1, 1]))
                dp = rec.get("div_progress", 0.0)
                sizes[t].append(_PT_SIZE_MIN + (_PT_SIZE_MAX - _PT_SIZE_MIN) * dp)

        # Adhesion bonds — nearest-neighbour edges within each cluster only.
        for group in snap.get("cluster_groups", []):
            if len(group) < 2:
                continue
            pos = np.array(group, dtype=np.float32)   # (N, 2) world (µm) coords
            drawn: set = set()
            for i in range(len(pos)):
                dists = np.linalg.norm(pos - pos[i], axis=1)
                dists[i] = np.inf
                j = int(np.argmin(dists))
                edge = (min(i, j), max(i, j))
                if edge not in drawn:
                    drawn.add(edge)
                    bond_lines.append(np.array([
                        [float(t_idx), pos[i][0], pos[i][1]],
                        [float(t_idx), pos[j][0], pos[j][1]],
                    ], dtype=np.float32))

    points_by_type = {
        k: (np.array(v, dtype=np.float32) if v else np.empty((0, 3), dtype=np.float32))
        for k, v in pts.items()
    }
    edge_colors_by_type = {
        k: (np.array(v, dtype=np.float32) if v else np.empty((0, 4), dtype=np.float32))
        for k, v in ecols.items()
    }
    sizes_by_type = {
        k: (np.array(v, dtype=np.float32) if v else np.empty((0,), dtype=np.float32))
        for k, v in sizes.items()
    }
    return image_stack, points_by_type, edge_colors_by_type, bond_lines, sizes_by_type


# ── info dock widget ──────────────────────────────────────────────────────────

def _make_info_widget(reward_params: Optional[Dict]) -> Tuple:
    """
    Returns (container_widget, task_label, selection_label).
    Both labels should be updated by the caller as the time slider moves.
    """
    from qtpy.QtWidgets import QLabel, QWidget, QVBoxLayout
    from qtpy.QtCore import Qt

    rp       = reward_params or {}
    base     = rp.get("base",      1.0)
    simple   = rp.get("simple",    3.0)
    complex_ = rp.get("complex",   5.0)
    triple   = rp.get("triple",    7.0)
    drain    = rp.get("drain",     1.5)
    coop     = rp.get("coop_cost", 0.3)

    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setAlignment(Qt.AlignTop)
    layout.setSpacing(6)
    layout.setContentsMargins(12, 12, 12, 12)

    # hex colours for region tile swatches (opaque versions of the translucent tile fill)
    _tile_hex = {"AND": "#0EE059", "OR": "#FF8C00", "XOR": "#B81AFF", "NAND": "#00E0F2"}

    def _swatch(hex_col: str, alpha: float) -> str:
        """Inline coloured square for the legend."""
        return (
            f"<span style='"
            f"display:inline-block;width:14px;height:14px;"
            f"background:{hex_col};opacity:{alpha:.2f};"
            f"border:1px solid #555;vertical-align:middle;margin-right:6px;"
            f"'></span>"
        )

    reward_html = (
        "<style>"
        "  * { font-family: 'Courier New', monospace; font-size: 11px; }"
        "  td.k { color: #555; padding-right: 10px; white-space: nowrap; }"
        "  td.v { font-weight: bold; }"
        "  hr   { border: 0; border-top: 1px solid #ccc; margin: 5px 0; }"
        "  .sec { font-size: 12px; font-weight: bold; }"
        "</style>"

        "<span class='sec'>REWARD SYSTEM</span><hr>"
        "<table cellspacing='2'>"
        f"<tr><td class='k'>Survival (lone)</td><td class='v'>+{base:.1f}/tick</td></tr>"
        f"<tr><td class='k'>1-step task</td>    <td class='v'>+{simple:.1f}/tick</td></tr>"
        f"<tr><td class='k'>2-step task</td>    <td class='v'>+{complex_:.1f}/tick</td></tr>"
        f"<tr><td class='k'>3-step task</td>    <td class='v'>+{triple:.1f}/tick</td></tr>"
        f"<tr><td class='k'>Defector drain</td> <td class='v'>&minus;{drain:.1f}/defector</td></tr>"
        f"<tr><td class='k'>Coop cost</td>      <td class='v'>&minus;{coop:.1f}/tick</td></tr>"
        "</table>"

        "<br><span class='sec'>REWARD VECTOR REGIONS</span><hr>"
        "<small style='color:#555;'>Each tile = 12-entry vector summing to 15.<br>"
        "Entries = reward pts per task combination.</small><br><br>"
        "<b style='font-size:11px;'>Tile colour = dominant op blend</b><br>"
        "<table cellspacing='3'>"
        f"<tr><td>{_swatch(_tile_hex['AND'],  0.9)}</td><td><b>AND</b></td> <td style='color:#555'>A &amp; B</td></tr>"
        f"<tr><td>{_swatch(_tile_hex['OR'],   0.9)}</td><td><b>OR</b></td>  <td style='color:#555'>A | B</td></tr>"
        f"<tr><td>{_swatch(_tile_hex['XOR'],  0.9)}</td><td><b>XOR</b></td> <td style='color:#555'>A ^ B</td></tr>"
        f"<tr><td>{_swatch(_tile_hex['NAND'], 0.9)}</td><td><b>NAND</b></td><td style='color:#555'>~(A&amp;B)</td></tr>"
        "</table>"
        "<small style='color:#888;'>"
        "Mixed colours = shared reward across ops.<br>"
        "Brighter tile = more specialised region.<br>"
        "2-step idx 4-7, 3-step idx 8-11."
        "</small>"

        "<br><br><span class='sec'>CELL GENOME</span><hr>"
        "<table cellspacing='3'>"
        f"<tr><td><span style='color:{_OP_HEX['AND']};font-size:15px;'>&#9679;</span></td>"
        f"    <td><b>AND</b></td></tr>"
        f"<tr><td><span style='color:{_OP_HEX['OR']};font-size:15px;'>&#9679;</span></td>"
        f"    <td><b>OR</b></td></tr>"
        f"<tr><td><span style='color:{_OP_HEX['XOR']};font-size:15px;'>&#9679;</span></td>"
        f"    <td><b>XOR</b></td></tr>"
        f"<tr><td><span style='color:{_OP_HEX['NAND']};font-size:15px;'>&#9679;</span></td>"
        f"    <td><b>NAND</b></td></tr>"
        "</table>"
        "<small style='color:#888;'>(point edge colour = encoded op)</small>"
    )
    reward_label = QLabel(reward_html)
    reward_label.setWordWrap(True)

    task_label = QLabel()
    task_label.setWordWrap(True)

    selection_label = QLabel()
    selection_label.setWordWrap(True)

    layout.addWidget(reward_label)
    layout.addSpacing(6)
    layout.addWidget(task_label)
    layout.addSpacing(6)
    layout.addWidget(selection_label)
    layout.addStretch()

    return widget, task_label, selection_label


def _task_html(task_str: str) -> str:
    return (
        "<style>"
        "  * { font-family: 'Courier New', monospace; font-size: 12px; }"
        "  hr { border: 0; border-top: 1px solid #ccc; margin: 6px 0; }"
        "  .task { font-size: 13px; font-weight: bold; color: #224488; }"
        "</style>"
        "<b style='font-size:13px;'>CURRENT TASK</b><hr>"
        f"<span class='task'>{task_str}</span>"
    )


def _selection_html(s: Dict) -> str:
    """
    Per-phenotype fitness rate (fitness/age) and the two selection-pressure
    signals: group-selection gradient (multi_advantage) and individual-
    selection gradient (coop_advantage).
    """
    def _col(v: float) -> str:
        if v > 0.05:
            return "#1a8a3a"   # green
        if v < -0.05:
            return "#b83030"   # red
        return "#555"

    def _sign(v: float) -> str:
        return f"+{v:.2f}" if v >= 0 else f"{v:.2f}"

    multi = s.get("multi_advantage", 0.0)
    coop  = s.get("coop_advantage",  0.0)

    return (
        "<style>"
        "  * { font-family: 'Courier New', monospace; font-size: 11px; }"
        "  td.k { color: #555; padding-right: 10px; white-space: nowrap; }"
        "  td.v { font-weight: bold; text-align: right; }"
        "  hr   { border: 0; border-top: 1px solid #ccc; margin: 5px 0; }"
        "  .sec { font-size: 12px; font-weight: bold; }"
        "  .note{ color: #777; font-size: 10px; }"
        "</style>"

        "<span class='sec'>SELECTION PRESSURE</span><hr>"
        "<span class='note'>fitness per tick, by phenotype</span>"
        "<table cellspacing='2'>"
        f"<tr><td class='k'>Clust. coop</td><td class='v'>{s.get('coop_rate_clustered',0.0):.2f}</td></tr>"
        f"<tr><td class='k'>Clust. def&nbsp;</td><td class='v'>{s.get('def_rate_clustered',0.0):.2f}</td></tr>"
        f"<tr><td class='k'>Lone coop</td> <td class='v'>{s.get('coop_rate_lone',0.0):.2f}</td></tr>"
        f"<tr><td class='k'>Lone def&nbsp;</td><td class='v'>{s.get('def_rate_lone',0.0):.2f}</td></tr>"
        "</table>"

        "<hr>"
        "<table cellspacing='2'>"
        f"<tr><td class='k'>Group gradient<br><span class='note'>(cluster &minus; lone)</span></td>"
        f"    <td class='v' style='color:{_col(multi)};'>{_sign(multi)}</td></tr>"
        f"<tr><td class='k'>Indiv. gradient<br><span class='note'>(coop &minus; def in cluster)</span></td>"
        f"    <td class='v' style='color:{_col(coop)};'>{_sign(coop)}</td></tr>"
        "</table>"
        "<span class='note'>"
        "Green = this strategy is currently paying.<br>"
        "Both green &rarr; multicellular cooperation wins."
        "</span>"
    )


# ── viewer ────────────────────────────────────────────────────────────────────

def launch_viewer(
    snapshots: List[Dict],
    grid_width: float,
    grid_height: float,
    reward_params: Optional[Dict] = None,
    cell_radius: float = 0.5,
    regional_tasks: Optional[List] = None,
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

    image_stack, points_by_type, edge_colors_by_type, bond_lines, sizes_by_type = build_napari_data(
        snapshots, grid_width, grid_height
    )

    viewer = napari.Viewer(title="Multicellularity Simulation")

    # µm per pixel — napari spatial scale for 2D layers (time axis is handled by dims slider)
    um_per_px = grid_width / VIS_RES   # e.g. 1000 µm / 400 px = 2.5 µm/px
    spatial_scale = (um_per_px, um_per_px)

    # Set agarose background on the VisPy canvas (visible when zoomed out)
    try:
        viewer.window._qt_viewer.canvas.bgcolor = (
            float(AGAROSE_BG[0]), float(AGAROSE_BG[1]), float(AGAROSE_BG[2]), 1.0
        )
    except Exception:
        pass

    # ── image layer ───────────────────────────────────────────────────────────
    viewer.add_image(
        image_stack,
        name="environment",
        rgb=True,
        opacity=1.0,
        blending="opaque",
        scale=spatial_scale,
    )

    # ── regional reward map (static — generated once at startup) ─────────────
    if regional_tasks is not None:
        task_map = _render_task_map(regional_tasks, grid_width, grid_height, VIS_RES)
        viewer.add_image(
            task_map,
            name="Reward regions",
            rgb=True,
            opacity=0.7,
            blending="translucent",
            scale=spatial_scale,
        )

    # ── native scale bar ──────────────────────────────────────────────────────
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit    = "µm"

    # ── adhesion bonds (clustered cells only) ─────────────────────────────────
    if bond_lines:
        bonds_layer = viewer.add_shapes(
            data=bond_lines,
            shape_type="line",
            ndim=3,
            edge_color="#5588CC",
            edge_width=0.5,
            face_color="transparent",
            name="Adhesion bonds",
        )
        bonds_layer.opacity = 0.70

    # ── border rectangle (simulation wall) — in µm coords ────────────────────
    border = np.array([[0, 0], [0, grid_height], [grid_width, grid_height], [grid_width, 0]], dtype=float)
    border_layer = viewer.add_shapes(
        [border],
        shape_type="rectangle",
        ndim=2,
        edge_color="#222222",
        face_color="transparent",
        edge_width=2,
        name="Border",
    )
    border_layer.opacity = 0.85

    # ── cell point clouds ─────────────────────────────────────────────────────
    for ctype, layer_name, face_color in _LAYER_META:
        pts = points_by_type[ctype]
        if len(pts) == 0:
            continue
        ecols = edge_colors_by_type[ctype]
        viewer.add_points(
            pts,
            ndim=3,
            name=layer_name,
            face_color=face_color,
            edge_color=ecols,
            edge_width=0.05,
            size=sizes_by_type[ctype] * 3.5,
            symbol="disc",
            blending="translucent",
        )

    viewer.dims.axis_labels = ("time", "x", "y")

    # ── dock widget ───────────────────────────────────────────────────────────
    info_widget, task_label, selection_label = _make_info_widget(reward_params)
    viewer.window.add_dock_widget(info_widget, name="Simulation Info", area="right")
    task_label.setText(_task_html(snapshots[0].get("current_task", "(A AND B) XOR C")))
    selection_label.setText(_selection_html(snapshots[0]))

    # ── title bar + slider callback ───────────────────────────────────────────
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
            task_label.setText(_task_html(snapshots[t].get("current_task", "(A AND B) XOR C")))
            selection_label.setText(_selection_html(snapshots[t]))

    napari.run()
