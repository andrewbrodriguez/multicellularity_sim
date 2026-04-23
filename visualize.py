"""
Entry point for the Napari visualizer.

Examples
--------
# default: sparse start so early competition is visible
python visualize.py

# record every single tick — full temporal resolution
python visualize.py --sample-every 1 --ticks 300

# denser start (less empty space, faster cluster formation)
python visualize.py --initial-cells 100 --ticks 600

# MVG extension: test environment fluctuation
python visualize.py --task-flip-period 100 --initial-cells 100 --ticks 600
"""

import argparse
from src.simulation import Simulation
from src.visualizer import launch_viewer
from src.environment import BASE_REWARD, SIMPLE_REWARD, COMPLEX_REWARD, TRIPLE_REWARD, DEFECTOR_DRAIN, COOP_COST, REPULSION_RADIUS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multicellularity simulation and open Napari viewer."
    )
    parser.add_argument("--grid-size",      type=int,   default=50,
                        help="Width and height of the 2-D spatial grid (default: 50)")
    parser.add_argument("--initial-cells",  type=int,   default=40,
                        help="Starting cell count (default: 40)")
    parser.add_argument("--mutation-rate",  type=float, default=0.005,
                        help="Per-bit flip probability per replication (default: 0.02)")
    parser.add_argument("--ticks",          type=int,   default=500,
                        help="Total simulation ticks to run (default: 500)")
    parser.add_argument("--sample-every",   type=int,   default=10,
                        help="Record a frame every N ticks; 1=every tick (default: 10)")
    parser.add_argument("--seed",           type=int,   default=42,
                        help="RNG seed for reproducibility (default: 42)")
    parser.add_argument("--quiet",          action="store_true",
                        help="Suppress per-tick console output")
    
    # New MVG Extension Argument
    parser.add_argument("--task-flip-period", type=int, default=None,
                        help="Ticks between environmental task structure flips (MVG extension)")
    
    args = parser.parse_args()

    # Formatted terminal output for the new argument
    flip_status = f"  task_flip_period={args.task_flip_period}" if args.task_flip_period else "  task_flip_period=None (Static)"

    print(
        f"Multicellularity Simulation\n"
        f"  grid={args.grid_size}×{args.grid_size}  "
        f"cells={args.initial_cells}  "
        f"ticks={args.ticks}  "
        f"sample_every={args.sample_every}  "
        f"seed={args.seed}\n"
        f"{flip_status}\n"
        f"{'─' * 75}"
    )

    # Pass the new parameter into the Simulation instantiation
    sim = Simulation(
        grid_size=args.grid_size,
        initial_cells=args.initial_cells,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
        task_flip_period=args.task_flip_period,
    )

    original_print = sim._print_stats
    if args.quiet:
        sim._print_stats = lambda _s: None   # silence tick output

    sim.run(ticks=args.ticks, record_every=args.sample_every)

    if args.quiet:
        sim._print_stats = original_print

    n_frames = len(sim.history)
    print(f"{'─' * 75}")
    print(f"Collected {n_frames} frames.  Launching Napari…\n")
    print("Controls:")
    print("  • Scrub the bottom slider to move through time")
    print("  • Toggle layer visibility (eye icon) to isolate lone/coop/defector cells")
    print("  • Scroll / pinch to zoom; click-drag to pan")

    launch_viewer(
        snapshots=sim.history,
        grid_width=args.grid_size,
        grid_height=args.grid_size,
        cell_radius=REPULSION_RADIUS,
        regional_tasks=sim.env.regional_rewards,
        reward_params={
            "base":     BASE_REWARD,
            "simple":   SIMPLE_REWARD,
            "complex":  COMPLEX_REWARD,
            "triple":   TRIPLE_REWARD,
            "drain":    DEFECTOR_DRAIN,
            "coop_cost": COOP_COST,
        },
    )


if __name__ == "__main__":
    main()