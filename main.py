from src.simulation import Simulation


def main() -> None:
    sim = Simulation(
        grid_size=50,
        initial_cells=40,
        mutation_rate=0.01,
        seed=42,
    )
    print("Multicellularity Simulation")
    print("Genome: 8-bit  |  Task: (A AND B) XOR C  |  Replication: cluster-level")
    print("-" * 75)
    stats = sim.run(ticks=500, record_every=25)
    print("-" * 75)

    final = stats[-1]
    print(f"\nFinal state (tick {final['tick']}):")
    for k, v in final.items():
        print(f"  {k:<20}: {v:.2f}" if isinstance(v, float) else f"  {k:<20}: {v}")


if __name__ == "__main__":
    main()
