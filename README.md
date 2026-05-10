# multicellularity_sim

CS2212 final project 

A 2D agent-based simulation of the emergence of multicellularity. Cells carry an
8-bit genome encoding a logical operation (AND/OR/XOR/NAND), an adhesion bit, and
a cooperator/defector bit. Cells join into clusters via kin-gated adhesion;
clusters earn shared rewards by chaining their members' operations to compute
multi-step tasks against a regional reward landscape. Defectors infiltrate
through mutation, drain the cluster, and force the question the simulation is
built around: under what conditions does cooperation persist?


## Setup

```bash
python -m venv sim_venv
source sim_venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py                                          # headless smoke run
python visualize.py                                     # full Napari GUI
python visualize.py --task-flip-period 100              # MVG mode
python experiment.py --sweep coop_cost --quiet          # one sweep → results/*.csv
python experiment.py --sweep all --ticks 1500 --quiet   # all sweeps
python graph_results.py --sweep coop_cost --save-dir figures/
```

