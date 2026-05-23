# RZ Neuron Interactive Simulator

An interactive Python tool for exploring the **Resonate-and-Fire (RZ)** neuron model using [Brian2](https://brian2.readthedocs.io/). Three sliders let you tune the inter-spike interval (dt), damping (b_res), and resonance frequency (W_res) in real time, and watch how the neuron responds.

---

## Table of Contents

- [Background – The RZ Neuron Model](#background--the-rz-neuron-model)
- [What the Simulation Does](#what-the-simulation-does)
- [Parameters & Sliders](#parameters--sliders)
- [Installation](#installation)
- [Usage](#usage)
- [Quick Examples](#quick-examples)
- [Code Structure](#code-structure)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Background – The RZ Neuron Model

The **Resonate-and-Fire** model was introduced by Izhikevich (2001) as a minimal model that captures **sub-threshold oscillations** – a key feature of many biological neurons. Unlike the simpler Integrate-and-Fire model, an RZ neuron is *frequency selective*: it fires preferentially when its inputs arrive at its resonant frequency.

### Equations

The two state variables form a **complex oscillator**:

```
dv/dt = b_res · v  −  W_res · u
du/dt = W_res · v  +  b_res · u
```

This is equivalent to writing a single complex equation:

```
d(v + i·u)/dt = (b_res + i·W_res)(v + i·u)
```

| Symbol | Role | Sign convention |
|--------|------|-----------------|
| `v`    | Membrane potential (real part) | Fires when `v > 1` |
| `u`    | Auxiliary variable (imaginary part) | Reset to 0 at spike |
| `b_res`| Damping coefficient | **Negative** → stable decay; **Positive** → unstable growth |
| `W_res`| Resonance (angular) frequency | Higher → faster oscillation |

### Threshold & Reset

```
Threshold : v > 1
Reset     : v = 1,  u = 0
```

Each input spike adds **0.7** to `v` via the synapse `v += 0.7`.

---

## What the Simulation Does

```
t = 0 ms
  │
  ├─ 10 ms    → Spike 1 arrives (v jumps +0.7)
  ├─ 10+dt ms → Spike 2 arrives (v jumps +0.7)
  └─ 10+2·dt ms → Spike 3 arrives (v jumps +0.7)

Simulation runs for 10·dt ms total.
```

Three input spikes are delivered to the RZ neuron. Whether the neuron fires depends on whether those spikes arrive in phase with its natural oscillation. The plot shows:

1. **Membrane potential trace** – the evolution of `v` over time, with the threshold line.
2. **Input raster** – the three input spike times (from `P`).
3. **Output raster** – any spikes fired by the RZ neuron.

---

## Parameters & Sliders

| Slider | Range | Default | Effect |
|--------|-------|---------|--------|
| `dt` | 140 – 200 ms | 144 | Inter-spike interval of the input pulse train |
| `b_res` | −11 – 0 | −5 | Damping: closer to 0 = less decay between spikes |
| `W_res` | 0 – 50 | 40 | Resonance frequency: higher = faster oscillation |

### Intuition

- **Matching dt to 1/W_res** causes spikes to arrive *in phase* with the oscillation, making each subsequent spike add constructively → higher peak `v` → more likely to fire.
- **More negative b_res** damps the oscillation faster → the "memory" of previous spikes fades quicker.
- **b_res → 0** gives a nearly undamped oscillator → the neuron accumulates sub-threshold oscillations across many spikes.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/rz-neuron-simulator.git
cd rz-neuron-simulator

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate.bat      # Windows

# 3. Install dependencies
pip install brian2 matplotlib numpy

# 4. (Optional) install an interactive Qt backend for the live sliders
pip install pyqt5               # or: pip install pyqt6
```

> **Python version**: 3.8 or later recommended.

---

## Usage

```bash
python rz_slider.py
```

A window opens with three panels and three sliders at the bottom.  
Drag any slider to re-run the simulation and see the updated plots instantly.

### Running Inside Jupyter

If you want to run this inside a Jupyter notebook, comment out the `%matplotlib qt` line at the top and use:

```python
%matplotlib widget   # requires: pip install ipympl
```

Then call `run_simulation(dt, b_res, W_res)` directly and plot the returned monitors.

---

## Quick Examples

### Example 1 – Default (resonant input)

```python
from rz_slider import run_simulation

MP, MRZ, st = run_simulation(dt=144, b_res=-5, W_res=40)
print("RZ fires at (ms):", st.t[MRZ.i == 0] / 1e-3)
```

At `dt=144 ms` and `W_res=40` the input period roughly matches 1/W_res (≈ 25 ms), producing constructive summation → the neuron fires.

---

### Example 2 – No spike (strongly damped)

```python
MP, MRZ, st = run_simulation(dt=144, b_res=-11, W_res=40)
print("Spikes:", len(MRZ.t))  # → 0
```

With `b_res=-11` the oscillation decays so rapidly that the sub-threshold kicks from each spike fade before the next arrives → no output spike.

---

### Example 3 – Sweeping dt to find the resonant ISI

```python
import numpy as np
import matplotlib.pyplot as plt
from rz_slider import run_simulation

dt_values = np.arange(140, 201, 2)
spike_counts = []

for dt in dt_values:
    _, MRZ, _ = run_simulation(dt=dt, b_res=-5, W_res=40)
    spike_counts.append(len(MRZ.t))

plt.bar(dt_values, spike_counts, width=1.5)
plt.xlabel("dt (ms)")
plt.ylabel("Number of RZ spikes")
plt.title("Resonance curve: spikes vs. input ISI")
plt.tight_layout()
plt.show()
```

This reproduces a **resonance curve**: a peak at the ISI that best matches the neuron's natural frequency.

---

## Code Structure

```
rz-neuron-simulator/
│
├── rz_slider.py          # Main script (simulation + interactive plot)
└── README.md             # This file
```

### Inside `rz_slider.py`

```
rz_slider.py
│
├── run_simulation(dt, b_res, W_res) → (MP, MRZ, st_MRZ)
│       Build Brian2 network, run, return monitors.
│
├── Figure setup
│       ax_v    – membrane potential line plot
│       ax_p    – input spike raster
│       ax_rz   – RZ output spike raster
│
├── Slider setup
│       sl_dt, sl_b_res, sl_W_res
│
└── update(_val)
        Slider callback: re-runs simulation, refreshes all axes.
```

### Key Brian2 objects

| Object | Type | Purpose |
|--------|------|---------|
| `P` | `SpikeGeneratorGroup` | Delivers the 3 input spikes |
| `RZ` | `NeuronGroup` | The resonate-and-fire neuron |
| `S1` | `Synapses` | `v += 0.7` on each pre-synaptic spike |
| `MP` | `SpikeMonitor` | Records P spike times |
| `MRZ` | `SpikeMonitor` | Records RZ spike times |
| `st_MRZ` | `StateMonitor` | Records v and u traces |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: brian2` | `pip install brian2` |
| Black / unresponsive plot window | Install a Qt backend: `pip install pyqt5` |
| `%matplotlib qt` error in Jupyter | Replace with `%matplotlib widget` and install `ipympl` |
| Simulation is slow | Brian2 compiles C++ on the first run; subsequent calls are fast |
| Plot doesn't update on slider drag | Ensure you are NOT in inline (`%matplotlib inline`) mode |

---

## References

- Izhikevich, E. M. (2001). *Resonate-and-fire neurons.* Neural Networks, 14(6-7), 883–894. <https://doi.org/10.1016/S0893-6080(01)00078-8>
- Brian2 documentation: <https://brian2.readthedocs.io/>
- Matplotlib Widgets: <https://matplotlib.org/stable/api/widgets_api.html>

---

## License

MIT License – see [LICENSE](LICENSE) for details.
