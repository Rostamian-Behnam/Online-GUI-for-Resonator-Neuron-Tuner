"""
rz_slider.py
============
Interactive simulation of a Resonate-and-Fire (RZ) neuron using Brian2.

The RZ neuron model is defined by a pair of coupled ODEs that produce
oscillatory sub-threshold dynamics, allowing the neuron to act as a
frequency-selective (resonant) integrator:

    dv/dt = b_res * v - W_res * u
    du/dt = W_res * v + b_res * u

where:
    v       – membrane potential (real part of complex oscillator)
    u       – auxiliary variable  (imaginary part)
    b_res   – damping / growth rate  (negative = damped, positive = growing)
    W_res   – natural resonance frequency (rad/s scaled)

Three input spikes are injected at times [t0, t0+dt, t0+2*dt].
Interactive sliders let you tune dt, b_res, and W_res in real time and
observe how the neuron's spike output changes.

Dependencies
------------
    pip install brian2 matplotlib numpy

Usage
-----
    python rz_slider.py

    Move the sliders to explore parameter space.  The plot updates live.

Author : <your name>
License: MIT
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from brian2 import (
    start_scope, ms, second,
    SpikeGeneratorGroup, NeuronGroup, Synapses,
    SpikeMonitor, StateMonitor,
    Network, collect, array,
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from IPython import get_ipython

# Switch matplotlib to the Qt backend so the interactive window works
# outside a notebook (comment out if running inside Jupyter).
_ip = get_ipython()
if _ip is not None:
    _ip.run_line_magic("matplotlib", "qt")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(dt: float, b_res: float, W_res: float):
    """
    Build and run a single-neuron Brian2 network.

    Parameters
    ----------
    dt : float
        Inter-spike interval (ms) between the three input spikes.
        Controls how closely spaced the stimuli are.
    b_res : float
        Damping coefficient of the RZ oscillator.
        Negative values → decaying oscillations (stable).
        Positive values → growing oscillations (unstable).
    W_res : float
        Resonance (angular) frequency of the RZ oscillator (rad/s, scaled).
        Higher values → faster oscillations → higher preferred input frequency.

    Returns
    -------
    MP : SpikeMonitor
        Spike monitor for the input (Poisson generator) group P.
    MRZ : SpikeMonitor
        Spike monitor for the RZ neuron.
    st_MRZ : StateMonitor
        State monitor recording v and u of the RZ neuron over time.

    Example
    -------
    >>> MP, MRZ, st_MRZ = run_simulation(dt=144, b_res=-5, W_res=40)
    >>> print("RZ spike times (ms):", MRZ.t / ms)
    """
    # Reset Brian2 state so repeated calls don't accumulate objects
    start_scope()

    # ------------------------------------------------------------------
    # Input: three spikes at t0=10 ms, t0+dt, t0+2*dt
    # All spikes target neuron index 0 in the SpikeGeneratorGroup.
    # ------------------------------------------------------------------
    spike_times = array([10, 10 + dt, 10 + 2 * dt]) * ms
    spike_indices = array([0, 0, 0])                 # neuron 0 fires each time
    P = SpikeGeneratorGroup(1, spike_indices, spike_times)

    # ------------------------------------------------------------------
    # RZ neuron model
    #
    # The two state variables (v, u) form a complex oscillator:
    #   d/dt [v + i*u] = (b_res + i*W_res)(v + i*u)
    #
    # b_res and W_res are stored as per-neuron *parameters* so that Brian2
    # can export them and sliders can adjust them without rebuilding the
    # network (not used here for live update, but good practice).
    # ------------------------------------------------------------------
    eqs_resonate = """
        dv/dt = b_res * v / second - W_res * u / second : 1
        du/dt = W_res * v / second + b_res * u / second : 1
        b_res : 1   (shared)
        W_res : 1   (shared)
    """

    # Fire when membrane potential v exceeds 1; reset to (v=1, u=0)
    threshold_condition = "v > 1"
    reset_rule = """
        v = 1
        u = 0
    """

    RZ = NeuronGroup(
        1,
        eqs_resonate,
        threshold=threshold_condition,
        reset=reset_rule,
        refractory=0 * ms,
        method="euler",
    )

    # Initial conditions
    RZ.b_res = b_res
    RZ.W_res = W_res
    RZ.v = 0.0
    RZ.u = 0.0      # start auxiliary variable at rest too

    # ------------------------------------------------------------------
    # Synapse: each input spike adds 0.7 to the RZ membrane potential v
    # ------------------------------------------------------------------
    S1 = Synapses(P, RZ, on_pre="v += 0.7")
    S1.connect()

    # ------------------------------------------------------------------
    # Monitors
    # ------------------------------------------------------------------
    MP     = SpikeMonitor(P)                        # input spike times
    MRZ    = SpikeMonitor(RZ)                       # RZ output spike times
    st_MRZ = StateMonitor(RZ, ["v", "u"], record=0) # v and u traces

    # Run for 10× the ISI so there is plenty of post-stimulus time
    simulation_time = 10 * dt * ms
    net = Network(collect())
    net.run(simulation_time)

    return MP, MRZ, st_MRZ


# ---------------------------------------------------------------------------
# Initial parameter values
# ---------------------------------------------------------------------------
INIT_DT    = 144   # ms  – ISI between input spikes
INIT_B_RES = -5    # damping (negative → stable decay)
INIT_W_RES = 40    # resonance frequency

# Run the first simulation before drawing anything
MP, MRZ, st_MRZ = run_simulation(INIT_DT, INIT_B_RES, INIT_W_RES)


# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig, (ax_v, ax_p, ax_rz) = plt.subplots(
    3, 1,
    figsize=(9, 7),
    gridspec_kw={"height_ratios": [3, 1, 1]},
)
plt.subplots_adjust(left=0.1, bottom=0.35, hspace=0.55)

# -- Membrane potential trace -----------------------------------------------
time_ms   = st_MRZ.t / ms          # Brian quantity → plain float array (ms)
voltage   = st_MRZ.v[0]            # dimensionless (model units)

(line_v,) = ax_v.plot(time_ms, voltage, "b-", linewidth=1.4, label="v (membrane)")
ax_v.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="Threshold (v=1)")
ax_v.set_xlabel("Time (ms)")
ax_v.set_ylabel("v (a.u.)")
ax_v.set_title("RZ Neuron – Membrane Potential")
ax_v.set_xlim(0, 10 * INIT_DT)
ax_v.legend(loc="upper right", fontsize=8)

# -- Input spike raster (P) -------------------------------------------------
ax_p.eventplot(MP.t / ms, lineoffsets=0.5, colors="steelblue", linewidths=1.5)
ax_p.set_xlim(0, 10 * INIT_DT)
ax_p.set_ylim(0, 1)
ax_p.set_yticks([])
ax_p.set_xlabel("Time (ms)")
ax_p.set_title("Input Spike Train (P)")

# -- RZ output spike raster -------------------------------------------------
ax_rz.eventplot(MRZ.t / ms, lineoffsets=0.5, colors="green", linewidths=1.5)
ax_rz.set_xlim(0, 10 * INIT_DT)
ax_rz.set_ylim(0, 1)
ax_rz.set_yticks([])
ax_rz.set_xlabel("Time (ms)")
ax_rz.set_title("RZ Neuron Output Spikes")


# ---------------------------------------------------------------------------
# Slider widgets
# ---------------------------------------------------------------------------
# Each slider occupies a thin axes strip below the main plots.
ax_sl_dt    = plt.axes([0.18, 0.22, 0.65, 0.03])
ax_sl_b_res = plt.axes([0.18, 0.15, 0.65, 0.03])
ax_sl_W_res = plt.axes([0.18, 0.08, 0.65, 0.03])

sl_dt    = Slider(ax_sl_dt,    "dt (ms)",  valmin=140, valmax=200, valinit=INIT_DT,    valstep=1)
sl_b_res = Slider(ax_sl_b_res, "b_res",    valmin=-11, valmax=0,   valinit=INIT_B_RES, valstep=1)
sl_W_res = Slider(ax_sl_W_res, "W_res",    valmin=0,   valmax=50,  valinit=INIT_W_RES, valstep=1)


# ---------------------------------------------------------------------------
# Slider callback
# ---------------------------------------------------------------------------

def update(_val):
    """
    Called whenever any slider is moved.

    Re-runs the full Brian2 simulation with the current slider values and
    refreshes all three sub-plots without recreating the figure.

    Parameters
    ----------
    _val : float
        The new slider value (passed automatically by Matplotlib; unused
        directly because we read all three sliders inside the function).
    """
    dt    = sl_dt.val
    b_res = sl_b_res.val
    W_res = sl_W_res.val

    # Run simulation with updated parameters
    MP_new, MRZ_new, st_new = run_simulation(dt, b_res, W_res)

    # Update membrane potential line data
    new_time = st_new.t / ms
    new_volt = st_new.v[0]
    line_v.set_xdata(new_time)
    line_v.set_ydata(new_volt)
    ax_v.set_xlim(0, 10 * dt)
    ax_v.relim()
    ax_v.autoscale_view(scalex=False)  # keep x fixed, auto-scale y

    # Redraw input raster
    ax_p.clear()
    ax_p.eventplot(MP_new.t / ms, lineoffsets=0.5, colors="steelblue", linewidths=1.5)
    ax_p.set_xlim(0, 10 * dt)
    ax_p.set_ylim(0, 1)
    ax_p.set_yticks([])
    ax_p.set_xlabel("Time (ms)")
    ax_p.set_title("Input Spike Train (P)")

    # Redraw RZ output raster
    ax_rz.clear()
    ax_rz.eventplot(MRZ_new.t / ms, lineoffsets=0.5, colors="green", linewidths=1.5)
    ax_rz.set_xlim(0, 10 * dt)
    ax_rz.set_ylim(0, 1)
    ax_rz.set_yticks([])
    ax_rz.set_xlabel("Time (ms)")
    ax_rz.set_title("RZ Neuron Output Spikes")

    fig.canvas.draw_idle()   # efficient redraw (only dirty regions)


# Connect sliders to the update callback
sl_dt.on_changed(update)
sl_b_res.on_changed(update)
sl_W_res.on_changed(update)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    plt.show()
