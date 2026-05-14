import numpy as np
import matplotlib.pyplot as plt
from load import load_h_data

# Data
ha_100 = load_h_data("atlantic_100")
ha_200 = load_h_data("atlantic_200")
ha_300 = load_h_data("atlantic_300")
ha_450 = load_h_data("atlantic_450")
ha_600 = load_h_data("atlantic_600")
ha_750 = load_h_data("atlantic_750")
ha_900 = load_h_data("atlantic_900")
ha_1050 = load_h_data("atlantic_1050")
ha_1200 = load_h_data("atlantic_1200")

ha = [
    [ha_100, 100, 1000, "Atlantic"],
    [ha_200, 200, 1000, "Atlantic"],
    [ha_300, 300, 1000, "Atlantic"],
    [ha_450, 450, 1000, "Atlantic"],
    [ha_600, 600, 1000, "Atlantic"],
    [ha_750, 750, 1000, "Atlantic"],
    [ha_900, 900, 1000, "Atlantic"],
    [ha_1050, 1050, 1000, "Atlantic"],
    [ha_1200, 1200, 1000, "Atlantic"]
]

# =========================================================
# BALTIC
# =========================================================

hb_100 = load_h_data("baltic_100")
hb_200 = load_h_data("baltic_200")
hb_300 = load_h_data("baltic_300")
hb_450 = load_h_data("baltic_450")
hb_600 = load_h_data("baltic_600")
hb_750 = load_h_data("baltic_750")
hb_900 = load_h_data("baltic_900")
hb_1050 = load_h_data("baltic_1050")
hb_1200 = load_h_data("baltic_1200")

hb = [
    [hb_100, 100, 30, "Baltic"],
    [hb_200, 200, 30, "Baltic"],
    [hb_300, 300, 30, "Baltic"],
    [hb_450, 450, 30, "Baltic"],
    [hb_600, 600, 30, "Baltic"],
    [hb_750, 750, 30, "Baltic"],
    [hb_900, 900, 30, "Baltic"],
    [hb_1050, 1050, 30, "Baltic"],
    [hb_1200, 1200, 30, "Baltic"]
]

grids_plot = np.array([100, 200, 300, 450, 600, 750, 900, 1050, 1200])

hs = [ha,hb]

times = np.arange(12, 21*24 + 12, 12)
g = 0.0981


def theo(x, c, t, Lx, Ly):    
    f = 1e-4              # Coriolis parameter [1/s]
    h0 = 5                # amplitude [m]
    Lw = Lx / 10
    R = c / f

    x = x * 1000  # km -> m
    t = t * 3600  # hours -> seconds

    # Wave center position (periodic)
    x0 = (0.5 * Lx + c * t) % Lx

    # Periodic distance
    dx = x - x0
    dx = (dx + Lx/2) % Lx - Lx/2

    # Kelvin-wave Gaussian
    Gt = h0 * np.exp(-(dx**2) / (Lw**2)) * np.exp(-0 / R)

    return Gt


def phase_diff(xmax, xmax_theo, c_theo, time, Lx):

    # periodic distance
    x_diff = xmax - xmax_theo
    x_diff = (x_diff + Lx/2) % Lx - Lx/2

    c_diff = x_diff / (time * 3600)

    return c_theo + c_diff


def rel_error(val1, val2):
    return 100 * np.abs(val1 - val2) / val2

results_a = []
results_b = []

for h_type in hs:
    for (h_grid, grid, H, name) in h_type:
        c_theo = np.sqrt(g * H)
        Lx = float(h_grid.x.max())
        Ly = float(h_grid.y.max())

        ratios = []
        for t in times:

            # Select time and y=0
            hnow = h_grid.sel(time=t, method="nearest")
            ht = hnow.sel(y=0, method="nearest")

            # Convert to numpy
            h_values = ht.values
            x_values = ht.x.values
            Nx = len(x_values)

            # Numerical peak
            idx = np.argmax(h_values)
            x_peak = x_values[idx]

            # Theoretical peak
            x_theo_peak = (0.5 * Lx + c_theo * t * 3600) % (Lx)

            # Phase speed
            c_num = phase_diff(x_peak, x_theo_peak, c_theo, t, Lx)
            ratio = c_num/c_theo

            # Store result
            ratios.append(ratio)
        
        ratios = np.array(ratios)
        std = np.std(ratios)
        mean = np.mean(ratios)
        end = ratios[-1]
        min = mean-std
        max = mean+std
        
        print(name, f"{grid}x{grid} -> num/theo = ", end)
        if name == "Atlantic":
            results_a.append((end, min, max))
        if name == "Baltic":
            results_b.append((mean, min, max))


# =========================================================
# PLOT
# =========================================================

results_a = np.array(results_a)
results_b = np.array(results_b)

fig, ax = plt.subplots(figsize=(5,3))

# Atlantic
ax.plot(
    grids_plot,
    results_a[:,0],
    marker="o",
    label="Atlantic"
)

ax.fill_between(
    grids_plot,
    results_a[:,1],
    results_a[:,2],
    alpha=0.3
)

# Baltic
ax.plot(
    grids_plot,
    results_b[:,0],
    marker="o",
    label="Baltic"
)

ax.fill_between(
    grids_plot,
    results_b[:,1],
    results_b[:,2],
    alpha=0.3
)

ax.grid(True, linestyle='--', alpha=0.6)


# Reference line
ax.axhline(1.0, linestyle="--", c="black")

# Labels
all_vals = np.concatenate([results_a[:, 0], results_b[:, 0]])
ax.set_ylim(0.998*all_vals.min(), 1.005*all_vals.max())

ax.set_xlabel("Grid resolution")
ax.set_ylabel(r"$c_{\mathrm{num}} / c_{\mathrm{theo}}$")
ax.set_title("Phase speed convergence", size=14)

ax.legend()
plt.tight_layout()
plt.savefig("plots/phase.png", dpi=300)
plt.savefig("kelvin-waves-report/Figures/phase.png", dpi=300)

plt.show()