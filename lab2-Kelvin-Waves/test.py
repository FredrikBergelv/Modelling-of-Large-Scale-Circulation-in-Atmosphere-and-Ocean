import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from load import load_h_data

times_a = [150]
times_b = [250]

# =========================================================
# DATA
# =========================================================

ha_100 = load_h_data("atlantic_100")
ha_200 = load_h_data("atlantic_200")
ha_300 = load_h_data("atlantic_300")
ha_450 = load_h_data("atlantic_450")
ha_600 = load_h_data("atlantic_600")
ha_750 = load_h_data("atlantic_750")
ha_900 = load_h_data("atlantic_900")
ha_1050 = load_h_data("atlantic_1050")
ha_1200 = load_h_data("atlantic_1200")

hb_100 = load_h_data("baltic_100")
hb_200 = load_h_data("baltic_200")
hb_300 = load_h_data("baltic_300")
hb_450 = load_h_data("baltic_450")
hb_600 = load_h_data("baltic_600")
hb_750 = load_h_data("baltic_750")
hb_900 = load_h_data("baltic_900")
hb_1050 = load_h_data("baltic_1050")
hb_1200 = load_h_data("baltic_1200")

cases = [
    [ha_100, 1000, r"Atlantic 100$\times$100"],
    [ha_200, 1000, r"Atlantic 200$\times$200"],
    [ha_300, 1000, r"Atlantic 300$\times$300"],
    [ha_450, 1000, r"Atlantic 450$\times$450"],

    [hb_100, 30, r"Baltic 100$\times$100"],
    [hb_200, 30, r"Baltic 200$\times$200"],
    [hb_300, 30, r"Baltic 300$\times$300"],
    [hb_450, 30, r"Baltic 450$\times$450"],
]

# =========================================================
# THEORY
# =========================================================

def theo(h0, y, c):
    """
    Kelvin-wave decay away from boundary
    """
    f = 1e-4
    R = c / f

    y = y * 1000  # km -> m

    return h0 * np.exp(-y / R)

# =========================================================
# PLOTTING
# =========================================================

fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharey=True)

g = 0.0981

for j, (dataset, H, name) in enumerate(cases):

    h = dataset

    # Wave speed
    c = np.sqrt(g * H)

    # Domain
    Lx = float(h.x.max())
    Nx = len(h.x.values)
    dx = Lx / Nx

    # Constant theoretical amplitude
    h0 = 5.0

    # =====================================================
    # TIME
    # =====================================================

    if j < len(cases)/2:
        t = times_a[0]
        ax = axs[0]
    else:
        t = times_b[0]
        ax = axs[1]

    # Select time
    hnow = h.sel(time=t, method="nearest")

    # =====================================================
    # FIND WAVE PEAK
    # =====================================================

    hx = hnow.sel(y=0, method="nearest")

    h_values = hx.values
    x_values = hx.x.values

    # Center-of-mass peak (stable)
    weights = h_values - np.min(h_values)

    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
        xmax = np.sum(x_values * weights)
    else:
        idx_max = np.argmax(h_values)
        xmax = x_values[idx_max]

    # =====================================================
    # Y SECTION THROUGH PEAK
    # =====================================================

    ht = hnow.sel(x=xmax, method="nearest")

    y_vals = ht.y.values / 1000  # km
    hvals = ht.values

    # Shift boundary to y=0
    y_vals = y_vals - np.min(y_vals)

    # =====================================================
    # THEORY
    # =====================================================

    if j == 0 or j == int(len(cases)/2):

        y_theo = np.linspace(0, np.max(y_vals), 1000)

        h_theo = theo(h0, y_theo, c)

        ax.plot(
            y_theo,
            h_theo / h0,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Theory"
        )

    # =====================================================
    # NUMERICAL
    # =====================================================

    ax.plot(
        y_vals,
        hvals / h0,
        label=name
    )

    # =====================================================
    # AXIS SETTINGS
    # =====================================================

    actual_time = float(ht["time"].values)

    ax.set_title(f"t = {actual_time:.1f} h")

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 1.1)

    ax.set_xlabel(r"$y$ [km]")
    ax.set_ylabel(r"$\eta/h_0$")

    ax.grid(True, linestyle="--", alpha=0.6)

    ax.legend()

# =========================================================
# FINALIZE
# =========================================================

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.suptitle(
    r"Kelvin-wave decay away from the boundary",
    fontsize=16
)

plt.savefig("plots/yaxis_grids.png", dpi=300)
plt.savefig("kelvin-waves-report/Figures/yaxis_grids.png", dpi=300)

plt.show()