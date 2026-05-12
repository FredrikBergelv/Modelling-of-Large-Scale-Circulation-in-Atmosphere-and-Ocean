import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from load import load_data

# =========================================================
# Parameters
# =========================================================

g = 0.01 * 9.81
H = 1000

times = [0, 24, 100, 150, 200]   # hours

# =========================================================
# Load data
# =========================================================

h, u, v = load_data("pacific")

Lx = float(h.x.max())
Ly = float(h.y.max())

Nx = len(h.x)
dx = Lx / Nx

min_height = float(h.min())
max_height = float(h.max())

# =========================================================
# Plot setup
# =========================================================

fig, axs = plt.subplots(
    1,
    len(times),
    figsize=(3 * len(times), 5),
    sharey=True
)

# =========================================================
# Loop over times
# =========================================================

for i, t in enumerate(times):

    # ---------------------------------------------
    # Select nearest time
    # ---------------------------------------------

    hnow = h.sel(time=t, method="nearest")
    unow = u.sel(time=t, method="nearest")
    vnow = v.sel(time=t, method="nearest")

    # ---------------------------------------------
    # Equatorial section
    # ---------------------------------------------

    hx = hnow.sel(y=Ly/2, method="nearest")

    x = h.x.values
    hx_vals = hx.values

    x0 = Lx / 2

    # =====================================================
    # Kelvin wave (eastward)
    # =====================================================

    east_mask = x > x0

    hx_east = hx_vals[east_mask]
    x_east = x[east_mask]

    idx_kelvin = np.argmax(hx_east)

    x_kelvin = x_east[idx_kelvin]

    # =====================================================
    # Rossby wave (westward)
    # =====================================================

    west_mask = x < x0

    hx_west = hx_vals[west_mask]
    x_west = x[west_mask]

    idx_rossby = np.argmax(hx_west)

    x_rossby = x_west[idx_rossby]

    # =====================================================
    # Extract meridional cross sections
    # =====================================================

    ht_ke = hnow.sel(x=x_kelvin, method="nearest")
    ht_ro = hnow.sel(x=x_rossby, method="nearest")

    vt_ke = vnow.sel(x=x_kelvin, method="nearest")
    vt_ro = vnow.sel(x=x_rossby, method="nearest")

    # =====================================================
    # Coordinates
    # =====================================================

    yh = h.y.values / 1000
    yv = v.y.values / 1000

    # =====================================================
    # Diagnostics
    # =====================================================

    print(f"\nTime = {t} h")

    print(
        "Kelvin wave location:",
        round(x_kelvin / 1000),
        "km"
    )

    print(
        "Rossby wave location:",
        round(x_rossby / 1000),
        "km"
    )

    print(
        "max |v_kelvin| =",
        np.max(np.abs(vt_ke.values))
    )

    print(
        "max |v_rossby| =",
        np.max(np.abs(vt_ro.values))
    )

    # =====================================================
    # Plot
    # =====================================================

    ax = axs[i]

    actual_time = float(hnow.time.values)

    ax.set_title(f"t = {actual_time:.1f} h")

    # Kelvin
    ax.plot(
        yh,
        ht_ke.values,
        lw=2,
        c="C0",
        label="Kelvin"
    )

    # Rossby
    ax.plot(
        yh,
        ht_ro.values,
        lw=2,
        ls="--",
        c="C1",
        label="Rossby"
    )

    # Equator
    ax.axvline(
        Ly / 2000,
        color="k",
        lw=1,
        alpha=0.4
    )

    ax.grid(True, linestyle="--", alpha=0.5)

    ax.set_xlim(0, Ly / 1000)
    ax.set_ylim(min_height, max_height)

    ax.set_xlabel(r"$y$ [km]")

    if i == 0:
        ax.set_ylabel(r"$\eta$ [m]")
        ax.legend()

# =========================================================
# Final layout
# =========================================================

plt.tight_layout(rect=[0, 0, 1, 0.93])

plt.suptitle(
    "Meridional Cross Sections of Kelvin and Rossby Waves",
    fontsize=16
)

plt.savefig("plots/equatorial_cross_sections.png", dpi=300)

plt.show()