import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from load import load_data

# ============================================================
# LOAD DATA
# ============================================================

data_in = np.loadtxt("speeds_topo.txt")
alpha_vals = data_in[:, 0]

data = [
    "depth_diff=-20",
    "depth_diff=0",
    "depth_diff=20",
    "depth_diff=40",
    "depth_diff=70",
    "depth_diff=90"
]

times = [0, 24*10, 24*20, 24*30]   # hours

# ============================================================
# SETTINGS
# ============================================================

vmin = -1
vmax = 5

levels = np.linspace(vmin, vmax, 1000)

# quiver density
skip = 6

# ============================================================
# QUIVER FUNCTION
# ============================================================
def compute_uv(u, v):
    u_i = 0.5 * (u.values[:, :-1] + u.values[:, 1:])
    v_i = 0.5 * (v.values[:-1, :] + v.values[1:, :])
    return np.nan_to_num(u_i), np.nan_to_num(v_i)


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    len(data),
    len(times),
    figsize=(8/0.6, 13),
    sharex=True,
    sharey=True
)

# ============================================================
# LOOP OVER DATASETS
# ============================================================

for j, name in enumerate(data):

    alpha = alpha_vals[j]

    # ---- load data ----
    h, u, v = load_data(name)

    # ---- coordinates in km ----
    x = h.x.values / 1000
    y = h.y.values / 1000

    X, Y = np.meshgrid(x, y)

    # ========================================================
    # LOOP OVER TIMES
    # ========================================================

    for i, tt in enumerate(times):

        ax = axes[j, i]

        # ---- select nearest time ----
        hs = h.sel(time=tt, method="nearest")
        us = u.sel(time=tt, method="nearest")
        vs = v.sel(time=tt, method="nearest")

        # ---- contour plot ----
        cf = ax.contourf(
            X,
            Y,
            hs.values,
            levels=levels,
            cmap="ocean",
            vmin=vmin,
            vmax=vmax,
            extend="both"
        )
        

        # ---- contour lines (optional but nice) ----

        # ====================================================
        # VELOCITIES ON T GRID
        # ====================================================

        # u is on staggered x-grid
        u_center = 0.5 * (
            us.values[:-1, :] +
            us.values[1:, :]
        )

        # v is on staggered y-grid
        v_center = 0.5 * (
            vs.values[:, :-1] +
            vs.values[:, 1:]
        )

        # transpose to match meshgrid orientation
        u_plot = u_center
        v_plot = v_center
        

        # ====================================================
        # LABELS
        # ====================================================

        if j == 0:
            ax.set_title(
                f"t = {tt/24:.0f} d"
            )

        if i == 0:
            ax.set_ylabel("y [km]")

        if j == len(data)-1:
            ax.set_xlabel("x [km]")

        ax.set_aspect("auto")
        
        u_i0, v_i0 = compute_uv(us, vs)
        step = 5
        Q = ax.quiver(
            X[::step, ::step],
            Y[::step, ::step],
            u_i0[::step, ::step],
            v_i0[::step, ::step],
            color="white",
            scale= 0.1,
            alpha=0.4,
            width=0.004
        )
        
        ax.set_xlim(2500, 4500)
        ax.set_ylim(-530, 530)
        
        if i == 0:
            ax.text(
                0.02, 0.95,
                rf"$\alpha={alpha:.2e}$",
                transform=ax.transAxes,
                fontsize=11,
                va="top",
                ha="left",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
            )


# ============================================================
# COLORBAR
# ============================================================

cbar = fig.colorbar(
    cf,
    ticks=[-1,0,1,2,3,4,5],
    ax=axes,
    orientation="vertical",
    fraction=0.01,
    pad=0.02
)

cbar.set_label(r"$\eta$ [m]")

# ============================================================
# FINAL TOUCHES
# ============================================================

plt.suptitle(
    "Rossby Waves over Sloping Topography",
    fontsize=16
)

save_name = "topography_contours"

plt.savefig(
    f"plots/{save_name}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    f"rossby-waves-report/Figures/{save_name}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()