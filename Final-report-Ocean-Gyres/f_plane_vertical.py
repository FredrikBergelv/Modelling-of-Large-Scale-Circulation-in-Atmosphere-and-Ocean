import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cmocean

D = 3000
Lx = Ly = 7e6
Nx = Ny = 200
dx = dy = Lx / Nx
rho = 1027
f0 = 7.27e-5
beta = 1.98e-11

Ekman_ratio = D / 50


def sverdrup(x, y, u10, Lx=Lx):

    dx_here = Lx / Nx

    tau0 = 1.225 * 1.5 * 0.001 * (u10**2)

    curl_tau = (
        -np.pi / Ly
        * tau0
        * np.cos(np.pi * y / Ly)
    )

    v_s = curl_tau / (rho * beta)

    psi = np.zeros((len(y), len(x)))

    for j in range(len(y)):

        integral = 0.0

        for i in reversed(range(len(x))):

            if i < len(x) - 1:
                integral += v_s[j] * dx_here

            psi[j, i] = integral

    return -psi


def munk(x, y, A, Lx):

    psi_s = sverdrup(x, y, u10=5, Lx=Lx)

    dm = (A / beta) ** (1 / 3)

    factor_x = (
        1
        - np.exp(-x / (2 * dm))
        * (
            np.cos(x * np.sqrt(3) / (2 * dm))
            + (1 / np.sqrt(3))
            * np.sin(x * np.sqrt(3) / (2 * dm))
        )
    )

    return psi_s * factor_x[None, :]


def extract_centerline(x, y, field):
    j0 = np.argmin(np.abs(y - 0))
    return field[j0, :]


def plot_streamfunction(
    folder,
    times,
    times_cross=None,
    show=False,
    u10=False,
    A=False,
    Lx=Lx
):
    """
    times        — list of times for the contourf panels (one column each)
    times_cross  — list of times to overlay on the cross-section rows
                   defaults to times if not provided
    """

    if not isinstance(times, list):
        times = [times]

    if times_cross is None:
        times_cross = times

    if not isinstance(times_cross, list):
        times_cross = [times_cross]

    h_ds = xr.open_dataset(f"data/{folder}/h.nc")
    v_ds = xr.open_dataset(f"data/{folder}/v.nc")

    x = h_ds["x"].values
    y = h_ds["y"].values

    # ======================================================
    # LOAD CONTOURF TIMES
    # ======================================================
    psi_list    = []
    v_list      = []
    actual_times = []

    for time in times:

        v = v_ds["v"].sel(time=time, method="nearest")
        h = h_ds["h"].sel(time=time, method="nearest")

        actual_time = h["time"].values
        actual_times.append(actual_time)

        psi = -np.cumsum(
            (v.values * D * dx)[:, ::-1],
            axis=1
        )[:, ::-1]

        if psi.shape[0] != len(y):
            y_plot = np.linspace(y[0], y[-1], psi.shape[0])
        else:
            y_plot = y

        psi_list.append((psi / 1e6, y_plot))
        v_list.append(v.values)

    # ======================================================
    # LOAD CROSS-SECTION TIMES
    # ======================================================
    psi_cross    = []
    v_cross      = []
    actual_times_cross = []

    for time in times_cross:

        v = v_ds["v"].sel(time=time, method="nearest")
        h = h_ds["h"].sel(time=time, method="nearest")

        actual_time = h["time"].values
        actual_times_cross.append(actual_time)

        psi = -np.cumsum(
            (v.values * D * dx)[:, ::-1],
            axis=1
        )[:, ::-1]

        if psi.shape[0] != len(y):
            y_plot = np.linspace(y[0], y[-1], psi.shape[0])
        else:
            y_plot = y

        psi_cross.append(extract_centerline(x, y_plot, psi / 1e6))
        v_cross.append(Ekman_ratio * extract_centerline(x, y_plot, v.values))

    # ======================================================
    # COLOR SCALE (contourf)
    # ======================================================
    all_psi  = np.concatenate([p[0].ravel() for p in psi_list])
    vmin_psi = all_psi.min()
    vmax_psi = all_psi.max()

    # ======================================================
    # Y-LIMITS (cross-section) — include all cross times
    # ======================================================
    psi_min = np.min([p.min() for p in psi_cross])
    psi_max = np.max([p.max() for p in psi_cross])
    v_min   = np.min([v.min() for v in v_cross])
    v_max   = np.max([v.max() for v in v_cross])

    n = len(times)

    # ======================================================
    # FIGURE: 3 rows x n columns
    # ======================================================
    fig, axes = plt.subplots(
        3, n,
        figsize=(4.7 * n, 7),
        sharex=True,
        sharey=False
    )

    if n == 1:
        axes = axes.reshape(3, 1)

    colors_cross = plt.cm.viridis(
        np.linspace(0.15, 0.85, len(times_cross))
    )

    # ======================================================
    # PLOTTING
    # ======================================================
    for i in range(n):

        psi, y_raw  = psi_list[i]
        actual_time = actual_times[i]

        x_plot = x / Lx
        y_plot = y_raw / Ly

        days  = actual_time // 24
        hours = actual_time % 24

        ax_contour = axes[0, i]
        ax_psi     = axes[1, i]
        ax_v       = axes[2, i]

        # --------------------------------------------------
        # ROW 0: CONTOURF
        # --------------------------------------------------
        cf = ax_contour.contourf(
            x_plot, y_plot, psi,
            cmap="cmo.haline",
            levels=8,
            vmin=vmin_psi,
            vmax=vmax_psi
        )

        if A:
            psi_theo = munk(x, y_raw, A, Lx)
            ax_contour.contour(x_plot, y_plot, psi_theo, colors="red")

        if u10:
            psi_theo = sverdrup(x, y_raw, u10, Lx)
            ax_contour.contour(x_plot, y_plot, psi_theo, colors="red")

        ax_contour.set_title(f"t={days:.0f}d {hours:.0f}h", fontsize=13)
        ax_contour.set_ylabel(r"Longitude, $y/L_y$ [-]", fontsize=11)

        # --------------------------------------------------
        # ROW 1 & 2: all cross-section times overlaid
        # --------------------------------------------------
        for k, (psi_k, v_k, t_k) in enumerate(
            zip(psi_cross, v_cross, actual_times_cross)
        ):
            days_k  = t_k // 24
            hours_k = t_k % 24
            lbl = f"t={days_k:.0f}d {hours_k:.0f}h"

            ax_psi.plot(x_plot, psi_k, color=colors_cross[k], linewidth=1.8, label=lbl)
            ax_v.plot(  x_plot, v_k,   color=colors_cross[k], linewidth=1.8, label=lbl)

        ax_psi.set_ylim(psi_min, psi_max)
        ax_psi.grid(alpha=0.3)
        ax_psi.set_ylabel(r"Streamfunction, $\Psi$ [Sv]", fontsize=11)

        ax_v.set_ylim(v_min, v_max)
        ax_v.grid(alpha=0.3)
        ax_v.set_xlabel(r"Latitude, $x/L_x$ [-]", fontsize=11)
        ax_v.set_ylabel(r"Meridional velocity, $v$ [m/s]", fontsize=11)

    # ======================================================
    # LEGEND — only on first column to avoid repetition
    # ======================================================
    axes[1, 0].legend(fontsize=9, loc="upper right")
    axes[2, 0].legend(fontsize=9, loc="upper right")

    # ======================================================
    # COLORBAR FOR ROW 0
    # ======================================================
    cbar = fig.colorbar(
        cf,
        ax=axes[0, :],
        location="right",
        shrink=0.85,
        pad=0.01
    )
    cbar.set_label(r"Streamfunction, $\Psi$ [Sv]", fontsize=11)

    plt.suptitle("Ocean Gyres Streamfunction", size=16)
    fig.set_constrained_layout(True)

    save_name = "f_plane_vertical"
    plt.savefig(f"plots/{save_name}.png", dpi=300, bbox_inches="tight")
    print(f"Saved as {save_name}")

    if show:
        plt.show()


# ==========================================================
# RUN
# ==========================================================
plot_streamfunction(
    "no_drag_f_plane",
    times=[2*24],
    times_cross=[2*24, 7*24, 40*24]
)