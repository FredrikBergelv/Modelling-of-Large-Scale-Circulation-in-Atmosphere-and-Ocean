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


def plot_streamfunction(folder, times, show=False, u10=False, A=False, Lx=Lx):

    if not isinstance(times, list):
        times = [times]

    h_ds = xr.open_dataset(f"data/{folder}/h.nc")
    u_ds = xr.open_dataset(f"data/{folder}/u.nc")
    v_ds = xr.open_dataset(f"data/{folder}/v.nc")

    x = h_ds["x"].values
    y = h_ds["y"].values

    psi_list = []
    v_list = []
    actual_times = []

    # ======================================================
    # COMPUTE ALL FIELDS
    # ======================================================
    for time in times:

        h = h_ds["h"].sel(time=time, method="nearest")
        v = v_ds["v"].sel(time=time, method="nearest")

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
    # COMMON COLOR SCALE (contourf row)
    # ======================================================
    all_psi = np.concatenate([p[0].ravel() for p in psi_list])
    vmin_psi = all_psi.min()
    vmax_psi = all_psi.max()

    # ======================================================
    # COMMON Y-LIMITS (cross-section rows)
    # ======================================================
    psi_profiles = [extract_centerline(x, psi_list[i][1], psi_list[i][0]) for i in range(len(times))]
    v_profiles   = [Ekman_ratio * extract_centerline(x, psi_list[i][1], v_list[i]) for i in range(len(times))]

    psi_min = np.min([p.min() for p in psi_profiles])
    psi_max = np.max([p.max() for p in psi_profiles])
    v_min   = np.min([v.min() for v in v_profiles])
    v_max   = np.max([v.max() for v in v_profiles])

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

    # ======================================================
    # PLOTTING
    # ======================================================
    for i in range(n):

        psi, y_raw = psi_list[i]
        v_field    = v_list[i]
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
            ax_contour.contour(
                x_plot, y_plot, psi_theo,
                colors="red"
            )

        if u10:
            psi_theo = sverdrup(x, y_raw, u10, Lx)
            ax_contour.contour(
                x_plot, y_plot, psi_theo,
                colors="red"
            )

        #ax_contour.set_title(f"t={days:.0f}d {hours:.0f}h", fontsize=13)

        ax_contour.set_ylabel(r"Longitude, $y/L_y$ [-]", fontsize=11)

        # --------------------------------------------------
        # ROW 1: PSI CROSS-SECTION at y=0
        # --------------------------------------------------
        ax_psi.plot(x_plot, psi_profiles[i], color="C0")
        ax_psi.set_ylim(psi_min, psi_max)
        ax_psi.grid(alpha=0.3)

        ax_psi.set_ylabel(r"Streamfunction, $\Psi$ [Sv]", fontsize=11)

        # --------------------------------------------------
        # ROW 2: V CROSS-SECTION at y=0
        # --------------------------------------------------
        ax_v.plot(x_plot, v_profiles[i], color="C1")
        ax_v.set_ylim(v_min, v_max)
        ax_v.grid(alpha=0.3)
        
        ax_v.set_xlabel(r"Latitude, $x/L_x$ [-]", fontsize=11)
   


        ax_v.set_ylabel(r"Meridional velocity, $v$ [m/s]", fontsize=11)

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
times_long  = [2*24, 7*24, 40*24]
times_short = [2*24, 7*24, 14*24]
times_end   = 1e10
times_early = 24

plot_streamfunction("no_drag_f_plane", times_end)