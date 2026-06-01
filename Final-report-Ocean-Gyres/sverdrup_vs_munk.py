import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cmocean
from matplotlib.gridspec import GridSpec

# =========================
# PARAMETERS
# =========================
D = 3000
Lx = Ly = 7e6
Nx = Ny = 200
dx = dy = Lx / Nx
rho = 1027
f0 = 7.27e-5
beta = 1.98e-11

time_start = 1 * 24


# =========================
# THEORY
# =========================
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

    psi_s = sverdrup(x, y, u10=10, Lx=Lx)

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


# =========================
# STREAMFUNCTION FROM MODEL
# =========================
def compute_psi(h_ds, v_ds, time):

    h = h_ds["h"].sel(time=time, method="nearest")
    v = v_ds["v"].sel(time=time, method="nearest")

    psi = -np.cumsum(
        (v.values * D * dx)[:, ::-1],
        axis=1
    )[:, ::-1]

    return psi[:-1], h["time"].values


def compute_psi_timeavg(v_ds, h_ds, t_upper):

    v = v_ds["v"].sel(time=slice(time_start, t_upper))
    x = h_ds["x"].values
    y = h_ds["y"].values

    if len(v.time) == 0:
        return None, x, y

    v_mean = v.mean(dim="time").values

    psi = -np.cumsum(
        (v_mean * D * dx)[:, ::-1],
        axis=1
    )[:, ::-1]

    if psi.shape[0] != len(y):
        y = np.linspace(y[0], y[-1], psi.shape[0])

    return psi[:-1] / 1e6, x, y


def compute_psi_instant(v_ds, h_ds, t):

    v = v_ds["v"].sel(time=t, method="nearest")
    x = h_ds["x"].values
    y = h_ds["y"].values

    psi = -np.cumsum(
        (v.values * D * dx)[:, ::-1],
        axis=1
    )[:, ::-1]

    if psi.shape[0] != len(y):
        y = np.linspace(y[0], y[-1], psi.shape[0])

    return psi[:-1] / 1e6, x, y


# =========================
# ERROR METRIC
# =========================
def relative_error_1d(numerical, theoretical):
    diff = numerical - theoretical
    return np.sqrt(np.sum(diff**2)) / np.sqrt(np.sum(theoretical**2))


# =========================
# MAIN PLOT
# =========================
def plot_2xN(models, times, labels, u10=None, A=None, Lx=Lx, show=False):

    nrows = len(models)
    ncols = len(times)

    fig = plt.figure(figsize=(3.7 * ncols, 7/3 * nrows + 3))

    gs = GridSpec(
        nrows + 1, ncols,
        figure=fig,
        height_ratios=[7/3] * nrows + [3.]
    )

    axes = np.array([
        [fig.add_subplot(gs[row, col]) for col in range(ncols)]
        for row in range(nrows)
    ])

    ax_conv = fig.add_subplot(gs[nrows, :])


    data_store = []
    cf_rows = [None] * nrows

    # =========================
    # LOAD DATA + PLOT
    # =========================
    for model_idx, model in enumerate(models):

        h_ds = xr.open_dataset(f"data/{model}/h.nc")
        v_ds = xr.open_dataset(f"data/{model}/v.nc")

        x = h_ds["x"].values
        y = h_ds["y"].values

        psi_row = []
        time_row = []

        for t in times:
            psi, actual_time = compute_psi(h_ds, v_ds, t)
            psi = psi / 1e6
            psi_row.append(psi)
            time_row.append(actual_time)

        data_store.append((model, h_ds, v_ds, x, y, psi_row, time_row))

        for j, ax in enumerate(axes[model_idx]):
            
            if j!=0:
                ax.tick_params(axis="y", labelleft=False)
            else:
                ax.set_ylabel("Longitude, $y/L_y$ [-]")
        
            psi = psi_row[j]

            x_plot = x / Lx
            y_plot = y / Ly

            # =========================
            # FIX: ROW-SPECIFIC SCALING
            # =========================
            if model_idx == 1:
                cf = ax.contourf(
                    x_plot, y_plot, psi,
                    levels=8,
                    cmap="cmo.haline",
                    vmin=-1, vmax=8   # <-- FORCED BOTTOM ROW SCALE
                )
            else:
                cf = ax.contourf(
                    x_plot, y_plot, psi,
                    levels=8,
                    cmap="cmo.haline",
                    vmin=0, vmax=24
                )
                actual_time = time_row[j]
                days  = actual_time // 24
                hours = actual_time % 24
                ax.set_title(f"t={days:.0f}d {hours:.0f}h", fontsize=13)

            if j == 0:
                cf_rows[model_idx] = cf

            if labels[model_idx] == "Munk drag ON":
                psi_theo = munk(x, y, 1e5, Lx)
                cs = ax.contour(x_plot, y_plot, psi_theo/1e6, colors="red")
                ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")
                text = labels[model_idx]
                ax.tick_params(axis="x", labelbottom=False)

            else:
                psi_theo = sverdrup(x, y, u10, Lx)
                cs = ax.contour(x_plot, y_plot, psi_theo/1e6, colors="red")
                ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")
                text = labels[model_idx]
                ax.set_xlabel("Latitude, $x/L_x$ [-]")

            ax.text(
                0.98, 0.97,
                text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
            )
            
            

    # =========================
    # TWO COLORBARS (ONE PER ROW)
    # =========================
    for i in range(nrows):
        cbar = fig.colorbar(
            cf_rows[i],
            ax=axes[i, :],
            shrink=0.8,
            pad=0.02
        )
        cbar.set_label(r"Streamfunction [Sv]")

    # =========================
    # CONVERGENCE (UNCHANGED)
    # =========================
    ref_h_ds = xr.open_dataset(f"data/{data_store[0][0]}/h.nc")
    all_times = ref_h_ds["time"].values
    t_uppers = all_times[all_times > time_start]

    colors = ["C0", "C1", "C2", "C3"]

    for i, (model, h_ds, v_ds, x, y, psi_row, time_row) in enumerate(data_store):

        errors_mean = []
        errors_inst = []
        t_axis = []

        for t_upper in t_uppers:

            psi_mean, x_num, y_num = compute_psi_timeavg(v_ds, h_ds, t_upper)
            if psi_mean is None:
                continue

            psi_inst, _, _ = compute_psi_instant(v_ds, h_ds, t_upper)

            if labels[i] == "Munk drag ON":
                psi_theo = munk(x_num, y_num, 1e5, Lx) / 1e6
            else:
                psi_theo = sverdrup(x_num, y_num, u10, Lx) / 1e6

            y_idx = np.argmin(np.abs(y_num - 0))

            num_mean = psi_mean[y_idx, :]
            num_inst = psi_inst[y_idx, :]
            theo = psi_theo[y_idx, :]

            errors_mean.append(relative_error_1d(num_mean, theo))
            errors_inst.append(relative_error_1d(num_inst, theo))
            t_axis.append(t_upper / 24)

        ax_conv.plot(t_axis, errors_mean, color=colors[i], label=f"{labels[i]} mean")
        ax_conv.plot(t_axis, errors_inst, color=colors[i], ls="--", label=f"{labels[i]} inst")

    ax_conv.set_xlabel("Time [days]")
    ax_conv.set_ylabel("relative error [-]")
    ax_conv.set_title("Convergence of Model to Theory", fontsize=14)
    ax_conv.legend()
    ax_conv.grid(alpha=0.3)

    plt.suptitle("Ocean Gyres Streamfunction", size=16)
    fig.set_constrained_layout(True)

    save_name = "sverdrup_vs_munk"
    plt.savefig(f"plots/{save_name}.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()


# =========================
# RUN
# =========================
plot_2xN(
    models=["drag_intermediate", "no_drag_normal_u10"],
    times=[2 * 24, 14 * 24, 50 * 24],
    labels=["Munk drag ON", "Munk drag OFF"],
    u10=10,
    show=False
)