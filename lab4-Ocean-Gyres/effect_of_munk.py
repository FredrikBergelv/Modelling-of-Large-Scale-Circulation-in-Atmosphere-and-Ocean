import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cmocean

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


# =========================
# THEORY
# =========================
def sverdrup(x, y, u10, Lx=Lx):

    dx_here = Lx / Nx

    # wind stress amplitude
    tau0 = 1.225 * 1.5 * 0.001 * (u10**2)

    # curl(tau)
    curl_tau = (
        -np.pi / Ly
        * tau0
        * np.cos(np.pi * y / Ly)
    )

    # Sverdrup meridional transport
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
        - np.exp(-x / (2 * dm)) *
        (
            np.cos(x * np.sqrt(3) / (2 * dm))
            + (1 / np.sqrt(3)) * np.sin(x * np.sqrt(3) / (2 * dm))
        )
    )

    return psi_s * factor_x[None, :]


# =========================
# STREAMFUNCTION FROM MODEL
# =========================
def compute_psi(h_ds, v_ds, time):
    h = h_ds["h"].sel(time=time, method="nearest")
    v = v_ds["v"].sel(time=time, method="nearest")

    psi = -np.cumsum((v.values * D * dx)[:, ::-1], axis=1)[:, ::-1]
    return psi, h["time"].values


# =========================
# MAIN PLOT: 2 ROWS × N COLS
# =========================
def plot_2xN(models, times, labels, u10=None, A=None, Lx=Lx, show=False):

    nrows = len(models)
    ncols = len(times)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 4 * nrows),
        sharex=True,
        sharey=True
    )

    axes = np.atleast_2d(axes)

    all_psi = []

    data_store = []

    # =========================
    # LOAD ALL DATA FIRST
    # =========================
    for model in models:

        h_ds = xr.open_dataset(f"data/{model}/h.nc")
        v_ds = xr.open_dataset(f"data/{model}/v.nc")

        x = h_ds["x"].values
        y = h_ds["y"].values

        psi_row = []
        time_row = []

        for t in times:
            psi, actual_time = compute_psi(h_ds, v_ds, t)

            psi = psi / 1e6  # Sv

            # FIX GRID ISSUE (important!)
            if psi.shape[0] != len(y):
                y_plot = np.linspace(y[0], y[-1], psi.shape[0])
            else:
                y_plot = y

            psi_row.append((psi, y_plot))
            time_row.append(actual_time)

            all_psi.append(psi.ravel())

        data_store.append((model, x, psi_row, time_row))

    all_psi = np.concatenate(all_psi)
    vmin, vmax = all_psi.min(), all_psi.max()

    # =========================
    # PLOTTING
    # =========================
    for i, (model, x, psi_row, time_row) in enumerate(data_store):

        for j, ax in enumerate(axes[i]):

            psi, y_plot = psi_row[j]
            t = time_row[j]

            cf = ax.contourf(
                x / 1000,
                y_plot / 1000,
                psi,
                levels=20,
                cmap="cmo.haline",
                vmin=0,
                vmax=8
            )

            # -------- THEORY --------
            if A is not None:
                psi_theo = munk(x, y_plot, A, Lx)
                ax.contour(x / 1000, y_plot / 1000, psi_theo, colors="red", linewidths=1)

            if u10 is not None:
                psi_theo = sverdrup(x, y_plot, u10, Lx)
                ax.contour(x / 1000, y_plot / 1000, psi_theo, colors="red", linewidths=1)

            # -------- LABELS --------
            days = t // 24
            hours = t % 24

            if i ==0:
                ax.set_title(f"t={days:.0f}d {hours:.0f}h",fontsize=13)

            # MODEL LABEL (upper right)
            ax.text(
                0.98, 0.95,
                labels[i],
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=11,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
            )

            if i == nrows - 1:
                ax.set_xlabel(r"Latitude, $x$ [km]",fontsize=11)
            if j == 0:
                ax.set_ylabel(r"Longitude, $y$ [km]",fontsize=11)

    # =========================
    # COLORBAR
    # =========================
    cbar = fig.colorbar(cf, ax=axes, shrink=0.5, pad=0.02)
    cbar.set_label(r"Streamfunction, $\Psi$ [Sv],fontsize=11")

    plt.suptitle("Ocean Gyres Streamfunction", size=16)
    fig.set_constrained_layout(True)
    
    save_name = "steamfuntion_munk_compare"
    plt.savefig(f"plots/{save_name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"ocean-gyres-report/Figures/{save_name}.png", dpi=300, bbox_inches="tight")
    print(f"Saved as {save_name}")
    if show:
        plt.show()


# =========================
# RUN
# =========================
plot_2xN(
    models=[
        "drag_normal",
        "no_drag_normal_u10"
    ],
    times=[2 * 24, 14 * 24, 30 * 24],
    labels=[
        "Munk drag ON",
        "Munk drag OFF"
    ],
    u10=10,
    show=False
)