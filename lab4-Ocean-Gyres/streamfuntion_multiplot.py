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


# ==========================================================
# THEORETICAL SOLUTIONS
# ==========================================================

def sverdrup(x, y, u10):

    dx_here = Ly / Nx

    tau_w = 1.225 * 1.5 * 0.001 * (u10**2)

    wE = (
        (1 / rho)
        * (1 / f0)
        * (-np.pi / Ly)
        * (tau_w / (rho * D))
        * np.cos(y * np.pi / Ly)
    )

    psi = np.zeros((len(y), len(x)))

    for j in range(len(y)):

        integral = 0.0

        for i in reversed(range(len(x))):

            if i < len(x) - 1:
                integral += wE[j] * dx_here

            psi[j, i] = -f0 / beta * integral

    return psi


def munk(x, y, A, u10=5):

    psi_s = sverdrup(x, y, u10=u10)

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

    psi = psi_s * factor_x[None, :]

    return psi


# ==========================================================
# SINGLE PANEL FUNCTION
# ==========================================================

def compute_streamfunction(folder, time):

    h_ds = xr.open_dataset(f"data/{folder}/h.nc")
    v_ds = xr.open_dataset(f"data/{folder}/v.nc")

    h = h_ds["h"].sel(time=time, method="nearest")
    v = v_ds["v"].sel(time=time, method="nearest")

    x = h_ds["x"].values
    y = h_ds["y"].values

    psi = -np.cumsum((v.values * D * dx)[:, ::-1], axis=1)[:, ::-1]

    if psi.shape[0] != len(y):
        y_plot = np.linspace(y[0], y[-1], psi.shape[0])
    else:
        y_plot = y

    return x, y_plot, psi / 1e6


# ==========================================================
# MULTI-PANEL PLOTTER
# ==========================================================

def plot_case_row(
    folders,
    values,
    label,
    save_name,
    time=1e10,
    u10=False,
    A=False,
    show=False
):

    n = len(folders)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(4*n, 4.4),
        sharex=True,
        sharey=True,
        squeeze=False
    )

    axes = axes[0]

    psi_all = []

    # ------------------------------------------------------
    # FIRST PASS (for common colorbar)
    # ------------------------------------------------------

    for folder in folders:

        x, y_plot, psi = compute_streamfunction(folder, time)

        psi_all.append(psi)

    vmin = np.min([p.min() for p in psi_all])
    vmax = np.max([p.max() for p in psi_all])

    # ------------------------------------------------------
    # SECOND PASS (actual plotting)
    # ------------------------------------------------------

    for i, ax in enumerate(axes):

        folder = folders[i]
        value = values[i]

        x, y_plot, psi = compute_streamfunction(folder, time)

        cf = ax.contourf(
            x / 1000,
            y_plot / 1000,
            psi,
            cmap="cmo.haline",
            levels=20,
            vmin=vmin,
            vmax=vmax
        )

        # ----------------------------------------------
        # THEORETICAL SOLUTIONS
        # ----------------------------------------------

        if A:

            psi_theo = munk(
                x,
                y_plot,
                A=value,
            )

            ax.contour(
                x / 1000,
                y_plot / 1000,
                psi_theo,
                colors="red",
                linewidths=1
            )
            text=f"{label} = {value:.0e}"+" m²/s"

        if u10:

            psi_theo = sverdrup(
                x,
                y_plot,
                u10=value,
            )

            ax.contour(
                x / 1000,
                y_plot / 1000,
                psi_theo,
                colors="red",
                linewidths=1
            )
            text=f"{label} = {value}"+" m/s"

        # ----------------------------------------------
        # LABELS
        # ----------------------------------------------

        ax.set_title(
            folder.replace("_", " ").title(),
            fontsize=13
        )

        ax.text(
            0.98,
            0.97,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(
                facecolor="white",
                alpha=0.8,
                edgecolor="none"
            )
        )

        ax.set_xlabel(r"Latitude, $x$ [km]",fontsize=11)

        if i == 0:
            ax.set_ylabel(r"Longitude, $y$ [km]",fontsize=11)

    # ------------------------------------------------------
    # COLORBAR
    # ------------------------------------------------------

    cbar = fig.colorbar(
        cf,
        ax=axes,
        location="right",
        shrink=0.85,
        pad=0.01
    )

    cbar.set_label(
        r"Streamfunction, $\Psi$ [Sv]",fontsize=11)

    plt.suptitle("Ocean Gyre Streamfunction", size=16)
    fig.set_constrained_layout(True)

    plt.savefig(
        f"plots/{save_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        f"ocean-gyres-report/Figures/{save_name}.png",
        dpi=300,
        bbox_inches="tight"
    )


    print(f"Saved as {save_name}")

    if show:
        plt.show()


# ==========================================================
# U10 COMPARISON
# ==========================================================

plot_case_row(
    folders=[
        "no_drag_low_u10",
        "no_drag_normal_u10",
        "no_drag_high_u10"],

    values=[4, 5, 6],
    label=r"$U_{10}$",
    save_name="u10_comparison",
    u10=True,
    time=2*24
    )


# ==========================================================
# MUNK COMPARISON
# ==========================================================

plot_case_row(
    folders=[
        "drag_low",
        "drag_normal",
        "drag_high"
    ],

    values=[100e3, 200e3, 400e3],
    label="A",
    save_name="munk_comparison",
    A=True,
    time=1e10
)