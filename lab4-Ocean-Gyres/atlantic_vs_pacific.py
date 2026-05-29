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


# =========================
# THEORETICAL MODELS
# =========================

def sverdrup(x, y, u10, Nx, Lx):

    # ------------------------------------------------------
    # USE ACTUAL GRID SIZE
    # ------------------------------------------------------
    Nx = len(x)

    dx_here = Lx / Nx

    # wind stress amplitude
    tau0 = 1.225 * 1.5 * 0.001 * (u10**2)

    # basin size
    Ly = np.max(y) - np.min(y)

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

    # IMPORTANT
    return -psi


def munk(x, y, A, Lx, Nx=Nx):

    psi_s = sverdrup(
        x,
        y,
        u10=5,
        Lx=Lx,
        Nx=Nx
    )

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


# =========================
# NEW: COMPARISON FUNCTION
# =========================

def plot_streamfunction_compare(
    folders,
    labels=None,
    u10=False,
    A=False,
    Lx=Lx,
    Nx=Nx,
    time_start=2*24,
    show=False
):

    fig, axes = plt.subplots(
        1,
        len(folders),
        figsize=(4 * len(folders), 4),
        sharey=True
    )

    axes = np.atleast_1d(axes)

    all_psi = []

    results = []

    # ------------------------------------------------------
    # LOAD ALL DATASETS
    # ------------------------------------------------------
    for folder in folders:

        h_ds = xr.open_dataset(f"data/{folder}/h.nc")
        v_ds = xr.open_dataset(f"data/{folder}/v.nc")

        x = h_ds["x"].values
        y = h_ds["y"].values

        dx_local = x[1] - x[0]

        # --------------------------------------------------
        # TIME-AVERAGED VELOCITY
        # --------------------------------------------------
        v_mean = v_ds["v"].where(
            v_ds["time"] >= time_start,
            drop=True
        ).mean(dim="time")

        psi = -np.cumsum(
            (v_mean.values * D * dx_local)[:, ::-1],
            axis=1
        )[:, ::-1] / 1e6

        # --------------------------------------------------
        # FIX y mismatch
        # --------------------------------------------------
        if psi.shape[0] != len(y):
            y_plot = np.linspace(
                y[0],
                y[-1],
                psi.shape[0]
            )
        else:
            y_plot = y

        all_psi.append(psi.ravel())

        results.append(
            (
                folder,
                x,
                y_plot,
                psi
            )
        )

    # ------------------------------------------------------
    # COMMON COLOR LIMITS
    # ------------------------------------------------------
    all_psi = np.concatenate(all_psi)

    vmin = all_psi.min()
    vmax = all_psi.max()

    # ------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------
    for i, (folder, x, y, psi) in enumerate(results):

        ax = axes[i]

        cf = ax.contourf(
            x / 1000,
            y / 1000,
            psi,
            cmap="cmo.haline",
            levels=8,
            vmin=vmin,
            vmax=vmax,
        )

        if u10:

            if (folder == "no_drag_intermediate_u10"):
                Lx = 7000000
                Nx = 200
            elif (folder == "no_drag_beta_plane_large_domain"):
                Lx = 14000000
                Nx = 400

            psi_theo = sverdrup(
                x,
                y,
                u10,
                Lx,
                Nx
            )

            cs=ax.contour(
                x / 1000,
                y / 1000,
                psi_theo,
                colors="red",
                linewidths=1
            )
            

            # Inline contour labels
            ax.clabel(
                cs,
                inline=True,
                fontsize=8,
                fmt="%.1f"
            )



        # --------------------------------------------------
        # LABELS
        # --------------------------------------------------
        label = labels[i] if labels else folder

        ax.set_title(
            label,
            fontsize=13
        )

        ax.set_xlabel(
            r"Latitude, $x$ [km]",
            fontsize=11
        )

        if i == 0:

            ax.set_ylabel(
                r"Longitude, $y$ [km]",
                fontsize=11
            )
        
        if (label == "North Atlantic"):
            txt = 7000000
        elif (label == "North Pacific"):
            txt = 14000000

        ax.text(
            0.95,
            0.95,
            rf"$L_x={txt/1e6:.0f}\times10^6$ m",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(
                facecolor="white",
                alpha=0.8
            )
        )

    # ------------------------------------------------------
    # COLORBAR
    # ------------------------------------------------------
    cbar = fig.colorbar(
        cf,
        ax=axes,
        shrink=0.8
    )

    cbar.set_label(
        r"Streamfunction, $\Psi$ [Sv]",
        fontsize=11
    )

    # ------------------------------------------------------
    # LAYOUT
    # ------------------------------------------------------
    fig.set_constrained_layout(True)

    plt.suptitle(
        "Ocean Gyre Streamfunction",
        size=16
    )

    save_name = "steamfuntion_domain_compare"

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


# =========================
# RUN YOUR CASE
# =========================

plot_streamfunction_compare(
    folders=[
        "no_drag_intermediate_u10",
        "no_drag_beta_plane_large_domain"
    ],

    labels=[
        "North Atlantic",
        "North Pacific"
    ],

    u10=True,

    Lx=14e6,
    Nx=400,

    time_start=2 * 24,

    show=False
)