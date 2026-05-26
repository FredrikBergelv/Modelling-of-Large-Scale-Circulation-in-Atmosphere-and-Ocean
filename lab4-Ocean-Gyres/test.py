import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cmocean

# ==========================================================
# PARAMETERS
# ==========================================================
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

    dx_here = Lx / Nx

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
            + (1 / np.sqrt(3)) * np.sin(x * np.sqrt(3) / (2 * dm))
        )
    )

    return psi_s * factor_x[None, :]


# ==========================================================
# MODEL STREAMFUNCTION
# ==========================================================
def compute_streamfunction(folder, time):

    h_ds = xr.open_dataset(f"data/{folder}/h.nc")
    v_ds = xr.open_dataset(f"data/{folder}/v.nc")

    h = h_ds["h"].sel(time=time, method="nearest")
    v = v_ds["v"].sel(time=time, method="nearest")

    x = h_ds["x"].values
    y = h_ds["y"].values

    psi = -np.cumsum((v.values * D * dx)[:, ::-1], axis=1)[:, ::-1]

    return x, y, psi / 1e6


# ==========================================================
# CROSS SECTION EXTRACTOR (y = 0)
# ==========================================================
def extract_centerline(x, y, psi):

    # find index closest to y=0
    j0 = np.argmin(np.abs(y - 0))

    return psi[j0, :]


def extract_theory_centerline(x, y, psi_theo):

    j0 = np.argmin(np.abs(y - 0))

    return psi_theo[j0, :]


# ==========================================================
# PLOT ROW FUNCTION (CROSS SECTIONS)
# ==========================================================
def plot_case_row_cross(
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

    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), sharey=True)

    if n == 1:
        axes = [axes]

    # ------------------------------------------------------
    # FIRST PASS (model data)
    # ------------------------------------------------------
    model_profiles = []
    theory_profiles = []
    x_global = None

    for i, folder in enumerate(folders):

        x, y, psi = compute_streamfunction(folder, time)
        x_global = x

        model_profiles.append(extract_centerline(x, y, psi))

        # THEORY
        if A:
            psi_theo = munk(x, y, A=values[i])
            theory_profiles.append(extract_theory_centerline(x, y, psi_theo))

        if u10:
            psi_theo = sverdrup(x, y, u10=values[i])
            theory_profiles.append(extract_theory_centerline(x, y, psi_theo))

    # ------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------
    for i, ax in enumerate(axes):

        x_km = x_global / 1000

        # MODEL (solid)
        ax.plot(
            x_km,
            model_profiles[i],
            linewidth=2,
            label="Model"
        )

        # THEORY (dotted)
        if (A or u10):
            ax.plot(
                x_km,
                theory_profiles[i],
                linestyle="dotted",
                linewidth=2,
                color="black",
                label="Theory"
            )

        # LABELS
        ax.set_title(folders[i].replace("_", " ").title())

        ax.set_xlabel("x [km]")
        ax.grid(alpha=0.3)

        # parameter text
        if A:
            ax.text(
                0.95, 0.95,
                rf"$A={values[i]:.0e}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(facecolor="white", alpha=0.8)
            )

        if u10:
            ax.text(
                0.95, 0.95,
                rf"$U_{{10}}={values[i]}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(facecolor="white", alpha=0.8)
            )

    axes[0].set_ylabel(r"$\Psi$ [Sv]")

    plt.suptitle("Gyre Streamfunction Cross-Section (y = 0)", size=14)
    fig.set_constrained_layout(True)

    plt.savefig(f"plots/{save_name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"ocean-gyres-report/Figures/{save_name}.png", dpi=300, bbox_inches="tight")

    print(f"Saved as {save_name}")

    if show:
        plt.show()


# ==========================================================
# RUN CASES
# ==========================================================

# WIND SPEED COMPARISON
plot_case_row_cross(
    folders=[
        "no_drag_low_u10",
        "no_drag_normal_u10",
        "no_drag_high_u10"
    ],
    values=[4, 5, 6],
    label=r"$U_{10}$",
    save_name="u10_cross_section",
    u10=True,
    time=2 * 24
)

# MUNK COMPARISON
plot_case_row_cross(
    folders=[
        "drag_low",
        "drag_normal",
        "drag_high"
    ],
    values=[100e3, 200e3, 400e3],
    label="A",
    save_name="munk_cross_section",
    A=True,
    time=1e10
)