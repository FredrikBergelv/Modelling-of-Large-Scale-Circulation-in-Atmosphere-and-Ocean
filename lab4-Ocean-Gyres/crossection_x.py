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

Ekman_ratio = D / 50


# ==========================================================
# THEORETICAL SOLUTIONS
# ==========================================================
def sverdrup(x, y, u10):

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

    return -psi / 1e6


def munk(x, y, A, u10=10):

    psi_s = sverdrup(x, y, u10=u10)

    dm = (A / beta) ** (1 / 3)

    print("dm/dx = ", dm / dx)

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


# ==========================================================
# MODEL STREAMFUNCTION (UPDATED)
# ==========================================================
def compute_streamfunction(folder, time_start=None):

    h_ds = xr.open_dataset(f"data/{folder}/h.nc")
    v_ds = xr.open_dataset(f"data/{folder}/v.nc")

    x = h_ds["x"].values
    y = h_ds["y"].values

    # ------------------------------------------------------
    # TIME AVERAGING (NEW)
    # ------------------------------------------------------
    if time_start is None:
        v = v_ds["v"].isel(time=-1)  # fallback: last timestep
    else:
        v = v_ds["v"].sel(time=slice(time_start, None)).mean("time")

    v = v.values

    # ------------------------------------------------------
    # STREAMFUNCTION
    # ------------------------------------------------------
    psi = -np.cumsum(
        (v * D * dx)[:, ::-1],
        axis=1
    )[:, ::-1]

    return x, y, psi / 1e6, v


# ==========================================================
# CROSS SECTION EXTRACTOR
# ==========================================================
def extract_centerline(x, y, field):
    j0 = np.argmin(np.abs(y - 0))
    return field[j0, :]


# ==========================================================
# PLOT ROW FUNCTION
# ==========================================================
def plot_case_row_cross(
    folders,
    values,
    label,
    save_name,
    time_start=24*2,
    u10=False,
    A=False,
    show=False
):

    n = len(folders)

    fig, axes = plt.subplots(
        2,
        n,
        figsize=(3.7 * n, 6),
        sharex=True
    )

    model_profiles = []
    theory_profiles = []

    model_v_profiles = []
    theory_v_profiles = []

    x_global = None

    # ======================================================
    # LOAD DATA
    # ======================================================
    for i, folder in enumerate(folders):

        x, y, psi, v = compute_streamfunction(folder, time_start)

        x_global = x

        # MODEL
        model_profile = extract_centerline(x, y, psi)
        model_profiles.append(model_profile)

        model_v = extract_centerline(x, y, v)
        model_v_profiles.append(Ekman_ratio * model_v)

        # THEORY
        if A:
            psi_theo = munk(x, y, A=values[i])
        if u10:
            psi_theo = sverdrup(x, y, u10=values[i])

        theory_profile = extract_centerline(x, y, psi_theo)
        theory_profiles.append(theory_profile)

        theory_v = np.gradient(
            theory_profile * 1e6,
            dx
        ) / D

        theory_v_profiles.append(Ekman_ratio * theory_v)

    # ======================================================
    # COMMON LIMITS
    # ======================================================
    psi_all = np.concatenate(model_profiles + theory_profiles)
    v_all = np.concatenate(model_v_profiles + theory_v_profiles)

    psi_min, psi_max = psi_all.min(), psi_all.max()
    v_min, v_max = v_all.min(), v_all.max()

    # ======================================================
    # PLOTTING
    # ======================================================
    for i in range(n):

        ax_top = axes[0, i]
        ax_bottom = axes[1, i]

        x_km = x_global / 1000

        # PSI
        ax_top.plot(x_km, model_profiles[i], linewidth=2, label="Model")
        ax_top.plot(x_km, theory_profiles[i], "--", linewidth=2, color="black", label="Theory")
        ax_top.set_ylim(psi_min, psi_max)

        # V
        ax_bottom.plot(x_km, model_v_profiles[i], linewidth=2, color="C1", label="Model")
        ax_bottom.plot(x_km, theory_v_profiles[i], "--", linewidth=2, color="black", label="Theory")
        ax_bottom.set_ylim(v_min, v_max)

        ax_top.set_title(folders[i].replace("_", " ").title())

        ax_top.grid(alpha=0.3)
        ax_bottom.grid(alpha=0.3)

        ax_bottom.set_xlabel("Latitude, x [km]")

        # TEXT
        if A:
            ax_top.text(
                0.95, 0.95,
                rf"$A={values[i]:.0e}$ m²/s",
                transform=ax_top.transAxes,
                ha="right",
                va="top",
                bbox=dict(facecolor="white", alpha=0.8)
            )

        if u10:
            ax_top.text(
                0.95, 0.95,
                rf"$U_{{10}}={values[i]}$ m/s",
                transform=ax_top.transAxes,
                ha="right",
                va="top",
                bbox=dict(facecolor="white", alpha=0.8)
            )

    # ======================================================
    axes[0, 0].set_ylabel(r"Streamfunction, $\Psi$ [Sv]")
    axes[1, 0].set_ylabel(r"Meridional Ekman velocity, $v$ [m/s]")

    plt.suptitle("Time-averaged, Gyre Streamfunction Cross-Section (y = 0)", size=14)

    fig.set_constrained_layout(True)
    plt.legend(loc="center right")

    plt.savefig(f"plots/{save_name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"ocean-gyres-report/Figures/{save_name}.png", dpi=300, bbox_inches="tight")

    print(f"Saved as {save_name}")

    if show:
        plt.show()


# ==========================================================
# RUN CASES
# ==========================================================
plot_case_row_cross(
    folders=["drag_low", "drag_intermediate", "drag_high"],
    values=[30e3, 100e3, 500e3],
    label="A",
    save_name="munk_cross_section",
    A=True,
    time_start=24 * 2
)