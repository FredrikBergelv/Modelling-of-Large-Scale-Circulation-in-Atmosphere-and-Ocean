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
# MODEL STREAMFUNCTION
# ==========================================================
def compute_streamfunction_average(folder, time_start):

    h_ds = xr.open_dataset(f"data/{folder}/h.nc")
    v_ds = xr.open_dataset(f"data/{folder}/v.nc")

    v = v_ds["v"].sel(time=slice(time_start, None))

    x = h_ds["x"].values
    y = h_ds["y"].values

    psi_all = -np.cumsum(
        (v.values * D * dx)[:, :, ::-1],
        axis=2
    )[:, :, ::-1]

    psi_mean = np.mean(psi_all, axis=0)
    psi_std = np.std(psi_all, axis=0)
    v_mean = np.mean(v.values, axis=0)
    v_std = np.std(v.values, axis=0)

    return x, y, psi_mean[:-1] / 1e6, 10*v_mean, psi_std / 1e6, 10*v_std


# ==========================================================
# CROSS SECTION EXTRACTOR
# ==========================================================
def extract_centerline(x, y, field):
    j0 = np.argmin(np.abs(y - 0))
    return field[j0, :]


# ==========================================================
# MERGED PLOT FUNCTION (3 rows: contourf + psi + v)
# ==========================================================
def plot_merged(
    folders,
    values,
    label,
    save_name,
    time_start=2*24,
    u10=False,
    A=False,
    show=False
):

    n = len(folders)

    fig, axes = plt.subplots(
        3,
        n,
        figsize=(3.7 * n, 7),
        sharex=False
    )

    model_profiles = []
    model_stds= []
    theory_profiles = []
    model_v_profiles = []
    model_v_stds = []
    theory_v_profiles = []
    contour_data = []

    x_global = None

    # ======================================================
    # LOAD DATA
    # ======================================================
    for i, folder in enumerate(folders):

        x, y, psi, v, psi_std, v_std = compute_streamfunction_average(
            folder,
            time_start
        )


        # Contourf data
        contour_data.append((x, y, psi))

        # Cross-section profiles
        model_profile = extract_centerline(x, y, psi)
        model_profiles.append(model_profile)
        
        model_profile_std = extract_centerline(x, y, psi_std)
        model_stds.append(model_profile_std)

        model_v = extract_centerline(x, y, v)
        model_v_profiles.append(Ekman_ratio * model_v)
        model_v_std = extract_centerline(x, y, v_std)
        model_v_stds.append(Ekman_ratio * model_v_std)

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
    # CONTOURF COLOR RANGE (row 0)
    # ======================================================
    vmin = np.min([d[2].min() for d in contour_data])
    vmax = np.max([d[2].max() for d in contour_data])

    # ======================================================
    # CROSS-SECTION LIMITS (rows 1 & 2)
    # ======================================================
    psi_all = np.concatenate(model_profiles + theory_profiles)
    v_all = np.concatenate(model_v_profiles + theory_v_profiles)

    psi_min, psi_max = psi_all.min(), psi_all.max()
    v_min, v_max = v_all.min(), v_all.max()

    # ======================================================
    # PLOTTING
    # ======================================================
    for i in range(n):

        ax_contour = axes[0, i]
        ax_psi     = axes[1, i]
        ax_v       = axes[2, i]
        
        if i != 0:
            ax_contour.tick_params(axis="y", labelleft=False)
            ax_psi.tick_params(axis="y", labelleft=False)
            ax_v.tick_params(axis="y", labelleft=False)
        ax_contour.tick_params(axis="x", labelbottom=False)
        ax_psi.tick_params(axis="x", labelbottom=False)
        
        
        x, y, psi = contour_data[i]
        x_plot = x / Lx
        y_plot = y / Ly
        
        ax_v.grid(alpha=0.3)
        ax_psi.grid(alpha=0.3)

        # --------------------------------------------------
        # ROW 0: CONTOURF
        # --------------------------------------------------
        cf = ax_contour.contourf(
            x_plot,
            y_plot,
            psi,
            cmap="cmo.haline",
            levels=8,
            vmin=vmin,
            vmax=vmax
        )

        if u10:

            psi_theo_2d = sverdrup(x, y, u10=values[i])

            cs = ax_contour.contour(
                x_plot,
                y_plot,
                psi_theo_2d,
                colors="red"
            )

            ax_contour.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

            text = f"{label} = {values[i]} m/s"

        if A:

            psi_theo_2d = munk(x, y, A=values[i])

            cs = ax_contour.contour(
                x_plot,
                y_plot,
                psi_theo_2d,
                colors="red"
            )

            ax_contour.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

            text = f"{label} = {values[i]:.0e} m²/s"

        ax_contour.set_title(
            folders[i].replace("_", " ").title(),
            fontsize=13
        )

        ax_contour.text(
            0.98, 0.97,
            text,
            transform=ax_contour.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
        )

        if i == 0:
            ax_contour.set_ylabel(r"Longitude, $y/L_y$ [-]", fontsize=11)

        # --------------------------------------------------
        # ROW 1: PSI CROSS-SECTION
        # --------------------------------------------------
        ax_psi.plot(x_plot, model_profiles[i],  label="Model mean")
        ax_psi.plot(x_plot, theory_profiles[i], "--",  color="black", label="Theory")
        ax_psi.fill_between(
            x_plot,
            model_profiles[i] - model_stds[i],
            model_profiles[i] + model_stds[i],
            color="C0",
            alpha=0.3,
            label="Model ±1 std"
        )
        ax_psi.set_ylim(psi_min, psi_max + np.max(model_stds))

        # --------------------------------------------------
        # ROW 2: V CROSS-SECTION
        # --------------------------------------------------
        ax_v.plot(x_plot, model_v_profiles[i],  color="C1", label="Model mean")
        ax_v.plot(x_plot, theory_v_profiles[i], "--",  color="black", label="Theory")
        ax_v.fill_between(
            x_plot,
            model_v_profiles[i] - model_v_stds[i],
            model_v_profiles[i] + model_v_stds[i],
            color="C1",
            alpha=0.3,
            label="Model ±1 std"
        )
        ax_v.set_ylim(v_min, v_max+np.max(model_v_stds))
        ax_v.set_xlabel("Latitude, $x/L_x$ [-]")

    # ======================================================
    # SHARED Y-LABELS
    # ======================================================
    axes[1, 0].set_ylabel(r"Streamfunction, $\Psi$ [Sv]")
    axes[2, 0].set_ylabel(r"Meridional Ekman velocity, $v$ [m/s]")

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

    # ======================================================
    # LEGEND FOR CROSS-SECTION ROWS
    # ======================================================
    axes[2, -1].legend(loc="center right")
    axes[1, -1].legend(loc="upper right")

    plt.suptitle("Time-averaged, Ocean Gyre Streamfunction", size=16)
    fig.set_constrained_layout(True)

    plt.savefig(f"plots/{save_name}.png", dpi=300, bbox_inches="tight")

    print(f"Saved as {save_name}")

    if show:
        plt.show()


# ==========================================================
# RUN
# ==========================================================
plot_merged(
    folders=[
        "no_drag_low_u10",
        "no_drag_intermediate_u10",
        "no_drag_high_u10"
    ],
    values=[4, 5, 6],
    label=r"$U_{10}$",
    save_name="sverdrup_u10",
    u10=True,
    time_start=2 * 24
)