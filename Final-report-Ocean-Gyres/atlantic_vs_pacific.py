import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cmocean
from matplotlib.gridspec import GridSpec

# ==========================================================
# PARAMETERS
# ==========================================================
D = 3000
Ly_default = 7e6

rho = 1027
f0 = 7.27e-5
beta = 1.98e-11

Ekman_ratio = D / 50


# ==========================================================
# THEORETICAL SOLUTIONS
# ==========================================================
def sverdrup(x, y, u10, Lx, Nx=None):

    if Nx is None:
        Nx = len(x)

    dx_here = Lx / Nx

    tau0 = 1.225 * 1.5 * 0.001 * (u10**2)

    Ly = np.max(y) - np.min(y)

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


# ==========================================================
# MODEL STREAMFUNCTION
# ==========================================================
def compute_streamfunction(folder, time_start):

    h_ds = xr.open_dataset(f"data/{folder}/h.nc")
    v_ds = xr.open_dataset(f"data/{folder}/v.nc")

    v = v_ds["v"].where(
        v_ds["time"] >= time_start,
        drop=True
    )

    x = h_ds["x"].values
    y = h_ds["y"].values

    dx = x[1] - x[0]

    # --------------------------------------------------
    # TIME SERIES STREAMFUNCTION (before averaging)
    # --------------------------------------------------
    psi_all = -np.cumsum(
        (v.values * D * dx)[:, :, ::-1],
        axis=2
    )[:, :, ::-1]   # shape: (time, y, x)

    # --------------------------------------------------
    # STATISTICS
    # --------------------------------------------------
    psi_mean = np.mean(psi_all, axis=0)
    psi_std  = np.std(psi_all, axis=0)

    v_mean = np.mean(v.values, axis=0)
    v_std  = np.std(v.values, axis=0)

    # --------------------------------------------------
    # GRID FIX
    # --------------------------------------------------
    if psi_mean.shape[0] != len(y):
        y_plot = np.linspace(y[0], y[-1], psi_mean.shape[0])
    else:
        y_plot = y

    return x, y_plot, psi_mean / 1e6, psi_std / 1e6, v_mean, v_std


# ==========================================================
# CROSS SECTION EXTRACTOR
# ==========================================================
def extract_centerline(y, field):
    j0 = np.argmin(np.abs(y - 0))
    return field[j0, :]


# ==========================================================
# MERGED PLOT FUNCTION (3 rows: contourf + psi + v)
# ==========================================================
def plot_domain_compare_merged(
    folders,
    labels,
    Lx_values,
    Nx_values,
    Lx_ref=7e6,
    u10_value=5,
    save_name="domain_compare_merged",
    time_start=2*24,
    show=False
):

    n = len(folders)

    # Width ratios: proportional to Lx
    width_ratios = [Lx / Lx_ref for Lx in Lx_values]

    fig = plt.figure(figsize=(sum(width_ratios) * 3.7, 7))

    gs = GridSpec(
        3, n,
        figure=fig,
        width_ratios=width_ratios
    )

    axes = np.array([
        [fig.add_subplot(gs[row, col]) for col in range(n)]
        for row in range(3)
    ])

    # ======================================================
    # STORAGE
    # ======================================================
    model_profiles = []
    model_stds = []
    theory_profiles = []
    model_v_profiles = []
    model_v_stds = []
    theory_v_profiles = []
    contour_data = []
    x_profiles = []

    # ======================================================
    # LOAD DATA
    # ======================================================
    for i, folder in enumerate(folders):

        x, y, psi, psi_std, v, v_std = compute_streamfunction(folder, time_start)

        dx = x[1] - x[0]

        x_profiles.append(x)
        contour_data.append((x, y, psi))

        model_profile = extract_centerline(y, psi)
        model_profiles.append(model_profile)
        model_profile_std = extract_centerline(y, psi_std)
        model_stds.append(model_profile_std)

        model_v = extract_centerline(y, v)
        model_v_profiles.append(Ekman_ratio * model_v)
        model_v_std = extract_centerline(y, v_std)
        model_v_stds.append(Ekman_ratio * model_v_std)

        psi_theo = sverdrup(
            x, y,
            u10=u10_value,
            Lx=Lx_values[i],
            Nx=Nx_values[i]
        )

        theory_profile = extract_centerline(y, psi_theo)
        theory_profiles.append(theory_profile)

        theory_v = np.gradient(theory_profile * 1e6, dx) / D
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

        # Normalise x by Lx_ref so Pacific runs 0→2
        x_plot = x / Lx_ref
        y_plot = y / Ly_default

        # --------------------------------------------------
        # ROW 0: CONTOURF
        # --------------------------------------------------
        cf = ax_contour.contourf(
            x_plot, y_plot, psi,
            cmap="cmo.haline",
            levels=8,
            vmin=vmin,
            vmax=vmax
        )

        psi_theo_2d = sverdrup(
            x, y,
            u10=u10_value,
            Lx=Lx_values[i],
            Nx=Nx_values[i]
        )

        cs = ax_contour.contour(
            x_plot, y_plot, psi_theo_2d,
            colors="red"
        )

        ax_contour.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

        ax_contour.set_title(labels[i], fontsize=13)

        ax_contour.text(
            0.95, 0.97,
            rf"$L_x={Lx_values[i]/1e6:.0f}\times10^6$ m",
            transform=ax_contour.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
        )

        if i == 0:
            ax_contour.set_ylabel(r"Longitude, $y/L_y$ [-]", fontsize=11)

        # --------------------------------------------------
        # ROW 1: PSI CROSS-SECTION
        # --------------------------------------------------
        x_plot_1d = x_profiles[i] / Lx_ref

        ax_psi.plot(x_plot_1d, model_profiles[i], label="Model mean")
        ax_psi.plot(x_plot_1d, theory_profiles[i], "--", color="black", label="Theory")
        ax_psi.fill_between(
            x_plot_1d,
            model_profiles[i] - model_stds[i],
            model_profiles[i] + model_stds[i],
            color="C0", alpha=0.3, label="Model ±1 std"
        )
        ax_psi.set_ylim(psi_min, 22)
        ax_psi.grid(alpha=0.3)


        # --------------------------------------------------
        # ROW 2: V CROSS-SECTION
        # --------------------------------------------------
        ax_v.plot(x_plot_1d, model_v_profiles[i], color="C1", label="Model mean")
        ax_v.plot(x_plot_1d, theory_v_profiles[i], "--", color="black", label="Theory")
        ax_v.fill_between(
            x_plot_1d,
            model_v_profiles[i] - model_v_stds[i],
            model_v_profiles[i] + model_v_stds[i],
            color="C1", alpha=0.3, label="Model ±1 std"
        )
        ax_v.set_ylim(v_min, 9)
        ax_v.grid(alpha=0.3)
        ax_v.set_xlabel(r"Latitude, $x/L_{x_{ref}}$ [-]")

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
        shrink=0.8,
        pad=0.01
    )
    cbar.set_label(r"Streamfunction, $\Psi$ [Sv]", fontsize=11)

    # ======================================================
    # LEGEND
    # ======================================================
    axes[2, -1].legend(loc="center right")
    axes[1, -1].legend(loc="upper right")

    plt.suptitle("Ocean Gyres Streamfunction", size=16)

    fig.set_constrained_layout(True)

    plt.savefig(f"plots/{save_name}.png", dpi=300, bbox_inches="tight")

    print(f"Saved as {save_name}")

    if show:
        plt.show()


# ==========================================================
# RUN
# ==========================================================
plot_domain_compare_merged(
    folders=[
        "no_drag_intermediate_u10",
        "no_drag_beta_plane_large_domain"
    ],
    labels=[
        "North Atlantic",
        "North Pacific"
    ],
    Lx_values=[7e6, 14e6],
    Nx_values=[200, 400],
    Lx_ref=7e6,
    u10_value=5,
    save_name="atlantic_vs_pacific",
    time_start=2 * 24,
    show=False
)