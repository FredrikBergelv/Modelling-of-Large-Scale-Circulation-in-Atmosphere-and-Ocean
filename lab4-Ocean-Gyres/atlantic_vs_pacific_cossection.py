import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cmocean

# ==========================================================
# PARAMETERS
# ==========================================================
D = 3000

rho = 1027
f0 = 7.27e-5
beta = 1.98e-11

Ekman_ratio = D / 50


# ==========================================================
# THEORETICAL SOLUTIONS
# ==========================================================
def sverdrup(x, y, u10, Lx):

    Nx = len(x)
    dx = Lx / Nx

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
                integral += v_s[j] * dx

            psi[j, i] = integral

    return -psi / 1e6


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

    dx = x[1] - x[0]

    psi = -np.cumsum(
        (v.values * D * dx)[:, ::-1],
        axis=1
    )[:, ::-1]

    return x, y, psi / 1e6, v.values


# ==========================================================
# CROSS SECTION EXTRACTOR
# ==========================================================
def extract_centerline(y, field):

    j0 = np.argmin(np.abs(y - 0))

    return field[j0, :]


# ==========================================================
# DOMAIN COMPARISON FUNCTION
# ==========================================================
def plot_domain_compare_cross(
    folders,
    labels,
    Lx_values,
    u10_value=5,
    save_name="domain_compare_cross",
    time=2*24,
    show=False
):

    n = len(folders)

    fig, axes = plt.subplots(
        2,
        n,
        figsize=(3.7*n, 6),
        sharex=False
    )

    # ------------------------------------------------------
    # STORAGE
    # ------------------------------------------------------
    model_profiles = []
    theory_profiles = []

    model_v_profiles = []
    theory_v_profiles = []

    x_profiles = []

    # ------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------
    for i, folder in enumerate(folders):

        x, y, psi, v = compute_streamfunction(
            folder,
            time
        )

        dx = x[1] - x[0]

        x_profiles.append(x)

        # --------------------------------------------------
        # MODEL
        # --------------------------------------------------
        model_profile = extract_centerline(y, psi)
        model_profiles.append(model_profile)

        model_v = extract_centerline(y, v)
        model_v_profiles.append(Ekman_ratio * model_v)

        # --------------------------------------------------
        # THEORY
        # --------------------------------------------------
        psi_theo = sverdrup(
            x,
            y,
            u10=u10_value,
            Lx=Lx_values[i]
        )

        theory_profile = extract_centerline(
            y,
            psi_theo
        )

        theory_profiles.append(theory_profile)

        # v = dpsi/dx
        theory_v = np.gradient(
            theory_profile * 1e6,
            dx
        ) / D

        theory_v_profiles.append(
            Ekman_ratio * theory_v
        )

    # ------------------------------------------------------
    # COMMON Y LIMITS
    # ------------------------------------------------------
    psi_all = np.concatenate(
        model_profiles + theory_profiles
    )

    v_all = np.concatenate(
        model_v_profiles + theory_v_profiles
    )

    psi_min = psi_all.min()
    psi_max = psi_all.max()

    v_min = v_all.min()
    v_max = v_all.max()

    # ------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------
    for i in range(n):

        ax_top = axes[0, i]
        ax_bottom = axes[1, i]

        x_km = x_profiles[i] / 1000

        # ==================================================
        # TOP ROW : PSI
        # ==================================================
        ax_top.plot(
            x_km,
            model_profiles[i],
            linewidth=2,
            label="Model"
        )

        ax_top.plot(
            x_km,
            theory_profiles[i],
            linestyle="--",
            linewidth=2,
            color="black",
            label="Theory"
        )

        ax_top.set_ylim(psi_min, psi_max)

        # ==================================================
        # BOTTOM ROW : V
        # ==================================================
        ax_bottom.plot(
            x_km,
            model_v_profiles[i],
            linewidth=2,
            color="C1",
            label="Model"
        )

        ax_bottom.plot(
            x_km,
            theory_v_profiles[i],
            linestyle="--",
            linewidth=2,
            color="black",
            label="Theory"
        )

        ax_bottom.set_ylim(v_min, v_max)

        # ==================================================
        # LABELS
        # ==================================================
        ax_top.set_title(labels[i])

        ax_top.grid(alpha=0.3)
        ax_bottom.grid(alpha=0.3)

        ax_bottom.set_xlabel("Latitude, x [km]")

        # ==================================================
        # TEXT
        # ==================================================
        ax_top.text(
            0.95,
            0.95,
            rf"$L_x={Lx_values[i]/1e6:.0f}\times10^6$ m",
            transform=ax_top.transAxes,
            ha="right",
            va="top",
            bbox=dict(
                facecolor="white",
                alpha=0.8
            )
        )

    # ======================================================
    # AXIS LABELS
    # ======================================================
    axes[0, 0].set_ylabel(
        r"Streamfunction, $\Psi$ [Sv]"
    )

    axes[1, 0].set_ylabel(
        r"Meridional Ekman velocity, $v$ [m/s]"
    )

    # ======================================================
    # LAYOUT
    # ======================================================
    plt.suptitle(
        "Domain Size Comparison (y = 0)",
        size=14
    )

    fig.set_constrained_layout(True)

    plt.legend(loc="center right")

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
# RUN CASE
# ==========================================================
plot_domain_compare_cross(
    folders=[
        "no_drag_intermediate_u10",
        "no_drag_beta_plane_large_domain"
    ],

    labels=[
        "North Atlantic",
        "North Pacific"
    ],

    Lx_values=[
        7e6,
        14e6
    ],

    u10_value=5,

    save_name="domain_compare_cross",

    time=2 * 24,

    show=False
)