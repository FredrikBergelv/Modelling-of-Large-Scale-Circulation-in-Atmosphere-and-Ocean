import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# --- Load data ---
def load_hcase(path):
    ds = xr.open_dataset(path)
    return ds["h"]

def load_vcase(path):
    ds = xr.open_dataset(path)
    return ds["v"]

h_smallf = load_hcase('data_smallf/h.nc')
h_largef = load_hcase('data_normal/h.nc')
h_smallH = load_hcase('data_smallH/h.nc')
h_largeH = load_hcase('data_largeH/h.nc')

v_smallf = load_vcase('data_smallf/v.nc')
v_largef = load_vcase('data_normal/v.nc')
v_smallH = load_vcase('data_smallH/v.nc')
v_largeH = load_vcase('data_largeH/v.nc')

cases = [
    (h_smallH, v_smallH, 500, 1e-4),
    (h_largef, v_largef, 4000, 1e-4),
    (h_largeH, v_largeH, 10000, 1e-4),
    (h_smallf, v_smallf, 4000, 1e-5)
]

titles = [
    r"$f=10^{-4}s^{-1}$, $H=500$ m",
    r"$f=10^{-4}s^{-1}$, $H=4000$ m",
    r"$f=10^{-4}s^{-1}$, $H=10000$ m",
    r"$f=10^{-5}s^{-1}$, $H=4000$ m"
]

# --- constants ---
g = 9.81
eta0 = 5

def h_theo_func(x, H, f, x_middle):
    c = np.sqrt(g * H)
    R = c / f
    x = np.asarray(x)

    return np.where(
        x >= x_middle,
        eta0 * (-1 + np.exp(-(x - x_middle) / R)) ,
        eta0 * (1 - np.exp((x - x_middle) / R))
        )
    
def v_theo_func(x, H, f, x_middle):
    c = np.sqrt(g * H)
    R = c / f
    x = np.asarray(x)

    return -g * eta0 / (R * f) * np.exp(-np.abs(x - x_middle) / R)

def l2_error_func(h_num, h_ref):
    return 100 * np.sqrt(np.mean((h_num - h_ref)**2)) / np.sqrt(np.mean(h_ref**2))

# --- FIGURE ---
fig, axs = plt.subplots(2, 4, figsize=(12, 6), sharey=True)

for i, (h, v, H, f) in enumerate(cases):

    x = h.x.values
    x_middle = float(x.mean())

    # theoretical profile (1D)
    h_theo = h_theo_func(x, H, f, x_middle)

    # numerical (time, x)
    h_num = h.mean(dim="y").values

    # compute error over time
    errors = [
        l2_error_func(h_num[t, :], h_theo)
        for t in range(h_num.shape[0])
        ]
    """
    if errors[-1] > 50:
        fig2, ax2 = plt.subplots()

        # pick last timestep (or any interesting one)
        t_idx = -1

        ax2.plot(x, h_num[t_idx, :], label="Numerical")
        ax2.plot(x, h_theo, "--", label="Theoretical")

        ax2.set_title(titles[i] + f"\n(final time, large error)")
        ax2.set_xlabel("x")
        ax2.set_ylabel(r"$\eta$")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.show()
    """
    time = h.time.values
    
    R = np.sqrt(g*H) / f
    errors = np.array(errors)
    mask = errors < 15   # since your errors are in \%

    axs[0, i].fill_between(
        time, 0, 200,
        where=mask,
        color="green",
        alpha=0.3,
        label="< 15% error"
    )
    legend = r"$\left\|\eta(x,t)-\eta_{\text{theo}}(x,\infty)\right\| / \left\|\eta_{\text{theo}}(x,\infty)\right\|$"
    
    axs[0, i].plot(time, errors, c=f"C{i}", label=legend)
    axs[0, i].set_title(titles[i]+f"\n (R={R/1000:.0f} km)")
    axs[0, i].set_xlabel("Time [h]")
    axs[0, i].set_ylim(0, 100)

    if i == 0:
        axs[0, i].set_ylabel("Surface height L2 error [%]")
    if i == 3:
        axs[0, i].legend(loc="lower left")

    axs[0, i].grid(True, linestyle="--", alpha=0.6)
    
    
    
    # ---- Plot v-section ----
    
    x = v.x.values
    x_middle = float(x.mean())

    # theoretical profile (1D)
    v_theo = v_theo_func(x, H, f, x_middle)

    # numerical (time, x)
    v_num = v.mean(dim="y").values

    # compute error over time
    errors = [
        l2_error_func(v_num[t, :], v_theo)
        for t in range(v_num.shape[0])
        ]

    time = v.time.values

    R = np.sqrt(g*H) / f
    errors = np.array(errors)
    mask = errors < 15   # since your errors are in \%

    axs[1, i].fill_between(
        time, 0, 200,
        where=mask,
        color="green",
        alpha=0.3,
        label="< 15% error"
    )
    legend = r"$\left\|v(x,t)-v_{\text{theo}}(x,\infty)\right\| / \left\|v_{\text{theo}}(x,\infty)\right\|$"

    axs[1, i].plot(time, errors, c=f"C{i}", label=legend)
    axs[1, i].set_xlabel("Time [h]")
    axs[1, i].set_ylim(0, 100)

    if i == 0:
        axs[1, i].set_ylabel("Meridional velocity L2 error [%]")
    if i == 3:
        axs[1, i].legend()

    axs[1, i].grid(True, linestyle="--", alpha=0.6)

# --- layout ---
plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.suptitle("Convergence Towards Theoretical Solution", fontsize=14)

plt.savefig("plots/convergence.png", dpi=300)
plt.savefig("geostrophic-adjustment-report/Figures/convergence.png", dpi=300)

plt.show()