import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cmocean

D = 3000
dx = 7e6 / 300


def plot_streamfunction(folder, times):

    # make sure times is always a list
    if not isinstance(times, list):
        times = [times]

    print("\n", folder)
    h_ds = xr.open_dataset(f"data/{folder}/h.nc")
    u_ds = xr.open_dataset(f"data/{folder}/u.nc")
    v_ds = xr.open_dataset(f"data/{folder}/v.nc")

    # --- coordinates ---
    x = h_ds["x"].values
    y = h_ds["y"].values

    # store all psi fields
    psi_list = []
    actual_times = []

    # --- compute all streamfunctions first ---
    for time in times:

        # correct time selection
        h = h_ds["h"].sel(time=time, method="nearest")
        u = u_ds["u"].sel(time=time, method="nearest")
        v = v_ds["v"].sel(time=time, method="nearest")

        actual_time = h["time"].values
        actual_times.append(actual_time)

        print("Actual time:", actual_time)

        # --- STREAMFUNCTION ---
        psi = -np.cumsum((v.values * D * dx)[:, ::-1], axis=1)[:, ::-1]

        # --- ALIGN GRID ---
        if psi.shape[0] != len(y):
            y_plot = np.linspace(y[0], y[-1], psi.shape[0])
        else:
            y_plot = y

        psi_list.append((psi / (10**6), y_plot))

    # --- common color scale ---
    all_psi = np.concatenate([p[0].ravel() for p in psi_list])

    vmin = np.min(all_psi)
    vmax = np.max(all_psi)

    # --- figure and subplots ---
    fig, axes = plt.subplots(1, len(times), figsize=(3 * len(times), 3), squeeze=False, sharey=True, sharex=True)

    axes = axes[0]

    # --- plotting ---
    for i, ax in enumerate(axes):

        psi, y_plot = psi_list[i]
        actual_time = actual_times[i]

        cf = ax.contourf(x / 1000, y_plot / 1000, psi, cmap='cmo.haline', levels=30, vmin=vmin, vmax=vmax)

        ax.contour(x / 1000, y_plot / 1000, psi, colors="k", linewidths=0.5)

        ax.set_xlabel("Latitude, $x$ [km]")

        if i == 0:
            ax.set_ylabel("Longitude, $y$ [km]")

        # convert to days + hours
        days = actual_time // 24
        hours = actual_time % 24

        ax.set_title(f"t={days:.0f}d {hours:.0f}h")

    # --- single colorbar on last subplot ---
    cbar = fig.colorbar(cf, ax=axes, location="right", shrink=1.0, pad=0.01)
    
    plt.suptitle(f"Streamfunction for {folder.replace("_", " ")}", size=16)

    cbar.set_label(r"Streamfunction, $\Psi$ [Sv]")

    fig.set_constrained_layout(True)
    save_name = folder + "_streamfunction"

    plt.savefig(f"plots/{save_name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"ocean-gyres-stommel-report/Figures/{save_name}.png", dpi=300, bbox_inches="tight")
    print(f"Saved as {save_name}")
    plt.show()

times_long = [2*24, 7*24, 14*24]
times_short = [2*24, 7*24, 14*24]


plot_streamfunction("no_drag_f_plane", times_long)

plot_streamfunction("no_drag_beta_plane", times_short)

plot_streamfunction("no_drag_normal_u10", times_short)

plot_streamfunction("no_drag_low_u10", times_short)

plot_streamfunction("no_drag_high_u10", times_short)

plot_streamfunction("no_drag_beta_plane_large_domain", times_short)



