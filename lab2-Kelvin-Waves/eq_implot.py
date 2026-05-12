import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from load import load_data

h, u, v = load_data("pacific")

# ---- Plotting settings ----
times = [0, 24, 100, 150, 200]   # hours to plot

Lx = float(h.x.max())
Ly = float(h.y.max())



# WE LOOK FOR THE GLOBAL MAXIMUM VELOCITY TO SCALE THE ARROWS PROPERLY
u_i_all = 0.5 * (u.values[:, :, :-1] + u.values[:, :, 1:])
v_i_all = 0.5 * (v.values[:, :-1, :] + v.values[:, 1:, :])

mag_all = np.sqrt(u_i_all**2 + v_i_all**2)
mag_all = np.nan_to_num(mag_all)
global_max = mag_all.max()
minval = float(h.min())
maxval = float(h.max())
value_scale = 0.5

def vel_arrows(ax, h, u, v, step=20,  global_max=True):
    # Interpolate to cell centers 
    u_i = 0.5 * (u.values[:, :-1] + u.values[:, 1:])
    v_i = 0.5 * (v.values[:-1, :] + v.values[1:, :])
    
    # Compute velocity magnitude for scaling
    magnitude = np.sqrt(0.5*(u.values[:, :-1]**2 + u.values[:, 1:]**2) + 0.5*(v.values[:-1, :]**2 + v.values[1:, :]**2))
    
    # fin the h-points (note unit is km for plotting)
    x = h.x.values
    y = h.y.values
    X, Y = np.meshgrid(x/1000, y/1000)
    
    ax.quiver(
        X[::step, ::step],
        Y[::step, ::step],
        u_i[::step, ::step],
        v_i[::step, ::step],
        color='peru',
        width=0.006,
        scale=global_max*2 if global_max is not None else None,
        )

# ---- FIGURE ----
fig, axs = plt.subplots(1, len(times)+1, figsize=(3*len(times), 4.1), gridspec_kw=dict(width_ratios=[1]*len(times) + [0.1]), sharey=True, sharex=True)


for i, t in enumerate(times):

    h_t = h.sel(time=t, method="nearest")
    u_t = u.sel(time=t, method="nearest")
    v_t = v.sel(time=t, method="nearest")

    im = axs[i].imshow(
        h_t,
        origin="lower",
        aspect="equal", 
        vmin=minval*value_scale, vmax=maxval*value_scale,
        cmap="ocean",
        extent=[0, Lx/1000, 0, Ly/1000]
        )
    
    # here we plot the arrows for the velocity field
    vel_arrows(axs[i], h_t, u_t, v_t, global_max=global_max)
    
    # Time formatting
    actual_time = float(h_t["time"].values)
    axs[i].set_title(f"t = {actual_time:.1f} h")
    axs[i].set_xlabel(r"$x$ [km]")
    
    if i == 0:
        axs[i].set_ylabel(r"$y$ [km]")


# colorbar
cbar =fig.colorbar(im, ax=axs[len(times)], pad=0, fraction=1, shrink=0.8,  extend="both", label=r"Height [m]")
cbar.set_ticks([0, 1, 2, 3, 4, 5])
axs[len(times)].axis("off")

plt.suptitle(r"$\eta(x,y,t)$ at Different Times", size=16)
plt.tight_layout()

plt.savefig("plots/equator.png", dpi=300)
plt.savefig("kelvin-waves-report/Figures/equator.png", dpi=300)

plt.show()