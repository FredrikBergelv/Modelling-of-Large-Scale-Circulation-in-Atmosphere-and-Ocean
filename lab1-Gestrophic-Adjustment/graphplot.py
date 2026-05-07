import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Load data
h_ds = xr.open_dataset('data/h.nc')
u_ds = xr.open_dataset('data/u.nc')
v_ds = xr.open_dataset('data/v.nc')

h = h_ds["h"]
u = u_ds["u"]
v = v_ds["v"]

# ---- Plotting settings ----
times = [0, 2, 4, 4.5, 5000]   # hours to plot

#  ----- Theoretical solution for geostrophic adjustment -----

x_middle = float(h_ds.x.mean())
H = 4000
g = 9.81
f = 1e-4
eta0 = 5

def h_theo(x):
    c = np.sqrt(g*H)
    R = c/f
    
    x = np.asarray(x) 

    return np.where(
        x > x_middle,
        eta0 * (-1 + np.exp(-(x - x_middle) / R)),
        eta0 * (1 - np.exp((x - x_middle) / R))
        )

def v_theo(x):
    c = np.sqrt(g*H)
    R = c/f
    
    x = np.asarray(x)  
    return -g* eta0 / (R * f) * np.exp(-np.abs(x - x_middle) / R)


#### ===== PLOTTING ===== ####

fig, axs = plt.subplots(2, len(times), figsize=(2.5*len(times), 5), sharex=True)

for i, t in enumerate(times):
    
    
    h_t = h.sel(time=t, method="nearest")
    u_t = u.sel(time=t, method="nearest")
    v_t = v.sel(time=t, method="nearest")

    # Coordinates (convert to km)
    xh = h_t.x.values / 1000
    xu = u_t.x.values / 1000
    xv = v_t.x.values / 1000
    
    spounge_start = xh[:20] 
    spounge_end = xh[-20:]
    
    ax = axs[0, i]
    if i != 0:
        ax.set_yticklabels([])
        
    actual_time = float(h_t["time"].values)
    ax.set_title(f"t = {actual_time:.1f} h")
    
    # ---- h-section ----
    ax.plot(xh, h_theo(xh*1000), c="darkblue", linestyle="--", label=r"Theo. $\eta(x)$")
    ax.plot(xh, h_t.mean(dim="y").values, c="C0", label=r"$\eta(x)$")

    
    ax.set_ylim(-5.2, 5.2)
    if i == 0:
        ax.set_ylabel(r"Height [m]")
        ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.fill_between(spounge_start, -10, 10, color="gray", alpha=0.3)
    ax.fill_between(spounge_end, -10, 10, color="gray", alpha=0.3)

    ax = axs[1, i]
    if i != 0:
        ax.set_yticklabels([])
        
    # ---- u/v-section ----
    ax.plot(xv, v_theo(xv*1000), c="darkgreen", linestyle="--", label=r"Theo. $v(x)$")
    ax.plot(xu, xu*0, c="peru", linestyle="--", label=r"Theo. $u(x)$")

    ax.plot(xu, u_t.mean(dim="y").values, c="C1", label=r"$u(x)$")
    ax.plot(xv, v_t.mean(dim="y").values, c="C2", label=r"$v(x)$")
    
    ax.set_xlabel("x [km]")
    if i == 0:
        ax.set_ylabel(r"Speed [m/s]")
        ax.legend()
    ax.set_ylim(-0.3, 0.3)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.fill_between(spounge_start, -10, 10, color="gray", alpha=0.3)
    ax.fill_between(spounge_end, -10, 10, color="gray", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.suptitle(r"Time Evolution ofSurface Height and Velocities", fontsize=14)
#plt.savefig("plots/graph.png", dpi=300)
#plt.savefig("geostrophic-adjustment-report/Figures/graph.png", dpi=300)

plt.show()