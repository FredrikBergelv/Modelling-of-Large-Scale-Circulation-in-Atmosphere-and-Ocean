import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Load data
h_ds = xr.open_dataset('data_smallf/h.nc')
u_ds = xr.open_dataset('data_smallf/u.nc')
v_ds = xr.open_dataset('data_smallf/v.nc')

h_smallf = h_ds["h"]
u_smallf = u_ds["u"]
v_smallf = v_ds["v"]

h_ds = xr.open_dataset('data_normal/h.nc')
u_ds = xr.open_dataset('data_normal/u.nc')
v_ds = xr.open_dataset('data_normal/v.nc')

h_largef = h_ds["h"]
u_largef = u_ds["u"]
v_largef = v_ds["v"]

h_ds = xr.open_dataset('data_smallH/h.nc')
u_ds = xr.open_dataset('data_smallH/u.nc')
v_ds = xr.open_dataset('data_smallH/v.nc')

h_smallH = h_ds["h"]
u_smallH = u_ds["u"]
v_smallH = v_ds["v"]

h_ds = xr.open_dataset('data_largeH/h.nc')
u_ds = xr.open_dataset('data_largeH/u.nc')
v_ds = xr.open_dataset('data_largeH/v.nc')

h_largeH = h_ds["h"]
u_largeH = u_ds["u"]
v_largeH = v_ds["v"]

### --- Theoretical solution for geostrophic adjustment --- ###
eta0 = 5
g = 9.81

def h_theo(x, f, H, h):
    x_middle = float(h.x.mean())
    c = np.sqrt(g*H)
    R = c/f
    
    x = np.asarray(x) 

    return np.where(
        x > x_middle,
        eta0 * (-1 + np.exp(-(x - x_middle) / R)),
        eta0 * (1 - np.exp((x - x_middle) / R))
        )

def v_theo(x, f, H, v):
    x_middle = float(v.x.mean())
    c = np.sqrt(g*H)
    R = c/f
    
    x = np.asarray(x)  
    return -g* eta0 / (R * f) * np.exp(-np.abs(x - x_middle) / R)


h_theo_smallf = h_theo(h_smallf.x.values, f=1e-5, H=4000, h=h_smallf)
h_theo_largef = h_theo(h_largef.x.values, f=1e-4, H=4000, h=h_largef)
h_theo_smallH = h_theo(h_smallH.x.values, f=1e-4, H=500, h=h_smallH)
h_theo_largeH = h_theo(h_largeH.x.values, f=1e-4, H=10000, h=h_largeH)

v_theo_smallf = v_theo(v_smallf.x.values, f=1e-5, H=4000, v=v_smallf)
v_theo_largef = v_theo(v_largef.x.values, f=1e-4, H=4000, v=v_largef)
v_theo_smallH = v_theo(v_smallH.x.values, f=1e-4, H=500, v=v_smallH)
v_theo_largeH = v_theo(v_largeH.x.values, f=1e-4, H=10000, v=v_largeH)

R_smallf = np.sqrt(g*4000) / 1e-5
R_largef = np.sqrt(g*4000) / 1e-4
R_smallH = np.sqrt(g*500) / 1e-4
R_largeH = np.sqrt(g*10000) / 1e-4

# ---- Plotting settings ----
time = 21*24   # hours to plot


#### ===== PLOTTING ===== ####

fig, axs = plt.subplots(2, 4, figsize=(3*4, 5), sharex=True)

titles = [r"$f=10^{-4}s^{-1}$ $H=500$ m" + f"\n (R={R_smallH/1000:.0f} km)",
          r"$f=10^{-4}s^{-1}$ $H=4000$ m" + f"\n (R={R_largef/1000:.0f} km)",
          r"$f=10^{-4}s^{-1}$ $H=10000$ m" + f"\n (R={R_largeH/1000:.0f} km)",
          r"$f=10^{-5}s^{-1}$ $H=4000$ m" + f"\n (R={R_smallf/1000:.0f} km)"]

cases = [(1e-4, 500), (1e-4, 4000), (1e-4, 10000), (1e-5, 4000)]

for i, (h, u, v) in enumerate([(h_smallH, u_smallH, v_smallH), (h_largef, u_largef, v_largef), (h_largeH, u_largeH, v_largeH), (h_smallf, u_smallf, v_smallf)]):
    
    h_t = h.sel(time=time, method="nearest")
    u_t = u.sel(time=time, method="nearest")
    v_t = v.sel(time=time, method="nearest")

    # Coordinates (convert to km)
    xh = h_t.x.values / 1000
    xu = u_t.x.values / 1000
    xv = v_t.x.values / 1000
    
    spounge_start = xh[:20] 
    spounge_end = xh[-20:]
    
    # Theoretical
    f, H = cases[i]
    h_theo_i = h_theo(h_t.x.values, f, H, h)
    v_theo_i = v_theo(v_t.x.values, f, H, v)
    
    # Plot height
    axs[0,i].set_title(titles[i])
    axs[0, i].plot(xh, h_theo_i, c="darkblue", linestyle="--", label=r"$\eta_{theo}(x)$")
    axs[0, i].plot(xh, h_t.mean(dim="y").values, c="C0", label=r"$\eta(x)$")
    
    axs[0, i].fill_between(spounge_start, -10, 10, color="gray", alpha=0.3)
    axs[0, i].fill_between(spounge_end, -10, 10, color="gray", alpha=0.3)
    
    if i == 0:
        axs[0, i].set_ylabel(r"Height [m]")
        axs[0, i].legend()
    else:
        axs[0, i].set_yticklabels([])
    axs[0, i].grid(True, linestyle='--', alpha=0.6)
    axs[0, i].set_ylim(-5.2, 5.2)
    
    # ---- u/v-section ----
    axs[1, i].plot(xu, xu*0, c="peru", linestyle="--", label=r"$u_{theo}(x)$")
    axs[1, i].plot(xv, v_theo_i, c="darkgreen", linestyle="--", label=r"$v_{theo}(x)$")

    axs[1, i].plot(xu, u_t.mean(dim="y").values, c="C1", label=r"$u(x)$")
    axs[1, i].plot(xv, v_t.mean(dim="y").values, c="C2", label=r"$v(x)$")
    
    axs[1, i].fill_between(spounge_start, -10, 10, color="gray", alpha=0.3)
    axs[1, i].fill_between(spounge_end, -10, 10, color="gray", alpha=0.3)
    
    if i == 0:
        axs[1, i].set_ylabel(r"Speed [m/s]")
        axs[1, i].legend()
    else:
        axs[1, i].set_yticklabels([])
    axs[1, i].grid(True, linestyle='--', alpha=0.6)
    axs[1, i].set_ylim(-0.3, 0.3)
    
    axs[1, i].set_xlabel("x [km]")


plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.suptitle(r"Surface Height and Velocities for Different Parameters, $t = 504$ h", fontsize=14)
plt.savefig("plots/graph_comparison.png", dpi=300)
plt.savefig("geostrophic-adjustment-report/Figures/graph_comparison.png", dpi=300)

plt.show()