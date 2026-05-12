
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from load import give_all_data, load_data

#Global data
g = 0.01*9.81
times = [0, 24, 100, 150, 200]   # hours to plot
H = 1000


# Load data
h, u, v = load_data("pacific")
Lx = float(h.x.max())
Nx = len(h.values)
dx = Lx/Nx
Ly = float(h.y.max())
min_speed = min(float(u.min()), float(v.min()))
max_speed = max(float(u.max()), float(v.max()))
min_height = float(h.min())
max_height = float(h.max())

def theo(x, c, t, Lx, Ly):    
    return 
    


# ===================== PLOTTING =====================

fig, axs = plt.subplots(1, len(times), figsize=(2.5*len(times), 3), sharey=True, sharex=True)

# Loop

for i, t in enumerate(times):
    
    # Select time and space
    hnow = h.sel(time=t, method="nearest")
    unow = u.sel(time=t, method="nearest")
    vnow = v.sel(time=t, method="nearest")
    
    hx = hnow.sel(y=Ly/2, method="nearest")

    x = h.x.values
    hx_vals = hx.values

    x0 = Lx / 2

    # Kelvin wave go eastward

    east_mask = x > x0

    hx_east = hx_vals[east_mask]
    x_east = x[east_mask]

    idx_kelvin = np.argmax(hx_east)

    x_kelvin = x_east[idx_kelvin]

    # Rossby wave are westward
    west_mask = x < x0

    hx_west = hx_vals[west_mask]
    x_west = x[west_mask]

    idx_rossby = np.argmax(hx_west)

    x_rossby = x_west[idx_rossby]

    # Extract meridional cross sections
    ht_ke = hnow.sel(x=x_kelvin, method="nearest")
    ht_ro = hnow.sel(x=x_rossby, method="nearest")
    
    vt_ke = vnow.sel(x=x_kelvin, method="nearest")
    vt_ro = vnow.sel(x=x_rossby, method="nearest")
    
    ut_ke = unow.sel(x=x_kelvin, method="nearest")
    ut_ro = unow.sel(x=x_rossby, method="nearest")
    
    # Convert to numpy arrays
    hvals_ro = ht_ro.values
    uvals_ro = ut_ro.values
    vvals_ro = vt_ro.values
    
    hvals_ke = ht_ke.values
    uvals_ke = ut_ke.values
    vvals_ke = vt_ke.values

    # Coordinates (convert to km)
    yh = h.y.values / 1000
    yu = u.y.values / 1000
    yv = v.y.values / 1000
    
    # ---- h-section ----
    ax = axs[i]
        
    # Title
    actual_time = float(ht_ro["time"].values)
    ax.set_title(f"t = {actual_time:.1f} h")
    
        
    ax.plot(yh, ht_ro, c=f"C0", label=r"$\eta\left(x_\text{west max},y,\right)$")
    ax.plot(yh, ht_ke, c=f"C1", label=r"$\eta\left(x_\text{east max},y,\right)$")

    
    ax.set_ylim(min_height, max_height)
    if i == 0:
        ax.set_ylabel(r"Height [m]")
        ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.set_ylim(-0.1, 5.1)
    ax.set_xlim(0, Ly/1000)
     
    ax.set_xlabel(r"$y$ [km]")
  

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.suptitle(r"Meridional Cross Sections of Kelvin and Rossby Waves", fontsize=16)
plt.savefig("plots/meridional_crosssection.png", dpi=300)
plt.savefig("kelvin-waves-report/Figures/meridional_crosssection.png", dpi=300)

plt.show()

    
