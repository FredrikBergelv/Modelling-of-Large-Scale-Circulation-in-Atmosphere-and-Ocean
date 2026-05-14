
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from load import give_all_data, load_data

#Global data
g = 0.01*9.81
times = [0, 24, 100, 150, 200]   # hours to plot
H = 1000
f = 1e-4
c = np.sqrt(g*H)

# Load data
h, u, v = load_data("pacific")
Lx = float(h.x.max())
Nx = len(h.values)
dx = Lx/Nx
Ly = float(h.y.max())
Ly_min = float(h.y.min())
min_speed = min(float(u.min()), float(v.min()))
max_speed = max(float(u.max()), float(v.max()))
min_height = float(h.min())
max_height = float(h.max())

def theo_h(h0, x, y, t, c=c):    
    b = 2e-11             # Coriolis parameter [1/s]
    Lw = Lx / 10
    R = np.sqrt(c / b)
    y = y*1000 - Ly/2 #convert to m
    t = t*3600 # Convert to s

    # Wave center position (periodic)
    x0 = (0.5 * Lx + c * t) % Lx

    # Periodic distance
    dx = x - x0

    # Wrap periodic distance
    dx = (dx + Lx/2) % Lx - Lx/2

    # Kelvin-wave Gaussian
    Gt = h0 * np.exp(-(dx**2) / (Lw**2)) * np.exp(-y**2 / (2*R**2))

    return Gt

def theo_u(h0, x, y, t, c=c):    
    b = 2e-11             # Coriolis parameter [1/s]
    Lw = Lx / 10
    R = np.sqrt(c / b)
    y = y*1000 - Ly/2 #convert to m
    t = t*3600 # Convert to s

    # Wave center position (periodic)
    x0 = (0.5 * Lx + c * t) % Lx

    # Periodic distance
    dx = x - x0

    # Wrap periodic distance
    dx = (dx + Lx/2) % Lx - Lx/2

    # Kelvin-wave Gaussian
    Gt = np.sqrt(g/H) * h0*np.exp(-(dx**2) / (Lw**2)) * np.exp(-y**2 / (2*R**2))

    return Gt


# ===================== PLOTTING =====================

fig, axs = plt.subplots(2, len(times), figsize=(2.5*len(times), 5), sharex=True)

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
    ax = axs[0, i]
        
    # Title
    actual_time = float(ht_ro["time"].values)
    ax.set_title(f"t = {actual_time:.1f} h")
    
    # Theo
    h0 = max(hvals_ke)
    yh_theo = np.linspace(Ly_min/1000,Ly/1000,1000)
    h_theo = theo_h(h0, x_kelvin, yh_theo, t)
    ax.plot(yh_theo, h_theo, c=f"darkblue", linestyle="--", label=r"$\eta_\text{theo. modified}\left(x_\text{Kelvin},y,\right)$")
    ax.plot(yh, ht_ro, c=f"C1", label=r"$\eta\left(x_\text{Rossby},y,\right)$", alpha=0.6)
    ax.plot(yh, ht_ke, c=f"C0", label=r"$\eta\left(x_\text{Kelvin},y,\right)$")
    


    
    ax.set_ylim(min_height, max_height)
    if i == 0:
        ax.set_ylabel(r"Height [m]")
        ax.legend()
    else:
        ax.set_yticklabels([])
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.set_ylim(-max_height*0.05, max_height)
    ax.set_xlim(0, Ly/1000)
    
    # ---- velocity-section ----
    ax = axs[1, i]
        
    # Title
    u_theo = theo_u(h0, x_kelvin, yh_theo, t)
    ax.plot(yh_theo, u_theo, c=f"darkgreen", linestyle="--", label=r"$u_\text{theo. modified}\left(x_\text{Kelvin},y,\right)$")
    ax.plot(yu, ut_ke, c=f"C2", label=r"$u\left(x_\text{Kelvin},y,\right)$")
    ax.plot(yv, vt_ke, c=f"C3", label=r"$v\left(x_\text{Kelvin},y,\right)$")

    if i == 0:
        ax.set_ylabel(r"Velocity [m/s]")
        ax.legend()
    else:
        ax.set_yticklabels([])
        
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.set_ylim(-max_speed*0.05, max_speed)
    ax.set_xlim(0, Ly/1000)
     
    ax.set_xlabel(r"$y$ [km]")
  

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.suptitle(r"Meridional Cross Sections of Kelvin and Rossby Waves", fontsize=16)
plt.savefig("plots/eq_yaxis.png", dpi=300)
plt.savefig("kelvin-waves-report/Figures/eq_yaxis.png", dpi=300)

plt.show()

    
