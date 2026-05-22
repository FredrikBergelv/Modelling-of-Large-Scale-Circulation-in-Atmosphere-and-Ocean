
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from load import give_all_data, load_data

#Global data
g = 0.01*9.81
times = [0, 24, 100, 150, 200]   # hours to plot
H = 1000
c=np.sqrt(g*H)

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

def theo(x, t):    
    b = 2e-11             # Coriolis parameter [1/s]
    h0 = 5                # amplitude [m]
    Lw = Lx / 10
    R = np.sqrt(c / b)
    y = 0
    x = x * 1000 #convert to m
    t = t*3600 # Convert to s

    # Wave center position (periodic)
    x0 = (0.5 * Lx + c * t) % Lx

    # Periodic distance
    dx = x - x0

    # Wrap periodic distance
    dx = (dx + Lx/2) % Lx - Lx/2

    # Kelvin-wave Gaussian
    Gt = h0 * np.exp(-(dx**2) / (Lw**2)) * np.exp(-y**2 / R)

    return Gt


# ===================== PLOTTING =====================
print(r"\begin{table}[h!]")
print(r"\centering")
print(r"\begin{tabular}{ccccc}")
print(r"\hline")
print(r"$t$ [h] & $x_{\max}$ [km] & $c_{\mathrm{num}}$ [m/s] & $c_{\mathrm{theo}}$ [m/s] & Error [\%] \\")
print(r"\hline")
x_old = Lx/2
t_old = 0

fig, axs = plt.subplots(1, len(times), figsize=(2.5*len(times), 3), sharey=True, sharex=True)

# Loop

for i, t in enumerate(times):
    
    # Select time and space
    hnow = h.sel(time=t, method="nearest")
    unow = u.sel(time=t, method="nearest")
    vnow = v.sel(time=t, method="nearest")
    
    
    # Extract meridional cross sections
    ht = hnow.sel(y=Ly/2, method="nearest")    
    vt = vnow.sel(y=Ly/2, method="nearest")
    ut = unow.sel(y=Ly/2, method="nearest")
    
    # Convert to numpy arrays
    hvals = ht.values
    uvals = ut.values
    vvals = vt.values

    # Coordinates (convert to km)
    yh = h.y.values / 1000
    yu = u.y.values / 1000
    yv = v.y.values / 1000
    
    # ---- h-section ----
    ax = axs[i]
        
    # Title
    actual_time = float(ht["time"].values)
    ax.set_title(f"t = {actual_time:.1f} h")
    
    h_theo = theo(yh, t)
    ax.plot(yh, ht, c=f"C0", label=r"$\eta\left(x,\frac{L_y}{2},\right)$")
    ax.plot(yh, h_theo, c=f"darkblue", linestyle="--", label=r"$\eta_\text{theo.}\left(x,\frac{L_y}{2},\right)$")
    
    ax.set_ylim(min_height, max_height)
    if i == 0:
        ax.set_ylabel(r"Height [m]")
        ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.set_ylim(-0.1, 5.1)
    ax.set_xlim(0, Ly/1000)
     
    ax.set_xlabel(r"$x$ [km]")
    
    
    
    # ===== PHASE SPEED ====
    idx = np.argmax(ht.values)
    x_peak = h.x.values[idx]
    x_diff= x_peak - x_old
    t_now = t
    t_diff = (t_now - t_old)*3600
    c_num = x_diff / t_diff

    # Avoid division by zero at t=0
    if t == 0:
        rel_err = np.nan
    else:
        rel_err = 100 * np.abs(c_num - c) / c

    print(
        f"{t_old:.0f}" + r" $\rightarrow$ " + f"{t_now:.0f} & "
        f"{x_peak/1000:.1f} & "
        f"{c_num:.2f} & "
        f"{c:.2f} & "
        f"{rel_err:.2f} \\\\"
    )
    x_old = x_peak
    t_old = t_now

x_diff = x_old - Lx/2
c_num = x_diff / (times[-1]*3600)
rel_err = 100 * np.abs(c_num - c) / c
print(f"0" + r" $\rightarrow$ " + f"{t_now:.0f} & "
        f"{x_peak/1000:.1f} & "
        f"{c_num:.2f} & "
        f"{c:.2f} & "
        f"{rel_err:.2f} \\\\"
        )
print(r"\hline")
print(r"\end{tabular}")
print(r"\caption{Kelvin-wave phase speed diagnostics.}")
print(r"\label{tab:kelvin_speed}")
print(r"\end{table}")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.suptitle(r"Zonal Cross Sections of Kelvin and Rossby Waves", fontsize=16)
plt.savefig("plots/eq_xaxis.png", dpi=300)
plt.savefig("kelvin-waves-report/Figures/eq_xaxis.png", dpi=300)

plt.show()

    
