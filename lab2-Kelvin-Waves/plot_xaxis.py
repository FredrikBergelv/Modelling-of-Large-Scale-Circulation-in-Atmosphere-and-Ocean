
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from load import give_all_data

#Global data
g = 0.01*9.81
times = [0, 12, 24, 24*3, 21*24]


# Load data
atlantic, atlantic_fine, baltic, baltic_fine = give_all_data(
    "atlantic",
    "atlantic_fine",
    "baltic",
    "baltic_fine"
)

h_a,  u_a,  v_a  = atlantic
h_af, u_af, v_af = atlantic_fine
h_b,  u_b,  v_b  = baltic
h_bf, u_bf, v_bf = baltic_fine

cases = [[atlantic, 1000, "Atlantic"],
         [atlantic_fine, 1000, "Atlantic fine grid"],
         [baltic, 30, "Baltic"],
         [baltic_fine, 30, "Baltic fine grid"]
         ]

def theo(x, c, t, Lx, Ly):    
    Lx = Lx
    Ly = Ly
    f = 1e-4              # Coriolis parameter [1/s]
    h0 = 5                # amplitude [m]
    Lw = Lx / 10
    R = c / f
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
    Gt = h0 * np.exp(-(dx**2) / (Lw**2)) * np.exp(-y / R)

    return Gt
    


# ===================== PLOTTING =====================
linestyles = ["--", ":", "-."]
ypos = [0.01, 0.05]   # y positions to plot

fig, axs = plt.subplots(2, len(times), figsize=(2.5*len(times), 5), sharey=True)

# Loop
for j, (dataset,H,name) in enumerate(cases):
    h, u, v = dataset
    
    # Local data
    c = np.sqrt(g*H)
    Lx = float(h.x.max())
    Ly = float(h.y.max())
    min_speed = min(float(u.min()), float(v.min()))
    max_speed = max(float(u.max()), float(v.max()))
    min_height = float(h.min())
    max_height = float(h.max())
    
    
    for i, t in enumerate(times):
    
        # Select time and space
        hnow = h.sel(time=t, method="nearest")
        unow = u.sel(time=t, method="nearest")
        vnow = v.sel(time=t, method="nearest")
        
        ht = hnow.sel(y=0, method="nearest")
        ut = unow.sel(y=0, method="nearest")
        vt = vnow.sel(y=0, method="nearest")
        
        # Convert to numpy arrays
        hvals = ht.values
        uvals = ut.values
        vvals = vt.values

        # Coordinates (convert to km)
        xh = h.x.values / 1000
        xu = u.x.values / 1000
        xv = v.x.values / 1000

        
        # ---- h-section ----
        if j==0 or j==1:
            ax = axs[0, i]
        if j==3 or j==2:
            ax = axs[1, i]
            
        if i != 0:
            ax.set_yticklabels([])
            
        # Title
        actual_time = float(ht["time"].values)
        if j==0:
            ax.set_title(f"t = {actual_time:.1f} h")
            
        #x_pos = ((c * actual_time * 60 * 60 + Lx/2) % Lx) / 1000
        #ax.axvline(x=x_pos, color="red", label=r"$x=ct$")
        
        h_theo= theo(xh, c, t, Lx, Ly)
        
        if j==0 or j==2:
            ax.plot(xh, h_theo, c=f"C4", label=fr"$\eta_{{\text{{theo.}}}}(x,0)$", linestyle="--", alpha=0.8)
            
        ax.plot(xh, ht, c=f"C{j}", label=fr"$\eta_{{\text{{{name}}}}}(x,0)$")
        
        """
        for k, frac in enumerate(ypos):
            y1 = Ly*frac
            ht1 = hnow.sel(y=y1, method="nearest")
            ut1 = unow.sel(y=y1, method="nearest")
            vt1 = vnow.sel(y=y1, method="nearest")
        
            ax.plot(xh, ht1, c=f"C{j}", linestyle=linestyles[k], label=fr"$\eta(x, y={frac}L_y)$", alpha=0.3)
        """
        ax.set_ylim(min_height, max_height)
        if i == 0:
            ax.set_ylabel(r"Height [m]")
            ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        ax.set_ylim(-0.1, 5.1)
        ax.set_xlim(0, Lx/1000)
     
        ax.set_xlabel(r"$x$ [km]")
  

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.suptitle(r"Time Evolution of Surface Height at $y=0$", fontsize=16)
plt.savefig("plots/xaxis.png", dpi=300)
plt.savefig("kelvin-waves-report/Figures/xaxis.png", dpi=300)

plt.show()

    
