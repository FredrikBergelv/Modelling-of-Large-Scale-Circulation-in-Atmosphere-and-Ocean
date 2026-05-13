
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from load import give_all_data

#Global data
g = 0.01*9.81
times_a = [0, 50, 150]
times_b = [0, 200, 250]



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

def theo(h0, y, c, t, Lx, Ly):    
    Lx = Lx
    Ly = Ly
    f = 1e-4              # Coriolis parameter [1/s]
    Lw = Lx / 10
    R = c  / f 
    y = y *1000 # convert to m
    x = 0
    t = t*3600       # convert to s

    # Wave center position (periodic)
    x0 = (0.5 * Lx + c * t) % Lx

    # Periodic distance
    dx = 0

    # Wrap periodic distance
    #dx = (dx + Lx/2) % Lx - Lx/2

    # Kelvin-wave Gaussian
    Gt = h0 * np.exp(-(dx**2) / (Lw**2)) * np.exp(-y / R)
    
    return Gt 

def gauss(h0, c, t, Lx, Ly):    
    Lx = Lx
    Ly = Ly
    f = 1e-4              # Coriolis parameter [1/s]
    Lw = Lx / 10
    R = c  / f 
    x = 0
    t = t*3600       # convert to s

    # Wave center position (periodic)
    x0 = (0.5 * Lx + c * t) % Lx

    # Periodic distance
    dx = 0

    # Wrap periodic distance
    #dx = (dx + Lx/2) % Lx - Lx/2

    # Kelvin-wave Gaussian
    Gt = h0 * np.exp(-(dx**2) / (Lw**2))
    
    return Gt 
    


# ===================== PLOTTING =====================
linestyles = ["--", ":", "-."]
ypos = [0.01, 0.05]   # y positions to plot

fig, axs = plt.subplots(2, len(times_a), figsize=(3*len(times_a), 5), sharey=True)

# Loop
for j, (dataset,H,name) in enumerate(cases):
    h, u, v = dataset
    
    # Local data
    c = np.sqrt(g*H)
    Lx = float(h.x.max())
    Nx = len(h.x.values)
    dx = float(Lx/Nx)
    Ly = float(h.y.max())
    min_speed = min(float(u.min()), float(v.min()))
    max_speed = max(float(u.max()), float(v.max()))
    min_height = float(h.min())
    max_height = float(h.max())
    
    
    for i, ö in enumerate(times_a):
        if j == 0 or j==1:
            t = times_a[i]
        if j == 2 or j==3:
            t = times_b[i]
            
        # Select time and space
        hnow = h.sel(time=t, method="nearest")
        unow = u.sel(time=t, method="nearest")
        vnow = v.sel(time=t, method="nearest")
        
        hx = hnow.sel(y=0, method="nearest")
        idx_max = np.argmax(hx.values)
        xmax = dx*idx_max
        
        ht = hnow.sel(x=xmax, method="nearest")
        ut = unow.sel(x=xmax, method="nearest")
        vt = vnow.sel(x=xmax, method="nearest")
        
        # Convert to numpy arrays
        hvals = ht.values
        uvals = ut.values
        vvals = vt.values
        
        h0 = hvals.max()

        # Coordinates (convert to km)
        yh = h.y.values / 1000
        yu = u.y.values / 1000
        yv = v.y.values / 1000

        
        # ---- h-section ----
        if j==0 or j==1:
            ax = axs[0, i]
        if j==3 or j==2:
            ax = axs[1, i]
            
        # Title
        actual_time = float(ht["time"].values)
        ax.set_title(f"t = {actual_time:.1f} h")
 
        # THEORY 
        if j==0 or j==2:
            yh_theo = np.linspace(0,Ly/1000,1000)
            h_theo = theo(h0, yh_theo, c, t, Lx, Ly)
            ax.plot(yh_theo, h_theo/h0, c=f"C4", label=fr"$\eta_{{\text{{theo. modified.}}}}\left(x_\text{{p.}},y\right)$", linestyle="--", alpha=0.8)
            
        ax.plot(yh, ht/h0, c=f"C{j}", label=fr"$\eta_{{\text{{{name}}}}}\left(x_\text{{p.}},y\right)$")
        
        ax.set_ylim(min_height, max_height)
        if i == 0:
            ax.set_ylabel(r"$\eta/h_0$ [-]")
            ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        ax.set_ylim(-0.1, 1.1)
        if i==0:
          ax.set_xlim(0, Ly/1000)  
        else:
            ax.set_xlim(0, 0.04*Ly/1000)

        ax.set_xlabel(r"$y$ [km]")
    
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.suptitle(r"Time Evolution of Surface Height at $x=x_\text{peak}$", fontsize=16)
plt.savefig("plots/yaxis.png", dpi=300)
plt.savefig("kelvin-waves-report/Figures/yaxis.png", dpi=300)

plt.show()

    
