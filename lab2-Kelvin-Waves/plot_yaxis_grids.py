
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from load import give_all_data, load_h_data

times_a = [150]
times_b = [250]

## Data
ha_100 = load_h_data("atlantic_100")
ha_200 = load_h_data("atlantic_200")
ha_300 = load_h_data("atlantic_300")
ha_450 = load_h_data("atlantic_450")
ha_600 = load_h_data("atlantic_600")
ha_750 = load_h_data("atlantic_750")
ha_900 = load_h_data("atlantic_900")
ha_1050 = load_h_data("atlantic_1050")
ha_1200 = load_h_data("atlantic_1200")

hb_100 = load_h_data("baltic_100")
hb_200 = load_h_data("baltic_200")
hb_300 = load_h_data("baltic_300")
hb_450 = load_h_data("baltic_450")
hb_600 = load_h_data("baltic_600")
hb_750 = load_h_data("baltic_750")
hb_900 = load_h_data("baltic_900")
hb_1050 = load_h_data("baltic_1050")
hb_1200 = load_h_data("baltic_1200")


cases = [[ha_100, 1000, r"100$\times$100"],
         [ha_200, 1000, r"200$\times$200"],
         [ha_300, 1000, r"300$\times$300"],
         [ha_450, 1000, r"450$\times$450"],
         [ha_600, 1000, r"600$\times$600"],
         [ha_750, 1000, r"750$\times$750"],
         [ha_900, 1000, r"900$\times$900"],
         [ha_1050, 1000, r"1050$\times$1050"],
         [ha_1200, 1000, r"1200$\times$1200"],
         [hb_100, 30, r"100$\times$100"],
         [hb_200, 30, r"200$\times$200"],
         [hb_300, 30, r"300$\times$300"],
         [hb_450, 30, r"450$\times$450"],
         [hb_600, 30, r"600$\times$600"],
         [hb_750, 30, r"750$\times$750"],
         [hb_900, 30, r"900$\times$900"],
         [hb_1050, 30, r"1050$\times$1050"],
         [hb_1200, 30, r"1200$\times$1200"],
         ]

print(len(cases))

def theo(h0, y, c, t, Lx, Ly, Ly_min):  
    
    fix_factor =  np.sqrt(Ly/Ly_min)
    print(fix_factor)
    Lx = Lx
    Ly = Ly
    f = 1e-4              # Coriolis parameter [1/s]
    Lw = Lx / 10
    R = (c  / f ) * fix_factor
    y = y *1000 # convert to m
    x = 0
    t = t*3600       # convert to s
    
    # Kelvin-wave Gaussian
    decay = h0 * np.exp(-np.abs(y-Ly_min) / R)
    
    return decay 

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

fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharey=True, sharex=True)

g = 0.0981
# Loop
ic = -1
for j, (dataset,H,name) in enumerate(cases):
    h = dataset
    
    # Local data
    c = np.sqrt(g*H)
    Lx = float(h.x.max())
    Nx = len(h.x.values)
    dx = float(Lx/Nx)
    Ly = float(h.y.max())
    Ly_min = float(h.y.min())
    min_height = float(h.min())
    max_height = float(h.max())
        
    for i, ö in enumerate(times_a):
        if j<len(cases)/2:
            ax = axs[0]
            name_title = "Atlantic"
            t = times_a[i]
        else:
            ax = axs[1]
            name_title = "Baltic"
            t = times_b[i]
        
        if j == len(cases)/2:
            ic = -1
        
        ic += 1
                
        # Select time and space
        hnow = h.sel(time=t, method="nearest")
        
        hx = hnow.sel(y=0, method="nearest")
        idx_max = np.argmax(hx.values)
        xmax = dx*idx_max
        
        ht = hnow.sel(x=xmax, method="nearest")
        
        # Convert to numpy arrays
        hvals = ht.values
        
        h0 = hvals.max()

        # Coordinates (convert to km)
        yh = h.y.values / 1000

        
        # ---- h-section ----
            
        # Title
        actual_time = float(ht["time"].values)
        ax.set_title(f"{name_title} at t = {actual_time:.1f} h")
 
        # THEORY 
        if j==0 or j==int(len(cases)/2):
            yh_theo = np.linspace(Ly_min/1000,Lx/1000,1000)
            h_theo = theo(h0, yh_theo, c, t, Lx, Ly, Ly_min)
            ax.plot(yh_theo/Ly_min, h_theo/h0, c=f"black", label=fr"Theo. modified", linestyle="--", alpha=0.8)
            
        ax.plot(yh/Ly_min, ht/h0, c=f"C{ic}", label=name)
        
        ax.set_ylim(min_height, max_height)
        ax.set_ylabel(r"$\eta/h_0$ [-]")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        ax.set_xlim(0, 0.05)
        ax.set_ylim(0, 1.1)

        if name_title =="Baltic":
            ax.set_xlabel(r"$y/y_\text{min}$")
            ax.legend()

    
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.suptitle(r"Time Evolution of Surface Height at $x=x_\text{peak}$", fontsize=16*(9/7)*(6/8))
plt.savefig("plots/yaxis_grids.png", dpi=300)
plt.savefig("kelvin-waves-report/Figures/yaxis_grids.png", dpi=300)

plt.show()

    
