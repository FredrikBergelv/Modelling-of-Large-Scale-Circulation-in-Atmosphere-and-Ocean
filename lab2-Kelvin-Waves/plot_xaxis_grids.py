
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from load import load_h_data

#Global data
g = 0.01*9.81
times = [21*24]


### Data
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


cases = [[ha_200, 1000, r"200$\times$200"],
         [ha_450, 1000, r"450$\times$450"],
         [ha_600, 1000, r"600$\times$600"],
         [ha_750, 1000, r"750$\times$750"],
         [ha_900, 1000, r"900$\times$900"],
         [ha_1050, 1000, r"1050$\times$1050"],
         [ha_1200, 1000, r"1200$\times$1200"],
         [hb_200, 30, r"200$\times$200"],
         [hb_450, 30, r"450$\times$450"],
         [hb_600, 30, r"600$\times$600"],
         [hb_750, 30, r"750$\times$750"],
         [hb_900, 30, r"900$\times$900"],
         [hb_1050, 30, r"1050$\times$1050"],
         [hb_1200, 30, r"1200$\times$1200"],
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

fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharey=True)
ic = -1

# Loop
for j, (dataset,H,name) in enumerate(cases):
    h = dataset
    
    # Local data
    c = np.sqrt(g*H)
    Lx = float(h.x.max())
    Ly = float(h.y.max())
    min_height = float(h.min())
    max_height = float(h.max())
    
    
    for i, t in enumerate(times):
        if j<len(cases)/2:
            ax = axs[0]
            name_title = "Atlantic"
        else:
            ax = axs[1]
            name_title = "Baltic"
    
        # Select time and space
        hnow = h.sel(time=t, method="nearest")
 
        ht = hnow.sel(y=0, method="nearest")
        
        # Convert to numpy arrays
        hvals = ht.values

        # Coordinates (convert to km)
        xh = h.x.values / 1000

        # COLOR
        if j == len(cases)/2:
            ic = -1
        
        ic += 1
        
        # ---- h-section ----
            
        # Title
        actual_time = float(ht["time"].values)
        if j==0:
            ax.set_title(f"t = {actual_time:.1f} h")
            
        #x_pos = ((c * actual_time * 60 * 60 + Lx/2) % Lx) / 1000
        #ax.axvline(x=x_pos, color="red", label=r"$x=ct$")
        
        h_theo= theo(xh, c, t, Lx, Ly)
        
        if j==0 or j==int(len(cases)/2):
            ax.plot(xh, h_theo, c=f"black", label=fr"$\eta_{{\text{{theo.}}}}(x,0)$", linestyle="--", alpha=0.8)
            
        ax.plot(xh, ht, c=f"C{ic}", label=name)
        

        ax.set_ylim(min_height, max_height)
        if i == 0:
            ax.set_ylabel(r"Height [m]")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        ax.set_ylim(-0.1, 5.1)
        ax.set_xlim(0, Lx/1000)
     
        if name_title == "Baltic":
            ax.legend()
            
        ax.set_xlabel(r"$x$ [km]")
  

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.suptitle(r"Time Evolution of Surface Height at $y=0$", fontsize=16*(9/7)*(6/8))
plt.savefig("plots/xaxis_grids.png", dpi=300)
plt.savefig("kelvin-waves-report/Figures/xaxis_grids.png", dpi=300)

plt.show()

    
