import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# --- Load data ---
def load_hcase(path):
    ds = xr.open_dataset(path)
    return ds["h"]

def load_h0case(path):
    ds = xr.open_dataset(path)
    return ds["h0"]

def load_ucase(path):
    ds = xr.open_dataset(path)
    return ds["u"]

def load_vcase(path):
    ds = xr.open_dataset(path)
    return ds["v"]

def load_time(path):
    ds = xr.open_dataset(path)
    return ds["time"]

h0_smallf = load_h0case('data_smallf/h_init_cond.nc')
t_smallf = load_time('data_smallf/h.nc')
h_smallf, u_smallf, v_smallf = load_hcase('data_smallf/h.nc'), load_ucase('data_smallf/u.nc'), load_vcase('data_smallf/v.nc')

h0_largef = load_h0case('data_normal/h_init_cond.nc')
t_largef = load_time('data_normal/h.nc')
h_largef, u_largef, v_largef = load_hcase('data_normal/h.nc'), load_ucase('data_normal/u.nc'), load_vcase('data_normal/v.nc')

h0_smallH = load_h0case('data_smallH/h_init_cond.nc')
t_smallH = load_time('data_smallH/h.nc')
h_smallH, u_smallH, v_smallH = load_hcase('data_smallH/h.nc'), load_ucase('data_smallH/u.nc'), load_vcase('data_smallH/v.nc')

h0_largeH = load_h0case('data_largeH/h_init_cond.nc')
t_largeH = load_time('data_largeH/h.nc')
h_largeH, u_largeH, v_largeH = load_hcase('data_largeH/h.nc'), load_ucase('data_largeH/u.nc'), load_vcase('data_largeH/v.nc')

cases = [
    (h_smallH, u_smallH, v_smallH, h0_smallH, t_smallH, 500, 1e-4),
    (h_largef, u_largef, v_largef, h0_largef, t_largef, 4000, 1e-4),
    (h_largeH, u_largeH, v_largeH, h0_largeH, t_largeH, 10000, 1e-4),
    (h_smallf, u_smallf, v_smallf, h0_smallf, t_smallf, 4000, 1e-5),
]



# Constants
eta0 = 5
Nx = 300
Ny = 300
Lx = 6.E6
Ly = 45.E5
dx = Lx / Nx
dy = Ly / Ny
g = 9.81
rho = 1027

R_smallf = np.sqrt(g*4000) / 1e-5
R_largef = np.sqrt(g*4000) / 1e-4
R_smallH = np.sqrt(g*500) / 1e-4
R_largeH = np.sqrt(g*10000) / 1e-4

# ---- Relative error function ----
def relerr(values, ref):
    return 100 * (values - ref) / ref

fig, axs = plt.subplots(3, 4, figsize=(3*4, 8), sharex=True)

titles = [r"$f=10^{-4}s^{-1}$ $H=500$ m" + f"\n (R={R_smallH/1000:.0f} km)",
          r"$f=10^{-4}s^{-1}$ $H=4000$ m" + f"\n (R={R_largef/1000:.0f} km)",
          r"$f=10^{-4}s^{-1}$ $H=10000$ m" + f"\n (R={R_largeH/1000:.0f} km)",
          r"$f=10^{-5}s^{-1}$ $H=4000$ m" + f"\n (R={R_smallf/1000:.0f} km)"]

def theo_EP(R):
    return -rho*g*eta0**2*R*Ly*(3/2-2*np.exp(-Lx/(2*R)) + 1/2*np.exp(-Lx/R))

def theo_EK(R):
    return 0.5*rho*g*eta0**2*R*Ly*(1 - np.exp(-Lx/R))

for i, (h, u, v, h0, t, H,  f) in enumerate(cases):
    
    
    # ---- Energies ----
    # Interpolate data for C-grid to cell centers
    u2_c = 0.5 * (u.values[:, :, :-1]**2 + u.values[:, :, 1:]**2)
    v2_c = 0.5 * (v.values[:, :-1, :]**2 + v.values[:, 1:, :]**2)

    EK = 0.5 * rho * (H * (u2_c + v2_c)).sum(axis=(1, 2)) * dx * dy

    EP = 0.5 * g * rho * (h.values**2).sum(axis=(1, 2)) * dx * dy
    
    EP0 = 0.5 * g * rho * (h0.values**2).sum() * dx * dy
    print(f"EP0: {EP0:.2e} J")
    
    #V = h.sum(dim=["x", "y"]) * dx * dy
    #V0 = h0.sum(dim=["x", "y"]) * dx * dyrelerr(E_total, EP0), color="C3", label="Total loss")

    E_total = EP + EK
    
    # ---- Theoretical Energies ----
    R = np.sqrt(g*H) / f
    eta0 = 5
    
    EP0_theo = Lx*Ly*0.5*g*rho*eta0**2
    
    delEK_theo = theo_EK(R)
    delEP_theo = theo_EP(R)
    EK_theo = t*0 + delEK_theo
    
    EP_theo = t*0 + (EP0_theo + delEP_theo)
    
    delEtot_theo = delEK_theo + delEP_theo
    Etot_theo = EP0_theo + delEtot_theo + t*0
    
    if EP_theo[-1] < 0:
        EP_theo = t*0
        print("EP theo negative, setting to zero")
    if Etot_theo[-1] < 0:
        Etot_theo = t*0
        print("Etot theo negative, setting to zero")

    axs[0, i].plot(t, E_total*1e-18, color="C3", label="Total energy")
    axs[0, i].plot(t, Etot_theo*1e-18, color="darkred", linestyle="--", label=r"Theo. steady state")
    axs[0, i].plot(t, t*0 + EP0_theo*1e-18, color="pink", linestyle="-.", label="Theo. initial state")
    if i == 0:
        axs[0, i].legend()
        axs[0, i].set_ylabel("Energy [EJ]")
    elif i!= 0:
        axs[0, i].set_yticklabels([])
    axs[0, i].set_ylim(0, EP0*1e-18)
    axs[0, i].grid(True, linestyle='--', alpha=0.6)
    axs[0, i].set_title("Total Energy loss")
    axs[0, i].set_title(titles[i])
    
    axs[1, i].plot(t, EK*1e-18, color="C0", label="Kinetic")
    axs[1, i].plot(t, EP*1e-18, color="C1", label="Potential")
    axs[1, i].plot(t, EK_theo*1e-18, color="darkblue", linestyle="--", label="Theo. kinetic steady state")
    axs[1, i].plot(t, EP_theo*1e-18, color="peru", linestyle="--", label="Theo. potential steady state")
    
    axs[1, i].grid(True, linestyle='--', alpha=0.6)
    axs[1, i].set_ylim(0, EP0*1e-18)
    if i == 3:
        axs[1, i].legend(loc="upper left")
    if i == 0:
        axs[1, i].set_ylabel("Energy [EJ]")
    elif i != 0:
        axs[1, i].set_yticklabels([])
    
    delEK = 0 - EK
    delEP = EP0 - EP   
    
    axs[2, i].plot(t, delEK/delEP, color="C2", label=r"$\Delta EK / \Delta EP$")
    axs[2, i].plot(t, t*0  -1/3, color="darkgreen", linestyle="--", label=r"$-1/3$")
    axs[2, i].plot(t, t*0 + delEK_theo/delEP_theo, color="teal", linestyle="-.", label="Theo. steady state ratio")

    
    axs[2, i].set_xlabel("Time [h]")
    #axs[2, i].set_ylim(-1, 0)
    if i == 1:
        axs[2, i].legend()
    axs[2, i].grid(True, linestyle='--', alpha=0.6)
    if i != 0:
        axs[2, i].set_yticklabels([])
    if i == 0:
        axs[2, i].set_ylabel("Ratio [-]")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.suptitle(r"Energy Comparison and Evolution", fontsize=14)
plt.savefig("plots/energy_comparison.png", dpi=300)
plt.savefig("geostrophic-adjustment-report/Figures/energy_comparison.png", dpi=300)

plt.show()    
    

