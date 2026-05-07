import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Load data
h_ds = xr.open_dataset('data/h.nc')
u_ds = xr.open_dataset('data/u.nc')
v_ds = xr.open_dataset('data/v.nc')
h_init_cond_ds = xr.open_dataset('data/h_init_cond.nc')

h = h_ds["h"]
u = u_ds["u"]
v = v_ds["v"]
h_initial = h_init_cond_ds["h0"]

t = h_ds["time"]

# Constants
Nx = 300
Ny = 300
Lx = 12.e6
Ly = 90.e5
dx = Lx / Nx
dy = Ly / Ny
g = 9.81
f= 1e-4
H = 4000
rho = 1027


# ---- Energies ----

# Interpolate data for C-grid to cell centers
u2_c = 0.5 * (u.values[:, :, :-1]**2 + u.values[:, :, 1:]**2)
v2_c = 0.5 * (v.values[:, :-1, :]**2 + v.values[:, 1:, :]**2)

EK = 0.5 * rho * (H * (u2_c + v2_c)).sum(axis=(1, 2)) * dx * dy

EP = 0.5 * g * rho * (h**2).sum(dim=["x", "y"]) * dx * dy

EP0 = 0.5 * g * rho * (h_initial**2).sum(dim=["x", "y"]) * dx * dy

V = h.sum(dim=["x", "y"]) * dx * dy

V0 = h_initial.sum(dim=["x", "y"]) * dx * dy

E_total = EP + EK


# ---- Plot ----
fig, axs = plt.subplots(1, 3, figsize=(11, 4))

# Energy evolution
R = np.sqrt(g*H) / f
eta0 = 5

def theo_EP(R):
    return -rho*g*eta0**2*R*Ly*(-2*np.exp(-Lx/(2*R)) + 1/2*np.exp(-Lx/R) + 3/2 )

def theo_EK(R):
    return 0.5*rho*g*eta0**2*R*Ly*(1 - np.exp(-Lx/R))

delEP_theo = theo_EP(R)
delEK_theo = theo_EK(R)


EP0_theo = Lx*Ly*0.5*g*rho*eta0**2


EP_theo = EP0_theo + delEP_theo
EK_theo = delEK_theo

Etot_theo = EP_theo + EK_theo


print(f"EP0: {EP0:.2e} J")
print(f"EP0_theo: {EP0_theo:.2e} J")


print(f"loss_EP: {theo_EP(R):.2e} J")
print(f"gain_EK: {theo_EK(R):.2e} J")
print(f"EP: {EP[0]:.2e} J")
print(f"EK: {EK[0]:.2e} J")
print(f"EP: {EP[-1]:.2e} J")
print(f"EK: {EK[-1]:.2e} J")

axs[0].plot(t, E_total*1e-18, color="C3", label="Total energy")
axs[0].plot(t, t*0 + Etot_theo*1e-18, color="darkred", linestyle="--", label=r"Theo. steady state")
axs[0].plot(t, t*0 + EP0_theo*1e-18, color="pink", linestyle="-.", label="Theo. initial state")


print(Etot_theo*1e-18)
print(EK_theo*1e-18)
print(EP_theo*1e-18)

axs[0].legend()
axs[0].set_ylabel("Energy [EJ]")
axs[0].set_ylim(0, 1.1*EP0_theo*1e-18)
axs[0].grid(True, linestyle='--', alpha=0.6)
axs[0].set_title("Total Energy")
  



axs[1].plot(t, EK*1e-18, label="Kinetic", c="C0")
axs[1].plot(t, EP*1e-18, label="Potential", c="C1")
axs[1].plot(t, t*0 + EK_theo*1e-18, linestyle="--", label="Theo. kinetic steady state", c="darkblue")
axs[1].plot(t, t*0 + EP_theo*1e-18, linestyle="--", label="Theo. potential steady state", c="peru")
axs[1].set_ylim(0, 1.1*EP0_theo*1e-18)

axs[1].set_xlabel("Time [h]")
axs[1].set_ylabel("Energy [EJ]")
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.6)
axs[1].set_title("Energy evolution")


# Relative error
axs[2].plot(t, -EK/(EP0-EP), color="C2", label=r"$\Delta EK/ \Delta EP$")
axs[2].plot(t, t*0 + delEK_theo/(delEP_theo), color="teal", linestyle="-.", label=r"Theo. steady state ratio")
axs[2].plot(t, t*0  -1/3, color="darkgreen", linestyle="--", label=r"$-1/3$")
axs[2].set_xlabel("Time [h]")
axs[2].set_ylabel("Relative parts [-]")
axs[2].grid(True, linestyle='--', alpha=0.6)
axs[2].set_title("Energy partition")
axs[2].legend()


plt.suptitle("Energy Conservation, for Four Times as Large Grid \n" + r"$f=10^{-4}s^{-1}$ $H=4000$m, " + f"(R={R/1000:.0f} km)", size=14)

plt.tight_layout()

plt.savefig("plots/energy.png", dpi=300)
plt.savefig("geostrophic-adjustment-report/Figures/energy.png", dpi=300)

plt.show()