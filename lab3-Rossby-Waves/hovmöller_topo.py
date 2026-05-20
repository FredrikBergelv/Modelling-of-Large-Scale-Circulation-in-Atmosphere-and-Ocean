import numpy as np
import matplotlib.pyplot as plt
from load import load_data

# ---- load saved velocities ----
data_in = np.loadtxt("speeds_topo.txt")
alpha_vals = data_in[:, 0]
cp_vals = data_in[:, 1]
cg_vals = data_in[:, 2]
cp_theo_vals = data_in[:, 3]
cg_theo_vals = data_in[:, 4]

beta = 2e-11
f = 1e-4
g = 0.0981
H0 = 4000
R = np.sqrt(g*H0)/f
kR = 1.73
k = kR / R



data = [("depth_diff=-20", -0.2),
        ("depth_diff=0", 0),
        ("depth_diff=20", 0.2),
        ("depth_diff=40", 0.4),
        ("depth_diff=70", 0.7),
        ("depth_diff=90", 0.8)]

fig, axes = plt.subplots(int(len(data)/2), 2, figsize=(8, 10), sharey=True)

for i, (name, ratio) in enumerate(data):

    alpha = alpha_vals[i]
    
    h, u, v = load_data(name)

    hy = h.sel(y=h.y.mean(), method="nearest")
    hx = hy.transpose("time", "x")

    x = hx.x.values / 1000 # to km

    # time in seconds (robust)
    t = hx.time.values

    ax = axes.flat[i]

    im = ax.pcolormesh(
        x, t, hx.values,
        shading="auto",
        cmap="RdBu_r",
        vmin=-1, 
        vmax=5
    )


    # ---- COLORBAR FIX ----
    if (i %2) == 1:
        cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.01)
        cbar.set_label(r"$\eta$ [m]")

    ax.set_title(fr"$\alpha$ = {alpha:.2e}")

    # ---- get velocities in h ----
    cp = cp_vals[i]*3600
    cg = cg_vals[i]*3600
    cp_theo = cp_theo_vals[i]*3600
    cg_theo = cg_theo_vals[i]*3600

    # ---- reference starting point (center of domain) -

    x0 = x[len(x)//2]*1000

    t_line = np.linspace(t.min(), t.max(), 100)

    # numerical speeds
    x_group = x0 + cg * (t_line - t_line[0])
    x_phase = x0 + cp * (t_line - t_line[0])
    
    x_group = x_group / 1000
    x_phase = x_phase / 1000
    
    # theoretical speeds
    x_theo_group = x0 + cg_theo * (t_line - t_line[0])
    x_theo_phase = x0 + cp_theo * (t_line - t_line[0])
    
    x_theo_group = x_theo_group / 1000
    x_theo_phase = x_theo_phase / 1000

    ax.plot(
        x_group,
        t_line,
        color="green",
        linestyle="--",
        linewidth=2,
        label="group velocity")
    
    ax.plot(
        x_phase,
        t_line,
        color="green",
        linestyle=":",
        linewidth=2,
        label="phase velocity")
    
    ax.plot(
        x_theo_group,
        t_line,
        color="yellow",
        linestyle="--",
        linewidth=2,
        label="theo. group velocity")
    
    ax.plot(
        x_theo_phase,
        t_line,
        color="yellow",
        linestyle=":",
        linewidth=2,
        label="theo. phase velocity")

    ax.set_ylim(0, 729)

    scaling = x0/1000-min(x_phase) + 450
    ax.set_xlim(x0/1000 - scaling, x0/1000 + scaling)
   
    
    if (i %2) == 0:
        ax.set_ylabel(r"$t$ [h]")
               
    if i==(4):
        ax.legend(loc="lower right")

axes[-1, 0].set_xlabel(r"$x$ [km]")
axes[-1, 1].set_xlabel(r"$x$ [km]")
plt.suptitle("Hovmöller Diagram of Rossby Topographic Waves", size=16)

plt.tight_layout()


save_name = "hovmöller_topo"
plt.savefig(f"plots/{save_name}.png", dpi=300)
plt.savefig(f"rossby-waves-report/Figures/{save_name}.png", dpi=300)


plt.show()