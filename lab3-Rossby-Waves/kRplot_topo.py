import numpy as np
from load import load_data
import matplotlib.pyplot as plt

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

phases = []
groups = []
phases_theo = []
groups_theo = []
alphas = []


def omega_theo(alpha):
    return - (beta-alpha*f/H0) * k / (k**2 + 1/(R**2))

def phase_theo(kR, alpha):
    k = kR/R
    return - (beta-alpha*f/H0) / (k**2 + 1/(R**2))

def group_theo(kR, alpha):
    k = kR/R
    return  (beta-alpha*f/H0) * (k**2 - 1/(R**2)) / ((k**2 + 1/(R**2))**2)

def phase(hy):
    x_pos = []

    for tt in hy.time:
        slice_t = hy.sel(time=tt)

        # location of max amplitude
        x_max = slice_t.x.where(slice_t == slice_t.max(), drop=True).values

        x_pos.append(x_max[0])


    x_pos = np.array(x_pos)

    # remove NaNs
    valid = ~np.isnan(x_pos)
    t_valid = 3600*t.values[valid] # Note, convert to seconds
    x_valid = x_pos[valid]

    # linear fit → phase speed
    cp = np.polyfit(t_valid, x_valid, 1)[0]
    
    return cp
    

def group(hy):

    x = hy.x.values
    t = 3600*hy.time.values

    # ---- t=0: largest point ----
    h0 = hy.isel(time=0).values
    i_max0 = np.argmax(h0)
    x0 = x[i_max0]

    # ---- find smallest value over ALL time ----
    hmin = hy.values.min()

    # find where that minimum occurs (first occurrence)
    idx = np.argwhere(hy.values == hmin)[0]
    it_min = idx[0]
    ix_min = idx[1]

    x1 = x[ix_min]
    t1 = t[it_min]

    # ---- convert time ----
    try:
        t0 = 0
        t1 = (t1 - t[0]) / np.timedelta64(1, "s")
    except:
        t0 = 0

    # ---- dx/dt ----
    dx = x1 - x0
    dt = t1 - t0

    c = dx / dt

    return c



for (data_name, ratio) in data:

    # Load data (xarray)
    h, u, v = load_data(data_name)

    # Ensure coordinates exist
    x = h.x
    y = h.y
    t = h.time
    
    alpha = ratio*2*H0/np.max(h.x)

    # Extract mid-latitude slice 
    hy = h.sel(y=h.y.mean(), method="nearest")

    # Obtain phase and group speeds
    cp = phase(hy)
    cg = group(hy)
    
    if ratio==0.7:
        cp=0
        cg=0
    
    phases.append(cp)
    groups.append(cg)
    
    # Obtain theoretical phase and group speeds
    cp_theo = phase_theo(kR, alpha)
    cg_theo = group_theo(kR, alpha)
    
    phases_theo.append(cp_theo)
    groups_theo.append(cg_theo)
    
    # Store alphas
    alphas.append(float(alpha))
    


    print("\n======================")
    print(data_name)
    print("phase speed cp:", cp, " (error:", np.round(float(100*np.abs(cp-cp_theo)/cp_theo)), "%)")
    print("group speed cg:", cg, " (error:", np.round(float(100*np.abs(cg-cg_theo)/cg_theo)),"%)")



# ---- FIGURE WITH 3 SUBFIGURES ----
fig, axes = plt.subplots(1, 3, figsize=(8/0.6, 4), sharey=True)

kRs = np.linspace(0, 8, 1000)

# always include i=1
base_index = 1

# second member of each panel
panel_indices = [0, 3, 5]

for ax, extra_i in zip(axes, panel_indices):

    # plot i=1 and chosen extra case
    for i in [base_index, extra_i]:

        ph = phases[i]
        gr = groups[i]
        al = alphas[i]

        # ---- theoretical curves ----
        phase_theos = phase_theo(kRs, al)
        group_theos = group_theo(kRs, al)

        ax.plot(
            kRs,
            phase_theos,
            linestyle="--",
            c=f"C{i}",
            label=fr"Theo. phase "
        )

        ax.plot(
            kRs,
            group_theos,
            linestyle="-",
            c=f"C{i}",
            label=fr"Theo. group "
        )

        # ---- theoretical points ----
        ax.scatter(
            kR,
            phase_theo(kR, al),
            marker="v",
            alpha=0.8,
            c=f"C{i}"
        )

        ax.scatter(
            kR,
            group_theo(kR, al),
            marker="^",
            alpha=0.8,
            c=f"C{i}"
        )

        # ---- numerical points ----
        ax.scatter(
            kR,
            ph,
            c=f"C{i}",
            label=fr"Num. phase "
        )

        ax.scatter(
            kR,
            gr,
            marker="s",
            c=f"C{i}",
            label=fr"Num. group "
        )

        # ---- optional error arrows ----
        """
        ax.annotate(
            "",
            xy=(kR, ph),
            xytext=(kR, phase_theo(kR, al)),
            arrowprops=dict(
                arrowstyle="<->",
                color=f"C{i}",
                lw=1.5,
                alpha=0.4
            )
        )

        ax.annotate(
            "",
            xy=(kR, gr),
            xytext=(kR, group_theo(kR, al)),
            arrowprops=dict(
                arrowstyle="<->",
                color=f"C{i}",
                lw=1.5,
                alpha=0.4
            )
        )
        """

    # ---- formatting ----
    ax.axhline(0, color="black")
    ax.grid(True, linestyle="--", alpha=0.6)

    ax.set_xlim(0, 8)

    ax.set_title(
        fr"$\alpha={alphas[base_index]:.2e}$ vs $\alpha={alphas[extra_i]:.2e}$"
    )

    ax.set_xlabel(r"$kR$ [-]")
    ax.legend()

axes[0].set_ylabel("Velocity [m/s]")

fig.suptitle(r"Topographic Phase and Group Speeds", fontsize=16)

plt.tight_layout()

save_name = "dispersion_topo"

plt.savefig(f"plots/{save_name}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"rossby-waves-report/Figures/{save_name}.png", dpi=300, bbox_inches="tight")

plt.show()



data_out = np.column_stack([alphas, phases, groups, phases_theo, groups_theo])

np.savetxt(
    "speeds_topo.txt",
    data_out,
    comments=""
)