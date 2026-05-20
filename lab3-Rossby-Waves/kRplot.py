import numpy as np
from load import load_data
import matplotlib.pyplot as plt

beta = 2e-11
f = 1e-4
g = 0.0981
H0 = 4000
R = np.sqrt(g*H0)/f

data = [("kR=050", 0.5),
        ("kR=100", 1),
        ("kR=173", 1.73),
        ("kR=300", 3),
        ("kR=500", 5),
        ("kR=700", 7)]

phases = []
groups = []
phases_theo = []
groups_theo = []
my_points = [0.5, 1, 1.73, 3, 5, 7]

def omega_theo(kR):
    k = kR / R
    return - beta * k / (k**2 + 1/(R**2))

def phase_theo(kR):
    k = kR / R
    return - beta / (k**2 + 1/(R**2))

def group_theo(kR):
    k = kR / R
    return  beta * (k**2 - 1/(R**2)) / ((k**2 + 1/(R**2))**2)

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



for (data_name, kR) in data:

    # Load data (xarray)
    h, u, v = load_data(data_name)

    # Ensure coordinates exist
    x = h.x
    y = h.y
    t = h.time


    # Extract mid-latitude slice 
    hy = h.sel(y=h.y.mean(), method="nearest")

    # Obtain phase and group speeds
    cp = phase(hy)
    cg = group(hy)
    
    phases.append(cp)
    groups.append(cg)
    
    # Obtain theoretical phase and group speeds
    cp_theo = phase_theo(kR)
    cg_theo = group_theo(kR)
    
    phases_theo.append(cp_theo)
    groups_theo.append(cg_theo)


    print("\n======================")
    print(data_name)
    print("phase speed cp:", cp, " (error:", np.round(100*np.abs(cp-cp_theo)/cp_theo), "%)")
    print("group speed cg:", cg, " (error:", np.round(100*np.abs(cg-cg_theo)/cg_theo),"%)")



# ---- FIGURE ----
plt.figure(figsize=(8,4))

kRs = np.linspace(0,8,1000)
phase_theos = phase_theo(kRs)
group_theos = group_theo(kRs)

max_val = max(max(phase_theos),max(group_theos))
min_val = min(min(phase_theos),min(group_theos))

a = np.argmax(group_theos)

print(f"R = {R/1000:.0f} km")
print(f"kR_max = {kRs[a]:.2f}")

# Plotting theoretical solutions
plt.plot(kRs, phase_theos, c="C0",linestyle="--", label="Theo. phase speed")
plt.plot(kRs, group_theos, linestyle="--", label="Theo. group speed", c="C1")

plt.scatter(my_points, phase_theo(my_points), marker="x", alpha=0.8, c="C0", label="Theo. phase speed points")
plt.scatter(my_points, group_theo(my_points), marker="x", alpha=0.8, c="C1", label="Theo. group speed points")

# Plotting num solutions
plt.scatter(my_points, phases, label="Num. phase speed points", c="C0")
plt.scatter(my_points, groups, label="Num. group speed points", c="C1")

# Error arrows
for ph, gr, kr in zip(phases, groups, my_points):
    plt.annotate("",
            xy=(kr, ph),
            xytext=(kr, phase_theo(kr)),
            arrowprops=dict(
                arrowstyle="<->",
                color="gray",
                lw=1.5,
                alpha=0.8))

    plt.annotate("",
            xy=(kr, gr),
            xytext=(kr, group_theo(kr)),
            arrowprops=dict(
                arrowstyle="<->",
                color="gray",
                lw=1.5,
                alpha=0.8))

# Other stuff
plt.hlines(0, min(kRs), max(kRs), color="black")
plt.vlines(1, min_val, max_val, color="black", label=r"$R=k$", linestyle=":")


plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.suptitle(r"Phase and Group Speeds", size=16)
plt.tight_layout()

save_name = "dispersion"
plt.savefig("plots/dispersion.png", dpi=300)
plt.savefig("rossby-waves-report/Figures/dispersion.png", dpi=300)

plt.show()



data_out = np.column_stack([my_points, phases, groups, phases_theo, groups_theo])

np.savetxt(
    "speeds.txt",
    data_out,
    comments=""
)