import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from load import load_data


# --- Load data ---
h, u, v = load_data("pacific")

# --- Compute global velocity scale ---
u_i_all = 0.5 * (u.values[:, :, :-1] + u.values[:, :, 1:])
v_i_all = 0.5 * (v.values[:, :-1, :] + v.values[:, 1:, :])

mag_all = np.sqrt(u_i_all**2 + v_i_all**2)
mag_all = np.nan_to_num(mag_all)
global_max = mag_all.max()

# --- Helper: interpolate to cell centers ---
def compute_uv(u, v):
    u_i = 0.5 * (u.values[:, :-1] + u.values[:, 1:])
    v_i = 0.5 * (v.values[:-1, :] + v.values[1:, :])
    return np.nan_to_num(u_i), np.nan_to_num(v_i)

# --- FIGURE SETUP ---
fig, ax = plt.subplots(figsize=(6, 5))

# Initial frame
h0 = h.isel(time=0)
u0 = u.isel(time=0)
v0 = v.isel(time=0)

# Coordinates (convert to km)
x = h0.x.values / 1000
y = h0.y.values / 1000
X, Y = np.meshgrid(x, y)

# Initial velocity
u_i0, v_i0 = compute_uv(u0, v0)

# --- FIXED COLOR SCALE (CORRECTED) ---
vmin = float(h.values.min())
vmax = float(h.values.max())

norm = colors.Normalize(vmin=vmin, vmax=vmax)

# --- Plot scalar field ---
im = ax.imshow(
    h0.values,
    origin="lower",
    cmap="ocean",
    norm=norm,
    extent=[x.min(), x.max(), y.min(), y.max()]
)

# --- Plot quiver ONCE ---
step = 20
Q = ax.quiver(
    X[::step, ::step],
    Y[::step, ::step],
    u_i0[::step, ::step],
    v_i0[::step, ::step],
    color="white",
    scale=global_max * 3,
    alpha=0.4,
    width=0.004
)

# Labels
ax.set_xlabel(r"$x$ [km]")
ax.set_ylabel(r"$y$ [km]")
plt.suptitle(r"$\eta(x,y,t)$ at different times", size=14)

# Colorbar (stable)
cbar = plt.colorbar(
    im,
    ax=ax,
    shrink=0.8,
    extend='both',
    label=r"$h(x,y,t)$ [m]"
)

# --- UPDATE FUNCTION ---
def update(frame):
    h_t = h.isel(time=frame)
    u_t = u.isel(time=frame)
    v_t = v.isel(time=frame)

    # update scalar field (NO rescaling)
    im.set_data(h_t.values)

    # update velocity field
    u_i, v_i = compute_uv(u_t, v_t)
    Q.set_UVC(
        u_i[::step, ::step],
        v_i[::step, ::step]
    )

    ax.set_title(f"t = {float(h_t.time.values):.2f} h")

    return [im, Q]

# --- ANIMATION ---
ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(h.time),
    interval=100
)

# Save
ani.save("plots/animation_equator.gif", writer="pillow", fps=10)

print("Animation saved as 'plots/animation_equator.gif'")