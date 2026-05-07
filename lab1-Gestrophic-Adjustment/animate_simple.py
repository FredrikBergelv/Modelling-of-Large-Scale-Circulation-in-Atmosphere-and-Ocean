
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Open the NetCDF files
h_ds = xr.open_dataset('data/h.nc')
u_ds = xr.open_dataset('data/u.nc')
v_ds = xr.open_dataset('data/v.nc')
h_init_cond_ds = xr.open_dataset('data/h_init_cond.nc')

fig, ax = plt.subplots()

# Initial frame
h0 = h_ds["h"].isel(time=0)

# Create image ONCE
im = h0.plot(ax=ax, cmap="ocean", add_colorbar=True)

def update(frame):
    h = h_ds["h"].isel(time=frame)

    # Update the data instead of re-plotting
    im.set_array(h.values.ravel())

    ax.set_title(f"t = {h_ds.time[frame]:.2f} h")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    
    return [im]

ani = animation.FuncAnimation(fig, update, frames=100)
ani.save("plots/animation.gif", writer="pillow", fps=10)
plt.show()
