import xarray as xr
import matplotlib.pyplot as plt

# Load data
h_ds = xr.open_dataset('data/h.nc')
u_ds = xr.open_dataset('data/u.nc')
v_ds = xr.open_dataset('data/v.nc')

h = h_ds["h"]
u = u_ds["u"]
v = v_ds["v"]

print("v min:", v.min().values, "v max:", v.max().values)

print("u min:", u.min().values, "u max:", u.max().values)

print("h min:", h.min().values, "h max:", h.max().values)