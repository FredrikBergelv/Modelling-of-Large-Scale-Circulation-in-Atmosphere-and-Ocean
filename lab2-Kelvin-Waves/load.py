import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def load_data(name="", ds=False):

    folder = "data" if name == "" else f"data_{name}"

    h_ds = xr.open_dataset(f"{folder}/h.nc")
    u_ds = xr.open_dataset(f"{folder}/u.nc")
    v_ds = xr.open_dataset(f"{folder}/v.nc")

    h = h_ds["h"]
    u = u_ds["u"]
    v = v_ds["v"]

    if not ds:
        return h, u, v
    elif ds:
        return h_ds, u_ds, v_ds
    
def load_h_data(name="", ds=False):

    folder = "data" if name == "" else f"data_{name}"

    h_ds = xr.open_dataset(f"{folder}/h.nc")

    h = h_ds["h"]

    if not ds:
        return h
    elif ds:
        return h_ds
    
def give_all_data(*names):

    datasets = []

    for name in names:
        datasets.append(load_data(name=name))

    return datasets