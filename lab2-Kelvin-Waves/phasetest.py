
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from load import load_data

#Global data
g = 0.01*9.81
times = [2, 10, 24*1, 24*7, 24*14, 21*24]


# Load data
h,u,v = load_data()
H  = 30

def phase_beginning(h, c, times=times):

    Lx = float(h.x.max())
    c_theory = c

    # --------------------------------------------------------
    # collect peak positions
    # --------------------------------------------------------

    x_raw = []
    t_raw = []

    for t in [0] + times:

        h_now = h.sel(time=t, method="nearest").sel(y=0, method="nearest")

        idx = np.argmax(h_now.values)

        x_raw.append(float(h_now.x.values[idx]))
        t_raw.append(float(h_now.time.values) * 3600)

    x_raw = np.array(x_raw)
    t_raw = np.array(t_raw)

    # --------------------------------------------------------
    # BUILD CONTINUOUS UNWRAPPED TRAJECTORY
    # --------------------------------------------------------

    x_unwrapped = np.zeros_like(x_raw)
    x_unwrapped[0] = x_raw[0]

    offset = 0.0

    for i in range(1, len(x_raw)):

        dx = x_raw[i] - x_raw[i - 1]

        # detect wrap (THIS is your key idea)
        if dx < 0:
            offset += Lx

        x_unwrapped[i] = x_raw[i] + offset

    # --------------------------------------------------------
    # compute speed
    # --------------------------------------------------------

    dt = t_raw - t_raw[0]
    dx = x_unwrapped - x_unwrapped[0]

    valid = dt != 0

    c_num = np.full_like(dt, np.nan, dtype=float)
    c_num[valid] = dx[valid] / dt[valid]

    rel_error = np.abs(c_num - c_theory) / c_theory * 100

    return t_raw, np.zeros_like(t_raw), c_num, c_theory, rel_error




# Loop
c = np.sqrt(g*H)

t_now, t_prev, c_num, c_theory, rel_error = phase_beginning(h, c)


    
for i, (tp, tn, c1, err) in enumerate(zip(t_prev, t_now, c_num, rel_error)):
        print(fr"{err:.2f}")
        

    
    
