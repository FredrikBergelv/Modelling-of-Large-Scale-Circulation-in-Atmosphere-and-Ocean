
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from load import give_all_data

#Global data
g = 0.01*9.81
times = [2, 10, 24*1, 24*7, 24*14, 21*24]


# Load data
atlantic, atlantic_fine, baltic, baltic_fine = give_all_data(
    "atlantic",
    "atlantic_fine",
    "baltic",
    "baltic_fine"
)

h_a,  u_a,  v_a  = atlantic
h_af, u_af, v_af = atlantic_fine
h_b,  u_b,  v_b  = baltic
h_bf, u_bf, v_bf = baltic_fine

cases = [[atlantic, 1000, "Atlantic"],
         [atlantic_fine, 1000, "Atlantic fine grid"],
         [baltic, 30, "Baltic"],
         [baltic_fine, 30, "Baltic fine grid"]
         ]

# Functions
def phase(h, c, times=times):
    
    # Extract values
    Lx = float(h.x.max())
    c_theory = c

    # ---- Time spacing in saved data ----
    time_array = h.time.values
    dt = float(time_array[1] - time_array[0])   # hours

    # ---- Lists ----
    x_now = []
    x_prev = []

    t_now = []
    t_prev = []

    # ---- Loop over times ----
    for t in times:

        # Current snapshot
        h_now = h.sel(time=t, method="nearest")
        hs_now = h_now.sel(y=0, method="nearest")

        idx_now = np.argmax(hs_now.values)
        x_peak_now = hs_now.x.values[idx_now]

        # Previous snapshot
        h_prev = h.sel(time=t-dt, method="nearest")
        hs_prev = h_prev.sel(y=0, method="nearest")

        idx_prev = np.argmax(hs_prev.values)
        x_peak_prev = hs_prev.x.values[idx_prev]

        # Save
        x_now.append(x_peak_now)
        x_prev.append(x_peak_prev)

        t_now.append(float(hs_now.time.values) * 3600)
        t_prev.append(float(hs_prev.time.values) * 3600)

    # ---- Convert to arrays ----
    x_now = np.array(x_now)
    x_prev = np.array(x_prev)

    # ---- Handle periodic wrapping ----
    dx = x_now - x_prev

    # Correct periodic jumps
    dx[dx < -Lx/2] += Lx
    dx[dx >  Lx/2] -= Lx

    # ---- Numerical phase speed ----
    dt_seconds = np.array(t_now) - np.array(t_prev)

    c_num = dx / dt_seconds

    # ---- Relative error ----
    rel_error = np.abs(c_num - c_theory) / c_theory * 100
    
    return t_now, t_prev, c_num, c_theory, rel_error

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


# ---- Print LaTeX table ----
print(r"\begin{table}[H]")
print(r"\setlength{\tabcolsep}{3pt}")
print(r"\renewcommand{\arraystretch}{0.4}")
print(r"\centering")
print(r"\begin{tabular}{ccccc}")
print(r"\hline")
print(r"Scheme & Time [h] & Numerical $c$ [m/s] & Theoretical $c$ [m/s] & Relative error [\%] \\")

# Loop
for i, (dataset,H,name) in enumerate(cases):
    h, u, v = dataset
    
    # Local data
    c = np.sqrt(g*H)
    """Lx = float(h.x.max())
    Ly = float(h.y.max())
    min_speed = min(float(u.min()), float(v.min()))
    max_speed = max(float(u.max()), float(v.max()))
    min_height = float(h.min())
    max_height = float(h.max())"""

    t_now1, t_prev1, c_num1, c_theory1, rel_error1 = phase_beginning(h, c)
    t_now, t_prev, c_num, c_theory, rel_error = phase(h, c)

    t_now = np.concatenate([t_now, [t_now1[-1]]])
    t_prev = np.concatenate([t_prev, [t_prev1[-1]]])
    c_num = np.concatenate([c_num, [c_num1[-1]]])
    rel_error = np.concatenate([rel_error, [rel_error1[-1]]])
    
    for i, (tp, tn, c1, err) in enumerate(zip(t_prev, t_now, c_num, rel_error)):
        if i==0:
            print(r"\hline")
            print(f"{name} & ")
        else: 
            print(" & ")
        print(fr"{tp/3600:.0f} $\rightarrow$ {tn/3600:.0f} & "
              fr"{c1:.2f} & "
              fr"{c_theory:.2f} & "
              fr"{err:.2f} \\\\"
              )
        
print(r"\hline")
print(r"\end{tabular}")
print(r"\caption{Instantaneous and average Kelvin wave phase speed compared with theoretical phase speed. The average phase speed can be seen at the bottom of each scheme.}")
print(r"\label{tab:phase}")
print(r"\end{table}")
    
    
    
    
