import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from load import give_all_data

# Global data
g = 0.01 * 9.81
times = [1, 10, 24*1, 24*7, 24*14, 21*24]

# Load data
atlantic, atlantic_fine, baltic, baltic_fine = give_all_data(
    "atlantic",
    "atlantic_fine",
    "baltic",
    "baltic_fine"
)

cases = [
    [atlantic, 1000, "Atlantic"],
    [atlantic_fine, 1000, "Atlantic fine grid"],
    [baltic, 30, "Baltic"],
    [baltic_fine, 30, "Baltic fine grid"]
]

def theo(x, c, t, Lx, Ly):    
    f = 1e-4              # Coriolis parameter [1/s]
    h0 = 5                # amplitude [m]
    Lw = Lx / 10
    R = c / f

    x = x * 1000  # km -> m
    t = t * 3600  # hours -> seconds

    # Wave center position (periodic)
    x0 = (0.5 * Lx + c * t) % Lx

    # Periodic distance
    dx = x - x0
    dx = (dx + Lx/2) % Lx - Lx/2

    # Kelvin-wave Gaussian
    Gt = h0 * np.exp(-(dx**2) / (Lw**2)) * np.exp(-0 / R)

    return Gt


def phase_diff(xmax, xmax_theo, c_theo, time):
    x_diff = xmax_theo - xmax
    c_diff = x_diff / (time * 3600)
    return c_theo + c_diff


def rel_error(val1, val2):
    return 100 * np.abs(val1 - val2) / val2


# Store results for LaTeX table
results = []

# =========================================================
# PHASE SPEED LOOP
# =========================================================

for dataset, H, name in cases:
    h_da, u_da, v_da = dataset

    c = np.sqrt(g * H)
    Lx = float(h_da.x.max())
    Ly = float(h_da.y.max())

    for t in times:

        # Select time and y=0
        hnow = h_da.sel(time=t, method="nearest")
        ht = hnow.sel(y=0, method="nearest")

        # Convert to numpy
        h_values = ht.values
        x_values = ht.x.values

        # Numerical peak
        idx = np.argmax(h_values)
        x_peak = x_values[idx]

        # Theoretical peak
        x_theo_peak = (0.5 * Lx + c * t * 3600) % (Lx)

        # Phase speed
        c_num = phase_diff(x_peak, x_theo_peak, c, t)

        # Error
        err = rel_error(c_num, c)

        # Store result
        results.append([name, t, c_num, c, err])


# =========================================================
# LaTeX TABLE OUTPUT
# =========================================================

print("\n% ================= LaTeX TABLE =================\n")

print(r"\begin{table}[h]")
print(r"\centering")
print(r"\begin{tabular}{lrrrr}")
print(r"\hline")
print(r"Case & Time (h) & $c_{num}$ & $c_{theo}$ & Rel. error (\%) \\")
print(r"\hline")

prev_name = None

for name, t, c_num, c_theo, err in results:

    # Add horizontal line between case groups
    if prev_name is not None and name != prev_name:
        print(r"\hline")

    # Only print case name once per group
    name_cell = name if name != prev_name else ""

    print(f"{name_cell} & 0 $\\rightarrow$ {t} & {c_num:.4f} & {c_theo:.4f} & {err:.2f} \\\\")

    prev_name = name

print(r"\hline")
print(r"\end{tabular}")
print(r"\caption{Kelvin wave phase speed: numerical vs theoretical comparison.}")
print(r"\end{table}")