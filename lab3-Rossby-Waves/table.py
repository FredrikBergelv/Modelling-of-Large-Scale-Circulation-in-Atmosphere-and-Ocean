import numpy as np

beta = 2e-11
f = 1e-4
g = 0.0981
H0 = 4000
Ly = 7e6

R = np.sqrt(g * H0) / f
my_points = [0.5, 1, 1.73, 3, 5, 7]

def L_w(kR):
    return 1.3 * (np.pi / kR) * (1 / np.sqrt(5)) * R

# --- Build LaTeX table ---
latex = []
latex.append("\\begin{table}[h]")
latex.append("\\centering")
latex.append("\\begin{tabular}{c c c c}")
latex.append("\\hline")
latex.append("$kR$ & $L_w$ (km) & \\% of domain \\\\")
latex.append("\\hline")

for p in my_points:
    lw = L_w(p)
    prec = 100 * lw / Ly
    latex.append(f"{p:.2f} & {lw/1000:.2f} & {prec:.4f} \\\\")

latex.append("\\hline")
latex.append("\\end{tabular}")
latex.append("\\end{table}")

print("\n".join(latex))