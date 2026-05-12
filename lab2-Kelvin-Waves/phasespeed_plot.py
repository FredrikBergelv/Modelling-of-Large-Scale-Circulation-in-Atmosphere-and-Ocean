import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([100, 300, 600, 900, 1200, 1500])
A = np.array([97.77, 77.56, 64.06, 54.90, 45.07, 40.00])  # Atlantic
B = np.array([87.15, 67.71, 48.28, 38.85, 32.13, 27.31])  # Baltic

plt.figure(figsize=(5,3))

# Plot with markers (important for discrete grid cases)
plt.plot(x, A, marker='o', linewidth=2, label="Atlantic")
plt.plot(x, B, marker='s', linewidth=2, label="Baltic")

# Labels
plt.xlabel("Grid points [-]")
plt.ylabel("Relative error [%]")
plt.gca().invert_xaxis()

# Styling
plt.title("Phase Speed Error vs Grid Points", size=13)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

# Optional: invert x-axis (coarser → finer visually left to right)
plt.gca().invert_xaxis()

plt.tight_layout()
plt.savefig("plots/phase.png", dpi=300)
plt.show()
plt.savefig("kelvin-waves-report/Figures/phase.png", dpi=300)