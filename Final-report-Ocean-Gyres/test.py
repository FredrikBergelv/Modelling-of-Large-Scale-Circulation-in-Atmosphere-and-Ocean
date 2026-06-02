import numpy as np
import matplotlib.pyplot as plt

v1 = 0.030092148
v2 = 0.047018446
v3 = 0.06770732

u1 = 4
u2 = 5
u3 = 6

# -------------------------
# DATA ARRAYS
# -------------------------
x = np.array([u1, u2, u3])
y = np.array([v1, v2, v3])

# -------------------------
# LINEAR FIT: y = a + b x
# -------------------------
b, a = np.polyfit(x, y, 1)  # slope b, intercept a

# fitted line
x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = a + b * x_fit

# -------------------------
# PRINT RESULTS
# -------------------------
print(f"Fit: y = a + b x")
print(f"a (intercept) = {a:.6f}")
print(f"b (slope)     = {b:.6f}")

# -------------------------
# PLOT
# -------------------------
plt.plot(x, y, "o", color="black", label="data")
plt.plot(x_fit, y_fit, "-", color="C0", label="linear fit")

# show equation on plot
plt.text(
    0.05, 0.95,
    f"y = {a:.4f} + {b:.4f}x",
    transform=plt.gca().transAxes,
    va="top",
    bbox=dict(facecolor="white", alpha=0.8)
)

plt.xlabel("Wind stress, $u_{10}$ [m/s]")
plt.ylabel("Meridional velocity, $v$ [m/s]")
plt.legend()
plt.grid(alpha=0.3)

plt.show()