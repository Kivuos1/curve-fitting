import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# 1) Load data (two columns: f, B)
# -----------------------------
DATA_FILE = "dataset3.txt"  # rename if needed

data = np.loadtxt(DATA_FILE)
f = data[:, 0].astype(float)   # frequency (Hz)
B = data[:, 1].astype(float)   # intensity

# Sort by frequency (helps plotting / sanity)
idx = np.argsort(f)
f = f[idx]
B = B[idx]

# Planck constants (SI)
h_TRUE  = 6.62607015e-34        # J*s (exact by definition)
kB_TRUE = 1.380649e-23          # J/K (exact by definition)
c_TRUE  = 299792458.0           # m/s (exact by definition)

# -----------------------------
# 2) Planck model: B(f, T)
# Using frequency form (often written B_nu)
# B(f, T) = (2 h f^3 / c^2) * 1/(exp(h f / (kB T)) - 1)
# -----------------------------
def planck_Bf_T(freq, T, h=h_TRUE, kB=kB_TRUE, c=c_TRUE):
    x = (h * freq) / (kB * T)
    # Avoid overflow for huge x: exp(x) overflows, but then 1/(exp(x)-1) ~ 0
    # We'll clip x to keep exp stable.
    x = np.clip(x, 1e-12, 700.0)
    return (2.0 * h * freq**3 / (c**2)) * (1.0 / (np.exp(x) - 1.0))

# -----------------------------
# 3) Preprocess: handle negative noisy values
#    Planck curve is nonnegative; your data has noise that can go negative.
#    We'll fit using only positive B values (simple + defensible).
# -----------------------------
mask = B > 0
f_fit = f[mask]
B_fit = B[mask]

# -----------------------------
# 4) Initial guess for T:
#    For B_nu (frequency form), peak occurs near x ≈ 2.821439 where x = h f / (kB T)
#    => T0 ≈ h * f_peak / (2.821439 * kB)
# -----------------------------
peak_i = np.argmax(B_fit)
f_peak = f_fit[peak_i]
T0 = (h_TRUE * f_peak) / (2.821439 * kB_TRUE)
T0 = float(np.clip(T0, 50.0, 50000.0))  # keep it sane

print(f"[Init] f_peak = {f_peak:.3e} Hz, T0 ≈ {T0:.2f} K")

# -----------------------------
# Solution A: Fit only T (h,kB,c known)
# -----------------------------
def model_T_only(freq, T):
    return planck_Bf_T(freq, T)

# Bounds to help convergence; adjust if needed
popt_T, pcov_T = curve_fit(
    model_T_only,
    f_fit,
    B_fit,
    p0=[T0],
    bounds=([1.0], [1e6]),
    maxfev=20000
)
T_hat = popt_T[0]
T_std = float(np.sqrt(pcov_T[0, 0])) if pcov_T.size else float("nan")
print(f"[Solution A] Estimated T = {T_hat:.4f} K (std ~ {T_std:.4g} K)")

# -----------------------------
# Solution B: Fit h, kB, c, T
# This is typically poorly-conditioned because parameters trade off.
# We'll use bounds + decent starting points to make it converge.
# -----------------------------
def model_all(freq, h, kB, c, T):
    x = (h * freq) / (kB * T)
    x = np.clip(x, 1e-12, 700.0)
    return (2.0 * h * freq**3 / (c**2)) * (1.0 / (np.exp(x) - 1.0))

p0_all = [h_TRUE, kB_TRUE, c_TRUE, T_hat]

# Tight-ish bounds around physical reality to prevent nonsense fits
bounds_lo = [h_TRUE * 0.5,  kB_TRUE * 0.5,  c_TRUE * 0.8,  1.0]
bounds_hi = [h_TRUE * 1.5,  kB_TRUE * 1.5,  c_TRUE * 1.2,  1e6]

popt_all, pcov_all = curve_fit(
    model_all,
    f_fit,
    B_fit,
    p0=p0_all,
    bounds=(bounds_lo, bounds_hi),
    maxfev=50000
)

h_hat, kB_hat, c_hat, T_hat2 = popt_all
print("[Solution B] Estimated (h, kB, c, T):")
print(f"  h  = {h_hat:.8e}   (true {h_TRUE:.8e})")
print(f"  kB = {kB_hat:.8e}  (true {kB_TRUE:.8e})")
print(f"  c  = {c_hat:.6f}   (true {c_TRUE:.6f})")
print(f"  T  = {T_hat2:.4f} K (Solution A gave {T_hat:.4f} K)")

# -----------------------------
# Plot results
# -----------------------------
f_plot = np.linspace(f.min(), f.max(), 2000)

plt.figure()
plt.plot(f, B, label="Measured data (noisy)")
plt.plot(f_plot, model_T_only(f_plot, T_hat), label="Fit: T only")
plt.plot(f_plot, model_all(f_plot, *popt_all), label="Fit: h,kB,c,T")

plt.xscale("log")
plt.xlabel("Frequency f (Hz)")
plt.ylabel("Intensity B(f)")
plt.title("Blackbody curve fitting (Planck law)")
plt.legend()
plt.tight_layout()
plt.savefig("curve3_fits.png", dpi=200)
plt.show()
