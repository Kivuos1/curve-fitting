import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def estimate_f0_fft(t, y):
    """
    Estimate fundamental frequency using FFT peak detection.
    """
    dt = np.mean(np.diff(t))
    y0 = y - np.mean(y)  # remove DC offset

    Y = np.fft.rfft(y0)
    freqs = np.fft.rfftfreq(len(y0), d=dt)

    idx = np.argmax(np.abs(Y[1:])) + 1
    return freqs[idx]

def least_squares_two_sines(t, y, f0):
    """
    LS model:
    y ≈ a1 sin(wt) + b1 cos(wt) + a3 sin(3wt) + b3 cos(3wt)
    """
    w = 2*np.pi*f0

    M = np.column_stack([
        np.sin(w*t),
        np.cos(w*t),
        np.sin(3*w*t),
        np.cos(3*w*t)
    ])

    theta, *_ = np.linalg.lstsq(M, y, rcond=None)
    a1, b1, a3, b3 = theta

    A1 = np.sqrt(a1**2 + b1**2)
    A3 = np.sqrt(a3**2 + b3**2)

    return a1, b1, a3, b3, A1, A3, M

def sine_model(t, A1, phi1, A3, phi3, f0):
    w = 2*np.pi*f0
    return A1*np.sin(w*t + phi1) + A3*np.sin(3*w*t + phi3)

def main():
    data = np.loadtxt("dataset2.txt")
    t = data[:, 0]
    y = data[:, 1]

    # 1) Periodicity estimation
    f0 = estimate_f0_fft(t, y)
    T0 = 1.0 / f0

    print("FFT Periodicity Estimate")
    print("------------------------")
    print(f"f0 ≈ {f0:.6f}")
    print(f"T  ≈ {T0:.6f}")

    # 2) Least Squares
    a1, b1, a3, b3, A1_ls, A3_ls, M = least_squares_two_sines(t, y, f0)

    print("\nLeast Squares Results")
    print("--------------------")
    print(f"A1 (LS) ≈ {A1_ls:.6f}")
    print(f"A3 (LS) ≈ {A3_ls:.6f}")
    print("First 3 rows of M:")
    print(M[:3])

    w = 2*np.pi*f0
    y_ls = (
        a1*np.sin(w*t) + b1*np.cos(w*t) +
        a3*np.sin(3*w*t) + b3*np.cos(3*w*t)
    )

    # 3) curve_fit
    p0 = [A1_ls, 0.0, A3_ls, 0.0]
    popt, _ = curve_fit(
        lambda tt, A1, phi1, A3, phi3: sine_model(tt, A1, phi1, A3, phi3, f0),
        t, y, p0=p0, maxfev=20000
    )

    A1_cf, phi1_cf, A3_cf, phi3_cf = popt

    print("\ncurve_fit Results")
    print("----------------")
    print(f"A1 (curve_fit) ≈ {A1_cf:.6f}")
    print(f"A3 (curve_fit) ≈ {A3_cf:.6f}")

    print("\nDifference")
    print("----------")
    print(f"|A1_cf − A1_ls| ≈ {abs(A1_cf - A1_ls):.6e}")
    print(f"|A3_cf − A3_ls| ≈ {abs(A3_cf - A3_ls):.6e}")

    # Plot
    t_plot = np.linspace(t.min(), t.max(), 3000)
    y_ls_plot = (
        a1*np.sin(w*t_plot) + b1*np.cos(w*t_plot) +
        a3*np.sin(3*w*t_plot) + b3*np.cos(3*w*t_plot)
    )
    y_cf_plot = sine_model(t_plot, A1_cf, phi1_cf, A3_cf, phi3_cf, f0)

    plt.figure(figsize=(9,5))
    plt.plot(t, y, '.', markersize=3, label="Noisy data")
    plt.plot(t_plot, y_ls_plot, linewidth=2, label="Least Squares fit")
    plt.plot(t_plot, y_cf_plot, '--', linewidth=2, label="curve_fit fit")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("Dataset 2: Two-Sine Amplitude Estimation")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("curve2_fits.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
