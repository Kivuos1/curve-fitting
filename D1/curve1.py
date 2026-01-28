import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load dataset (two columns: x, y)
    data = np.loadtxt("dataset1.txt")
    x = data[:, 0]
    y = data[:, 1]

    # Construct least-squares matrix M = [1, x]
    M = np.column_stack((np.ones_like(x), x))

    # Solve normal equations
    theta_hat = np.linalg.inv(M.T @ M) @ M.T @ y
    b_hat, m_hat = theta_hat

    print("Least Squares Results")
    print("---------------------")
    print(f"Estimated intercept (b): {b_hat:.4f}")
    print(f"Estimated slope (m):     {m_hat:.4f}")
    print("\nFirst 5 rows of M matrix:")
    print(M[:5])

    # Predicted values and noise estimate
    y_hat = b_hat + m_hat * x
    sigma = np.std(y - y_hat)

    # Plot
    plt.figure(figsize=(8, 5))

    plt.plot(x, y, 'o', markersize=3, label="Noisy data")

    # Error bars every 25 points
    idx = np.arange(0, len(x), 25)
    plt.errorbar(
        x[idx], y[idx],
        yerr=sigma,
        fmt='none',
        capsize=3,
        label="Error bars (every 25 points)"
    )

    # Fitted line
    x_fit = np.linspace(x.min(), x.max(), 500)
    y_fit = b_hat + m_hat * x_fit
    plt.plot(x_fit, y_fit, 'r', linewidth=2, label="Least squares fit")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Dataset 1: Least Squares Line Fit")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("curve1_fits.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
