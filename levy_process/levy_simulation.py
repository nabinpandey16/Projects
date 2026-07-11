"""
Lévy Process Simulation
========================
Implements and compares three canonical Lévy process models built from the
Lévy-Khintchine decomposition X_t = drift + Brownian + jumps:

  1. Merton Jump-Diffusion   -- Brownian motion + compound Poisson jumps
  2. Variance Gamma (VG)     -- Brownian motion subordinated by a Gamma process
  3. Alpha-Stable            -- pure-jump, heavy-tailed self-similar process
                                 (Chambers-Mallows-Stuck algorithm)

Author: generated for Nabin's econophysics work
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable

# ----------------------------- Configuration -----------------------------
np.random.seed(42)

T = 1.0             # total time horizon (e.g. 1 year, or 1 trading day)
N_STEPS = 2000       # number of time discretization steps
N_PATHS = 5          # number of sample paths to draw per model
dt = T / N_STEPS
t_grid = np.linspace(0, T, N_STEPS + 1)

# Merton jump-diffusion parameters
MERTON_MU = 0.05        # drift
MERTON_SIGMA = 0.2       # Brownian volatility
MERTON_LAMBDA = 5.0       # jump intensity (jumps per unit time)
MERTON_JUMP_MU = -0.02    # mean jump size (negative = crash-prone)
MERTON_JUMP_SIGMA = 0.08  # jump size std dev

# Variance Gamma parameters
VG_THETA = -0.1     # skewness of the Brownian drift under subordination
VG_SIGMA = 0.2       # Brownian vol under subordination
VG_KAPPA = 0.2       # variance rate of the Gamma subordinator (controls kurtosis)

# Alpha-stable parameters
STABLE_ALPHA = 1.7   # tail index (2 = Gaussian, lower = heavier tails)
STABLE_BETA = 0.0    # skewness
STABLE_SCALE = 0.05  # scale per unit time
STABLE_LOC = 0.0     # location/drift

print("Configuration loaded. Simulating", N_PATHS, "paths x 3 models over",
      N_STEPS, "steps...")


# --------------------------- Model 1: Merton JD ---------------------------
def simulate_merton(n_paths=N_PATHS):
    """Brownian motion with drift + compound Poisson jumps (Normal jump sizes)."""
    paths = np.zeros((n_paths, N_STEPS + 1))
    for p in range(n_paths):
        # continuous (Brownian) part
        dW = np.random.normal(0, np.sqrt(dt), N_STEPS)
        cont = (MERTON_MU - 0.5 * MERTON_SIGMA**2) * dt + MERTON_SIGMA * dW

        # jump part: Poisson-distributed number of jumps per step, Normal sizes
        dN = np.random.poisson(MERTON_LAMBDA * dt, N_STEPS)
        jumps = np.zeros(N_STEPS)
        for i, n_jumps in enumerate(dN):
            if n_jumps > 0:
                jumps[i] = np.sum(np.random.normal(MERTON_JUMP_MU, MERTON_JUMP_SIGMA, n_jumps))

        increments = cont + jumps
        paths[p, 1:] = np.cumsum(increments)
    return paths


# --------------------------- Model 2: Variance Gamma ---------------------------
def simulate_vg(n_paths=N_PATHS):
    """Brownian motion subordinated by an independent Gamma process."""
    paths = np.zeros((n_paths, N_STEPS + 1))
    for p in range(n_paths):
        # Gamma subordinator increments: shape = dt/kappa, scale = kappa
        dG = np.random.gamma(shape=dt / VG_KAPPA, scale=VG_KAPPA, size=N_STEPS)
        dG = np.maximum(dG, 1e-12)  # avoid zero time-change

        # Brownian motion evaluated at the random subordinated time dG
        dW = np.random.normal(0, 1, N_STEPS) * np.sqrt(dG)
        increments = VG_THETA * dG + VG_SIGMA * dW

        paths[p, 1:] = np.cumsum(increments)
    return paths


# --------------------------- Model 3: Alpha-Stable ---------------------------
def simulate_alpha_stable(n_paths=N_PATHS):
    """Pure-jump stable Lévy process via scipy's Chambers-Mallows-Stuck sampler."""
    paths = np.zeros((n_paths, N_STEPS + 1))
    # scale per step follows dt^(1/alpha) self-similarity
    step_scale = STABLE_SCALE * dt ** (1 / STABLE_ALPHA)
    for p in range(n_paths):
        increments = levy_stable.rvs(
            STABLE_ALPHA, STABLE_BETA, loc=STABLE_LOC * dt, scale=step_scale, size=N_STEPS
        )
        paths[p, 1:] = np.cumsum(increments)
    return paths


# --------------------------------- Run ---------------------------------
merton_paths = simulate_merton()
vg_paths = simulate_vg()
stable_paths = simulate_alpha_stable()
print("Simulation complete.")

# --------------------------------- Plots ---------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

models = [
    ("Merton Jump-Diffusion", merton_paths, "#2563eb"),
    ("Variance Gamma", vg_paths, "#16a34a"),
    ("Alpha-Stable (α=%.1f)" % STABLE_ALPHA, stable_paths, "#dc2626"),
]

for col, (name, paths, color) in enumerate(models):
    ax_path = axes[0, col]
    for p in range(paths.shape[0]):
        ax_path.plot(t_grid, paths[p], lw=1, alpha=0.8, color=color)
    ax_path.set_title(name)
    ax_path.set_xlabel("t")
    ax_path.set_ylabel("$X_t$")
    ax_path.grid(alpha=0.3)

    ax_hist = axes[1, col]
    final_increments = np.diff(paths, axis=1).flatten()
    ax_hist.hist(final_increments, bins=80, color=color, alpha=0.7, density=True)
    ax_hist.set_title(f"Increment distribution")
    ax_hist.set_xlabel(r"$\Delta X$")
    ax_hist.set_yscale("log")
    ax_hist.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("levy_comparison.png", dpi=150)
print("Saved plot: levy_comparison.png")

# --------------------------------- Tail comparison ---------------------------------
fig2, ax = plt.subplots(figsize=(8, 6))
for name, paths, color in models:
    final_increments = np.diff(paths, axis=1).flatten()
    sorted_abs = np.sort(np.abs(final_increments))[::-1]
    rank = np.arange(1, len(sorted_abs) + 1) / len(sorted_abs)
    ax.loglog(sorted_abs, rank, label=name, color=color, lw=1.5)

ax.set_xlabel(r"$|\Delta X|$ (log scale)")
ax.set_ylabel("P(|ΔX| > x)  (log scale)")
ax.set_title("Tail comparison (log-log survival function)")
ax.legend()
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("levy_tails.png", dpi=150)
print("Saved plot: levy_tails.png")

print("\nSummary statistics (per-step increments):")
for name, paths, _ in models:
    inc = np.diff(paths, axis=1).flatten()
    print(f"  {name:28s}  mean={inc.mean():.6f}  std={inc.std():.6f}  "
          f"skew={((inc-inc.mean())**3).mean()/inc.std()**3:.3f}  "
          f"kurtosis={((inc-inc.mean())**4).mean()/inc.std()**4:.3f}")
