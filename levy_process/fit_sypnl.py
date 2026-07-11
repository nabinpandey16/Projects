"""
Lévy Process Calibration & Forecasting for SYPNL (NEPSE)
===========================================================
Fits Merton Jump-Diffusion, Variance Gamma, and Alpha-Stable models to
SYPNL daily log-returns using cumulant/MLE matching, then Monte Carlo
simulates forward price paths to build a probabilistic price forecast.

Key idea used throughout: for ANY Lévy process, cumulants of X_t scale
linearly in t. So per-day cumulants (mean, variance, skewness, excess
kurtosis of daily log-returns) can be matched directly to each model's
theoretical per-unit-time cumulant formulas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import levy_stable

np.random.seed(7)

# --------------------------- Load & prepare data ---------------------------
df = pd.read_csv("levy_process/sypnl_prices.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)
df["log_price"] = np.log(df["LTP"])
df["log_return"] = df["log_price"].diff()
df = df.dropna(subset=["log_return"]).reset_index(drop=True)

# Full sample vs. "post-stabilization" sample (after circuit-limit listing rally)
POST_STAB_START = "2026-02-01"
full_returns = df["log_return"].values
post_df = df[df["Date"] >= POST_STAB_START]
post_returns = post_df["log_return"].values

print(f"Full sample: {len(full_returns)} daily returns ({df['Date'].min().date()} to {df['Date'].max().date()})")
print(f"Post-stabilization sample: {len(post_returns)} daily returns ({post_df['Date'].min().date()} to {post_df['Date'].max().date()})")

last_price = df["LTP"].iloc[-1]
last_date = df["Date"].iloc[-1]
print(f"Last price: {last_price} on {last_date.date()}")


def empirical_cumulants(r):
    mean = r.mean()
    var = r.var()
    std = r.std()
    skew = ((r - mean) ** 3).mean() / std ** 3
    kurt = ((r - mean) ** 4).mean() / std ** 4 - 3  # excess kurtosis
    return mean, var, skew, kurt


# --------------------------- Merton Jump-Diffusion fit ---------------------------
def fit_merton(r):
    mean, var, skew, kurt = empirical_cumulants(r)
    c1, c2 = mean, var
    c3 = skew * var ** 1.5
    c4 = kurt * var ** 2

    # Robust jump detection: threshold at 3x MAD-based sigma
    med = np.median(r)
    mad = np.median(np.abs(r - med)) * 1.4826
    is_jump = np.abs(r - med) > 3 * mad
    lam = max(is_jump.sum() / len(r), 0.02)  # jump rate per day, floor to avoid 0

    def eqs(x):
        mu, sigma, muJ, sigmaJ = x
        e1 = mu + lam * muJ - c1
        e2 = sigma ** 2 + lam * (muJ ** 2 + sigmaJ ** 2) - c2
        e3 = lam * (muJ ** 3 + 3 * muJ * sigmaJ ** 2) - c3
        e4 = lam * (muJ ** 4 + 6 * muJ ** 2 * sigmaJ ** 2 + 3 * sigmaJ ** 4) - c4
        return [e1, e2, e3, e4]

    # initial guess: diffusive vol from non-jump returns, jump stats from jump returns
    sigma0 = r[~is_jump].std() if is_jump.sum() < len(r) else r.std()
    muJ0 = r[is_jump].mean() if is_jump.sum() > 0 else 0.0
    sigmaJ0 = r[is_jump].std() if is_jump.sum() > 1 else abs(muJ0) + 1e-3
    x0 = [c1, max(sigma0, 1e-4), muJ0, max(sigmaJ0, 1e-4)]
    sol = fsolve(eqs, x0, full_output=False)
    mu, sigma, muJ, sigmaJ = sol
    return {"mu": mu, "sigma": abs(sigma), "lambda": lam, "muJ": muJ, "sigmaJ": abs(sigmaJ)}


# --------------------------- Variance Gamma fit ---------------------------
def fit_vg(r):
    mean, var, skew, kurt = empirical_cumulants(r)
    c1, c2 = mean, var
    c3 = skew * var ** 1.5

    def eqs(x):
        theta, sigma, kappa = x
        e1 = theta - c1
        e2 = (sigma ** 2 + theta ** 2 * kappa) - c2
        e3 = (2 * theta ** 3 * kappa ** 2 + 3 * sigma ** 2 * theta * kappa) - c3
        return [e1, e2, e3]

    x0 = [c1, np.sqrt(max(var, 1e-6)), 0.3]
    sol = fsolve(eqs, x0, full_output=False)
    theta, sigma, kappa = sol
    return {"theta": theta, "sigma": abs(sigma), "kappa": abs(kappa)}


# --------------------------- Alpha-Stable fit (MLE) ---------------------------
def fit_stable(r):
    alpha, beta, loc, scale = levy_stable.fit(r)
    return {"alpha": alpha, "beta": beta, "loc": loc, "scale": scale}


# --------------------------- Run fits on both samples ---------------------------
results = {}
for label, r in [("full", full_returns), ("post_stab", post_returns)]:
    results[label] = {
        "merton": fit_merton(r),
        "vg": fit_vg(r),
        "stable": fit_stable(r),
        "empirical": dict(zip(["mean", "var", "skew", "kurt"], empirical_cumulants(r))),
    }

for label in results:
    print(f"\n=== {label} sample fit ===")
    for model, params in results[label].items():
        print(f"  {model:10s}: {params}")


# --------------------------- Monte Carlo forward simulation ---------------------------
def simulate_forward(model, params, n_paths, n_days, s0):
    paths = np.zeros((n_paths, n_days + 1))
    paths[:, 0] = np.log(s0)
    for p in range(n_paths):
        if model == "merton":
            mu, sigma, lam, muJ, sigmaJ = (params[k] for k in ["mu", "sigma", "lambda", "muJ", "sigmaJ"])
            dW = np.random.normal(0, 1, n_days)
            cont = mu + sigma * dW
            dN = np.random.poisson(lam, n_days)
            jumps = np.array([np.sum(np.random.normal(muJ, sigmaJ, n)) if n > 0 else 0.0 for n in dN])
            inc = cont + jumps
        elif model == "vg":
            theta, sigma, kappa = (params[k] for k in ["theta", "sigma", "kappa"])
            dG = np.maximum(np.random.gamma(shape=1 / kappa, scale=kappa, size=n_days), 1e-9)
            inc = theta * dG + sigma * np.sqrt(dG) * np.random.normal(0, 1, n_days)
        else:  # stable
            alpha, beta, loc, scale = (params[k] for k in ["alpha", "beta", "loc", "scale"])
            inc = levy_stable.rvs(alpha, beta, loc=loc, scale=scale, size=n_days)
        paths[p, 1:] = paths[p, 0] + np.cumsum(inc)
    return np.exp(paths)  # back to price level


N_PATHS = 3000
N_DAYS = 30  # ~1.5 trading months ahead

forecasts = {}
for model in ["merton", "vg", "stable"]:
    forecasts[model] = simulate_forward(model, results["post_stab"][model], N_PATHS, N_DAYS, last_price)

future_dates = pd.bdate_range(last_date, periods=N_DAYS + 1)

# --------------------------- Plot: fan chart per model ---------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
percentiles = [5, 25, 50, 75, 95]
colors = {"merton": "#2563eb", "vg": "#16a34a", "stable": "#dc2626"}
titles = {"merton": "Merton Jump-Diffusion", "vg": "Variance Gamma", "stable": "Alpha-Stable"}

# recent history to show for context
hist_tail = df.tail(60)

for ax, model in zip(axes, ["merton", "vg", "stable"]):
    ax.plot(hist_tail["Date"], hist_tail["LTP"], color="black", lw=1.2, label="Historical")
    paths = forecasts[model]
    pct = np.percentile(paths, percentiles, axis=0)
    ax.fill_between(future_dates, pct[0], pct[4], color=colors[model], alpha=0.15, label="5-95%")
    ax.fill_between(future_dates, pct[1], pct[3], color=colors[model], alpha=0.3, label="25-75%")
    ax.plot(future_dates, pct[2], color=colors[model], lw=2, label="Median")
    ax.set_title(titles[model])
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

axes[0].set_ylabel("Price (NPR)")
plt.suptitle(f"SYPNL {N_DAYS}-Trading-Day Price Forecast (fitted on post-stabilization data)", y=1.02)
plt.tight_layout()
plt.savefig("/home/claude/levy/nepse/sypnl_forecast.png", dpi=150, bbox_inches="tight")
print("\nSaved: sypnl_forecast.png")

# --------------------------- Summary table ---------------------------
print(f"\nForecast summary at t+{N_DAYS} trading days (from {last_price:.2f} NPR on {last_date.date()}):")
summary_rows = []
for model in ["merton", "vg", "stable"]:
    final_prices = forecasts[model][:, -1]
    row = {
        "model": titles[model],
        "p05": np.percentile(final_prices, 5),
        "p25": np.percentile(final_prices, 25),
        "median": np.percentile(final_prices, 50),
        "mean": final_prices.mean(),
        "p75": np.percentile(final_prices, 75),
        "p95": np.percentile(final_prices, 95),
    }
    summary_rows.append(row)
    print(f"  {row['model']:22s}  p05={row['p05']:8.1f}  p25={row['p25']:8.1f}  "
          f"median={row['median']:8.1f}  mean={row['mean']:8.1f}  p75={row['p75']:8.1f}  p95={row['p95']:8.1f}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("/home/claude/levy/nepse/sypnl_forecast_summary.csv", index=False)
print("\nSaved: sypnl_forecast_summary.csv")
