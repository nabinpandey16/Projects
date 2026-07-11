# Lévy Process Models for NEPSE Equity Returns

A personal econophysics project exploring how Lévy process models — the kind used in high-frequency finance and anomalous diffusion — apply to a real Nepal Stock Exchange (NEPSE) stock. Built for learning, not investment advice.

## What this is

Every Lévy process decomposes into a drift, a Brownian (diffusion) part, and a jump part (the Lévy–Khintchine decomposition). This project implements three standard models built from that same decomposition, compares their behavior on simulated data, then calibrates all three to real daily price data for **SY Panel Nepal Limited (SYPNL)**, a stock listed on NEPSE in December 2025.

**Models implemented:**
- **Merton Jump-Diffusion** — Brownian motion + compound Poisson jumps (Merton, 1976)
- **Variance Gamma (VG)** — Brownian motion time-changed by a Gamma process (Madan, Carr & Chang, 1998)
- **Alpha-Stable (Lévy flight)** — heavy-tailed, self-similar process (Chambers–Mallows–Stuck algorithm, 1976; Mandelbrot, 1963)

## What's in this repo

| File | Description |
|---|---|
| `levy_process_theory.md` | Theory: Lévy–Khintchine formula, derivation of each model, comparison table |
| `levy_simulation.py` | Simulates all three models from scratch, plots sample paths, increment histograms, and a log-log tail comparison |
| `sypnl_prices.csv` | Daily OHLC price history for SYPNL, Dec 10 2025 – Jul 10 2026 (source: Merolagani) |
| `fit_sypnl.py` | Calibrates all three models to SYPNL log-returns (cumulant matching for Merton/VG, MLE for alpha-stable) and Monte Carlo simulates a 30-day forward price forecast |
| `sypnl_levy_writeup.tex` / `.pdf` | Full write-up: theory, methodology, results, and — importantly — limitations |

## Method, briefly

Cumulants of a Lévy process scale linearly in time, so daily-return cumulants (mean, variance, skewness, kurtosis) can be matched directly to each model's theoretical per-day cumulants:

- **Merton**: jump rate is estimated first via a robust outlier threshold (3× median absolute deviation), then the remaining four parameters are solved from the four cumulant equations.
- **VG**: exactly identified — three parameters, three cumulant equations.
- **Alpha-stable**: fit by maximum likelihood instead, since variance is infinite for α < 2 and can't be moment-matched.

The calibrated models are then used to Monte Carlo simulate thousands of forward price paths, giving a probability fan chart rather than a single-number forecast.

## Key caveats (read before trusting any numbers)

- SYPNL had only ~104 usable trading days after excluding its initial listing-rally period (which repeatedly hit NEPSE's 10% circuit limit and isn't representative i.i.d. behavior). That's a small sample for pinning down skew/kurtosis.
- Merton's fit is under-identified (5 parameters, 4 usable moment equations) and the jump-rate-fixing heuristic is a simplification, not a rigorous MLE.
- Lévy processes assume independent, identically distributed increments — real price momentum/mean-reversion isn't captured.
- This is not a trading signal. It's a demonstration of the calibration methodology on real, messy data.

## Running it

```bash
pip install numpy scipy matplotlib pandas
python levy_simulation.py      # general 3-model comparison
python fit_sypnl.py            # SYPNL calibration + forecast
```

## References

- Merton, R.C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics*, 3(1–2), 125–144.
- Madan, D.B., Carr, P.P., & Chang, E.C. (1998). The Variance Gamma process and option pricing. *European Finance Review*, 2(1), 79–105.
- Chambers, J.M., Mallows, C.L., & Stuck, B.W. (1976). A method for simulating stable random variables. *JASA*, 71(354), 340–344.
- Mandelbrot, B. (1963). The variation of certain speculative prices. *Journal of Business*, 36(4), 394–419.

---
*Part of ongoing coursework in econophysics at Tri-Chandra Multiple Campus, Tribhuvan University.*# Projects
