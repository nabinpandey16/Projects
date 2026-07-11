# Lévy Processes: Theory and Model Construction

## 1. Definition

A stochastic process $\{X_t\}_{t \ge 0}$ with $X_0 = 0$ is a **Lévy process** if it satisfies:

1. **Independent increments**: for $0 \le t_1 < t_2 < \dots < t_n$, the increments $X_{t_2}-X_{t_1}, \dots, X_{t_n}-X_{t_{n-1}}$ are independent.
2. **Stationary increments**: $X_{t+s} - X_s$ has the same distribution as $X_t$ for all $s,t \ge 0$.
3. **Stochastic continuity**: $\lim_{h\to 0} P(|X_{t+h}-X_t|>\epsilon) = 0$ for all $\epsilon>0$ (rules out fixed jump times, not path continuity).

Brownian motion and the Poisson process are the two building blocks; every Lévy process is (in law) a combination of a drift, a Brownian component, and a jump component.

## 2. Lévy–Khintchine Representation

The defining object is the **characteristic function**:

$$
\mathbb{E}\left[e^{i u X_t}\right] = e^{t\,\psi(u)}
$$

where the **characteristic exponent** $\psi(u)$ has the Lévy–Khintchine form:

$$
\psi(u) = i\mu u - \tfrac{1}{2}\sigma^2 u^2 + \int_{\mathbb{R}\setminus\{0\}} \left( e^{iux} - 1 - iux\,\mathbb{1}_{|x|<1} \right) \nu(dx)
$$

The triplet $(\mu, \sigma^2, \nu)$ fully characterizes the process:

- $\mu$: drift
- $\sigma^2$: Gaussian (diffusive) variance rate
- $\nu(dx)$: the **Lévy measure** — governs jump frequency and size, must satisfy $\int \min(1,x^2)\,\nu(dx) < \infty$

This is the mathematical statement of "Brownian motion plus jumps": every Lévy process decomposes as

$$
X_t = \mu t + \sigma W_t + J_t
$$

where $J_t$ is a pure-jump Lévy process built from $\nu$.

## 3. The Models

### 3.1 Merton Jump-Diffusion

Adds a **compound Poisson** jump term to standard Brownian motion with drift. Jumps arrive at rate $\lambda$ (Poisson), with i.i.d. jump sizes $Y_i \sim \mathcal{N}(\mu_J, \sigma_J^2)$:

$$
X_t = \mu t + \sigma W_t + \sum_{i=1}^{N_t} Y_i, \qquad N_t \sim \text{Poisson}(\lambda t)
$$

Lévy measure: $\nu(dx) = \lambda \cdot \phi(x; \mu_J, \sigma_J^2)\,dx$ (finite total mass $\lambda$ — finite activity, finite number of jumps in any interval).

Used for: asset log-returns with rare large moves (crashes, news shocks) — this is what Merton (1976) introduced for option pricing with jump risk.

### 3.2 Variance Gamma (VG)

Built by **time-changing (subordinating)** Brownian motion with an independent Gamma process $G_t$ (a subordinator: increasing pure-jump Lévy process):

$$
X_t = \theta G_t + \sigma W_{G_t}, \qquad G_t \sim \text{Gamma}(t/\kappa,\, \kappa)
$$

Here $\kappa$ controls the variance of the time change (kurtosis of returns), $\theta$ controls skewness. VG has **infinite activity** (infinitely many small jumps in any interval) but finite variation, and no Brownian component at all — it is pure jump. Widely used in high-frequency finance (Madan, Carr, Chang 1998) because it captures fat tails and skew better than Merton with fewer parameters.

### 3.3 Alpha-Stable (Lévy Flight)

Characteristic exponent has the self-similar form:

$$
\psi(u) = i\mu u - |\sigma u|^\alpha \left[1 - i\beta\, \text{sgn}(u) \tan(\pi\alpha/2)\right], \quad \alpha \in (0,2]
$$

- $\alpha=2$: recovers Gaussian (Brownian motion)
- $\alpha<2$: **infinite variance**, heavy power-law tails $P(|X|>x) \sim x^{-\alpha}$
- $\beta$: skewness parameter

This is the process behind **Lévy flights** — long-range, self-similar jumps used to model anomalous diffusion, foraging patterns, and (in econophysics) the heavy-tailed nature of high-frequency price changes (Mandelbrot's original 1963 cotton-price model used $\alpha \approx 1.7$).

## 4. Comparison Table

| Property | Merton JD | Variance Gamma | Alpha-Stable |
|---|---|---|---|
| Activity | Finite | Infinite | Infinite |
| Variation | Finite | Finite | Infinite (for $\alpha<1$) |
| Variance | Finite | Finite | **Infinite** (for $\alpha<2$) |
| Brownian part | Yes | No (pure jump) | No (pure jump, $\alpha<2$) |
| Tail behavior | Light (Gaussian jumps) | Semi-heavy (exponential) | Heavy (power-law) |
| Typical use | Crash risk, options | HF equity returns | Extreme events, anomalous diffusion |

## 5. Simulation Strategy

- **Merton**: simulate directly — Euler-discretize the Brownian part, and independently simulate Poisson jump times/sizes and add them in.
- **VG**: simulate the Gamma subordinator $G_t$ first (independent Gamma increments), then simulate Brownian motion evaluated at the *random* subordinated time.
- **Alpha-stable**: cannot use rejection/inversion easily since there's no closed-form density in general (except $\alpha=1$ Cauchy, $\alpha=2$ Gaussian). Use the **Chambers–Mallows–Stuck (1976)** algorithm, which generates exact stable random variables from two uniforms.

See the accompanying `levy_simulation.py` for implementation of all three.
