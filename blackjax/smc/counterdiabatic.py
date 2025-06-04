import jax
import jax.numpy as jnp
import optax
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# 1) GENERAL POISSON BRACKET FUNCTION (scalar)
# =============================================================================
def poisson_bracket_fn(f, g):
    """
    Given f(q,p)->scalar and g(q,p)->scalar,
    returns bracket(q,p) = {f,g} computed via autodiff.
    """
    df_dq = jax.grad(lambda qr, pr: f(qr, pr), argnums=0)
    df_dp = jax.grad(lambda qr, pr: f(qr, pr), argnums=1)
    dg_dq = jax.grad(lambda qr, pr: g(qr, pr), argnums=0)
    dg_dp = jax.grad(lambda qr, pr: g(qr, pr), argnums=1)

    def bracket(q, p):
        return df_dq(q, p) * dg_dp(q, p) - df_dp(q, p) * dg_dq(q, p)

    return bracket

# =============================================================================
# 2) HAMILTONIAN AND μ(t)
# =============================================================================
m = 1.0  # Mass of the particle

# Harmonic oscillator Hamiltonian with moving center mu(t)
def H0_mu(q, p, mu):
    """
    Hamiltonian for a 1D harmonic oscillator with moving center mu(t).
    Args:
        q: position
        p: momentum
        mu: center of the potential (can be time-dependent)
    Returns:
        Scalar energy value
    """
    return 0.5 * (p ** 2) / m + 0.5 * (q - mu) ** 2

# Compute dH0/dmu using automatic differentiation
# This replaces the manual formula with autodiff for generality
dH0_dmu = jax.grad(lambda q, p, mu: H0_mu(q, p, mu), argnums=2)

v = 3.0  # Velocity of the moving center, set well above typical particle speed
t_move = 2.0  # You may also increase t_move if you want the well to move for longer

def mu_fn(t):
    return jnp.where(t < t_move, v * t, v * t_move)

dot_mu_fn = jax.grad(mu_fn)

# =============================================================================
# 3) PARAMETRIC ANSATZ A(q,p; θ)
# =============================================================================
def A_ansatz(params, q, p):
    θ1, θ2 = params
    return θ1 * p + θ2 * (q * p)

# =============================================================================
# 4) FIT FUNCTION USING GENERAL POISSON BRACKET
# =============================================================================
def fit_gauge_potential(mu, samples, init_params, num_iters=200, lr=0.05):
    """
    Fit A(q,p; θ) by minimizing mean_{samples}[ ( {A,H} - ∂H/∂μ )^2 ].
    """
    qp_batch = jnp.array(samples)  # shape (N,2)
    params = jnp.array(init_params)

    H_fixed = lambda q, p: H0_mu(q, p, mu)
    dH_fixed = lambda q, p: dH0_dmu(q, p, mu)

    def bracket_A_H(params, q, p):
        f = lambda qr, pr: A_ansatz(params, qr, pr)
        return poisson_bracket_fn(f, H_fixed)(q, p)

    def R(params, q, p):
        return bracket_A_H(params, q, p) - dH_fixed(q, p)

    vmap_R = jax.vmap(lambda qr, pr, θ: R(θ, qr, pr), in_axes=(0, 0, None))

    def loss_fn(params, qp_batch):
        qs = qp_batch[:, 0]
        ps = qp_batch[:, 1]
        R_vals = vmap_R(qs, ps, params)
        return jnp.mean(R_vals ** 2)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, qp_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, qp_batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for _ in range(num_iters):
        params, opt_state, _ = update(params, opt_state, qp_batch)

    return params

# =============================================================================
# 5) LEAPFROG STEPS (SINGLE PARTICLE)
# =============================================================================
def naive_leapfrog_step(q, p, mu, mu_next, eps):
    p_half = p - 0.5 * eps * (q - mu)
    q_new = q + eps * (p_half / m)
    p_new = p_half - 0.5 * eps * (q_new - mu_next)
    return q_new, p_new

# Scalar ∂A derivatives
dA_dq_scalar = jax.grad(lambda q, p, θ: A_ansatz(θ, q, p), argnums=0)
dA_dp_scalar = jax.grad(lambda q, p, θ: A_ansatz(θ, q, p), argnums=1)

def cd_leapfrog_step(q, p, theta, mu, mu_next, dot_mu, dot_mu_next, eps):
    # First half‐kick on p
    p_half = p - 0.5 * eps * ((q - mu) + dot_mu * dA_dq_scalar(q, p, theta))
    # Drift on q
    q_new = q + eps * ((p_half / m) + dot_mu * dA_dp_scalar(q, p_half, theta))
    # Second half‐kick on p
    p_new = p_half - 0.5 * eps * ((q_new - mu_next) + dot_mu_next * dA_dq_scalar(q_new, p_half, theta))
    return q_new, p_new

# Vectorized versions for arrays of shape (M,)
naive_step = jax.vmap(naive_leapfrog_step, in_axes=(0, 0, None, None, None))
cd_step = jax.vmap(cd_leapfrog_step, in_axes=(0, 0, None, None, None, None, None, None))

# =============================================================================
# 6) SIMULATION: NAÏVE HMC VS CD WITH ONLINE FITTING
# =============================================================================
M = 3000  # Number of particles
N_steps = 100  # Increased number of steps for longer simulation
delta_t = 0.1  # Time step size
eps = 0.1      # Leapfrog step size
momentum_refresh_interval = 20  # Resample momenta every n steps

# Initialize particle ensembles
key = jax.random.PRNGKey(0)
key, sub = jax.random.split(key)
q_naive = jax.random.normal(sub, (M,))
key, sub = jax.random.split(key)
p_naive = jax.random.normal(sub, (M,)) * jnp.sqrt(m)

q_cd = q_naive.copy()
p_cd = p_naive.copy()

theta = jnp.array([0.0, 0.0])  # initial θ

theta_history = []
snapshots = {'naive': [], 'cd': [], 'mu': []}

for k in range(N_steps + 1):
    t_k = k * delta_t
    mu_k = float(mu_fn(t_k))
    dot_mu_k = float(dot_mu_fn(t_k))

    # Record histograms every 25 steps
    if k % 10 == 0:
        snapshots['naive'].append(np.array(q_naive))
        snapshots['cd'].append(np.array(q_cd))
        snapshots['mu'].append(mu_k)

    # Re-fit A every 20 steps
    if (k % 20 == 0) and (k < N_steps):
        samples = np.stack([np.array(q_cd), np.array(p_cd)], axis=1)
        theta = fit_gauge_potential(mu_k, samples, init_params=theta,
                                    num_iters=100, lr=0.05)

    theta_history.append(np.array(theta))

    # Randomize momenta for naive HMC every momentum_refresh_interval steps
    if (k % momentum_refresh_interval == 0) and (k < N_steps):
        key, sub = jax.random.split(key)
        p_naive = jax.random.normal(sub, (M,)) * jnp.sqrt(m)
        p_cd = p_naive.copy()

    if k == N_steps:
        break

    mu_k1 = float(mu_fn(t_k + delta_t))
    dot_mu_k1 = float(dot_mu_fn(t_k + delta_t))

    # --- Naïve step ---
    q_naive, p_naive = naive_step(q_naive, p_naive, mu_k, mu_k1, eps)

    # Check for NaNs in naive HMC
    if jnp.isnan(q_naive).any():
        print(f"Warning: NaNs detected in q_naive at step {k} (count: {jnp.isnan(q_naive).sum()})")
    if jnp.isnan(p_naive).any():
        print(f"Warning: NaNs detected in p_naive at step {k} (count: {jnp.isnan(p_naive).sum()})")

    # --- CD step ---
    q_cd, p_cd = cd_step(q_cd, p_cd, theta, mu_k, mu_k1, dot_mu_k, dot_mu_k1, eps)

# Convert θ history to NumPy
theta_history = np.stack(theta_history, axis=0)

# =============================================================================
# 7) PLOTTING RESULTS
# =============================================================================
# Create a 3x5 grid for 15 histogram plots plus the parameter plot
fig, axes = plt.subplots(3, 5, figsize=(24, 14))

# 7a) Plot θ₁ and θ₂ over time in the first subplot (top-left)
times = np.arange(N_steps + 1) * delta_t
axes = axes.flatten()
axes[0].plot(times, theta_history[:, 0], label=r'$\theta_1$')
axes[0].plot(times, theta_history[:, 1], label=r'$\theta_2$')
axes[0].axhline(1.0, color='k', linestyle='--', label='ideal θ₁=1, θ₂=0')
axes[0].set_title("Learned parameters over time")
axes[0].set_xlabel("t")
axes[0].set_ylabel("θ value")
axes[0].legend()

# 7b) Histogram snapshots for naïve vs CD, with legend
num_hist_axes = 14  # 15 total plots, 1 for theta, 14 for histograms
num_snaps = len(snapshots['naive'])
if num_snaps > num_hist_axes:
    selected_indices = np.linspace(0, num_snaps - 1, num_hist_axes, dtype=int)
else:
    selected_indices = np.arange(num_snaps)

for plot_idx, snap_idx in enumerate(selected_indices):
    ax = axes[plot_idx + 1]  # axes[0] is for theta plot
    naive_snap = snapshots['naive'][snap_idx]
    cd_snap = snapshots['cd'][snap_idx]
    mu_val = snapshots['mu'][snap_idx]

    print(naive_snap, "naive_snap")

    sns.histplot(naive_snap, bins=50, stat='density',
                 color='C0', alpha=0.4, label='Naïve', ax=ax)
    sns.histplot(cd_snap, bins=50, stat='density',
                 color='C1', alpha=0.4, label='CD', ax=ax)
    # Use a dynamic xs range centered on mu_val for the true density
    xs = np.linspace(mu_val - 4, mu_val + 4, 400)
    rho = np.exp(-0.5 * (xs - mu_val) ** 2)
    rho /= np.trapz(rho, xs)
    ax.plot(xs, rho, 'r-', lw=2, label='True')
    ax.set_title(f"t={snap_idx*25*delta_t:.2f}, μ={mu_val:.2f}")
    ax.set_xlabel("q")
    ax.set_ylabel("Density")
    # Dynamically set x-limits to follow the moving well
    x_min = min(-2, mu_val - 4)
    x_max = max(5, mu_val + 4)
    ax.set_xlim(x_min, x_max)
    ax.legend()

plt.tight_layout()
plt.savefig("counterdiabatic.png")
