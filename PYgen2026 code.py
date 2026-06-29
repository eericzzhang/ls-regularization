"""
Paper : "On Model Selection Consistency of l^s-regularized Linear Models
         with s ∈ (0,1)" — Peng, Zhang, and Zeng (2026)
         Electronic Journal of Statistics

Install dependencies once:
    pip install numpy matplotlib scikit-learn


"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV

"""
    L^s linear regularised model — Algorithm 2 in Peng, Zhang & Zeng (2026).

    Parameters
    ----------
    X      : (n, d) ndarray  — design matrix
    Y      : (n,)   ndarray  — response vector
    s      : float           — penalty order, 0 < s < 1
    lam    : float           — regularisation parameter λ
    beta   : (d,)   ndarray  — initial coefficient vector
    nstep  : int             — maximum Newton iterations per coordinate
    a      : float           — Newton step-size scaling parameter

    Returns
    -------
    (d+1,) ndarray  — [intercept, β₁, …, β_d]
    """

def GLM(X, Y, s, lam, beta, nstep, a):
    d    = X.shape[1]
    beta = np.array(beta, dtype=float).flatten()

    for j in range(d):
        mu        = float(X[:, j] @ X[:, j])
        idx_other = np.r_[np.arange(j), np.arange(j + 1, d)]
        c_j       = float(X[:, j] @ (Y - X[:, idx_other] @ beta[idx_other])) / mu

        # Threshold: if penalty exceeds this, shrink coefficient to zero
        lam_crit = (
            mu * abs(c_j) / (2 - s)
            * (abs(2 - 2 * s) / (2 - s) * abs(c_j)) ** (1 - s)
        )

        if lam_crit <= lam:
            beta[j] = 0.0
        else:
            # Newton's method with scaled step
            def g1(x, _mu=mu, _c=c_j, _lam=lam, _s=s):
                return _mu * (x - _c) + _lam * _s * np.sign(x) * abs(x) ** (_s - 1)

            def g2(x, _mu=mu, _lam=lam, _s=s):
                return _mu + _lam * _s * (_s - 1) * abs(x) ** (_s - 2)

            x = float(c_j)
            for _ in range(nstep):
                step = a * g1(x) / g2(x)
                if abs(step) < 1e-5:
                    break
                x -= step
            beta[j] = x

    beta0 = float(np.mean(Y - X @ beta))
    return np.concatenate([[beta0], beta])


# ──────────────────────────────────────────────────────────────────────────────
#  CV Tuner for s and λ
# ──────────────────────────────────────────────────────────────────────────────

"""
    Tune s and λ for L^s regularisation via a 75/25 hold-out split.

    Searches over:
        s ∈ {1/(S+1), 2/(S+1), …, S/(S+1)}   (S values)
        λ ∈ {1/L,     2/L,     …, 1}           (L values)

    Returns
    -------
    [s_opt, λ_opt] as a length-2 ndarray
    """

def tune_s_lambda(X, Y, S, L, beta, nstep, a, seed=None):
    R_mat = np.full((S, L), 1e8)
    n     = X.shape[0]
    Z     = np.column_stack([np.ones(n), X])   # augmented with intercept column

    # Replicate R's sample() with set.seed(seed)
    rng   = np.random.default_rng(seed)
    train = rng.choice(n, int(n * 3 / 4), replace=False)
    test  = np.setdiff1d(np.arange(n), train)

    s_grid = np.arange(1, S + 1) / (S + 1)
    l_grid = np.arange(1, L + 1) / L

    for si, s_val in enumerate(s_grid):
        for li, l_val in enumerate(l_grid):
            b          = GLM(X[train], Y[train], s_val, l_val, beta, nstep, a)
            R_mat[si, li] = np.mean((Y[test] - Z[test] @ b) ** 2)

    best_si, best_li = np.unravel_index(np.argmin(R_mat), R_mat.shape)
    return np.array([(best_si + 1) / (S + 1), (best_li + 1) / L])


# ──────────────────────────────────────────────────────────────────────────────
#  Helper: Lasso with CV  (mirrors cv.glmnet + glmnet in R)
# ──────────────────────────────────────────────────────────────────────────────

"""
    Equivalent to R's:
        cv_model   = cv.glmnet(X, Y, alpha=1)
        best_model = glmnet(X, Y, alpha=1, lambda=cv_model$lambda.min)
        coef(best_model)

    Returns length-(d+1) array [intercept, β₁, …, β_d].
    """

def run_lasso_cv(X, Y, seed=3):
    cv = LassoCV(
        cv=10,
        fit_intercept=True,
        max_iter=20_000,
        alphas=100,
        random_state=seed,
    )
    cv.fit(X, Y)
    # LassoCV is itself the fitted model at the best alpha — no second fit needed
    return np.concatenate([[cv.intercept_], cv.coef_])


# ──────────────────────────────────────────────────────────────────────────────
#  Shared plot helper
# ──────────────────────────────────────────────────────────────────────────────

def _save_show(fig, filename):
    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"  Saved → {filename}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE 1.1
#  Variable selection — SIC FAILS  (β₁=2, β₂=3, β₃=0)
#  Figures 1.1.1 (Lasso) and 1.1.2 (L^s)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Example 1.1  —  SIC fails  (β=[2,3,0])")
print("⚠  Intensive: tune_s_lambda on 100×100 grid × 181 sample sizes.")
print("   Reduce S=L to e.g. 10 for a quick test.")
print("=" * 60)

beta1 = np.array([2.0, 3.0])
beta  = np.append(beta1, 0.0)   # initial guess [2, 3, 0]
nstep = 100
S = L = 100
a     = 1.0

n_start, n_end = 20, 200
N = np.arange(1, n_end - n_start + 2)          # 1..181  (x-axis, as in R)

bhat_11       = np.zeros((n_end - n_start + 1, 4))
bhat_lasso_11 = np.zeros((n_end - n_start + 1, 4))
SIC_11        = None

for n in range(n_start, n_end + 1):
    i = n - n_start

    # ── Generate X  (seed 1 resets each iteration, matching R's set.seed(1)) ──
    np.random.seed(1)
    X1 = np.column_stack([np.random.normal(0, 1, n),
                           np.random.normal(0, 1, n)])
    X2 = ((2 / 3) * (X1[:, 0] + X1[:, 1])).reshape(-1, 1)
    X  = np.column_stack([X1, X2])

    C11   = (X1.T @ X1) / n
    C21   = (X2.T @ X1) / n
    SIC_11 = abs((C21 @ np.linalg.solve(C11, np.eye(2))) @ np.sign(beta1))

    # ── Generate noise  (seed 2) ──────────────────────────────────────────────
    np.random.seed(2)
    e = np.random.normal(0, 1, n)
    Y = X @ beta + e

    # ── Lasso with CV  (seed 3 for fold shuffle) ─────────────────────────────
    out_lasso = run_lasso_cv(X, Y, seed=3)
    bhat_lasso_11[i, :] = out_lasso[:4]

    # ── L^s regularisation  (seed 4 for train/test split) ────────────────────
    tune       = tune_s_lambda(X, Y, S, L, beta, nstep, a, seed=4)
    s_opt, lam_opt = tune
    out        = GLM(X, Y, s_opt, lam_opt, beta, nstep, a)
    bhat_11[i, :] = out[:4]

    if (n - n_start) % 20 == 0:
        print(f"  n={n:3d}  s_opt={s_opt:.3f}  λ_opt={lam_opt:.4f}")

print(f"\nSIC (n={n_end}): {SIC_11}")

# Figure 1.1.1 — Lasso estimates
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(N, bhat_lasso_11[:, 3], color='red',   s=8,
           label=r'$\beta_3$', zorder=3)
ax.plot(N,    bhat_lasso_11[:, 1], color='blue',  linestyle='--',
        label=r'$\beta_1$')
ax.plot(N,    bhat_lasso_11[:, 2], color='black', linestyle='--',
        label=r'$\beta_2$')
ax.axhline(2, color='blue',  linewidth=0.8)
ax.axhline(0, color='red',   linewidth=0.8)
ax.axhline(3, color='black', linewidth=0.8)
ax.set_ylim(-2, 4)
ax.set_xlabel('Sample Size n')
ax.set_ylabel(r'$\beta_i$')
ax.set_title(r'Estimates of $\beta_i$ from the Lasso')
ax.legend(loc='lower left')
_save_show(fig, 'figure_1_1_1_lasso.png')

# Figure 1.1.2 — L^s estimates
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(N, bhat_11[:, 3], color='red',   s=8,
           label=r'$\beta_3$', zorder=3)
ax.plot(N,    bhat_11[:, 1], color='blue',  linestyle='--',
        label=r'$\beta_1$')
ax.plot(N,    bhat_11[:, 2], color='black', linestyle='--',
        label=r'$\beta_2$')
ax.axhline(2, color='blue',  linewidth=0.8)
ax.axhline(0, color='red',   linewidth=0.8)
ax.axhline(3, color='black', linewidth=0.8)
ax.set_ylim(-2, 4)
ax.set_xlabel('Sample Size n')
ax.set_ylabel(r'$\beta_i$')
ax.set_title(r'Estimates of $\beta_i$ from $\ell^s$-regularization')
ax.legend(loc='lower left')
_save_show(fig, 'figure_1_1_2_ls.png')


# ══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE 1.2
#  Variable selection — SIC HOLDS  (β₁=-2, β₂=3, β₃=0)
#  Figures 1.2.1 (Lasso) and 1.2.2 (L^s)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Example 1.2  —  SIC holds  (β=[-2,3,0])")
print("⚠  Same runtime caveat as Example 1.1.")
print("=" * 60)

beta1 = np.array([-2.0, 3.0])
beta  = np.append(beta1, 0.0)
nstep = 100
S = L = 100
a     = 1.0

n_start, n_end = 20, 200
N = np.arange(1, n_end - n_start + 2)

bhat_12       = np.zeros((n_end - n_start + 1, 4))
bhat_lasso_12 = np.zeros((n_end - n_start + 1, 4))
SIC_12        = None

for n in range(n_start, n_end + 1):
    i = n - n_start

    np.random.seed(1)
    X1 = np.column_stack([np.random.normal(0, 1, n),
                           np.random.normal(0, 1, n)])
    X2 = ((2 / 3) * (X1[:, 0] + X1[:, 1])).reshape(-1, 1)
    X  = np.column_stack([X1, X2])

    C11    = (X1.T @ X1) / n
    C21    = (X2.T @ X1) / n
    SIC_12 = abs((C21 @ np.linalg.solve(C11, np.eye(2))) @ np.sign(beta1))

    np.random.seed(2)
    e = np.random.normal(0, 1, n)
    Y = X @ beta + e

    out_lasso = run_lasso_cv(X, Y, seed=3)
    bhat_lasso_12[i, :] = out_lasso[:4]

    tune       = tune_s_lambda(X, Y, S, L, beta, nstep, a, seed=4)
    s_opt, lam_opt = tune
    out        = GLM(X, Y, s_opt, lam_opt, beta, nstep, a)
    bhat_12[i, :] = out[:4]

    if (n - n_start) % 20 == 0:
        print(f"  n={n:3d}  s_opt={s_opt:.3f}  λ_opt={lam_opt:.4f}")

print(f"\nSIC (n={n_end}): {SIC_12}")

# Figure 1.2.1 — Lasso
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(N, bhat_lasso_12[:, 3], color='red',   s=8,
           label=r'$\beta_3$', zorder=3)
ax.plot(N,    bhat_lasso_12[:, 1], color='blue',  linestyle='--',
        label=r'$\beta_1$')
ax.plot(N,    bhat_lasso_12[:, 2], color='black', linestyle='--',
        label=r'$\beta_2$')
ax.axhline(-2, color='blue',  linewidth=0.8)
ax.axhline( 0, color='red',   linewidth=0.8)
ax.axhline( 3, color='black', linewidth=0.8)
ax.set_ylim(-6, 4)
ax.set_xlabel('Sample Size n')
ax.set_ylabel(r'$\beta_i$')
ax.set_title(r'Estimates of $\beta_i$ from the Lasso')
ax.legend(loc='lower left')
_save_show(fig, 'figure_1_2_1_lasso.png')

# Figure 1.2.2 — L^s
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(N, bhat_12[:, 3], color='red',   s=8,
           label=r'$\beta_3$', zorder=3)
ax.plot(N,    bhat_12[:, 1], color='blue',  linestyle='--',
        label=r'$\beta_1$')
ax.plot(N,    bhat_12[:, 2], color='black', linestyle='--',
        label=r'$\beta_2$')
ax.axhline(-2, color='blue',  linewidth=0.8)
ax.axhline( 0, color='red',   linewidth=0.8)
ax.axhline( 3, color='black', linewidth=0.8)
ax.set_ylim(-6, 4)
ax.set_xlabel('Sample Size n')
ax.set_ylabel(r'$\beta_i$')
ax.set_title(r'Estimates of $\beta_i$ from $\ell^s$-regularization')
ax.legend(loc='lower left')
_save_show(fig, 'figure_1_2_2_ls.png')


# ══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE 2.1
#  Solution paths vs ‖β‖_s — SIC FAILS  (β₁=2, β₂=3, β₃=0, n=1000)
#  Figures 2.1.1 (L^s) and 2.1.2 (Lasso)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Example 2.1  —  Solution paths, SIC fails  (β=[2,3,0], n=1000)")
print("=" * 60)

beta1 = np.array([2.0, 3.0])
beta  = np.append(beta1, 0.0)
nstep = 100
a     = 1.0
n     = 1000

bhat_21           = np.zeros((n, 4))
beta_norm_21      = np.zeros(n)
bhat_lasso_21     = np.zeros((n, 4))
beta_norm_lasso21 = np.zeros(n)

# Generate X and Y once (fixed dataset)
np.random.seed(1)
X1 = np.column_stack([np.random.normal(0, 1, n),
                       np.random.normal(0, 1, n)])
X2 = ((2 / 3) * (X1[:, 0] + X1[:, 1])).reshape(-1, 1)
X  = np.column_stack([X1, X2])

C11    = (X1.T @ X1) / n
C21    = (X2.T @ X1) / n
SIC_21 = abs((C21 @ np.linalg.solve(C11, np.eye(2))) @ np.sign(beta1))

np.random.seed(2)
e = np.random.normal(0, 1, n)
Y = X @ beta + e

# Fix s for solution path
s = 0.9

for m in range(1, n + 1):
    # Lasso: sweep λ linearly
    lam_lasso = 0.001 * m
    mdl       = Lasso(alpha=lam_lasso, fit_intercept=True, max_iter=20_000)
    mdl.fit(X, Y)
    out_lasso = np.concatenate([[mdl.intercept_], mdl.coef_])

    # L^s: sweep λ exponentially (large → small regularisation → dense → sparse)
    lam_ls = np.exp(m)
    out    = GLM(X, Y, s, lam_ls, beta, nstep, a)

    beta_norm_21[m - 1]      = (np.sum(np.abs(out[1:4]) ** s)) ** (1 / s)
    beta_norm_lasso21[m - 1] = np.sum(np.abs(out_lasso[1:4]))

    bhat_21[m - 1, :]     = out[:4]
    bhat_lasso_21[m - 1, :] = out_lasso[:4]

print(f"  SIC: {SIC_21}")

# Sort by ascending norm (solution path order)
ord_ls    = np.argsort(beta_norm_21)
ord_lasso = np.argsort(beta_norm_lasso21)

norm_ls_sorted    = beta_norm_21[ord_ls]
norm_lasso_sorted = beta_norm_lasso21[ord_lasso]

bhat_21_sorted       = bhat_21[ord_ls, :]
bhat_lasso_21_sorted = bhat_lasso_21[ord_lasso, :]

# Figure 2.1.1 — L^s solution path
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(norm_ls_sorted, bhat_21_sorted[:, 3], color='blue',  linestyle='--',
        label=r'$\beta_3$')
ax.plot(norm_ls_sorted, bhat_21_sorted[:, 1], color='red',   linestyle='--',
        label=r'$\beta_1$')
ax.plot(norm_ls_sorted, bhat_21_sorted[:, 2], color='black', linestyle='--',
        label=r'$\beta_2$')
ax.set_ylim(-3.5, 3.5)
ax.set_xlabel(r'$\|\beta\|_s$')
ax.set_ylabel(r'Estimate of $\beta_i$')
ax.set_title(r'Solution Paths of $L^s$-Linear Regularization against $\|\beta\|_s$')
ax.legend(loc='lower left')
_save_show(fig, 'figure_2_1_1_ls.png')

# Figure 2.1.2 — Lasso solution path
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(norm_lasso_sorted, bhat_lasso_21_sorted[:, 3], color='blue',  linestyle='--',
        label=r'$\beta_3$')
ax.plot(norm_lasso_sorted, bhat_lasso_21_sorted[:, 1], color='red',   linestyle='--',
        label=r'$\beta_1$')
ax.plot(norm_lasso_sorted, bhat_lasso_21_sorted[:, 2], color='black', linestyle='--',
        label=r'$\beta_2$')
ax.set_ylim(-3.5, 3.5)
ax.set_xlabel(r'$\|\beta\|_1$')
ax.set_ylabel(r'Estimate of $\beta_i$')
ax.set_title(r'Solution Paths of Lasso against $\|\beta\|_1$')
ax.legend(loc='lower left')
_save_show(fig, 'figure_2_1_2_lasso.png')


# ══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE 2.2
#  Solution paths vs ‖β‖_s — SIC HOLDS  (β₁=-2, β₂=3, β₃=0, n=1000)
#  Figures 2.2.1 (L^s) and 2.2.2 (Lasso)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Example 2.2  —  Solution paths, SIC holds  (β=[-2,3,0], n=1000)")
print("=" * 60)

beta1 = np.array([-2.0, 3.0])
beta  = np.append(beta1, 0.0)
nstep = 100
a     = 1.0
n     = 1000

bhat_22           = np.zeros((n, 4))
beta_norm_22      = np.zeros(n)
bhat_lasso_22     = np.zeros((n, 4))
beta_norm_lasso22 = np.zeros(n)

np.random.seed(1)
X1 = np.column_stack([np.random.normal(0, 1, n),
                       np.random.normal(0, 1, n)])
X2 = ((2 / 3) * (X1[:, 0] + X1[:, 1])).reshape(-1, 1)
X  = np.column_stack([X1, X2])

C11    = (X1.T @ X1) / n
C21    = (X2.T @ X1) / n
SIC_22 = abs((C21 @ np.linalg.solve(C11, np.eye(2))) @ np.sign(beta1))

np.random.seed(2)
e = np.random.normal(0, 1, n)
Y = X @ beta + e

s = 0.9

for m in range(1, n + 1):
    lam_lasso = 0.001 * m
    mdl       = Lasso(alpha=lam_lasso, fit_intercept=True, max_iter=20_000)
    mdl.fit(X, Y)
    out_lasso = np.concatenate([[mdl.intercept_], mdl.coef_])

    lam_ls = np.exp(m)
    out    = GLM(X, Y, s, lam_ls, beta, nstep, a)

    beta_norm_22[m - 1]      = (np.sum(np.abs(out[1:4]) ** s)) ** (1 / s)
    beta_norm_lasso22[m - 1] = np.sum(np.abs(out_lasso[1:4]))

    bhat_22[m - 1, :]     = out[:4]
    bhat_lasso_22[m - 1, :] = out_lasso[:4]

print(f"  SIC: {SIC_22}")

ord_ls    = np.argsort(beta_norm_22)
ord_lasso = np.argsort(beta_norm_lasso22)

norm_ls_sorted    = beta_norm_22[ord_ls]
norm_lasso_sorted = beta_norm_lasso22[ord_lasso]

bhat_22_sorted       = bhat_22[ord_ls, :]
bhat_lasso_22_sorted = bhat_lasso_22[ord_lasso, :]

# Figure 2.2.1 — L^s solution path
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(norm_ls_sorted, bhat_22_sorted[:, 3], color='blue',  linestyle='--',
        label=r'$\beta_3$')
ax.plot(norm_ls_sorted, bhat_22_sorted[:, 1], color='red',   linestyle='--',
        label=r'$\beta_1$')
ax.plot(norm_ls_sorted, bhat_22_sorted[:, 2], color='black', linestyle='--',
        label=r'$\beta_2$')
ax.set_ylim(-3.5, 3.5)
ax.set_xlabel(r'$\|\beta\|_s$')
ax.set_ylabel(r'Estimate of $\beta_i$')
ax.set_title(r'Solution Paths of $L^s$-Linear Regularization against $\|\beta\|_s$')
ax.legend(loc='lower left')
_save_show(fig, 'figure_2_2_1_ls.png')

# Figure 2.2.2 — Lasso solution path
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(norm_lasso_sorted, bhat_lasso_22_sorted[:, 3], color='blue',  linestyle='--',
        label=r'$\beta_3$')
ax.plot(norm_lasso_sorted, bhat_lasso_22_sorted[:, 1], color='red',   linestyle='--',
        label=r'$\beta_1$')
ax.plot(norm_lasso_sorted, bhat_lasso_22_sorted[:, 2], color='black', linestyle='--',
        label=r'$\beta_2$')
ax.set_ylim(-3.5, 3.5)
ax.set_xlabel(r'$\|\beta\|_1$')
ax.set_ylabel(r'Estimate of $\beta_i$')
ax.set_title(r'Solution Paths of Lasso against $\|\beta\|_1$')
ax.legend(loc='lower left')
_save_show(fig, 'figure_2_2_2_lasso.png')

print("\nDone. All 8 figures saved to the working directory.")
