import cvxpy as cp
import numpy as np

import scipy.stats
import scipy.optimize


def MMD(M, w1, w2):
    """Calculates Maximum Mean Discrepancy between two discrete distributions
    w.r.t. kernel matrix M.
    Note that MMD^2(P, Q) = (P - Q)^T M (P - Q)
    """
    x = w1 - w2
    return float(np.sqrt(np.dot(x.T, np.dot(M, x))).flatten())

def worst_context_distribution_DRO(M, w_ref, UCB_x, epsilon):
    """Finds the worst distribution in ambiguity set with radius epsilon."""
    w = cp.Variable(w_ref.shape)
    x = w_ref - w

    eps_sqr = epsilon*epsilon
    
    cons1 = cp.sum(w) == 1
    cons2 = w >= 0
    cons3 = cp.quad_form(x, M, assume_PSD=True) <= eps_sqr

    constraints = [cons1, cons2, cons3]

    problem = cp.Problem(
        cp.Minimize(w.T @ UCB_x),
        constraints=constraints
    )

    if 'GUROBI' in cp.installed_solvers():
        problem.solve(solver="GUROBI")
    else:
        problem.solve()

    return w.value

def generate_pmf(n):
    """Generates a random PMF with support cardinality n."""
    cmf = np.concatenate(([0], np.sort(np.random.rand(n-1)), [1]))
    pmf = np.diff(cmf)
    return cmf.reshape(-1, 1), pmf.reshape(-1, 1)

def sample_index_from_p(p):
    """Returns the index of a random sample from p"""
    CDF_true = np.cumsum(p)
    x = np.random.rand()
    c = np.sum(CDF_true <= x)
    return c

def discretized_normal_distribution(discretization_pts, mean, var):
    """Creates a normal distribution on support discretization_pts."""
    P_ref = scipy.stats.norm.pdf((discretization_pts - mean) / np.sqrt(var))
    P_ref /= P_ref.sum()

    return P_ref.reshape(-1, 1)

def kappa_x_w(M, w_ref, w, UCB_x, tau, clip=True):
    """Calculates the expression inside the maximum operation of kappa;
    for specific x, w_ref and w.
    Clip parameter is for inspection when inifinities are present."""
    w = w.reshape(-1, 1)
    numerator = tau - np.dot(UCB_x.T, w)
    denominator = MMD(M, w_ref, w)

    tmp_kappa = numerator / (denominator + 1e-9)
    if clip:
        tmp_kappa = np.clip(tmp_kappa, -10, 10)
    return tmp_kappa

def w_x_t(M, w_ref, UCB_x, tau):
    """
    If given ucb_x, calculates w_bar_x_t.
    If given f_x instead, calculates w_doublebar_x_t.
    """
    
    # Since we want to maximize,
    # define lambda functions as negatives to minimize using SciPy.
    neg_kappa_x_w = lambda w: -kappa_x_w(M, w_ref, w, UCB_x, tau, clip=False)

    # Random starting point for solver.
    _, w0 = generate_pmf(w_ref.shape[0])

    # Inequalities to ensure w constructs a valid PMF.
    constraints = [
        {'type': 'ineq', 'fun': lambda w:  w},
        {'type': 'ineq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: 1 - np.sum(w)}

        # Below formulation is same as above.
        # The one above is used since COBYLA does not allow equalities.
        # {'type': 'ineq', 'fun': lambda w:  w},
        # {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]

    result = scipy.optimize.minimize(
        neg_kappa_x_w, w0.flatten(), method='COBYLA',
        constraints=constraints, options={"maxiter": 1000}  # "disp":True,
    )
    return result.x
