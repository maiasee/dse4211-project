import numpy as np
from scipy.optimize import minimize

def portfolio_return(w, mu):
    return np.dot(w, mu)

def portfolio_variance(w, cov):
    return np.dot(w.T, np.dot(cov, w))

def objective(w, nu, cov, lamda):
    return -(portfolio_return(w, nu) - lamda * portfolio_variance(w, cov) )

def optimize_portfolio(mu, cov, lamda=1.0, max_weight=0.3):
    mu = np.array(mu)
    cov = np.array(cov)

    n = len(mu)
    w0 = np.ones(n) / n
    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # weights sum to 1
    )
    bounds = tuple((0, max_weight) for _ in range(n))
    result = minimize(objective, w0, args=(mu, cov, lamda), method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        return w0
    return result.x