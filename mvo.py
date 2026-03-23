import numpy as np
from scipy.optimize import minimize

def portfolio_return(w, mu):
    '''
    Compute portfolio expected return
    '''
    return np.dot(w, mu)

def portfolio_variance(w, cov):
    '''
    Compute portfolio variance
    '''
    return np.dot(w.T, np.dot(cov, w))

def objective(w, nu, cov, lamda):
    '''
    Objective function for MVO:
    Maximise = portfolio return - lambda * portfolio variance
    '''
    return -(portfolio_return(w, nu) - lamda * portfolio_variance(w, cov) )

def optimize_portfolio(mu, cov, lamda = 1.0):
    '''
    Compute optimal portfolio weights using MVO
    
    Params:
        mu (np.array): Expected returns for each asset
        cov (np.array): Covariance matrix of asset returns
        lamda (float): Risk aversion parameter

    Returns:
        w (np.array): Optimal portfolio weights
    '''

    mu = np.array(mu)
    cov = np.array(cov)

    n = len(mu)

    # initial guess (equal weights)
    w0 = np.ones(n) / n

    # constraints
    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # weights sum to 1
    )

    # bounds
    bounds = tuple((0, 0.3) for _ in range(n))  # max 30% allocation per asset -> need to justify!!!

    # optimize
    result = minimize(objective, w0, args=(mu, cov, lamda), method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        return w0
    
    return result.x