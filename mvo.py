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

def optimize_portfolio(mu, cov, lamda=1.0, max_weight=0.3):
    '''
    Compute optimal portfolio weights using MVO

    Params:
        mu (np.array): Expected returns for each asset
        cov (np.array): Covariance matrix of asset returns
        lamda (float): Risk aversion parameter. Default 1.0 balances return
            and variance equally on a normalised scale; increase to penalise
            risk more heavily, decrease to chase higher expected return.
        max_weight (float): Maximum weight allowed for any single asset.
            Default 0.3 (30%) enforces diversification and prevents the
            optimiser from concentrating the portfolio in one or two assets,
            which commonly occurs with unconstrained MVO due to estimation
            error in mu and cov.

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
    bounds = tuple((0, max_weight) for _ in range(n))

    # optimize
    result = minimize(objective, w0, args=(mu, cov, lamda), method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        return w0
    
    return result.x