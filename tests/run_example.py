from functools import partial

import numpy as np
import pandas as pd
from pounders.solve import solve_pounders


data = pd.read_csv("example_data/data.csv")
endog = np.asarray(data["y"])
exog = np.asarray(data["t"])


def func(x: np.ndarray, exog: np.ndarray, endog: np.ndarray) -> np.ndarray:
    """User provided residual function."""
    return endog - np.exp(-x[0] * exog) / (x[1] + x[2] * exog)


f = partial(func, exog=exog, endog=endog)

x0 = np.array([0.15, 0.008, 0.01])
nobs = exog.shape[0]

# Default params
delta = 0.1
delta_min = 1e-6
delta_max = 1e3
gamma0 = 0.5
gamma1 = 2.0
theta1 = 1e-5
theta2 = 1e-4
eta0 = 0.0
eta1 = 0.1
c1 = np.sqrt(x0.shape[0])
c2 = 10
gtol_sub = 1e-8
maxiter = 200

solution, gradient = solve_pounders(
    x0=x0,
    nobs=nobs,
    criterion=f,
    delta=delta,
    delta_min=delta_min,
    delta_max=delta_max,
    gamma0=gamma0,
    gamma1=gamma1,
    theta1=theta1,
    theta2=theta2,
    eta0=eta0,
    eta1=eta1,
    c1=c1,
    c2=c2,
    gtol_sub=gtol_sub,
    maxiter=maxiter,
)
