import numpy as np
import cvxopt as opt
from cvxopt import matrix, spmatrix, sparse
from cvxopt.solvers import qp, options
from cvxopt import blas
import numpy as np
import pandas as pd
import json
import util
options['show_progress'] = False

regions = util.regions
path = util.path

global_forecast = util.load_forecast(path+f"agg_experts.csv")

region_forcasts = {}
for region in regions:
    region_forcasts[region] = util.load_forecast(path+f"{region}-fcast.csv")

all_forecasts = pd.concat([global_forecast] + [region_forcasts[region] for region in regions], axis=1)
all_forecasts.dropna(inplace=True)
Y = all_forecasts.values.T
m = len(regions)
N = len(all_forecasts)

with open(path + 'boundaries.txt') as json_file:
    region_boundaries = json.load(json_file)

def ols_reconciliation(Y, quantile=None):
    S = np.ones((m,))
    S = np.vstack((S, np.eye(m)))

    P = np.linalg.inv((S.T @ S)) @ S.T

    Y_new = S @ P @ Y

    return Y_new

def bottom_up_reconciliation(Y, quantile=None):
    S = np.ones((m,))
    S = np.vstack((S, np.eye(m)))

    P_bottom_up = np.hstack((np.zeros((m,1)), np.eye(m)))
    Y_new = S @ P_bottom_up @ Y

    return Y_new

def gtop_reconciliation(Y, quantile=10):
    q = quantile // 10 - 1
    w_t = m
    a = 1
    w_i = np.ones((m,)) * a
    boundaries = np.ones((m,)) 
    Y_new = np.zeros_like(Y)

    for i, region_forcast in enumerate(region_forcasts):
        boundaries[i] = region_boundaries[region_forcast][q]

    for i in range(N):
        Y_ = Y[:, i]
        A2 = np.diag(np.insert(w_i, 0, w_t, axis=0))
        Q = matrix(A2)
        r = matrix(2 * A2 @ Y_)

        A = np.ones((1,m+1))
        A[0, 0] = -1
        A = matrix(A)
        b = matrix(0.0)
        
        G_up = np.hstack((np.zeros((m,1)), np.eye(m)))
        G_down = np.hstack((np.zeros((m,1)), -np.eye(m)))
        G = np.vstack((G_up, G_down))
        h = G @ Y_ + np.concatenate((boundaries, boundaries))
        h = matrix(h)
        G = matrix(G)

        # Solve and retrieve solution
        sol = qp(Q, r, G, h, A, b)['x']
        Y_new[:, i] = np.array(sol.T)[0]
    return Y_new

def reconciliate(reconciliation_method, quantile=None):
    Y_new = reconciliation_method(Y, quantile)
    region_corrected = {}
    for i, region_forcast in enumerate(region_forcasts):
        region_corrected[region_forcast] = pd.Series(Y_new[i+1]).set_axis(global_forecast.index)

    new_gb_forecast = pd.Series(Y_new[0]).set_axis(global_forecast.index)
    return new_gb_forecast, region_corrected
