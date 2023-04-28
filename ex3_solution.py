# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:39:38 2023

@author: Tobias Kallehauge
"""

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
fn = 'data/temp_anomaly_ex3.nc'
ds = nc.Dataset(fn)
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

X = np.array(ds['tempanomaly'])
X[X == 2**15 - 1] = np.nan # set missing values to nan
# flatten array
N = X.shape[0]
X = X.reshape(N,100)


# split into training validation and test
N_val = 100
N_tst = 12
N_trn = N - N_val - N_tst
X_trn = X[:N_trn]
X_val = X[N_trn: N_trn + N_val]
X_tst = X[-N_tst:]

# =============================================================================
# About the solution
# =============================================================================

"""
The solution is purely spatial. It estimates the covariances between each point
in the grid and then uses the best linear predictior based on estimated mean
and covariance.  Note that temporal dependencies is not accounted for, however
analysing the data, the spatial correlation is close to 1, so modelling 
temporal dependencies would properly give only minor improvements. 
"""


# =============================================================================
# Infer mean and covariance parameters
# =============================================================================

mu = np.mean(X_trn, axis = 0)
C = np.cov(X_trn.T)

# =============================================================================
# Extract mean and coviariance parameters
# =============================================================================

# # setup unknown locations as mask
idx_pred = np.where(np.isnan(X_tst[0]))[0] # which index in flattened array is unknown
mask = np.ones(100, dtype = bool)
mask[idx_pred] = False    
N_pred = len(idx_pred)

# Y are the unknown locations and X are the known locations

# get means
mu_Y = mu[idx_pred]
mu_X = mu[mask]

# # get covairance parameters
C_YX = [C[i, mask] for i in idx_pred]
C_XX = C[mask][:,mask]
C_YY = [C[i,i] for i in idx_pred]

# get predictve variance (does not depend on data)
var_Y_cond = [C_YY[i] - C_YX[i] @ np.linalg.inv(C_XX) @ C_YX[i] for i in range(N_pred)]

# # =============================================================================
# # Predict on validation data using projection theorem for multivariate normal
# # =============================================================================

mu_Y_cond = np.zeros((N_pred, N_val))
for j in range(N_pred):
    for i in range(N_val):
        mu_Y_cond[j,i] = mu_Y[j] + (C_YX[j] @ np.linalg.inv(C_XX)) @ (X_val[i,mask] - mu_X)
    

# compare prediction with reald data
for i in range(N_pred):
    plt.fill_between(np.arange(N_val),
                      mu_Y_cond[i] - 1.96*np.sqrt(var_Y_cond[i]), 
                      mu_Y_cond[i] + 1.96*np.sqrt(var_Y_cond[i]),
                      alpha =1, color= 'r')
    plt.plot(mu_Y_cond[i], 'o-')
    plt.plot(X_val[:,idx_pred[i]])
    plt.show()

# =============================================================================
# Predict on test data
# =============================================================================

mu_Y_cond = np.zeros((N_pred,N_tst))
for j in range(N_pred):
    for i in range(N_tst):
        mu_Y_cond[j,i] = mu_Y[j] + (C_YX[j] @ np.linalg.inv(C_XX)) @ (X_tst[i,mask] - mu_X)
        

# =============================================================================
# save output in as cvs (only prediction at (6,6)
# =============================================================================

idx_66 = np.where(idx_pred == 55)[0][0] # of the four predictions

pred = pd.DataFrame({'Prediction at (6,6)': mu_Y_cond[idx_66]})
pred.index = np.arange(N_trn + N_val + 1, N + 1)
pred.index.name = 'Index'
pred.to_csv('results/ex3_prediction.csv')
