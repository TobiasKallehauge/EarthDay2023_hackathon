# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:39:38 2023

@author: Tobias Kallehauge
"""

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
fn = 'data/temp_anomaly_ex1.nc'
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
# Infer mean and covariance parameters
# =============================================================================

mu = np.mean(X_trn, axis = 0)
C = np.cov(X_trn.T)

# =============================================================================
# Extract mean and coviariance parameters
# =============================================================================

# setup unknown locations as mask
idx_pred = np.where(np.isnan(X_tst))[1][0] # which index in flattened array is unknown
mask = np.ones(100, dtype = bool)
mask[idx_pred] = False    

# Y are the unknown locations and X are the known locations

# get means
mu_Y = mu[idx_pred]
mu_X = mu[mask]

# get covairance parameters
C_YX = C[idx_pred, mask]
C_XX = C[mask][:,mask]
C_YY = C[idx_pred,idx_pred]

# get predictve variance (does not depend on data)
var_Y_cond = C_YY - C_YX @ np.linalg.inv(C_XX) @ C_YX

# =============================================================================
# Predict on validation data using projection theorem for multivariate normal
# =============================================================================

mu_Y_cond = np.zeros(N_val)
for i in range(N_val):
    mu_Y_cond[i] = mu_Y + (C_YX @ np.linalg.inv(C_XX)) @ (X_val[i,mask] - mu_X)
    

# compare prediction with reald data
plt.fill_between(np.arange(N_val),
                 mu_Y_cond - 1.96*np.sqrt(var_Y_cond), 
                 mu_Y_cond + 1.96*np.sqrt(var_Y_cond),
                 alpha =1, color= 'r')
plt.plot(mu_Y_cond, 'o-')
plt.plot(X_val[:,idx_pred])

# =============================================================================
# Predict on test data
# =============================================================================

mu_Y_cond = np.zeros(N_tst)
for i in range(N_tst):
    mu_Y_cond[i] = mu_Y + (C_YX @ np.linalg.inv(C_XX)) @ (X_tst[i,mask] - mu_X)

# =============================================================================
# save output in as cvs
# =============================================================================

pred = pd.DataFrame({'Prediction at (6,6)': mu_Y_cond})
pred.index = np.arange(N_trn + N_val + 1, N + 1)
pred.index.name = 'Index'
pred.to_csv('results/ex1_prediction.csv')
