# libraries
import os
from pathlib import Path
from collections import defaultdict
import scipy
import random
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sklearn
# from skimage.filters import sobel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, max_error, mean_squared_error, mean_absolute_error, median_absolute_error
import keras
from keras import Sequential, regularizers
from keras.layers import Dense, BatchNormalization, Dropout
from statsmodels.nonparametric.smoothers_lowess import lowess
from glob import glob

#===============================================
# NN functions
#===============================================

def build_nn(num_features, neurons=[512,256], act='relu', use_drop=True, drop_rate=0.5, learning_rate=0.01, reg=0.001):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(num_features,)))
    for i in range(len(neurons)):
        model.add(Dense(units=neurons[i], activation=act, kernel_regularizer=regularizers.l2(reg)))
        if use_drop:
            model.add(Dropout(drop_rate))
    model.add(Dense(units=1))

    model.compile(keras.optimizers.Adam(lr=learning_rate), loss='mse', metrics=['mse'])

    return model

def build_nn_vf(num_features, act='relu', learning_rate=0.01, reg=0.001):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(num_features,)))
    model.add(Dense(units=500, activation=act, kernel_regularizer=regularizers.l2(reg)))
    model.add(Dense(units=500, activation=act, kernel_regularizer=regularizers.l2(reg)))
    model.add(Dense(units=1))

    model.compile(keras.optimizers.Adam(lr=learning_rate), loss='mse', metrics=['mse'])

    return model

#===============================================
# Evaluation functions
#===============================================

def centered_rmse(y,pred):
    y_mean = np.mean(y)
    pred_mean = np.mean(pred)
    return np.sqrt(np.square((pred - pred_mean) - (y - y_mean)).sum()/pred.size)

def evaluate_test(y, pred):
    scores = {
        'mse':mean_squared_error(y, pred),
        'mae':mean_absolute_error(y, pred),
        'medae':median_absolute_error(y, pred),
        'max_error':max_error(y, pred),
        'bias':pred.mean() - y.mean(),
        'r2':r2_score(y, pred),
        'corr':np.corrcoef(y,pred)[0,1],
        'cent_rmse':centered_rmse(y,pred),
        'stdev' :np.std(pred),
        'amp_ratio':(np.max(pred)-np.min(pred))/(np.max(y)-np.min(y)), # added when doing temporal decomposition
        'stdev_ref':np.std(y),
        'range_ref':np.max(y)-np.min(y),
        'iqr_ref':np.subtract(*np.percentile(y, [75, 25]))
        }
    return scores

#===============================================
# Saving functions
#===============================================

#def save_clean_data(df, data_output_dir, ens, member):
#    print("Starting data saving process")
#    output_dir = f"{data_output_dir}/{ens}/member_{member}"
#    Path(output_dir).mkdir(parents=True, exist_ok=True)
#    fname = f"{output_dir}/data_clean_2D_mon_{ens}_{member}_1x1_198201-201701.pkl"
#    df.to_pickle(fname)
#    print("Save complete")

def save_model(model, model_output_dir, approach, ens, member, run=None):
    print("Starting model saving process")
    model_dir = f"{model_output_dir}/{approach}/{ens}/member_{member}"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    if approach == 'nn':
        if run is None:
            run = 0
        model_fname = f"{model_dir}/{approach}_model_pC02_2D_mon_{ens}_{member}_{run}_1x1_198201-201701.h5"
        model.save(model_fname)
    else:
        model_fname = f"{model_dir}/{approach}_model_pC02_2D_mon_{ens}_{member}_1x1_198201-201701.joblib"
        joblib.dump(model, model_fname)
    print("Save complete")

#def save_recon(DS_recon, recon_output_dir, approach, ens, member, run=None):
#    print("Starting reconstruction saving process")
#    recon_dir = f"{recon_output_dir}/{approach}/{ens}/member_{member}"
#    Path(recon_dir).mkdir(parents=True, exist_ok=True)
#    if approach == "nn":
#        if run is None:
#            run = 0
#        recon_fname = f"{recon_dir}/{approach}_recon_pC02_2D_mon_{ens}_{member}_{run}_1x1_198201-201701.nc"
#    else:
#        recon_fname = f"{recon_dir}/{approach}_recon_pC02_2D_mon_{ens}_{member}_1x1_198201-201701.nc"
#    DS_recon.to_netcdf(recon_fname)
#    print("Save complete")