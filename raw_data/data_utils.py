#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import xarray as xr

import os
from pathlib import Path
from collections import defaultdict
import scipy
import random
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import joblib
#import sklearn
#from skimage.filters import sobel

def import_model_data(ens, member, date_start, date_end):
    
    base_url = '/home/julias/MLEE-final-project/raw_data/{}_2D_mon_{}{}_1x1_198201-201701.nc'
    XCO2_url = '/home/julias/MLEE-final-project/raw_data/{}_1D_mon_{}{}_native_198201-201701.nc'
    mask_path = '/home/julias/MLEE-final-project/raw_data/create_mask.nc'

    variables = {'Chl','FG-CO2','iceFrac','MLD','pATM','pCO2','SSS','SST','U10','XCO2'}

    full_potential_variables_list = []
    inputs = {}
    inputs['socat_mask'] = xr.open_dataset(mask_path).socat_mask.sel(time=slice(date_start,date_end))
    inputs['net_mask'] = xr.open_dataset(mask_path).net_mask.sel(time=slice(date_start,date_end))
#time = inputs['socat_mask']

    for variable in variables:

        if variable=='Chl':    
            Chl_path = base_url.format(variable,ens,member)
            inputs['Chl'] = xr.open_dataset(Chl_path).Chl.sel(time=slice(date_start,date_end)).assign_coords(time=inputs['socat_mask'].time)
            full_potential_variables_list.append(Chl_path)
        if variable=='MLD':    
            MLD_path = base_url.format(variable,ens,member)
            inputs['MLD'] = xr.open_dataset(MLD_path).MLD.sel(time=slice(date_start,date_end)).assign_coords(time=inputs['socat_mask'].time)
            full_potential_variables_list.append(MLD_path)
        if variable=='pATM':    
            pATM_path = base_url.format(variable,ens,member)
            inputs['pATM'] = xr.open_dataset(pATM_path).pATM.sel(time=slice(date_start,date_end)).assign_coords(time=inputs['socat_mask'].time)
            full_potential_variables_list.append(pATM_path)
        if variable=='pCO2':    
            pCO2_path = base_url.format(variable,ens,member)
            inputs['pCO2'] = xr.open_dataset(pCO2_path).pCO2.sel(time=slice(date_start,date_end)).assign_coords(time=inputs['socat_mask'].time)
            full_potential_variables_list.append(pCO2_path)
        if variable=='SSS':    
            SSS_path = base_url.format(variable,ens,member)
            inputs['SSS'] = xr.open_dataset(SSS_path).SSS.sel(time=slice(date_start,date_end)).assign_coords(time=inputs['socat_mask'].time)    
            full_potential_variables_list.append(SSS_path)
        if variable=='SST':    
            SST_path = base_url.format(variable,ens,member)
            inputs['SST'] = xr.open_dataset(SST_path).SST.sel(time=slice(date_start,date_end)).assign_coords(time=inputs['socat_mask'].time)
            full_potential_variables_list.append(SST_path)
        
    # Note that XCO2 requires a slightly different URL
        if variable=='XCO2':
            XCO2_path = XCO2_url.format(variable,ens,member)
            inputs['XCO2'] = xr.open_dataset(XCO2_path).XCO2.sel(time=slice(date_start,date_end)).assign_coords(time=inputs['socat_mask'].time)
            full_potential_variables_list.append(XCO2_path)
        
        if variable=='FG-CO2' or variable=='iceFrac' or variable=='U10':
            full_potential_variables_list.append(base_url.format(variable,ens,member))
    
    for i in inputs:        
        if i != 'XCO2' and i != 'socat_mask' and i != 'net_mask':
            inputs[i] = inputs[i].assign_coords(xlon=(((inputs[i].xlon+180)%360)-180)) # wraps from 0 to 360 to -180 to 180
            inputs[i] = inputs[i].sortby(inputs[i].xlon)    
        
    DS = xr.merge([inputs['SSS'], inputs['SST'], inputs['MLD'], inputs['Chl'], inputs['pCO2'], inputs['socat_mask'], inputs['net_mask']])
    
    DS.xlon.attrs['long_name'] = 'longitude'
    DS.xlon.attrs['standard_name'] = 'longitude'
    DS.xlon.attrs['units'] = 'degrees_east'

    return DS, inputs['XCO2'], full_potential_variables_list



##################################################################################################

def create_features(ds, ds_XCO2, N_time, N_batch): #NOTE maybe change time back to 421
    
    XCO2_full = np.repeat(ds_XCO2.values, 360*180)
    test_XCO2 = np.reshape(np.ravel(XCO2_full), (420, 180, 360))
    ds['XCO2'] = ds['SSS']
    ds['XCO2'].values = test_XCO2
    
    days_idx = ds['time'].dt.dayofyear #df.index.get_level_values('time').dayofyear
    ds['T0'], ds['T1'] = [np.cos(days_idx * 2 * np.pi / 365), np.sin(days_idx * 2 * np.pi / 365)]
    
    
    lon_rad = np.radians(ds.xlon)
    lat_rad = np.radians(ds.ylat)
    ds['A'], ds['B'], ds['C'] = [np.sin(lat_rad), np.sin(lon_rad)*np.cos(lat_rad), -np.cos(lon_rad)*np.cos(lat_rad)]
    
    return ds
