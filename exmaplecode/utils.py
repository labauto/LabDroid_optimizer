#!/usr/bin/python
#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd
#from StringIO import StringIO
from io import StringIO
from RPE_data import FGRi_conc_txt
import copy


X_names = [
        "Pre-con Conc.",
        "Pre-con Period",
        "Detach Period",
        "Detach Speed",
        "Detach Length",
        "KSR Conc.",
        "3i Period"
]
X_bounds = [
    (10., 510.), 
    (1., 6.),
    (0., 30.),
    (10., 100.),
    (0., 1.),
    (1., 19.),
    (3., 19.)
]

def get_space(normalize=True, logscale=["Detach Speed"]):
    bounds = copy.copy(X_bounds)
    space= {
         "Pre-con Conc.": np.array([float(c) for c in FGRi_conc_txt.split("\n")], dtype=np.float),
        "Pre-con Period": np.arange(1,6+1, dtype=np.float),
        "Detach Period": np.linspace(0, 30, 100),
        "Detach Speed": np.arange(10, 100+1, dtype=np.float),
        "Detach Length": np.array([0,1], dtype=np.float),
        "KSR Conc.": np.arange(1,19+1, dtype=np.float),
        "3i Period": np.arange(3, 19+1, dtype=np.float)
    }
    for k in logscale:
        space[k] = np.log10(space[k])
        i = X_names.index(k)
        bounds[i]=(np.log10(X_bounds[i][0]),np.log10(X_bounds[i][1]))
        
    if normalize:
        # normalize to [0,10]
        for i,k in enumerate(X_names):
            d = bounds[i][1] - bounds[i][0]
            if d > 0:
                space[k] = (space[k] - bounds[i][0]) / d
    
    return space

def get_space_forsum(normalize=True, logscale=["Detach Speed"]):
    bounds = copy.copy(X_bounds)
    space= {
         "Pre-con Conc.": np.array([float(c) for c in FGRi_conc_txt.split("\n")], dtype=np.float),
        "Pre-con Period": np.arange(1,6+1, dtype=np.float),
        "Detach Period": np.linspace(5, 26, 8),
        "Detach Speed": np.linspace(10, 100, 10),
        "Detach Length": np.array([0,1], dtype=np.float),
        "KSR Conc.": np.arange(1,19+1, dtype=np.float),
        "3i Period": np.arange(3, 19+1, dtype=np.float)
    }
    for k in logscale:
        space[k] = np.log10(space[k])
        i = X_names.index(k)
        bounds[i]=(np.log10(X_bounds[i][0]),np.log10(X_bounds[i][1]))
        
    if normalize:
        # normalize to [0,10]
        for i,k in enumerate(X_names):
            d = bounds[i][1] - bounds[i][0]
            if d > 0:
                space[k] = (space[k] - bounds[i][0]) / d
    
    return space

def get_domain(normalize=True, logscale=["Detach Speed"]):
    space = get_space(normalize, logscale)
    
    return [
        {"name":"Pre-con Conc.", "type":"discrete", "domain":space["Pre-con Conc."]},
        {"name":"Pre-con Period", "type":"discrete", "domain":space["Pre-con Period"]},
        {"name":"Detach Period", "type":"discrete", "domain":space["Detach Period"]},
        {"name":"Detach Speed", "type":"discrete", "domain":space["Detach Speed"]},
        {"name":"Detach Length", "type":"categorical", "domain":space["Detach Length"]},
        {"name":"KSR Conc.", "type":"discrete", "domain":space["KSR Conc."]},
        {"name":"3i Period", "type":"discrete", "domain":space["3i Period"]}
    ]

def get_batch_context(normalize=True,mintrypsintime = 5.):
    # fixed parameters
    batch_context = []
    for i_plate in range(8):
        for i_well in range(6):
            batch_context.append({"Detach Period": mintrypsintime + 3*i_well})
            if normalize:
                batch_context[-1]["Detach Period"] /= (X_bounds[2][1]-X_bounds[2][0])
    return batch_context

def get_batch_context24(normalize=True,mintrypsintime = 5.):
    # fixed parameters
    batch_context = []
    for i_plate in range(4):
        for i_well in range(6):
            batch_context.append({"Detach Period": mintrypsintime + 3*i_well})
            if normalize:
                batch_context[-1]["Detach Period"] /= (X_bounds[2][1]-X_bounds[2][0])
    return batch_context
        
def get_init_design(arr, normalize=False, logscale=["Detach Speed"]):
    init_design_arr = copy.copy(arr)
    bounds = copy.copy(X_bounds)

    for k in logscale:
        i = X_names.index(k)
        bounds[i]=(np.log10(X_bounds[i][0]),np.log10(X_bounds[i][1]))
        init_design_arr[:,i] = np.log10(init_design_arr[:,i])
    init_design_arr[np.isinf(init_design_arr)] = 0 # replace log(0)=-inf to log(1)=0

    if normalize:
        # normalize to [0,1]
        for i,k in enumerate(X_names):
            d = bounds[i][1] - bounds[i][0]
            if d > 0:
                init_design_arr[:,i] = (init_design_arr[:,i] - bounds[i][0]) / d
    
    return init_design_arr

def rescale(X, logscale=["Detach Speed"]):
    bounds = copy.copy(X_bounds)
    one = np.ones(X.shape)
    init = np.ones(X.shape)
    
    for k in logscale:
        i = X_names.index(k)
        bounds[i]=(np.log10(X_bounds[i][0]),np.log10(X_bounds[i][1]))
    
    for i,k in enumerate(X_names):
        d = bounds[i][1] - bounds[i][0]
        #print(d)
        one[:,i] = d
        init[:,i] = bounds[i][0]
    rescaledX = np.array(one*X + init)
    
    for k in logscale:
        i = X_names.index(k)
        rescaledX[:,i] = 10**(rescaledX[:,i])
    rescaledX = np.round(rescaledX, 0)   
    
    return rescaledX

def getNearestValue(_list, num):
    idx = np.abs(np.asarray(_list) - num).argmin()
    return _list[idx]