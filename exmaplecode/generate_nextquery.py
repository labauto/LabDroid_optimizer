import os
import numpy as np
import pandas as pd
import GPyOpt
import GPy 
import pickle

import utils
from utils import X_names, X_bounds, get_domain
from utils import get_batch_context, get_init_design, rescale

#params
pastdatapath = "../data/pastresults/253G1_results_r5.csv"
savepath= "../data/nextquery/" 
normalize = True # if normalize == True, normalize domain in [0,1]^d & generate gaussian process in [0,1]^d
batch_contextFlag = True
_no = 0

def calc_EIstep(X_init, Y_init, jitter,batch_context=True,normalize = True, savepath = None):
    space = GPyOpt.core.task.space.Design_space(get_domain(normalize=normalize), None)
    model_gp = GPyOpt.models.GPModel(
        kernel=GPy.kern.RBF(input_dim=X_init.shape[1],variance = 0.25 ,ARD=True),verbose=False, noise_var=9e-4)
    objective = GPyOpt.core.task.SingleObjective(None)
    acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)
    acquisition_ = GPyOpt.acquisitions.AcquisitionEI(model_gp, space, acquisition_optimizer, jitter=jitter)
    acquisition = GPyOpt.acquisitions.LP.AcquisitionLP(model_gp, space, acquisition_optimizer,acquisition_)
    evaluator = GPyOpt.core.evaluators.LocalPenalization(acquisition, batch_size=48)
    
    bo_EI = GPyOpt.methods.ModularBayesianOptimization(
    model=model_gp,
    space=space,
    objective=objective,
    acquisition=acquisition,
    evaluator=evaluator,
    X_init=X_init,
    Y_init=Y_init,   
    normalize_Y=True
    )
    if batch_context:
        nextX = bo_EI.suggest_next_locations(batch_context=get_batch_context(normalize=normalize))
    else:
        nextX = bo_EI.suggest_next_locations()
        
    if normalize:
        nextX = rescale(nextX)
    
    with open( savepath+"model/eimodel.pkl", "wb") as f:
        pickle.dump(bo_EI, f, protocol=2)
    
    return nextX

def x2csv(nextX, _no,savepath = None):
    wells = np.array([])
    dishes = np.array([])
    for dishno in np.arange(1,9):
        wells = np.append(wells,[1,2,3,4,5,6])
        dishes = np.append(dishes,np.array([1,1,1,1,1,1])*dishno)
    wells = np.array([wells]).T
    dishes = np.array([dishes]).T
    test = np.concatenate((dishes,wells),axis=1)
    result = np.concatenate((test,nextX),axis=1)
    result = pd.DataFrame(result, columns=["Plate", "Well","Pre-con Conc.","Pre-con Period","Detach Period","Detach Speed","Detach Length","KSR Conc.","3i Period"])
    result.to_csv(savepath+"/csv/ei"+str(_no)+".csv")
    return 0


if __name__ == "__main__":
    
    # load past data
    init_design_df =  pd.read_csv(pastdatapath)
    
    # Reshape
    score=np.array([init_design_df["Area%"].as_matrix()],dtype=np.float).T
    init_design_arr = np.array([
        init_design_df["Pre-con Conc."].as_matrix(),
        init_design_df["Pre-con Period"].as_matrix(),
        init_design_df["Detach Period"].as_matrix(),
        init_design_df["Detach Speed"].as_matrix(),
        init_design_df["Detach Length"].as_matrix(),
        init_design_df["KSR Conc."].as_matrix(),
        init_design_df["3i Period"].as_matrix()
    ], dtype=np.float).T
    
    Y_init = -1*np.array(score)
    X_init = get_init_design(init_design_arr, normalize=normalize)
    
    #Generate next query
    nextX = calc_EIstep(X_init, Y_init, jitter=0, batch_context=batch_contextFlag, normalize = normalize, savepath = savepath)
    x2csv(nextX, _no,savepath= savepath )
    
