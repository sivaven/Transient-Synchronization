# -*- coding: utf-8 -*-
"""
@author: Siva
"""
import utils
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
import EI_wrapper
import pandas as pd
import random

I_EXC = 36
OUTPUT_DIR='outputs/'
N_CLUST=3
N_PERT=50
PERT_W=0.5
OBS_TIMEPTS_FOLL_PERT=10
DUR=10
SEED=234
random.seed(SEED)
perturb_times=random.sample(range(500, 1000), N_PERT)
print('perturbation times ', perturb_times)
n_nrns_to_perturb_arr=[i for i in range(0, 21)] # Experiment

metrics_lows=[]

for n_nrns_to_perturb in n_nrns_to_perturb_arr:
    print(n_nrns_to_perturb)    
    dfs =EI_wrapper.sims(I_exc_=I_EXC, nrn_to_perturb_=n_nrns_to_perturb, perturb_times_=perturb_times, pert_w_=PERT_W, seed=SEED)    
    print('n dataframes ', len(dfs))
    for i in range(len(dfs)):
        utils.write_inst_phase_pd(df=dfs[i], fname='n_pert_nrns_'+str(n_nrns_to_perturb)+'__pert_trial_'+str(i), dir_=OUTPUT_DIR) #'0.csv' is always baseline
    
    aris=[[] for i in range(len(perturb_times))]    
    tt=0
    while tt<DUR:            
        cluster_base=utils.get_cluster_labels(dfs[0], tt, n_clusters = N_CLUST)        
        for i in range(len(perturb_times)):
            cluster_pert=utils.get_cluster_labels(dfs[i+1], tt, n_clusters = N_CLUST)
            aris[i].append(adjusted_rand_score(cluster_base, cluster_pert))            
        tt=tt+0.1
        
    metrics_low=[]    
    for i in range(len(perturb_times)):
        p_time=int(perturb_times[i]/100)    
        metrics_low.append(np.min(aris[i][p_time-1:p_time+OBS_TIMEPTS_FOLL_PERT])) 
    metrics_lows.append(metrics_low)

df_lows=pd.DataFrame(metrics_lows, index=n_nrns_to_perturb_arr)
response=1-df_lows
response.to_csv(OUTPUT_DIR+'crc.csv')
