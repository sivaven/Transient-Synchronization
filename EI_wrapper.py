# -*- coding: utf-8 -*-
"""
@author: Siva
"""

import EI_attractors_main 
import utils
import numpy as np
import pandas as pd


def freqs_from_spike_monitor(SP, nrn_id=0):
    netwrk_spikes=SP.spike_trains()
    return np.diff(netwrk_spikes[nrn_id])
    
def get_inst_phase_pd(SMv, SMu):
    #print('inst phases..')
    times, inst_phases_all_t, conv_cons, mem_vs, mem_us = utils.inst_phases(SMv, SMu)
    inst_phase_pd=pd.DataFrame(inst_phases_all_t)
    inst_phase_pd=np.rad2deg(inst_phase_pd)
    return inst_phase_pd
    
def get_degree_of_conn(W):
    degree_of_struct_conn=[]
    for row in W:
        degree_of_struct_conn.append(sum([1 for i in row if np.isnan(i)==False]))
    return degree_of_struct_conn

def get_degree_of_conn2(W):
    W=np.transpose(W)    
    degree_of_struct_conn=[]
    for row in W:
        degree_of_struct_conn.append(sum([1 for i in row if np.isnan(i)==False]))
    return degree_of_struct_conn

    
def sims(I_exc_, nrn_to_perturb_, perturb_times_, pert_w_, seed=123):    
    inst_phase_pd=[]    
    SM, W=EI_attractors_main.configure_and_run_network_EI(I_exc=I_exc_, sd=seed, perturb=False)    
    inst_phase_pd.append(get_inst_phase_pd(SM[0], SM[1])) #baseline df    
    for i in range(len(perturb_times_)):
         SMp, W=EI_attractors_main.configure_and_run_network_EI(I_exc=I_exc_, sd=seed, perturb=True, n_i_nrns=nrn_to_perturb_, perturb_time=perturb_times_[i], perturb_W=pert_w_)
         inst_phase_pd.append(get_inst_phase_pd(SMp[0], SMp[1]))         
   
    return inst_phase_pd

def sims_only4_stability(I_exc_, seed=123):
    SM, W=EI_attractors_main.configure_and_run_network_EI(I_exc=I_exc_, sd=seed, perturb=False)
    times, inst_phases_all_t, conv_cons, mem_vs, mem_us = utils.inst_phases(SM[0], SM[1])
    ph_diff,ns=utils.synch_metric(inst_phases_all_t)    
    isis_e=[]
    isis_i=[]
    NE=100
    NI=50    
    for ne in range(NE):
        isis_e.append(freqs_from_spike_monitor(SM[2], ne))
    for ni in range(NI):
        isis_i.append(freqs_from_spike_monitor(SM[3], ni))
        
    return ph_diff, ns, isis_e, isis_i, conv_cons, mem_vs, mem_us
