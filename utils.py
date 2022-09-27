# -*- coding: utf-8 -*-
"""
@author: Siva
"""
import scipy.signal
from brian2 import *
import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd

def process_negative_phases(inst_phases_2dlist):
    for i in range(len(inst_phases_2dlist)):
        for j in range(len(inst_phases_2dlist[i])):
            if inst_phases_2dlist[i][j]<0:
                inst_phases_2dlist[i][j]=360+inst_phases_2dlist[i][j]                
    return inst_phases_2dlist
    
def parser(string):
    strs=string[1:-1].split(',')
    float_=[float(i) for i in strs]
    return float_

def synch_metric(inst_phases_all_osc_all_t, n_osc=50):
    ph_diff=[]
    for i in range(n_osc):
        for j in range(n_osc):
            if j>i:
                ph_diff = np.concatenate((ph_diff , (inst_phases_all_osc_all_t[i]- inst_phases_all_osc_all_t[j])))      
    n1=np.abs(np.mean(np.exp(1*np.complex(0,1)*ph_diff)))
    n2=np.abs(np.mean(np.exp(2*np.complex(0,1)*ph_diff)))    
    n3=np.abs(np.mean(np.exp(3*np.complex(0,1)*ph_diff)))    
    n4=np.abs(np.mean(np.exp(4*np.complex(0,1)*ph_diff)))
    n5=np.abs(np.mean(np.exp(5*np.complex(0,1)*ph_diff)))
    n6=np.abs(np.mean(np.exp(6*np.complex(0,1)*ph_diff)))
    n7=np.abs(np.mean(np.exp(7*np.complex(0,1)*ph_diff)))        
    
    return ph_diff[0::5], [n1, n2*(1-n1), n3*(1-n1)*(1-n2), n4*(1-n1)*(1-n2)*(1-n3), 
    n5*(1-n1)*(1-n2)*(1-n3)*(1-n4), n6*(1-n1)*(1-n2)*(1-n3)*(1-n4)*(1-n5), 
    n7*(1-n1)*(1-n2)*(1-n3)*(1-n4)*(1-n5)*(1-n6)]

def write_inst_phase_pd(df, fname, dir_='outputs/perts/'):
    df.to_csv(dir_+fname+'.csv')
    
def get_cluster_labels(inst_phase_pd, t, n_clusters = 2, csv_read=False):    
    t=int(t*1000*10)
    if csv_read:
        t=str(t)
    gmm_model = GaussianMixture(n_components=n_clusters, random_state=1).fit(inst_phase_pd[t].values.reshape(-1, 1))   
    cluster_labels = gmm_model.predict(inst_phase_pd[t].values.reshape(-1, 1))
    return cluster_labels

def conv(mem_v, f_thresh=0.007):
    order=5
    thresh_freq=f_thresh
    n_coeffs, d_coeffs=scipy.signal.butter(order, thresh_freq)
    return scipy.signal.filtfilt(n_coeffs, d_coeffs, mem_v, padtype=None)

def inst_phases(mm_evs, mm_evs_u=None, f_thresh=0.007):
    times = mm_evs.t/ms
    inst_phases = []
    mem_vs=[]
    mem_us=[]
    conv_cons=[]    
    n_neurons = len(mm_evs.v)
    for nrn_id in range(n_neurons):
        mem_v = mm_evs.v[nrn_id]
        mem_u=None
        if mm_evs_u!=None:
            mem_u = mm_evs_u.u[nrn_id]        
        #mem_v=mem_v[::10]
        convolved1 = conv(mem_v, f_thresh)
        convolved1 = (convolved1 - np.mean(convolved1)) / np.std(convolved1)
        inst_phases.append(np.angle(scipy.signal.hilbert(convolved1)))
        mem_vs.append(mem_v)
        mem_us.append(mem_u)
        conv_cons.append(convolved1)        
    return times, inst_phases, conv_cons, mem_vs, mem_us

def inst_phases_select(mm_evs, mm_evs_u, nrns, f_thresh=0.007):
    times = mm_evs.t/ms
    inst_phases = []
    mem_vs=[]
    mem_us=[]
    conv_cons=[]   
    for nrn_id in nrns:
        mem_v = mm_evs.v[nrn_id]
        mem_u = mm_evs_u.u[nrn_id]
        #mem_v=mem_v[::10]
        convolved1 = conv(mem_v, f_thresh)
        convolved1 = (convolved1 - np.mean(convolved1)) / np.std(convolved1)
        inst_phases.append(np.angle(scipy.signal.hilbert(convolved1)))
        mem_vs.append(mem_v)
        mem_us.append(mem_u)
        conv_cons.append(convolved1)        
    return times, inst_phases, conv_cons, mem_vs, mem_us


def plot_polar(inst_phases_all_t, n_nrns_to_plot=10,
               fname='/data/fig.png'):
    ph_diff = []
    n_neurons=50
    rand_nrns1=random.sample(range(0, n_neurons), n_nrns_to_plot)
    rand_nrns2=random.sample(range(0, n_neurons), n_nrns_to_plot)
    for i in rand_nrns1:
        for j in rand_nrns2:
            if i==j:
                continue
            ph_diff.append(list(inst_phases_all_t[i] - inst_phases_all_t[j]))    
    flat_list = list(itertools.chain(*ph_diff))    
    plt.subplot(111, polar=True)
    plt.hist(flat_list, bins=100)    
    #plt.savefig(fname, dpi=500)


