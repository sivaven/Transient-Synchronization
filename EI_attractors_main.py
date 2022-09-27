# -*- coding: utf-8 -*-
"""
@author: Siva
"""

from brian2 import *
from matplotlib import pyplot as plt
import itertools
import random
import numpy as np
import pandas as pd

def configure_and_run_network_EI(I_exc, sd=123, perturb=False, n_i_nrns=0, perturb_time=1000, perturb_W=0.3):
    #print(n_i_nrns)
    ne = 100    
    ae = 0.02/ms
    be = 0.2/ms
    ce = -65 * mV
    de = 8*mV/ms
    var_Ie = I_exc    
    var_Ie2 = 10    
    ni = 50    
    ai = 0.1/ms
    bi = 0.2/ms
    ci = -45 * mV 
    di = 2*mV/ms
    var_Ii = 0      
    cw_e2i=perturb_W*mV
    
    eqs_e = '''
    dv/dt = (0.04/ms/mV)*v**2+(5/ms)*v+140*mV/ms-u+Ie : volt
    du/dt = ae*(be*v-u)                                : volt/second  
    Ie                                                : volt/second
    '''    
    reset_e = '''
    v = ce
    u = u + de
    '''
    
    eqs_i = '''
    dv/dt = (0.04/ms/mV)*v**2+(5/ms)*v+140*mV/ms-u+Ii : volt
    du/dt = ai*(bi*v-u)                                : volt/second
    Ii                                                : volt/second
    '''    
    reset_i = '''
    v = ci
    u = u + di
    '''    
    
    G_e = NeuronGroup(ne, eqs_e, threshold = 'v >= 30*mV', reset = reset_e, method = 'euler')
    G_e.v = ce
    G_e.u = be*ce  
    G_e.Ie=var_Ie*mV/ms
    
    G_e2 = NeuronGroup(10, eqs_e, threshold = 'v >= 30*mV', reset = reset_e, method = 'euler')
    G_e2.v = ce
    G_e2.u = be*ce  
    G_e2.Ie=0*mV/ms
        
    G_i = NeuronGroup(ni, eqs_i, threshold = 'v >= 30*mV', reset = reset_i, method = 'euler')
    G_i.v = ci
    G_i.u = bi*ci
    G_i.Ii=var_Ii*mV/ms
    
    seed(sd)
    
    S_ei = Synapses(G_e, G_i, on_pre = 'v+=0.3*mV', delay=1*ms )
    S_ei.connect(p = 0.7)      
    S_ii = Synapses(G_i, G_i, on_pre = 'v+=-0.3*mV', delay=1*ms)
    S_ii.connect(p = 0.4)     
    
    #to perturb
    if n_i_nrns>0:
        S_e2i = Synapses(G_e2, G_i, on_pre = 'v+=cw_e2i', delay=1*ms)
        post_indices=random.sample(range(1,50), n_i_nrns)  
        S_e2i.connect(i=0, j=post_indices)  
    
    SM = StateMonitor(G_i, 'v', record=True)   
    SMu = StateMonitor(G_i, 'u', record=True)      
    SPe = SpikeMonitor(G_e, 'v')
    SPi = SpikeMonitor(G_i, 'v')    
    
    if n_i_nrns>0:
        net = Network(G_e, G_i, G_e2, S_ei, S_ii, S_e2i, SM, SPe, SPi, SMu)
    else:
        net = Network(G_e, G_i, G_e2, S_ei, S_ii, SM, SPe, SPi, SMu)
    
    G_e2.Ie=0*mV/ms    
    net.run((perturb_time-100)*ms)
    
    if perturb:
        G_e2.Ie=var_Ie2*mV/ms
    else:
        G_e2.Ie=0*mV/ms   
    net.run(100*ms)
    
    G_e2.Ie=0*mV/ms    
    net.run((10000-perturb_time)*ms)
    
    W = np.full((len(G_i), len(G_i)), np.nan)
    Wex = np.full((len(G_e), len(G_i)), np.nan)

    
    return [SM, SMu, SPe, SPi], [W, Wex]

