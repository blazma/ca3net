# -*- coding: utf8 -*-
"""
Mostly a copy-paste of `spw_network.py` but this one is optimized for gamma oscillation
(Creates AdExpIF PC and BC populations in Brian2, loads in recurrent connection matrix for PC population
runs simulation and checks the dynamics)
authors: András Ecker, Szabolcs Káli last update: 03.2021
"""

import os
import shutil
import numpy as np
import random as pyrandom
from brian2 import *
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt
from helper import load_wmx, save_vars
from spw_network import analyse_results


base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])

# ----- base parameters are COPY-PASTED from `spw_network.py` -----
# population size
nPCs = 8000
nBCs = 150
# sparseness
connection_prob_PC = 0.1
connection_prob_BC = 0.25

# synaptic time constants:
# rise time constants
rise_PC_E = 1.3 * ms  # Guzman 2016 (only from Fig.1 H - 20-80%)
rise_PC_MF = 0.65 * ms  # Vyleta ... Jonas 2016 (20-80%)
rise_PC_I = 0.3 * ms  # Bartos 2002 (20-80%)
rise_BC_E = 1. * ms  # Lee 2014 (data from CA1)
rise_BC_I = 0.25 * ms  # Bartos 2002 (20-80%)
# decay time constants
decay_PC_E = 9.5 * ms  # Guzman 2016 ("needed for temporal summation of EPSPs")
decay_PC_MF = 5.4 * ms  # Vyleta ... Jonas 2016
decay_PC_I = 3.3 * ms  # Bartos 2002
decay_BC_E = 4.1 * ms  # Lee 2014 (data from CA1)
decay_BC_I = 1.2 * ms  # Bartos 2002
# Normalization factors (normalize the peak of the PSC curve to 1)
tp = (decay_PC_E * rise_PC_E)/(decay_PC_E - rise_PC_E) * np.log(decay_PC_E/rise_PC_E)  # time to peak
norm_PC_E = 1.0 / (np.exp(-tp/decay_PC_E) - np.exp(-tp/rise_PC_E))
tp = (decay_PC_MF * rise_PC_MF)/(decay_PC_MF - rise_PC_MF) * np.log(decay_PC_MF/rise_PC_MF)
norm_PC_MF = 1.0 / (np.exp(-tp/decay_PC_MF) - np.exp(-tp/rise_PC_MF))
tp = (decay_PC_I * rise_PC_I)/(decay_PC_I - rise_PC_I) * np.log(decay_PC_I/rise_PC_I)
norm_PC_I = 1.0 / (np.exp(-tp/decay_PC_I) - np.exp(-tp/rise_PC_I))
tp = (decay_BC_E * rise_BC_E)/(decay_BC_E - rise_BC_E) * np.log(decay_BC_E/rise_BC_E)
norm_BC_E = 1.0 / (np.exp(-tp/decay_BC_E) - np.exp(-tp/rise_BC_E))
tp = (decay_BC_I * rise_BC_I)/(decay_BC_I - rise_BC_I) * np.log(decay_BC_I/rise_BC_I)
norm_BC_I = 1.0 / (np.exp(-tp/decay_BC_I) - np.exp(-tp/rise_BC_I))
# synaptic delays:
delay_PC_E = 2.2 * ms  # Guzman 2016
delay_PC_I = 1.1 * ms  # Bartos 2002
delay_BC_E = 0.9 * ms  # Geiger 1997 (data from DG)
delay_BC_I = 0.6 * ms  # Bartos 2002
# synaptic reversal potentials
Erev_E = 0.0 * mV
Erev_I = -70.0 * mV

z = 1 * nS
# AdExpIF parameters for PCs (re-optimized by Szabolcs)
g_leak_PC = 4.31475791937223 * nS
tau_mem_PC = 41.7488927175169 * ms
Cm_PC = tau_mem_PC * g_leak_PC
Vrest_PC = -75.1884554193901 * mV
Vreset_PC = -29.738747396665072 * mV
theta_PC = -24.4255910105977 * mV
tref_PC = 5.96326930945599 * ms
delta_T_PC = 4.2340696257631 * mV
spike_th_PC = theta_PC + 5 * delta_T_PC
a_PC = -0.274347065652738 * nS
b_PC = 206.841448096415 * pA
tau_w_PC = 84.9358017225512 * ms
""" comment this back to run with ExpIF PC model...
# ExpIF parameters for PCs (optimized by Szabolcs)
g_leak_PC = 4.88880734814042 * nS
tau_mem_PC = 70.403501012992 * ms
Cm_PC = tau_mem_PC * g_leak_PC
Vrest_PC = -76.59966923496779 * mV
Vreset_PC = -58.8210432444992 * mV
theta_PC = -28.7739788756 * mV
tref_PC = 1.07004414539699 * ms
delta_T_PC = 10.7807538634886 * mV
spike_th_PC = theta_PC + 5 * delta_T_PC
a_PC = 0. * nS
b_PC = 0. * pA
tau_w_PC = 1 * ms
"""
# parameters for BCs (re-optimized by Szabolcs)
g_leak_BC = 7.51454086502288 * nS
tau_mem_BC = 15.773412296065 * ms
Cm_BC = tau_mem_BC * g_leak_BC
Vrest_BC = -74.74167987795019 * mV
Vreset_BC = -64.99190523539687 * mV
theta_BC = -57.7092044103536 * mV
tref_BC = 1.15622717832178 * ms
delta_T_BC = 4.58413312063091 * mV
spike_th_BC = theta_BC + 5 * delta_T_BC
a_BC = 3.05640210724374 * nS
b_BC = 0.916098931234532 * pA
tau_w_BC = 178.581099914024 * ms

eqs_PC = """
dvm/dt = (-g_leak_PC*(vm-Vrest_PC) + g_leak_PC*delta_T_PC*exp((vm- theta_PC)/delta_T_PC) - w + depol_ACh - ((g_ampa+g_ampaMF)*z*(vm-Erev_E) + g_gaba*z*(vm-Erev_I)))/Cm_PC : volt (unless refractory)
dw/dt = (a_PC*(vm-Vrest_PC) - w) / tau_w_PC : amp
dg_ampa/dt = (x_ampa - g_ampa) / rise_PC_E : 1
dx_ampa/dt = -x_ampa / decay_PC_E : 1
dg_ampaMF/dt = (x_ampaMF - g_ampaMF) / rise_PC_MF : 1
dx_ampaMF/dt = -x_ampaMF / decay_PC_MF : 1
dg_gaba/dt = (x_gaba - g_gaba) / rise_PC_I : 1
dx_gaba/dt = -x_gaba/decay_PC_I : 1
depol_ACh: amp
"""

eqs_BC = """
dvm/dt = (-g_leak_BC*(vm-Vrest_BC) + g_leak_BC*delta_T_BC*exp((vm- theta_BC)/delta_T_BC) - w - (g_ampa*z*(vm-Erev_E) + g_gaba*z*(vm-Erev_I)))/Cm_BC : volt (unless refractory)
dw/dt = (a_BC*(vm-Vrest_BC) - w) / tau_w_BC : amp
dg_ampa/dt = (x_ampa - g_ampa) / rise_BC_E : 1
dx_ampa/dt = -x_ampa/decay_BC_E : 1
dg_gaba/dt = (x_gaba - g_gaba) / rise_BC_I : 1
dx_gaba/dt = -x_gaba/decay_BC_I : 1
"""

def run_simulation(params, save, seed, verbose=True):
    """
    Sets up the network and runs simulation
    :param wmx_PC_E: np.array representing the recurrent excitatory synaptic weight matrix
    :param save: bool flag to save PC spikes after the simulation (used by `bayesian_decoding.py` later)
    :param seed: random seed used for running the simulation
    :param verbose: bool flag to report status of simulation
    :return SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC: Brian2 monitors (+ array of selected cells used by multi state monitor)
    """

    w_PC_I, w_BC_E, w_BC_I, wmx_PC_E, w_PC_MF, rate_MF, g_leak_PC, tau_mem_PC, Cm_PC, Vrest_PC, Vrest_BC = params

    np.random.seed(seed)
    pyrandom.seed(seed)

    PCs = NeuronGroup(nPCs, model=eqs_PC, threshold="vm>spike_th_PC",
                      reset="vm=Vreset_PC; w+=b_PC", refractory=tref_PC, method="exponential_euler")
    PCs.vm = Vrest_PC; PCs.g_ampa = 0.0; PCs.g_ampaMF = 0.0; PCs.g_gaba = 0.0
    PCs.depol_ACh = 40 * pA  # ACh induced ~10 mV depolarization in PCs...

    BCs = NeuronGroup(nBCs, model=eqs_BC, threshold="vm>spike_th_BC",
                      reset="vm=Vreset_BC; w+=b_BC", refractory=tref_BC, method="exponential_euler")
    BCs.vm  = Vrest_BC; BCs.g_ampa = 0.0; BCs.g_gaba = 0.0

    MF = PoissonGroup(nPCs, rate_MF)
    C_PC_MF = Synapses(MF, PCs, on_pre="x_ampaMF+=norm_PC_MF*w_PC_MF")
    C_PC_MF.connect(j="i")

    # weight matrix used here
    C_PC_E = Synapses(PCs, PCs, "w_exc:1", on_pre="x_ampa+=norm_PC_E*w_exc", delay=delay_PC_E)
    nonzero_weights = np.nonzero(wmx_PC_E)
    C_PC_E.connect(i=nonzero_weights[0], j=nonzero_weights[1])
    C_PC_E.w_exc = wmx_PC_E[nonzero_weights].flatten()
    del wmx_PC_E

    C_PC_I = Synapses(BCs, PCs, on_pre="x_gaba+=norm_PC_I*w_PC_I", delay=delay_PC_I)
    C_PC_I.connect(p=connection_prob_BC)

    C_BC_E = Synapses(PCs, BCs, on_pre="x_ampa+=norm_BC_E*w_BC_E", delay=delay_BC_E)
    C_BC_E.connect(p=connection_prob_PC)

    C_BC_I = Synapses(BCs, BCs, on_pre="x_gaba+=norm_BC_I*w_BC_I", delay=delay_BC_I)
    C_BC_I.connect(p=connection_prob_BC)

    SM_PC = SpikeMonitor(PCs)
    SM_BC = SpikeMonitor(BCs)
    RM_PC = PopulationRateMonitor(PCs)
    RM_BC = PopulationRateMonitor(BCs)

    selection = np.arange(0, nPCs, 4)   # subset of neurons for recoring variables
    StateM_PC = StateMonitor(PCs, variables=["vm", "w", "g_ampa", "g_ampaMF", "g_gaba"],
                             record=selection.tolist(), dt=0.1*ms)
    StateM_BC = StateMonitor(BCs, "vm", record=[nBCs/2], dt=0.1*ms)

    if verbose:
        run(10000*ms, report="text")
    else:
        run(10000*ms)

    if save:
        save_vars(SM_PC, RM_PC, StateM_PC, selection, seed)

    return SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC

def copy_plots(subdir):
    fig_dir = os.path.join(base_path, "figures_simultaneous")
    fig_subdir = os.path.join(fig_dir, subdir)
    if not os.path.isdir(fig_subdir):
        os.mkdir(fig_subdir)
    for path, subdirs, files in os.walk(fig_dir):
        for name in files:
            filename = os.path.join(path, name)
            shutil.copy(filename, fig_subdir)

def gamma_network(wmx_PC_E, save, seed, verbose):
    # parameters start out the same as SWR
    w_PC_I = 1.37
    w_BC_E = 1.21
    w_BC_I = 5.27
    wmx_mult = 1.704
    w_PC_MF = 22.96
    rate_MF = 17.47 * Hz

    # some of them get tuned as described in manuscript
    wmx_mult = (0.02 / 0.15) * wmx_mult
    w_PC_I = (2.0 / 4.0) * w_PC_I
    w_BC_E = (0.3 / 1.5) * w_BC_E

    g_leak_PC = (2.5 / 3.3333) * 4.31475791937223 * nS
    tau_mem_PC = (80. / 60.) * 41.7488927175169 * ms
    Cm_PC = tau_mem_PC * g_leak_PC
    Vrest_PC = (-75.1884554193901 + 10.0) * mV
    Vrest_BC = (-74.74167987795019 + 10.0) * mV
    wmx_PC_E = wmx_mult * wmx_PC_E

    params = [w_PC_I, w_BC_E, w_BC_I, wmx_PC_E, w_PC_MF, rate_MF, g_leak_PC, tau_mem_PC, Cm_PC, Vrest_PC, Vrest_BC]
    SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC = run_simulation(params, save=save, seed=seed, verbose=verbose)
    results = analyse_results(SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC, seed=seed,
                              multiplier=1, linear=True, pklf_name=None, dir_name=None,
                              analyse_replay=False, TFR=False, save=save, verbose=False)
    #copy_plots("gamma")

def swr_network(wmx_PC_E, save, seed, verbose):
    w_PC_I = 1.37
    w_BC_E = 1.21
    w_BC_I = 5.27
    wmx_mult = 1.704
    w_PC_MF = 22.96
    rate_MF = 17.47 * Hz
    g_leak_PC = 4.31475791937223 * nS
    tau_mem_PC = 41.7488927175169 * ms
    Cm_PC = tau_mem_PC * g_leak_PC
    Vrest_PC = -75.1884554193901 * mV
    Vrest_BC = -74.74167987795019 * mV
    wmx_PC_E = wmx_mult * wmx_PC_E

    params = [w_PC_I, w_BC_E, w_BC_I, wmx_PC_E, w_PC_MF, rate_MF, g_leak_PC, tau_mem_PC, Cm_PC, Vrest_PC, Vrest_BC]
    SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC = run_simulation(params, save=save, seed=seed, verbose=verbose)
    results = analyse_results(SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC, seed=seed,
                              multiplier=1, linear=True, pklf_name=None, dir_name=None,
                              analyse_replay=False, TFR=False, save=save, verbose=False)
    #copy_plots("swr")


if __name__ == "__main__":
    seed = 12345
    save = False
    verbose = True

    f_in = "wmx_sym_0.5_linear.pkl"
    wmx_PC_E = load_wmx(os.path.join(base_path, "files", f_in)) * 1e9  # *1e9 nS conversion

    #swr_network(wmx_PC_E, save, seed, verbose)
    #start_scope()
    gamma_network(wmx_PC_E, save, seed, verbose)
    plt.show()
