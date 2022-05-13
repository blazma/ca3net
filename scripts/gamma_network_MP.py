# -*- coding: utf8 -*-
"""
Mostly a copy-paste of `spw_network.py` but this one is optimized for gamma oscillation
(Creates AdExpIF PC and BC populations in Brian2, loads in recurrent connection matrix for PC population
runs simulation and checks the dynamics)
authors: András Ecker, Szabolcs Káli last update: 03.2021
"""

import os
import sys
import shutil
import numpy as np
import random as pyrandom
from brian2 import *
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt
from helper import load_wmx, save_vars
from spw_network import analyse_results
from collections import OrderedDict
import traceback
import time
import subprocess
import multiprocessing

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

rate_MF = 3 * 15.0 * Hz  # mossy fiber input freq - gamma (manual)

z = 1 * nS
# AdExpIF parameters for PCs (re-optimized by Szabolcs)


g_leak_PC = (2.5 / 3.3333) * 4.31475791937223 * nS      # gamma
tau_mem_PC = (80. / 60.) * 41.7488927175169 * ms        # gamma
Cm_PC = tau_mem_PC * g_leak_PC
Vrest_PC =  (-75.1884554193901 + 10.0) * mV       # gamma
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
Vrest_BC = (-74.74167987795019 + 10.0) * mV          # gamma
Vreset_BC = -64.99190523539687 * mV
theta_BC = -57.7092044103536 * mV
tref_BC = 1.15622717832178 * ms
delta_T_BC = 4.58413312063091 * mV
spike_th_BC = theta_BC + 5 * delta_T_BC
a_BC = 3.05640210724374 * nS
b_BC = 0.916098931234532 * pA
tau_w_BC = 178.581099914024 * ms

eqs_PC = """
dvm/dt = (-g_leak_PC*(vm-Vrest_PC) + g_leak_PC*delta_T_PC*exp((vm- theta_PC)/delta_T_PC) - w - ((g_ampa+g_ampaMF)*z*(vm-Erev_E) + g_gaba*z*(vm-Erev_I)))/Cm_PC : volt (unless refractory)
dw/dt = (a_PC*(vm-Vrest_PC) - w) / tau_w_PC : amp
dg_ampa/dt = (x_ampa - g_ampa) / rise_PC_E : 1
dx_ampa/dt = -x_ampa / decay_PC_E : 1
dg_ampaMF/dt = (x_ampaMF - g_ampaMF) / rise_PC_MF : 1
dx_ampaMF/dt = -x_ampaMF / decay_PC_MF : 1
dg_gaba/dt = (x_gaba - g_gaba) / rise_PC_I : 1
dx_gaba/dt = -x_gaba/decay_PC_I : 1
"""

eqs_BC = """
dvm/dt = (-g_leak_BC*(vm-Vrest_BC) + g_leak_BC*delta_T_BC*exp((vm- theta_BC)/delta_T_BC) - w - (g_ampa*z*(vm-Erev_E) + g_gaba*z*(vm-Erev_I)))/Cm_BC : volt (unless refractory)
dw/dt = (a_BC*(vm-Vrest_BC) - w) / tau_w_BC : amp
dg_ampa/dt = (x_ampa - g_ampa) / rise_BC_E : 1
dx_ampa/dt = -x_ampa/decay_BC_E : 1
dg_gaba/dt = (x_gaba - g_gaba) / rise_BC_I : 1
dx_gaba/dt = -x_gaba/decay_BC_I : 1
"""


def run_simulation(wmx_PC_E, g1, g2, g3, g4, save, seed, verbose=True):
    """
    Sets up the network and runs simulation
    :param wmx_PC_E: np.array representing the recurrent excitatory synaptic weight matrix
    :param save: bool flag to save PC spikes after the simulation (used by `bayesian_decoding.py` later)
    :param seed: random seed used for running the simulation
    :param verbose: bool flag to report status of simulation
    :return SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC: Brian2 monitors (+ array of selected cells used by multi state monitor)
    """

    np.random.seed(seed)
    pyrandom.seed(seed)

    # synaptic weights (see `/optimization/optimize_network.py`)
    #wmx_PC_E = g1 * (0.02 / 0.15) * wmx_PC_E       # gamma
    #w_PC_I = g2 * (2.0 / 4.0) * 0.65  # nS       # gamma
    #w_BC_E = g3 * (0.3 / 1.5) * 0.85             # gamma
    #w_BC_I = g4 * (0.1) * 5.                     # gamma (manual)

    wmx_PC_E = g1 * wmx_PC_E
    w_PC_I = g2
    w_BC_E = g3
    w_BC_I = g4
    w_PC_MF = 19.15

    PCs = NeuronGroup(nPCs, model=eqs_PC, threshold="vm>spike_th_PC",
                      reset="vm=Vreset_PC; w+=b_PC", refractory=tref_PC, method="exponential_euler")
    PCs.vm = Vrest_PC; PCs.g_ampa = 0.0; PCs.g_ampaMF = 0.0; PCs.g_gaba = 0.0

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

    selection = np.arange(0, nPCs, 20)   # subset of neurons for recoring variables
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


def grid_search_worker(g1, g2, g3, g4, wmx_PC_E, save, seed, verbose):
    start_time = time.time()
    print("# creating directory for g1_{}__g2_{}__g3_{}__g4_{}".format(g1, g2, g3, g4))
    gridsearch_output_dir = os.path.join(base_path, "gridsearch",
                                         "g1_{}__g2_{}__g3_{}__g4_{}".format(g1, g2, g3, g4))
    if not os.path.exists(gridsearch_output_dir):
        os.mkdir(gridsearch_output_dir)
    with open(os.path.join(gridsearch_output_dir, "params.txt"), "w") as gridparams_file:
        gridparams_file.writelines([str(g1), "\n",
                                    str(g2), "\n",
                                    str(g3), "\n",
                                    str(g4)])

    print("# starting simulation")
    start_scope()
    SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC = run_simulation(wmx_PC_E,
                                                                                 g1, g2, g3, g4, save=save,
                                                                                 seed=seed, verbose=verbose)

    print("# starting analysis")
    results = analyse_results(SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC, seed=seed,
                              multiplier=1, linear=True, pklf_name=None, dir_name=None,
                              analyse_replay=False, TFR=False, save=save, verbose=False)

    print("# save results to file")
    with open(os.path.join(gridsearch_output_dir, "results.txt"), "w") as gridresults_file:
        for result_key, result_value in results.items():
            gridresults_file.writelines([result_key, "=", str(result_value), "\n"])

    print("# copying figures")
    figures_dir = os.path.join(base_path, "figures")
    subprocess.call(['cp', '-a', figures_dir, gridsearch_output_dir])

    print("# DONE!")

if __name__ == "__main__":
    seed = 12345
    save = False
    verbose = True
    selected_only = False

    f_in = "wmx_sym_0.5_linear.pkl"
    wmx_PC_E = load_wmx(os.path.join(base_path, "files", f_in)) * 1e9  # *1e9 nS conversion

    pool_size = 2
    pool = multiprocessing.Pool(pool_size)

    # set of problematic runs, to be re-run separately
    selections = [
        [0.5, 1.0, 2.0, 2.0],
        [1.0, 1.0, 2.0, 2.0],
        [1.0, 2.0, 0.5, 2.0],
        [2.0, 1.0, 2.0, 2.0],
        [2.0, 2.0, 0.5, 1.0],
        [2.0, 2.0, 0.5, 2.0]
    ]
    if selected_only:
        for selection in selections:
            g1, g2, g3, g4 = selection
            worker = pool.apply_async(grid_search_worker, (g1, g2, g3, g4, wmx_PC_E, save, seed, verbose))
    else:

        measured_conductances = {
            "w_BC_E": {    #g3
                "g_preCCh": 3.51,
                "g_postCCh": 1.09,
            },
            "w_PC_E": {    # g1
                "g_preCCh": 2.34,
                "g_postCCh": 0.44
            },
            "w_PC_I": {    #g2
                "g_preCCh": 2.82,
                "g_postCCh": 0.9
            },
            "w_BC_I": {    #g4
                "g_preCCh": 2.0,
                "g_postCCh": 2.0
            }
        }

        gridpoints = OrderedDict()
        for g in measured_conductances:
            if "w_PC_E" in g:
                continue
            g0 = measured_conductances[g]["g_preCCh"]
            g2 = measured_conductances[g]["g_postCCh"]
            g1 = (g0 + g2) / 2
            gridpoints[g] = [g0, g1, g2]
        pctl = 100 - 0.92  # 0.92% is the mean connection probability between PCs in Guzman et al (2016)
        wmx_PC_E_pctl = numpy.percentile(wmx_PC_E, pctl)
        wmx_PC_E_filt = wmx_PC_E[wmx_PC_E > wmx_PC_E_pctl]

        # here we can't give an exact grid point but a multiplier for all elements of the weight matrix
        # which will land the mean of the highest conductances to the desired measured conductances
        gridpoints["w_PC_E"] = [0.0, 0.0, 0.0]
        gridpoints["w_PC_E"][0] = measured_conductances["w_PC_E"]["g_preCCh"] / np.mean(wmx_PC_E_filt)
        gridpoints["w_PC_E"][2] = measured_conductances["w_PC_E"]["g_postCCh"] / np.mean(wmx_PC_E_filt)
        gridpoints["w_PC_E"][1] = (gridpoints["w_PC_E"][0] + gridpoints["w_PC_E"][2]) / 2

        # since this one is the same for both preCch and PostCCh we'll perturbate it a little bit
        gridpoints["w_BC_I"][0] = gridpoints["w_BC_I"][1] * 0.5
        gridpoints["w_BC_I"][2] = gridpoints["w_BC_I"][1] * 2.0

        for g1 in gridpoints["w_PC_E"]:
            for g2 in gridpoints["w_PC_I"]:
                for g3 in gridpoints["w_BC_E"]:
                    for g4 in gridpoints["w_BC_I"]:
                        worker = pool.apply_async(grid_search_worker, (g1,g2,g3,g4,wmx_PC_E,save,seed,verbose))
    pool.close()
    pool.join()
