# -*- coding: utf8 -*-
"""
Optimizes connection parameters (synaptic weights)
authors: Bence Bagi, András Ecker last update: 12.2021
"""

import os, sys, logging
import numpy as np
import pandas as pd
import bluepyopt as bpop
import multiprocessing as mp
import sim_evaluator_double_sim_opt4scale
base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-3])
# add "scripts" directory to the path (to import modules)
sys.path.insert(0, os.path.sep.join([base_path, "scripts"]))
from helper import load_wmx
from plots import plot_evolution

# print info into console
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_checkpoints(pklf_name):
    """
    Loads in saved checkpoints from pickle file (used e.g. to repeat the analysis part...)
    :param pklf_name: name of the saved pickle file
    :return: obejects saved by BluePyOpt"""
    import pickle
    with open(pklf_name, "rb") as f:
        cp = pickle.load(f)
    return cp["generation"], cp["halloffame"], cp["logbook"], cp["history"]


def hof2csv(pnames, hof, f_name):
    """
    Creates pandas DaataFrame from hall of fame and saves it to csv
    :param pnames: names of optimized parameters
    :param hof: BluePyOpt HallOfFame object
    :param f_name: name of the saved file
    """
    data = np.zeros((len(hof), len(pnames)))
    for i in range(len(hof)):
        data[i, :] = hof[i]
    df = pd.DataFrame(data=data, columns=pnames)
    df.to_csv(f_name)

def optconf_experimental_range():
    # preCCh/SWR-presenting conductances (from various papers)
    g_experimental = {'w_BC_E': 4.5,
                      'w_BC_I': 3.95,
                      'w_PC_E': 0.54,
                      'w_PC_I': 5.28}
    # from the manuscript:
    CCh_scaling_factors = {
        "s_PC_E": 0.255,
        "s_PC_I": 0.28,
        "s_BC_E": 0.4
    }
    # wmx_mult shall start at the value where it scales the average PC-PC conductance (0.2 nS) to the experimental one
    wmx_mult = g_experimental["w_PC_E"] / 0.2

    # range boundaries (%)
    s1, s2 = 0.5, 1.5

    optconf = [("w_PC_I_", s1 * g_experimental["w_PC_I"], s2 * g_experimental["w_PC_I"]),
               ("w_BC_E_", s1 * g_experimental["w_BC_E"], s2 * g_experimental["w_BC_E"]),
               ("w_BC_I_", s1 * g_experimental["w_BC_I"], s2 * g_experimental["w_BC_I"]),
               ("wmx_mult_", s1 * wmx_mult, s2 * wmx_mult),
               ("w_PC_MF_", 15.0, 25.0),
               ("rate_MF_", 5.0, 20.0),
               ("s_PC_E", s1 * CCh_scaling_factors["s_PC_E"], s2 * CCh_scaling_factors["s_PC_E"]),
               ("s_PC_I", s1 * CCh_scaling_factors["s_PC_I"], s2 * CCh_scaling_factors["s_PC_I"]),
               ("s_BC_E", s1 * CCh_scaling_factors["s_BC_E"], s2 * CCh_scaling_factors["s_BC_E"]),
               ("s_BC_I", 0.1, 0.9)]  # no experimental data so no assumption is made
    return optconf

def optconf_broad_range():
    s1, s2 = 0.75, 1.25
    optconf = [("w_PC_I_", 0.1, 8.0),
               ("w_BC_E_", 0.1, 8.0),
               ("w_BC_I_", 1.0, 8.0),
               ("wmx_mult_", 0.5, 3.0),
               ("w_PC_MF_", 15.0, 25.0),
               ("rate_MF_", 5.0, 20.0),
               ("s_PC_E", s1 * 0.255, s2 * 0.255),
               ("s_PC_I", s1 * 0.28, s2 * 0.28),
               ("s_BC_E", s1 * 0.4, s2 * 0.4),
               ("s_BC_I", 0.1, 0.9)]
    return optconf

if __name__ == "__main__":
    try:
        STDP_mode = sys.argv[1]
    except:
        STDP_mode = "sym"
    assert STDP_mode in ["sym", "asym"]
    linear = True
    place_cell_ratio = 0.5
    f_in = "wmx_%s_%.1f_linear.pkl" % (STDP_mode, place_cell_ratio) if linear else "wmx_%s_%.1f.pkl" % (STDP_mode, place_cell_ratio)
    cp_f_name = os.path.join(base_path, "scripts", "optimization", "checkpoints", "checkpoint_%s" % f_in[4:])
    hof_f_name = os.path.join(base_path, "scripts", "optimization", "checkpoints", "hof_%s.csv" % f_in[4:-4])

    # parameters to be fitted as a list of: (name, lower bound, upper bound)
    # the order matters! if you want to add more parameters - update `run_sim.py` too
    optconf = optconf_broad_range()
    pnames = [name for name, _, _ in optconf]

    offspring_size = 100
    max_ngen = 100

    pklf_name = os.path.join(base_path, "files", f_in)
    wmx_PC_E = load_wmx(pklf_name) * 1e9  # *1e9 nS conversion

    # Create multiprocessing pool for parallel evaluation of fitness function
    n_proc = 100
    pool = mp.Pool(processes=n_proc)
    # Create BluePyOpt optimization and run
    evaluator = sim_evaluator_double_sim_opt4scale.Brian2Evaluator(linear, wmx_PC_E, optconf)
    opt = bpop.optimisations.DEAPOptimisation(evaluator, offspring_size=offspring_size, map_function=pool.map,
                                              eta=20, mutpb=0.3, cxpb=0.7)

    print("Started running %i simulations on %i cores..." % (offspring_size*max_ngen, n_proc))
    pop, hof, log, hist = opt.run(max_ngen=max_ngen, cp_filename=cp_f_name)
    del pool, opt
    # ====================================== end of optimization ======================================

    # summary figure (about optimization)
    plot_evolution(log.select("gen"), np.array(log.select("min")), np.array(log.select("avg")),
                   np.array(log.select("std")), "fittnes_evolution")

    # save hall of fame to csv, get best individual, and rerun with best parameters to save figures
    hof2csv(pnames, hof, hof_f_name)
    best = hof[0]
    for pname, value in zip(pnames, best):
        print("%s = %.3f" % (pname, value))
    _ = evaluator.evaluate_with_lists(best, verbose=True, plots=True)
