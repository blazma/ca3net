# -*- coding: utf8 -*-
"""
BluePyOpt evaluator for optimization
authors: Bence Bagi, András Ecker, Szabolcs Káli last update: 12.2021
"""

import os, sys, traceback, gc
import numpy as np
from brian2 import *
import bluepyopt as bpop
import run_sim as sim
base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-3])
# add "scripts" directory to the path (to import modules)
sys.path.insert(0, os.path.sep.join([base_path, "scripts"]))
from helper import preprocess_monitors
from detect_oscillations import analyse_rate, gamma, lowfreq, ripple
from detect_replay import slice_high_activity, replay_circular
import multiprocessing


class Brian2Evaluator(bpop.evaluators.Evaluator):
    """Evaluator class required by BluePyOpt"""

    def __init__(self, linear, Wee, params):
        """
        :param Wee: weight matrix (passing Wee with cPickle to the slaves (as BluPyOpt does) is still the fastest solution)
        :param params: list of parameters to fit - every entry must be a tuple: (name, lower bound, upper bound)
        """
        super(Brian2Evaluator, self).__init__()
        self.linear = linear
        self.Wee = Wee
        self.params = params
        self.gen_id = 1

        # Parameters to be optimized
        self.params = [bpop.parameters.Parameter(name, bounds=(minval, maxval))
                       for name, minval, maxval in self.params]
        self.swr_objectives = ["ripple_peakE", "ripple_peakI", "no_gamma_peakI", "ripple_powerE", "ripple_powerI", "ripple_rateE", "ripple_rateI"]
        self.gam_objectives = ["gamma_peakE", "gamma_peakI",  "no_subgamma_peakE", "no_subgamma_peakI", "gamma_powerE", "gamma_powerI", "gamma_rateE", "gamma_rateI", "no_replay"]
        self.objectives = self.swr_objectives + self.gam_objectives

    def generate_model(self, individual, verbose=False):
        """Runs single simulation (see `run_sim.py`) and returns monitors"""
        SM_PC, SM_BC, RM_PC, RM_BC = sim.run_simulation(self.Wee, *individual, verbose=verbose)
        return SM_PC, SM_BC, RM_PC, RM_BC

    def evaluate_errors_swr(self, SM_PC, SM_BC, RM_PC, RM_BC):
        if SM_PC.num_spikes > 0 and SM_BC.num_spikes > 0:  # check if there is any activity
            # analyse spikes
            spike_times_PC, spiking_neurons_PC, rate_PC, ISI_hist_PC, bin_edges_PC = preprocess_monitors(SM_PC,RM_PC)
            spike_times_BC, spiking_neurons_BC, rate_BC = preprocess_monitors(SM_BC, RM_BC, calc_ISI=False)
            del SM_PC, SM_BC, RM_PC, RM_BC
            gc.collect()
            # analyse rates
            slice_idx = [] if not self.linear else slice_high_activity(rate_PC, th=2, min_len=260)
            mean_rate_PC, rate_ac_PC, max_ac_PC, t_max_ac_PC, f_PC, Pxx_PC = analyse_rate(rate_PC, 1000.0, slice_idx, normalize=True)
            mean_rate_BC, rate_ac_BC, max_ac_BC, t_max_ac_BC, f_BC, Pxx_BC = analyse_rate(rate_BC, 1000.0, slice_idx, normalize=True)
            avg_ripple_freq_PC, ripple_power_PC = ripple(f_PC, Pxx_PC, slice_idx)
            avg_ripple_freq_BC, ripple_power_BC = ripple(f_BC, Pxx_BC, slice_idx)
            avg_gamma_freq_PC, _, gamma_power_PC = gamma(f_PC, Pxx_PC, slice_idx)
            avg_gamma_freq_BC, _, gamma_power_BC = gamma(f_BC, Pxx_BC, slice_idx)

            # look for significant ripple peak close to 180 Hz
            ripple_peakE = np.exp(-1 / 2 * (avg_ripple_freq_PC - 180.) ** 2 / 20 ** 2) if not np.isnan(avg_ripple_freq_PC) else 0.
            ripple_peakI = np.exp(-1 / 2 * (avg_ripple_freq_BC - 180.) ** 2 / 20 ** 2) if not np.isnan(avg_ripple_freq_BC) else 0.
            # penalize gamma peak (in inhibitory pop) - binary variable, which might not be the best for this algo.
            no_gamma_peakI = 1. if np.isnan(avg_gamma_freq_BC) else 0.
            # look for "low" exc. population rate (around 2.5 Hz)
            ripple_rateE = np.exp(-1 / 2 * (mean_rate_PC - 1.0) ** 2 / 3.0 ** 2)    # from Hájos et al 2013
            ripple_rateI = np.exp(-1 / 2 * (mean_rate_PC - 10.0) ** 2 / 10.0 ** 2)   # from Hájos et al 2013
            # *-1 since the algorithm tries to minimize...
            errors = -1. * np.array(
                [ripple_peakE, ripple_peakI, no_gamma_peakI, ripple_power_PC / 100., ripple_power_BC / 100.,
                 ripple_rateE, ripple_rateI])
            return errors.tolist()
        else:
            return [0.]*len(self.swr_objectives)

    def evaluate_errors_gam(self, SM_PC, SM_BC, RM_PC, RM_BC):
        if SM_PC.num_spikes > 0 and SM_BC.num_spikes > 0:  # check if there is any activity
            # analyse spikes
            spike_times_PC, spiking_neurons_PC, rate_PC, ISI_hist_PC, bin_edges_PC = preprocess_monitors(SM_PC, RM_PC)
            spike_times_BC, spiking_neurons_BC, rate_BC = preprocess_monitors(SM_BC, RM_BC, calc_ISI=False)
            del SM_PC, SM_BC, RM_PC, RM_BC
            gc.collect()
            # analyse rates
            slice_idx = []  # if not self.linear else slice_high_activity(rate_PC, th=2, min_len=260)
            mean_rate_PC, rate_ac_PC, max_ac_PC, t_max_ac_PC, f_PC, Pxx_PC = analyse_rate(rate_PC, 1000.0, slice_idx, normalize=True)
            mean_rate_BC, rate_ac_BC, max_ac_BC, t_max_ac_BC, f_BC, Pxx_BC = analyse_rate(rate_BC, 1000.0, slice_idx, normalize=True)
            avg_gamma_freq_PC, absolute_gamma_power_PC, relative_gamma_power_PC = gamma(f_PC, Pxx_PC, slice_idx)
            avg_gamma_freq_BC, absolute_gamma_power_BC, relative_gamma_power_BC = gamma(f_BC, Pxx_BC, slice_idx)
            avg_subgamma_freq_PC, subgamma_power_PC = lowfreq(f_PC, Pxx_PC, slice_idx)
            avg_subgamma_freq_BC, subgamma_power_BC = lowfreq(f_BC, Pxx_BC, slice_idx)

            # look for significant gamma peak close to 30 Hz
            gamma_peakE = np.exp(-1 / 2 * (avg_gamma_freq_PC - 30.) ** 2 / 10 ** 2) if not np.isnan(avg_gamma_freq_PC) else 0.
            gamma_peakI = np.exp(-1 / 2 * (avg_gamma_freq_BC - 30.) ** 2 / 10 ** 2) if not np.isnan(avg_gamma_freq_BC) else 0.
            # penalize sub gamma peaks - binary variable, which might not be the best for this algo.
            no_subgamma_peakE = 1. if np.isnan(avg_subgamma_freq_PC) else 0.
            no_subgamma_peakI = 1. if np.isnan(avg_subgamma_freq_BC) else 0.
            # look for "low" exc. population rate (around 1.0 Hz)
            gamma_rateE = np.exp(-1 / 2 * (mean_rate_PC - 5.0) ** 2 / 3.0 ** 2)    # search for mean of 5.0 with a std of 3.0
            gamma_rateI = np.exp(-1 / 2 * (mean_rate_PC - 25.0) ** 2 / 10.0 ** 2)   # search for mean of 25. with a std. of 10.0
            # penalize replay (only in circular env)
            if not self.linear:
                replay_ROI = np.where((150 <= bin_edges_PC) & (bin_edges_PC <= 850))
                no_replay = 3 if np.isnan(replay_circular(ISI_hist_PC[replay_ROI])) else 0.
            else:
                no_replay = 0.
            # *-1 since the algorithm tries to minimize...
            errors = -1. * np.array([gamma_peakE, gamma_peakI, no_subgamma_peakE, no_subgamma_peakI,
                                     2 * relative_gamma_power_PC / 100., 2 * relative_gamma_power_BC / 100.,
                                     gamma_rateE, gamma_rateI, no_replay])
            return errors.tolist()
        else:
            return [0.]*len(self.gam_objectives)

    def save_output(self, individual_swr, individual_gam, errors):
        if not os.path.isdir("./errors"):
            os.mkdir("./errors")
        with open("errors/GEN-{}_{}_errors.txt".format(self.gen_id, multiprocessing.current_process().name), "w") as errors_file:
            for idx_obj, obj in enumerate(self.objectives):
                errors_file.writelines([obj, "=", str(errors[idx_obj]), "\n"])
        param_names = ["w_PC_I_", "w_BC_E_", "w_BC_I_", "wmx_mult_", "w_PC_MF_", "rate_MF_", "g_leak_PC", "tau_mem_PC", "Cm_PC", "Vrest_PC", "Vrest_BC"]
        with open("errors/GEN-{}_{}_params.txt".format(self.gen_id, multiprocessing.current_process().name), "w") as params_file:
            params_file.writelines(["param_name", ",", "swr", ",", "gamma", "\n"])
            for idx_param, param_name in enumerate(param_names):
                params_file.writelines([param_name, ",", str(individual_swr[idx_param]), ",", str(individual_gam[idx_param]), "\n"])

    def evaluate_with_lists(self, individual, verbose=False, plots=False):
        """Fitness error used by BluePyOpt for the optimization"""

        # original, SWR parameters
        print("GEN: {},\tWORKER: {}\tSWR".format(self.gen_id, multiprocessing.current_process().name))
        w_PC_I_, w_BC_E_, w_BC_I_, wmx_mult_, w_PC_MF_, rate_MF_ = individual
        g_leak_PC = 4.31475791937223 * nS
        tau_mem_PC = 41.7488927175169 * ms
        Cm_PC = tau_mem_PC * g_leak_PC
        Vrest_PC = -75.1884554193901 * mV
        Vrest_BC = -74.74167987795019 * mV
        individual_swr = [w_PC_I_, w_BC_E_, w_BC_I_, wmx_mult_, w_PC_MF_, rate_MF_, g_leak_PC, tau_mem_PC, Cm_PC, Vrest_PC, Vrest_BC]
        SM_PC_swr, SM_BC_swr, RM_PC_swr, RM_BC_swr = self.generate_model(individual_swr, verbose=verbose)

        # reset simulation
        start_scope()

        # gamma parameters, change SWR ones using proportions from experimental paper
        print("GEN: {}\tWORKER: {}\tGAM".format(self.gen_id, multiprocessing.current_process().name))
        wmx_mult_ = (0.02 / 0.15) * wmx_mult_
        w_PC_I_ = (2.0 / 4.0) * w_PC_I_
        w_BC_E_ = (0.3 / 1.5) * w_BC_E_
        g_leak_PC = (2.5 / 3.3333) * 4.31475791937223 * nS
        tau_mem_PC = (80. / 60.) * 41.7488927175169 * ms
        Cm_PC = tau_mem_PC * g_leak_PC
        Vrest_PC = (-75.1884554193901 + 10.0) * mV
        Vrest_BC = (-74.74167987795019 + 10.0) * mV
        individual_gam = [w_PC_I_, w_BC_E_, w_BC_I_, wmx_mult_, w_PC_MF_, rate_MF_, g_leak_PC, tau_mem_PC, Cm_PC, Vrest_PC, Vrest_BC]
        SM_PC_gam, SM_BC_gam, RM_PC_gam, RM_BC_gam = self.generate_model(individual_gam, verbose=verbose)

        try:
            swr_errors = self.evaluate_errors_swr(SM_PC_swr, SM_BC_swr, RM_PC_swr, RM_BC_swr)
            gam_errors = self.evaluate_errors_gam(SM_PC_gam, SM_BC_gam, RM_PC_gam, RM_BC_gam)
            errors = swr_errors + gam_errors

            self.save_output(individual_swr, individual_gam, errors)
            self.gen_id += 1
            return errors
        except Exception:
            # Make sure exception and backtrace are thrown back to parent process
            raise Exception("".join(traceback.format_exception(*sys.exc_info())))
