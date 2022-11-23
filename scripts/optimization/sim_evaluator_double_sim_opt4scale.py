# -*- coding: utf8 -*-
"""
BluePyOpt evaluator for optimization
authors: Bence Bagi, András Ecker, Szabolcs Káli last update: 12.2021
"""

import os, sys, traceback, gc
import numpy as np
from brian2 import *
import bluepyopt as bpop
base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-3])
# add "scripts" directory to the path (to import modules)
sys.path.insert(0, os.path.sep.join([base_path, "scripts"]))
from helper import preprocess_monitors
from detect_oscillations import analyse_rate, gamma, lowfreq, ripple
from detect_replay import slice_high_activity, replay_circular
import multiprocessing


class Brian2Evaluator(bpop.evaluators.Evaluator):
    """Evaluator class required by BluePyOpt"""

    def __init__(self, linear, Wee, params, gen_id=1):
        """
        :param Wee: weight matrix (passing Wee with cPickle to the slaves (as BluPyOpt does) is still the fastest solution)
        :param params: list of parameters to fit - every entry must be a tuple: (name, lower bound, upper bound)
        """
        super(Brian2Evaluator, self).__init__()
        self.linear = linear
        self.Wee = Wee
        self.params = params
        self.gen_id = gen_id

        # Parameters to be optimized
        self.params = [bpop.parameters.Parameter(name, bounds=(minval, maxval))
                       for name, minval, maxval in self.params]
        self.swr_objectives = ["ripple_peakE", "ripple_peakI", "no_gamma_power_PC", "no_gamma_power_BC", "ripple_powerE", "ripple_powerI", "ripple_rateE", "ripple_rateI"]
        self.gam_objectives = ["gamma_peakE", "gamma_peakI",  "no_subgamma_power_PC", "no_subgamma_power_BC", "gamma_powerE", "gamma_powerI", "gamma_rateE", "gamma_rateI"]
        self.exp_objectives = ["diff_wmx_mult", "diff_w_PC_I", "diff_w_BC_E", "diff_w_BC_I"]
        self.objectives = self.swr_objectives + self.gam_objectives + self.exp_objectives

    def generate_model(self, individual, verbose=False):
        from run_sim import run_simulation

        """Runs single simulation (see `run_sim.py`) and returns monitors"""
        SM_PC, SM_BC, RM_PC, RM_BC = run_simulation(self.Wee, *individual, verbose=verbose)
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
            mean_rate_PC, rate_ac_PC, max_ac_PC, t_max_ac_PC, f_PC, Pxx_PC = analyse_rate(rate_PC, 1000.0, slice_idx)
            mean_rate_BC, rate_ac_BC, max_ac_BC, t_max_ac_BC, f_BC, Pxx_BC = analyse_rate(rate_BC, 1000.0, slice_idx)
            avg_ripple_freq_PC, relative_ripple_power_PC = ripple(f_PC, Pxx_PC, slice_idx, p_th=1) # p_th set to 1 so that any swr oscillation will be considered significant
            avg_ripple_freq_BC, relative_ripple_power_BC = ripple(f_BC, Pxx_BC, slice_idx, p_th=1)
            avg_gamma_freq_PC, _, relative_gamma_power_PC = gamma(f_PC, Pxx_PC, slice_idx, lb=20)
            avg_gamma_freq_BC, _, relative_gamma_power_BC = gamma(f_BC, Pxx_BC, slice_idx, lb=20)

            # look for significant ripple peak close to 180 Hz
            ripple_peakE = np.exp(-1 / 2 * (avg_ripple_freq_PC - 180.) ** 2 / 20 ** 2) if not np.isnan(
                avg_ripple_freq_PC) else 0.
            ripple_peakI = np.exp(-1 / 2 * (avg_ripple_freq_BC - 180.) ** 2 / 20 ** 2) if not np.isnan(
                avg_ripple_freq_BC) else 0.
            # penalize gamma power
            no_gamma_power_PC = 1-relative_gamma_power_PC/100.
            no_gamma_power_BC = 1-relative_gamma_power_BC/100.
            # look for "low" exc. population rate (around 2.5 Hz)
            ripple_rateE = np.exp(-1 / 2 * (mean_rate_PC - 1.0) ** 2 / 3.0 ** 2)    # from Hájos et al 2013
            ripple_rateI = np.exp(-1 / 2 * (mean_rate_PC - 10.0) ** 2 / 10.0 ** 2)   # from Hájos et al 2013
            # *-1 since the algorithm tries to minimize...
            errors = -1. * np.array([ripple_peakE, ripple_peakI,
                                     2 * no_gamma_power_PC, 2 * no_gamma_power_BC,
                                     2 * relative_ripple_power_PC / 100., 2 * relative_ripple_power_BC / 100.,
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
            mean_rate_PC, rate_ac_PC, max_ac_PC, t_max_ac_PC, f_PC, Pxx_PC = analyse_rate(rate_PC, 1000.0, slice_idx,
                                                                                          normalize=True)
            mean_rate_BC, rate_ac_BC, max_ac_BC, t_max_ac_BC, f_BC, Pxx_BC = analyse_rate(rate_BC, 1000.0, slice_idx,
                                                                                          normalize=True)
            avg_gamma_freq_PC, _, relative_gamma_power_PC = gamma(f_PC, Pxx_PC, slice_idx, p_th=1, lb=20)  # p_th set to 1 so that any gamma oscillation will be considered significant
            avg_gamma_freq_BC, _, relative_gamma_power_BC = gamma(f_BC, Pxx_BC, slice_idx, p_th=1, lb=20)
            avg_subgamma_freq_PC, subgamma_power_PC = lowfreq(f_PC, Pxx_PC, slice_idx, gamma_lb=30)
            avg_subgamma_freq_BC, subgamma_power_BC = lowfreq(f_BC, Pxx_BC, slice_idx, gamma_lb=30)

            # look for significant gamma peak close to 30 Hz
            gamma_peakE = np.exp(-1 / 2 * (avg_gamma_freq_PC - 30.) ** 2 / 10 ** 2) if not np.isnan(avg_gamma_freq_PC) else 0.
            gamma_peakI = np.exp(-1 / 2 * (avg_gamma_freq_BC - 30.) ** 2 / 10 ** 2) if not np.isnan(avg_gamma_freq_BC) else 0.
            # penalize sub gamma power
            no_subgamma_power_PC = 1.-subgamma_power_PC/100.
            no_subgamma_power_BC = 1.-subgamma_power_BC/100.
            # look for "low" exc. population rate (around 1.0 Hz)
            gamma_rateE = np.exp(-1 / 2 * (mean_rate_PC - 5.0) ** 2 / 3.0 ** 2)    # search for mean of 5.0 with a std of 3.0
            gamma_rateI = np.exp(-1 / 2 * (mean_rate_BC - 25.0) ** 2 / 10.0 ** 2)   # search for mean of 25. with a std. of 10.0
            # *-1 since the algorithm tries to minimize...
            errors = -1. * np.array([gamma_peakE, gamma_peakI,
                                     2 * no_subgamma_power_PC, 2 * no_subgamma_power_BC,
                                     2 * relative_gamma_power_PC / 100., 2 * relative_gamma_power_BC / 100.,
                                     gamma_rateE, gamma_rateI])
            return errors.tolist()
        else:
            return [0.]*len(self.gam_objectives)

    def evaluate_errors_exp(self, individual):
        w_PC_I_, w_BC_E_, w_BC_I_, wmx_mult_, _, _, _, _, _, _ = individual

        # preCCh/SWR-presenting conductances (from various papers)
        g_experimental = {'w_BC_E': 4.5,
                          'w_BC_I': 3.95,
                          'w_PC_E': 0.54,
                          'w_PC_I': 5.28}

        sigma = 0.5
        wmx_mult_exp = g_experimental["w_PC_E"] / 0.2

        diff_wmx_mult = np.exp(-1 / 2 * ((wmx_mult_ - wmx_mult_exp) ** 2 / sigma ** 2))
        diff_w_PC_I = np.exp(-1 / 2 * ((w_PC_I_ - g_experimental['w_PC_I']) ** 2 / sigma ** 2))
        diff_w_BC_E = np.exp(-1 / 2 * ((w_BC_E_ - g_experimental['w_BC_E']) ** 2 / sigma ** 2))
        diff_w_BC_I = np.exp(-1 / 2 * ((w_BC_I_ - g_experimental['w_BC_I']) ** 2 / sigma ** 2))

        errors = -1. * np.array([0.25 * diff_wmx_mult,
                                 0.25 * diff_w_PC_I,
                                 0.25 * diff_w_BC_E,
                                 0.25 * diff_w_BC_I])  # weighted by .25 so that it won't overpower the rest
        return errors.tolist()

    def save_output(self, individual_swr, individual_gam, errors):
        if not os.path.isdir("./errors"):
            os.mkdir("./errors")
        with open("errors/{}_errors.txt".format(multiprocessing.current_process().name), "w") as errors_file:
            for idx_obj, obj in enumerate(self.objectives):
                errors_file.writelines([obj, "=", str(errors[idx_obj]), "\n"])
        param_names = ["w_PC_I_", "w_BC_E_", "w_BC_I_", "wmx_mult_", "w_PC_MF_", "rate_MF_", "g_leak_PC", "tau_mem_PC", "Cm_PC", "Vrest_PC", "Vrest_BC"]
        with open("errors/{}_params.txt".format(multiprocessing.current_process().name), "w") as params_file:
            params_file.writelines(["param_name", ",", "swr", ",", "gamma", "\n"])
            for idx_param, param_name in enumerate(param_names):
                params_file.writelines([param_name, ",", str(individual_swr[idx_param]), ",", str(individual_gam[idx_param]), "\n"])

    def evaluate_with_lists(self, individual, verbose=False, plots=False):
        """Fitness error used by BluePyOpt for the optimization"""

        # original, SWR parameters
        print("WORKER: {}\tSWR".format(multiprocessing.current_process().name))
        w_PC_I_, w_BC_E_, w_BC_I_, wmx_mult_, w_PC_MF_, rate_MF_, s_PC_E, s_PC_I, s_BC_E, s_BC_I = individual
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
        print("WORKER: {}\tGAM".format(multiprocessing.current_process().name))
        wmx_mult_ = s_PC_E * wmx_mult_
        w_PC_I_ = s_PC_I * w_PC_I_
        w_BC_E_ = s_BC_E * w_BC_E_
        w_BC_I_ = s_BC_I * w_BC_I_

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
            exp_errors = self.evaluate_errors_exp(individual)
            errors = swr_errors + gam_errors + exp_errors

            self.save_output(individual_swr, individual_gam, errors)
            self.gen_id += 1
            return errors
        except Exception:
            # Make sure exception and backtrace are thrown back to parent process
            raise Exception("".join(traceback.format_exception(*sys.exc_info())))
