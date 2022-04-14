# -*- coding: utf8 -*-
"""
BluePyOpt evaluator for optimization
authors: Bence Bagi, András Ecker, Szabolcs Káli last update: 12.2021
"""

import os, sys, traceback, gc
import numpy as np
import bluepyopt as bpop
import run_sim as sim
base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-3])
# add "scripts" directory to the path (to import modules)
sys.path.insert(0, os.path.sep.join([base_path, "scripts"]))
from helper import preprocess_monitors
from detect_oscillations import analyse_rate, gamma, lowfreq
from detect_replay import slice_high_activity, replay_circular


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

        # Parameters to be optimized
        self.params = [bpop.parameters.Parameter(name, bounds=(minval, maxval))
                       for name, minval, maxval in self.params]
        self.objectives = ["gamma_peakE", "gamma_peakI",  "no_subgamma_peakE", "no_subgamma_peakI",
                           "gamma_ratioE", "gamma_ratioI", "rateE", "no_replay", "absolute_gamma_power_PC",
                           "absolute_gamma_power_BC"]

    def generate_model(self, individual, verbose=False):
        """Runs single simulation (see `run_sim.py`) and returns monitors"""
        SM_PC, SM_BC, RM_PC, RM_BC = sim.run_simulation(self.Wee, *individual, verbose=verbose)
        return SM_PC, SM_BC, RM_PC, RM_BC

    def evaluate_with_lists(self, individual, verbose=False, plots=False):
        """Fitness error used by BluePyOpt for the optimization"""
        SM_PC, SM_BC, RM_PC, RM_BC = self.generate_model(individual, verbose=verbose)

        try:
            wc_errors = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]  # worst case scenario
            if SM_PC.num_spikes > 0 and SM_BC.num_spikes > 0:  # check if there is any activity
                # analyse spikes
                spike_times_PC, spiking_neurons_PC, rate_PC, ISI_hist_PC, bin_edges_PC = preprocess_monitors(SM_PC, RM_PC)
                spike_times_BC, spiking_neurons_BC, rate_BC = preprocess_monitors(SM_BC, RM_BC, calc_ISI=False)
                del SM_PC, SM_BC, RM_PC, RM_BC
                gc.collect()
                # analyse rates
                slice_idx = [] #if not self.linear else slice_high_activity(rate_PC, th=2, min_len=260)
                mean_rate_PC, rate_ac_PC, max_ac_PC, t_max_ac_PC, f_PC, Pxx_PC = analyse_rate(rate_PC, 1000.0, slice_idx)
                mean_rate_BC, rate_ac_BC, max_ac_BC, t_max_ac_BC, f_BC, Pxx_BC = analyse_rate(rate_BC, 1000.0, slice_idx)
                avg_gamma_freq_PC, absolute_gamma_power_PC, relative_gamma_power_PC = gamma(f_PC, Pxx_PC, slice_idx)
                avg_gamma_freq_BC, absolute_gamma_power_BC, relative_gamma_power_BC = gamma(f_BC, Pxx_BC, slice_idx)
                avg_subgamma_freq_PC, subgamma_power_PC = lowfreq(f_PC, Pxx_PC, slice_idx)
                avg_subgamma_freq_BC, subgamma_power_BC = lowfreq(f_BC, Pxx_BC, slice_idx)

                # these 2 flags are only used for the last rerun, but not during the optimization
                if verbose:
                    print("Mean excitatory rate: %.3f" % mean_rate_PC)
                    print("Mean inhibitory rate: %.3f" % mean_rate_BC)
                    print("Average exc. gamma freq: %.3f" % avg_gamma_freq_PC)
                    #print("Exc. gamma power: %.3f" % gamma_power_PC)
                    print("Average inh. gamma freq: %.3f" % avg_gamma_freq_BC)
                    #print("Inh. gamma power: %.3f" % gamma_power_BC)
                if plots:
                    from plots import plot_raster, plot_PSD, plot_zoomed
                    plot_raster(spike_times_PC, spiking_neurons_PC, rate_PC, [ISI_hist_PC, bin_edges_PC],
                                slice_idx, "blue", multiplier_=1)
                    plot_PSD(rate_PC, rate_ac_PC, f_PC, Pxx_PC, "PC_population", "blue", multiplier_=1)
                    plot_PSD(rate_BC, rate_ac_BC, f_BC, Pxx_BC, "BC_population", "green", multiplier_=1)
                    _ = plot_zoomed(spike_times_PC, spiking_neurons_PC, rate_PC, "PC_population", "blue", multiplier_=1)
                    plot_zoomed(spike_times_BC, spiking_neurons_BC, rate_BC, "BC_population", "green",
                                multiplier_=1, PC_pop=False)

                # look for significant gamma peak close to 30 Hz
                gamma_peakE = 2 * np.exp(-1/2*(avg_gamma_freq_PC-30.)**2/10**2) if not np.isnan(avg_gamma_freq_PC) else 0.
                gamma_peakI = np.exp(-1/2*(avg_gamma_freq_BC-30.)**2/10**2) if not np.isnan(avg_gamma_freq_BC) else 0.
                # penalize sub gamma peaks - binary variable, which might not be the best for this algo.
                no_subgamma_peakE = 1. if np.isnan(avg_subgamma_freq_PC) else 0.
                no_subgamma_peakI = 1. if np.isnan(avg_subgamma_freq_BC) else 0.
                # look for high gamma/sub gamma (alpha, beta) power ratio
                gamma_ratioE = np.clip(relatve_gamma_power_PC/subgamma_power_PC, 0., 5.)
                gamma_ratioI = np.clip(relatve_gamma_power_BC/subgamma_power_BC, 0., 5.)
                # look for "low" exc. population rate (around 1.0 Hz)
                rateE = np.exp(-1/2*(mean_rate_PC-1.0)**2/0.5**2)
                # penalize replay (only in circular env)
                if not self.linear:
                    replay_ROI = np.where((150 <= bin_edges_PC) & (bin_edges_PC <= 850))
                    no_replay = 3 if np.isnan(replay_circular(ISI_hist_PC[replay_ROI])) else 0.
                else:
                    no_replay = 0.
                # *-1 since the algorithm tries to minimize...
                errors = -1. * np.array([gamma_peakE, gamma_peakI, no_subgamma_peakE, no_subgamma_peakI,
                                         gamma_ratioE, gamma_ratioI,  rateE, no_replay, np.log(absolute_gamma_power_BC),
                                         np.log(absolute_gamma_power_PC)])
                filename = "_".join(list([str(i) for i in individual]))
                with open("./errors/{}.txt".format(filename), "w") as errors_file:
                    errors_file.writelines(["gamma_peakE=", str(gamma_peakE), "\n",
                                            "gamma_peakI=", str(gamma_peakI), "\n",
                                            "no_subgamma_peakE=", str(no_subgamma_peakE), "\n",
                                            "no_subgamma_peakI=", str(no_subgamma_peakI), "\n",
                                            "gamma_ratioE=", str(gamma_ratioE), "\n",
                                            "gamma_ratioI=", str(gamma_ratioI), "\n",
                                            "rateE=", str(rateE), "\n",
                                            "no_replay=", str(no_replay), "\n",
                                            "absolute_gamma_power_PC", str(absolute_gamma_power_PC), "\n",
                                            "absolute_gamma_power_BC", str(absolute_gamma_power_BC)])
                return errors.tolist()
            else:
                return wc_errors
        except Exception:
            # Make sure exception and backtrace are thrown back to parent process
            raise Exception("".join(traceback.format_exception(*sys.exc_info())))
