import os
import sys
import csv
from glob import glob
import pandas
import numpy
import seaborn
from matplotlib import pyplot as plt


data = []
for subdir in glob("{}/*/".format(sys.argv[1])):
    if "output" in subdir:
        continue
    with open("{}/params.txt".format(subdir), "r") as params:
        g1, g2, g3, g4 = [float(p.strip("\n")) for p in params.readlines()]

    wmx_mult = round(g1, 2)       # * wmx_PC_E
    w_PC_I = round(g2, 2)
    w_BC_E = round(g3, 2)
    w_BC_I = round(g4, 2)

    results_dict = {}
    with open("{}/results.txt".format(subdir), "r") as results_file:
        results = csv.reader(results_file, delimiter="=")
        for result in results:
            try:
                 results_dict[result[0]] = float(result[1])
            except TypeError:
                 results_dict[result[0]] = result[1]
    results_dict["wmx_mult"] = wmx_mult
    results_dict["w_PC_I"] = w_PC_I
    results_dict["w_BC_E"] = w_BC_E
    results_dict["w_BC_I"] = w_BC_I

    data.append(results_dict)

data_df = pandas.DataFrame(data)

wmx_mult_sorted = sorted(list(pandas.unique(data_df["wmx_mult"])))
w_PC_I_sorted = sorted(list(pandas.unique(data_df["w_PC_I"])), reverse=True)
w_BC_E_sorted = sorted(list(pandas.unique(data_df["w_BC_E"])))
w_BC_I_sorted = sorted(list(pandas.unique(data_df["w_BC_I"])), reverse=True)

def calculate_plausibility(variable):
    # calculate if ripple osc is biologically plausible or has too high a frequency
    # variable: peak_freq_PC, peak_freq_LFP, peak_freq_BC
    plausibles = []
    for datapoint in data_df[variable]:
        if datapoint > 150.0 and datapoint < 220.0:  # SWR range
            plausibles.append(1.0)
        elif datapoint > 25.0 and datapoint < 100.0: # gamma range
            plausibles.append(-1.0)
        else:
            plausibles.append(0.0)
    data_df.insert(len(data_df.columns), "is_{}_plausible".format(variable), plausibles)


def plot_variable_as_heatmap(variable, colormap="rocket", vmin=None, vmax=None):
    fig, axes = plt.subplots(3, 3, figsize=(12,12))
    fig.suptitle(variable)
    cbar_ax = fig.add_axes([.925, .3, .03, .4])

    # colorbar min max values
    vmax_, vmin_ = data_df[variable].max(), 0
    if vmax:
        vmax_ = vmax
    if vmin:
        vmin_ = vmin

    for outer_x, wmx_mult in enumerate(wmx_mult_sorted):
        axes[0][outer_x].set_title(wmx_mult, fontsize=20)
        for outer_y, w_PC_I in enumerate(w_PC_I_sorted):
            axes[:,0][outer_y].set_ylabel(w_PC_I, fontsize=20)
            heatmap_matrix = numpy.zeros((3,3))
            for inner_x, w_BC_E in enumerate(w_BC_E_sorted):
                for inner_y, w_BC_I in enumerate(w_BC_I_sorted):
                    datapoint = data_df[
                        (data_df["w_BC_E"] == w_BC_E) &
                        (data_df["w_BC_I"] == w_BC_I) &
                        (data_df["wmx_mult"] == wmx_mult) &
                        (data_df["w_PC_I"] == w_PC_I)
                    ]
                    val = datapoint[variable] #/ datapoint["ripple_power_LFP"]
                    heatmap_matrix[inner_y, inner_x] = float(val)
            seaborn.heatmap(heatmap_matrix, ax=axes[outer_y, outer_x],
                            vmin=vmin_, vmax=vmax_, cbar_ax=cbar_ax,
                            xticklabels = w_BC_E_sorted,
                            yticklabels = w_BC_I_sorted, cmap=colormap)
            axes[outer_y, outer_x].set_xlabel("w_BC_E")
            axes[outer_y, outer_x].set_ylabel("w_BC_I")
            fig.text(0.5, 0.94, 'wmx_mult', ha='center', fontsize=15)
            fig.text(0.04, 0.5, 'w_PC_I', va='center', rotation='vertical', fontsize=15)
    save_folder = "{}/output".format(sys.argv[1])
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    plt.savefig("{}/{}.png".format(save_folder, variable))

variables = ["absolute_gamma_power_PC",
             "relative_gamma_power_PC",
             "absolute_gamma_power_BC",
             "relative_gamma_power_BC",
             "absolute_gamma_power_LFP",
             "relative_gamma_power_LFP",
             "absolute_ripple_power_PC",
             "relative_ripple_power_PC",
             "absolute_ripple_power_BC",
             "relative_ripple_power_BC",
             "absolute_ripple_power_LFP",
             "relative_ripple_power_LFP"]

peak_vars = ["peak_freq_BC", "peak_freq_PC", "peak_freq_LFP"]
for variable in peak_vars:
    calculate_plausibility(variable)
    plot_variable_as_heatmap("is_{}_plausible".format(variable), colormap="coolwarm", vmin=-3.0, vmax=3.0)

total_plausibility = numpy.zeros(data_df.shape[0])
for variable in peak_vars:
    total_plausibility += data_df["is_{}_plausible".format(variable)]
data_df.insert(len(data_df.columns), "total_plausibility", total_plausibility)
plot_variable_as_heatmap("total_plausibility", colormap="coolwarm", vmin=-3.0, vmax=3.0)

for variable in variables:
    if "relative" in variable:
        plot_variable_as_heatmap(variable, vmin=0, vmax=100)
        plot_variable_as_heatmap(variable, vmin=0, vmax=100)
    else:
        plot_variable_as_heatmap(variable)
        plot_variable_as_heatmap(variable)