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
    with open("{}/params.txt".format(subdir), "r") as params:
        g1, g2, g3, g4 = [float(p.strip("\n")) for p in params.readlines()]

    w_PC_E = round(g1 * (0.02 / 0.15) * 1, 2)       # * wmx_PC_E
    w_PC_I = round(g2 * (2.0 / 4.0) * 0.65, 2)
    w_BC_E = round(g3 * (0.3 / 1.5) * 0.85, 2)
    w_BC_I = round(g4 * (0.1) * 5., 2)

    results_dict = {}
    with open("{}/results.txt".format(subdir), "r") as results_file:
        results = csv.reader(results_file, delimiter="=")
        for result in results:
            try:
                 results_dict[result[0]] = float(result[1])
            except TypeError:
                 results_dict[result[0]] = result[1]
    results_dict["w_PC_E"] = w_PC_E
    results_dict["w_PC_I"] = w_PC_I
    results_dict["w_BC_E"] = w_BC_E
    results_dict["w_BC_I"] = w_BC_I

    data.append(results_dict)

data_df = pandas.DataFrame(data)
#del data_df['multiplier'], data_df['np.nan']

pairs = [
    ("w_PC_E", "w_PC_I"),
    ("w_PC_E", "w_BC_E"),
    ("w_PC_E", "w_BC_I"),
    ("w_PC_I", "w_BC_E"),
    ("w_PC_I", "w_BC_I"),
    ("w_BC_E", "w_BC_I")
]

#fig, axes = plt.subplots(3,2)
#cbar_ax = fig.add_axes([.91, .3, .03, .4])
#vmax = max([data_df.groupby([w1, w2]).var()["gamma_power_PC"].max() for w1, w2 in pairs])
#idx = 0
#for w1, w2 in pairs:
#    data_grouped = data_df.groupby([w1, w2]).var()["gamma_power_PC"].reset_index().pivot(w1, w2, "gamma_power_PC")
#    seaborn.heatmap(data_grouped, ax=axes[idx % 3,idx % 2], vmin=0, vmax=vmax, cbar_ax=cbar_ax)
#    idx += 1
#plt.bar(list(range(len(data))), gamma_power_PC)

w_PC_E_sorted = sorted(list(pandas.unique(data_df["w_PC_E"])))
w_PC_I_sorted = sorted(list(pandas.unique(data_df["w_PC_I"])), reverse=True)
w_BC_E_sorted = sorted(list(pandas.unique(data_df["w_BC_E"])))
w_BC_I_sorted = sorted(list(pandas.unique(data_df["w_BC_I"])), reverse=True)


def plot_variable_as_heatmap(variable):
    fig, axes = plt.subplots(3, 3)
    fig.suptitle(variable)
    cbar_ax = fig.add_axes([.95, .3, .03, .4])
    vmax = data_df[variable].max() #/ data_df["ripple_power_LFP"]).max()
    for outer_x, w_PC_E in enumerate(w_PC_E_sorted):
        axes[0][outer_x].set_title(w_PC_E, fontsize=20)
        for outer_y, w_PC_I in enumerate(w_PC_I_sorted):
            axes[:,0][outer_y].set_ylabel(w_PC_I, fontsize=20)
            heatmap_matrix = numpy.zeros((3,3))
            for inner_x, w_BC_E in enumerate(w_BC_E_sorted):
                for inner_y, w_BC_I in enumerate(w_BC_I_sorted):
                    datapoint = data_df[
                        (data_df["w_BC_E"] == w_BC_E) &
                        (data_df["w_BC_I"] == w_BC_I) &
                        (data_df["w_PC_E"] == w_PC_E) &
                        (data_df["w_PC_I"] == w_PC_I)
                    ]
                    val = datapoint[variable] #/ datapoint["ripple_power_LFP"]
                    heatmap_matrix[inner_y, inner_x] = float(val)
            seaborn.heatmap(heatmap_matrix, ax=axes[outer_y, outer_x], vmin=0, vmax=vmax, cbar_ax=cbar_ax,
                            xticklabels = w_BC_E_sorted,
                            yticklabels = w_BC_I_sorted)
            axes[outer_y, outer_x].set_xlabel("w_BC_E")
            axes[outer_y, outer_x].set_ylabel("w_BC_I")
            fig.text(0.5, 0.94, 'w_PC_E', ha='center', fontsize=15)
            fig.text(0.04, 0.5, 'w_PC_I', va='center', rotation='vertical', fontsize=15)

variables = ["absolute_gamma_power_PC",
             "relative_gamma_power_PC",
             "absolute_gamma_power_BC",
             "relative_gamma_power_BC",
             "absolute_gamma_power_LFP",
             "relative_gamma_power_LFP"]

for variable in variables:
    plot_variable_as_heatmap(variable)
    plot_variable_as_heatmap(variable)
plt.show()
