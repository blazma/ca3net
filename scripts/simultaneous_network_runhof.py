import csv
from distutils.dir_util import copy_tree
from pathlib import Path
from simultaneous_network import *
from optimization.sim_evaluator_double_sim_opt4scale import Brian2Evaluator


evaluator = Brian2Evaluator(linear=None, Wee=None, params=[])  # no need to actually use it, only to access a function
seed = 12345
save = False
verbose = True
f_in = "wmx_sym_0.5_linear.pkl"
wmx_PC_E = load_wmx(os.path.join(base_path, "files", f_in)) * 1e9  # *1e9 nS conversion
hof_filename = "hof_opt4scale_nov16"

with open(f"../hof/{hof_filename}.csv") as hof_file:
    hof_content = csv.reader(hof_file)
    next(hof_content)
    for line in hof_content:
        id, *params = line
        params = [float(param) for param in params]
        w_PC_I, w_BC_E, w_BC_I, wmx_mult, w_PC_MF, rate_MF, s_PC_E, s_PC_I, s_BC_E, s_BC_I = params
        rate_MF = rate_MF * Hz

        # create output subdirectories
        subdir_path = Path(f"../hof/{id}")
        swr_subdir_path = Path(f"../hof/{id}/swr")
        gam_subdir_path = Path(f"../hof/{id}/gam")
        for path in [subdir_path, swr_subdir_path, gam_subdir_path]:
            path.mkdir(exist_ok=True)

        # run swr simulation, copy output to subdirectories
        params_swr = [w_PC_I, w_BC_E, w_BC_I, wmx_mult, w_PC_MF, rate_MF]
        SM_PC_swr, SM_BC_swr, RM_PC_swr, RM_BC_swr = swr_network(params_swr, wmx_PC_E, save, seed, verbose)
        copy_tree("../figures", str(swr_subdir_path))

        # run gamma simulation, copy output to subdirectories
        params_gam = params_swr + [s_PC_E, s_PC_I, s_BC_E, s_BC_I]
        SM_PC_gam, SM_BC_gam, RM_PC_gam, RM_BC_gam = gamma_network(params_gam, wmx_PC_E, save, seed, verbose)
        copy_tree("../figures", str(gam_subdir_path))

        # calculate and save errors
        swr_errors = evaluator.evaluate_errors_swr(SM_PC_swr, SM_BC_swr, RM_PC_swr, RM_BC_swr)
        gam_errors = evaluator.evaluate_errors_gam(SM_PC_gam, SM_BC_gam, RM_PC_gam, RM_BC_gam)
        errors = swr_errors + gam_errors
        with open(f"{str(subdir_path)}/errors.txt", "w") as errors_file:
            for idx_obj, obj in enumerate(evaluator.objectives):
                errors_file.writelines([obj, "=", str(errors[idx_obj]), "\n"])
