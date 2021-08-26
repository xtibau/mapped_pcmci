# This file, joints the eval methods of mapped-pcmci and grid level.
import pickle
from copy import deepcopy
import numpy as np
from mpi4py import MPI

from tigramite.independence_tests import ParCorr
from tigramite.pcmci import PCMCI
from tigramite.models import LinearMediation
import tigramite.data_processing as pp

from experiment_classes import Evaluation


def split(container, count):
    """
    Simple function splitting a the range of selected variables (or range(N))
    into equal length chunks. Order is not preserved.
    """
    return [container[i::count] for i in range(count)]


def fix_grid_phi(grid_results, dm_method, verbose=True):
    """
    Sets Grid as expected by the evaluation class.
    Move time to beginning and fill the first time-step with ones in the diagonal.
    """
    if verbose:
        print("Starting the fixing function")
    pcmi_dict, corr_dict = grid_results
    dataframe = pp.DataFrame(dm_method.data_field.T)

    dm_method.tg_grid_results["corr"] = corr_dict

    # Corr
    if verbose:
        print("created pcmci for corr")
    dm_method.grid_pcmci["corr"] = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr(),
        verbosity=False
    )
    corr_grd_results = deepcopy(dm_method.tg_grid_results["corr"])
    variance_vars = dm_method.grid_pcmci["corr"].dataframe.values.std(axis=0)

    # Get Phi from val_matrix
    Phi = corr_grd_results['val_matrix']

    # If p_value not enought set it to 0
    Phi[[corr_grd_results['p_matrix'] > dm_method.pc_alpha]] = 0

    # Now we do the coefficient by Val_matrix[i, j, tau]*std(j)/std(i)
    Phi = (Phi * variance_vars[:, None]) / variance_vars[:, None, None]

    dm_method.grid_phi["corr"] = np.moveaxis(deepcopy(Phi), 2, 0)
    np.fill_diagonal(dm_method.grid_phi["corr"][0, ...], 1)  # Fill the diagonal of tau 0 with ones

    if verbose:
        print("Phi for corr, obtained")

    # PCMCI
    if verbose:
        print("Staring pcmci for pcmci")
    dm_method.grid_pcmci["pcmci"] = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr(),
        verbosity=True,
    )

    dm_method.tg_grid_results["pcmci"] = pcmi_dict

    if verbose:
        print("Getting parents")

    dm_method.parents_predict["pcmci"] = dm_method.grid_pcmci["pcmci"].return_significant_links(
        pq_matrix=dm_method.tg_grid_results["pcmci"]["p_matrix"],
        val_matrix=dm_method.tg_grid_results["pcmci"]["val_matrix"],
        alpha_level=dm_method.parents_alpha,
        include_lagzero_links=False,
    )["link_dict"]

    # Get grid phi
    if verbose:
        print("Getting linear mediators")
    med = LinearMediation(dataframe=dataframe)
    if verbose:
        print("Fitting model")
    med.fit_model(all_parents=dm_method.parents_predict["pcmci"], tau_max=dm_method.tau_max)
    dm_method.grid_phi["pcmci"] = med.phi

    # Set it so its not performed again
    dm_method.is_grid_done = True

    if verbose:
        print("Finished")
    return dm_method

models = range(100)

# Default communicator
COMM = MPI.COMM_WORLD

if COMM.rank == 0:
    splitted_jobs = split(models, COMM.size)
    print("Split selected_variables = {}".format(splitted_jobs))
else:
    splitted_jobs = None

scattered_jobs = COMM.scatter(splitted_jobs, root=0)

for n_model in scattered_jobs:
    dm_method_file = "./experiment/dm_methods/dm_method_synthetic_{}.pkl".format(n_model)
    dm_grid_results = "./experiment/dm_methods/results/grid_results_synthetic_{}.pkl".format(n_model)
    results_file_dict = "./experiment/dm_methods/results/final_results_synthetic_{}.pkl".format(n_model)

    with open(dm_method_file, "rb") as f:
        dm_method = pickle.load(f)

    with open(dm_grid_results, "rb") as f:
        grid_results = pickle.load(f)

    dm_method = fix_grid_phi(grid_results, dm_method)
    eval = Evaluation(dm_method, None, grid_threshold=0)
    eval._obtain_grid_metrics()
    with open(results_file_dict, "wb") as f:
        pickle.dump(eval, f, protocol=5)
