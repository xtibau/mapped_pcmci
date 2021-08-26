import pickle
from tqdm import tqdm
from mpi4py import MPI

from experiment_classes import DmMethod
import tigramite.data_processing as pp
import numpy as np

# Default communicator
COMM = MPI.COMM_WORLD


def split(container, count):
    """
    Simple function splitting a the range of selected variables (or range(N))
    into equal length chunks. Order is not preserved.
    """
    return [container[i::count] for i in range(count)]


n_models = 100

# The master
if COMM.rank == 0:
    splitted_jobs = split(range(n_models), COMM.size)
else:
    splitted_jobs = None

scattered_jobs = COMM.scatter(splitted_jobs, root=0)

for n_model in scattered_jobs:
    print("Perfomirng model {}".format(n_model))

    # Config
    dm_methods_file = './experiment/dm_methods/dm_method_synthetic_{}.pkl'.format(n_model)
    results_file = './experiment/dm_methods/results/grid_results_synthetic_{}.pkl'.format(n_model)

    with open(dm_methods_file, "rb") as f:
        dm_method: DmMethod = pickle.load(f)

    dataframe = pp.DataFrame(dm_method.data_field.T)
    tau_max = dm_method.tau_max
    tau_min = 1
    T = dm_method.savar.time_length
    N = dm_method.savar.spatial_resolution

    pc_alpha = dm_method.pc_alpha
    parents_alpha = dm_method.parents_alpha
    max_conds_dim = None
    max_conds_px = None

    selected_variables = range(N)

    # Collect all results in dictionaries
    print("\nCollecting results...")

    pcmci_results = {}
    corr_results = {}
    for v in tqdm(selected_variables):
        piece_folder = "./temporal/model_synthetic_{}/".format(n_model)
        piece_file = piece_folder + "piece_synthetic_{}.pkl".format(v)

        with open(piece_file, "rb") as f:
            res = pickle.load(f)

        # PCMCI
        j, pcmci_results_j, _ = res
        for key in pcmci_results_j.keys():
            if key in ["p_matrix", "val_matrix", "conf_matrix"]:
                if pcmci_results_j[key] is None:
                    pcmci_results[key] = None
                else:
                    if key not in pcmci_results.keys():
                        if key == 'p_matrix':
                            pcmci_results[key] = np.ones(pcmci_results_j[key].shape)
                        else:
                            pcmci_results[key] = np.zeros(pcmci_results_j[key].shape)
                        pcmci_results[key][:, j, :] = pcmci_results_j[key][:, j, :]
                    else:
                        pcmci_results[key][:, j, :] = pcmci_results_j[key][:, j, :]

        # Corr
        j, _, corr_results_j = res
        for key in corr_results_j.keys():
            if key in ["p_matrix", "val_matrix", "conf_matrix"]:
                if corr_results_j[key] is None:
                    corr_results[key] = None
                else:
                    if key not in corr_results.keys():
                        if key == 'p_matrix':
                            corr_results[key] = np.ones(corr_results_j[key].shape)
                        else:
                            corr_results[key] = np.zeros(corr_results_j[key].shape)
                        corr_results[key][:, j, :] = corr_results_j[key][:, j, :]
                    else:
                        corr_results[key][:, j, :] = corr_results_j[key][:, j, :]

    pcmci_dict = {"p_matrix": pcmci_results['p_matrix'],
                  "val_matrix": pcmci_results['val_matrix'],
                  "conf_matrix": pcmci_results['conf_matrix']
                  }

    corr_dict = {"p_matrix": corr_results['p_matrix'],
                 "val_matrix": corr_results['val_matrix'],
                 "conf_matrix": corr_results['conf_matrix']
                 }

    with open(results_file, "wb") as f:
        pickle.dump((pcmci_dict, corr_dict), f)

    print("Model {} Finished!".format(n_model))
