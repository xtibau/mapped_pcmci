# This script performs the pcmci at grid level parallelizing for each variable
# Pipeline. We get the results of DmMethod.tg_grid_results. which is a dict with "_matrix" and
# "val_matrix" as output of this script.
# Starting from a DmMethod with specific name in specific folder, we compute those values.
# Ths script is based on the script of Tigramite
import numpy as np
from mpi4py import MPI
import argparse
import numpy
import pickle
from pathlib import Path
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr
import tigramite.data_processing as pp


def split(container, count):
    """
    Simple function splitting a the range of selected variables (or range(N))
    into equal length chunks. Order is not preserved.
    """
    return [container[i::count] for i in range(count)]


def run_pcmciplus_parallel(j):
    # Instatntiete PCMCI
    print("var {} staring PCMCI".format(j))
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        verbosity=verbosity
    )

    # Run pcmciplus
    # print(selected_links[j])
    pcmci_res = pcmci.run_pcmciplus(selected_links=selected_links[j],
                                    tau_min=tau_min,
                                    tau_max=tau_max,
                                    pc_alpha=pc_alpha,
                                    )

    print("var {} going for correlation".format(j))

    corr_res = pcmci.get_lagged_dependencies(
        selected_links=selected_links[j],
        tau_min=tau_min,
        tau_max=tau_max,
        val_only=False
    )

    print("Var {} WORKD DONE!!".format(j))


    # Save the results in an intermediary folder
    piece_folder = "./temporal/model_synthetic_{}/".format(n_model)
    piece_file = piece_folder + "piece_synthetic_{}.pkl".format(j)

    Path(piece_folder).mkdir(parents=True, exist_ok=True)
    with open(piece_file, "wb") as f:
        pickle.dump((j, pcmci_res, corr_res), f, protocol=5)

    # return j, pcmci_res, corr_res


# Default communicator
COMM = MPI.COMM_WORLD

from experiment_classes import DmMethod

parser = argparse.ArgumentParser()
parser.add_argument('-n', "--n_model", type=int)
args = parser.parse_args()
n_model = args.n_model

test = False
gather = False  # Then, we use another script to put everything together

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

if test:
    numpy.random.seed(42)  # Fix random seed
    links_coeffs = {0: [((0, -1), 0.7)],
                    1: [((1, -1), 0.8), ((0, -1), 0.8)],
                    2: [((2, -1), 0.5), ((1, -2), 0.5)],
                    }

    T = 500  # time series length
    data, true_parents_neighbors = pp.var_process(links_coeffs, T=T)
    T, N = data.shape

    # Optionally specify variable names
    var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^3$']

    # Initialize dataframe object
    dataframe = pp.DataFrame(data, var_names=var_names)


cond_ind_test, confidence = ParCorr(), 'analytic'
verbosity = False
selected_links = {n: {m: [(i, -t) for i in range(N) for \
                          t in range(tau_min, tau_max)] if m == n else [] for m in range(N)} for n in range(N)}
selected_variables = range(N)

if COMM.rank == 0:
    # Only the master node (rank=0) runs this
    print("\n##\n## Running Parallelized Tigramite PC algorithm\n##"
          "\n\nParameters:")
    print("\nindependence test = %s" % cond_ind_test.measure
          + "\ntau_min = %d" % tau_min
          + "\ntau_max = %d" % tau_max
          + "\npc_alpha = %s" % pc_alpha
          + "\nmax_conds_dim = %s" % max_conds_dim)
    print("\n")

    # Split selected_variables into however many cores are available.
    splitted_jobs = split(selected_variables, COMM.size)
    print("Split selected_variables = {}".format(splitted_jobs))
else:
    splitted_jobs = None

##
##  PC algo condition-selection step
##
# Scatter jobs across cores.
scattered_jobs = COMM.scatter(splitted_jobs, root=0)

for j in scattered_jobs:
    # Estimate conditions
    run_pcmciplus_parallel(j)

if gather:

    # TO POLISH
    if COMM.rank == 0:
        # Collect all results in dictionaries
        print("\nCollecting results...")

        results = []
        for j in selected_variables:
            piece_folder = "./temporal/model_synthetic_{}/".format(n_model)
            piece_file = piece_folder + "piece_synthetic_{}.pkl".format(j)

            with open(piece_file, "rb") as f:
                j, pcmci_res, corr_res = pickle.load(f)


        # PCMCI Results
        pcmci_results = {}
        for res in results:
            for j, pcmci_results_j, _ in res:
                for key in pcmci_results_j.keys():
                    if key in ["p_matrix", "val_matrix", "conf_matrix"]:
                        if pcmci_results_j[key] is None:
                            pcmci_results[key] = None
                        else:
                            if key not in pcmci_results.keys():
                                if key == 'p_matrix':
                                    pcmci_results[key] = np.ones(pcmci_results_j[key].shape)
                                else:
                                    pcmci_results[key] = numpy.zeros(pcmci_results_j[key].shape)
                                pcmci_results[key][:, j, :] = pcmci_results_j[key][:, j, :]
                            else:
                                pcmci_results[key][:, j, :] = pcmci_results_j[key][:, j, :]

        #Corr results
        # PCMCI Results
        corr_results = {}
        for res in results:
            for j, _, corr_results_j in res:
                for key in corr_results_j.keys():
                    if key in ["p_matrix", "val_matrix", "conf_matrix"]:
                        if corr_results_j[key] is None:
                            corr_results[key] = None
                        else:
                            if key not in corr_results.keys():
                                if key == 'p_matrix':
                                    corr_results[key] = np.ones(corr_results_j[key].shape)
                                else:
                                    corr_results[key] = numpy.zeros(corr_results_j[key].shape)
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

        print("Work Done!")
