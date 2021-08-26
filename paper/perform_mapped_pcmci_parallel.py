# This file takes the models and performs mapped-pcmci. Then returns the results

import pickle

from savar.savar import SAVAR
from experiment_classes import DmMethod
from mpi4py import MPI

# Default communicator
COMM = MPI.COMM_WORLD


def split(container, count):
    """
    Simple function splitting a the range of selected variables (or range(N))
    into equal length chunks. Order is not preserved.
    """
    return [container[i::count] for i in range(count)]


n_models = 100

#model_folder = "./experiment/data_small/"
model_folder = "./experiment/data_synthetic/"
dm_method_folder = './experiment/dm_methods/'

# The master
if COMM.rank == 0:
    splitted_jobs = split(range(n_models), COMM.size)
else:
    splitted_jobs = None

scattered_jobs = COMM.scatter(splitted_jobs, root=0)

for n in scattered_jobs:
    model_file = "{}savar_model_synthetic_{}.pkl".format(model_folder, n)
    dm_method_file = "{}dm_method_synthetic_{}.pkl".format(dm_method_folder, n)

    with open(model_file, "rb") as f:
        savar_model: SAVAR = pickle.load(f)

    print("Starting {} job".format(n))

    dm_method = DmMethod(savar_model)
    dm_method.perform_dm()
    dm_method.get_pcmci_results()
    dm_method.get_phi()

    with open(dm_method_file, "wb") as f:
        pickle.dump(dm_method, f, protocol=5)
