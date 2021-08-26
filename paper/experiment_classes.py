import pickle
from copy import deepcopy
import itertools as it
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

from tigramite.independence_tests import ParCorr
from tigramite.models import LinearMediation
from tigramite.pcmci import PCMCI
import tigramite.data_processing as pp

from savar.savar import SAVAR
from savar.functions import cg_to_est_phi
from varimax_plus.varimax_plus import VarimaxPlus

# Internals
from mapped_pcmci.functions import find_permutation


# Volem DM method, that uses PCA and Varimax, to get the signals
# Volem la Evaluation class que fara l'output dels resultats que volem.

class DmMethod:
    """
        This method handles the different methods evaluated. Is the input of the evaluation class.
        Performs the different experimetns

        """

    def __init__(self, savar: SAVAR, tau_max: int = None, max_comps: int = None,
                 significance: str = "analytic", ind_test: str = "ParCorr", pc_alpha: float = 0.2,
                 parents_alpha: float = 0.05, verbose: bool = True):

        """

        :param savar:
        :param tau_max:
        :param max_comps:
        :param significance:
        :param ind_test:
        :param pc_alpha:
        :param parents_alpha:
        :param correct_permutation: Should permutation be corrected
        :param perform_analysis:
        :param verbose:
        """

        # Input objects
        self.savar = savar

        # Checks
        if self.savar.data_field is None:
            raise ValueError("SAVAR is empty did you run the method for create data?")

        # Tigramite configuration
        self.max_lag = max(abs(lag)
                           for (_, lag), _, in it.chain.from_iterable(self.savar.links_coeffs.values()))
        if tau_max is None:
            self.tau_max = self.max_lag
        else:
            if tau_max < self.max_lag:
                raise ValueError("Tau max is smaller than the true tau_max")
            self.tau_max = tau_max
        self.significance = significance
        self.ind_test = ind_test
        self.pc_alpha = pc_alpha
        self.parents_alpha = parents_alpha

        # Set configuration
        if max_comps is None:  # Assign the true components
            self.max_comps = self.savar.n_vars
        else:
            self.max_comps = max_comps

            # Check that components are <= savar.variables
            if max_comps < self.savar.n_vars:
                raise ValueError("Savar Models has {0} components while max_comps = {1}, the latter should be"
                                 "equal or biggerto {0}".format(savar.n_vars, self.max_comps))
        self.verbose = verbose

        # Extract savar elements
        self.data_field = deepcopy(self.savar.data_field)  # L times T

        # Emtpy elements
        # Savar
        self.phi = {}
        self.weights = {}
        self.weights_inv = {}
        self.signal = {}

        # Tigramite
        self.pcmci = {}
        self.grid_pcmci = {}
        self.tg_results = {}
        self.tg_grid_results = {}
        self.parents_predict = {}
        self.grid_phi = {}

        # Others
        self.varimax_out = None
        self.pca_out = None
        self.permutation_dict = {}
        self.is_permuted = False
        self.is_grid_done = False

    def __call__(self):  # If perform analisis = True does all the steps
        if self.verbose:
            print("Starting DR Methods")
        self.perform_dm()
        self.get_pcmci_results()
        self.get_phi()
        if self.verbose:
            print("DR Methods finished")

    def perform_dm(self):

        # Do Varimax, and get weights signal and W_inv from varimax method.
        self._perform_varimax()

        # Do PCA and get weights signal and W_inv from PCA method
        self._perform_pca()

    def get_pcmci_results(self):

        # Varimax
        self._perform_pcmci("varimax")

        # PCA
        self._perform_pcmci("pca")

    def get_phi(self):

        # Method 1: Varimax Corr
        corr_results_var_corr = deepcopy(self.tg_results["varimax_corr"])

        # Used to convert linear OLS result from pearson coefficient
        variance_vars = self.pcmci["varimax_corr"].dataframe.values.std(axis=0)

        # Get Phi from val_matrix
        Phi = corr_results_var_corr['val_matrix']

        # If p_value not enought set it to 0
        Phi[[corr_results_var_corr['p_matrix'] > self.pc_alpha]] = 0

        # Now we do the coefficient by Val_matrix[i, j, tau]*std(j)/std(i)
        Phi = (Phi * variance_vars[:, None]) / variance_vars[:, None, None]

        self.phi["varimax_corr"] = np.moveaxis(deepcopy(Phi), 2, 0)
        np.fill_diagonal(self.phi["varimax_corr"][0, ...], 1)  # Fill the diagonal of tau 0 with ones

        # Method 2: Varimax PCMCI
        # Get parents
        self.parents_predict["varimax_pcmci"] = self.pcmci["varimax_pcmci"].return_significant_links(
            pq_matrix=self.tg_results["varimax_pcmci"]["p_matrix"],
            val_matrix=self.tg_results["varimax_pcmci"]["val_matrix"],
            alpha_level=self.parents_alpha,
            include_lagzero_links=False,
        )['link_dict']

        # Get model and phi
        dataframe = deepcopy(self.pcmci["varimax_pcmci"].dataframe)
        med = LinearMediation(dataframe=dataframe)
        med.fit_model(all_parents=self.parents_predict["varimax_pcmci"], tau_max=self.tau_max)
        self.phi["varimax_pcmci"] = med.phi

        # Method 3: PCA Corr
        corr_results_pca_corr = deepcopy(self.tg_results["pca_corr"])
        # Used to convert linear OLS result from pearson coefficient
        variance_vars = self.pcmci["pca_corr"].dataframe.values.std(axis=0)

        # Get Phi from val_matrix
        Phi = corr_results_pca_corr['val_matrix']

        # If p_value not enough set it to 0
        Phi[[corr_results_pca_corr['p_matrix'] > self.pc_alpha]] = 0

        # Now we do the coefficient by Val_matrix[i, j, tau]*std(j)/std(i)
        Phi = (Phi * variance_vars[:, None]) / variance_vars[:, None, None]

        self.phi["pca_corr"] = np.moveaxis(deepcopy(Phi), 2, 0)  # Phi is in the other cases [tau, i, j],
        np.fill_diagonal(self.phi["pca_corr"][0, ...], 1)  # Fill the diagonal of tau 0 with ones

        # Method 4
        self.parents_predict["pca_pcmci"] = self.pcmci["pca_pcmci"].return_significant_links(
            pq_matrix=self.tg_results["pca_pcmci"]["p_matrix"],
            val_matrix=self.tg_results["pca_pcmci"]["val_matrix"],
            alpha_level=self.parents_alpha,
            include_lagzero_links=False,
        )['link_dict']
        dataframe = deepcopy(self.pcmci["pca_pcmci"].dataframe)

        # Get model and phi
        med = LinearMediation(dataframe=dataframe)
        med.fit_model(all_parents=self.parents_predict["pca_pcmci"], tau_max=self.tau_max)
        self.phi["pca_pcmci"] = med.phi

    def get_grid_phi(self, results_dict_file):

        with open(results_dict_file, "rb") as f:
            pcmci_dict, corr_dict = pickle.load(f)

        self.tg_grid_results["pcmci"] = pcmci_dict
        self.tg_grid_results["corr"] = corr_dict

        if self.verbose:
            print("Starting pcmci at grid level")

        if self.significance == "analytic" and self.ind_test == "ParCorr":
            ind_test = ParCorr(significance=self.significance)
        else:
            raise ValueError("Only ParrCorr test implemented your option: {} not yet implemented".format(self.ind_test))

        dataframe = pp.DataFrame(self.savar.data_field.transpose())  # Input data for PCMCI T times K

        # PCMCI #
        self.grid_pcmci["pcmci"] = PCMCI(
            dataframe=dataframe,
            cond_ind_test=ind_test,
            verbosity=self.verbose,
        )

        self.parents_predict["pcmci"] = self.grid_pcmci["pcmci"].return_significant_links(
            pq_matrix=self.tg_grid_results["pcmci"]["p_matrix"],
            val_matrix=self.tg_grid_results["pcmci"]["val_matrix"],
            alpha_level=self.parents_alpha,
            include_lagzero_links=False,
        )["link_dict"]

        # Get grid phi
        med = LinearMediation(dataframe=dataframe)
        med.fit_model(all_parents=self.parents_predict["pcmci"], tau_max=self.tau_max)
        self.grid_phi["pcmci"] = med.phi

        # CORR

        self.grid_pcmci["corr"] = PCMCI(
            dataframe=dataframe,
            cond_ind_test=ind_test,
            verbosity=self.verbose,
        )

        corr_grd_results = deepcopy(self.tg_grid_results["corr"])
        variance_vars = self.grid_pcmci["corr"].dataframe.values.std(axis=0)

        # Get Phi from val_matrix
        Phi = corr_grd_results['val_matrix']

        # If p_value not enought set it to 0
        Phi[[corr_grd_results['p_matrix'] > self.pc_alpha]] = 0

        # Now we do the coefficient by Val_matrix[i, j, tau]*std(j)/std(i)
        Phi = (Phi * variance_vars[:, None]) / variance_vars[:, None, None]

        self.grid_phi["corr"] = np.moveaxis(deepcopy(Phi), 2, 0)
        np.fill_diagonal(self.grid_phi["corr"][0, ...], 1)  # Fill the diagonal of tau 0 with ones

        # Set it so its not performed again
        self.is_grid_done = True


    def old_get_grid_phi(self):

        if self.is_grid_done:
            print("Grid level already performed")
            return None

        if self.verbose:
            print("Starting pcmci at grid level")

        if self.significance == "analytic" and self.ind_test == "ParCorr":
            ind_test = ParCorr(significance=self.significance)
        else:
            raise ValueError("Only ParrCorr test implemented your option: {} not yet implemented".format(self.ind_test))

        dataframe = pp.DataFrame(self.savar.data_field.transpose())  # Input data for PCMCI T times K

        # PCMCI #
        self.grid_pcmci["pcmci"] = PCMCI(
            dataframe=dataframe,
            cond_ind_test=ind_test,
            verbosity=self.verbose,
        )

        self.tg_grid_results["pcmci"] = self.grid_pcmci["pcmci"].run_pcmciplus(tau_min=1, tau_max=self.tau_max,
                                                                               pc_alpha=self.pc_alpha)

        self.parents_predict["pcmci"] = self.grid_pcmci["pcmci"].return_significant_links(
            pq_matrix=self.tg_grid_results["pcmci"]["p_matrix"],
            val_matrix=self.tg_grid_results["pcmci"]["val_matrix"],
            alpha_level=self.parents_alpha,
            include_lagzero_links=False,
        )["link_dict"]

        # Get grid phi
        med = LinearMediation(dataframe=dataframe)
        med.fit_model(all_parents=self.parents_predict["pcmci"], tau_max=self.tau_max)
        self.grid_phi["pcmci"] = med.phi

        #### Correlation ####

        self.grid_pcmci["corr"] = PCMCI(
            dataframe=dataframe,
            cond_ind_test=ind_test,
            verbosity=self.verbose,
        )

        self.tg_grid_results["corr"] = self.grid_pcmci["corr"].get_lagged_dependencies(selected_links=None,
                                                                                       tau_min=1,
                                                                                       tau_max=self.tau_max,
                                                                                       val_only=False)

        corr_grd_results = deepcopy(self.tg_grid_results["corr"])
        variance_vars = self.grid_pcmci["corr"].dataframe.values.std(axis=0)

        # Get Phi from val_matrix
        Phi = corr_grd_results['val_matrix']

        # If p_value not enought set it to 0
        Phi[[corr_grd_results['p_matrix'] > self.pc_alpha]] = 0

        # Now we do the coefficient by Val_matrix[i, j, tau]*std(j)/std(i)
        Phi = (Phi * variance_vars[:, None]) / variance_vars[:, None, None]

        self.grid_phi["corr"] = np.moveaxis(deepcopy(Phi), 2, 0)
        np.fill_diagonal(self.grid_phi["corr"][0, ...], 1)  # Fill the diagonal of tau 0 with ones

        # Set it so its not performed again
        self.is_grid_done = True

    def _perform_varimax(self):
        var_dict = {
            "truncate_by": "max_comps",
            "max_comps": self.max_comps,
            "fraction_explained_variance": 0.9,
            "verbose": self.verbose
        }

        # Data must be in shape T x L for Varimax
        data = self.savar.data_field.transpose()  # Data has the shape T x L
        var = VarimaxPlus(data, **var_dict)
        self.varimax_out = var()
        self.weights["varimax"] = self.varimax_out["weights"].transpose()  # Now they are N times L

        # signal:  N times L @ L times T -> N times T (signal)
        self.signal["varimax"] = self.weights["varimax"] @ self.data_field

        # Reorder the signal weights and get w_plus
        self._correct_permutation("varimax")

    def _perform_pca(self):

        self.pca_out = PCA(n_components=self.max_comps)

        # Input PCA: T times L
        self.pca_out.fit(self.data_field.transpose())
        self.weights["pca"] = self.pca_out.components_  # Shape N x L

        # signal =   N times L @ L times T -> N times T (signal)
        self.signal['pca'] = self.weights["pca"] @ self.data_field

        # Reorder the signal weights and get w_plus
        self._correct_permutation("pca")

    def _correct_permutation(self, met):

        # NxT = NxL @ LxT
        savar_signal = self.savar.mode_weights.reshape(self.savar.n_vars, -1) @ self.savar.data_field
        self.permutation_dict[met] = find_permutation(savar_signal.transpose(), self.signal[met])
        idx_permutation = [self.permutation_dict[met][i] for i in range(self.savar.n_vars)]

        # Correct weights and  get signal
        self.weights[met] = self.weights[met][idx_permutation, ...]
        self.signal[met] = self.weights[met] @ self.data_field  #

        # Get inverse
        self.weights_inv[met] = np.linalg.pinv(self.weights[met])  # L times K

        self.is_permuted = True

    def _perform_pcmci(self, met):

        if self.significance == "analytic" and self.ind_test == "ParCorr":
            ind_test = ParCorr(significance=self.significance)
        else:
            raise ValueError("Only ParrCorr test implemented your option: {} not yet implemented".format(self.ind_test))

        self.pcmci[str(met + "_pcmci")] = PCMCI(
            dataframe=pp.DataFrame(self.signal[met].transpose()),  # Input data for PCMCI T times N
            cond_ind_test=ind_test,
            verbosity=self.verbose,
        )
        self.pcmci[met + "_corr"] = deepcopy(self.pcmci[met + "_pcmci"])
        self.tg_results[met + "_pcmci"] = self.pcmci[met + "_pcmci"].run_pcmciplus(tau_min=1, tau_max=self.tau_max,
                                                                                   pc_alpha=self.pc_alpha)
        self.tg_results[met + "_corr"] = self.pcmci[met + "_corr"].get_lagged_dependencies(selected_links=None,
                                                                                           tau_min=1,
                                                                                           tau_max=self.tau_max,
                                                                                           val_only=False)


# TODO: Needs to be chekced
# We will use isa_pcmci to get Phi before any iteration.

class Evaluation:
    """
    This class is used to output the evaluation metrics of the experiments
    """

    def __init__(self, dm_object: DmMethod, results_dict_file, grid_threshold: float = None, grid_threshold_per: float = 95,
                 verbose: bool = True,
                 methods: list = (),  # ("varimax_corr", "varimax_pcmci", "pca_corr", "pca_pcmci"),
                 grid_methods: list = ("pcmci", "corr", "varimax_pcmci_w", "varimax_corr_w",
                                       "pca_pcmci_w", "pca_corr_w")):
        # inputs
        self.dm_object = dm_object
        self.results_dict_file = results_dict_file

        # Extract dm_object
        self.n_variables = self.dm_object.savar.n_vars
        self.tau_max = self.dm_object.tau_max
        self.dm_phi = self.dm_object.phi
        self.dm_cg = deepcopy(self.dm_phi)
        for key in self.dm_cg:
            self.dm_cg[key][np.abs(self.dm_phi[key]) > 0] = 1
        self.dm_phi = self.dm_object.phi
        self.dm_weights = self.dm_object.weights

        # savar
        self.savar_phi = cg_to_est_phi(self.dm_object.savar.links_coeffs, tau_max=self.dm_object.tau_max)
        self.savar_weights = self.dm_object.savar.mode_weights.reshape(self.n_variables, -1)
        self.savar_cg = deepcopy(self.savar_phi)
        self.savar_cg[np.abs(self.savar_cg) > 0] = 1

        # Other
        self.verbose = verbose
        self.grid_threshold = grid_threshold
        self.grid_threshold_per = grid_threshold_per
        self.metrics = {metric: {} for metric in methods + grid_methods}
        self.methods = methods
        self.grid_methods = grid_methods

        # Empty attributes
        self.cg_conf_matrix = {}
        self.savar_grid_phi = None
        self.grid_phi = {method: {} for method in grid_methods}
        self.savar_grid_cg = None
        self.grid_cg = {method: {} for method in grid_methods}
        self.dm_grid_cg = None
        self.cg_grid_conf_matrix = {}
        self.dm_grid_phi = None

    def _obtain_individual_metrics(self, method):

        # MSE and RMAE
        savar_phi = self.savar_phi[1:, ...]
        idx = np.nonzero(savar_phi)  # Non_zero elements of True phi
        dm_phi = deepcopy(self.dm_object.phi[method][1:, ...])
        # For non-zero elements of savar phi = (|Phi-\tilde(Phi)|)/|Phi|
        self.metrics[method]["mse"] = np.square(savar_phi[idx] - dm_phi[idx]).mean()
        self.metrics[method]["rmae"] = (np.abs(savar_phi[idx] - dm_phi[idx]) / np.abs(savar_phi[idx])).mean()

        # Precision and Recall
        savar_cg = deepcopy(self.savar_cg[1:, ...].flatten())
        dm_cg = self.dm_cg[method][1:, ...].flatten()
        self.cg_conf_matrix[method] = confusion_matrix(savar_cg, dm_cg, labels=(0, 1))
        tn, fp, fn, tp = self.cg_conf_matrix[method].ravel()

        if (tp + fp) != 0:
            self.metrics[method]["precision"] = tp / (tp + fp)
        else:
            self.metrics[method]["precision"] = 0
        if tp / (tp + fn) != 0:
            self.metrics[method]["recall"] = tp / (tp + fn)
        else:
            self.metrics[method]["recall"] = 0

        # Weights
        if method in ("varimax_pcmci", "varimax_corr"):
            method_dm = "varimax"
        elif method in ("pca_pcmci", "pca_corr"):
            method_dm = "pca"
        else:
            raise ValueError("Method {} not correct".format(method))
        N = self.dm_object.savar.n_vars
        corr_weights = np.array([np.corrcoef(self.dm_object.weights[method_dm][i, ...],
                                             self.dm_object.savar.mode_weights[i, ...].flatten())[0, 1]
                                 for i in range(N)])

        self.metrics[method]["corr_weights"] = np.abs(corr_weights)

        # Signal
        # (K times T) = K times L @ L times T
        savar_signal = self.dm_object.savar.mode_weights.reshape(N, -1) @ self.dm_object.savar.data_field
        corr_signal = np.array([np.corrcoef(self.dm_object.signal[method_dm][i, ...], savar_signal[i, ...])[0, 1]
                                for i in range(N)])
        self.metrics[method]["corr_signal"] = np.abs(corr_signal)

    def _obtain_grid_metrics(self):

        dm_grid_methods_dict = {
            "varimax_pcmci_w": ("varimax_pcmci", "varimax"),
            "varimax_corr_w": ("varimax_corr", "varimax"),
            "pca_pcmci_w": ("pca_pcmci", "pca"),
            "pca_corr_w": ("pca_corr", "pca"),
        }

        # self.dm_object.get_grid_phi(self.results_dict_file)
        # Grid_phi = W^+ Phi W
        self.savar_grid_phi = self.compute_grid_phi(self.savar_phi, self.savar_weights)
        idx = np.nonzero(self.savar_grid_phi)  # Non_zero elements of True phi

        for method in self.grid_methods:
            if method in ("pcmci", "corr"):
                # MSE and RMAR
                self.grid_phi[method] = deepcopy(self.dm_object.grid_phi[method][1:, ...])
                dm_grid_phi = deepcopy(self.grid_phi[method])
                savar_grid = deepcopy(self.savar_grid_phi)
                # For non-zero elements of savar phi = (|Phi-\tilde(Phi)|)/|Phi|
                self.metrics[method]["grid_mse"] = np.square(savar_grid[idx] - dm_grid_phi[idx]).mean()
                self.metrics[method]["grid_rmae"] = (
                        np.abs(savar_grid[idx] - dm_grid_phi[idx]) / np.abs(savar_grid[idx])).mean()

                # CG metrics (precisions and Recall)
                self.savar_grid_cg = deepcopy(self.savar_grid_phi)
                self.savar_grid_cg[np.abs(self.savar_grid_cg) > 0] = 1
                self.grid_cg[method] = deepcopy(self.grid_phi[method])
                self.grid_cg[method][np.abs(self.grid_cg[method]) > 0] = 1

                savar_grid_cg = self.savar_grid_cg.flatten()
                grid_cg = self.grid_cg[method].flatten()

                self.cg_grid_conf_matrix[method] = confusion_matrix(savar_grid_cg, grid_cg, labels=(0, 1))
                tn, fp, fn, tp = self.cg_grid_conf_matrix[method].ravel()

                self.metrics[method]["grid_precision"] = tp / (tp + fp)
                self.metrics[method]["grid_recall"] = tp / (tp + fn)

            if method in ("varimax_pcmci_w", "varimax_corr_w",
                          "pca_pcmci_w", "pca_corr_w"):

                latent_method = dm_grid_methods_dict[method][0]  # e.j: varimax_pcmci
                dm_method = dm_grid_methods_dict[method][1]  # e.j: varimax

                self.grid_phi[method] = self.compute_grid_phi(
                    self.dm_object.phi[latent_method],
                    self.dm_weights[dm_method])

                dm_grid_phi = deepcopy(self.grid_phi[method])
                savar_grid = deepcopy(self.savar_grid_phi)

                # For non-zero elements of savar phi = (|Phi-\tilde(Phi)|)/|Phi|
                self.metrics[method]["grid_mse"] = np.square(savar_grid[idx] - dm_grid_phi[idx]).mean()
                self.metrics[method]["grid_rmae"] = (np.abs(savar_grid[idx] - dm_grid_phi[idx])
                                                     / np.abs(savar_grid[idx])).mean()

                self.savar_grid_cg = deepcopy(self.savar_grid_phi)
                self.savar_grid_cg[np.abs(self.savar_grid_cg) > 0] = 1

                self.grid_cg[method] = deepcopy(self.grid_phi[method])
                # Apply a threshold
                if self.grid_threshold is not None:
                    grid_threshold = self.grid_threshold

                    if self.grid_threshold != 0:
                        self.grid_cg[method][np.abs(self.grid_cg[method]) <= grid_threshold] = 0
                else:
                    grid_threshold = np.percentile(np.abs(self.grid_cg[method]), self.grid_threshold_per)
                    print("printing percent threshold", grid_threshold)
                    self.grid_cg[method][np.abs(self.grid_cg[method]) <= grid_threshold] = 0

                self.grid_cg[method][np.abs(self.grid_cg[method]) > 0] = 1

                savar_grid_cg = self.savar_grid_cg.flatten()
                grid_cg = self.grid_cg[method].flatten()

                self.cg_grid_conf_matrix[method] = confusion_matrix(savar_grid_cg, grid_cg, labels=(0, 1))
                tn, fp, fn, tp = self.cg_grid_conf_matrix[method].ravel()

                self.metrics[method]["grid_precision"] = tp / (tp + fp)
                self.metrics[method]["grid_recall"] = tp / (tp + fn)

    def obtain_score_metrics(self, perform_grid=False):
        """
        Computes the following metrics for the 4 methods implemented
        If perfrom_grid, then also performs the metrics at grid level
        :return:
        """
        for method in self.methods:
            self._obtain_individual_metrics(method)

        if perform_grid:
            self._obtain_grid_metrics()

    @staticmethod
    def compute_grid_phi(phi, weights):
        """ Remove time 0 in tau and computes the grid version
        :param phi: Phi at mode level
        :param weights: weights of shape K x L"""
        # Estimated Grid_Phi
        phi = phi[1:, ...]
        weights_inv = np.linalg.pinv(weights)
        return weights_inv[None, ...] @ phi @ weights[None, ...]
