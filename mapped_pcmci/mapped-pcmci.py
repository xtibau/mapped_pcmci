import numpy as np
from copy import deepcopy
from varimax_plus.varimax_plus import VarimaxPlus
from tigramite.independence_tests import ParCorr
from tigramite.pcmci import PCMCI
import tigramite.data_processing as pp
from tigramite.models import LinearMediation


class MappedPCMCI:
    def __init__(self, dataset: np.ndarray, weights: np.ndarray = None,
                 var_config: dict = None, ind_test: str = "ParCorr",
                 significance: str = "analityc", pc_alpha: float = 0.2, parents_alpha: float = 0.05, tau_max: int = 3,
                 verbose: bool = False):
        """
        :param weights: Shape components, Lat, Lon
        :param dataset: Shape Space x Time
        """
        self.dataset = dataset
        self.weights = weights
        self.var_config = var_config

        # PCMCI
        self.significance = significance
        self.ind_test = ind_test
        self.pc_alpha = pc_alpha
        self.parents_alpha = parents_alpha
        self.tau_max = tau_max

        # Others
        self.verbose = verbose

    def perform_dm(self):
        # In case the weights have been computed or are provided by other methods
        if self.weights is not None:
            return None

        self._get_varimax_weights()

    def perform_pcmci(self):

        if self.significance == "analytic" and self.ind_test == "ParCorr":
            ind_test = ParCorr(significance=self.significance)
        else:
            raise ValueError("Only ParrCorr test implemented your option: {} not yet implemented".format(self.ind_test))

        n_comps = self.weights.shape[0]
        self.signal = self.weights.reshape(n_comps, -1) @ self.dataset

        self.pcmci = PCMCI(
            dataframe=pp.DataFrame(self.signal.transpose()),  # Input data for PCMCI T times N
            cond_ind_test=ind_test,
            verbosity=self.verbose,
        )
        self.pcmci_results = self.pcmci.run_pcmciplus(tau_min=1, tau_max=self.tau_max,
                                                      pc_alpha=self.pc_alpha)
        self.corr_results = self.pcmci.get_lagged_dependencies(selected_links=None,
                                           tau_min=1,
                                           tau_max=self.tau_max,
                                           val_only=False)

    def get_grid_phi(self):

        self.parents_predict = self.pcmci.return_significant_links(
            pq_matrix=self.pcmci_results["p_matrix"],
            val_matrix=self.pcmci_results["val_matrix"],
            alpha_level=self.parents_alpha,
            include_lagzero_links=False,
        )["link_dict"]

        # Get grid phi
        med = LinearMediation(dataframe=pp.DataFrame(self.signal.transpose()))
        med.fit_model(all_parents=self.parents_predict["pcmci"], tau_max=self.tau_max)
        self.phi = med.phi

        weights_inv = np.linalg.pinv(self.weights)
        self.grid_phi = weights_inv[None, ...] @ self.phi @ self.weights[None, ...]

    def _get_varimax_weights(self):

        if self.var_config is None:

            var_config = {
                "truncate_by": "fraction_explained_variance",
                "max_comps": 60,
                "fraction_explained_variance": 0.9,
                "boot_axis": 1,
                "boot_rep": 100,
                "boot_samples": self.dataset.shape[1],
                "alpha_level": 0.01,
                "verbose": self.verbose
            }
        else:
            var_config = self.var_config

        dataset = deepcopy(self.dataset)  # time * Lat x Lon

        if self.verbose:
            print("computing Varimax+")
        var_p = VarimaxPlus(data=dataset, **var_config)
        var_p.varimax_plus()
        self.var_results = var_p.results
        self.weights = var_p.results["weights"]

    def __call__(self, *args, **kwargs):

        self.perform_dm()
        self.perform_pcmci()
        self.get_grid_phi()
        return self.grid_phi
