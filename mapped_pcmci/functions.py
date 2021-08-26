from copy import deepcopy
import numpy as np


def find_permutation(true, permuted):
    """
    Finds the most probable permutation of true time series in between permuted time series
    :param true: true ordered time series of shape T times N
    :param permuted: Permuted time series of shape P times T. P > K
    :return: A dict containing {true idx: permuted idx}
    """

    N = true.shape[1]
    max_comps = permuted.shape[0]

    corr_matrix = np.zeros((N, max_comps))

    # Find correlations
    for i in range(N):
        for j in range(max_comps):
            corr_matrix[i, j] = np.corrcoef(true[:, i], permuted[j, :])[0, 1]

    permutation_dict = {}
    used_comps = []

    # Find best order
    per_matrix = np.argsort(-np.abs(corr_matrix), axis=1)

    for i in range(N):
        for j in per_matrix[i, :]:
            if j in used_comps:
                continue
            else:
                permutation_dict[i] = j
                used_comps.append(j)
                break

    return permutation_dict


def threshold_link_strenght(links_coeff, threshold):
    links_coeff = deepcopy(links_coeff)
    for var, link_list in links_coeff.items():
        links_coeff[var] = [((v, t), s) for ((v, t), s) in link_list if abs(s) >= threshold]

    return links_coeff
