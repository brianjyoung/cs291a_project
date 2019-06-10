from numba import jit
import numpy as np


# returns the set of connected subsets containing r elements, containing x
@jit(nopython=True, parallel=True)
def connected_subsets(data: np.ndarray, r: int, x: int) -> np.ndarray:
    # end bound is *one after* starting point of last interval
    end_bound = len(data) - r + 1 if x + r > len(data) else x + 1
    # start bound is starting point of first interval
    start_bound = 0 if x - r < -1 else x - r + 1

    # set up loop
    start_bounds = range(start_bound, end_bound)
    end_bounds = range(start_bound + r, end_bound + r)
    subsets = np.empty((end_bound - start_bound, r))
    for i, (j, k) in enumerate(zip(start_bounds, end_bounds)):
        subsets[i] = data[j:k]
    return subsets

# closing sieve, minima extrema processing
@jit(nopython=True)
def o_sieve(data: np.ndarray, r: int, x: int) -> float:
    # Array stores min of starting point and next r values
    subsets = connected_subsets(data, r, x)
    subset_minima = np.empty(subsets.shape[0])
    for i in range(subsets.shape[0]):
        subset_minima[i] = np.min(subsets[i])
    # subset_minima = np.min(subsets, 1)
    return np.max(subset_minima)

# one iteration of the multiscale analysis
@jit(nopython=True, parallel=True)
def multiscale_step(r: int, previous: np.ndarray) -> np.ndarray:
    post = np.empty_like(previous)
    for i in range(len(previous)):
        post[i] = o_sieve(previous, r, i)
    return post

# full multiscale analysis
@jit(nopython=True, parallel=True)
def multiscale_full(initial: np.ndarray) -> np.ndarray:
    filters = np.empty((61, 4800))
    filters[0] = initial
    for i in range(1, 61):
        filters[i] = multiscale_step(i+1, initial)
    return filters


# if __name__ == "__main__":
#     filters = np.empty(61, 40000)
#     filters[0] = 0  # img.flatten('F')
#     for index in range(1, 61):
#         filters[index] = multiscale_step(index + 1, filters[index - 1])
#     differences = filters[1:] - filters[:-1]
#     sums = np.sum(differences, 1)
