import numpy as np

# supercomputer benchmark parameters:
#  ``c`` is a scale factor in units of seconds, obtained by the LINPACK
#  benchmarks, satisfying

# t = c * n**3 * 2 ** (n / 2)

# where ``t`` is the time it takes to compute a Hafnian of size ``n x n`` on a
# particular computer.
c_niagara = 5.42e-15
c_fugaku = c_niagara / 122.8


def gbs_runtime(N_c, G, modes, c=c_fugaku):
    """Simulation time of a GBS sample.

    Args:
        N_c (int): number of nonzero detector outcomes
        G (float): collision parameter
        modes (int): number of modes
        c (float): a supercomputer benchmark parameter (see reference
            above)

    Returns:
        float: runtime in seconds
    """

    return 0.5 * c_fugaku * modes * N_c**3 * G ** (N_c / 2)


def ncg(sample):
    """Calculates and returns the number of nonzero events ``N_c`` and the collision parameter ``G`` of
    a sample.

    Args:
        sample (array[int]): a photon-number array of shape
        ``(temporal_modes,)``

    Returns:
        tuple[float]: the number of nonzero events ``N_c`` and the collision parameter ``G`` of a sample
    """
    N_c = int(np.heaviside(sample, 0).sum())

    # avoid division by 0 when `samples` has no photons
    if N_c != 0:
        G = np.prod(np.float64(sample) + 1) ** (1 / N_c)
    else:
        G = 1

    return N_c, G


def gbs_sample_runtime(sample, c=c_fugaku, return_ncg=False):
    """Simulation time of a GBS sample.

    Args:
        sample (array[int]): a photon-number array of shape
            ``(temporal_modes,)``
        c (float): a supercomputer benchmark parameter (see reference
            above)
        return_ncg (bool): if ``True`` return not only runtime but also
            the number of non-zero detector events ``N_c`` and the
            collision parameter ``G``

    Returns:
        float: the simulation time of a sample in seconds;
        if ``return_ncg`` also returns ``N_c`` and ``G``
    """
    N_c, G = ncg(sample)
    modes = len(sample)
    r = gbs_runtime(N_c, G, modes, c)

    if not return_ncg:
        return r
    return r, N_c, G
