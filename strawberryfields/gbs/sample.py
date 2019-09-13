# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Sampling functions
==================

**Module name:** :mod:`strawberryfields.gbs.sample`

.. currentmodule:: strawberryfields.gbs.sample

This module provides functionality for generating samples from both a quantum device and the
uniform distribution.

The :func:`quantum_sampler` function allows users to simulate Gaussian boson sampling (GBS) by
choosing a symmetric input matrix to sample from :cite:`bradler2018gaussian`. For a symmetric
:math:`N \times N` input matrix :math:`A`, corresponding to an undirected graph,
an :math:`N`-mode GBS device with threshold detectors generates samples that are binary strings
of length ``N``. Various problems can be encoded in the matrix :math:`A` so that the output
samples are informative and can be used as a form of randomness to improve solvers, as outlined
in Refs. :cite:`arrazola2018using` and :cite:`arrazola2018quantum`.

On the other hand, the :func:`uniform_sampler` function allows users to generate samples where a
subset of modes are selected using the uniform distribution.

Summary
-------

.. autosummary::
    QUANTUM_BACKENDS
    BACKEND_DEFAULTS
    quantum_sampler
    uniform_sampler
    random_seed

Code details
^^^^^^^^^^^^
"""
from typing import Optional

import numpy as np
import strawberryfields as sf

from strawberryfields.gbs import utils

QUANTUM_BACKENDS = ("gaussian",)
"""tuple[str]: Available quantum backends for sampling."""

BACKEND_DEFAULTS = {"remote": False, "backend": "gaussian", "threshold": True, "postselect": 0}
"""dict[str, Any]: Dictionary to specify default parameters of options in backend sampling for
:func:`quantum_sampler`.
"""


def quantum_sampler(
    A: np.ndarray, n_mean: float, samples: int = 1, backend_options: Optional[dict] = None
) -> list:
    r"""Generate samples from GBS

    Perform quantum sampling of adjacency matrix :math:`A` using the Gaussian boson sampling
    algorithm.

    Sampling can be controlled with the optional ``backend_options`` argument, which should be a
    dict containing any of the following:

    .. glossary::

        key: ``"remote"``, value: *bool*
            Performs sampling on a remote server if ``True``. Remote sampling is
            required for sampling on hardware. If not specified, sampling will be performed locally.

        key: ``"backend"``, value: *str*
            Requested backend for sampling. Available backends are listed in
            ``QUANTUM_BACKENDS``, these are:

            - ``"gaussian"``: for simulating the output of a GBS device using the
              :mod:`strawberryfields.backends.gaussianbackend`

        key: ``"threshold"``, value: *bool*
            Performs sampling using threshold (on/off click) detectors if ``True``
            or using photon number resolving detectors if ``False``. Defaults to ``True`` for
            threshold sampling.

        key: ``"postselect"``, value: *int*
            Causes samples with a photon number or click number less than the
            specified value to be filtered out. Defaults to ``0`` if unspecified, resulting in no
            postselection. Note that the number of samples returned in :func:`quantum_sampler` is
            still equal to the ``samples`` parameter.

    Args:
        A (array): the (real or complex) :math:`N \times N` adjacency matrix to sample from
        n_mean (float): mean photon number
        samples (int): number of samples; defaults to 1
        backend_options (dict[str, Any]): dictionary specifying options used by backends during
            sampling; defaults to :const:`BACKEND_DEFAULTS`

    Returns:
        list[list[int]]: a list of length ``samples`` whose elements are length :math:`N` lists of
        integers indicating events (e.g., photon numbers or clicks) in each of the :math:`N`
        modes of the detector
    """
    backend_options = {**BACKEND_DEFAULTS, **(backend_options or {})}

    if not utils.is_undirected(A):
        raise ValueError("Input must be a NumPy array corresponding to a symmetric matrix")

    if samples < 1:
        raise ValueError("Number of samples must be at least one")

    nodes = len(A)
    p = sf.Program(nodes)

    mean_photon_per_mode = n_mean / float(nodes)

    # pylint: disable=expression-not-assigned,pointless-statement
    with p.context as q:
        sf.ops.GraphEmbed(A, mean_photon_per_mode=mean_photon_per_mode) | q
        sf.ops.Measure | q

    p = p.compile("gbs")

    postselect = backend_options["postselect"]
    threshold = backend_options["threshold"]

    if postselect == 0:
        s = _sample_sf(p, shots=samples, backend_options=backend_options)

        if threshold:
            s[s >= 1] = 1

        s = s.tolist()

    elif postselect > 0:
        s = []

        while len(s) < samples:
            samp = np.array(_sample_sf(p, shots=1, backend_options=backend_options))

            if threshold:
                samp[samp >= 1] = 1

            counts = np.sum(samp)

            if counts >= postselect:
                s.append(samp.tolist())

    else:
        raise ValueError("Can only postselect on nonnegative values")

    return s


def _sample_sf(p: sf.Program, shots: int = 1, backend_options: Optional[dict] = None) -> np.ndarray:
    """Generate samples from Strawberry Fields.

    Args:
        p (sf.Program): the program to sample from
        shots (int): the number of samples; defaults to 1
        backend_options (dict[str, Any]): dictionary specifying options used by backends during
        sampling; defaults to :const:`BACKEND_DEFAULTS`

    Returns:
        array: an array of ``len(shots)`` samples, with each sample the result of running a
        :class:`~strawberryfields.program.Program` specified by ``p``
    """
    backend_options = {**BACKEND_DEFAULTS, **(backend_options or {})}

    if not backend_options["backend"] in QUANTUM_BACKENDS:
        raise ValueError("Invalid backend selected")

    if backend_options["remote"]:
        raise ValueError("Remote sampling not yet available")

    eng = sf.LocalEngine(backend=backend_options["backend"])

    return np.array(eng.run(p, run_options={"shots": shots}).samples)


def uniform_sampler(modes: int, sampled_modes: int, samples: int = 1) -> list:
    """Perform classical sampling using the uniform distribution to randomly select a subset of
    modes of a given size.

    Args:
        modes (int): number of modes to sample from
        sampled_modes (int): number of modes to sample
        samples (int): number of samples; defaults to 1

    Returns:
        list[list[int]]: a ``len(samples)`` list whose elements are ``len(modes)`` lists which
        contain zeros and ones. The number of ones is ``samples_modes``, corresponding to the modes
        selected.
    """

    if modes < 1 or sampled_modes < 1 or sampled_modes > modes:
        raise ValueError(
            "Modes and sampled_modes must be greater than zero and sampled_modes must not exceed "
            "modes "
        )
    if samples < 1:
        raise ValueError("Number of samples must be greater than zero")

    base_sample = [1] * sampled_modes + [0] * (modes - sampled_modes)

    output_samples = [list(np.random.permutation(base_sample)) for _ in range(samples)]

    return output_samples


def random_seed(seed: int = None) -> None:
    """Seed for random number generators.

    Wrapper function for `numpy.random.seed <https://docs.scipy.org/doc/numpy//reference/generated
    /numpy.random.seed.html>`_ to seed NumPy-based random number generators used in
    :func:`quantum_sampler` and :func:`uniform_sampler`. This allows for repeatable sampling.

    Args:
        seed (int): random seed; defaults to ``None``
    """
    np.random.seed(seed)
