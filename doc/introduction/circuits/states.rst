.. _state_class:

State objects
=============

In Strawberry Fields, the statevector simulators return ``state`` objects,
which provide useful information and processing of the quantum state of
the system. These methods may differ depending on the backend, but there
are a few that are implemented for all backends.

.. note::
    In the following, keyword arguments are denoted ``**kwargs``, and allow additional
    options to be passed to the underlying State class - these are documented where
    available. For more details on relevant keyword arguments, please
    consult the backend documentation directly.

Common methods and attributes
-----------------------------

.. currentmodule:: strawberryfields.backends.states.BaseState

.. autosummary::
    data
    hbar
    is_pure
    num_modes
    mode_names
    mode_indices
    reduced_dm
    fock_prob
    mean_photon
    fidelity
    fidelity_vacuum
    fidelity_coherent
    wigner
    quad_expectation
    poly_quad_expectation


Gaussian states
---------------

Backend that represent the quantum state using the Gaussian formalism
(such as the ``gaussian`` backend) will return a ``GaussianState`` object,
with the following methods and attributes.

.. currentmodule:: strawberryfields.backends.gaussianbackend.states.GaussianState

.. autosummary::
    means
    cov
    reduced_gaussian
    is_coherent
    is_squeezed
    displacement
    squeezing
    reduced_dm
    fock_prob
    mean_photon
    fidelity
    fidelity_vacuum
    fidelity_coherent


Fock states
-----------

Backend that represent the quantum state in the Fock basis
(such as the ``fock`` or ``tf`` backends) will return a ``FockState``
object.

.. currentmodule:: strawberryfields.backends.states.BaseFockState

.. autosummary::
    cutoff_dim
    ket
    dm
    trace
    all_fock_probs
