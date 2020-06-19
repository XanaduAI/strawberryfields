.. role:: html(raw)
   :format: html

States
======

In Strawberry Fields, the statevector simulator backends return ``state`` objects,
which provide useful information and processing of the quantum state of
the system. For example, consider the following program:

.. code-block:: python3

    prog = sf.Program(3)

    with prog.context as q:
        ops.Sgate(0.54) | q[0]
        ops.Sgate(0.54) | q[1]
        ops.Sgate(0.54) | q[2]
        ops.BSgate(0.43, 0.1) | (q[0], q[2])
        ops.BSgate(0.43, 0.1) | (q[1], q[2])

    eng = sf.Engine("fock", backend_options={"cutoff_dim": 10})
    result = eng.run(prog)
    state = result.state

By executing this program on a local simulator backend, we can use the state
object to determine the trace, return the density matrix, and calculate particular
Fock basis state probabilities:

>>> state.trace()
0.9989783190545866
>>> rho = state.dm()
>>> state.fock_prob([0, 0, 2])
0.07933909728557098

These methods may differ depending on the backend, but there
are a few that are implemented for all backends.

Common methods and attributes
-----------------------------

.. currentmodule:: strawberryfields.backends.BaseState

:html:`<div class="summary-table">`

.. autosummary::
    data
    hbar
    is_pure
    num_modes
    mode_names
    mode_indices
    ket
    dm
    reduced_dm
    fock_prob
    all_fock_probs
    mean_photon
    fidelity
    fidelity_vacuum
    fidelity_coherent
    wigner
    quad_expectation
    poly_quad_expectation
    number_expectation
    parity_expectation
    p_quad_values
    x_quad_values

:html:`</div>`


Gaussian states
---------------

Backend that represent the quantum state using the Gaussian formalism
(such as the ``gaussian`` backend) will return a ``GaussianState`` object,
with the following methods and attributes.

.. currentmodule:: strawberryfields.backends.BaseGaussianState

:html:`<div class="summary-table">`

.. autosummary::
    means
    cov
    reduced_gaussian
    is_coherent
    is_squeezed
    displacement
    squeezing

:html:`</div>`


Fock states
-----------

Backend that represent the quantum state in the Fock basis
(such as the ``fock`` or ``tf`` backends) will return a ``FockState``
object.

.. currentmodule:: strawberryfields.backends.BaseFockState

:html:`<div class="summary-table">`

.. autosummary::
    cutoff_dim
    trace

:html:`</div>`
