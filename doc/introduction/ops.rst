.. role:: html(raw)
   :format: html

Operations
==========

Strawberry Fields supports a wide variety of photonic quantum operations --- including gates,
state preparations and measurements. These operations are used to construct quantum programs,
as in the following example:

.. code-block:: python3

    with prog.context as q:
        Fock(2)       | q[0]
        Fock(3)       | q[1]
        BSgate()      | (q[0], q[1])
        MeasureFock() | q[0]

This program uses Fock state preparation (:class:`~.ops.Fock`), beamsplitters (:class:`~.BSgate`),
and a photon number resolving detector (:class:`~.MeasureFock`).

Below is a list of all quantum operations supported by Strawberry Fields.

.. note::

    In the Strawberry Fields we use the convention :math:`\hbar=2` by default, however
    this can be changed using the global variable :py:data:`sf.hbar`.

    See :ref:`conventions` for more details.

.. currentmodule:: strawberryfields.ops

Single-mode gates
-----------------

:html:`<div class="summary-table">`

.. autosummary::
    Dgate
    Xgate
    Zgate
    Sgate
    Rgate
    Pgate
    Vgate
    Fouriergate

:html:`</div>`

Two-mode gates
--------------

:html:`<div class="summary-table">`

.. autosummary::
    BSgate
    MZgate
    S2gate
    CXgate
    CZgate
    CKgate

:html:`</div>`

State preparation
-----------------

:html:`<div class="summary-table">`

.. autosummary::
    Vacuum
    Coherent
    Squeezed
    DisplacedSqueezed
    Thermal
    Fock
    Catstate
    GKP
    Ket
    DensityMatrix
    Gaussian

:html:`</div>`

Measurements
------------

:html:`<div class="summary-table">`

.. autosummary::
    MeasureFock
    MeasureThreshold
    MeasureHomodyne
    MeasureHeterodyne
    MeasureX
    MeasureP
    MeasureHD

:html:`</div>`


Channels
--------

:html:`<div class="summary-table">`

.. autosummary::
    LossChannel
    ThermalLossChannel
    MSgate
    PassiveChannel

:html:`</div>`

.. _decompositions:

Decompositions
--------------

:html:`<div class="summary-table">`

.. autosummary::
    Interferometer
    GraphEmbed
    BipartiteGraphEmbed
    GaussianTransform
    Gaussian

:html:`</div>`


