.. |ps| replace:: post-selection
.. |PS| replace:: Post-selection

.. _ps_tutorial:

Measurements and |ps| tutorial
##############################

.. sectionauthor:: Josh Izaac <josh@xanadu.ai>

.. role:: html(raw)
   :format: html

.. highlight:: pycon

In this tutorial, we will walk through the use of measurement operators in Strawberry Fields, and how they may be used to perform post-selection. Be sure to read through the introductory :ref:`teleportation tutorial <tutorial>` before attempting this tutorial.


Measurement operators
=====================

The Blackbird programming language supports the following measurement operations:

.. rst-class:: docstable

+----------------------------------+----------------------------------------------------+---------------------------+-------------------------+
|           Measurement            |                     Operation                      |         Shortcuts         |     Backend Support     |
+==================================+====================================================+===========================+=========================+
| Homodyne detection               | :class:`MeasureHomodyne(phi=0) <.MeasureHomodyne>` | ``MeasureX`` ``MeasureP`` | :class:`~.BaseFock`     |
| of quadrature angle :math:`\phi` |                                                    |                           | :class:`~.BaseGaussian` |
+----------------------------------+----------------------------------------------------+---------------------------+-------------------------+
| Heterodyne detection             | :class:`MeasureHeterodyne() <.MeasureHeterodyne>`  | ``MeasureHD``             | :class:`~.BaseGaussian` |
+----------------------------------+----------------------------------------------------+---------------------------+-------------------------+
| Photon-counting                  | :class:`MeasureFock() <.MeasureFock>`              | ``Measure``               | :class:`~.BaseFock`     |
|                                  |                                                    |                           | :class:`~.BaseGaussian` |
+----------------------------------+----------------------------------------------------+---------------------------+-------------------------+

.. note:: While all backends support homodyne detection, the Gaussian backend is the only backend to support heterodyne detection. On the other hand, Fock-basis measurements are supported in all backends, though the Gaussian backend does not update the post-measurement quantum state, which would be non-Gaussian.

The measurement operators are used in the same manner as all other quantum transformation operations in Blackbird:

``MeasurementOperator | (q[0], q[1], q[2], ...)``

where the left-hand side represents the measurement operator (along with any required or optional arguments), and the right-hand side signifies the modes which are to be measured.

To see how this works in practice, consider the following circuit, where two incident Fock states :math:`\ket{n}` and :math:`\ket{m}` are directed on a beamsplitter, with two photon detectors at the output modes.

|

.. image:: ../_static/bs_measure.svg
    :align: center
    :width: 30%
    :target: javascript:void(0);

|

Due to the definition of the beamsplitter, we know that it preserves the photon number of the system; thus, the two output states :math:`\ket{n'}` and :math:`\ket{m'}` must be such that :math:`n+m=n'+m'`.

Constructing this circuit in Strawberry Fields with :math:`n=2,~m=3`, let's perform only the first Fock measurement.

.. code-block:: python3

    import strawberryfields as sf
    from strawberryfields.ops import *

    prog = sf.Program(2)
    eng = sf.Engine("fock", backend_options={"cutoff_dim": 6})

    with prog.context as q:
        Fock(2)       | q[0]
        Fock(3)       | q[1]
        BSgate()      | (q[0], q[1])
        MeasureFock() | q[0]

    results = eng.run(prog)

.. note:: If the :class:`~.BSgate` parameters are not specified, by default a 50-50 beamsplitter ``BSgate(pi/4,0)`` is applied.

The default action after every measurement is to reset the measured modes to the vacuum state. However, we can extract the measured value of mode ``q[0]`` via the ``results``
object returned by the engine after it has finished execution:

>>> results.samples[0]
1

.. note:: Since measurement is a stochastic process, your results might differ when executing this code.

Since no measurement has yet been applied to the second mode, ``results.samples[1]`` will return ``None``.

>>> results.samples
[1, None]

Therefore, we know that, to preserve the photon number, ``q[1]`` must be in the state :math:`\ket{m+n-k}`, where :math:`m` and :math:`n` are the photon numbers of the initial states
and :math:`k` is value returned in :code:`result.samples`.
Executing the backend again, and this time applying the second Fock measurement:

.. code-block:: python3

    prog2 = sf.Program(2)
    with prog2.context as q:
        MeasureFock() | q[1]

    results = eng.run(prog2)


We will find that the second measurement yields :math:`m+n-k`. In this case, we get

>>> results.samples[1]
4


|PS|
==========

In addition, StrawberryFields also allows the specification or |ps| of a required measurement output, and will condition the remaining unmeasured modes based on this post-selected value.  When applying the measurement operators, the optional keyword argument ``select`` can be passed to the operator. The value should be an **integer** (or **list of integers**) for :class:`~.MeasureFock`, a **float** for :class:`~.MeasureHomodyne`, and a **complex value** for :class:`~.MeasureHeterodyne`.

For example, we can rewrite the example above using post-selection:

.. code-block:: python3

    import strawberryfields as sf
    from strawberryfields.ops import *

    prog = sf.Program(2)
    eng = sf.Engine("fock", backend_options={"cutoff_dim": 6})

    with prog.context as q:
        Fock(2) | q[0]
        Fock(3) | q[1]
        BSgate() | (q[0], q[1])
        MeasureFock(select=0) | q[0]
        MeasureFock() | q[1]

    result = eng.run(prog)

Since we are post-selecting a measurement of 0 photons in mode ``q[0]``, we expect ``result.samples[0]`` to be ``0`` and ``result.samples[1]`` to be ``5``. Indeed,

>>> result.samples
[0, 5]

.. warning::

    If we attempt to post-select on Fock measurement results that have zero probability given the circuit/state of the simulation, the Fock backend returns a ``ZeroDivisionError``. For example, in the previous code snippet, if we instead attempt to post-select two values that do not preserve the photon number,

    >>> eng.run("fock", cutoff_dim=6, select=[1,2])
    ZeroDivisionError: Measurement has zero probability.

    This check is provided for convenience, but the user should always be aware of post-selecting on zero-probability events. The current implementation of homodyne measurements in the Fock backend *does not* currently perform this check.

Example
---------

Consider the following circuit:


|

.. image:: ../_static/s_measure.svg
    :align: center
    :width: 30%
    :target: javascript:void(0);

|

Here, we have two vacuum states incident on a two-mode squeezed gate. Homodyne detection in the :math:`x` quadrature of the first output mode is then performed; as a result, the output mode ``q[1]`` is conditionally displaced depending on the measured value.

We can simulate this conditional displacement using post-selection. Utilizing the Gaussian backend, the above circuit can be simulated in Strawberry Fields as follows:

.. code-block:: python3

    import strawberryfields as sf
    from strawberryfields.ops import *

    prog = sf.Program(2)
    eng = sf.Engine("gaussian")

    with prog.context as q:
    with eng:
        S2gate(1)                    | (q[0], q[1])
        MeasureHomodyne(0,select=1)  | q[0]

    state = eng.run('gaussian').state

To check the displacement of the second output mode, we can use the :meth:`~.BaseGaussianState.reduced_gaussian` state method to extract the vector of means and the covariance matrix:

>>> mu, cov = state.reduced_gaussian([1])

The vector of means contains the mean quadrature displacements, and for a single mode is of the form :math:`\bar{\mathbf{r}} = (\bar{\mathbf{x}}, \bar{\mathbf{p}})`. Therefore, looking at the first index of the vector of means for ``q[1]``:

>>> print(mu[0])
0.964027569826

The :math:`x` quadrature displacement of the second mode is conditional to the post-selected value in the circuit construction above.


Measurement control and processing
=============================================

In addition to the features already explored above, Strawberry Fields also allows the measurement results of qumodes to be used as subsequent gate parameters. This is simple and intuitive as well - simply pass the register referencing the measured mode as the gate argument, for example like

``MeasureX | q[0]``

``Rgate(q[0]) | q[1]``

and the Strawberry Fields engine will, in the background, ensure that the measured value of that mode is used as the gate parameter during the circuit simulation.

Note that, the return type of the measurement determines the parameter type, potentially restricting the resulting gates which can be measurement-controlled.

.. rst-class:: docstable

+----------------------------------------------------+----------------+-------------------------------------------------------+
|                    Measurement                     |  Return type   |           Gates with matching parameter type          |
+====================================================+================+=======================================================+
| :class:`MeasureHomodyne(phi=0) <.MeasureHomodyne>` | Real number    | All                                                   |
+----------------------------------------------------+----------------+-------------------------------------------------------+
| :class:`MeasureHeterodyne() <.MeasureHeterodyne>`  | Complex number | :class:`~.Dgate`, :class:`~.Sgate`, :class:`~.S2gate` |
+----------------------------------------------------+----------------+-------------------------------------------------------+
| :class:`MeasureFock() <.MeasureFock>`              | Integer        | All                                                   |
+----------------------------------------------------+----------------+-------------------------------------------------------+


Classical processing
---------------------

Sometimes, additional classical processing needs to be performed on the measured value before using it as a gate parameter; Strawberry Fields provides some simple classical processing functions (known as **register transforms**) in the module :mod:`strawberryfields.utils`:

.. rst-class:: docstable

+-------------------------------------------+------------------------------------------------------------------+
|       Classical Processing function       |                           Description                            |
+===========================================+==================================================================+
| :func:`neg(q) <.neg>`                     | Negates the measured mode value, returns :math:`-q`              |
+-------------------------------------------+------------------------------------------------------------------+
| :func:`mag(q) <.mag>`                     | Returns the magnitude :math:`|q|` of a measured mode value.      |
+-------------------------------------------+------------------------------------------------------------------+
| :func:`phase(q) <.phase>`                 | Returns the phase :math:`\phi` of a complex measured mode value  |
+-------------------------------------------+------------------------------------------------------------------+
| :func:`scale(q,a) <.scale>`               | Returns :math:`aq`                                               |
+-------------------------------------------+------------------------------------------------------------------+
| :func:`shift(q,b) <.shift>`               | Returns :math:`q+b`                                              |
+-------------------------------------------+------------------------------------------------------------------+
| :func:`scale_shift(q,a,b) <.scale_shift>` | Returns :math:`aq+b`                                             |
+-------------------------------------------+------------------------------------------------------------------+
| :func:`power(q,a) <.power>`               | Returns :math:`q^a`. :math:`a` can be negative and/or fractional |
+-------------------------------------------+------------------------------------------------------------------+

These only need to be used when passing a measured mode value as a gate parameter. For example, if we wish to perform a Heterodyne measurement on a mode, and then use the measured **phase** to perform a controlled beamsplitter on other modes, we could do the following:

``MeasureHD | q[0]``

``BSgate(phase(q[0]), 0) | (q[1],q[2])``

In this particular example, we are casting the complex-valued Heterodyne measurement to a real value using the ``phase`` classical processing function, allowing us to pass it as a beamsplitter parameter.


User-defined processing functions
-----------------------------------

If you need a classical processing function beyond the basic ones provided in the :mod:`strawberryfields.utils` module, you can use the :func:`strawberryfields.convert` decorator to create your own. For example, consider the case where you might need to take the *logarithm* of a measured value, but only within a certain range, and use this as a subsequent gate parameter.

.. code-block:: python3

    import numpy as np
    import strawberryfields as sf
    from strawberryfields.ops import *

    @sf.convert
    def log(q):
        if 0.5<q<1:
            return np.log(q)
        else:
            return q

    prog = sf.Program(2)

    with prog.context as q:
        MeasureX      | q[0]
        Xgate(log(q)) | q[1]

By using the ``@sf.convert`` decorator directly above our user-defined custom processing function ``log(q)``, we convert this function into a register transform that can be applied directly to a measured mode as a gate parameter.

:html:`<div class="aside admonition" id="aside1"><a data-toggle="collapse" data-parent="#aside1" href="#content1" class="collapsed"><p class="first admonition-title">Advanced: RegRefTransforms (click to expand) <i class="fas fa-chevron-circle-down"></i></p></a><div id="content1" class="collapse" data-parent="#aside1" style="height: 0px;">`


Under the hood, the convert decorator is converting the user-defined processing function to a :class:`~.RegRefTransform` instance, which is how the Strawberry Fields engine understands transformations on qumodes. While it is always advised to use the built in classical processing functions, or the :func:`strawberryfields.convert` decorator for custom functions, the more advanced :class:`~.RegRefTransform` class can be used when more functionality is needed, for example processing functions on multiple qumodes.

The ``RegRefTransform`` is initialised as follows:

``RR([q[0],q[1],...], func(q0,q1,...))``

where the first argument is a sequence of :math:`n` qumodes, and the second argument is an :math:`n` argument function, with each argument corresponding to a qumode.

For example, the above user defined ``log`` function can be rewritten using an explicit ``RegRefTransform``:

.. code-block:: python3

    def log(q):
        if 0.5<q<1:
            return np.log(q)
        else:
            return q

    prog = sf.Program(2)

    with prog.context as q:
        MeasureX      | q[0]
        Xgate(RR(q[0],log)) | q[1]

However, ``RegRefTransform`` allows for more flexibility, by allowing us to define a classical processing function that acts on multiple qubits. For example, we can combine two Homodyne measurement results to form a single complex argument for a displacement gate:


.. code-block:: python3

    prog = sf.Program(3)

    with prog.context as q:
        MeasureX      | q[0]
        MeasureP      | q[1]
        Dgate(RR([q[0],q[1]], lambda q0,q1: q0+1j*q1)) | q[2]

:html:`</div></div>`
