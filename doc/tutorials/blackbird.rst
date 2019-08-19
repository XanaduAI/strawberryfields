.. _blackbird:

Blackbird programming language
#######################################

.. sectionauthor:: Nathan Killoran <nathan@xanadu.ai>, Josh Izaac <josh@xanadu.ai>

In this section, we provide an overview of the **Blackbird quantum programming language**. This simple and elegant language breaks down a quantum circuit into a set of instructions detailing the quantum operations we would like to apply, as well as the the subsystems that these operations act on [#f1]_. The Blackbird language is built-in to Strawberry Fields, but also exists as a separate Python `package <https://quantum-blackbird.readthedocs.io/en/latest/>`_.

Using the Strawberry Fields :mod:`.io` module, Blackbird scripts can be serialized and deserialized between files and Strawberry Fields Programs.

Operations
============

Blackbird is a quantum assembly language, capable of representing the basic continuous-variable (CV) states, gates, and measurements outlined in the :ref:`introduction`. Collectively, these are all considered as *Operations*. In Blackbird, there are four main types of Operations:

* State preparation

* Gate application

* Measurements

* Adding and removing subsystems

These all use the following general syntax:

``Operation(args) | q``

where ``args`` represents a list of parameters for the operation, and ``q`` is the qumode (or sequence of qumodes) the quantum operation acts upon. In Blackbird code, the symbol ``|`` *always* separates the operation to be applied (on the left) and the subsystem(s) it acts on (on the right).

State preparation
==================

States can be prepared using the state preparation Operators :class:`~.Vacuum`, :class:`~.Fock`, :class:`~.Coherent`, :class:`~.Squeezed`, :class:`~.DisplacedSqueezed`, and :class:`~.Thermal`. By default, all qumode subsystems are assumed initialised in the vacuum state.

.. code-block:: python3

    # State preparation in Blackbird
    Fock(1) | q[0]
    Coherent(0.5+2j) | q[1]

In the above example, we have prepared a single photon state in qumode ``q[0]``, and a coherent state with :math:`\alpha=0.5+2i` in qumode ``q[1]``. Note that we can also construct a state preparation operator with a particular set of parameters, and reuse it repeatedly:


.. code-block:: python3

    S = Squeezed(1)
    S | q[0]
    S | q[1]

Gate application
==================

Gates are applied within Blackbird code in the exact same manner as state preparations. Consider the following example:

.. code-block:: python3

    # Apply the Displacement gate to qumode 0
    alpha = 2.0 + 1j
    Dgate(alpha) | q[0]

    # Apply the Rotation gate
    phi = 3.14 / 2
    Rgate(phi) | q[0]

    # Apply the Squeezing gate
    Sgate(2.0, 0.17) | q[0]

    # Apply the Beamsplitter gate to qumodes 0 & 1
    BSgate(3.14 / 10, 0.223) | (q[0], q[1])

    # Apply the Cubic Phase gate (VGate) to qumode 0
    gamma = 0.1
    Vgate(gamma) | q[0]

Here, we are applying various gates, including the displacement gate (:class:`~.Dgate`), rotation gate (:class:`~.Rgate`), squeezing gate (:class:`~.Sgate`), beamsplitter (:class:`~.BSgate`, a two-mode gate), and the cubic phase gate (:class:`~.Vgate`). For more details on the gates available, as well as the parameters they take, see :ref:`gates`.

Note that gate Operations have some subtle differences to state preparation operators:

* Unlike state preparation operators, some gates (such as the beamsplitter above) can be applied to multiple qumodes.

  .. note:: The number of qumodes the gate acts upon and the sequence of qumodes to the right of the ``|`` operator must *always* match---we cannot apply the beamsplitter to a single qumode.

* We can also apply the Hermitian conjugate of a gate operator; this is specified by appending ``.H`` to the operator. For example:

  .. code-block:: python3

      V = Vgate(gamma)
      V.H | q[0]

.. note:: Operations must be applied in temporal order, from top to bottom.

Measurements
==================

In Blackbird, several CV measurement Operations are available; these include homodyne detection (:class:`.MeasureHomodyne`, as well as the shortcuts ``MeasureX`` and ``MeasureP``), heterodyne detection (:class:`MeasureHD <.MeasureHeterodyne>`), and photon detection (:class:`.MeasureFock`). These are applied directly to the qumodes to be measured:

.. code-block:: python3

    # Homodyne measurement at angle phi
    phi = 0.25 * 3.14
    MeasureHomodyne(phi) | q[0]

    # Special homodyne measurements
    MeasureX | q[0]
    MeasureP | q[1]

    # Heterodyne measurement
    MeasureHeterodyne() | q[0]
    MeasureHD           | q[1]  # shorthand

    # Number state measurements of various qumodes
    MeasureFock() | q[0]
    MeasureFock() | (q[1], q[2]) # multiple modes

For more details on measurements, as well as advanced features such as postselection, see the :ref:`ps_tutorial`.

Otherwise, to see how Blackbird programs are used in practice within Strawberry Fields, continue on to the :ref:`tutorial`.

.. rubric:: Footnotes

.. [#] Note: the Blackbird syntax is modeled after that of `Project Q <https://projectq.ch/>`_, but specialized to the CV setting.
