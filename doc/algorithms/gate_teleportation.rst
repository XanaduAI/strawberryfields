.. role:: html(raw)
   :format: html

.. _gate_teleport:

Gate teleportation
====================

    "Entanglement-assisted communication becomes entanglement-assisted computation" - Furusawa :cite:`furusawa2011`

In the quantum state teleportation algorithm, the quantum state is transferred from the sender to the receiver exactly. However, quantum teleportation can be used in a much more powerful manner, by simultaneously processing and manipulating the teleported state; this is known as **gate teleportation**.

But the biggest departure from its namesake is the method in which the gate to be 'teleported' is applied; rather than applying a quantum unitary directly to the first qumode in the system, the unitary is applied via the projective measurement of the first qumode onto a particular basis. This measurement-based approach provides significant advantages over applying unitary gates directly, for example by reducing resources, and in the application of experimentally hard-to-implement gates :cite:`furusawa2011`. In fact, gate teleportation forms a universal quantum computing primitive, and is a precursor to cluster state models of quantum computation :cite:`gottesman1999`:cite:`gu2009`.


CV implementation
-----------------------------------

First described by Gottesman and Chuang :cite:`gottesman1999` in the case of qubits, gate teleportation was generalized for the CV case by Bartlett and Munro in 2003 :cite:`bartlett2003`. In an analogous process to the discrete-variable case, you begin with the algorithm for **local state teleportation**:

.. image:: ../_static/gate_teleport1.svg
    :align: center
    :width: 50%
    :target: javascript:void(0);

:html:`<br>`

Note that:

* Unlike the spatially-separated quantum state teleportation we considered in the previous section, **local teleportation** can transport the state using only two qumodes; the state we are teleporting is entangled directly with the squeezed vacuum state in the momentum space through the use of a controlled-phase gate.


* The state is then teleported to qumode :math:`q_1` via a homodyne measurement in the computational basis (the position quadrature).


* Like in the previous section, to recover the teleported state exactly, we must perform Weyl-Heisenberg corrections to :math:`q_1`; here, that would be :math:`F^\dagger X(m)^\dagger`. However, for convenience and simplicity, we write the circuit without the corrections applied explicitly.

Rather than simply teleporting the state as-is, we can introduce an arbitrary unitary :math:`U` that acts upon :math:`\ket{\psi}`, as follows:

.. image:: ../_static/gate_teleport2.svg
    :align: center
    :width: 50%
    :target: javascript:void(0);

:html:`<br>`

Now, the action of the unitary :math:`U` is similarly teleported along with the initial state --- this is a trivial extension of the local teleportation circuit. In order to view this in as a measurement-based universal quantum computing primitive, we make a couple of important changes:

* The inverse Fourier gate is absorbed into the measurement, making it a homodyne detector in the momentum quadrature

* The unitary gate :math:`U`, if diagonal in the computational basis (i.e., it is of the form :math:`U=e^{i f(\hat{x}^i)}`), commutes with the controlled-phase gate (:math:`CZ(s)=e^{i s ~\hat{x_1}\otimes\hat{x_2}/\hbar}`), and can be moved to the right of it. It is then also absorbed into the projective measurement.

.. image:: ../_static/gate_teleport3.svg
    :align: center
    :width: 50%
    :target: javascript:void(0);

:html:`<br>`

Additional gates can now be added simply by introducing additional qumodes with the appropriate projective measurements, all 'stacked vertically' (i.e., coupled to the each consecutive qumode via a controlled-phase gate). From this primitive, the model of cluster state quantum computation can be derived :cite:`gu2009`.

.. note:: What happens if the unitary is *not* diagonal in the computational basis? In this case, **feedforward** is required; additional qumodes and projective measurements are introduced, with successive measurements dependent on the previous result :cite:`loock2007`.


Blackbird code
---------------

Consider the following gate teleportation circuit,

.. image:: ../_static/gate_teleport_ex.svg
    :align: center
    :width: 70%
    :target: javascript:void(0);

:html:`<br>`

Here, the state :math:`\ket{\psi}`, a squeezed state with :math:`r=0.1`, is teleported to the final qumode, with the quadratic phase gate (:class:`~strawberryfields.ops.Pgate`) :math:`P(s)=e^{is\hat{x}^2/2\hbar}` teleported to act on it - with the quadratic phase gate chosen as it is diagonal in the :math:`\x` quadrature. This can be easily implemented using the Blackbird quantum circuit language:

.. literalinclude:: ../../examples/gate_teleportation.py
   :language: python
   :linenos:
   :dedent: 4
   :tab-width: 4
   :start-after: with gate_teleportation.context as q:
   :end-before: # compare


Some important notes:

* As with the state teleportation circuit above, perfectly squeezed vacuum states are not physically realizable; preparing the states with a squeezing factor of :math:`|r|=2` (:math:`\sim 18\text{dB}`) is a reasonable approximation.

..

* The Blackbird notation ``Operator.H`` denotes the Hermitian conjugate of the corresponding operator.

..

* Here, we do not make the corrections to the final state; this is left as an exercise to the reader. For additional details, see the gate teleportation commutation relations derived by van Loock :cite:`loock2007`.


To easily check that the output of the circuit is as expected, we can make sure that it agrees with the (uncorrected) state

.. math:: X({q_1})FP(0.5)X(q_0)F \ket{z}

which can be generated as follows:

.. literalinclude:: ../../examples/gate_teleportation.py
   :language: python
   :dedent: 4
   :tab-width: 4
   :start-after: # not including the corrections
   :end-before: # end circuit

.. note:: A fully functional Strawberry Fields simulation containing the above Blackbird code is included at :download:`examples/gate_teleportation.py <../../examples/gate_teleportation.py>`.
