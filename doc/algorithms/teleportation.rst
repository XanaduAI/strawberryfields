.. _state_teleportation:

State teleportation
===================

   "A trick that quantum magicians use to produce phenomena that cannot be imitated by classical magicians." -  A. Peres :cite:`bru2002`


.. role:: html(raw)
   :format: html

Quantum teleportation - sometimes referred to as state teleportation to avoid confusion with gate teleportation - is the reliable transfer of an unknown quantum state across spatially separated qubits or qumodes, through the use of a classical transmission channel and quantum entanglement :cite:`bennett1993`. Considered a fundamental quantum information protocol, it has applications ranging from quantum communication to enabling distributed information processing in quantum computation :cite:`furusawa2011`.

In general, all quantum teleportation circuits work on the same basic principle. Two distant observers, Alice and Bob, share a maximally entangled quantum state (in discrete variables, any one of the four `Bell states <https://en.wikipedia.org/wiki/Bell_state>`_; or in CV, a maximally entangled state for a fixed energy), and have access to a classical communication channel. Alice, in possession of an unknown state which she wishes to transport to Bob, makes a joint measurement of the unknown state and her half of the entangled state, by projecting onto the Bell basis. By transmitting the results of her measurement to Bob, Bob is then able to transform his half of the entangled state to an accurate replica of the original unknown state, by performing a conditional phase flip (for qubits) or displacement (for qumodes) :cite:`steeb2006`.

CV implementation
----------------------------------

While originally designed for discrete-variable quantum computation with qubits, the (spatially separated) quantum teleportation algorithm described above can be easily translated to CV qumodes; the result is shown in the following circuit:

:html:`<br>`

.. image:: ../_static/teleport.svg
   :width: 60%
   :align: center
   :target: javascript:void(0);


:html:`<br>`

This process can be explained as follows:

1. Here, qumodes :math:`q_1` and :math:`q_2` are initially prepared as (the unphysical) infinitely squeezed vacuum states in momentum and position space respectively,

   .. math::
    &\ket{0}_x \sim \lim_{z\rightarrow\infty} S(z)\ket{0}\\
    &\ket{0}_p \sim \lim_{z\rightarrow-\infty} S(z)\ket{0}=\frac{1}{\sqrt{\pi}}\int_{-\infty}^\infty \ket{x}~dx

   before being maximally entangled by a 50-50 beamsplitter:

   .. math:: BS(\pi/4,0)(\ket{0}_p\otimes\ket{0}_x)


2. These two qumodes are now spatially separated, with :math:`\ket{q_1}` held by Alice, and :math:`\ket{q_2}` held by Bob, with the two connected via the classical communication channels :math:`c_0` and :math:`c_1`.


3. To teleport her unknown state :math:`\ket{\psi}` to Bob, Alice now performs a projective measurement of her entire system onto the maximally entangled basis states. This is done by entangling :math:`\ket{\psi}` and :math:`\ket{q_1}` via another 50-50 beamsplitter, before performing two homodyne measurements, in the :math:`x` and :math:`p` quadratures respectively.


4. The results of these measurements are then transmitted to Bob, who performs both a position displacement (conditional on the :math:`x` measurement) and a momentum displacement (conditional on the :math:`p` measurement) to recover exactly the transmitted state :math:`\ket{\psi}`.

Blackbird code
---------------
This can be easily implemented using the Blackbird quantum circuit language:

.. literalinclude:: ../../examples/teleportation.py
   :language: python
   :linenos:
   :dedent: 4
   :tab-width: 4
   :start-after: with teleportation.context as q:
   :end-before: # end circuit

Some important notes:

* Infinite squeezed vacuum states are not physically realizable; preparing the states with a squeezing factor of :math:`|r|=2` (:math:`\sim 18\text{dB}`) is a reasonable approximation.

* The function :func:`~.scale` can be imported from :mod:`strawberryfields.utils`, and allows classical processing on the measured register value; in this case, multiplying it by the correctional factor :math:`\sqrt{2}`. Other simple classical processing functions are available in the ``utils`` module; however if more advanced classical processing is required, custom classical processing functions can be created using the :func:`strawberryfields.convert` decorator.

* :meth:`~strawberryfields.ops.BSgate` accepts two arguments, ``theta`` and ``phi``. A variable storing the value of :math:`\pi` is used for setting these parameters - in Python, this can be imported from NumPy,

  .. code-block:: python

    from numpy import pi

.. note:: A fully functional Strawberry Fields simulation containing the above Blackbird code is included at :download:`examples/teleportation.py <../../examples/teleportation.py>`.
