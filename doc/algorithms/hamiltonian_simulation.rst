.. role:: html(raw)
   :format: html

.. _ham_sim:

Hamiltonian simulation
======================

	"The problem of simulating the dynamics of quantum systems was the original motivation for quantum computers and remains one of their major potential applications." - Berry et al. :cite:`berry2015`

The simulation of atoms, molecules and other biochemical systems is another application uniquely suited to quantum computation. For example, the ground state energy of large systems, the dynamical behaviour of an ensemble of molecules, or complex molecular behaviour such as protein folding, are often computationally hard or downright impossible to determine via classical computation or experimentation :cite:`aspuruguzik2005hamiltonian,whitfield2011hamiltonian`.

In the discrete-variable qubit model, efficient methods of Hamiltonian simulation have been discussed at-length, providing several implementations depending on properties of the Hamiltonian, and resulting in a linear simulation time :cite:`childs2012hamiltonian,berry2006hamiltonian`. Efficient implementations of Hamiltonian simulation also exist in the CV formulation :cite:`kalajdzievski2018hamiltonian`, with specific application to `Bose-Hubbard Hamiltonians <https://en.wikipedia.org/wiki/Bose%E2%80%93Hubbard_model>`_ (describing a system of interacting bosonic particles on a lattice of orthogonal position states :cite:`sowinski2012hamiltonian`). As such, this method is ideally suited to photonic quantum computation.

CV implementation
----------------------------------

For a quick example, consider a lattice composed of two adjacent nodes:


.. raw:: html

    <br>

.. image:: ../_static/graph.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);


.. raw:: html

    <br>

This graph is represented by the :math:`2\times 2` adjacency matrix :math:`A=\begin{bmatrix}0&1\\1&0\end{bmatrix}`. Here, each node in the graph represents a qumode, so we can model the dynamics of Bosons on this structure via a 2-qumode CV circuit.

The `Bose-Hubbard Hamiltonian <https://en.wikipedia.org/wiki/Bose%E2%80%93Hubbard_model>`_ with on-site interactions is given by

.. math:: H = J\sum_{i}\sum_j A_{ij} \ad_i\a_j + \frac{1}{2}U\sum_i \hat{n}_i(\hat{n}_i-1)= J(\ad_1 \a_2 + \ad_2\a_1) + \frac{1}{2}U (  \hat{n}_1^2 - \hat{n}_1 + \hat{n}_2^2 - \hat{n}_2)

where :math:`J` represents the transfer integral or hopping term of the boson between nodes, and :math:`U` is the on-site interaction potential. Here, :math:`\ad_1 \a_2` represents a boson transitioning from node 1 to node 2, while :math:`\ad_2\a_1` represents a boson transitioning from node 2 to node 1, and :math:`\hat{n}_i=\ad_i\a_i` is the number operator applied to mode :math:`i`. Applying the Lie-product formula, we find that

.. math:: e^{-iHt} = \left[\exp\left({-i\frac{ J t}{k}(\ad_1 \a_2 + \ad_2\a_1)}\right)\exp\left(-i\frac{Ut}{2k}\hat{n}_1^2\right)\exp\left(-i\frac{Ut}{2k}\hat{n}_2^2\right)\exp\left(i\frac{Ut}{2k}\hat{n}_1\right)\exp\left(i\frac{Ut}{2k}\hat{n}_2\right)\right]^k+\mathcal{O}\left(t^2/k\right),

where :math:`\mathcal{O}\left(t^2/k\right)` is the order of the error term, derived from the Lie product formula. Comparing this to the form of various CV gates, we can write this as the product of :ref:`symmetric beamsplitters <beamsplitter>` (:class:`~.BSgate`), :ref:`Kerr gates <kerr>` (:class:`~.Kgate`), and :ref:`rotation gates <rotation>` (:class:`~.Rgate`):

.. math:: e^{iHt} = \left[BS\left(\theta,\phi\right)\left(K(r)R(-r)\otimes K(r)R(-r)\right)\right]^k+\mathcal{O}\left(t^2/k\right).

where :math:`\theta=-Jt/k`, :math:`\phi=\pi/2`, and :math:`r=-Ut/2k`.

For the case :math:`k=2`, this can be drawn as the circuit diagram

|

.. image:: ../_static/hamsim.svg
    :align: center
    :width: 70%
    :target: javascript:void(0);

|

For more complex CV decompositions, including those with interactions, see Kalajdzievski et al. :cite:`kalajdzievski2018hamiltonian` for more details.

Blackbird code
---------------

The Hamiltonian simulation circuit displayed above for the 2-node lattice, can be implemented using the Blackbird quantum circuit language:

.. literalinclude:: ../../examples/hamiltonian_simulation.py
   :language: python
   :linenos:
   :dedent: 4
   :tab-width: 4
   :start-after: with ham_simulation.context as q:
   :end-before: # end circuit

where, for this example, we have set ``J=1``, ``U=1.5``, ``k=20``, ``t=1.086``, ``theta = -J*t/k``, and ``r = -U*t/(2*k)``. After constructing the circuit and running the engine,

.. literalinclude:: ../../examples/hamiltonian_simulation.py
   :language: python
   :linenos:
   :start-after: # run the engine
   :end-before: # the output state probabilities

the site occupation probabilities can be calculated via

>>> state = results.state
>>> state.fock_prob([2,0])
0.52240124572001989
>>> state.fock_prob([1,1])
0.23565287685672454
>>> state.fock_prob([0,2])
0.24194587742325951

As Hamiltonian simulation is particle preserving, these probabilities should add up to one; indeed, summing them results in a value :math:`\sim1.0000000000000053`.

We can compare this result to the analytic matrix exponential :math:`e^{-iHt}`, where the matrix elements of :math:`H` can be computed in the Fock basis. Considering the diagonal interaction terms,

.. math::
    & \braketT{0,2}{H}{0,2} = \frac{1}{2}U\braketT{0}{(\hat{n}^2-\hat{n})}{0} + \frac{1}{2}U\braketT{2}{(\hat{n}^2-\hat{n})}{2} = \frac{1}{2}U(2^2-2) = U\\[7pt]
    & \braketT{1,1}{H}{1,1} = 0\\[7pt]
    & \braketT{2,0}{H}{2,0} = U

as well as the off-diagonal hopping terms,

.. math::
    & \braketT{1,1}{H}{0,2} = J\braketT{1,1}{\left(\ad_1\a_2 + \a_1\ad_2\right)}{0,2} = J(\sqrt{1}\sqrt{2} + \sqrt{0}\sqrt{3}) = J\sqrt{2}\\[7pt]
    & \braketT{1,1}{H}{2,0} = J\sqrt{2}

and taking into account the Hermiticity of the system, we arrive at

.. math:: H = \begin{bmatrix}U&J\sqrt{2}&0\\ J\sqrt{2} & 0 & J\sqrt{2}\\ 0 & J\sqrt{2} & U\end{bmatrix}

which acts on the Fock basis :math:`\{\ket{0,2},\ket{1,1},\ket{2,0}\}`. Using the SciPy matrix exponential function :py:func:`scipy.linalg.expm`:

>>> from scipy.linalg import expm
>>> H = J*np.sqrt(2)*np.array([[0,1,0],[1,0,1],[0,1,0]]) + U*np.diag([1,0,1])
>>> init_state = np.array([0,0,1])
>>> np.abs(np.dot(expm(-1j*t*H), init_state))**2
[ 0.52249102,  0.23516277,  0.24234621]

which agrees, within the expected error margin, with our Strawberry Fields Hamiltonian simulation.

.. note::
  A fully functional Strawberry Fields simulation containing the above Blackbird code is included at :download:`examples/hamiltonian_simulation.py <../../examples/hamiltonian_simulation.py>`.
