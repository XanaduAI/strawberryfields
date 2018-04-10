.. role:: raw-latex(raw)
   :format: latex
   
.. role:: html(raw)
   :format: html

.. _introduction:

Introduction
=============================================

.. sectionauthor:: Nathan Killoran <nathan@xanadu.ai>

Many physical systems are intrinsically continuous, with light being the
prototypical example. Such systems reside in an infinite-dimensional
Hilbert space, offering a paradigm for quantum computation which is
distinct from the qubit model. This **continuous-variable model** takes
its name from the fact that the quantum operators underlying the model
have continuous spectra. 

The CV model is a natural fit for simulating bosonic systems
(electromagnetic fields, harmonic oscillators, phonons, Bose-Einstein condensates, or
optomechanical resonators) and for settings where continuous quantum
operators – such as position & momentum – are present. Even in classical
computing, recent advances from deep learning have demonstrated the
power and flexibility of a continuous picture of computation
:cite:`graves2014neural`:cite:`graves2016hybrid`.

From qubits to qumodes
----------------------

A high-level comparison of CV quantum computation with the qubit model
is depicted in the following table: 

.. rst-class:: docstable

+---------------------+--------------------------------------------------------------+---------------------------------------------------------------------+
|                     | .. centered::                                                | .. centered::                                                       |
|                     |  CV                                                          |  Qubit                                                              |
+=====================+==============================================================+=====================================================================+
| Basic element       | Qumodes                                                      | Qubits                                                              |
+---------------------+--------------------------------------------------------------+---------------------------------------------------------------------+
| Information unit    | 1 nat (:math:`\log_2e` bits)                                 | 1 bit                                                               |
+---------------------+--------------------------------------------------------------+---------------------------------------------------------------------+
| Relevant operators  | Quadrature operators                                         | Pauli operators                                                     |
|                     | :math:`\x,\p`                                                | :math:`\hat{\sigma}_x, \hat{\sigma}_y, \hat{\sigma}_z`              |
|                     |                                                              |                                                                     |
|                     | Mode operators                                               |                                                                     |
|                     | :math:`\a, \ad`                                              |                                                                     |
+---------------------+--------------------------------------------------------------+---------------------------------------------------------------------+
| Common states       | Coherent states :math:`\ket{\alpha}`                         | Pauli eigenstates :math:`\ket{0/1}, \ket{\pm}, \ket{\pm i}`         |
|                     |                                                              |                                                                     |
|                     | Squeezed states :math:`\ket{z}`                              |                                                                     |
|                     |                                                              |                                                                     |
|                     | Number states :math:`\ket{n}`                                |                                                                     |
+---------------------+--------------------------------------------------------------+---------------------------------------------------------------------+
| Common gates        | Rotation, Displacement, Squeezing, Beamsplitter, Cubic Phase | Phase Shift, Hadamard, CNOT, T Gate                                 |
|                     |                                                              |                                                                     |
+---------------------+--------------------------------------------------------------+---------------------------------------------------------------------+
| Common measurements | Homodyne :math:`\hat{x}_\phi`, Heterodyne :math:`Q(\alpha)`, | Pauli-basis measurements                                            |
|                     | Photon-counting :math:`\ketbra{n}{n}`                        | :math:`\ketbra{0/1}{0/1}, \ketbra{\pm}{\pm}, \ketbra{\pm i}{\pm i}` |
+---------------------+--------------------------------------------------------------+---------------------------------------------------------------------+

.. note:: Qubit-based computations can be embedded into the CV picture, e.g., by using the *Gottesman-Knill-Preskill (GKP) embedding* :cite:`lloyd1998analog`:cite:`braunstein1998error`:cite:`gottesman2001encoding`, so the CV model is as computationally powerful as its qubit counterparts. 


The most elementary CV system is the bosonic harmonic
oscillator, defined via the canonical mode operators :math:`\a` and
:math:`\ad`. These satisfy the well-known commutation relation
:math:`[\a,\ad]=\I`. It is also common to work with the **quadrature
operators**

.. math::

   \begin{aligned}
    \hat{x} := \sqrt{\frac{\hbar}{2}}(\a + \ad), \\
    \hat{p} := -i\sqrt{\frac{\hbar}{2}}(\a - \ad),\end{aligned}

with :math:`[\x,\p]=i \hbar`. These self-adjoint
operators are proportional to the hermitian and antihermitian parts of
the operator :math:`\a`. 

.. note:: Strawberry Fields uses the convention that :math:`\hbar=2` by default, but this can be changed as required.

We can picture a fixed harmonic oscillator
mode (say, within an optical fibre or waveguide on a photonic chip) as a
single ‘wire’ in a quantum circuit. These **qumodes** are the fundamental
information-carrying units of CV quantum computers. By combining
multiple qumodes (each with corresponding operators :math:`\a_i` and
:math:`\ad_i`) and interacting them via sequences of suitable quantum
gates, we can implement a general CV quantum computation.

CV states
---------

The dichotomy between qubit and CV systems is perhaps most evident in
the basis expansions of quantum states:

.. math::

   \begin{aligned}
     &\rm{Qubit} &\ket{\phi} & = \phi_0 \ket{0} + \phi_1 \ket{1}, \\
     &\rm{Qumode} &\ket{\psi} & = \int dx~\psi(x) \ket{x}. \end{aligned}

For qubits, we use a discrete set of coefficients; for CV systems, we
can have a *continuum*. The states :math:`\ket{x}` are the eigenstates of
the :math:`\x` quadrature, :math:`\x\ket{x}=x\ket{x}`, with
:math:`x\in\mathbb{R}`. These quadrature states are special cases of a
more general family of CV states, the **Gaussian states**, which we now
introduce.

Gaussian states
~~~~~~~~~~~~~~~

Our starting point is the vacuum state :math:`\ket{0}`. Other states can
be created by evolving the vacuum state according to

.. math::

    \ket{\psi} = \exp(-itH)\ket{0},

where :math:`H` is a bosonic Hamiltonian and :math:`t` is the evolution
time. States where the Hamiltonian :math:`H` is at most quadratic in the
operators :math:`\x` and :math:`\p` are called *Gaussian*. 

For a single
qumode, Gaussian states are parameterized by two continuous complex
variables: a displacement parameter :math:`\alpha\in\mathbb{C}` and a
squeezing parameter :math:`z\in\mathbb{C}` (often expressed as
:math:`z=r\exp(i\phi)`, with :math:`r \geq 0`). Gaussian states are
so-named because we can identify each Gaussian state, through its
displacement and squeezing parameters, with a corresponding Gaussian
distribution. The displacement gives the centre of the Gaussian, while
the squeezing determines the variance and rotation of the distribution.

.. note:: Many important pure states in the CV model are special cases of the pure Gaussian states; these are summarized in the following table.

.. rst-class:: docstable

+--------------------------------------------------+--------------------------------------------------------------------+---------------------------------------------------------------------+
| .. centered::                                    | .. centered::                                                      | .. centered::                                                       |
|   State family                                   |  Displacement                                                      |  Squeezing                                                          |
+==================================================+====================================================================+=====================================================================+
| Coherent states :math:`\ket{\alpha}`             | .. centered::                                                      | .. centered::                                                       |
|                                                  |   :math:`\alpha\in\mathbb{C}`                                      |   :math:`z=0`                                                       |
+--------------------------------------------------+--------------------------------------------------------------------+---------------------------------------------------------------------+
| Squeezed states :math:`\ket{z}`                  | .. centered::                                                      | .. centered::                                                       |
|                                                  |   :math:`\alpha=0`                                                 |   :math:`z\in\mathbb{C}`                                            |
+--------------------------------------------------+--------------------------------------------------------------------+---------------------------------------------------------------------+
| Displaced squeezed states :math:`\ket{\alpha,z}` | .. centered::                                                      | .. centered::                                                       |
|                                                  |   :math:`\alpha\in\mathbb{C}`                                      |   :math:`z\in\mathbb{C}`                                            |
+--------------------------------------------------+--------------------------------------------------------------------+---------------------------------------------------------------------+
| :math:`\x` eigenstates :math:`\ket{x}`           | .. centered::                                                      | .. centered::                                                       |
|                                                  |   :math:`\alpha\in\mathbb{C}`, :math:`x\propto\mathrm{Re}(\alpha)` |   :math:`\phi=0`, :math:`r\rightarrow\infty`                        |
+--------------------------------------------------+--------------------------------------------------------------------+---------------------------------------------------------------------+
| :math:`\p` eigenstates :math:`\ket{p}`           | .. centered::                                                      | .. centered::                                                       |
|                                                  |   :math:`\alpha\in\mathbb{C}`, :math:`p\propto\mathrm{Im}(\alpha)` |   :math:`\phi=\pi`, :math:`r\rightarrow\infty`                      |
+--------------------------------------------------+--------------------------------------------------------------------+---------------------------------------------------------------------+
| Vacuum state :math:`\ket{0}`                     | .. centered::                                                      | .. centered::                                                       |
|                                                  |   :math:`\alpha=0`                                                 |   :math:`z=0`                                                       |
+--------------------------------------------------+--------------------------------------------------------------------+---------------------------------------------------------------------+

Number states
~~~~~~~~~~~~~

Complementary to the continuous Gaussian states are the discrete **number
states** (or **Fock states**) :math:`\ket{n}`, :math:`n\in\mathbb{N}`.
These are the eigenstates of the number operator :math:`\n=\ad\a`. The
number states form a discrete countable basis for the states of a single
qumode. Thus, each of the Gaussian states considered in the previous
section can be expanded in the number state basis. For example, coherent
states have the form

.. math::

    \ket{\alpha} = \exp\left(-\tfrac{|\alpha|^2}{2}\right) \sum_{n=0}^\infty \frac{\alpha^n}{\sqrt{n!}}\ket{n},

while (undisplaced) squeezed states only have even number states in their expansion:

.. math::

    \ket{z} = \frac{1}{\sqrt{\cosh r}}\sum_{n=0}^\infty\frac{\sqrt{(2n)!}}{2^n n!}[-e^{i\phi}\tanh (r)]^n\ket{2n}.

Mixed states
~~~~~~~~~~~~

Mixed Gaussian states are also important in the CV picture, for
instance, the **thermal state**

.. math::

    \rho(\nbar) := \sum_{n=0}^\infty\frac{\nbar^n}{(1+\nbar)^{n+1}}\ketbra{n}{n},

which is parameterized via the mean photon number
:math:`\nbar:=\tr{(\rho(\nbar)\hat{n})}`. Starting from this state, we
can consider a mixed-state-creation process, similar to above, namely

.. math::

    \rho = \exp(-itH)\rho(\nbar)\exp(itH).

Analogously to pure states, by applying quadratic-order Hamiltonians to
thermal states, we generate the family of Gaussian mixed states.

CV gates
--------

Unitary operations can always be associated with a generating
Hamiltonian :math:`H` via the recipe 

.. math::

    U = \exp{(-itH)}.

For convenience, we can classify unitaries by the degree of their
generating Hamiltonians. We can build an N-mode unitary by
applying a sequence of gates from a **universal gate set**, each which acts
only on one or two modes. 

.. note:: A CV quantum computer is said to be universal if it can implement, to arbitrary precision and with a finite number of steps, any unitary which is polynomial in the mode operators :cite:`qccv1999`.

We focus on a universal gate set which contains the
following two components:

Gaussian gates
    One-mode and two-mode gates which are quadratic in the mode operators,
    e.g., *displacement, rotation, squeezing, and beamsplitter* gates. These are 
    equivalent to the Clifford group of gates from the qubit model.

Non-Gaussian gates
    A single-mode gate which is degree 3 or higher, e.g., the *cubic
    phase gate*. These are equivalent to the non-Clifford gates in the 
    qubit model.

A number of fundamental CV gates are presented in the following table:

.. rst-class:: docstable

+---------------+------------------------------------------------------------------------------------------+---------------------------------------------------------------------+
| .. centered:: | .. centered::                                                                            | .. centered::                                                       |
|   Gate        |   Unitary                                                                                |  Symbol                                                             |
+===============+==========================================================================================+=====================================================================+
| Displacement  | .. centered::                                                                            | .. image:: _static/Dgate.svg                                        |
|               |   :math:`D_i(\alpha)=\exp{(\alpha\ad_i - \alpha^*\a_i)}`                                 |   :align: center                                                    |
|               |                                                                                          |   :target: javascript:void(0);                                      |
+---------------+------------------------------------------------------------------------------------------+---------------------------------------------------------------------+
| Rotation      | .. centered::                                                                            | .. image:: _static/Rgate.svg                                        |
|               |   :math:`R_i(\phi)=\exp{(i\phi\hat{n}_i)}`                                               |   :align: center                                                    |
|               |                                                                                          |   :target: javascript:void(0);                                      |
+---------------+------------------------------------------------------------------------------------------+---------------------------------------------------------------------+
| Squeezing     | .. centered::                                                                            | .. image:: _static/Sgate.svg                                        |
|               |   :math:`S_i(z)=\exp{(\frac{1}{2}(z^* \a_i^2 - z \a_i^{\dagger 2}))}`                    |   :align: center                                                    |
|               |                                                                                          |   :target: javascript:void(0);                                      |
+---------------+------------------------------------------------------------------------------------------+---------------------------------------------------------------------+
| Beamsplitter  | .. centered::                                                                            | .. image:: _static/BSgate.svg                                       |
|               |   :math:`BS_{i,j}(\theta,\phi)=\exp{(\theta(e^{i\phi}\ad_i\a_j - e^{-i\phi}\a_i\ad_j))}` |   :align: center                                                    |
|               |                                                                                          |   :target: javascript:void(0);                                      |
+---------------+------------------------------------------------------------------------------------------+---------------------------------------------------------------------+
| Cubic Phase   | .. centered::                                                                            | .. image:: _static/Vgate.svg                                        |
|               |   :math:`V_i(\gamma)=\exp{(i\frac{\gamma}{6}\x_i^3)}`                                    |   :align: center                                                    |
|               |                                                                                          |   :target: javascript:void(0);                                      |
+---------------+------------------------------------------------------------------------------------------+---------------------------------------------------------------------+

    
.. note::

  We often also use the position displacement (:math:`X(x)=D(x/2)=\exp(-ix\p/2)` where :math:`x\in\mathbb{R}`) and momentum displacement (:math:`Z(p)=D(ip/2)=\exp(ip\x/2)` where :math:`p\in\mathbb{R}`) gates, specific cases of displacement defined above. 

  Together, these are known as the Weyl-Heisenberg group, satisfying the relation :math:`X(x)Z(p)=e^{-ixp/2}Z(p)X(x)`, and are analogous to the Pauli group operators in the discrete-variable formulation.

We can see that many of the Gaussian states from the previous
section are connected to a corresponding Gaussian gate. Any multimode
Gaussian gate can be implemented through a suitable combination of
Displacement, Rotation, Squeezing, and Beamsplitter Gates
:cite:`gqi2012`, making these gates sufficient
for quadratic unitaries. The cubic phase gate is presented as an
exemplary higher-order gate, but any other non-Gaussian gate could also
be used to achieve universality.

CV measurements
---------------

As with CV states and gates, we can distinguish between Gaussian and
non-Gaussian measurements. The Gaussian class consists of two
(continuous) types: **homodyne** and **heterodyne** measurements, while the key
non-Gaussian measurement is **photon counting**.

.. rst-class:: docstable

+---------------------+-----------------------------------------------------+-----------------------------------+
|                     | .. centered::                                       | .. centered::                     |
| Measurement         |   Measurement operator                              |   Measurement values              |
+=====================+=====================================================+===================================+
| Homodyne            | .. centered::                                       | .. centered::                     |
|                     |   :math:`\ket{\x_\phi}\bra{\x_\phi}`                |   :math:`q\in\mathbb{R}`          |
+---------------------+-----------------------------------------------------+-----------------------------------+
| Heterodyne          | .. centered::                                       | .. centered::                     |
|                     |   :math:`\frac{1}{\pi}|\alpha\rangle\langle\alpha|` |   :math:`\alpha\in\mathbb{C}`     |
+---------------------+-----------------------------------------------------+-----------------------------------+
| Photon counting     | .. centered::                                       | .. centered::                     |
|                     |   :math:`\ketbra{n}{n}`                             |   :math:`n\in\mathbb{N},n\geq 0`  |
+---------------------+-----------------------------------------------------+-----------------------------------+

Homodyne measurements
~~~~~~~~~~~~~~~~~~~~~

Ideal homodyne detection is a projective measurement onto the
eigenstates of the quadrature operator :math:`\x`. These states form a
continuum, so homodyne measurements are inherently continuous, returning
values :math:`x\in\mathbb{R}`. More generally, we can consider
projective measurement onto the eigenstates of the Hermitian operator

.. math:: \x_\phi:=\cos\phi~\x + \sin\phi~\p,

which is equivalent to rotating the state by :math:`-\phi` and
performing an :math:`\x`-homodyne measurement. If we have a multimode
Gaussian state and we perform a homodyne measurement on one of the modes,
the conditional state on the remaining modes stays Gaussian.

Heterodyne measurements
~~~~~~~~~~~~~~~~~~~~~~~

Whereas homodyne is a measurement of :math:`\x`, heterodyne can be seen
as a simultaneous measurement of both :math:`\x` and :math:`\p`. Because
these operators do not commute, they cannot be simultaneously measured
without some degree of uncertainty. 

The output of a heterodyne measurement is usually phrased in terms of the 
*Q function* :math:`Q(\alpha)=\langle\alpha|\rho|\alpha\rangle` (where
:math:`\{|\alpha\rangle\}_{\alpha\in\mathbb{C}}` are the coherent states). Like 
homodyne, heterodyne measurements preserve the Gaussian character of
Gaussian states.

Photon counting
~~~~~~~~~~~~~~~

Photon counting is a complementary measurement method to the “-dyne”
measurements, revealing the particle-like, rather than the wave-like,
nature of qumodes. This measurement projects onto the number eigenstates
:math:`\ket{n}`, returning non-negative integer values
:math:`n\in\mathbb{N}` [#]_. 

Except for the outcome :math:`n=0`, a
photon-counting measurement on a single mode of a multimode Gaussian
state will cause the remaining modes to become non-Gaussian. Thus,
photon-counting can be used as an ingredient for implementing
non-Gaussian gates. 

.. rubric:: Footnotes

.. [#] A related process is *photodetection*, where a detector only resolves the vacuum state from non-vacuum states. This process has only two measurement projectors, namely :math:`\ketbra{0}{0}` and :math:`\I - \ketbra{0}{0}`.

