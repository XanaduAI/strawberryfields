
.. raw:: html

   <style>
      h1 {
         counter-reset: section;
      }
      h2::before {
        counter-increment: section;
        content: counter(section) ". ";
        white-space: pre;
      }
      h2 {
         border-bottom: 1px solid #ddd;
      }
      h2 a {
         color: black;
      }
      h3 a {
         color: black;
      }
      .local > ul {
         list-style-type: decimal;
      }
      .local > ul ul {
         list-style-type: initial;
      }
   </style>

States
==========================



.. note:: In Strawberry Fields we use the convention :math:`\hbar=2` by default, but other conventions can also be chosen on engine :func:`initialization <strawberryfields.Engine>`. In this document we keep :math:`\hbar` explicit.

.. contents:: Contents
   :local:
   :depth: 1


.. _gaussian_basis:

Gaussian States
---------------------------------------------

As mentioned in the introduction, a pure Gaussian state is one that can be obtained by applying gates
whose generators are at most **quadratic** in the quadrature (or creation and destruction) operators.
For this class of states, one can obtain a very concise description in terms of covariance matrices.
Let us consider a system of :math:`N` qumodes.  We can now define the following vector of quadrature operators

.. math:: \mathbf{\hat{ r}}=(\x_1, \ldots, \x_N, \p_1, \ldots, \p_N).


The commutation relations of the :math:`2N` position and momentum operators in the last equation can be easily summarized as :math:`[\hat r_i, \hat r_j] = \hbar i\Omega_{ij}`, where

.. math::  \Omega = \begin{bmatrix} 0 & \I_N \\-\I_N & 0 \\\end{bmatrix}

is the **symplectic matrix**. Using the symplectic matrix, we can define the **Weyl operator** :math:`D(\xi)` (a multimode displacement operator) and the **characteristic function** :math:`\chi(\xi)` of a quantum :math:`N` mode state :math:`\rho`:

.. math:: \hat D(\xi) = \exp\left(i \mathbf{\hat{ r}} \Omega \xi\right), ~~~~ \chi(\xi) = \langle \hat D(\xi) \rangle_{\rho}, 

where :math:`\xi \in \mathbb{R}^{2N}`.

We can now consider the Fourier transform of the characteristic function to obtain the `Wigner function <https://en.wikipedia.org/wiki/Wigner_quasiprobability_distribution>`_ of the state :math:`\hat \rho`:

.. math::  W(\mathbf{r}) = \int_{\mathbb{R}^{2N}} \frac{d^{2N} \xi}{(2 \pi)^{2N}} \exp( -i \mathbf{r} \Omega \xi )\chi(\xi).
	  
The :math:`2N` real arguments :math:`\mathbf{r}` of the Wigner function are the eigenvalues of the quadrature operators from :math:`\mathbf{\hat{r}}`.

The above recipe maps an :math:`N`-mode quantum state living in a Hilbert space to the real symplectic space :math:`\mathcal{K}:=(\mathbb{R}^{2N},\Omega)`, which is called **phase space**. Like a probability distribution over this phase space, the Wigner function is normalized to one; unlike a probability distribution, it may take negative values. Gaussian states have the special property that their characteristic function (and hence their Wigner function) is a Gaussian function of the variables :math:`\mathbf{r}`. In this case, the Wigner function takes the form
 
.. math::    W(\mathbf{r}) =\frac{\exp\left( -\frac{1}{2} (\mathbf{r} -\bar{ \mathbf{r}}) \mathbf{V}^{-1} (\mathbf{r}-\bar{\mathbf{r}} ) \right)}{(2 \pi)^N \sqrt{\text{det} \mathbf{V}}},

where :math:`\bar{ \mathbf{r}} = \langle \hat{\mathbf{r}} \rangle_\rho = \text{Tr}\left(\hat{ \mathbf{r}}  \hat \rho \right)` is the displacement or mean vector and :math:`\mathbf{V}_{ij} = \frac{1}{2} \langle \Delta r_i \Delta r_j+ \Delta r_j \Delta r_i \rangle` with :math:`\Delta \hat{\mathbf{ r}}=\hat{ \mathbf{r}}-\bar{ \mathbf{r}}` is the covariance matrix. Gaussian states are completely charaterized by the mean vector and covariance matrix (their first and second moments respectively).

.. note:: The diagonal elements of the covariance matrix are simply the *variances* of each quadrature :math:`V_{i,i} = (\Delta x_i)^2` and :math:`V_{i+N,i+N}=(\Delta p_i)^2` for :math:`i \leq N`, where :math:`\Delta x_i\Delta p_i\geq\frac{\hbar}{2}`.

.. note::  The only pure states that have non-negative Wigner functions are the pure Gaussian states. A Gaussian state is pure if and only if its covariance matrix satisfies :math:`\text{det}\mathbf{V}=\left(\frac{\hbar}{2}\right)^{2N}`.

Mean vector and covariance
^^^^^^^^^^^^^^^^^^^^^^^^^^

Each type of Gaussian state has a specific form of covariance matrix :math:`\mathbf{V}` and mean vector :math:`\bar{\mathbf{r}}`:

* For the **single-mode vacuum state**, we have :math:`\mathbf{V}=\frac{\hbar}{2}\I_2` and :math:`\bar{\mathbf{r}}=(0,0)^T`,

.. 

* A **thermal state** has the same (zero) displacement but a covariance matrix :math:`\mathbf{V}=(2 \bar n+1)\frac{\hbar}{2}\I_2`;

.. 

* A **coherent state**, obtained by displacing vacuum, has the same :math:`\mathbf{V}` as vacuum but a nonzero displacement vector :math:`\bar{ \mathbf{r}}=\sqrt{2\hbar}(\text{Re}(\alpha),\text{Im}(\alpha))`;

.. 

* and lastly, a **squeezed state** has zero displacement and covariance matrix  :math:`\mathbf{V} = \frac{\hbar}{2} \text{diag}(e^{-2r},e^{2r})`.

In the limit :math:`r \to \infty`, the squeezed state's variance in the :math:`\x` quadrature becomes zero and the state becomes proportional to the :math:`\x`-eigenstate :math:`\ket{x}` with eigenvalue 0. Consistent with the uncertainty principle, the squeezed state's variance in :math:`\p` blows up.

We can also consider the case :math:`r \to -\infty`, where we find a state proportional to the eigenstate :math:`\ket{p}` of the :math:`\p` quadrature with eigenvalue 0. In the laboratory and in numerical simulation we must approximate every quadrature eigenstate using a finitely squeezed state (being careful that the variance of the relevant quadrature is much smaller than any other uncertainty relevant to the system). Any other quadrature eigenstate can be obtained from the :math:`x=0` eigenstate by applying suitable displacement and rotation operators.

Finally, note that Gaussian operations will transform the vector of means via an affine transformation and the covariance matrix via similarity transformation; for a detailed discussion of these transformation, see Sec. 2 of *Gaussian quantum information* by Weedbrook et al. in the :ref:`further reading <further_reading>` section.

Given a :math:`2N \times 2N` real symmetric matrix, how can we check that it is a valid covariance matrix? And if it is valid, which operations (displacement, squeezing, multiport interferometers) should be performed to prepare the corresponding Gaussian state?

To answer the first question: a :math:`2N \times 2N` real symmetric matrix :math:`\tilde{ \mathbf{V}}` corresponds to a Gaussian quantum state if and only if :math:`\tilde{ \mathbf{V}}+i \frac{\hbar}{2}\Omega \geq 0` (the matrix inequality is understood in the sense that the eigenvalues of the quantity :math:`\tilde{\mathbf{V}}+i \frac{\hbar}{2} \Omega` are nonnegative).


The answer to the second question is provided by the *Bloch-Messiah reduction* :cite:`bloch1975canonical`:cite:`braunstein2005squeezing`:cite:`simon1994quantum`. This reduction shows that any :math:`N`-mode Gaussian state (equivalently any covariance matrix and vector of means) can be constructed by starting with a product of :math:`N` thermal states :math:`\bigotimes_i \rho_i(\bar n_i)` (with potentially different mean photon numbers), then applying a multiport interferometer :math:`\mathcal{V}`, followed by single-mode squeezing operations :math:`\bigotimes_i S_i(z_i)` followed by another multiport :math:`\mathcal{U}` followed by single-mode displacement operations :math:`\bigotimes_i D_i(\alpha_i)`. Explicitly,

.. math::  \rho_\text{Gaussian} = \mathcal{W} \left( \bigotimes_i \rho_i(\bar n_i) \right) \mathcal{W}^\dagger, ~~~~~
	   \mathcal{W} = \left(\bigotimes_i D_i(\alpha_i) \right)\mathcal{U} \left(\bigotimes_i S_i(z_i) \right) \mathcal{V}.

If the state is pure then :math:`\bigotimes_i \rho_i(\bar n_i) = \ket{0} \bra{0}` where :math:`\ket{0}` is the multimode vacuum state. Note that in this case one :math:`\mathcal{V} \ket{0}=\ket{0}` and thus one can write

.. math:: \ket{\psi}_\text{Gaussian} = \left(\bigotimes_i D_i(\alpha_i) \right)\mathcal{U} \left(\bigotimes_i S_i(z_i) \right) \ket{0}

Finally, just one minor note about conventions; the multiport interferometer is typically specified by providing its action on the destruction operators of the incoming modes

.. math:: \a_i \to \mathcal{U} \a_i \mathcal{U}^\dagger = \sum_j^N U_{i,j} \a_j

where the matrix :math:`U` is unitary :math:`U U^\dagger = U^\dagger U = \I_N`.



.. _fock_basis:

Fock basis
---------------------------------------------

.. warning:: The Fock basis is **non-Gaussian**, and thus states listed here can only be used in the Fock backends, *not* the Gaussian backend.

.. admonition:: Definition
   :class: defn

   A single mode state can be decomposed into the Fock basis as follows:

   .. math::
      \ket{\psi} = \sum_n c_n \ket{n}

   if there exists a unique integer :math:`m` such that :math:`\begin{cases}c_n=1& n=m\\c_n=0&n\neq m\end{cases}`, then the single mode is simply a Fock state or :math:`n` photon state.

.. tip::

   *Implemented in Strawberry Fields as a state preparation operator by* :class:`strawberryfields.ops.Fock`

   *Available in Strawberry Fields as a NumPy array by* :func:`strawberryfields.utils.fock_state`

   *Implemented in Strawberry Fields as a state preparation operator by* :class:`strawberryfields.ops.Ket`


When working with an :math:`N`-mode density matrix in the Fock basis,

.. math:: \rho = \sum_{n_1}\cdots\sum_{n_N} c_{n_1,\cdots,n_N}\ket{n_1,\cdots,n_N}\bra{n_1,\cdots,n_N}

we use the convention that every pair of consecutive dimensions corresponds to a subsystem; i.e.,

.. math:: \rho_{\underbrace{ij}_{\text{mode}~0}~\underbrace{kl}_{\text{mode}~1}~\underbrace{mn}_{\text{mode}~2}}

Thus, using index notation, we can calculate the reduced density matrix for mode 2 by taking the partial trace over modes 0 and 1:

.. math:: \braketT{n}{\text{Tr}_{01}[\rho]}{m} = \sum_{i}\sum_k \rho_{iikkmn}

.. _vacuum_state:

Vacuum state
---------------------------------------------

.. note:: By default, newly created modes in Strawberry Fields default to the vacuum state

.. admonition:: Definition
   :class: defn

   The vacuum state :math:`\ket{0}` is a Gaussian state defined by

   .. math::
      \ket{0} = \frac{1}{\sqrt[4]{\pi \hbar}}\int dx~e^{-x^2/(2 \hbar)}\ket{x} ~~\text{where}~~ \a\ket{0}=0

.. tip::

   *Implemented in Strawberry Fields as a state preparation operator by* :class:`strawberryfields.ops.Vacuum`

   *Available in Strawberry Fields as a NumPy array by* :func:`strawberryfields.utils.vacuum_state`

In the Fock basis, it is represented by Fock state :math:`\ket{0}`, and in the Gaussian formulation, by :math:`\bar{\mathbf{r}}=(0,0)` and :math:`\mathbf{V}= \frac{\hbar}{2} I`.

.. _coherent_state:

Coherent state
---------------------------------------------

.. admonition:: Definition
   :class: defn

   The coherent state :math:`\ket{\alpha}`, :math:`\alpha\in\mathbb{C}` is a displaced vacuum state defined by

   .. math::
      \ket{\alpha} = D(\alpha)\ket{0}

.. tip::

   *Implemented in Strawberry Fields as a state preparation operator by* :class:`strawberryfields.ops.Coherent`

   *Available in Strawberry Fields as a NumPy array by* :func:`strawberryfields.utils.coherent_state`

A coherent state is a minimum uncertainty state, and the eigenstate of the annihilation operator:

.. math:: \a\ket{\alpha} = \alpha\ket{\alpha}

In the Fock basis, it has the decomposition

.. math:: |\alpha\rangle = e^{-|\alpha|^2/2} \sum_{n=0}^\infty
        \frac{\alpha^n}{\sqrt{n!}}|n\rangle

whilst in the Gaussian formulation, :math:`\bar{\mathbf{r}}=2 \sqrt{\frac{\hbar}{2}}(\text{Re}(\alpha), \text{Im}(\alpha))` and :math:`\mathbf{V}= \frac{\hbar}{2} I`.

.. _squeezed_state:

Squeezed state
---------------------------------------------

.. admonition:: Definition
   :class: defn

   The squeezed state :math:`\ket{z}`, :math:`z=re^{i\phi}` is a squeezed vacuum state defined by

   .. math::
      \ket{z} = S(z)\ket{0}

.. tip::

   *Implemented in Strawberry Fields as a state preparation operator by* :class:`strawberryfields.ops.Squeezed`

   *Available in Strawberry Fields as a NumPy array by* :func:`strawberryfields.utils.squeezed_state`


A squeezed state is a minimum uncertainty state with unequal quadrature variances, and satisfies the following eigenstate equation:

.. math:: \left(\a\cosh(r)+\ad e^{i\phi}\sinh(r)\right)\ket{z} = 0

In the Fock basis, it has the decomposition

.. math:: |z\rangle = \frac{1}{\sqrt{\cosh(r)}}\sum_{n=0}^\infty
        \frac{\sqrt{(2n)!}}{2^n n!}(-e^{i\phi}\tanh(r))^n|2n\rangle

whilst in the Gaussian formulation, :math:`\bar{\mathbf{r}} = (0,0)`, :math:`\mathbf{V} = \frac{\hbar}{2}R(\phi/2)\begin{bmatrix}e^{-2r} & 0 \\0 & e^{2r} \\\end{bmatrix}R(\phi/2)^T`.

We can use the squeezed vacuum state to approximate the zero position and zero momentum eigenstates;

.. math:: \ket{0}_x \approx S(r)\ket{0}, ~~~~ \ket{0}_p \approx S(-r)\ket{0}

where :math:`z=r` is sufficiently large.

.. _displaced_squeezed_state:

Displaced squeezed state
---------------------------------------------

.. admonition:: Definition
   :class: defn

   The displaced squeezed state :math:`\ket{\alpha, z}`, :math:`\alpha\in\mathbb{C}`, :math:`z=re^{i\phi}` is a displaced and squeezed vacuum state defined by

   .. math::
      \ket{\alpha, z} = D(\alpha)S(z)\ket{0}

.. tip::

   *Implemented in Strawberry Fields as a state preparation operator by* :class:`strawberryfields.ops.DisplacedSqueezed`

   *Available in Strawberry Fields as a NumPy array by* :func:`strawberryfields.utils.displaced_squeezed_state`

In the Fock basis, it has the decomposition

.. math:: |\alpha,z\rangle = e^{-\frac{1}{2}|\alpha|^2-\frac{1}{2}{\alpha^*}^2 e^{i\phi}\tanh{(r)}} \sum_{n=0}^\infty\frac{\left[\frac{1}{2}e^{i\phi}\tanh(r)\right]^{n/2}}{\sqrt{n!\cosh(r)}} H_n\left[ \frac{\alpha\cosh(r)+\alpha^*e^{i\phi}\sinh(r)}{\sqrt{e^{i\phi}\sinh(2r)}} \right]|n\rangle


where :math:`H_n(x)` are the Hermite polynomials defined by :math:`H_n(x)=(-1)^n e^{x^2}\frac{d}{dx}e^{-x^2}`. Alternatively, in the Gaussian formulation, :math:`\bar{\mathbf{r}} = 2 \sqrt{\frac{\hbar}{2}}(\text{Re}(\alpha),\text{Im}(\alpha))` and :math:`\mathbf{V} = R(\phi/2)\begin{bmatrix}e^{-2r} & 0 \\0 & e^{2r} \\\end{bmatrix}R(\phi/2)^T`


We can use the displaced squeezed states to approximate the :math:`x` position and :math:`p` momentum eigenstates;

.. math:: \ket{x}_x \approx D\left(\frac{1}{2}x\right)S(r)\ket{0}, ~~~~ \ket{p}_p \approx D\left(\frac{i}{2}p\right)S(-r)\ket{0}

where :math:`z=r` is sufficiently large.

.. _thermal_state:

Thermal state
---------------------------------------------

.. admonition:: Definition
   :class: defn

   The thermal state is a mixed Gaussian state defined by 

   .. math:: \rho(\nbar) := \sum_{n=0}^\infty\frac{\nbar^n}{(1+\nbar)^{n+1}}\ketbra{n}{n}

   where :math:`\nbar:=\tr{(\rho(\nbar)\hat{n})}` is the mean photon number. In the Gaussian formulation one has :math:`\mathbf{V}=(2 \nbar +1) \frac{\hbar}{2} I` and :math:`\bar{\mathbf{r}}=(0,0)`.

.. tip::

   *Implemented in Strawberry Fields as a state preparation operator by* :class:`strawberryfields.ops.Thermal`

.. _cat_state:

Cat state
-------------

.. warning:: The cat state is **non-Gaussian**, and thus can only be used in the Fock backends, *not* the Gaussian backend.

.. admonition:: Definition
   :class: defn

   The cat state is a non-Gaussian superposition of coherent states

   .. math:: |cat\rangle = \frac{e^{-|\alpha|^2/2}}{\sqrt{2(1+e^{-2|\alpha|^2}\cos(\phi))}}
        \left(|\alpha\rangle +e^{i\phi}|-\alpha\rangle\right)

   with the even cat state given for :math:`\phi=0`, and the odd cat state given for :math:`\phi=\pi`.

.. tip::

   *Implemented in Strawberry Fields as a state preparation operator by* :class:`strawberryfields.ops.Catstate`

   *Implemented in Strawberry Fields as a NumPy array by* :class:`strawberryfields.utils.cat_state`

In the case where :math:`\alpha<1.2`, the cat state can be approximated by the squeezed single photon state :math:`S\ket{1}`.
