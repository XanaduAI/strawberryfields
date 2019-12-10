
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

Decompositions
==============

.. note:: In Strawberry Fields we use the convention :math:`\hbar=2` by default, but other conventions can also be chosen on engine :func:`initialization <strawberryfields.Engine>`. In this document we keep :math:`\hbar` explicit.

.. contents:: Contents
   :local:
   :depth: 1

This page contains some common continuous-variable (CV) decompositions. However, before beginning, it is useful to state some important properties of Gaussian unitaries and the symplectic group.

Gaussian unitaries
-------------------

Gaussian states are those with positive Wigner distributions, and are fully characterised by their first and second moments; the means vector :math:`\hat{\mathbf{r}}=(\x_1,\dots,\x_N,\p_1,\dots,\p_N)`, and the covariance matrix :math:`V_{ij}=\frac{1}{2}\langle\Delta r_i\Delta r_j + \Delta r_i\Delta r_j\rangle` respectively (for more details, see :ref:`gaussian_basis`).

Gaussian unitaries, it then follows, are quantum operations that retain the Gaussian character of the state; i.e., the set of unitary transformations :math:`U`  that transform Gaussian states into Gaussian states:

.. math:: \rho\rightarrow U\rho U^\dagger

In the Hilbert space, these are represented by unitaries of the form :math:`U=e^{-iH/2}` with Hamiltonians :math:`H` that are at most second-order polynomials in the quadrature operators :math:`\x` and :math:`\p`. In order to preserve the commutation relations :math:`[r_i,r_j]=i\hbar\Omega`, where

.. math:: \Omega = \begin{bmatrix}0 & \I_N \\-\I_N & 0 \end{bmatrix}

is the `symplectic matrix <https://en.wikipedia.org/wiki/Symplectic_matrix>`_, the set of Gaussian unitary transformations :math:`U` in the Hilbert space is represented by *real symplectic transformations* on the first and second moments of the Gaussian state:

.. math::  \rho\rightarrow U\rho U^\dagger ~~~\Leftrightarrow ~~~ \begin{cases}\mathbf{r}\rightarrow S\mathbf{r}+\mathbf{d}\\ V\rightarrow S V S^T\end{cases}

Here, :math:`\mathbf{d}\in\mathbb{R}^{2N}`, and :math:`S\in\mathbb{R}^{2N\times 2N}` is a symplectic matrix satisfying the condition :math:`S\Omega S^T=S`.


.. _williamson:

Williamson decomposition
-------------------------

.. admonition:: Definition
    :class: defn

    For every positive definite real matrix :math:`V\in\mathbb{R}^{2N\times 2N}`, there exists a symplectic matrix :math:`S` and diagonal matrix :math:`D` such that

    .. math:: V = S D S^T

    where :math:`D=\text{diag}(\nu_1,\dots,\nu_N,\nu_1,\dots,\nu_N)`, and :math:`\{\nu_i\}` are the eigenvalues of :math:`|i\Omega V|`, where :math:`||` represents the element-wise absolute value.

.. tip::

   *Implemented in Strawberry Fields as a state preparation decomposition by* :class:`strawberryfields.ops.Gaussian`


The Williamson decomposition allows an arbitrary Gaussian covariance matrix to be decomposed into a symplectic transformation acting on the state described by the diagonal matrix :math:`D`.

The matrix :math:`D` can always be decomposed further into a set of thermal states with mean photon number given by

.. math:: \bar{n}_i = \frac{1}{\hbar}\nu_i - \frac{1}{2}, ~~i=1,\dots,N

Pure states
^^^^^^^^^^^

In the case where :math:`V` represents a pure state (:math:`|V|-(\hbar/2)^{2N}=0`), the Williamson decomposition outputs :math:`D=\frac{1}{2}\hbar I_{2N}`; that is, a symplectic transformation :math:`S` acting on the vacuum. It follows that the original covariance matrix can therefore be recovered simply via :math:`V=\frac{\hbar}{2}SS^T`.

.. note:: :math:`V` must be a valid quantum state satisfying the uncertainty principle: :math:`V+\frac{1}{2}i\hbar\Omega\geq 0`. If this is not the case, the Williamson decomposition will return non-physical thermal states with :math:`\bar{n}_i<0`.


.. _bloch_messiah:

Bloch-Messiah (or Euler) decomposition
--------------------------------------

.. admonition:: Definition
    :class: defn

    For every symplectic matrix :math:`S\in\mathbb{R}^{2N\times 2N}`, there exists orthogonal symplectic matrices :math:`O_1` and :math:`O_2`, and diagonal matrix :math:`Z`, such that

    .. math:: S = O_1 Z O_2

    where :math:`Z=\text{diag}(e^{-r_1},\dots,e^{-r_N},e^{r_1},\dots,e^{r_N})` represents a set of one mode squeezing operations with parameters :math:`(r_1,\dots,r_N)`.

.. tip::

   *Implemented in Strawberry Fields as a gate decomposition by* :class:`strawberryfields.ops.GaussianTransform`

Gaussian symplectic transforms can be grouped into two main types; passive transformations (those which preserve photon number) and active transformations (those which do not). Compared to active transformation, passive transformations have an additional constraint - they must preserve the trace of the covariance matrix, :math:`\text{Tr}(SVS^T)=\text{Tr}(V)`; this only occurs when the symplectic matrix :math:`S` is also orthogonal (:math:`SS^T=\I`).

The Bloch-Messiah decomposition therefore allows any active symplectic transformation to be decomposed into two passive Gaussian transformations :math:`O_1` and :math:`O_2`, sandwiching a set of one-mode squeezers, an active transformation.

Acting on the vacuum
^^^^^^^^^^^^^^^^^^^^^^

In the case where the symplectic matrix :math:`S` is applied to a vacuum state :math:`V=\frac{\hbar}{2}\I`, the action of :math:`O_2` cancels out due to its orthogonality:

.. math:: SVS^T = (O_1 Z O_2)\left(\frac{\hbar}{2}\I\right)(O_1 Z O_2)^T = \frac{\hbar}{2} O_1 Z O_2 O_2^T Z O_1^T = \frac{\hbar}{2}O_1 Z^2 O_1^T

As such, a symplectic transformation acting on the vacuum is sufficiently characterised by single mode squeezers followed by a passive Gaussian transformation (:math:`S = O_1 Z`).


.. _rectangular:

Rectangular decomposition
-------------------------

The rectangular decomposition allows any passive Gaussian transformation to be decomposed into a series of beamsplitters and rotation gates.

.. admonition:: Definition
    :class: defn

    For every real orthogonal symplectic matrix

    .. math:: O=\begin{bmatrix}X&-Y\\ Y&X\end{bmatrix}\in\mathbb{R}^{2N\times 2N},

    the corresponding unitary matrix :math:`U=X+iY\in\mathbb{C}^{N\times N}` representing a multiport interferometer can be decomposed into a set of :math:`N(N-1)/2` beamsplitters and single mode rotations with circuit depth of :math:`N`.

    For more details, see :cite:`clements2016`.

.. tip::

   *Implemented in Strawberry Fields as a gate decomposition by* :class:`strawberryfields.ops.Interferometer`

.. note::

    The rectangular decomposition as formulated by Clements :cite:`clements2016` uses a different beamsplitter convention to Strawberry Fields:

    .. math:: BS_{clements}(\theta, \phi) = BS(\theta, 0) R(\phi)
