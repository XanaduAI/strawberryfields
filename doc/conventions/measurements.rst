
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

Measurements
==========================

.. note:: In the Strawberry Fields we use the convention :math:`\hbar=2` by default, but other conventions can also be chosen on engine :func:`initialization <strawberryfields.Engine>`. In this document we keep :math:`\hbar` explicit.

.. contents:: Contents
   :local:

.. _homodyne:

Homodyne measurement
---------------------------------------------

.. admonition:: Definition
   :class: defn

   Homodyne measurement is a Gaussian projective measurement given by projecting the state onto the states 

   .. math:: \ket{x_\phi}\bra{x_\phi},

   defined as eigenstates of the Hermitian operator

   .. math:: \hat{x}_\phi = \cos(\phi) \hat{x} + \sin(\phi)\hat{p}.

.. tip::

   *Implemented in Strawberry Fields as a measurement operator by* :class:`strawberryfields.ops.MeasureHomodyne`

In the Gaussian backend, this is done by projecting onto finitely squeezed states approximating the :math:`x` and :math:`p` eigenstates. Due to the finite squeezing approximation, this results in a measurement variance of :math:`\sigma_H^2`, where :math:`\sigma_H=2\times 10^{-4}`.

In the Fock backends, this is done by using Hermite polynomials to calculate the :math:`\x_\phi` probability distribution over a specific range and number of bins, before taking a random sample.

.. _heterodyne:

Heterodyne measurement
---------------------------------------------

.. warning:: The heterodyne measurement can only be performed in the Gaussian backend.

.. admonition:: Definition
   :class: defn

   Heterodyne measurement is a Gaussian projective measurement given by projecting the state onto the coherent states,

   .. math:: \frac{1}{\pi} \ket{\alpha}\bra{\alpha}


.. tip::

   *Implemented in Strawberry Fields as a measurement operator by* :class:`strawberryfields.ops.MeasureHeterodyne`


.. _photon_counting:

Photon counting measurement
---------------------------------------------

.. warning:: Photon counting is available in the Gaussian backend, but the state of the circuit is not updated after measurement (since it would be non-Gaussian).

.. admonition:: Definition
   :class: defn

   Photon counting is a non-Gaussian projective measurement given by

   .. math:: \ket{n_i}\bra{n_i}

.. tip::

   *Implemented in Strawberry Fields as a measurement operator by* :class:`strawberryfields.ops.MeasureFock`