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

.. _opcon:

Operators
===================

.. note:: In the Strawberry Fields we use the convention :math:`\hbar=2` by default, but other conventions can also be chosen on engine :func:`initialization <strawberryfields.Engine>`. In this document we keep :math:`\hbar` explicit.


.. contents:: Contents
   :local:

Annihilation and creation operators
---------------------------------------------

As noted in the introduction, some of the basic operators used in CV quantum computation are the **bosonic anhilation and creation operators** :math:`\a` and :math:`\ad`. The operators corresponding to two seperate modes, :math:`\a_1` and :math:`\a_2` respectively,  satisfy the following commutation relations:

.. math::
   &[\a_1,\ad_1] = [\a_2,\ad_2] = \I,\\
   &[\a_1,\a_1]=[\a_1,\ad_2]=[\a_1,\a_2]=[\a_2,\a_2]=0.

Quadrature operators
---------------------------------------------

The dimensionless **position and momentum quadrature operators** :math:`\x` and :math:`\p` are defined by 

.. math::
   \x = \sqrt{\frac{\hbar}{2}}(\a+\ad),~~~ \p = -i \sqrt{\frac{\hbar}{2}}(\a-\ad).

They fulfill the commutation relation

.. math::
   [\x, \p] = i \hbar,

and satisfy the eigenvector equations

.. math:: 
   \x\xket{x} = x \xket{x} ~~~~  \p\ket{p}_p = p\ket{p}_p

Here, :math:`\xket{x}` and :math:`\ket{p}_p` are the eigenstates of :math:`\x` and :math:`\p` with eigenvalues :math:`x` and :math:`p` respectively. The position and momentum operators generate shifts in each others' eigenstates:

.. math::
   e^{-i r \p/\hbar} \xket{x} = \xket{x+r},\\
   e^{i s \x/\hbar} \ket{p}_p = \ket{p+s}_p.

In the vacuum state, the variances of position and momentum are given by

.. math::
   \bra{0}\x^2\ket{0} = \bra{0}\p^2\ket{0} = \frac{\hbar}{2}.

Note that we can also write the annihilation and creation operators in terms of the quadrature operators:

.. math::
   \a := \sqrt{\frac{1}{2 \hbar}} (\x +i\p), ~~~~ \ad := \sqrt{\frac{1}{2 \hbar}} (\x -i\p).

Number operator
---------------------------------------------

The number operator is :math:`\hat{n} := \ad \a`, and satisfies the eigenvector equation

.. math:: \hat{n}\ket{n} = n\ket{n}

where :math:`\ket{n}` are the **Fock states** with eigenvalue :math:`n`. Furthermore, note that 

.. math:: \ad\ket{n} = \sqrt{n+1}\ket{n+1}~~~\text{and}~~~~\a\ket{n}=\sqrt{n}\ket{n-1}.

Using the position eigenstates :math:`\xket{x}`, we may represent the Fock states :math:`\ket{n}` as wavefunctions in the position representation:

.. math::
      \psi_n(x) = \braket{x_x|n} = \frac{1}{\sqrt{2^n\,n!}} \: \left(\frac{1}{\pi \hbar}\right)^{1/4} e^{- \frac{1}{2 \hbar}x^2} \: H_n\left(x/\sqrt{\hbar}  \right), \qquad n = 0,1,2,\ldots,

where :math:`H_n(x)` are the physicist's `Hermite polynomials <https://en.wikipedia.org/wiki/Hermite_polynomials>`_ :cite:`dlmf`.
