
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

Gates
==========================

.. note:: In Strawberry Fields we use the convention :math:`\hbar=2` by default, but other conventions can also be chosen on engine :func:`initialization <strawberryfields.Engine>`. In this document we keep :math:`\hbar` explicit.

.. contents:: Contents
   :local:


.. _displacement:

Displacement
---------------------------------------------

.. admonition:: Definition
   :class: defn

   .. math::
      D(\alpha) = \exp( \alpha \ad -\alpha^* \a) = \exp(r (e^{i\phi}\ad -e^{-i\phi}\a)),
      \quad D^\dagger(\alpha) \a D(\alpha)=\a +\alpha\I

   where :math:`\alpha=r e^{i \phi}` with :math:`r \geq 0` and :math:`\phi \in [0,2 \pi)`.

.. tip::

   *Implemented in Strawberry Fields as a quantum gate by* :class:`strawberryfields.ops.Dgate`



We obtain for the position and momentum operators

.. math::
   D^\dagger(\alpha) \x D(\alpha) = \x +\sqrt{2 \hbar } \re(\alpha) \I,\\
   D^\dagger(\alpha) \p D(\alpha) = \p +\sqrt{2 \hbar } \im(\alpha) \I.



.. admonition:: Definition
   :class: defn

   The pure position and momentum displacement operators are defined as

   .. math::
      X(x) &= D\left( x/\sqrt{2 \hbar}\right)  = \exp(-i x \p /\hbar),
      \quad X^\dagger(x) \x X(x) = \x +x\I,\\
      Z(p) &= D\left(i p/\sqrt{2 \hbar}\right) = \exp(i p \x /\hbar ),
      \quad Z^\dagger(p) \p Z(p) = \p +p\I.

.. tip::

   *Implemented in Strawberry Fields as a quantum gate by* :class:`strawberryfields.ops.Xgate`

   *Implemented in Strawberry Fields as a quantum gate by* :class:`strawberryfields.ops.Zgate`

The matrix elements of the displacement operator in the Fock basis were derived by Cahill and Glauber :cite:`cahill1969`:

.. math::
   \bra{m}\hat D(\alpha) \ket{n}  = \sqrt{\frac{n!}{m!}} \alpha^{m-n} e^{-|\alpha|^2/2} L_n^{m-n}\left( |\alpha|^2 \right)


where :math:`L_n^{m}(x)` is a generalized Laguerre polynomial :cite:`dlmf`.

.. _squeezing:

Squeezing
---------------------------------------------

.. admonition:: Definition
   :class: defn

   .. math::
      & S(z) = \exp\left(\frac{1}{2}\left(z^* \a^2-z {\ad}^{2} \right) \right) = \exp\left(\frac{r}{2}\left(e^{-i\phi}\a^2 -e^{i\phi}{\ad}^{2} \right) \right)\\
      & S^\dagger(z) \a S(z) = \a \cosh(r) -\ad e^{i \phi} \sinh r\\
      & S^\dagger(z) \ad S(z) = \ad \cosh(r) -\a e^{-i \phi} \sinh(r)

   where :math:`z=r e^{i \phi}` with :math:`r \geq 0` and :math:`\phi \in [0,2 \pi)`.

.. tip::

   *Implemented in Strawberry Fields as a quantum gate by* :class:`strawberryfields.ops.Sgate`

The squeeze gate affects the position and momentum operators as

.. math:: S^\dagger(z) \x_{\phi} S(z) = e^{-r}\x_{\phi}, ~~~ S^\dagger(z) \p_{\phi} S(z) = e^{r}\p_{\phi}

The Fock basis decomposition of displacement and squeezing operations was analysed by Krall :cite:`kral1990`, and the following quantity was calculated,

.. math::
   f_{n,m}(r,\phi,\beta)&=\bra{n}\exp\left(\frac{r}{2}\left(e^{i \phi} \a^2 -e^{-i \phi} \ad \right) \right) D(\beta) \ket{m} = \bra{n}S(z^*) D(\beta) \ket{m}\\
   &=\sqrt{\frac{n!}{\mu  m!}} e^{\frac{\beta ^2 \nu ^*}{2\mu }-\frac{\left| \beta \right| ^2}{2}}
   \sum_{i=0}^{\min(m,n)}\frac{\binom{m}{i} \left(\frac{1}{\mu  \nu }\right)^{i/2}2^{\frac{i-m}{2}+\frac{i}{2}-\frac{n}{2}} \left(\frac{\nu }{\mu }\right)^{n/2} \left(-\frac{\nu ^*}{\mu }\right)^{\frac{m-i}{2}} H_{n-i}\left(\frac{\beta }{\sqrt{2} \sqrt{\mu  \nu }}\right) H_{m-i}\left(-\frac{\alpha ^*}{\sqrt{2}    \sqrt{-\mu  \nu ^*}}\right)}{(n-i)!}


where :math:`\nu=e^{- i\phi} \sinh(r), \mu=\cosh(r), \alpha=\beta \mu - \beta^* \nu`.

Two important special cases of the last formula are obtained when :math:`r \to 0` and when :math:`\beta \to 0`:

* For :math:`r \to 0` we can take :math:`\nu \to 1, \mu \to r, \alpha \to \beta` and use the fact that for large :math:`x \gg 1` the leading order term of the Hermite polynomials is  :math:`H_n(x) = 2^n x^n +O(x^{n-2})` to obtain

  .. math:: f_{n,m}(0,\phi,\beta) = \bra{n}D(\beta) \ket{m}=\sqrt{\frac{n!}{  m!}} e^{-\frac{\left| \beta \right| ^2}{2}} \sum_{i=0}^{\min(m,n)} \frac{(-1)^{m-i}}{(n-i)!} \binom{m}{i} \beta^{n-i} (\beta^*)^{m-i}

* On the other hand if we let :math:`\beta\to 0` we use the fact that

  .. math::
     H_n(0) =\begin{cases}0,  & \mbox{if }n\mbox{ is odd} \\(-1)^{\tfrac{n}{2}} 2^{\tfrac{n}{2}} (n-1)!! , & \mbox{if }n\mbox{ is even} \end{cases}

  to deduce that :math:`f_{n,m}(r,\phi,0)` is zero if :math:`n` is even and :math:`m` is odd or vice versa.

When writing the Bloch-Messiah reduction :cite:`cariolaro2016`:cite:`cariolaro2016b` of a Gaussian state in the Fock basis one often needs the following matrix element

.. math::
   \bra{k} D(\alpha) R(\theta) S(r) \ket{l}  = e^{i \theta l }  \bra{k} D(\alpha) S(r e^{2i \theta}) \ket{l} = e^{i \theta l} f^*_{l,k}(-r,-2\theta,-\alpha)



..   f_{2n,2m}(r,\phi,\beta) = \bra{2n}S(z^*) \ket{2m}=\sqrt{\frac{(2n)!}{\mu  (2m)!}}    \sum_{i=0}^{\min(m,n)}\frac{\binom{2m}{2i} \left(\frac{1}{\mu  \nu }\right)^{i} \left(\frac{\nu }{\mu }\right)^{n} \left(-\frac{\nu ^*}{\mu }\right)^{m-i} (-1)^{n+m}  (m-i-1)!! (n-i-1)!!}{(n-i)!}

.. _rotation:

Rotation
---------------------------------------------

.. note:: We use the convention that a positive value of :math:`\phi` corresponds to an **anticlockwise** rotation in the phase space.

.. admonition:: Definition
   :class: defn

   We write the phase space rotation operator as

   .. math::
      R(\phi) = \exp\left(i \phi \ad \a\right)=\exp\left(i \frac{\phi}{2} \left(\frac{\x^2+  \p^2}{\hbar}-\I\right)\right), \quad R^\dagger(\phi) \a R(\phi) = \a e^{i \phi}

.. tip::

   *Implemented in Strawberry Fields as a quantum gate by* :class:`strawberryfields.ops.Rgate`


It rotates the position and momentum quadratures to each other:

.. math::
   R^\dagger(\phi)\x R(\phi) = \x \cos \phi -\p \sin \phi,\\
   R^\dagger(\phi)\p R(\phi) = \p \cos \phi +\x \sin \phi.

.. _fourier:

.. admonition:: Definition
   :class: defn

   A special case of the rotation operator is the case :math:`\phi=\pi/2`; this corresponds to the Fourier gate,

   .. math::
      F = R(\pi/2) = e^{i (\pi/2) \ad \a},

.. tip::

   *Implemented in Strawberry Fields as a quantum gate by* :class:`strawberryfields.ops.Fouriergate`

The Fourier gate transforms the quadratures as follows:

.. math::
   & F^\dagger\x F = -\p,\\
   & F^\dagger\p F = \x.


.. _quadratic:

Quadratic phase
---------------------------------------------

.. admonition:: Definition
   :class: defn

   .. math::
      P(s) = \exp\left(i  \frac{s}{2 \hbar} \x^2\right),
      \quad P^\dagger(s) \a P(s) = \a +i\frac{s}{2}(\a +\ad)

.. tip:: *Implemented in Strawberry Fields as a quantum gate by* :class:`strawberryfields.ops.Pgate`

It shears the phase space, preserving position:

.. math::
   P^\dagger(s) \x P(s) &= \x,\\
   P^\dagger(s) \p P(s) &= \p +s\x.

This gate can be decomposed as

.. math::
   P(s) = R(\theta) S(r e^{i \phi})

where :math:`\cosh(r) = \sqrt{1+(\frac{s}{2})^2}, \quad \tan(\theta) = \frac{s}{2}, \quad \phi = -\sign(s)\frac{\pi}{2} -\theta`.

.. _beamsplitter:

Beamsplitter
---------------------------------------------

.. admonition:: Definition
   :class: defn

   For the annihilation and creation operators of two modes, denoted :math:`\a_1` and :math:`\a_2`, the beamsplitter is defined by

   .. math::
      B(\theta,\phi) = \exp\left(\theta (e^{i \phi}\a_1 \ad_2 - e^{-i \phi} \ad_1 \a_2) \right)

.. tip:: *Implemented in Strawberry Fields as a quantum gate by* :class:`strawberryfields.ops.BSgate`

**Action on the creation and annihilation operators**

They will transform the operators according to

.. math::
   B^\dagger(\theta,\phi) \a_1  B(\theta,\phi) &= \a_1\cos \theta -\a_2 e^{-i \phi} \sin \theta  = t \a_1 -r^* \a_2,\\
   B^\dagger(\theta,\phi) \a_2  B(\theta,\phi) &= \a_2\cos \theta + \a_1  e^{i \phi} \sin \theta= t \a_2 +r \a_1.

where :math:`t = \cos \theta` and :math:`r = e^{i\phi} \sin \theta` are the transmittivity and reflectivity amplitudes of the beamsplitter respectively.

Therefore, the beamsplitter transforms two input coherent states to two output coherent states :math:`B(\theta, \phi) \ket{\alpha,\beta} = \ket{\alpha',\beta'}`, where

.. math::
   \alpha' &= \alpha\cos \theta-\beta e^{-i\phi}\sin\theta = t\alpha - r^*\beta\\
   \beta' &= \beta\cos \theta+\alpha e^{i\phi}\sin\theta = t\beta + r\alpha\\


**Action on the quadrature operators**

By substituting in the definition of the creation and annihilation operators in terms of the position and momentum operators, it is possible to derive an expression for how the beamsplitter transforms the quadrature operators:

.. math::
	&\begin{cases}
		B^\dagger(\theta,\phi) \x_1 B(\theta,\phi) = \x_1 \cos(\theta)-\sin(\theta) [\x_2\cos(\phi)+\p_2\sin(\phi)]\\
		B^\dagger(\theta,\phi) \p_1 B(\theta,\phi) = \p_1 \cos(\theta)-\sin(\theta) [\p_2\cos(\phi)-\x_2\sin(\phi)]\\
	\end{cases}\\[12pt]
	&\begin{cases}
		B^\dagger(\theta,\phi) \x_2 B(\theta,\phi) = \x_2 \cos(\theta)+\sin(\theta) [\x_1\cos(\phi)-\p_1\sin(\phi)]\\
		B^\dagger(\theta,\phi) \p_2 B(\theta,\phi) = \p_2 \cos(\theta)+\sin(\theta) [\p_1\cos(\phi)+\x_1\sin(\phi)]
	\end{cases}


**Action on the position and momentum eigenstates**

A 50% or **50-50 beamsplitter** has :math:`\theta=\pi/4` and :math:`\phi=0` or :math:`\phi=\pi`; consequently :math:`|t|^2 = |r|^2 = \frac{1}{2}`, and it acts as follows:

.. math::
	& B(\pi/4,0)\xket{x_1}\xket{x_2} = \xket{\frac{1}{\sqrt{2}}(x_1-x_2)}\xket{\frac{1}{\sqrt{2}}(x_1+x_2)}\\
	& B(\pi/4,0)\ket{p_1}_p\ket{p_2}_p = \xket{\frac{1}{\sqrt{2}}(p_1-p_2)}\xket{\frac{1}{\sqrt{2}}(p_1+p_2)}

and

.. math::
	& B(\pi/4,\pi)\xket{x_1}\xket{x_2} = \xket{\frac{1}{\sqrt{2}}(x_1+x_2)}\xket{\frac{1}{\sqrt{2}}(x_2-x_1)}\\
	& B(\pi/4,\pi)\ket{p_1}_p\ket{p_2}_p = \xket{\frac{1}{\sqrt{2}}(p_1+p_2)}\xket{\frac{1}{\sqrt{2}}(p_2-p_1)}

Alternatively, **symmetric beamsplitter** (one that does not distinguish between :math:`\a_1` and :math:`\a_2`) is obtained by setting :math:`\phi=\pi/2`.

.. _two_mode_squeezing:

Two-mode squeezing
---------------------------------------------

.. admonition:: Definition
   :class: defn

   .. math::
      S_2(z) = \exp\left(z^* \a_1\a_2 -z \ad_1 \ad_2 \right) = \exp\left(r (e^{-i\phi} \a_1\a_2 -e^{i\phi} \ad_1 \ad_2 \right)

   where :math:`z=r e^{i \phi}` with :math:`r \geq 0` and :math:`\phi \in [0,2 \pi)`.

.. tip:: *Implemented in Strawberry Fields as a quantum gate by* :class:`strawberryfields.ops.S2gate`


It can be decomposed into two opposite local squeezers sandwiched between two 50\% beamsplitters :cite:`ebs2002`:

.. math::
   S_2(z) = B^\dagger(\pi/4,0) \: \left[ S(z) \otimes S(-z)\right] \: B(\pi/4,0)


Two-mode squeezing will transform the operators according to

.. math::
   S_2(z)^\dagger \a_1  S_2(z) &= \a_1 \cosh(r)-\ad_2 e^{i \phi} \sinh(r),\\
   S_2(z)^\dagger \a_2  S_2(z) &= \a_2 \cosh(r) -\ad_1 e^{i \phi} \sinh(r),\\

where :math:`z=r e^{i \phi}` with :math:`r \geq 0` and :math:`\phi \in [0,2 \pi)`.

.. _CX:

Controlled-X gate
---------------------------------------------

.. admonition:: Definition
   :class: defn

   The controlled-X gate, also known as the addition gate or the sum gate, is a controlled displacement in position. It is given by

   .. math::
      \text{CX}(s) = \int dx \xket{x}\xbra{x} \otimes D\left(\frac{s x}{\sqrt{2\hbar}}\right) = \exp\left({-i \frac{s}{\hbar} \: \x_1 \otimes \p_2}\right).

.. tip:: *Implemented in Strawberry Fields as a quantum gate by* :class:`strawberryfields.ops.CXgate`

It is called addition because in the position basis
:math:`\text{CX}(s) \xket{x_1, x_2} = \xket{x_1, x_2+s x_1}`.

We can also write the action of the addition gate on the canonical operators:

.. math::
   \text{CX}(s)^\dagger \x_1 \text{CX}(s) &= \x_1\\
   \text{CX}(s)^\dagger \p_1 \text{CX}(s) &= \p_1- s \ \p_2\\
   \text{CX}(s)^\dagger \x_2 \text{CX}(s) &= \x_2+ s \ \x_1\\
   \text{CX}(s)^\dagger \p_2 \text{CX}(s) &= \p_2 \\
   \text{CX}(s)^\dagger \hat{a}_1 \text{CX}(s) &= \a_1+  \frac{s}{2} (\ad_2 -  \a_2)\\
   \text{CX}(s)^\dagger \hat{a}_2 \text{CX}(s) &= \a_2+  \frac{s}{2} (\ad_1 +  \a_1)\\

The addition gate can be decomposed in terms of single mode squeezers and beamsplitter as follows
:math:`\text{CX}(s) = B(\frac{\pi}{2}+\theta,0) \left(S(r,0) \otimes S(-r,0) \right) B(\theta,0)`
where
:math:`\sin(2 \theta) = \frac{-1}{\cosh r}, \ \cos(2 \theta)=-\tanh(r), \ \sinh(r) = -\frac{ s}{2}`

.. _CZ:

Controlled-phase
---------------------------------------------

.. admonition:: Definition
   :class: defn

   .. math::
      \text{CZ}(s) =  \iint dx dy \: e^{i s x_1 x_2/\hbar } \xket{x_1,x_2}\xbra{x_1,x_2} = \exp\left({i s \: \hat{x_1} \otimes \hat{x_2} /\hbar}\right).

.. tip:: *Implemented in Strawberry Fields as a quantum gate by* :class:`strawberryfields.ops.CZgate`


It is related to the addition gate by a phase space rotation in the second mode:
:math:`\text{CZ}(s) = R_{(2)}(\pi/2) \: \text{CX}(s) \: R_{(2)}^\dagger(\pi/2)`.


In the position basis
:math:`\text{CZ}(s) \xket{x_1, x_2} = e^{i  s x_1 x_2/\hbar} \xket{x_1, x_2}`.


We can also write the action of the controlled-phase gate on the canonical operators:

.. math::
   \text{CZ}(s)^\dagger \x_1 \text{CZ}(s) &= \x_1\\
   \text{CZ}(s)^\dagger \p_1 \text{CZ}(s) &= \p_1+ s \ \x_2\\
   \text{CZ}(s)^\dagger \x_2 \text{CZ}(s) &= \x_2\\
   \text{CZ}(s)^\dagger \p_2 \text{CZ}(s) &= \p_2+ s \ \x_1 \\
   \text{CZ}(s)^\dagger \hat{a}_1 \text{CZ}(s) &= \a_1+  i\frac{s}{2} (\ad_2 +  \a_2)\\
   \text{CZ}(s)^\dagger \hat{a}_2 \text{CZ}(s) &= \a_2+  i\frac{s}{2} (\ad_1 +  \a_1)\\

.. _cubic:

Cubic phase
---------------------------------------------

.. warning:: The cubic phase gate is **non-Gaussian**, and thus can only be used in the Fock backends, *not* the Gaussian backend.

.. warning:: The cubic phase gate can suffer heavily from numerical inaccuracies due to finite-dimensional cutoffs in the Fock basis. The gate implementation in Strawberry Fields is unitary, but it does not implement an exact cubic phase gate. The Kerr gate provides an alternative non-Gaussian gate.

.. admonition:: Definition
   :class: defn

   .. math::
      V(\gamma) = \exp\left(i \frac{\gamma}{3 \hbar} \x^3\right),
      \quad V^\dagger(\gamma) \a V(\gamma) = \a +i\frac{\gamma}{2\sqrt{2/\hbar}} (\a +\ad)^2

.. tip:: *Implemented in Strawberry Fields as a quantum gate by* :class:`strawberryfields.ops.Vgate`

It transforms the phase space as follows:

.. math::
   V^\dagger(\gamma) \x V(\gamma) &= \x,\\
   V^\dagger(\gamma) \p V(\gamma) &= \p +\gamma \x^2.

.. _kerr:

Kerr interaction
---------------------------------------------

.. warning:: The Kerr gate is **non-Gaussian**, and thus can only be used in the Fock backends, *not* the Gaussian backend.

.. admonition:: Definition
   :class: defn

   The Kerr interaction is given by the Hamiltonian

   .. math::
      H = (\hat{a}^\dagger\hat{a})^2=\hat{n}^2

   which is non-Gaussian and diagonal in the Fock basis.

.. tip:: *Implemented in Strawberry Fields as a quantum gate by* :class:`strawberryfields.ops.Kgate`

We can therefore define the Kerr gate, with parameter :math:`\kappa` as

.. math::
   K(\kappa) = \exp{(i\kappa\hat{n}^2)}.


.. _cross_kerr:

Cross-Kerr interaction
---------------------------------------------

.. warning:: The cross-Kerr gate is **non-Gaussian**, and thus can only be used in the Fock backends, *not* the Gaussian backend.

.. admonition:: Definition
   :class: defn

   The cross-Kerr interaction is given by the Hamiltonian

   .. math::
      H = \hat{n}_1\hat{n_2}

   which is non-Gaussian and diagonal in the Fock basis.

.. tip:: *Implemented in Strawberry Fields as a quantum gate by* :class:`strawberryfields.ops.CKgate`

We can therefore define the cross-Kerr gate, with parameter :math:`\kappa` as

.. math::
   CK(\kappa) = \exp{(i\kappa\hat{n}_1\hat{n_2})}.
