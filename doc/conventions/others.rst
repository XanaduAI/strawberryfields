
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

Other
======




.. note:: In Strawberry Fields we use the convention :math:`\hbar=2` by default, but other conventions can also be chosen on engine :func:`initialization <strawberryfields.Engine>`. In this document we keep :math:`\hbar` explicit.

.. _loss:

Loss channel
---------------------------------------------

Loss is implemented by a CPTP map whose Kraus representation is

.. math::
   \mathcal{N}(T)\left\{\ \cdot \ \right\} = \sum_{n=0}^{\infty} E_n(T) \  \cdot \ E_n(T)^\dagger , \quad E_n(T) = \left(\frac{1-T}{T} \right)^{n/2} \frac{\a^n}{\sqrt{n!}} \left(\sqrt{T}\right)^{\ad \a}

.. admonition:: Definition
    :class: defn

    Loss is implemented by coupling mode :math:`\a` to another bosonic mode :math:`\hat{b}` prepared in the vacuum state, by using the following transformation

    .. math::
       \a \to \sqrt{T} \a+\sqrt{1-T} \hat{b}

    and then tracing it out. Here, :math:`T` is the *energy* transmissivity. For :math:`T = 0` the state is mapped to the vacuum state, and for :math:`T=1` one has the identity map.

.. tip::

   *Implemented in Strawberry Fields as a quantum channel by* :class:`strawberryfields.ops.LossChannel`


One useful identity is

.. math::
   \mathcal{N}(T)\left\{\ket{n}\bra{m} \right\}=\sum_{l=0}^{\min(n,m)} \left(\frac{1-T}{T}\right)^l \frac{T^{(n+m)/2}}{l!} \sqrt{\frac{n! m!}{(n-l)!(m-l)!}} \ket{n-l}\bra{m-l}

In particular :math:`\mathcal{N}(T)\left\{\ket{0}\bra{0} \right\} =  \pr{0}`.


.. _thermal_loss:

Thermal loss channel
---------------------------------------------

.. admonition:: Definition
    :class: defn

    Thermal loss is implemented by coupling mode :math:`\a` to another bosonic mode :math:`\hat{b}` prepared in the thermal state :math:`\ket{\bar{n}}`, by using the following transformation

    .. math::
       \a \to \sqrt{T} \a+\sqrt{1-T} \hat{b}

    and then tracing it out. Here, :math:`T` is the *energy* transmissivity. For :math:`T = 0` the state is mapped to the thermal state :math:`\ket{\bar{n}}` with mean photon number :math:`\bar{n}`, and for :math:`T=1` one has the identity map.

.. tip::

   *Implemented in Strawberry Fields as a quantum channel by* :class:`strawberryfields.ops.ThermalLossChannel`


Note that if :math:`\bar{n}=0`, the thermal loss channel is equivalent to the :ref:`loss channel <loss>`.


Commutation relations
---------------------------------------------

A collection of commutation relations between the gates.

.. math::
   B^\dagger(\theta,\phi) D(z) B(\theta,\phi) = D(z \cos \theta) \otimes D(z e^{-i\phi} \sin \theta)


