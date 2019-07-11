.. _conventions:

========================
Conventions and formulas
========================

.. sectionauthor:: Nicolás Quesada <nicolas@xanadu.ai>, Ville Bergholm, Josh Izaac

|

    "The nice thing about standards is that you have so many to choose from." - Tanenbaum :cite:`tanenbaum2002computer`

In this section, we provide the definitions of various quantum operations used by Strawberry Fields, as well as introduce specific conventions chosen. We'll also provide some more technical details relating to the various operations.

.. raw:: html

   <style>
      div.topic.contents > ul {
         max-height: 600px;
      }
   </style>


.. rst-class:: contents local topic

.. toctree::
   :maxdepth: 2

   conventions/operators
   conventions/gates
   conventions/states
   conventions/measurements
   conventions/decompositions
   conventions/others


.. note:: In Strawberry Fields we use the convention :math:`\hbar=2` by default, but other
   conventions can also be chosen by setting the global variable :py:data:`strawberryfields.hbar`
   at the beginning of a session.
   In this document we keep :math:`\hbar` explicit.


.. note::

   More information about the definitions included on this page are available in :cite:`barnett2002` and :cite:`kok2010`.

   The Kraus representation of the loss channel is found in :cite:`albert2017` Eq. 1.4, which is related to the parametrization used here by taking :math:`1-\gamma = T`.

   The explicit expression for the harmonic oscillator wave functions can be found in :cite:`sakurai1994` Eq. A.4.3 of Appendix A.








