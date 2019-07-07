.. role:: html(raw)
   :format: html

.. _iqp:

Instantaneous quantum polynomial
================================

	"... although they are not the preferred problem of computer scientists, integrals of functions over many  variables  have  been  extensively  studied  with no known efficient algorithms known for arbitrary integrals." - Arrazola et al. :cite:`arrazola2017quantum` on the applications of IQP.

Like boson sampling and Gaussian boson sampling before it, the instantaneous quantum polynomial (IQP) protocol is a restricted, non-universal model of quantum computation, designed to implement a computation that is classically intractable and verifiable by a remote adjudicator.

..
  First introduced by :cite:`shepherd2009iqp` in the discrete variable qubit model, the protocol is devised with a two-party classical communication channel in mind; here, Alice designs a classically intractable problem with a 'hidden variable' she can use to verify the result. Bob then runs the IQP scheme with Alice's input; the scheme, due to its use of a non-universal gate set, is designed to scale in polynomial time. Finally, Alice verifies that Bob has achieved quantum supremacy, by considering the time taken to receive the result and using her hidden knowledge of the problem set.

First introduced by :cite:`shepherd2009iqp` in the discrete variable qubit model, the scheme, due to its use of a non-universal gate set, is designed to scale in polynomial time. In particular, IQP circuits are defined as follows:

1. there are :math:`N` inputs in state :math:`\ket{0}` acted on by Hadamard gates;


2. the resulting computation is diagonal in the computational basis by randomly selecting from the set :math:`U=\{R(\pi/4),\sqrt{CZ}\}`;


3. The output qubits are measured in the Pauli-X basis.

.. note:: Since all gates are diagonal in the computational basis, they all commute - therefore they can be applied in *any* temporal order. This is the origin of the term 'instantaneous' in IQP.

Efficient classically sampling the resulting probability distribution :math:`H^{\otimes N}UH^{\otimes N}\ket{0}^{\otimes N}` - even approximately :cite:`bremner2016iqp` or in the presence of noise :cite:`bremner2017achieving` - has been shown to be #P-hard, and would result in the collapse of the polynomial hierarchy to the third level :cite:`lund2017iqp,bremner2010iqp`.

Unlike boson sampling and Gaussian boson sampling, however, the IQP protocol was not constructed with quantum linear optics in mind. Nevertheless, the IQP model was recently extended to the CV formulation of quantum computation :cite:`douce2017iqp`, taking advantage of the ability to efficiently prepare large resource states, and the higher efficiencies afforded by homodyne detection. Furthermore, the computational complexity results of the discrete-variable case apply equally to the so-called CV-IQP model, assuming a specific input squeezing parameter dependent on the circuit size.

Moreover, the resulting probability distributions have been shown to be given by integrals of oscillating functions in large dimensions, which are belived to be intractable to compute by classical methods.  This leads to the result that, even in the case of finite squeezing effects and reduced measurement precision, approximate sampling from the output CV-IQP model remains classically intractable :cite:`douce2017iqp,arrazola2017quantum`.


CV implementation
----------------------------------

The CV-IQP model is defined as follows:

1. inputs are squeezed momentum states :math:`\ket{z}`, where :math:`z=r\in\mathbb{R}` and :math:`r<0` - these are implemented using :class:`~.Sgate`;


2. the unitary transformation is diagonal in the :math:`\hat{x}` quadrature basis, by randomly selecting from the set of gates :math:`U=\{Z(p),CZ(s),V(\gamma)\}` (available respectively as :class:`~.Zgate`, :class:`~.CZgate`, and :class:`~.Vgate`);

3. and finally, homodyne measurements are performed on all modes in the :math:`\hat{p}` quadrature.

A circuit diagram of a 4 mode IQP circuit is given below.

.. image:: ../_static/IQP.svg
    :align: center
    :width: 50%
    :target: javascript:void(0);


Blackbird code
---------------

The 4 mode IQP circuit displayed above, with gate parameters chosen randomly, can be implemented using the Blackbird quantum circuit language:

.. literalinclude:: ../../examples/IQP.py
   :language: python
   :linenos:
   :dedent: 4
   :tab-width: 4
   :start-after: with iqp_circuit.context as q:
   :end-before: # end circuit


In the above circuit, the homodyne measurements have been excluded to allow for analysis of the full quantum state; however these can be included using :class:`~.MeasureHomodyne`.

.. note::
  A fully functional Strawberry Fields simulation containing the above Blackbird code is included at :download:`examples/IQP.py <../../examples/IQP.py>`.
