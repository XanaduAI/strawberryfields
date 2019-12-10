.. role:: html(raw)
   :format: html

.. _gaussian_algorithm:

Gaussian boson sampling
========================

    "If you need to wait exponential time for [your single photon sources to emit simultaneously], then there would seem to be no advantage over classical computation.  This is the reason why so far, boson sampling has only been demonstrated with 3-4 photons. When faced with these problems, until recently, all we could do was shrug our shoulders." - `Scott Aaronson <https://www.scottaaronson.com/blog/?p=1579>`_

While boson sampling allows the experimental implementation of a sampling problem that is countably hard classically, one of the main issues it has in experimental setups is one of **scalability**, due to its dependence on an array of simultaneously emitting single photon sources. Currently, most physical implementations of boson sampling make use of a process known as
`spontaneous parametric down-conversion <http://en.wikipedia.org/wiki/Spontaneous_parametric_down-conversion>`_
to generate the single-photon source inputs. However, this method is non-deterministic ---
as the number of modes in the apparatus increases, the average time required until every photon source emits a simultaneous photon increases exponentially.

In order to simulate a deterministic single-photon source array, several variations on boson sampling have been proposed; the most well-known being scattershot boson sampling :cite:`lund2014`. However, a recent boson sampling variation by Hamilton et al. :cite:`hamilton2017` mitigates the need for single photon Fock states altogether, by showing that incident Gaussian states
--- in this case, single mode squeezed states --- can produce problems in the same computational
complexity class as boson sampling. Even more significantly, this negates the scalability problem with single photon sources, as single mode squeezed states can be deterministically generated simultaneously.

The Gaussian boson sampling scheme remains, on initial observation, quite similar to that of boson sampling:

* :math:`N` single mode squeezed states :math:`\ket{z}`, with squeezing parameter :math:`z=re^{i\phi}`, enter an :math:`N` mode linear interferometer described by unitary :math:`U` simultaneously.

* Each output mode of the interferometer (denoted by state :math:`\ket{\psi'}`) is then measured in the Fock basis, :math:`\bigotimes_i n_i\ket{n_i}\bra{n_i}`.

Without loss of generality, we can absorb the squeezing phase parameter :math:`\phi` into the interferometer, and set :math:`\phi=0` for convenience.

Using phase space methods, Hamilton et al. :cite:`hamilton2017` showed that the probability of measuring a Fock state containing only 0 or 1 photons per mode is given by

.. math:: \left|\left\langle{n_1,n_2,\dots,n_N}\middle|{\psi'}\right\rangle\right|^2 = \frac{\left|\text{Haf}[(U(\bigoplus_i\tanh(r_i))U^T)]_{st}\right|^2}{\prod_{i=1}^N \cosh(r_i)}

i.e., the sampled single photon probability distribution is proportional to the **hafnian** of a submatrix of :math:`U(\bigoplus_i\tanh(r_i))U^T`, dependent upon the output covariance matrix.

.. note::

    The hafnian of a matrix is defined by

    .. math:: \text{Haf}(A) = \frac{1}{n!2^n}\sum_{\sigma=S_{2N}}\prod_{i=1}^N A_{\sigma(2i-1)\sigma(2i)}

    where :math:`S_{2N}` is the set of all permutations of :math:`2N` elements. In graph theory, the hafnian calculates the number of perfect `matchings <https://en.wikipedia.org/wiki/Matching_(graph_theory)>`_ in an **arbitrary graph** with adjacency matrix :math:`A`.

    Compare this to the permanent, which calculates the number of perfect matchings on a *bipartite* graph - the hafnian turns out to be a generalization of the permanent, with the relationship

    .. math:: \text{Per(A)} = \text{Haf}\left(\left[\begin{matrix}
            0&A\\
            A^T&0
        \end{matrix}\right]\right)

As any algorithm that could calculate (or even approximate) the hafnian could also calculate the permanent - a #P-hard problem - it follows that calculating or approximating the hafnian must also be a classically hard problem.


CV implementation
------------------------------------

As with the boson sampling problem, the multimode linear interferometer can be decomposed into two-mode beamsplitters (:class:`~.BSgate`) and single-mode phase shifters (:class:`~.Rgate`) :cite:`reck1994`, allowing for a straightforward translation into a CV quantum circuit.

For example, in the case of a 4 mode interferometer, with arbitrary :math:`4\times 4` unitary :math:`U`, the CV quantum circuit for Gaussian boson sampling is given by


:html:`<br>`

.. image:: ../_static/gaussian_boson_sampling.svg
    :align: center
    :width: 70%
    :target: javascript:void(0);

:html:`<br>`

In the above, the single mode squeeze states all apply identical squeezing :math:`z=r`, the parameters of the beamsplitters and the rotation gates determine the unitary :math:`U`, and finally the detectors perform Fock state measurements on the output modes. As with boson sampling, for :math:`N` input modes, we must have a minimum of :math:`N+1` columns in the beamsplitter array :cite:`clements2016`.


Blackbird code
---------------

The boson sampling circuit displayed above, with randomly chosen rotation angles and beamsplitter parameters, can be implemented using the Blackbird quantum circuit language:

.. literalinclude:: ../../examples/gaussian_boson_sampling.py
   :language: python
   :linenos:
   :dedent: 4
   :tab-width: 4
   :start-after: with gbs.context as q:
   :end-before: # end circuit

If we wish to simulate Fock measurements, we can additionally include

.. code-block:: python

    MeasureFock() | q

after the beamsplitter array. After constructing the circuit and running the engine, the values of the Fock state measurements will be available within the :attr:`samples` attribute of the :class:`~.Result` object returned by the engine.
In order to sample from this distribution :math:`N` times, a :code:`shots` parameter can be included in :code:`run_options` during engine execution, i.e., :func:`eng.run(gbs, run_options={"shots": N})` (only supported for Gaussian backend).

Alternatively, you may omit the measurements, and extract the resulting Fock state probabilities directly via the state methods :meth:`~.BaseFockState.all_fock_probs` (supported by Fock backends) or :meth:`~.BaseState.fock_prob` (supported by all backends).

.. note::
  A fully functional Strawberry Fields simulation containing the above Blackbird code is included at :download:`examples/gaussian_boson_sampling.py <../../examples/gaussian_boson_sampling.py>`.

  For more details on running the above Blackbird code in Strawberry Fields, including calculations of how to determine the output Fock state probabilities using the matrix permanent and comparisons to the returned state, refer to the in-depth :ref:`Gaussian boson sampling tutorial <gaussian_boson_tutorial>`.
