GBS datasets
============

.. role:: html(raw)
   :format: html

.. currentmodule:: strawberryfields.apps.data

*Technical details are available in the API documentation:* :doc:`/code/api/strawberryfields.apps.data`

Strawberry Fields contains datasets of pre-generated samples from GBS for encoded problems,
including graphs for :doc:`graph optimization and machine learning problems <applications>`,
and molecules for calculating :doc:`vibronic spectra <apps/run_tutorial_vibronic>`.

Graphs
------

For :ref:`dense subgraph <apps-subgraph-tutorial>` and
:ref:`maximum clique <apps-clique-tutorial>` identification, we provide:

.. |planted| image:: /_static/graphs/planted.png
    :align: middle
    :width: 250px
    :target: javascript:void(0);

.. |tace_as| image:: /_static/graphs/TACE-AS.png
    :align: middle
    :width: 250px
    :target: javascript:void(0);

+---------------+---------------+
|   |planted|   |    |tace_as|  |
|               |               |
|  **Planted**  |   **TACE-AS** |
+---------------+---------------+

:html:`<div class="summary-table">`

.. autosummary::
    Planted
    TaceAs
    PHat

For :ref:`graph similarity <apps-sim-tutorial>`, we provide:

.. |mutag_0| image:: /_static/graphs/MUTAG_0.png
    :align: middle
    :width: 250px
    :target: javascript:void(0);

.. |mutag_1| image:: /_static/graphs/MUTAG_1.png
    :align: middle
    :width: 250px
    :target: javascript:void(0);

.. |mutag_2| image:: /_static/graphs/MUTAG_2.png
    :align: middle
    :width: 250px
    :target: javascript:void(0);

.. |mutag_3| image:: /_static/graphs/MUTAG_3.png
    :align: middle
    :width: 250px
    :target: javascript:void(0);

+-------------+-------------+
|  |mutag_0|  |  |mutag_1|  |
|             |             |
| **MUTAG_0** | **MUTAG_1** |
+-------------+-------------+
|  |mutag_2|  |  |mutag_3|  |
|             |             |
| **MUTAG_2** | **MUTAG_3** |
+-------------+-------------+

:html:`<div class="summary-table">`

.. autosummary::
    Mutag0
    Mutag1
    Mutag2
    Mutag3

Additionally, pre-calculated feature vectors of the following graph datasets are provided:

.. autosummary::
    MUTAG
    QM9Exact
    QM9MC

Molecules
---------

Using the :mod:`~.qchem.vibronic` module and :func:`~.qchem.vibronic.sample` function, GBS data has
been generated for formic acid at zero temperature. The GBS samples can be used to recover the
:ref:`vibronic spectrum <apps-vibronic-tutorial>` of the molecule.

.. autosummary::
    Formic
    Water
    Pyrrole

Dataset
-------

The :class:`~.SampleDataset` class provides the base functionality from which all datasets inherit.

Each dataset contains a variety of metadata relevant to the sampling:

- ``n_mean``: theoretical mean number of photons in the GBS device

-  ``threshold``: flag to indicate whether samples are generated with threshold detection or
   with photon-number-resolving detectors

- ``n_samples``: total number of samples in the dataset

- ``modes``: number of modes in the GBS device or, equivalently, number of nodes in the graph

- ``data``: the raw data accessible as a SciPy `csr sparse array
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`__

Graph and molecule datasets also contain some specific data, such as the graph adjacency matrix
or the input molecular information.

Note that datasets are simulated without photon loss.

Loading data
------------

We use the :class:`Planted` class as an example to show how to interact with the datasets. Datasets
can be loaded by running:

>>> data = Planted()

Simply use indexing and slicing to access samples from the dataset:

>>> sample_3 = data[3]
>>> samples = data[:10]

Datasets also contain metadata relevant to the GBS setup:

>>> data.n_mean
8

>>> len(data)
50000

The number of photons or clicks in each sample is available using the
:meth:`~.SampleDataset.counts` method:

>>> data.counts()
[2, 0, 8, 11, ... , 6]

For example, we see that the ``data[3]`` sample has 11 clicks.
