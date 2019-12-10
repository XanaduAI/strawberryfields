Machine learning
================

Quantum is a natural fit for machine learning, providing a high-dimensional embedding of data and
a naturally probabilistic platform.

Graph similarity
----------------

GBS can be used to construct a similarity measure between graphs, known as a graph kernel
:cite:`schuld2019quantum`. Kernels can be applied to graph-based data for machine learning tasks
such as classification using a support vector machine.

.. customgalleryitem::
    :tooltip: Graphs similarity
    :description: :doc:`../tutorials_apps/run_tutorial_similarity`
    :figure: ../_static/similarity_svm.png


.. raw:: html

    <div style='clear:both'></div>

Point processes
---------------

Point sampling can be a useful tool in machine learning, providing a source of randomness with
preference towards both diversity :cite:`kulesza2012determinantal` and similarity in data.

.. customgalleryitem::
    :tooltip: Point processes
    :description: :doc:`../tutorials_apps/run_tutorial_points`
    :figure: ../_static/points.png
