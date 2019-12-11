Machine learning
================

.. raw:: html

	<style>
		.sphx-glr-thumbcontainer {
			float: center;
			margin-top: 20px;
			margin-right: 50px;
            margin-bottom: 20px;
		}
        #right-column.card {
            box-shadow: none!important;
        }
        #right-column.card:hover {
            box-shadow: none!important;
        }
	</style>

Combining quantum devices with machine learning can unlock additional performance. This can be in
the form of **shorter training times** or **increased predictive power**. Quantum machine
learning is an exciting and rapidly-growing field with multiple approaches to integrating quantum
systems into the existing toolchain.

One hybrid approach is to treat quantum circuits as an analogue of a neural network, providing a
machine learning model by controlling parameters of fixed gates in the circuit. Design,
simulation, and hardware execution of these **variational circuits** is handled by Xanadu's
`PennyLane package <https://pennylane.ai/>`__.

Strawberry Fields focuses on providing the functionality for using near-term quantum photonic
devices for machine learning. Take a look below to see what you can do!

.. customgalleryitem::
    :tooltip: Graphs similarity
    :description: :doc:`Graph similarity and classification <../tutorials_apps/run_tutorial_similarity>`
    :figure: ../_static/similarity_svm.png

.. customgalleryitem::
    :tooltip: Point processes
    :description: :doc:`Point process sampling <../tutorials_apps/run_tutorial_points>`
    :figure: ../_static/points.png

.. raw:: html

    <div style='clear:both'></div>
