Sampling
========

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

Quantum computers are probabilistic and a common task is to take samples. Gaussian boson sampling
(GBS) is a photonic algorithm that can be realized using near-term devices, see the
:doc:`/introduction/introduction` for further details. Samples from GBS can used for
:ref:`graph optimization <graphs-intro>`, :ref:`machine learning <ml-intro>` and :ref:`chemistry
calculations <chemistry-intro>`. Strawberry Fields provides high-level tools for embedding
problems into GBS without needing to worry about designing a quantum circuit.

.. customgalleryitem::
    :tooltip: GBS Sampling
    :description: :doc:`Near-term quantum sampling <../tutorials_apps/run_tutorial_sample>`
    :figure: ../_static/thumbs/photon.png

.. customgalleryitem::
    :tooltip: Datasets
    :description: :doc:`Pre-generated datasets <data>`
    :figure: ../_static/graphs/MUTAG_3.png

.. raw:: html

    <div style='clear:both'></div>

.. toctree::
   :maxdepth: 2
   :hidden:

   data
