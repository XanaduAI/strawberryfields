.. _gallery:

Gallery and links
#########################

Notebook gallery
----------------

Check out some cool applications of Strawberry Fields in the gallery below. Notebooks in the gallery have been contributed by the awesome Strawberry Fields community, and curated by the Strawberry Fields team 🍓.

.. Add the location of your Jupyter notebook below!

.. toctree::
   :maxdepth: 1
   :hidden:
   :titlesonly:

   minimizing_correlations/minimizing_correlations.ipynb
   state_learner/StateLearning.ipynb
   gate_synthesis/GateSynthesis.ipynb
   fibonacci_classifier/fibonacci_classifier.ipynb


.. Copy the template below in order to create a link to your notebook, and a thumbnail.

.. _Minimizing correlations: minimizing_correlations/minimizing_correlations.html
.. |mc| image:: minimizing_correlations/thumbnail.gif
   :width: 260px
   :align: middle
   :target: minimizing_correlations/minimizing_correlations.html


.. _Fibonacci classifier: fibonacci_classifier/fibonacci_classifier.html
.. |fc| image:: fibonacci_classifier/thumbnail.png
   :width: 260px
   :align: middle
   :target: fibonacci_classifier/fibonacci_classifier.html


.. _Quantum state learning: state_learner/StateLearning.html
.. |sl| image:: state_learner/thumbnail.gif
   :width: 260px
   :align: middle
   :target: state_learner/StateLearning.html


.. _Gate synthesis: gate_synthesis/GateSynthesis.html
.. |gs| image:: gate_synthesis/thumbnail.gif
   :width: 260px
   :align: middle
   :target: gate_synthesis/GateSynthesis.html


.. Add your thumbnail to the table in the Gallery!

.. rst-class:: gallery-table

+----------------------------+------------------------------+------------------------------+
| |mc|                       | |sl|                         |   |gs|                       |
|                            |                              |                              |
| `Minimizing correlations`_ | `Quantum state learning`_    |   `Gate synthesis`_          |
+----------------------------+------------------------------+------------------------------+

Applications and plugins
------------------------

The following list of applications and plugins build on the Strawberry Fields framework; extending Strawberry Fields to allow it to tackle and solve new problems, or by providing an interface between other software frameworks.

* `Quantum Machine Learning Toolbox <https://github.com/XanaduAI/QMLT>`_

  The Quantum Machine Learning Toolbox (QMLT) is a Strawberry Fields application that simplifies the optimization of variational quantum circuits. It provides helper functions commonly used in machine learning and optimization (such as for regularization), and includes a numerical learner, that can be used to optimize with the Gaussian and Fock backends.

* `SFOpenBoson <https://github.com/XanaduAI/SFOpenBoson>`_

  A plugin for Strawberry Fields and OpenFermion, SFOpenBoson integrates the photonic quantum circuit simulation capability of Strawberry Fields with the OpenFermion quantum chemistry package. This allows bosonic Hamiltonians defined in OpenFermion to be decomposed and simulated directly in Strawberry Fields.


External resources
------------------

In addition to the notebooks in the gallery, below are some external web resources that use or highlight Strawberry Fields.

.. Add your external blog post/application/GitHub page below!

* `Verifying continuous-variable Bell correlations <https://peterwittek.com/verifying-cv-bell-correlations.html>`_ - Peter Wittek

  *Explore how Strawberry Fields can be used to understand core concepts in quantum physics, such as the violation of the Bell inequalities.*


* `Quantum state learning and gate synthesis <https://github.com/XanaduAI/quantum-learning>`_ - Xanadu

  *A collection of scripts to automate the process of quantum state learning and gate synthesis using Strawberry Fields, based on the paper "Machine learning method for state preparation and gate synthesis on photonic quantum computers" (arXiv:1807.10781). Also included are some useful Python functions for generating well-known CV states and gates.*


* `Getting Started with Quantum Programming <https://hackernoon.com/an-interactive-tutorial-on-quantum-programming-327da388f859>`_ - Tanisha Bassan

  *The timeless game of battleships has been updated for the 21st century; Quantum Battleships, powered by Strawberry Fields.*


* `The World of Photonic Quantum Computing <https://medium.com/@briannagopaul/the-world-of-photonic-quantum-computing-4787a2b12649>`_ - Brianna Gopaul

  *See how Strawberry Fields can be used to visualize how the elementary CV gate set transforms the vacuum state.*


Research papers
----------------

Finally, some links to studies and research papers that utilize Strawberry Fields.

#. D\. Su, K. K. Sabapathy, C. R. Myers, H. Qi, C. Weedbrook, and K. Brádler. *Implementing quantum algorithms on temporal photonic cluster states.* arXiv, 2018. `arXiv:1805.02645 <https://arxiv.org/abs/1805.02645>`_.

#. N\. Quesada, and A. M. Brańczyk. *Gaussian functions are optimal for waveguided nonlinear-quantum-optical processes.* arXiv, 2018. `arXiv:1805.06868 <https://arxiv.org/abs/1805.06868>`_

#. N\. Killoran, T. R. Bromley, J. M. Arrazola, M. Schuld, N. Quesada, and S. Lloyd. *Continuous-variable quantum neural networks.* arXiv, 2018. `arXiv:1806.06871 <https://arxiv.org/abs/1806.06871>`_.

#. J\. M\. Arrazola, T. R. Bromley, J. Izaac, C. R. Myers, K. Brádler, and N. Killoran. *Machine learning method for state preparation and gate synthesis on photonic quantum computers.* arXiv, 2018. `arXiv:1807.10781 <https://arxiv.org/abs/1807.10781>`_.
