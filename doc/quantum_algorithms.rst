.. _algorithms:

Quantum algorithms
###################

.. role:: html(raw)
   :format: html

.. sectionauthor:: Josh Izaac <josh@xanadu.ai>

Quantum information and quantum computation are, by nature, very highly interdisciplinary fields; while the underlying framework is built upon concepts in quantum mechanics, its applications are far-reaching, with real world applications ranging from network analysis and cryptography, to biochemistry and communication theory. 

Here we will discuss several continuous-variable (CV) quantum algorithm examples that can be run with Strawberry Fields.

.. toctree::
   :maxdepth: 2
   :hidden:

   algorithms/teleportation.rst
   algorithms/gate_teleportation.rst
   algorithms/gaussian_cloning.rst
   algorithms/boson_sampling.rst
   algorithms/gaussian_boson_sampling.rst
   algorithms/hamiltonian_simulation.rst
   algorithms/iqp.rst



.. |tp| image:: _static/teleport.svg    
   :scale: 100%
   :align: middle
   :target: algorithms/teleportation.html
.. |gate| image::  _static/gate_teleport_ex.svg    
   :scale: 50%
   :align: top
   :target: algorithms/gate_teleportation.html
.. |clone| image::  _static/cloning.svg    
   :scale: 50%
   :align: top
   :target: algorithms/gaussian_cloning.html
.. |bs| image::  _static/boson_sampling.svg    
   :scale: 50%
   :align: top
   :target: algorithms/boson_sampling.html
.. |gbs| image::  _static/gaussian_boson_sampling.svg    
   :scale: 50%
   :align: top
   :target: algorithms/gaussian_boson_sampling.html
.. |hs| image::  _static/hamsim.svg    
   :width: 60%
   :align: top
   :target: algorithms/hamiltonian_simulation.html
.. |iqp| image::  _static/IQP.svg    
   :scale: 50%
   :align: top
   :target: algorithms/iqp.html

.. raw:: html

   <style>
      .docstable a {
        color: #119a68;
      }
      .docstable a:hover {
        color: #119a68;
      }
      .docstable img {
        padding-bottom: 20px;
      }
   </style>



.. rst-class:: docstable docstable-nohead


+----------------------------+---------------------------+
| |tp|                       | |gate|                    |
| :ref:`state_teleportation` | :ref:`gate_teleport`      |
+----------------------------+---------------------------+
| |clone|                    | |bs|                      |
| :ref:`gaussian_cloning`    | :ref:`boson_sampling`     |
+----------------------------+---------------------------+
| |gbs|                      | |hs|                      |
| :ref:`gaussian_algorithm`  | :ref:`ham_sim`            |
+----------------------------+---------------------------+
| |iqp|                      |                           |
| :ref:`iqp`                 |                           |
+----------------------------+---------------------------+








.. Quantum chemistry simulations
   ==============================


.. Variational quantum eigensolver
   ===============================



