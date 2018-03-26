 .. role:: html(raw)
   :format: html


What is quantum computing?
=============================================
.. 
    The only thing we're really missing, after looking at Rigetti's, Microsoft's, and
    ProjectQ's documentation, is an introduction to quantum computing for the non-scientist.
.. 
    That is, a page that introduces the idea of quantum computation with **no equations** (or very few equations),
    **no references**, and **lots of figures and pictures**, and discusses more the **application**
    and **potential** than the theory.
.. 
    We could actually do something interesting here, which is introduce CV quantum computing
    *without* having to go via qubits - I've never seen this done before, which would be useful.

Quantum mechanics, one of the breakthrough theoretical physics results of the early 20th century, transformed the way we viewed nature on very small scales. Not only did we now have to view discrete particles such as electrons and protons as waves, it also meant that phenomena we long thought were described solely as waves also sometimes behaved like discrete particles; the most famous being the discrete unit of light, the **photon**. On top of that, the probabilistic nature of quantum mechanics also defied common sense; it allowed for particles to exist in **superpositions** of different states, something that can't exist in our classical world.

.. raw:: html

    <div class="aside admonition" id="aside1">
        <a data-toggle="collapse" data-parent="#aside1" href="#content1" class="collapsed">
            <p class="first admonition-title">
                Quantum superposition and entanglement (click to expand) <i class="fas fa-chevron-circle-down"></i>
            </p>
        </a>
        <div id="content1" class="collapse" data-parent="#aside1" style="height: 0px;">

For example, consider Alice, who is doing her families weekly wash. As is the universal law of washing machines [#]_, the number of socks exiting doesn't quite seem to match the number of socks Alice put in. Originally, she had thrown in her sons red pair of socks and her daughters green pair of socks (Alice's family has an eclectic sense of fashion), but only two socks remain. In a resigned rage, she grabs the remaining socks, and, without checking which is which, decides to pass on the problem to her children. One sock goes into her sons pile, the other into her daughters.

When her son Charlie comes home from school and checks the washing, he finds one lone green sock. At that moment, we know that his sister Eve must have the red sock; classically, she had the red sock all along, but only now do we have the information to know that as well. 

In the quantum version of this missing sock scenario, however, the underlying physics is significantly different. In this case, the washing machine is preparing a **superposition** of the red and green sock states; the state of each socked is inseparably described as an entanglement of the individual red and green sock states:

.. math:: \frac{1}{\sqrt{2}} \left( \ket{\text{red}}_{\text{charlie}}\ket{\text{green}}_{\text{eve}} + \ket{\text{green}}_{\text{charlie}}\ket{\text{red}}_{\text{eve}} \right)

From this point on, the socks evolve as part of this entangled system. It's only when we make a **measurement** by observing one of the socks that we collapse the quantum entanglement, at this point we know exactly what colour the other sock is.

This is the main difference from the classical case - in the classical case, if we knew every single piece of information about the system, we would be able to predict exactly which sock was which before the measurement. The socks *always* have a deterministic colour, regardless whether they have been observed or not. In the quantum case, this is not true - the entangled system, with each sock having an undetermined colour, evolves deterministically *until* the measurement, which is then a stochastic process [#]_.

:html:`</div></div>`

The advances in our understanding of quantum mechanics has enabled us to take advantage of these surprising quantum phenomena and correlations, bringing us **lasers**, **GPS**, extremely accurate **atomic clocks**, and medical imaging tools such as **MRIs**.

In parallel to these advances, we were also making incredible progress with silicon-based computers in the second half of the 20th century. Yet, as computing power slowly increases, we seem to be approaching the limit of what we can do with classical silicon-based technology. `Moore's Law <https://en.wikipedia.org/wiki/Moore%27s_law>`_, which we have relied upon for more-or-less linear increase in processing power, is starting to falter, as companies such as Intel are beginning to struggle increasing the number of transistors on smaller and smaller chips [#]_. Even more of a hurdle are the theoretical complexities of computer science; a majority of the problems that are currently 'hard to compute' are `generally believed by computer scientists <https://en.wikipedia.org/wiki/P%3Dnp>`_ to be *impossible* to compute on a computer constrained to the classical world [#]_.

And this is where quantum computing comes in.

At around the same time that personal computers were starting to enter the average middle class home, physicists were starting to grapple with a particular set of questions regarding quantum theory - can we harness the weird world of quantum superposition and entanglement to surpass what is theoretically possible with classical computers?

Two intricately linked fields have grown out of answering this question, **quantum information theory** and **quantum computation**, which have brought together scientists from disciplines such as mathematics, computer science, quantum physicists, quantum chemistry, biochemistry, and more. We now believe that we are exceptionally close to making this a reality, and producing devices that can efficiently implement algorithms that would take years on current supercomputers.



Applications
----------------------------------

Due to the quantum nature of quantum computation, it is incredibly suited to modelling phenomena that happen on small size scales, where quantum effects come into play. As such, it has applications in

* **Biochemistry**: efficiently and accurately modelling the time evolution and interactions of molecules and other biological processes such as photosynthesis and protein folding would lead to new scientific discoveries [#]_.

* **Medicine and pharmaceuticals**: the ability to easily simulate the effects of synthesized molecules could revolutionize the medical industry, and provide new methods of drug manufacture.

In addition, the ability of quantum computation to harness superposition to probe large networks exponentially faster than we can do classically provides further applications in graph and network theory. This includes 

* **Network analysis**: the ability to efficiently characterize and search networks is incredibly important due to the huge number of real-life phenomena we model as networks. For example, the spread of disease, communication, power grids, the internet, are all examples of systems modelled as networks.

* **Finance**: by utilizing quantum correlations, we can analyse stock markets and provide in-depth insight (that is not shown by classical methods) significantly faster

Finally, quantum versions of well-known classical algorithms allow us to extend fields beyond what we can do classically:

* **Quantum cryptography**: 

* **Quantum machine learning and optimization**:

* **Quantum communication protocols**: 



Continuous-variable quantum computing
--------------------------------------





.. rubric:: Footnotes

.. [#] Stephen Hawking has suggested that the missing socks conundrum has an entirely physical explanation: quantum gravitational effects cause the spontaneous creation of mini black holes that rapidly decay after swallowing the sock. As far as this author is aware, this theory has not yet cleared peer review.

.. [#] For more sock-based quantum mechanics, have a read of `Bertlmann's socks and the nature of reality <https://doi.org/10.1051/jphyscol:1981202>`_ by John Bell.

.. [#] "These transitions are a natural part of the history of Moore's Law and are a by-product of the technical challenges of shrinking transistors while ensuring they can be manufactured in high volume," - `Brian Krzanich, CEO of Intel Corporation <https://www.infoworld.com/article/2949153/hardware/intel-pushes-10nm-chipmaking-process-to-2017-slowing-moores-law.html>`_

.. [#] While this hasn't yet been proved, it is strongly believed that :math:`P\neq NP`. That is, a problem that can be verified efficiently in a polynomial amount of time (for example, checking whether :math:`x,y,\dots` are factors of some large number) *does not mean* there is an efficient way of solving the problem. This is one of the Millennium Prize Problems, with a $1 million dollar prize awardable to the first verified proof (or non-proof!). Despite the challenges involved in solving this problem, attempted proofs by enthusiasts are posted `almost monthly on the arXiv <http://www.win.tue.nl/~gwoegi/P-versus-NP.htm>`_. For more details on why these are almost always incorrect, see `Eight Signs A Claimed P≠NP Proof Is Wrong <https://www.scottaaronson.com/blog/?p=458>`_ by Scott Aaronson.

.. [#] For a example quantum simulation that can be performed by Strawberry Fields, see the :ref:`Hamiltonian simulation <ham_sim>` section of the quantum algorithms page.