# Release 0.16.0 (development release)

<h3>New features since last release</h3>

* Adds the ability to construct time domain multiplexing algorithms via the new
  `sf.TDMProgram` class, for highly scalable simulation of Gaussian states.
  [(#440)](https://github.com/XanaduAI/strawberryfields/pull/440)

  For example, creating and simulating a time domain program with 2 concurrent modes:

  ```pycon
  >>> import strawberryfields as sf
  >>> from strawberryfields import ops
  >>> prog = sf.TDMProgram(N=2)
  >>> with prog.context([1, 2], [3, 4], copies=3) as (p, q):
  ...     ops.Sgate(0.7, 0) | q[1]
  ...     ops.BSgate(p[0]) | (q[0], q[1])
  ...     ops.MeasureHomodyne(p[1]) | q[0]
  >>> eng = sf.Engine("gaussian")
  >>> results = eng.run(prog)
  >>> print(results.all_samples)
  {0: [array([1.26208025]), array([1.53910032]), array([-1.29648336]),
  array([0.75743215]), array([-0.17850101]), array([-1.44751996])]}
  ```

  For more details, see the [code
  documentation](https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.TDMProgram.html).

<h3>Improvements</h3>

<h3>Breaking Changes</h3>

<h3>Bug fixes</h3>

<h3>Documentation</h3>

 * Adds further testing and coverage descriptions.
  [(#461)](https://github.com/XanaduAI/strawberryfields/pull/461)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Theodor Isacsson, Josh Izaac, Fabian Laudenbach, Nicolás Quesada, Antal Száva

# Release 0.15.1 (current release)

<h3>Improvements</h3>

* Adds the ability to bypass recompilation of programs if they have been
  compiled already to the target device.
  [(#447)](https://github.com/XanaduAI/strawberryfields/pull/447)

<h3>Breaking Changes</h3>

* Changes the default compiler for devices that don't specify a default from `"Xcov"` to `"Xunitary"`.
  This compiler is slightly more strict and only compiles the unitary, not the initial squeezers,
  however avoids any unintentional permutations.
  [(#445)](https://github.com/XanaduAI/strawberryfields/pull/445)

<h3>Bug fixes</h3>

* Fixes a bug where a program that amounts to the identity operation would cause an error when
  compiled using the `xcov` compiler.
  [(#444)](https://github.com/XanaduAI/strawberryfields/pull/444)

<h3>Documentation</h3>

* Updates the `README.rst` file and hardware access links.
  [(#448)](https://github.com/XanaduAI/strawberryfields/pull/448)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Theodor Isacsson, Josh Izaac, Nathan Killoran, Nicolás Quesada, Antal Száva

# Release 0.15.0

<h3>New features since last release</h3>

* Adds the ability to train variational GBS circuits in the applications layer.
  [(#387)](https://github.com/XanaduAI/strawberryfields/pull/387)
  [(#388)](https://github.com/XanaduAI/strawberryfields/pull/388)
  [(#391)](https://github.com/XanaduAI/strawberryfields/pull/391)
  [(#393)](https://github.com/XanaduAI/strawberryfields/pull/393)
  [(#414)](https://github.com/XanaduAI/strawberryfields/pull/414)
  [(#415)](https://github.com/XanaduAI/strawberryfields/pull/415)

  Trainable parameters can be embedded into a VGBS class:

  ```python
  from strawberryfields.apps import data, train

  d = data.Mutag0()
  embedding = train.Exp(d.modes)
  n_mean = 5

  vgbs = train.VGBS(d.adj, 5, embedding, threshold=False, samples=np.array(d[:1000]))
  ```

  Properties of the variational GBS distribution for different choices of
  trainable parameters can then be inspected:

  ```pycon
  >>> params = 0.1 * np.ones(d.modes)
  >>> vgbs.n_mean(params)
  3.6776094165797364
  ```

  A cost function can then be created and its value and gradient accessed:

  ```pycon
  >>> h = lambda x: np.sum(x)
  >>> cost = train.Stochastic(h, vgbs)
  >>> cost(params, n_samples=1000)
  3.940396998165503
  >>> cost.grad(params, n_samples=1000)
  array([-0.54988876, -0.49270263, -0.6628071 , -1.13057762, -1.13568456,
       -0.70180571, -0.6266806 , -0.68803539, -1.11032533, -1.12853718,
       -0.59172261, -0.47830748, -0.96901676, -0.66938217, -0.85162006,
       -0.27188134, -0.26955011])
  ```

  For more details, see the [VGBS training
  demo](https://strawberryfields.ai/photonics/apps/run_tutorial_training.html).

* Feature vectors of graphs can now be calculated exactly in the `apps.similarity` module of the
  applications layer. Datasets of pre-calculated feature vectors are available in `apps.data`.
  [(#390)](https://github.com/XanaduAI/strawberryfields/pull/390)
  [(#401)](https://github.com/XanaduAI/strawberryfields/pull/401)

  ```pycon
  >>> from strawberryfields.apps import data
  >>> from strawberryfields.apps.similarity import feature_vector_sampling
  >>> samples = data.Mutag0()
  >>> feature_vector_sampling(samples, [2, 4, 6])
  [0.19035, 0.2047, 0.1539]
  ```

  For more details, see the [graph similarlity
  demo](https://strawberryfields.ai/photonics/apps/run_tutorial_similarity.html).

* A new `strawberryfields.apps.qchem` module has been introduced, centralizing all quantum chemistry
  applications. This includes various new features and improvements:

  * Adds the `apps.qchem.duschinsky()` function for generation of the
    Duschinsky rotation matrix and displacement vector which are needed to simulate a vibronic process
    with Strawberry Fields.
    [(#434)](https://github.com/XanaduAI/strawberryfields/pull/434)

  * Adds the `apps.qchem.dynamics` module for simulating vibrational quantum dynamics in molecules.
    [(#402)](https://github.com/XanaduAI/strawberryfields/pull/402)
    [(#411)](https://github.com/XanaduAI/strawberryfields/pull/411)
    [(#419)](https://github.com/XanaduAI/strawberryfields/pull/419)
    [(#421)](https://github.com/XanaduAI/strawberryfields/pull/421)
    [(#423)](https://github.com/XanaduAI/strawberryfields/pull/423)
    [(#430)](https://github.com/XanaduAI/strawberryfields/pull/430)

    This includes:

    - `dynamics.evolution()` constructs a custom operation that encodes the input chemical
      information. This custom operation can then be used within a Strawberry Fields `Program`.

    - `dynamics.sample_coherent()`, `dynamics.sample_fock()` and `dynamics.sample_tmsv()` functions
      allow for generation of samples from a variety of input states.

    - The probability of an excited state can then be estimated with the `dynamics.prob()` function,
      which calculates the relative frequency of the excited state among the generated samples.

    - Finally, the `dynamics.marginals()` function generates marginal distributions.

  * The `sf.apps.vibronic` module has been relocated to within the `qchem` module. As a result, the
    `apps.sample.vibronic()` function is now accessible under `apps.qchem.vibronic.sample()`,
    providing a single location for quantum chemistry functionality.
    [(#416)](https://github.com/XanaduAI/strawberryfields/pull/416)

  For more details, please see the [qchem
  documentation](https://strawberryfields.readthedocs.io/en/latest/code/api/strawberryfields.apps.qchem.html).

* The `GaussianState` returned from simulations using the Gaussian backend
  now has feature parity with the `FockState` object returned from the Fock backends.
  [(#407)](https://github.com/XanaduAI/strawberryfields/pull/407)

  In particular, it now supports the following methods:

  - `GaussianState.dm()`
  - `GaussianState.ket()`
  - `GaussianState.all_fock_probs()`

  In addition, the existing `GaussianState.reduced_dm()` method now supports
  multi-mode reduced density matrices.

* Adds the `sf.utils.samples_expectation`, `sf.utils.samples_variance` and
  `sf.utils.all_fock_probs_pnr` functions for obtaining counting statistics from samples.
  [(#399)](https://github.com/XanaduAI/strawberryfields/pull/399)

* Compilation of Strawberry Fields programs has been overhauled.

  * Strawberry Fields can now access the Xanadu Cloud device specifications API.
    The `Connection` class has a new method `Connection.get_device`,
    which returns a `DeviceSpec` class.
    [(#429)](https://github.com/XanaduAI/strawberryfields/pull/429)
    [(#432)](https://github.com/XanaduAI/strawberryfields/pull/432)

  * New `Xstrict`, `Xcov`, and `Xunitary` compilers for compiling programs into the X architecture
    have been added.
    [(#358)](https://github.com/XanaduAI/strawberryfields/pull/358)
    [(#438)](https://github.com/XanaduAI/strawberryfields/pull/438)

  * Finally, the `strawberryfields.circuitspecs` module has been renamed to
    `strawberryfields.compilers`.

* Adds `diagonal_expectation` method for the `BaseFockState` class, which returns
  the expectation value of any operator that is diagonal in the number basis.
  [(#389)](https://github.com/XanaduAI/strawberryfields/pull/389)

* Adds `parity_expectation` method as an instance of `diagonal_expectation` for
  the `BaseFockState` class, and its own function for `BaseGaussianState`.
  This returns the expectation value of the parity operator,
  defined as (-1)^N.
  [(#389)](https://github.com/XanaduAI/strawberryfields/pull/389)

<h3>Improvements</h3>

* Modifies the rectangular interferometer decomposition to make it more efficient for hardware
  devices. Rather than decomposing the interferometer using Clements :math:`T` matrices, the
  decomposition now directly produces Mach-Zehnder interferometers corresponding to on-chip phases.
  [(#363)](https://github.com/XanaduAI/strawberryfields/pull/363)

* Changes the `number_expectation` method for the `BaseFockState` class to be an instance of
  `diagonal_expectation`.
  [(#389)](https://github.com/XanaduAI/strawberryfields/pull/389)

* Increases the speed at which the following gates are generated: `Dgate`, `Sgate`, `BSgate` and
  `S2gate` by relying on a recursive implementation recently introduced in `thewalrus`. This has
  substantial effects on the speed of the `Fockbackend` and the `TFbackend`, especially for high
  cutoff values.
  [(#378)](https://github.com/XanaduAI/strawberryfields/pull/378)
  [(#381)](https://github.com/XanaduAI/strawberryfields/pull/381)

* All measurement samples can now be accessed via the `results.all_samples` attribute, which returns
  a dictionary mapping the mod index to a list of measurement values. This is useful for cases where
  a single mode may be measured multiple times.
  [(#433)](https://github.com/XanaduAI/strawberryfields/pull/433)

<h3>Breaking Changes</h3>

* Removes support for Python 3.5.
  [(#385)](https://github.com/XanaduAI/strawberryfields/pull/385)

* Complex parameters now are expected in polar form as two separate real parameters.
  [(#378)](https://github.com/XanaduAI/strawberryfields/pull/378)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Tom Bromley, Jack Ceroni, Aroosa Ijaz, Theodor Isacsson, Josh Izaac, Nathan
Killoran, Soran Jahangiri, Shreya P. Kumar, Filippo Miatto, Nicolás Quesada, Antal Száva


# Release 0.14.0

<h3>New features since last release</h3>

* Dark counts can now be added to the samples received from photon measurements in the
  Fock basis (sf.ops.MeasureFock) during a simulation.

* The `"tf"` backend now supports TensorFlow 2.0 and above.
  [(#283)](https://github.com/XanaduAI/strawberryfields/pull/283)
  [(#320)](https://github.com/XanaduAI/strawberryfields/pull/320)
  [(#323)](https://github.com/XanaduAI/strawberryfields/pull/323)
  [(#361)](https://github.com/XanaduAI/strawberryfields/pull/361)
  [(#372)](https://github.com/XanaduAI/strawberryfields/pull/372)
  [(#373)](https://github.com/XanaduAI/strawberryfields/pull/373)
  [(#374)](https://github.com/XanaduAI/strawberryfields/pull/374)
  [(#375)](https://github.com/XanaduAI/strawberryfields/pull/375)
  [(#377)](https://github.com/XanaduAI/strawberryfields/pull/377)

  For more details and demonstrations of the new TensorFlow 2.0-compatible backend,
  see our [optimization and machine learning tutorials](https://strawberryfields.readthedocs.io/en/stable/introduction/tutorials.html#optimization-and-machine-learning).

  For example, using TensorFlow 2.0 to train a variational photonic
  circuit:

  ```python
  eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 7})
  prog = sf.Program(1)

  with prog.context as q:
      # Apply a single mode displacement with free parameters
      Dgate(prog.params("a"), prog.params("p")) | q[0]

  opt = tf.keras.optimizers.Adam(learning_rate=0.1)

  alpha = tf.Variable(0.1)
  phi = tf.Variable(0.1)

  for step in range(50):
      # reset the engine if it has already been executed
      if eng.run_progs:
          eng.reset()

      with tf.GradientTape() as tape:
          # execute the engine
          results = eng.run(prog, args={'a': alpha, 'p': phi})
          # get the probability of fock state |1>
          prob = results.state.fock_prob([1])
          # negative sign to maximize prob
          loss = -prob

      gradients = tape.gradient(loss, [alpha, phi])
      opt.apply_gradients(zip(gradients, [alpha, phi]))
      print("Value at step {}: {}".format(step, prob))
  ```

* Adds the method `number_expectation`  that calculates the expectation value of the product of the
  number operators of a given set of modes.
  [(#348)](https://github.com/XanaduAI/strawberryfields/pull/348/)

  ```python
  prog = sf.Program(3)
  with prog.context as q:
      ops.Sgate(0.5) | q[0]
      ops.Sgate(0.5) | q[1]
      ops.Sgate(0.5) | q[2]
      ops.BSgate(np.pi/3, 0.1) |  (q[0], q[1])
      ops.BSgate(np.pi/3, 0.1) |  (q[1], q[2])
  ```

  Executing this on the Fock backend,

  ```python
  >>> eng = sf.Engine("fock", backend_options={"cutoff_dim": 10})
  >>> state = eng.run(prog).state
  ```

  we can compute the expectation value :math:`\langle \hat{n}_0\hat{n}_2\rangle`:

  ```python
  >>> state.number_expectation([0, 2])
  ```

<h3>Improvements</h3>

* Add details to the error message for failed remote jobs.
  [(#370)](https://github.com/XanaduAI/strawberryfields/pull/370)

* The required version of The Walrus was increased to version 0.12, for
  tensor number expectation support.
  [(#380)](https://github.com/XanaduAI/strawberryfields/pull/380)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Tom Bromley, Theodor Isacsson, Josh Izaac, Nathan Killoran, Filippo Miatto, Nicolás Quesada,
Antal Száva, Paul Tan.


# Release 0.13.0

<h3>New features since last release</h3>

* Adds initial support for the Xanadu's photonic quantum hardware.
  [(#101)](https://github.com/XanaduAI/strawberryfields/pull/101)
  [(#148)](https://github.com/XanaduAI/strawberryfields/pull/148)
  [(#294)](https://github.com/XanaduAI/strawberryfields/pull/294)
  [(#327)](https://github.com/XanaduAI/strawberryfields/pull/327)
  [(#328)](https://github.com/XanaduAI/strawberryfields/pull/328)
  [(#329)](https://github.com/XanaduAI/strawberryfields/pull/329)
  [(#330)](https://github.com/XanaduAI/strawberryfields/pull/330)
  [(#334)](https://github.com/XanaduAI/strawberryfields/pull/334)
  [(#336)](https://github.com/XanaduAI/strawberryfields/pull/336)
  [(#337)](https://github.com/XanaduAI/strawberryfields/pull/337)
  [(#339)](https://github.com/XanaduAI/strawberryfields/pull/339)

  Jobs can now be submitted to the Xanadu cloud platform to be run
  on supported hardware using the new `RemoteEngine`:

  ```python
  import strawberryfields as sf
  from strawberryfields import ops
  from strawberryfields.utils import random_interferometer

  # replace AUTHENTICATION_TOKEN with your Xanadu cloud access token
  con = sf.api.Connection(token="AUTH_TOKEN")
  eng = sf.RemoteEngine("X8", connection=con)
  prog = sf.Program(8)

  U = random_interferometer(4)

  with prog.context as q:
      ops.S2gate(1.0) | (q[0], q[4])
      ops.S2gate(1.0) | (q[1], q[5])
      ops.S2gate(1.0) | (q[2], q[6])
      ops.S2gate(1.0) | (q[3], q[7])

      ops.Interferometer(U) | q[:4]
      ops.Interferometer(U) | q[4:]
      ops.MeasureFock() | q

  result = eng.run(prog, shots=1000)
  ```

  For more details, see the
  [photonic hardware quickstart](https://strawberryfields.readthedocs.io/en/latest/introduction/photonic_hardware.html)
  and [tutorial](https://strawberryfields.readthedocs.io/en/latest/tutorials/tutorial_X8.html).

* Significantly speeds up the Fock backend of Strawberry Fields,
  through a variety of changes:

  - The Fock backend now uses The Walrus high performance implementations of
    the displacement, squeezing, two-mode squeezing, and beamsplitter operations.
    [(#287)](https://github.com/XanaduAI/strawberryfields/pull/287)
    [(#289)](https://github.com/XanaduAI/strawberryfields/pull/289)

  - Custom tensor contractions which make use of symmetry relations for the beamsplitter
    and the two-mode squeeze gate have been added, as well as more efficient contractions
    for diagonal operations in the Fock basis.
    [(#292)](https://github.com/XanaduAI/strawberryfields/pull/292)

<br>

* New `sf` command line program for configuring Strawberry Fields for access
  to the Xanadu cloud platform, as well as submitting and executing jobs from the
  command line.
  [(#146)](https://github.com/XanaduAI/strawberryfields/pull/146)
  [(#312)](https://github.com/XanaduAI/strawberryfields/pull/312)

  The new Strawberry Fields command line program `sf` provides several utilities
  including:

  * `sf configure [--token] [--local]`: configure the connection to the cloud platform

  * `sf run input [--output FILE]`: submit and execute quantum programs from the command line

  * `sf --ping`: verify your connection to the Xanadu cloud platform

  For more details, see the
  [documentation](https://strawberryfields.readthedocs.io/en/stable/code/sf_cli.html).

* New configuration functions to load configuration from
  keyword arguments, environment variables, and configuration files.
  [(#298)](https://github.com/XanaduAI/strawberryfields/pull/298)
  [(#306)](https://github.com/XanaduAI/strawberryfields/pull/306)

  This includes the ability to automatically store Xanadu cloud platform
  credentials in a configuration file using the new function

  ```python
  sf.store_account("AUTHENTICATION_TOKEN")
  ```

  as well as from the command line,

  ```bash
  $ sf configure --token AUTHENTICATION_TOKEN
  ```

  Configuration files can be saved globally, or locally on a per-project basis.
  For more details, see the
  [configuration documentation](https://strawberryfields.readthedocs.io/en/stable/introduction/configuration.html)

* Adds configuration functions for resetting, deleting configurations, as
  well as displaying available configuration files.
  [(#359)](https://github.com/XanaduAI/strawberryfields/pull/359)

* Adds the `x_quad_values` and `p_quad_values` methods to the `state` class.
  This allows calculation of x and p quadrature
  probability distributions by integrating across the Wigner function.
  [(#270)](https://github.com/XanaduAI/strawberryfields/pull/270)

* Adds support in the applications layer for node-weighted graphs.

  Sample from graphs with node weights using a special-purpose encoding
  [(#295)](https://github.com/XanaduAI/strawberryfields/pull/295):

  ```python
  from strawberryfields.apps import sample

  # generate a random graph
  g = nx.erdos_renyi_graph(20, 0.6)
  a = nx.to_numpy_array(g)

  # define node weights
  # and encode into the adjacency matrix
  w = [i for i in range(20)]
  a = sample.waw_matrix(a, w)

  s = sample.sample(a, n_mean=10, n_samples=10)
  s = sample.postselect(s, min_count=4, max_count=20)
  s = sample.to_subgraphs(s, g)
  ```

  Node weights can be input to search algorithms in the `clique` and `subgraph` modules
  [(#296)](https://github.com/XanaduAI/strawberryfields/pull/296)
  [(#297)](https://github.com/XanaduAI/strawberryfields/pull/297):

  ```python
  from strawberryfields.apps import clique
  c = [clique.shrink(s_, g, node_select=w) for s_ in s]
  [clique.search(c_, g, iterations=10, node_select=w) for c_ in c]
  ```

  ```python
  from strawberryfields.apps import subgraph
  subgraph.search(s, g, min_size=5, max_size=8, node_select=w)
  ```

<h3>Improvements</h3>

* Moved Fock backend apply-gate functions to `Circuit` class, and removed
  `apply_gate_einsum` and `Circuits._apply_gate`, since they were no longer used.
  [(#293)](https://github.com/XanaduAI/strawberryfields/pull/293/)

* Results returned from all backends now have a unified type and shape.
  In addition, attempting to use batching, post-selection and feed-foward together
  with multiple shots now raises an error.
  [(#300)](https://github.com/XanaduAI/strawberryfields/pull/300)

* Modified the rectangular decomposition to ensure that identity-like
  unitaries are implemented with no swaps.
  [(#311)](https://github.com/XanaduAI/strawberryfields/pull/311)

<h3>Bug fixes</h3>

* Symbolic Operation parameters are now compatible with TensorFlow 2.0 objects.
  [(#282)](https://github.com/XanaduAI/strawberryfields/pull/282)

* Added `sympy>=1.5` to the list of dependencies.
  Removed the `sympy.functions.atan2` workaround now that SymPy has been fixed.
  [(#280)](https://github.com/XanaduAI/strawberryfields/pull/280)

* Removed two unnecessary else statements that pylint complained about.
  [(#290)](https://github.com/XanaduAI/strawberryfields/pull/290)

* Fixed a bug in the `MZgate`, where the internal and external phases were
  in the wrong order in both the docstring and the argument list. The new
  signature is `MZgate(phase_in, phase_ex)`, matching the existing `rectangular_symmetric`
  decomposition.
  [(#301)](https://github.com/XanaduAI/strawberryfields/pull/301)

* Updated the relevant methods in `RemoteEngine` and `Connection` to derive `shots`
  from the Blackbird script or `Program` if not explicitly specified.
  [(#327)](https://github.com/XanaduAI/strawberryfields/pull/327)

* Fixed a bug in homodyne measurements in the Fock backend, where computed
  probability values could occasionally include small negative values
  due to floating point precision error.
  [(#364)](https://github.com/XanaduAI/strawberryfields/pull/364)

* Fixed a bug that caused an exception when printing results with no state.
  [(#367)](https://github.com/XanaduAI/strawberryfields/pull/367)

* Improves the Takagi decomposition, by making explicit use of the eigendecomposition of real symmetric matrices. [(#352)](https://github.com/XanaduAI/strawberryfields/pull/352)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ville Bergholm, Tom Bromley, Jack Ceroni, Theodor Isacsson, Josh Izaac, Nathan Killoran, Shreya P Kumar,
Leonhard Neuhaus, Nicolás Quesada, Jeremy Swinarton, Antal Száva, Paul Tan, Zeid Zabaneh.


# Release 0.12.1

<h3>New features</h3>

* A new `gaussian_unitary` circuitspec that can be used to compile any sequency of Gaussian
  transformations into a single `GaussianTransform` gate and a sequence of single mode `Dgate`s.
  [(#238)](https://github.com/XanaduAI/strawberryfields/pull/238)

<h3>Improvements</h3>

* Add new Strawberry Fields applications paper to documentation
  [(#274)](https://github.com/XanaduAI/strawberryfields/pull/274)

* Update figure for GBS device in documentation
  [(#275)](https://github.com/XanaduAI/strawberryfields/pull/275)

<h3>Bug fixes</h3>

* Fix installation issue with incorrect minimum version number for `thewalrus`
  [(#272)](https://github.com/XanaduAI/strawberryfields/pull/272)
  [(#277)](https://github.com/XanaduAI/strawberryfields/pull/277)

* Correct URL for image in `README`
  [(#273)](https://github.com/XanaduAI/strawberryfields/pull/273)

* Add applications data to `MANIFEST.in`
  [(#278)](https://github.com/XanaduAI/strawberryfields/pull/278)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ville Bergholm, Tom Bromley, Nicolás Quesada, Paul Tan


# Release 0.12.0

<h3>New features</h3>

* A new applications layer, allowing users to interface samples generated from near-term photonic
  devices with problems of practical interest. The `apps` package consists of the following
  modules:

  - The `apps.sample` module, for encoding graphs and molecules into Gaussian boson sampling
    (GBS) and generating corresponding samples.

  - The `apps.subgraph` module, providing a heuristic algorithm for finding dense subgraphs from GBS
    samples.

  - The `apps.clique` module, providing tools to convert subgraphs sampled from GBS into cliques and
    a heuristic to search for larger cliques.

  - The `apps.similarity` module, allowing users to embed graphs into high-dimensional feature
    spaces using GBS. Resulting feature vectors provide measures of graph similarity for machine
    learning tasks.

  - The `apps.points` module, allowing users to sample subsets of points according to new
    point processes that can be generated from a GBS device.

  - The `apps.vibronic` module, providing functionality to construct the vibronic absorption
    spectrum of a molecule from GBS samples.

<h3>Improvements</h3>

* The documentation was improved and refactored. Changes include:

  - A brand new theme, now matching PennyLane
    [(#262)](https://github.com/XanaduAI/strawberryfields/pull/262)

  - The documentation has been restructured to make it
    easier to navigate
    [(#266)](https://github.com/XanaduAI/strawberryfields/pull/266)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Tom Bromley, Josh Izaac, Soran Jahangiri, Nicolás Quesada


# Release 0.11.2

<h3>New features</h3>

* Adds the MZgate to ops.py, representing a Mach-Zehnder interferometer. This is
  not a primitive of the existing simulator backends; rather, `_decompose()` is
  defined, decomposing it into an external phase shift, two 50-50 beamsplitters,
  and an internal phase shift.
  [(#127)](https://github.com/XanaduAI/strawberryfields/pull/127)

* The `Chip0Spec` circuit class now defines a `compile` method, allowing
  arbitrary unitaries comprised of `{Interferometer, BSgate, Rgate, MZgate}`
  operations to be validated and compiled to match the topology of chip0.
  [(#127)](https://github.com/XanaduAI/strawberryfields/pull/127)

* `strawberryfields.ops.BipartiteGraphEmbed` quantum decomposition now added,
  allowing a bipartite graph to be embedded on a device that allows for
  initial two-mode squeezed states, and block diagonal unitaries.

* Added threshold measurements, via the new operation `MeasureThreshold`,
  and provided implementation of this operation in the Gaussian backend.
  [(#152)](https://github.com/XanaduAI/strawberryfields/pull/152)

* Programs can now have free parameters/arguments which are only bound to
  numerical values when the Program is executed, by supplying the actual
  argument values to the `Engine.run` method.
  [(#163)](https://github.com/XanaduAI/strawberryfields/pull/163)

<h3>API Changes</h3>

* The `strawberryfields.ops.Measure` shorthand has been deprecated in favour
  of `strawberryfields.ops.MeasureFock()`.
  [(#145)](https://github.com/XanaduAI/strawberryfields/pull/145)

* Several changes to the `strawberryfields.decompositions` module:
  [(#127)](https://github.com/XanaduAI/strawberryfields/pull/127)

  - The name `clements` has been replaced with `rectangular` to
    correspond with the shape of the resulting decomposition.

  - All interferometer decompositions (`rectangular`, `rectangular_phase_end`,
    `rectangular_symmetric`, and `triangular`) now have standardized outputs
    `(tlist, diag, tilist)`, so they can easily be swapped.

* Several changes to `ops.Interferometer`:
  [(#127)](https://github.com/XanaduAI/strawberryfields/pull/127)

  - The calculation of the ops.Interferometer decomposition has been moved from
    `__init__` to `_decompose()`, allowing the interferometer decomposition type
    to be set by a `CircuitSpec` during compilation.

  - `**kwargs` is now passed through from `Operation.decompose` -> `Gate.decompose`
    -> `SpecificOp._decompose`, allowing decomposition options to be passed during
    compilation.

  - `ops.Interferometer` now accepts the keyword argument `mesh` to be set during
    initialization, allowing the user to specify the decomposition they want.

* Moves the `Program.compile_seq` method to `CircuitSpecs.decompose`. This allows it
  to be accessed from the `CircuitSpec.compile` method. Furthermore, it now must also
  be passed the program registers, as compilation may sometimes require this.
  [(#127)](https://github.com/XanaduAI/strawberryfields/pull/127)

* Parameter class is replaced by `MeasuredParameter` and `FreeParameter`, both inheriting from
  `sympy.Symbol`. Fixed numeric parameters are handled by the built-in Python numeric
  classes and numpy arrays.
  [(#163)](https://github.com/XanaduAI/strawberryfields/pull/163)

* `Parameter`, `RegRefTransform` and `convert` are removed.
  [(#163)](https://github.com/XanaduAI/strawberryfields/pull/163)

<h3>Improvements</h3>

* Photon-counting measurements can now be done in the Gaussian backend for states with nonzero displacement.
  [(#154)](https://github.com/XanaduAI/strawberryfields/pull/154)

* Added a new test for the cubic phase gate
  [(#160)](https://github.com/XanaduAI/strawberryfields/pull/160)

* Added new integration tests for the Gaussian gates that are not primitive,
  i.e., P, CX, CZ, and S2.
  [(#173)](https://github.com/XanaduAI/strawberryfields/pull/173)

<h3>Bug fixes</h3>

* Fixed bug in `strawberryfields.decompositions.rectangular_symmetric` so its
  returned phases are all in the interval [0, 2*pi), and corrects the
  function docstring.
  [(#196)](https://github.com/XanaduAI/strawberryfields/pull/196)

* When using the `'gbs'` compilation target, the measured registers are now sorted in
  ascending order in the resulting compiled program.
  [(#144)](https://github.com/XanaduAI/strawberryfields/pull/144)

* Fixed typo in the Gaussian Boson Sampling example notebook.
  [(#133)](https://github.com/XanaduAI/strawberryfields/pull/133)

* Fixed a bug in the function `smeanxp` of the Gaussian Backend simulator.
  [(#154)](https://github.com/XanaduAI/strawberryfields/pull/154)

* Clarified description of matrices that are accepted by graph embed operation.
  [(#147)](https://github.com/XanaduAI/strawberryfields/pull/147)

* Fixed typos in the documentation of the CX gate and BSgate
  [(#166)](https://github.com/XanaduAI/strawberryfields/pull/166)
  [(#167)](https://github.com/XanaduAI/strawberryfields/pull/167)
  [(#169)](https://github.com/XanaduAI/strawberryfields/pull/169)


# Release 0.11.1

<h3>Improvements</h3>

* Added the `circuit_spec` attribute to `BaseBackend` to denote which CircuitSpecs class
  should be used to validate programs for each backend
  [(#125)](https://github.com/XanaduAI/strawberryfields/pull/125).

* Removed the `return_state` keyword argument from `LocalEngine.run()`. Now no state
  object is returned if `modes==[]`.
  [(#126)](https://github.com/XanaduAI/strawberryfields/pull/126)

* Fixed a typo in the boson sampling tutorial.
  [(#133)](https://github.com/XanaduAI/strawberryfields/pull/133)

<h3>Bug fixes</h3>

* Allows imported Blackbird programs to store `target` options
  as default run options. During eng.run, if no run options are provided
  as a keyword argument, the engine will fall back on the run options stored
  within the program.
  This fixes a bug where shots specified in Blackbird scripts were not being
  passed to `eng.run`.
  [(#130)](https://github.com/XanaduAI/strawberryfields/pull/130)

* Removes `ModuleNotFoundError` from the codebase, replacing all occurrences
  with `ImportError`. Since `ModuleNotFoundError` was only introduced in
  Python 3.6+, this fixes a bug where Strawberry Fields was not importable
  on Python 3.5
  [(#124)](https://github.com/XanaduAI/strawberryfields/pull/124).

* Updates the Chip0 template to use `MeasureFock() | [0, 1, 2, 3]`, which will
  allow correct fock measurement behaviour when simulated on the Gaussian backend
  [(#124)](https://github.com/XanaduAI/strawberryfields/pull/124).

* Fixed a bug in the `GraphEmbed` op, which was not correctly determining when a
  unitary was the identity
  [(#128)](https://github.com/XanaduAI/strawberryfields/pull/128).


# Release 0.11.0

This is a significant release, with breaking changes to how quantum programs are constructed and executed. For example, the following Strawberry Fields program, <= version 0.10:

```python
eng, q = sf.Engine(2, hbar=0.5)

with eng:
    Sgate(0.5) | q[0]
    MeasureFock() | q[0]

state = eng.run("fock", cutoff_dim=5)
ket = state.ket()
print(q[0].val)
```

would now be written, in v0.11, as follows:

```python
sf.hbar = 0.5
prog = sf.Program(2)
eng = sf.Engine("fock", backend_options={"cutoff_dim": 5})

with prog.context as q:
    Sgate(0.5) | q[0]
    MeasureFock() | q[0]

results = eng.run(prog)
ket = results.state.ket()
print(results.samples[0])
```

<h3>New features</h3>

- The functionality of the `Engine` class has been divided into two new classes: `Program`, which represents a quantum circuit or a fragment thereof, and `Engine`, which executes `Program` instances.

- Introduced the `BaseEngine` abstract base class and the `LocalEngine` child class. `Engine` is kept as an alias for `LocalEngine`.

- The Engine API has been changed slightly:

  The engine is initialized with the required backend, as well as a `backend_options` dictionary, which is passed to the backend:

    ```python
    eng = sf.Engine("fock", backend_options={"cutoff_dim": 5}
    ```

  `LocalEngine.run()` now accepts a program to execute, and returns a `Result` object that contains both a state object (`Result.state`) and measurement samples (`Result.samples`):

    ```python
    results = eng.run(prog)
    state = results.state
    samples = results.samples
    ```

  - `compile_options` can be provided when calling `LocalEngine.run()`. These are passed to the `compile()` method of the program before execution.

  - `run_options` can be provided when calling `LocalEngine.run()`. These are used to determine the characteristics of the measurements and state contained in the `Results` object returned after the program is finished executing.

  - `shots` keyword argument can be passed to `run_options`, enabling multi-shot sampling. Supported only
    in the Gaussian backend, and only for Fock measurements.

 - The Gaussian backend now officially supports Fock-basis measurements (`MeasureFock`), but does not update the quantum state after a Fock measurement.

- Added the `io` module, which is used to save/load standalone Blackbird scripts from/into Strawberry Fields. Note that the Blackbird DSL has been spun off as an independent package and is now a dependency of Strawberry Fields.

- Added a new interferometer decomposition `mach_zehnder` to the decompositions module.

- Added a `Configuration` class, which is used to load, store, save, and modify configuration options for Strawberry Fields.

- `hbar` is now set globally for the entire session, by setting the value of `sf.hbar` (default is 2).


- Added the ability to generate random real (orthogonal) interferometers and random block diagonal symplectic and covariance matrices.

- Added two top-level functions:
    - `about()`, which prints human-readable system info including installed versions of various Python packages.
    - `cite()`, which prints a bibtex citation for SF.

- Added a glossary to the documentation.

<h3>API Changes</h3>

- Added the `circuitspecs` subpackage, containing the `CircuitSpecs` class and a quantum circuit database.

  The database can be used to
    - Validate that a `Program` belongs in a specific circuit class.
    - Compile a `Program` for a desired circuit target, e.g., so that it can be executed on a given backend.
  The database includes a number of compilation targets, including Gaussian Boson Sampling circuits.

- The way hbar is handled has been simplified:
    - The backend API is now entirely hbar-independent, i.e., every backend API method is defined in terms of a and a^\dagger only, not x and p.
    - The backends always explicitly use `hbar=2` internally.
    - `hbar` is now a global, frontend-only variable that the user can set at the beginning of the session. It is used at the `Operation.apply()` level to scale the inputs and outputs of the backend API calls as needed, and inside the `State` objects.
    - The only backend API calls that need to do hbar scaling for the input parameters are the X, Z, and V gates, the Gaussian state decomposition, and homodyne measurements (both the returned value and postselection argument are scaled).

<h3>Improvements</h3>

- Removed TensorFlow as an explicit dependency of Strawberry Fields. Advanced users can still install TensorFlow manually using `pip install tensorflow==1.3` and use as before.

- The behaviour and function signature of the `GraphEmbed` operation has been updated.

- Remove the unused `Command.decomp` instance attribute.

- Better error messages for the `New` operation when used outside of a circuit.

- Docstrings updated in the decompositions module.

- Docstrings for Fock backend reformatted and cleaned up.

- Cleaning up of citations and `references.bib` file.

- Typos in documentation fixed.

<h3>Bug fixes</h3>

- Fixed a bug with installation on Windows for certain locales.
- Fixed a bug in the `New` operation.
- Bugfix in `Gate.merge()`
- Fixed bugs in `measure_fock` in the TensorFlow backend which caused samples to be evaluated independently and for conditional states to be potentially decoupled from the measurement results.
- Fixed a latent bug in `graph_embed`.
- Bugfix for Bloch-Messiah returning non-symplectic matrices when input is passive.

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ville Bergholm, Tom Bromley, Ish Dhand, Karel Dumon, Xueshi Guo, Josh Izaac, Nathan Killoran, Leonhard Neuhaus, Nicolás Quesada.



# Release 0.10

<h3>New features</h3>

- Added two new utility functions to extract a numerical representation of a circuit from an Engine object: `extract_unitary` and `extract_channel`.

- Added a LaTeX quantum circuit drawer, that outputs the engine queue or the applied operations as a qcircuit compatible circuit diagram.

- Added support for an alternative form of Clements decomposition, where the local phases occur at the end rather than in the middle of the beamsplitter array. This decomposition is more symmetric than the intermediate one, which could make it more robust. This form also makes it easier to implement a tensor-network simulation of linear optics.

- Adds the `GraphEmbed` quantum operation/decomposition to the Strawberry Fields frontend. This allows the embedding of an arbitrary (complex-valued) weighted adjacency matrix into a Gaussian boson sampler.

- Adds support for the Reck decomposition

- Added documentation to the Quantum Algorithms section on CV quantum neural networks

<h3>Improvements</h3>

- Test suite has been ported to pytest

- Linting improvements

- Made corrections to the Clements decomposition documentation and docstring, and fixed the Clements unit tests to ensure they are deterministic.

<h3>Bug fixes</h3>

- Fixed Bloch-Messiah bug arising when singular values were degenerate. Previously, the Bloch-Messiah decomposition did not return matrices in the canonical symplectic form if one or more of the Bloch-Messiah singular values were degenerate.

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Shahnawaz Ahmed, Thomas R. Bromley, Ish Dhand, Marcus Edwards, Christian Gogolin, Josh Izaac, Nathan Killoran, Filippo Miatto, Nicolás Quesada.


# Release 0.9

<h3>New features</h3>

- Updated the [Strawberry Fields gallery](https://strawberryfields.readthedocs.io/en/latest/gallery/gallery.html), featuring community-submitted content (tutorials, notebooks, repositories, blog posts, research papers, etc.) using Strawberry Fields

- Added the `@operation` decorator, which allows commonly-used algorithms and subroutines to be declared in blackbird code as one-liner operations

- Added a `ThermalLossChannel` to the Strawberry Fields API (currently supported by the Gaussian backend)

- Added a `poly_quad_expectation` method to the `state` objects for Gaussian and Fock backends

<h3>Improvements</h3>

- New and improved tests

- Fixed typos in code/documentation

<h3>Contributors</h3>

This release contains contributions from:

Juan Leni, Arthur Pesah, Brianna Gopaul, Nicolás Quesada, Josh Izaac, and Nathan Killoran.


# Release 0.8

<h3>New features</h3>
* You can now prepare multimode states in all backends, via the following new quantum operations in `strawberryfields.ops`:
    - `Ket`
    - `DensityMatrix`
    - `Gaussian`

  Both `Ket` and `DensityMatrix` work with the Fock backends, while `Gaussian` works with all three, applying the Williamson decomposition or, optionally, directly preparing the Gaussian backend with the provided Gaussian state.
* Added Gaussian decompositions to the front-end; these can be accessed via the new quantum operations `Interferometer`, `GaussianTransform`, `Gaussian`. These allow you to apply interferometers, Gaussian symplectic transformations, and prepare a state based on a covariance matrix respectively. You can also query the engine to determine the CV gate decompositions applied.
* Added the cross-Kerr interaction, accessible via the quantum operation `CKgate()`.
* Added utilities for creating random covariance, symplectic, and Gaussian unitary matrices in `strawberryfields.utils`.
* States can now be compared directly for equality - this is defined separately for Gaussian states and Fock basis states.


<h3>Improvements</h3>
* The engine logic and behaviour has been overhauled, making it simpler to use and understand.
    - `eng.run()` and `eng.reset()` now allow the user to alter parameters such as `cutoff_dim` between runs.
    - `eng.reset_backend()` has been renamed to `eng.reset()`, and now also implicitly resets the queue.
    - The engine can now be reset even in the case of modes having being added/deleted, with no side effects. This is due to the presence of register checkpoints, allowing the engine to keep track of register changes.
    - `eng.print_applied()` keeps track of multiple simulation runs, by using nested lists.
* A new parameter class is introduced - this is a developmental change, and does not affect the user-facing parts of Strawberry Fields. All parameters passed to quantum operations are 'wrapped' in this parameter class, which also contains several high level mathematical and array/tensor manipulation functions and methods.


<h3>Contributors</h3>
This release contains contributions from:

Ville Bergholm, Christian Gogolin, Nicolás Quesada, Josh Izaac, and Nathan Killoran.




# Release 0.7.3

<h3>New features</h3>
* Added Gaussian decompositions to the front-end; these can be accessed via the new quantum operations `Interferometer`, `GaussianTransform`, `CovarianceState`. These allow you to apply interferometers, Gaussian symplectic transformations, and prepare a state based on a covariance matrix respectively. You can also query the engine to determine the CV gate decompositions applied.
* Added utilities for creating random covariance, symplectic, and gaussian unitary matrices in `strawberryfields.utils`.

<h3>Improvements</h3>
* Created a separate package `strawberryfields-gpu` that requires `tensorflow-gpu`.
* Modified TFBackend to cache non-variable parts of the beamsplitter, to speed up computation.
* Minor performance improvement in `fock_prob()` by avoiding inverting a matrix twice.

<h3>Bug fixes</h3>
* Fixed bug #10 by adding the ability to reset the Fock modeMap and GaussianCircuit class
* Fixed bug #11 by reshaping the Fock probabilities if the state happens to be pure states
* Fixed Clements decomposition bug where some phase angles weren't applied
* Fixed typo in displaced squeezed formula in documentation
* Fix to prevent beamsplitter prefactor cache from breaking things if using two graphs
* Fix bug #13, GaussianBackend.state() raises an IndexError if all modes in the state have been deleted.


# Release 0.7.2

<h3>Bug fixes</h3>
* Fixed Tensorflow requirements in `setup.py`, so that installation will now work for versions of tensorflow>=1.3,<1.7

<h3>Known issues</h3>
* Tensorflow version 1.7 introduces some breaking API changes, so is currently not supported by Strawberry Fields.


# Release 0.7.1

Initial public release.

<h3>Contributors</h3>
This release contains contributions from:

Nathan Killoran, Josh Izaac, Nicolás Quesada, Matthew Amy, and Ville Bergholm.
