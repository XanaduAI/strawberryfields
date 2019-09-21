# Release 0.12.0-dev

### New features

* Adds in first version of an applications layer aimed at solving problems using
  Gaussian boson sampling. This layer focuses on graph-based problems and
  currently has algorithms for the densest ``k``-subgraph problem.
  [#164](https://github.com/XanaduAI/strawberryfields/pull/164)

* Adds the MZgate to ops.py, representing a Mach-Zehnder interferometer. This is
  not a primitive of the existing simulator backends; rather, `_decompose()` is
  defined, decomposing it into an external phase shift, two 50-50 beamsplitters,
  and an internal phase shift.
  [#127](https://github.com/XanaduAI/strawberryfields/pull/127)

* The `Chip0Spec` circuit class now defines a `compile` method, allowing
  arbitrary unitaries comprised of `{Interferometer, BSgate, Rgate, MZgate}`
  operations to be validated and compiled to match the topology of chip0.
  [#127](https://github.com/XanaduAI/strawberryfields/pull/127)

* `strawberryfields.ops.BipartiteGraphEmbed` quantum decomposition now added,
  allowing a bipartite graph to be embedded on a device that allows for
  initial two-mode squeezed states, and block diagonal unitaries.

* Added threshold measurements, via the new operation `MeasureThreshold`,
  and provided implementation of this operation in the Gaussian backend.
  [#152](https://github.com/XanaduAI/strawberryfields/pull/152)

* Adds new integration tests for the Gaussian gates that are not primitive, i.e., P, CX, CZ, and S2. Addresses issue [#171](https://github.com/XanaduAI/strawberryfields/issues/171)

### API Changes

* The `strawberryfields.ops.Measure` shorthand has been deprecated in favour
  of `strawberryfields.ops.MeasureFock()`.
  [#145](https://github.com/XanaduAI/strawberryfields/pull/145)

* Several changes to the `strawberryfields.decompositions` module:
  [#127](https://github.com/XanaduAI/strawberryfields/pull/127)

  - The name `clements` has been replaced with `rectangular` to
    correspond with the shape of the resulting decomposition.

  - All interferometer decompositions (`rectangular`, `rectangular_phase_end`,
    `rectangular_symmetric`, and `triangular`) now have standardized outputs
    `(tlist, diag, tilist)`, so they can easily be swapped.

* Several changes to `ops.Interferometer`:
  [#127](https://github.com/XanaduAI/strawberryfields/pull/127)

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
  [#127](https://github.com/XanaduAI/strawberryfields/pull/127)

### Improvements

* Photon-counting measurements can now be done in the Gaussian backend for states with nonzero displacement.
  [#154](https://github.com/XanaduAI/strawberryfields/pull/154)
  
* Added a new test for the cubic phase gate
  [#160](https://github.com/XanaduAI/strawberryfields/pull/160)

### Bug fixes

* Fixed bug in `strawberryfields.decompositions.rectangular_symmetric` so its
  returned phases are all in the interval [0, 2*pi), and corrects the
  function docstring.
  [#196](https://github.com/XanaduAI/strawberryfields/pull/196)

* When using the `'gbs'` compilation target, the measured registers are now sorted in
  ascending order in the resulting compiled program.
  [#144](https://github.com/XanaduAI/strawberryfields/pull/144)

* Fixed typo in the Gaussian Boson Sampling example notebook.
  [#133](https://github.com/XanaduAI/strawberryfields/pull/133)

* Fixed a bug in the function `smeanxp` of the Gaussian Backend simulator. 
  [#154](https://github.com/XanaduAI/strawberryfields/pull/154)
  
* Clarified description of matrices that are accepted by graph embed operation.
  [#147](https://github.com/XanaduAI/strawberryfields/pull/147)
  
* Fixed typos in the documentation of the CX gate and BSgate
  [#166](https://github.com/XanaduAI/strawberryfields/pull/166)
  [#167](https://github.com/XanaduAI/strawberryfields/pull/167)
  [#169](https://github.com/XanaduAI/strawberryfields/pull/169)

---

# Release 0.11.1

### Improvements

* Added the `circuit_spec` attribute to `BaseBackend` to denote which CircuitSpecs class
  should be used to validate programs for each backend
  [#125](https://github.com/XanaduAI/strawberryfields/pull/125).

* Removed the `return_state` keyword argument from `LocalEngine.run()`. Now no state
  object is returned if `modes==[]`.
  [#126](https://github.com/XanaduAI/strawberryfields/pull/126)

* Fixed a typo in the boson sampling tutorial.
  [#133](https://github.com/XanaduAI/strawberryfields/pull/133)

### Bug fixes

* Allows imported Blackbird programs to store `target` options
  as default run options. During eng.run, if no run options are provided
  as a keyword argument, the engine will fall back on the run options stored
  within the program.
  This fixes a bug where shots specified in Blackbird scripts were not being
  passed to `eng.run`.
  [#130](https://github.com/XanaduAI/strawberryfields/pull/130)

* Removes `ModuleNotFoundError` from the codebase, replacing all occurrences
  with `ImportError`. Since `ModuleNotFoundError` was only introduced in
  Python 3.6+, this fixes a bug where Strawberry Fields was not importable
  on Python 3.5
  [#124](https://github.com/XanaduAI/strawberryfields/pull/124).

* Updates the Chip0 template to use `MeasureFock() | [0, 1, 2, 3]`, which will
  allow correct fock measurement behaviour when simulated on the Gaussian backend
  [#124](https://github.com/XanaduAI/strawberryfields/pull/124).

* Fixed a bug in the `GraphEmbed` op, which was not correctly determining when a
  unitary was the identity
  [#128](https://github.com/XanaduAI/strawberryfields/pull/128).

---

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

### New features

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

### API Changes

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

### Improvements

- Removed TensorFlow as an explicit dependency of Strawberry Fields. Advanced users can still install TensorFlow manually using `pip install tensorflow==1.3` and use as before.

- The behaviour and function signature of the `GraphEmbed` operation has been updated.

- Remove the unused `Command.decomp` instance attribute.

- Better error messages for the `New` operation when used outside of a circuit.

- Docstrings updated in the decompositions module.

- Docstrings for Fock backend reformatted and cleaned up.

- Cleaning up of citations and `references.bib` file.

- Typos in documentation fixed.

## Bug fixes

- Fixed a bug with installation on Windows for certain locales.
- Fixed a bug in the `New` operation.
- Bugfix in `Gate.merge()`
- Fixed bugs in `measure_fock` in the TensorFlow backend which caused samples to be evaluated independently and for conditional states to be potentially decoupled from the measurement results.
- Fixed a latent bug in `graph_embed`.
- Bugfix for Bloch-Messiah returning non-symplectic matrices when input is passive.

### Contributors

This release contains contributions from (in alphabetical order):

Ville Bergholm, Tom Bromley, Ish Dhand, Karel Dumon, Xueshi Guo, Josh Izaac, Nathan Killoran, Leonhard Neuhaus, Nicolás Quesada.

---


# Release 0.10

### New features

- Added two new utility functions to extract a numerical representation of a circuit from an Engine object: `extract_unitary` and `extract_channel`.

- Added a LaTeX quantum circuit drawer, that outputs the engine queue or the applied operations as a qcircuit compatible circuit diagram.

- Added support for an alternative form of Clements decomposition, where the local phases occur at the end rather than in the middle of the beamsplitter array. This decomposition is more symmetric than the intermediate one, which could make it more robust. This form also makes it easier to implement a tensor-network simulation of linear optics.

- Adds the `GraphEmbed` quantum operation/decomposition to the Strawberry Fields frontend. This allows the embedding of an arbitrary (complex-valued) weighted adjacency matrix into a Gaussian boson sampler.

- Adds support for the Reck decomposition

- Added documentation to the Quantum Algorithms section on CV quantum neural networks

### Improvements

- Test suite has been ported to pytest

- Linting improvements

- Made corrections to the Clements decomposition documentation and docstring, and fixed the Clements unit tests to ensure they are deterministic.

## Bug fixes

- Fixed Bloch-Messiah bug arising when singular values were degenerate. Previously, the Bloch-Messiah decomposition did not return matrices in the canonical symplectic form if one or more of the Bloch-Messiah singular values were degenerate.

### Contributors

This release contains contributions from (in alphabetical order):

Shahnawaz Ahmed, Thomas R. Bromley, Ish Dhand, Marcus Edwards, Christian Gogolin, Josh Izaac, Nathan Killoran, Filippo Miatto, Nicolás Quesada.


# Release 0.9

## Summary of changes from 0.8

### New features

- Updated the [Strawberry Fields gallery](https://strawberryfields.readthedocs.io/en/latest/gallery/gallery.html), featuring community-submitted content (tutorials, notebooks, repositories, blog posts, research papers, etc.) using Strawberry Fields

- Added the `@operation` decorator, which allows commonly-used algorithms and subroutines to be declared in blackbird code as one-liner operations

- Added a `ThermalLossChannel` to the Strawberry Fields API (currently supported by the Gaussian backend)

- Added a `poly_quad_expectation` method to the `state` objects for Gaussian and Fock backends

### Improvements

- New and improved tests

- Fixed typos in code/documentation

### Contributors

This release contains contributions from:

Juan Leni, Arthur Pesah, Brianna Gopaul, Nicolás Quesada, Josh Izaac, and Nathan Killoran.


# Release 0.8

## Summary of changes from 0.7

### New features
* You can now prepare multimode states in all backends, via the following new quantum operations in `strawberryfields.ops`:
    - `Ket`
    - `DensityMatrix`
    - `Gaussian`

  Both `Ket` and `DensityMatrix` work with the Fock backends, while `Gaussian` works with all three, applying the Williamson decomposition or, optionally, directly preparing the Gaussian backend with the provided Gaussian state.
* Added Gaussian decompositions to the front-end; these can be accessed via the new quantum operations `Interferometer`, `GaussianTransform`, `Gaussian`. These allow you to apply interferometers, Gaussian symplectic transformations, and prepare a state based on a covariance matrix respectively. You can also query the engine to determine the CV gate decompositions applied.
* Added the cross-Kerr interaction, accessible via the quantum operation `CKgate()`.
* Added utilities for creating random covariance, symplectic, and Gaussian unitary matrices in `strawberryfields.utils`.
* States can now be compared directly for equality - this is defined separately for Gaussian states and Fock basis states.


### Improvements
* The engine logic and behaviour has been overhauled, making it simpler to use and understand.
    - `eng.run()` and `eng.reset()` now allow the user to alter parameters such as `cutoff_dim` between runs.
    - `eng.reset_backend()` has been renamed to `eng.reset()`, and now also implicitly resets the queue.
    - The engine can now be reset even in the case of modes having being added/deleted, with no side effects. This is due to the presence of register checkpoints, allowing the engine to keep track of register changes.
    - `eng.print_applied()` keeps track of multiple simulation runs, by using nested lists.
* A new parameter class is introduced - this is a developmental change, and does not affect the user-facing parts of Strawberry Fields. All parameters passed to quantum operations are 'wrapped' in this parameter class, which also contains several high level mathematical and array/tensor manipulation functions and methods.


### Contributors
This release contains contributions from:

Ville Bergholm, Christian Gogolin, Nicolás Quesada, Josh Izaac, and Nathan Killoran.


---


# Release 0.7.3

## New features
* Added Gaussian decompositions to the front-end; these can be accessed via the new quantum operations `Interferometer`, `GaussianTransform`, `CovarianceState`. These allow you to apply interferometers, Gaussian symplectic transformations, and prepare a state based on a covariance matrix respectively. You can also query the engine to determine the CV gate decompositions applied.
* Added utilities for creating random covariance, symplectic, and gaussian unitary matrices in `strawberryfields.utils`.

## Improvements
* Created a separate package `strawberryfields-gpu` that requires `tensorflow-gpu`.
* Modified TFBackend to cache non-variable parts of the beamsplitter, to speed up computation.
* Minor performance improvement in `fock_prob()` by avoiding inverting a matrix twice.

## Bug fixes
* Fixed bug #10 by adding the ability to reset the Fock modeMap and GaussianCircuit class
* Fixed bug #11 by reshaping the Fock probabilities if the state happens to be pure states
* Fixed Clements decomposition bug where some phase angles weren't applied
* Fixed typo in displaced squeezed formula in documentation
* Fix to prevent beamsplitter prefactor cache from breaking things if using two graphs
* Fix bug #13, GaussianBackend.state() raises an IndexError if all modes in the state have been deleted.

---

# Release 0.7.2

## Bug fixes
* Fixed Tensorflow requirements in `setup.py`, so that installation will now work for versions of tensorflow>=1.3,<1.7

## Known issues
* Tensorflow version 1.7 introduces some breaking API changes, so is currently not supported by Strawberry Fields.

---

# Release 0.7.1

Initial public release.

### Contributors
This release contains contributions from:

Nathan Killoran, Josh Izaac, Nicolás Quesada, Matthew Amy, and Ville Bergholm.

