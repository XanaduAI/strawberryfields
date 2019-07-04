#  Release 0.11.dev

### New features

- The functionality of the `Engine` class has been divided into two new classes: `Program`, which represents a quantum circuit or a fragment thereof, and `Engine`, which executes `Program` instances.
- Introduced the `BaseEngine` abstract base class and the `LocalEngine` child class. `Engine` is kept as an alias for `LocalEngine`.
- The Engine API has been changed slightly:
    - `LocalEngine.run()` returns a `Result` object that contains both a state object and measurement samples.
    - The way kwargs were used has been simplified by introducing the new kwargs-like arguments `backend_options` and `state_options`. `LocalEngine.run()` passes the actual kwargs only to `Operation.apply()`.
- The Gaussian backend now officially supports Fock-basis measurements (`MeasureFock`/`Measure`/`measure_fock`), but does not update the quantum state.
- New `shots` keyword argument added to `Engine.run()`, enabling multi-shot sampling. Supported only in the Gaussian backend, and only for Fock measurements.
- Added the ability to compile quantum programs to match a desired circuit target.
- Included a number of compilation targets, including Gaussian Boson Sampling circuits.
- Added a frontend validation database, the `devicespecs` submodule, for validating that quantum programs can be executed on certain backends and providing compilation methods.
- Added the `sf.io` module, which is used to save/load standalone Blackbird scripts from/into Strawberry Fields. Note that the Blackbird DSL has been spun off as an independent package and is now a dependency of Strawberry Fields.
- Added a new decomposition `mach_zehnder` to the decompositions module. 
- Added a `Configuration` class, which is used to load, store, save, and modify configuration options for Strawberry Fields.
- The way hbar is handled has been simplified:
    - The backend API is now entirely hbar-independent, i.e., every backend API method is defined in terms of a and a^\dagger only, not x and p.
    - The backends always explicitly use `hbar=2` internally.
    - `hbar` is now a global, frontend-only variable that the user can set at the beginning of the session. It is used at the `Operation.apply()` level to scale the inputs and outputs of the backend API calls as needed, and inside the `State` objects.
    - The only backend API calls that need to do hbar scaling for the input parameters are the X, Z, and V gates, the Gaussian state decomposition, and homodyne measurements (both the returned value and postselection argument are scaled).
- Added the ability to generate random real (orthogonal) interferometers and random block diagonal symplectic and covariance matrices.
- Added two top-level functions:
    - `about()`, which prints human-readable system info including installed versions of various Python packages.
    - `cite()`, which prints a bibtex citation for SF.
- Added a glossary to the documentation


### Improvements

- Removed TensorFlow as an explicit dependency of Strawberry Fields. This was causing complicated dependency issues due to version mismatches between TensorFlow, Strawberry Fields, and Python 3, which made installing difficult for new users. Advanced users can still install TensorFlow manually using `pip install tensorflow==1.3` and use as before.
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

