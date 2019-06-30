# :trophy: Strawberry Fields Challenges :trophy:

Looking for inspiration for contributing to the Strawberry Fields community? Or just in it for the
Xanadu Swag™? Check out some ideas below to get started. If you get stuck, or have any questions,
simply ask over at our [Slack community](https://u.strawberryfields.ai/slack).

For more details on contributing to Strawberry Fields, see our
[contributing guidelines](.github/CONTRIBUTING.md).

## Community :strawberry:

* **Blog post/tutorial**
   - Write a blog post or tutorial exploring your experience with Strawberry Fields, highlighting
  any cool results you generate, or unique applications for Strawberry Fields. Send us the link
  when you're done at support@xanadu.ai, and we may even add it to the Strawberry Fields gallery.

## Quantum theory and research :mortar_board:

* **Publish results using Strawberry Fields**
   - If you've published a paper that utilizes Strawberry Fields for simulations, let us know,
   and we will add it to our Strawberry Fields gallery.

* **Contribute continuous-variable theory or algorithms to our documentation**
   - Our aim is for the [Strawberry Fields documentation](http://strawberryfields.readthedocs.io) to
   become an open-access online resource for everything continuous-variable. If you have any particular
   algorithms or theory you wish to include, simply chat to us on our Slack channel.

* **Write a non-physicist introduction to CV quantum computation**
  - Quantum computation is a tricky field to break into for newcomers. One thing we are currently
    missing in our documentation is a non-physicist introduction to CV quantum computation.
    That is, a page that introduces the idea of quantum computation with no equations (or very few equations),
    few references, lots of figures and pictures, and discusses more the application and potential than the theory.
    The main goal for this challenge would be to introduce continuous-variable quantum computing
    without having to first introduce the discrete-variable qubit model.

## Coding contributions :computer:

*Note*: please contact us, either via the [Slack channel](https://u.strawberryfields.ai/slack)  or at support@xanadu.ai,
to let us know if you want to start working on any coding challenges currently marked as *open*, as well as the
corresponding GitHub repository. We will update the status below with a link to your repository.

 * **Add common expectation values to the state module**
   - *Status: open*
   - Currently, Strawberry Fields supports the `state.quad_expectation()` method to calculate the quadrature expectation
     values x and p, as well as the quadrature variance \Delta x and \Delta p. In this challenge, modify the
     Strawberry Fields `states.py` module to calculate other common expectation values, such as <x^2>, <p^2>, and others.

 * **Find an accurate and numerically efficient implementation of the cubic phase gate**
   - *Status: open*
   - This challenge would involve
     finding a more accurate and numerically efficient method of implementing the cubic phase gate
     in the Fock and/or Tensorflow backends of Strawberry Fields.

 * **Create a new Strawberry Fields backend**
   - Strawberry Fields currently includes 3 backends; two Fock-basis backends (using NumPy and Tensorflow),
     and one Gaussian backend (using NumPy). Some ideas for a new backend include:
     - **Gaussian-Tensorflow**: (*status: open*) Duplicate the included Gaussian backend, and add Tensorflow support
     - **MPI support**: (*status: open*) Add MPI support (using mpi4py) to one of the backends
     ```
