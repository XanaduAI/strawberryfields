# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports
"""
.. _gbs-vibronic-tutorial:

Vibronic Spectra Tutorial
=========================
In this tutorial, we study how GBS can be used to compute vibronic spectra. So let's start from
the beginning: what does vibronic spectra mean?  The
frequencies at which light is more strongly absorbed determine many important properties of a
molecule. Molecules absorb light at frequencies that
depend on the allowed energy transitions between different molecular configurations. These
transitions can be determined by the configuration of the electrons in the molecule and also by its
vibrational degrees of freedom, in which case the absorption lines are referred to as the
vibronic spectrum.

It is possible to determine vibronic spectra by running clever and careful experiments.
However, this can be slow and expensive, in which case it is valuable to predict vibronic spectra
using theoretical calculations. To model vibronic transitions, it is common to focus on only a few
relevant parameters, which we list below:

1. :math:`\Omega`: the frequencies of the vibrational modes of the electronic ground state.
2. :math:`\Omega'`: the frequencies of the vibrational modes of the electronic excited state.
3. :math:`U_\text{D}`: the Duschinsky matrix.
4. :math:`d`: the displacement vector.
5. :math:`T`: the temperature.

The vibrational mode frequencies determine the energies of the different molecular
configurations. The Duschinsky matrix and displacement vector encode information regarding how
vibrational modes are transformed when the molecule changes from a ground to excited electronic
state. At zero temperature, all initial modes are in the vibrational ground state. At finite
temperature, other vibrational states are also populated.

In the GBS algorithm for computing vibronic spectra :cite:`Huh_2015`, these parameters
are sufficient to determine the configuration of a GBS device. As opposed to other applications
that involve only single-mode squeezing and linear interferometry, for vibronic spectra we
prepare a Gaussian state using the following sequence of operations: two-mode squeezing, linear
interferometry, single-mode squeezing, linear interferometry, and displacements.

The module :mod:`~.gbs.vibronic` contains the function :func:`~.gbs_params`, which can be used to
obtain the GBS squeezing, interferometer, and displacement parameters from the input chemical
parameters listed above. In this tutorial, we study the vibronic spectrum of
`formic acid <https://en.wikipedia.org/wiki/Formic_acid>`_, which can be well approximated by
looking at only a few vibrational modes. Its chemical parameters, obtained from
:cite:`Huh_2015`, are listed below. They are then mapped to GBS parameters using the function
:func:`~.gbs_params:
"""
from strawberryfields.gbs import vibronic
import numpy as np
# ground state frequencies
w = np.array([3765.2386, 3088.1826, 1825.1799, 1416.9512, 1326.4684, 1137.0490, 629.7144])
# excited state frequencies
wp = np.array([3629.9472, 3064.9143, 1566.4602, 1399.6554, 1215.3421, 1190.9077, 496.2845])
# Duschinsky matrix
Ud = np.array(
     [
         [0.9934, 0.0144, 0.0153, 0.0268, 0.0638, 0.0751, -0.0428],
         [-0.0149, 0.9931, 0.0742, 0.0769, -0.0361, -0.0025, 0.0173],
         [-0.0119, -0.0916, 0.8423, 0.1799, -0.3857, 0.3074, 0.0801],
         [0.0381, 0.0409, -0.3403, -0.5231, -0.6679, 0.3848, 0.1142],
         [-0.0413, -0.0342, -0.4004, 0.7636, -0.1036, 0.4838, 0.0941],
         [0.0908, -0.0418, -0.0907, 0.3151, -0.5900, -0.7193, 0.1304],
         [-0.0325, 0.0050, -0.0206, 0.0694, -0.2018, 0.0173, -0.9759],
     ]
 )
# displacement vector
d = np.array([0.2254, 0.1469, 1.5599, -0.3784, 0.4553, -0.3439, 0.0618])
T = 0  # temperature

t, U1, r, U2, alpha = vibronic.gbs_params(w, wp, Ud, d, T)  # GBS parameters

##############################################################################
# It is important to note that since two-mode squeezing operators are involved, if we have :math:`N`
# input modes, the Gaussian state prepared is a :math:`2N`-mode Gaussian state, and the samples
# are vectors of length :math:`2N`. The first :math:`N` modes are those of the excited electronic
# state, the final :math:`N` modes those of the ground state. From above, :math:`t` is a vector
# of two-mode squeezing parameters, :math:`U1` and :math:`U2` are
# the # interferometer unitaries, :math:`r` is a vector of single-mode squeezing parameters,
# and `alpha` is a vector of displacements. Since we set the temperature to zero, the two-mode
# squeezing parameters, should all be equal to zero. Let's confirm this is the case:

print(t)

##############################################################################
# Great! 😁
# Photons detected at the output of the GBS device correspond to a specific transition energy.
# The GBS algorithm for computing vibronic spectra works because the device is programmed in such
# a way that energies sampled with high probability correspond to peaks of the vibronic spectrum.
# The function :func:`~.energies` can be used to compute the energies for a set of samples:

samples = [[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
e = vibronic.energies(samples, w, wp)
print(np.around(e, 4))  # precision of 4 decimals

##############################################################################
# Once the GBS parameters have been obtained, it is straightforward to run the GBS algorithm: we
# perform many samples, compute their energies, and make a histogram of the most observed
# energies. The :mod:`~.gbs.sample` module contains the function :func:`~.vibronic` that is tailored
# for use in vibronic spectra applications. Similarly, the :mod:`~.gbs.plot` module includes a
# :func:`~.spectrum` that generates the vibronic spectrum from the GBS samples. Let's take a look
# at how this is done for just a few samples:

from strawberryfields.gbs import sample, plot
nr_samples = 100
s = sample.gaussian(U1, r, U2, alpha, 100)
e = vibronic.energies(s, w, wp)
# plot.spectrum(e)

##############################################################################
# The bars in the plot are the histogram of energies, while the curve surrounding them is a
# Lorentzian broadening of the spectrum, which better represents what is actually observed in an
# experiment. Of course, 💯 samples are not quite enough to accurately reconstruct the
# vibronic spectrum. The :mod:`~.gbs.data` module contains a large number pre-generated samples,
# which we can use to compute a more accurate spectrum:

from strawberryfields.gbs import data
fa_samples = data.Formic()
e = vibronic.energies(fa_samples, w, wp)
# plot.spectrum(e)

##############################################################################
# Let's compare with a plot of the actual experimental spectrum of formic acid, taken from
#:cite:`Huh_2015`


##############################################################################
# An excellent match! Formic acid is a sufficiently small molecule that its vibronic spectrum
# can be computed using classical computers. However, for larger molecules, this task quickly
# becomes intractable, for much the same reason that simulating GBS cannot be done efficiently with
# classical devices. Photonic quantum computing therefore holds the potential to enable new
# computational capabilities in this area of quantum chemistry ⚛️.