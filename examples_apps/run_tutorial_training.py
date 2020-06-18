"""
.. _apps-clique-tutorial:

Training variational GBS distributions
======================================

*Technical details are available in the API documentation:*
:doc:`/code/api/strawberryfields.apps.train`

Many quantum algorithms rely on the ability to train the parameters of quantum circuits, a
strategy inspired by the success of neural networks in machine learning. Training is often
performed by evaluating gradients of a cost function with respect to circuit parameters,
then employing gradient-based optimization methods. In this demonstration, we outline the
theoretical principles for training Gaussian Boson Sampling (GBS) circuits, which were first
introduced in Ref.~[insert ref]. We then explain how to employ the Strawberry Fields Apps to
perform the training by looking at basic examples in stochastic optimization and unsupervised
learning. Let's go! 🚀

Theory
------

As explained in more detail in [insert link], for a GBS device, the probability :math:`\Pr(S)` of
observing an output :math:`S=(s_1, s_2, \ldots, s_m)`, where :math:`s_i` denotes the number of
photons detected in the :math:`i`-th mode device, can be expressed as

.. math::
    \Pr(S) = \frac{1}{\mathcal{N}} \frac{|\text{Haf}(A_S)|^2}{
    s_1!\ldots s_m!},

where :math:`A` is an arbitrary symmetric matrix with eigenvalues bounded between
:math:`-1` and :math:`1` and :math:`\mathcal{N}` is a normalization constant. The matrix
:math:`A` can also be rescaled by a constant factor, which is equivalent to fixing a total
mean photon number in the distribution.

Now, we want to *train* this distribution to perform a specific task. For example, we may want to
reproduce the statistical properties of a given dataset to generate new data, or optimize the
circuit to sample specific patterns with high probability. The usual strategy is to identify a
parametrization of the distribution and compute gradients of a suitable cost function with
respect to trainable parameters. These can then be used to optimize the parameters using
gradient-base techniques. We refer to these as variational GBS circuits, or VGBS for short.

So the staring point is a parametrization of the distribution. Gradient formulas can be
generally challenging to calculate, but there exists a particular strategy that leads to
gradients that are simpler to compute. Known as the WAW (wow!) parametrization, it involves
transforming the symmetric matrix :math:`A` as

.. math::
    A \rightarrow A_W = W A W,

where :math:`W = \text{diag}(w_1, w_2, \ldots, w_m)` is a diagonal *weight* matrix. The
usefulness of this parametrization is that the hafnian of :math:`A_W` factorizes into two
separate components

.. math::
    \text{Haf}(A_W) \rightarrow A_W = \text{Haf}(A)\text{det}(W),

a property that can be cleverly exploited to compute gradients more efficiently. More broadly,
it is convenient to embed trainable parameters :math:`\theta = (\theta_1, \ldots, \theta_d)` into
the weights :math:`w_k` in the diagonal of :math:`W`. Several choices are possible in principle,
but here we focus on an exponential embedding

.. math::
    w_k = \exp(-\theta^T f^{(k)}),

where each :math:`f^{(k)` is a :math:`d`-dimensional vector. The simplest case occurs when we
set :math:`d=m` and choose these vectors such that :math:`\theta^T f^{(k)} = theta_k` such that
:math:`w_k = \exp(-\theta_k)`.

In stochastic optimization, we are given a function :math:`h(S)` and the goal is optimize the 
parameters to sample from a distribution :math:`P_{\theta}(S)` that minimizes the 
expectation value 
.. math::
        C (\theta) = \sum_{S} h(S) P_{\theta}(S).
        
As shown in [insert ref], the gradient of the cost function :math:`C (\theta)` is given by 

.. math::
\partial_{\theta} C (\theta) = \sum_{S} h(S) P_{\theta}(S)
            \sum_{k=1}^{m}  (s_k - \langle s_{k} \rangle) \partial_{\theta} \log w_{k},

:math:`\langle s_k\rangle` denotes the average photon numbers in mode *k*. This gradient is an
expectation value with respect to the GBS distribution, so it can be estimated by generating
samples from the device.

In a standard unsupervised learning scenario, data are assumed to be sampled from an unknown
distribution and a common goal is to learn that distribution. Training can be performed by
minimizing the Kullback-Leibler (KL) divergence, which up to additive constants can be written as:

    .. math::

        KL = -\frac{1}{T}\sum_S \log[P(S)],

where :math:`S` is an element of the data, :math:`P(S)` is the probability of observing that
element when sampling from the GBS distribution, and :math:`T` is the total number of elements
in the data. For the GBS distribution in the WAW parametrization, the gradient of the KL
divergence can be written as

    .. math::

        \partial_\theta KL(\theta) = - \sum_{k=1}^m\frac{1}{w_k}(\langle s_k\rangle_{\text{data}}-
        \langle s_k\rangle_{\text{GBS}})\partial_\theta w_k,

where :math:`\langle s_k\rangle` denotes the average photon numbers in mode *k*. Remarkably,
this gradient can be evaluated without a quantum computer since for GBS the expectation values
:math:`\langle s_k\rangle_{\text{GBS}})` can be efficiently computed classically. As we'll see
soon, this leads to fast training. This is true even if sampling the distribution remains
classically intractable 🤯!

Stochastic optimization
-----------------------
We're ready to start using Strawberry Fields to train GBS distributions. The main functions we'll
need can be found in the :mod:`~.apps.train` module, so let's start by importing it. We'll also
use pre-generated GBS samples and graphs from the :mod:`~.apps.data` module
"""

import strawberryfields as sf
from strawberryfields.apps import train, data

##############################################################################
# We look at a basic example where the goal is to optimize the distribution to favour photons
# appearing in a specific subset of modes, while minimizing the number of photons in the
# remaining modes. This can be achieved with the following cost function

import numpy as np


def h(s):
    x = np.array(s)
    modes = np.arange(len(x))
    not_subset = [k for k in modes if k not in subset]
    return np.sum(x[not_subset]) - np.sum(x[subset])


##############################################################################
# The cost function is defined with respect to a subset of modes for which we want to observe
# many photons. This subset can be specified later on. Then, for a given sample ``s``,
# we want the total number of photons in the subset to be large, which we can achieve by minimizing
# its negative value. Similarly, for modes outside of the specified subset, we want to minimize
# their total sum. Now time to define the variational circuit. We'll train a distribution based on
# on a simple lollipop 🍭 graph with five vertices:

import networkx as nx
import plotly
from strawberryfields.apps import plot

graph = nx.lollipop_graph(3, 2)
A = nx.to_numpy_array(graph)
plot_graph = plot.graph(graph)
# plotly.offline.plot(plot_graph, filename="mutag1.html")

##############################################################################
# Defining a variational GBS circuit consists of three steps: (i) specifying the embedding,
# (ii) building the circuit, (iii) defining the cost function with respect to the circuit and
# embedding. We'll go through each step one at a time. For the embedding of trainable parameters,
# we'll use the simple form :math:`w_k = \exp(-\theta_k)` outlined above, which can be accessed
# through the ``Exp()`` method in the :mod:`~.apps.train.embed` submodule. Its only input is the
# number of modes in the device, which is equal to the number of vertices in the graph.

nr_modes = len(A)
weights = train.embed.Exp(nr_modes)

##############################################################################
# Easy! The GBS distribution is determined by the symmetric matrix :math:`A`, which we
# train using the WAW parametrization, and by the total mean photon number. In principle there is
# freedom in choosing :math:`A`, but here we'll just use the graph's adjacency matrix. The total
# mean photon number is a hyperparameter of the distribution: in general, different choices may
# lead to different results in training. Finally, besides the embedding, GBS devices can operate
# either with photon number-resolving detectors or threshold detectors, so there is an option to
# specify which one we intend to use.

n_mean = 6
vgbs = train.VGBS(A, n_mean, weights, threshold=False)

##############################################################################
# The last step before commencing training is to define the cost function with respect to our
# previous choices. Since this is a stochastic optimization task, we employ the
# ``Stochastic`` class and input our previously defined cost function ``h``:

cost = train.Stochastic(h, vgbs)

##############################################################################
# During training, we'll calculate gradients with respect to this cost function and evaluate it
# to keep track of progress. Both these actions require estimating expectation values from
# sampling, so the number of samples in the estimation also needs to be specified. The gradient and
# cost function are ultimately determined by the values of the parameters, which need to be
# initialized. There is freedom in this choice, but here we'll set them all to zero. Finally, we
# aim to increase the number of photons in the "candy" part of the lollippop graph,
# which corresponds to the subset of modes ``[0, 1, 2]``. Let's see how this is done:

np.random.seed(1969)  # for reproducibility of results
d = nr_modes
params = np.zeros(d)
subset = [0, 1, 2]
nr_samples = 100

print('Initial mean photon numbers = ', vgbs.mean_photons_by_mode(params))

##############################################################################
# If training is successful, we should see the mean photon numbers of the first three modes
# increasing, while those of the last two modes should be close to zero. We perform training over
# 200 steps of gradient descent with a learning rate of 0.01:

nr_steps = 200
rate = 0.01

for i in range(nr_steps):
    params -= rate*cost.gradient(params, nr_samples)
    if (i + 1) % 10 == 0:
        print('Cost = {:.3f}'.format(cost.evaluate(params, nr_samples)))

print('Final mean photon numbers = ', vgbs.mean_photons_by_mode(params))

##############################################################################
# Great! The cost function decreases smoothly and there is a clear increase in the mean photon
# numbers of the target modes, with a corresponding decrease in the remaining modes. Because the
# transformed matrix :math:`A_W = W A W` also needs to have eigenvalues bounded between -1 and 1,
# continuing training indefinitely can lead to unphysical distributions when weights become too
# large, so it's important to monitor this behaviour.
