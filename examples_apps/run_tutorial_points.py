# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports,invalid-name
r"""
.. _apps-points-tutorial:

Point processes
===============

*Technical details are available in the API documentation:* :doc:`/code/api/strawberryfields.apps.points`

This section shows how to generate GBS point process samples and use them to detect outlier
points in a data set. Point processes are models for generating random point patterns and GBS
devices can be programmed to operate as special types of point processes that generate clustered
random point patterns :cite:`jahangiri2019point`.

The probability of generating a specific pattern of points in GBS point processes depends on
matrix functions of a kernel matrix :math:`K` that describes the similarity between the points.
Matrix functions that appear in GBS point processes are typically
`permanents <https://en.wikipedia.org/wiki/Permanent_(mathematics)>`__ and
`hafnians <https://the-walrus.readthedocs.io/en/latest/hafnian.html>`__. Here we use
the permanental point process, in which the probability of observing a pattern of points :math:`S`
depends on the permanent of their corresponding kernel submatrix :math:`K_S` as
:cite:`jahangiri2019point`:

.. math::
    \mathcal{P}(S) = \frac{1}{\alpha(S)}\text{per}(K_S),

where :math:`\alpha` is a normalization function that depends on :math:`S` and the average number
of points. Let's look at a simple example to better understand the permanental point process.
"""

##############################################################################
# We first import the modules we need. Note that the :mod:`~.apps.points` module has most of
# the core functionalities exploring point processes.

import numpy as np
import plotly
from sklearn.datasets import make_blobs
from strawberryfields.apps import points, plot

##############################################################################
# We define a space where the GBS point process patterns are generated. This
# space is referred to as the state space and is defined by a set of points. The
# point process selects a subset of these points in each sample. Here we create
# a 20 :math:`\times` 20 square grid of points.

R = np.array([(i, j) for i in range(20) for j in range(20)])

##############################################################################
# The rows of R are the coordinates of the points.
#
# Next step is to create the kernel matrix for the points of this discrete space. We call
# the :func:`~.rbf_kernel` function which uses the *radial basis function* (RBF) kernel defined as:
#
# .. math::
#     K_{i,j} = e^{-\|\bf{r}_i-\bf{r}_j\|^2/2\sigma^2},
#
# where :math:`\bf{r}_i` are the coordinates of point :math:`i` and :math:`\sigma` is a kernel
# parameter that determines the scale of the kernel.
#
# In the RBF kernel, points that are much further than a distance :math:`\sigma` from each other
# lead to small entries of the kernel matrix, whereas points much closer than :math:`\sigma`
# generate large entries. Now consider a specific point pattern in which all points
# are close to each other, which simply means that their matrix elements have larger entries. The
# permanent of a matrix is a sum over the product of some matrix entries. Therefore,
# the submatrix that corresponds to those points has a large permanent and the probability of
# observing them in a sample is larger.
#
# For kernel matrices that are positive-semidefinite, such as the RBF kernel, there exist efficient
# quantum-inspired classical algorithms for permanental point process sampling
# :cite:`jahangiri2019point`. In this tutorial we use positive-semidefinite kernels and the
# quantum-inspired classical algorithm.
#
# Let's construct the RBF kernel with the parameter :math:`\sigma` set to 2.5.

K = points.rbf_kernel(R, 2.5)

##############################################################################
# We generate 10 samples with an average number of 50 points per sample by calling
# the :func:`~.points.sample` function of the :mod:`~.apps.points` module.

samples = points.sample(K, 50.0, 10)

##############################################################################
# We visualize the first sample by using the :func:`~.points` function of
# the :mod:`~.apps.plot` module. The point patterns generated by the permanental point process
# usually have a higher degree of clustering compared to a uniformly random pattern.

plot_1 = plot.points(R, samples[0], point_size=10)

plotly.offline.plot(plot_1, filename="Points.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_apps/Points.html
#
# .. note::
#     The command ``plotly.offline.plot()`` is used to display plots in the documentation. In
#     practice, you can simply use ``plot_1.show()`` to view your graph.

##############################################################################
# Outlier Detection
# -----------------
#
# When the distribution of points in a given space is inhomogeneous, GBS point processes
# sample points from the dense regions with higher probability. This feature of the GBS point
# processes can be used to detect outlier points in a data set. In this example, we create two
# dense clusters and place them in a two-dimensional space containing some randomly distributed
# points in the background. We consider the random background points as outliers to the clustered
# points and show that the permanental point process selects points from the dense clusters with
# a higher probability.
#
# We first create the data points. The clusters have 50 points each and the points have a
# standard deviation of 0.3. The clusters are centered at :math:`[x = 2, y = 2]` and :math:`[x = 4,
# y = 4]`, respectively. We also add 25 randomly generated points to the data set.

clusters = make_blobs(n_samples=100, centers=[[2, 2], [4, 4]], cluster_std=0.3)[0]

noise = np.random.rand(25, 2) * 6.0

R = np.concatenate((clusters, noise))

##############################################################################
# Then construct the kernel matrix and generate 10000 samples.

K = points.rbf_kernel(R, 1.0)

samples = points.sample(K, 10.0, 10000)

##############################################################################
# We obtain the indices of 100 points that appear most frequently in the permanental point
# process samples and visualize them. The majority of the commonly appearing points belong
# to the clusters and the points that do not appear frequently are the outlier points. Note that
# some of the background points might overlap with the clusters.

gbs_frequent_points = np.argsort(np.sum(samples, axis=0))[-100:]

plot_2 = plot.points(
    R, [1 if i in gbs_frequent_points else 0 for i in range(len(samples[0]))], point_size=10
)

plotly.offline.plot(plot_2, filename="Outliers.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_apps/Outliers.html

##############################################################################
# The two-dimensional examples considered here can be easily extended to higher dimensions. The
# GBS point processes retain their clustering property in higher dimensions but visual inspection
# of this clustering feature might not be very straightforward.
#
# GBS point processes can potentially be used in other applications such as clustering data points
# and finding correlations in time series data. Can you design your own example for using GBS point
# processes in a new application?
