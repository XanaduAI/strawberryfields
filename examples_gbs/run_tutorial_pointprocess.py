"""
Point Process Tutorial
======================

This tutorial shows how to generate GBS point process samples and use them to detect outlier
points in a data set. Point processes are models for generating random point patterns and GBS
devices can be programmed to operate as special types of point processes that generate clustered
point patterns. GBS point processes belong to a class of point processes in which the probability
of generating a specific pattern of points depends on matrix functions of a kernel matrix that
describes the similarity between the points. Matrix functions that appear in GBS point processes
are typically permanents and hafnians. In this tutorial we use the permanental point process in which
the probability of observing a pattern of points depends on the permanent of their corresponding
kernel matrix. 
"""

##############################################################################
# First we need to import the :mod:`~.gbs.points` and :mod:`~.gbs.plot` modules.

from strawberryfields.gbs import points, plot

##############################################################################
# We define a space in which the GBS point process patterns are generated. This
# space is referred to as the state space and is defined by a set of points. The
# point process selects a subset of these points in each sample. Here we create
# a discrete and homogeneous two-dimensional space containing 400 points as our
# state space. 

import numpy as np
R = np.array([(i, j) for i in range(20) for j in range(20)])

##############################################################################
# The kernel matrix for the points of this discrete space is constructed using 
# the :func:`~.kernel` function. The :func:`~.kernel` function uses the *radial
# basis function* (RBF) kernel:
#
# .. math::
#     K_{i,j} = e^{-\|\bf{r}_i-\bf{r}_j\|^2/2\sigma^2},
#
# where :math:`\bf{r}_i` are the coordinates of point :math:`i` and :math:`\sigma`
# is a kernel parameter that determines the scale of the kernel. Points that are much further
# than a distance :math:`\sigma` from each other lead to small entries of the kernel matrix,
# whereas points much closer than :math:`\sigma` generate large entries. The parameter
# :math:`\sigma` is set to 1.0 in this example. For kernel matrices that are 
# positive-semidefinite, there exist efficient quantum-inspired classical algorithms
# for permanental point process sampling :cite:`jahangiri2019point`. In this tutorial
# we restrict ourselves to such positive-semidefinite kernels and we use the quantum-inspired
# classical algorithm for generating the permanental point process samples.

K = points.kernel(R, 1.0)

##############################################################################
# We generate 10 samples with an average number of 50 points per sample by calling
# the :func:`~.sample` function of the :mod:`~.gbs.points` module.

samples = points.sample(K, 50.0, 10)

##############################################################################
# We visualize the first sample by using the :func:`~.plot_points` function of 
# the :mod:`~.gbs.plot` module.

import plotly
plotly.offline.plot(plot_points(R, samples[0]), filename="Points.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/Points.html

##############################################################################
# Outlier Detection
# -----------------
#
# When the distribution of points in a given space is inhomogenous, GBS point processes typically
# sample points from the dense regions. This feature can be used to detect high-density outlier 
# points in a data set. In this example, we create two dense clusters and place them in a 
# two-dimensional space containing evenly-distributed points. The GBS point process samples points 
# from the dense clusters, with a higher probability.
#
# We first create the data points.

from sklearn.datasets import make_blobs
R = np.concatenate((make_blobs(n_samples=50, centers=[[5, 5], [15, 15]], cluster_std=1)[0],
                    np.array([(i, j) for i in range(20) for j in range(20)])))

##############################################################################
# Then construct the kernel matrix.

K = points.kernel(R, 1.0)

##############################################################################
# And generate 500 samples.

samples = points.sample(K, 50.0, 500)

##############################################################################
# We obtain the indices of 50 points that appear most frequently in the GBS point
# process samples. 

gbs_frequent_points = np.argsort(np.sum(samples, axis=0))[-50:]

##############################################################################
# Then we visualize these most frequent points. Majority of the commonly appearing 
# points belong to the dense clusters.

plotly.offline.plot(plot_points(R, [1 if i in gbs_frequent_points else 0 for i in
                                    range(len(samples[0]))]), filename="Outliers.html")

##############################################################################
# .. raw:: html
#     :file: ../../examples_gbs/Outliers.html
