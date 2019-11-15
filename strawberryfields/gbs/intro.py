# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Overview
========

**Module name:** :mod:`strawberryfields.gbs.intro`

.. currentmodule:: strawberryfields.gbs.intro

Gaussian Boson Sampling (GBS) is a special-purpose model of photonic quantum computation, first
introduced in Ref. :cite:`hamilton2017`. In its most general form, GBS consists of preparing a
multi-mode Gaussian state and measuring it in the Fock basis. It differs from universal photonic
circuits simply because it does not employ non-Gaussian gates and it restricts measurements to the
Fock basis. Nevertheless, in general, the output distribution of a GBS device cannot be simulated
in polynomial time with classical computers.

After its inception, it was quickly realized that GBS offers significant versatility in the scope of
problems that can be encoded into the device. This led to the appearance of several GBS
algorithms with applications to quantum chemistry~\cite{huh2017vibronic}, graph optimization
~\cite{bradler2018gaussian, arrazola2018quantum, arrazola2018using}, molecular docking~\cite{
banchi2019molecular}, graph similarity~\cite{bradler2018graph,schuld2019quantum,
bradler2019duality}, and point processes~\cite{jahangiri2019point}. The GBS applications layer of
Strawberry Fieldsc is built with the goal of providing users with the capability to implement GBS
algorithms using only a few lines of code. Programming the GBS device, generating samples,
and classical post-processing of the outputs are taken care of automatically by built-in functions.

Gaussian Boson Sampling
=======================
A general pure Gaussian state can be prepared from a vacuum state by a sequence of single-mode
squeezing, multimode linear interferometry, and single-mode displacements. It was shown in Ref.
:cite:`{hamilton2017}` that for a Gaussian state with zero mean -- which can be prepared using
only squeezing followed by linear interferometry -- the probability :math:`\Pr(S)` of observing
an output :math:`S=(s_1, s_2, \ldots, s_m)`, where :math:`s_i` denotes the number of photons
detected in the :math:`i`-th mode, is given by

.. math::
\Pr(S) = \frac{1}{\sqrt{\text{det}(\mathbf{Q})} } \frac{\text{Haf}(\mathbf{\mathcal{A}}_S)}{s_1!\ldots s_m!},

where

.. math::
\begin{align}
\mathbf{Q}&:=\mathbf{\Sigma} +I/2\\
\mathbf{\mathcal{A}} &:= \mathbf{X} \left(I- \mathbf{Q}^{-1}\right),\\
\mathbf{X} &:=  \left[\begin{smallmatrix}
	0 &  I \\
	I & 0
\end{smallmatrix} \right],
\end{align}

and :math:`\mathbf{\Sigma}` is the covariance matrix of the Gaussian state. The matrix function
:math:`\text{Haf}(\cdot)` is the `hafnian <https://hafnian.readthedocs.io/en/stable/hafnian.html>`_.
When the state is pure, the matrix :math:`\mathbf{\mathcal{A}}` can be written as
:math:`\mathbf{\mathcal{A}}=\mathbf{A}\oplus \mathbf{A}^*` and the distribution becomes

.. math::
\Pr(S) = \frac{1}{\sqrt{\text{det}(\mathbf{Q})} } \frac{|\text{Haf}(\mathbf{A}_S)|^2}{
s_1!\ldots s_m!},

where :math:`\mathbm{A}` is an arbitrary symmetric matrix with eigenvalues bounded between -1 and 1.
Therefore, besides encoding Gaussian states and gates into a GBS device, it is also possible
to encode symmetric matrices :math:`\mathbm{A}`. Notably, these may include adjacency matrices of
graphs and kernel matrices.

Programming a GBS device
------------------------
In GBS without displacements, specifying gate parameters is equivalent to specifying the symmetric
matrix :math:`\mathbf{A}`. Employing the Takagi-Autonne decomposition, we can write

.. math::
\mathbf{A} = \mathbf{U} \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_m) \mathbf{U}^T,

where :math:`\mathbf{U}` is a unitary matrix. The matrix :math:`\mathbf{U}` is precisely the unitary
operation that specifies the linear interferometer of a GBS device. The values :math:`0\leq
\lambda_i < 1` uniquely determine the squeezing parameters :math:`r_i` via the relation
:math:`\tanh (r_i) = \lambda_i`, as well as the mean photon number :math:`\bar{n}` of the
distribution from the expression

.. math::
\bar{n} = \sum_{i=1}^M \frac{\lambda_i^2}{1-\lambda_i^2}.


It is possible to encode an arbitrary symmetric matrix :math:`\mathbf{A}` into a GBS device by
rescaling the matrix with a parameter :math:`c>0` so that :math:`cA` satisfies :math:`0\leq
\lambda_i < 1` as in the above decomposition. The parameter :math:`c` controls the squeezing
parameters :math:`r_i` and the mean photon number. Overall, a GBS device can be programmed as
follows:

#. Compute the Takagi-Autonne decomposition of :math:`\mathbf{A}` to determine the unitary
:math:`\mathbf{U}` and the values :math:`\lambda_1, \lambda_2, \ldots, \lambda_m`.
#. Program the linear interferometer according to the unitary :math:`\mathbf{U}`.
#. Solve for the constant :math:`c>0` such that
:math:`\bar{n} = \sum_{i=1}^M \frac{(c\lambda_i)^2}{1-(c\lambda_i)^2}`.
#. Program the squeezing parameter :math:`r_i` of the squeezing gate :math:`S(r_i)` acting on the
:math:`i`-th mode as :math:`r_i=\tanh^{-1}(c\lambda_i)`.

The GBS device then samples from the distribution

.. math::
\Pr(S) \propto c^{k} \frac{|\text{Haf}(\mathbf{A}_S)|^2}{s_1!\ldots s_m!},

with :math:`k = \sum_{i}s_{i}`.

The GBS applications layer includes functions for sampling from GBS devices that are programmed
in this manner. Such samples can then be post-processed by classical techniques to give rise to a
wide range of GBS algorithms, each of which has a dedicated module.
"""
