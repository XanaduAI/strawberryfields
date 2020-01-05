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
"""Tensorflow states development module"""

import numpy as np
import tensorflow as tf

from strawberryfields.backends.states import BaseFockState

class FockStateTF2batched(BaseFockState):
    """
    #TODO
    """
    def __init__(self, state_data, num_modes, pure, cutoff_dim, mode_names=None):
        #TODO: autobatching: for now we assume state_data is always
        # given with a batch dimension, but it could be not so.
        state_data = tf.convert_to_tensor(state_data, name="state_data")
        super().__init__(state_data, num_modes, pure, cutoff_dim, mode_names)

    @property
    def ket(self):
        return self._data if self.is_pure else None

    @property
    def dm(self):
        if self.is_pure:
            flatket = tf.reshape(self.ket, [-1, self.cutoff_dim**self.num_modes])
            batched_outer = tf.einsum('bj,bk->bjk', flatket, tf.math.conj(flatket))
            return self.zip_indices(tf.reshape(batched_outer, [-1] + [self.cutoff_dim]*2*self.num_modes))
        else:
            return self._data

    def trace(self):
        if self.is_pure:
            return tf.math.reduce_euclidean_norm(self.ket, axis=[-n for n in range(1, self.num_modes+1)])**2
        else:
            tr = self.dm
            for _ in range(self.num_modes):
                tr = tf.linalg.trace(tr)
            return tr

    def fock_prob(self, n):
        if self.is_pure:
            return (self.ket.__getitem__([None]+n)**2)[[0]]
        else:
            return (self.dm.__getitem__([None]+[x for pair in zip(n, n) for x in pair]))[[0]]

    def all_fock_probs(self):
        if self.is_pure:
            return tf.abs(self.ket) ** 2
        else:
            return tf.vectorized_map(tf.linalg.tensor_diag_part, self.unzip_indices(self.dm)) # TF2 bug (https://github.com/tensorflow/tensorflow/issues/34734)

    def reduced_dm(self, modes):
        if isinstance(modes, int): # or numbers.Integral?
            modes = [modes]
        keep = [i for m in modes for i in (2*m+1, 2*m+2)]
        perm = [0] + keep + list(set(range(1,2*self.num_modes+1)) - set(keep))

        out = self.dm
        while len(out.shape) > 2*len(modes) + 1:
            out = tf.linalg.trace(out)
        return out

    def mean_photon(self, modes):

        avg = self.reduced_dm(modes)
        for k in avg.shape[1:]:
            avg = avg@tf.range(k)

        avg2 = self.reduced_dm(modes) #TODO this is wasteful, make a copy instead
        for k in avg2.shape[1:]:
            avg2 = avg2@(tf.range(k)**2)

        return avg, avg2 - avg**2


    def quad_expectation(self, mode, phi=0.):
        # TODO: account for batch!

        N = np.arange(self.cutoff_dim)
        a = np.sqrt(np.arange(1, self.cutoff_dim))
        a2 = np.sqrt(np.arange(1, self.cutoff_dim-1)*np.arange(2, self.cutoff_dim))
        phi_arr = np.exp(1j*N*phi)

        rho = phi_arr[None, :]*self.reduced_dm(mode)*np.conj(phi_arr[:, None])
        avg = tf.math.real(tf.linalg.diag_part(rho, k=1))@a
        avg2 = (1 + 2*N@tf.math.real(tf.linalg.diag_part(rho)) + 2*a2@tf.math.real(tf.linalg.diag_part(rho, k=2)))/4

        return avg, avg2 - avg**2

    def fidelity(self, other_state, mode):
        return tf.vectorized_map(lambda x: tf.reduce_sum(tf.transpose(other_state)*x), self.reduced_dm(mode))


    def fidelity_coherent(self, alpha_list):
        #TODO: check *alpha_matrix and not the transpose
        alpha_matrix = (alpha_list**np.arange(self.cutoff_dim)[:, None])*np.exp(-np.abs(alpha_list)**2)[:, None]
        if self.is_pure:
            return tf.vectorized_map(lambda ket: tf.einsum(ket, *alpha_matrix, np.arange(self.num_modes), *[[n] for n in range(1, self.num_modes+1)])**2, self.ket)
        else:
            return tf.vectorized_map(lambda rho: tf.einsum(rho, *alpha_matrix, *alpha_matrix, np.arange(self.num_modes), *[[n] for n in range(1, self.num_modes+1)], *[[n] for n in range(1, self.num_modes+1)]), self.dm)

    def fidelity_vacuum(self):
        if self.is_pure:
            return tf.abs(self.ket[:,0])**2
        else:
            return tf.abs(self.dm[:,0,0])

    def is_vacuum(self, tol):
        fidel = self.fidelity_vacuum()
        return np.isclose(fidel, [1]*self.num_modes, atol=tol)


    def poly_quad_expectation(self, A, d=None, k=0, phi=0, **kwargs): # pragma: no cover
        pass


    def wigner(self, mode, xvec, pvec):
        pass

    # UTILITIES
    def zip_indices(self, t):
        """
        Assuming a first batched dimension, given a tensor of shape [b, D1a, ..., Dna, D1b, ..., Dnb]
        returns the transposed tensor of shape [b, D1a, D1b, ..., Dna, Dnb]
        """
        nmodes = len(t.shape)//2
        return tf.transpose(t, [0]+[x for n in range(1, nmodes+1) for x in (n, n+nmodes)])

    def unzip_indices(self, t):
        """
        Assuming a first batched dimension, given a tensor of shape [b, D1a, D1b, ..., Dna, Dnb]
        returns the transposed tensor of shape [b, D1a, ..., Dna, D1b, ..., Dnb]
        """
        nmodes = len(t.shape)//2
        return tf.transpose(t, [0]+list(np.arange(1, 2*nmodes+1, 2))+list(np.arange(2, 2*nmodes+2, 2)))