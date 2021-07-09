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

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2


class SymplecticOpt(optimizer_v2.OptimizerV2):
    r"""Symplectic Optimizer especially for the symplectic matrix of the Gaussian gate.
    """
    def __init__(self, learning_rate = 0.01, name = "SymplecticOpt", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        
    @tf.function
    def _resource_apply_dense(self, grad, var):
        """Perform one optimization step for one model variable
        """
        var_dtype = var.dtype.base_dtype
        lr_t = self.learning_rate

        grad_rieman = 0.5 * (grad - var@tf.transpose(grad)@var)
        new_var = tf.exp(tf.cast(lr_t, dtype = var_dtype) * grad_rieman @ tf.transpose(var))@var
        var.assign(new_var)

    def _resource_apply_sparse(self, grad, var):
        """Not implement for now
        """
        raise NotImplementedError

    def get_config(self):
        """Show the configuration of the optimizer
        """
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
        }
