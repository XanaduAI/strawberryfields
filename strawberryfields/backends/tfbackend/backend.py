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

"""
Tensorflow backend interface
============================

"""
# pylint: disable=too-many-public-methods
import numpy as np
import tensorflow as tf

from strawberryfields.backends import BaseFock, ModeMap
from .circuit import Circuit
from .ops import _check_for_eval, mixed, partial_trace, reorder_modes
from .states import FockStateTF


class TFBackend(BaseFock):
    r"""Implements a simulation of quantum optical circuits in a truncated
    Fock basis using `TensorFlow <http://www.numpy.org/>`_, returning a :class:`~.FockStateTF`
    state object.
    """

    short_name = "tf"
    circuit_spec = "tf"

    def __init__(self, graph=None):
        """Initialize a TFBackend object.

        Args:
            graph (tf.Graph): optional Tensorflow Graph object where circuit should be defined
        """
        super().__init__()
        self._supported["mixed_states"] = True
        self._supported["batched"] = True
        self._supported["symbolic"] = True
        if graph is None:
            self._graph = tf.get_default_graph()
        else:
            self._graph = graph
        self._init_modes = None  #: int: initial number of modes in the circuit
        self._modemap = None  #: Modemap: maps external mode indices to internal ones
        self.circuit = (
            None  #: ~.tfbackend.circuit.Circuit: representation of the simulated quantum state
        )

    def _remap_modes(self, modes):
        if isinstance(modes, int):
            modes = [modes]
            was_int = True
        else:
            was_int = False
        map_ = self._modemap.show()
        submap = [map_[m] for m in modes]
        if not self._modemap.valid(modes) or None in submap:
            raise ValueError("The specified modes are not valid.")
        remapped_modes = self._modemap.remap(modes)

        if was_int:
            remapped_modes = remapped_modes[0]
        return remapped_modes

    def begin_circuit(self, num_subsystems, **kwargs):
        r"""Instantiate a quantum circuit.

        Instantiates a representation of a quantum optical state with ``num_subsystems`` modes.
        The state is initialized to vacuum.

        The modes in the circuit are indexed sequentially using integers, starting from zero.
        Once an index is assigned to a mode, it can never be re-assigned to another mode.
        If the mode is deleted its index becomes invalid.
        An operation acting on an invalid or unassigned mode index raises an ``IndexError`` exception.

        Args:
            num_subsystems (int): number of modes in the circuit

        Keyword Args:
            cutoff_dim (int): Numerical Hilbert space cutoff dimension for the modes.
                For each mode, the simulator can represent the Fock states :math:`\ket{0}, \ket{1}, \ldots, \ket{\text{cutoff_dim}-1}`.
            pure (bool): If True (default), use a pure state representation (otherwise will use a mixed state representation).
            batch_size (None or int): Size of the batch-axis dimension. If None, no batch-axis will be used.
        """
        cutoff_dim = kwargs.get("cutoff_dim", None)
        pure = kwargs.get("pure", True)
        batch_size = kwargs.get("batch_size", None)

        if cutoff_dim is None:
            raise ValueError("Argument 'cutoff_dim' must be passed to the Tensorflow backend")

        if not isinstance(num_subsystems, int):
            raise ValueError("Argument 'num_subsystems' must be a positive integer")
        if not isinstance(cutoff_dim, int):
            raise ValueError("Argument 'cutoff_dim' must be a positive integer")
        if not isinstance(pure, bool):
            raise ValueError("Argument 'pure' must be either True or False")
        if batch_size == 1:
            raise ValueError(
                "batch_size of 1 not supported, please use different batch_size or set batch_size=None"
            )

        with tf.name_scope("Begin_circuit"):
            self._modemap = ModeMap(num_subsystems)
            circuit = Circuit(self._graph, num_subsystems, cutoff_dim, pure, batch_size)

        self._init_modes = num_subsystems
        self.circuit = circuit

    def reset(self, pure=True, **kwargs):
        """Reset the circuit so that all the modes are in the vacuum state.

        After the reset the circuit is in the same state as it was after
        the last :meth:`begin_circuit` call. It will have the original number
        of modes, all initialized in the vacuum state. Some circuit parameters
        may be changed during the reset, see the keyword args below.

        Args:
            pure (bool): if True, initialize the circuit in a pure state representation
                (will use a mixed state representation if pure is False)

        Keyword Args:
            cutoff_dim (int): new Hilbert space truncation dimension
            hard (bool): Whether to reset the underlying TensorFlow graph.
                If True (default), then resets the underlying tensor graph as well.
                If False, then the circuit is reset to its initial state, but ops that
                have already been declared are still accessible.
        """
        hard = kwargs.pop("hard", True)
        if hard:
            tf.reset_default_graph()
            self._graph = tf.get_default_graph()

        with tf.name_scope("Reset"):
            self._modemap.reset()
            self.circuit.reset(
                graph=self._graph, num_subsystems=self._init_modes, pure=pure, **kwargs
            )

    def get_cutoff_dim(self):
        return self.circuit.cutoff_dim

    def get_modes(self):
        # pylint: disable=protected-access
        return [i for i, j in enumerate(self._modemap._map) if j is not None]

    def prepare_vacuum_state(self, mode):
        with tf.name_scope("Prepare_vacuum"):
            remapped_mode = self._remap_modes(mode)
            self.circuit.prepare_vacuum_state(remapped_mode)

    def prepare_coherent_state(self, alpha, mode):
        with tf.name_scope("Prepare_coherent"):
            remapped_mode = self._remap_modes(mode)
            self.circuit.prepare_coherent_state(alpha, remapped_mode)

    def prepare_squeezed_state(self, r, phi, mode):
        with tf.name_scope("Prepare_squeezed"):
            remapped_mode = self._remap_modes(mode)
            self.circuit.prepare_squeezed_state(r, phi, remapped_mode)

    def prepare_displaced_squeezed_state(self, alpha, r, phi, mode):
        with tf.name_scope("Prepare_displaced_squeezed"):
            remapped_mode = self._remap_modes(mode)
            self.circuit.prepare_displaced_squeezed_state(alpha, r, phi, remapped_mode)

    def prepare_fock_state(self, n, mode):
        with tf.name_scope("Prepare_fock"):
            remapped_mode = self._remap_modes(mode)
            self.circuit.prepare_fock_state(n, remapped_mode)

    def prepare_ket_state(self, state, modes):
        with tf.name_scope("Prepare_state"):
            self.circuit.prepare_multimode(state, self._remap_modes(modes), True)

    def prepare_dm_state(self, state, modes):
        with tf.name_scope("Prepare_state"):
            self.circuit.prepare_multimode(state, self._remap_modes(modes), False)

    def prepare_thermal_state(self, nbar, mode):
        with tf.name_scope("Prepare_thermal"):
            remapped_mode = self._remap_modes(mode)
            self.circuit.prepare_thermal_state(nbar, remapped_mode)

    def rotation(self, phi, mode):
        with tf.name_scope("Rotation"):
            remapped_mode = self._remap_modes(mode)
            self.circuit.phase_shift(phi, remapped_mode)

    def displacement(self, alpha, mode):
        with tf.name_scope("Displacement"):
            remapped_mode = self._remap_modes(mode)
            self.circuit.displacement(alpha, remapped_mode)

    def squeeze(self, z, mode):
        with tf.name_scope("Squeeze"):
            remapped_mode = self._remap_modes(mode)
            self.circuit.squeeze(z, remapped_mode)

    def beamsplitter(self, t, r, mode1, mode2):
        with tf.name_scope("Beamsplitter"):
            if isinstance(t, complex):
                raise ValueError("Beamsplitter transmittivity t must be a float.")
            if isinstance(t, tf.Tensor):
                if t.dtype.is_complex:
                    raise ValueError("Beamsplitter transmittivity t must be a float.")
            remapped_modes = self._remap_modes([mode1, mode2])
            self.circuit.beamsplitter(t, r, remapped_modes[0], remapped_modes[1])

    def loss(self, T, mode):
        with tf.name_scope("Loss"):
            remapped_mode = self._remap_modes(mode)
            self.circuit.loss(T, remapped_mode)

    def cubic_phase(self, gamma, mode):
        with tf.name_scope("Cubic_phase"):
            remapped_mode = self._remap_modes(mode)
            self.circuit.cubic_phase(gamma, remapped_mode)

    def kerr_interaction(self, kappa, mode):
        with tf.name_scope("Kerr_interaction"):
            remapped_mode = self._remap_modes(mode)
            self.circuit.kerr_interaction(kappa, remapped_mode)

    def cross_kerr_interaction(self, kappa, mode1, mode2):
        with tf.name_scope("Cross-Kerr_interaction"):
            remapped_modes = self._remap_modes([mode1, mode2])
            self.circuit.cross_kerr_interaction(kappa, remapped_modes[0], remapped_modes[1])

    def state(self, modes=None, **kwargs):
        r"""Returns the state of the quantum simulation, restricted to the subsystems defined by `modes`.

        See :meth:`.BaseBackend.state`.

        Keyword Args:
            session (tf.Session): TensorFlow session
            feed_dict (Dict): Dictionary containing the desired numerical values for Tensors
                for numerically evaluating the state. Used with ``session``.

        Returns:
            FockStateTF: state description
        """
        with tf.name_scope("State"):
            s = self.circuit.state
            pure = self.circuit.state_is_pure
            num_modes = self.circuit.num_modes
            batched = self.circuit.batched

            # reduce rho down to specified subsystems
            if modes is None:
                # reduced state is full state
                reduced_state = s
                modes = list(range(num_modes))
            else:
                if isinstance(modes, int):
                    modes = [modes]
                if len(modes) != len(set(modes)):
                    raise ValueError("The specified modes cannot be duplicated.")
                if len(modes) > num_modes:
                    raise ValueError(
                        "The number of specified modes cannot be larger than the number of subsystems."
                    )

                if pure:
                    # convert to mixed state representation
                    reduced_state = mixed(s, batched)
                    pure = False
                else:
                    reduced_state = s

                    # trace our all modes not in modes
                    # todo: Doing this one by one is very inefficient. The partial trace function should be improved.
                for mode in sorted([m for m in range(num_modes) if m not in modes], reverse=True):
                    reduced_state = partial_trace(reduced_state, mode, False, batched)
                reduced_state_pure = False

            # unless the modes were requested in order, we need to swap indices around
            if modes != sorted(modes):
                mode_permutation = np.argsort(np.argsort(modes))
                reduced_state = reorder_modes(
                    reduced_state, mode_permutation, reduced_state_pure, batched
                )

            evaluate_results, session, feed_dict, close_session = _check_for_eval(kwargs)
            if evaluate_results:
                s = session.run(reduced_state, feed_dict=feed_dict)
                if close_session:
                    session.close()
            else:
                s = reduced_state

            modenames = ["q[{}]".format(i) for i in np.array(self.get_modes())[modes]]
            state_ = FockStateTF(
                s,
                len(modes),
                pure,
                self.circuit.cutoff_dim,
                graph=self._graph,
                batched=batched,
                mode_names=modenames,
                eval=evaluate_results,
            )
        return state_

    def measure_fock(self, modes, shots=1, select=None, **kwargs):
        """Measure the given modes in the Fock basis.

        See :meth:`.BaseFock.measure_fock`.

        Keyword Args:
            session (tf.Session): TensorFlow session
            feed_dict (Dict): Dictionary containing the desired numerical values for Tensors
                for numerically evaluating the measurement results. Used with ``session``.

        Returns:
            tuple[int] or tuple[Tensor]: measurement outcomes
        """
        if shots != 1:
            raise NotImplementedError(
                "TF backend currently does not support " "shots != 1 for Fock measurement"
            )
        with tf.name_scope("Measure_fock"):
            remapped_modes = self._remap_modes(modes)
            meas = self.circuit.measure_fock(remapped_modes, select=select, **kwargs)
        return meas

    def measure_homodyne(self, phi, mode, shots=1, select=None, **kwargs):
        """Perform a homodyne measurement on the specified modes.

        See :meth:`.BaseBackend.measure_homodyne`.

        Keyword Args:
            session (tf.Session): TensorFlow session
            feed_dict (Dict): Dictionary containing the desired numerical values for Tensors
                for numerically evaluating the measurement results. Used with ``session``.
            num_bins (int): Number of equally spaced bins for the probability distribution function
                (pdf) simulating the homodyne measurement (default: 100000).
            max (float): The pdf is discretized onto the 1D grid [-max,max] (default: 10).

        Returns:
            float or tf.Tensor: measurement outcome
        """
        if shots != 1:
            raise NotImplementedError(
                "TF backend currently does not support " "shots != 1 for homodyne measurement"
            )
        with tf.name_scope("Measure_homodyne"):
            remapped_mode = self._remap_modes(mode)
            meas = self.circuit.measure_homodyne(phi, remapped_mode, select, **kwargs)
        return meas

    def is_vacuum(self, tol=0.0, **kwargs):
        with tf.name_scope("Is_vacuum"):
            with self.circuit.graph.as_default():
                vac_elem = self.circuit.vacuum_element()
                if "eval" in kwargs and kwargs["eval"] is False:
                    v = vac_elem
                else:
                    sess = tf.Session()
                    v = sess.run(vac_elem)
                    sess.close()

            result = np.abs(v - 1) <= tol
        return result

    def del_mode(self, modes):
        with tf.name_scope("Del_mode"):
            remapped_modes = self._remap_modes(modes)
            if isinstance(remapped_modes, int):
                remapped_modes = [remapped_modes]
            self.circuit.del_mode(remapped_modes)
            self._modemap.delete(modes)

    def add_mode(self, n=1):
        with tf.name_scope("Add_mode"):
            self.circuit.add_mode(n)
            self._modemap.add(n)

    @property
    def graph(self):
        """
        Get the Tensorflow Graph object where the current quantum circuit is defined.

        Returns:
            tf.Graph: the circuit's graph
        """
        return self._graph
