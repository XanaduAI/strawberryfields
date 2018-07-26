# Copyright 2018 Xanadu Quantum Technologies Inc.

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
# pylint: disable=too-many-public-methods,not-context-manager
import numpy as np
import tensorflow as tf

from strawberryfields.backends import BaseFock, ModeMap
from .circuit import Circuit
from .ops import _check_for_eval, mixed, partial_trace, reorder_modes
from .states import FockStateTF

class TFBackend(BaseFock):
    """Tensorflow Backend implementation."""

    def __init__(self, graph=None):
        """
        Instantiate a TFBackend object.

        Args:
            graph (Graph): optional Tensorflow Graph object where circuit should be defined
        """
        super().__init__()
        self._supported["mixed_states"] = True
        self._supported["batched"] = True
        self._supported["symbolic"] = True
        self._short_name = "tf"
        if graph is None:
            self._graph = tf.get_default_graph()
        else:
            self._graph = graph

    def _remap_modes(self, modes):
        if isinstance(modes, int):
            modes = [modes]
            was_int = True
        else:
            was_int = False
        map_ = self._modemap.show()
        submap = [map_[m] for m in modes]
        if not self._modemap.valid(modes) or None in submap:
            raise ValueError('The specified modes are not valid.')
        else:
            remapped_modes = self._modemap.remap(modes)
        if was_int:
            remapped_modes = remapped_modes[0]
        return remapped_modes

    def begin_circuit(self, num_subsystems, cutoff_dim=None, hbar=2, pure=True, **kwargs):
        r"""Create a quantum circuit (initialized in vacuum state) with the number of modes
        equal to num_subsystems and a Fock-space cutoff dimension of cutoff_dim.

        Args:
            num_subsystems (int): number of modes the circuit should begin with
            cutoff_dim (int): numerical cutoff dimension in Fock space for each mode.
                ``cutoff_dim=D`` represents the Fock states :math:`|0\rangle,\dots,|D-1\rangle`.
                This argument is **required** for the Tensorflow backend.
            hbar (float): The value of :math:`\hbar` to initialise the circuit with, depending on the conventions followed.
                By default, :math:`\hbar=2`. See :ref:`conventions` for more details.
            pure (bool): whether to begin the circuit in a pure state representation
            **kwargs: optional keyword arguments which will be passed to the underlying circuit class

                * **batch_size** (*None* or *int*): the size of the batch-axis dimension. If None, no batch-axis will be used.
        """
        # pylint: disable=too-many-arguments,attribute-defined-outside-init
        with tf.name_scope('Begin_circuit'):
            batch_size = kwargs.get('batch_size', None)

            if cutoff_dim is None:
                raise ValueError("Argument 'cutoff_dim' must be passed to the Tensorflow backend")

            if not isinstance(num_subsystems, int):
                raise ValueError("Argument 'num_subsystems' must be a positive integer")
            elif not isinstance(cutoff_dim, int):
                raise ValueError("Argument 'cutoff_dim' must be a positive integer")
            elif not isinstance(pure, bool):
                raise ValueError("Argument 'pure' must be either True or False")
            elif batch_size == 1:
                raise ValueError("batch_size of 1 not supported, please use different batch_size or set batch_size=None")
            else:
                self._modemap = ModeMap(num_subsystems)
                circuit = Circuit(self._graph, num_subsystems, cutoff_dim, hbar, pure, batch_size)

        self._init_modes = num_subsystems
        self.circuit = circuit

    def reset(self, pure=True, **kwargs):
        r"""
        Resets the circuit state tensor back to an all-vacuum state.

        Args:
          pure (bool): whether to use a pure state representation upon reset

        Keyword Args:
          hard (bool): whether to reset the underlying tensorflow graph.
                  If hard reset is specified, then resets the underlying tensor graph as well.
                  If False, then the circuit is reset to its initial state, but ops that
                  have already been declared are still accessible.
          cutoff_dim (int): new cutoff dimension for the simulated circuit.
          hbar (float): New :math:`\hbar` value. See :ref:`conventions` for more details.

        """
        hard = kwargs.get('hard', True)
        if hard:
            tf.reset_default_graph()
            self._graph = tf.get_default_graph()

        with tf.name_scope('Reset'):
            self._modemap.reset()
            self.circuit.reset(pure, graph=self._graph, num_subsystems=self._init_modes, **kwargs)

    def get_cutoff_dim(self):
        """Returns the Hilbert space cutoff dimension used.

        Returns:
            int: cutoff dimension
        """
        return self.circuit.cutoff_dim

    def get_modes(self):
        """Return a list of the active mode indices for the circuit.

        Returns:
            list[int]: sorted list of active (assigned, not invalid) mode indices
        """
        # pylint: disable=protected-access
        return [i for i, j in enumerate(self._modemap._map) if j is not None]

    def prepare_vacuum_state(self, mode):
        """
        Prepare the vacuum state on the specified mode.
        Note: this may convert the state representation to mixed.

        Args:
            mode (int): index of mode where state is prepared
        """
        with tf.name_scope('Prepare_vacuum'):
            remapped_mode = self._remap_modes(mode)
            self.circuit.prepare_vacuum_state(remapped_mode)

    def prepare_coherent_state(self, alpha, mode):
        """
        Prepare a coherent state with parameter alpha on the specified mode.
        Note: this may convert the state representation to mixed.

        Args:
            alpha (complex): coherent state displacement parameter
            mode (int): index of mode where state is prepared
        """
        with tf.name_scope('Prepare_coherent'):
            remapped_mode = self._remap_modes(mode)
            self.circuit.prepare_coherent_state(alpha, remapped_mode)

    def prepare_squeezed_state(self, r, phi, mode):
        """
        Prepare a coherent state with parameters (r, phi) on the specified mode.
        Note: this may convert the state representation to mixed.

        Args:
            r (float): squeezing amplitude
            phi (float): squeezing phase
            mode (int): index of mode where state is prepared

        """
        with tf.name_scope('Prepare_squeezed'):
            remapped_mode = self._remap_modes(mode)
            self.circuit.prepare_squeezed_state(r, phi, remapped_mode)

    def prepare_displaced_squeezed_state(self, alpha, r, phi, mode):
        """
        Prepare a displaced squezed state with parameters (alpha, r, phi) on the specified mode.
        Note: this may convert the state representation to mixed.

        Args:
            alpha (complex): displacement parameter
            r (float): squeezing amplitude
            phi (float): squeezing phase
            mode (int): index of mode where state is prepared

        """
        with tf.name_scope('Prepare_displaced_squeezed'):
            remapped_mode = self._remap_modes(mode)
            self.circuit.prepare_displaced_squeezed_state(alpha, r, phi, remapped_mode)

    def prepare_fock_state(self, n, mode):
        """
        Prepare a Fock state on the specified mode.
        Note: this may convert the state representation to mixed.

        Args:
            n (int): number state to prepare
            mode (int): index of mode where state is prepared

        """
        with tf.name_scope('Prepare_fock'):
            remapped_mode = self._remap_modes(mode)
            self.circuit.prepare_fock_state(n, remapped_mode)

    def prepare_ket_state(self, state, modes):
        """
        Prepare an arbitrary pure state on the specified mode.
        Note: this may convert the state representation to mixed.

        Args:
            state (array): vector representation of ket state to prepare
            mode (int): index of mode where state is prepared

        """
        with tf.name_scope('Prepare_state'):
            self.circuit.prepare_multimode(state, self._remap_modes(modes), True)

    def prepare_dm_state(self, state, modes):
        """
        Prepare an arbitrary mixed state on the specified mode.
        Note: this converts the state representation to mixed.

        Args:
            state (array): matrix representation of the state to prepare
            mode (int): index of mode where state is prepared

        """
        with tf.name_scope('Prepare_state'):
            self.circuit.prepare_multimode(state, self._remap_modes(modes), False)

    def prepare_thermal_state(self, nbar, mode):
        """
        Prepare a thermal state on the specified mode.
        Note: this may convert the state representation to mixed.

        Args:
            nbar (float): mean photon number of the thermal state
            mode (int): index of mode where state is prepared

        """
        with tf.name_scope('Prepare_thermal'):
            remapped_mode = self._remap_modes(mode)
            self.circuit.prepare_thermal_state(nbar, remapped_mode)

    def rotation(self, phi, mode):
        """
        Perform a phase shift by angle phi on the specified mode.

        Args:
            phi (float):
            mode (int): index of mode where operation is carried out

        """
        with tf.name_scope('Rotation'):
            remapped_mode = self._remap_modes(mode)
            self.circuit.phase_shift(phi, remapped_mode)

    def displacement(self, alpha, mode):
        """
        Perform a displacement operation on the specified mode.

        Args:
            alpha (float): displacement parameter
            mode (int): index of mode where operation is carried out

        """
        with tf.name_scope('Displacement'):
            remapped_mode = self._remap_modes(mode)
            self.circuit.displacement(alpha, remapped_mode)

    def squeeze(self, z, mode):
        """
        Perform a squeezing operation on the specified mode.

        Args:
            z (complex): squeezing parameter
            mode (int): index of mode where operation is carried out

        """
        with tf.name_scope('Squeeze'):
            remapped_mode = self._remap_modes(mode)
            self.circuit.squeeze(z, remapped_mode)

    def beamsplitter(self, t, r, mode1, mode2):
        """
        Perform a beamsplitter operation on the specified modes.

        Args:
            t (float): transmittivity parameter
            r (complex): reflectivity parameter
            mode1 (int): index of first mode where operation is carried out
            mode2 (int): index of second mode where operation is carried out

        """
        with tf.name_scope('Beamsplitter'):
            if isinstance(t, complex):
                raise ValueError("Beamsplitter transmittivity t must be a float.")
            if isinstance(t, tf.Tensor):
                if t.dtype.is_complex:
                    raise ValueError("Beamsplitter transmittivity t must be a float.")
            remapped_modes = self._remap_modes([mode1, mode2])
            self.circuit.beamsplitter(t, r, remapped_modes[0], remapped_modes[1])

    def loss(self, T, mode):
        """
        Perform a loss channel operation on the specified mode.

        Args:
            T: loss parameter
            mode (int): index of mode where operation is carried out

        """
        with tf.name_scope('Loss'):
            remapped_mode = self._remap_modes(mode)
            self.circuit.loss(T, remapped_mode)

    def cubic_phase(self, gamma, mode):
        r"""Apply the cubic phase operation to the specified mode.

        .. warning:: The cubic phase gate can suffer heavily from numerical inaccuracies due to finite-dimensional cutoffs in the Fock basis.
                     The gate implementation in Strawberry Fields is unitary, but it does not implement an exact cubic phase gate.
                     The Kerr gate provides an alternative non-Gaussian gate.

        Args:
            gamma (float): cubic phase shift
            mode (int): which mode to apply it to
        """
        with tf.name_scope('Cubic_phase'):
            remapped_mode = self._remap_modes(mode)
            self.circuit.cubic_phase(gamma, remapped_mode)

    def kerr_interaction(self, kappa, mode):
        r"""Apply the Kerr interaction :math:`\exp{(i\kappa \hat{n}^2)}` to the specified mode.

        Args:
            kappa (float): strength of the interaction
            mode (int): which mode to apply it to
        """
        with tf.name_scope('Kerr_interaction'):
            remapped_mode = self._remap_modes(mode)
            self.circuit.kerr_interaction(kappa, remapped_mode)

    def cross_kerr_interaction(self, kappa, mode1, mode2):
        r"""Apply the two mode cross-Kerr interaction :math:`\exp{(i\kappa \hat{n}_1\hat{n}_2)}` to the specified modes.

        Args:
            kappa (float): strength of the interaction
            mode1 (int): first mode that cross-Kerr interaction acts on
            mode2 (int): second mode that cross-Kerr interaction acts on
        """
        with tf.name_scope('Cross-Kerr_interaction'):
            remapped_modes = self._remap_modes([mode1, mode2])
            self.circuit.cross_kerr_interaction(kappa, remapped_modes[0], remapped_modes[1])

    def state(self, modes=None, **kwargs):
        r"""Returns the state of the quantum simulation, restricted to the subsystems defined by `modes`.

        Args:
            modes (int or Sequence[int]): specifies the mode or modes to restrict the return state to.
                                          This argument is optional; the default value ``modes=None`` returns the state containing all modes.
            **kwargs: optional keyword args (`session`: a Tensorflow session; `feed_dict`: a Python dictionary feeding the desired numerical values for Tensors)
                      which will be used by the Tensorflow simulator for numerically evaluating the measurement results.
        Returns:
            An instance of the Strawberry Fields FockStateTF class.
        """
        with tf.name_scope('State'):
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
                    raise ValueError("The number of specified modes cannot be larger than the number of subsystems.")

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
                reduced_state = reorder_modes(reduced_state, mode_permutation, reduced_state_pure, batched)

            evaluate_results, session, feed_dict, close_session = _check_for_eval(kwargs)
            if evaluate_results:
                s = session.run(reduced_state, feed_dict=feed_dict)
                if close_session:
                    session.close()
            else:
                s = reduced_state

            modenames = ["q[{}]".format(i) for i in np.array(self.get_modes())[modes]]
            state_ = FockStateTF(s, len(modes), pure, self.circuit.cutoff_dim,
                                 graph=self._graph, batched=batched, hbar=self.circuit.hbar,
                                 mode_names=modenames, eval=evaluate_results)
        return state_

    def measure_fock(self, modes, select=None, **kwargs):
        """
        Perform a Fock measurement on the specified modes.

        Args:
            modes (Sequence[int]): indices of mode where operation is carried out
            select (Sequence[int]): (Optional) desired values of measurement results. Allows user to post-select on specific measurement results instead of randomly sampling.
            **kwargs: optional keyword args (`session`: a Tensorflow session; `feed_dict`: a Python dictionary feeding the desired numerical values for Tensors)
                                which will be used by the Tensorflow simulator for numerically evaluating the measurement results.

        Returns:
            tuple[int] or tuple[Tensor]: measurement outcomes
        """
        with tf.name_scope('Measure_fock'):
            remapped_modes = self._remap_modes(modes)
            meas = self.circuit.measure_fock(remapped_modes, select=select, **kwargs)
        return meas

    def measure_homodyne(self, phi, mode, select=None, **kwargs):
        """
        Perform a homodyne measurement on the specified modes.

        Args:
            phi (float): angle (relative to x-axis) for the measurement.
            select (float): (Optional) desired values of measurement results. Allows user to post-select on specific measurement results instead of randomly sampling.
            mode (Sequence[int]): index of mode where operation is carried out
            **kwargs: optional keyword args (`session`: a Tensorflow session; `feed_dict`: a Python dictionary feeding the desired numerical values for Tensors)
                                which will be used by the Tensorflow simulator for numerically evaluating the measurement results.
                                In addition, kwargs can be used to (optionally) pass user-specified numerical parameters `max` and `num_bins`.
                                These are used numerically to build the probability distribution function (pdf) for the homodyne measurement.
                                Specifically, the pdf is discretized onto the 1D grid [-max,max], with num_bins equally spaced bins.

        Returns:
            tuple[float] or tuple[Tensor]: measurement outcomes
        """
        with tf.name_scope('Measure_homodyne'):
            remapped_mode = self._remap_modes(mode)
            meas = self.circuit.measure_homodyne(phi, remapped_mode, select, **kwargs)
        return meas

    def is_vacuum(self, tol=0.0, **kwargs):
        r"""Test whether the current circuit state is in vacuum (up to tolerance tol).
        Args:
            tol (float): numerical tolerance for how close state must be to true vacuum state
        Returns:
            bool: True if vacuum state up to tolerance tol
        """
        with tf.name_scope('Is_vacuum'):
            with self.circuit.graph.as_default():
                vac_elem = self.circuit.vacuum_element()
                if "eval" in kwargs and kwargs["eval"] is False:
                    v = vac_elem
                else:
                    sess = tf.Session()
                    v = sess.run(vac_elem)
                    sess.close()

            result = (1 - v) <= tol
        return result

    def del_mode(self, modes):
        """
        Trace out the specified modes from the underlying circuit state.
        Note: This will reduce the number of indices used for the state representation,
        and also convert the state representation to mixed.

        Args:
            modes (Sequence[int]): the modes to be removed from the circuit
        """
        with tf.name_scope('Del_mode'):
            remapped_modes = self._remap_modes(modes)
            if isinstance(remapped_modes, int):
                remapped_modes = [remapped_modes]
            self.circuit.del_mode(remapped_modes)
            self._modemap.delete(modes)

    def add_mode(self, n=1):
        """
        Add n new modes to the underlying circuit state. Indices for new modes
        always occur at the end of the state tensor.
        Note: This will increase the number of indices used for the state representation.

        Args:
            n (int): the number of modes to be added to the circuit.
        """
        with tf.name_scope('Add_mode'):
            self.circuit.add_mode(n)
            self._modemap.add(n)

    @property
    def graph(self):
        """
        Get the Tensorflow Graph object where the current quantum circuit is defined.

        Returns:
            (Graph): the circuit's graph
        """
        return self._graph
