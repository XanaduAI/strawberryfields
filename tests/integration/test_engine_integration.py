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
r"""Integration tests for the frontend engine.py module with the backends"""
import numbers
import pytest

import numpy as np

import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.backends import BaseFock
from strawberryfields.backends import GaussianBackend, FockBackend
from strawberryfields.backends.states import BaseState


try:
    from strawberryfields.backends.tfbackend import TFBackend
except (ImportError, ModuleNotFoundError, ValueError) as e:
    eng_backend_params = [("gaussian", GaussianBackend), ("fock", FockBackend)]
else:
    eng_backend_params = [
        ("tf", TFBackend),
        ("gaussian", GaussianBackend),
        ("fock", FockBackend),
    ]
    from tensorflow import Tensor


# make test deterministic
np.random.random(42)
a = 0.1234
b = -0.543
c = 0.312


@pytest.mark.parametrize("name,expected", eng_backend_params)
def test_load_backend(name, expected):
    """Test backends can be correctly loaded via strings"""
    eng = sf.Engine(name)
    assert isinstance(eng.backend, expected)


class TestEngineReset:
    """Test engine reset functionality"""

    def test_init_vacuum(self, setup_eng, tol):
        """Test that the engine is initialized to the vacuum state"""
        eng, prog = setup_eng(2)
        eng.run(prog)  # run an empty program
        assert np.all(eng.backend.is_vacuum(tol))

    def test_reset_vacuum(self, setup_eng, tol):
        """Test that resetting the engine returns to the vacuum state"""
        eng, prog = setup_eng(2)

        with prog.context:
            ops.Dgate(0.5) | 0

        eng.run(prog)
        assert not np.all(eng.backend.is_vacuum(tol))

        eng.reset()
        assert np.all(eng.backend.is_vacuum(tol))

    @pytest.mark.backends("fock")
    def test_eng_reset(self, setup_eng, cutoff):
        """Test the Engine.reset() features."""
        eng, prog = setup_eng(2)

        state = eng.run(prog).state
        backend_cutoff = eng.backend.get_cutoff_dim()
        assert state._cutoff == backend_cutoff
        assert cutoff == backend_cutoff

        # change the cutoff dimension
        new_cutoff = cutoff + 1
        eng.reset({"cutoff_dim": new_cutoff})

        state = eng.run(prog).state
        backend_cutoff = eng.backend.get_cutoff_dim()
        assert state._cutoff == backend_cutoff
        assert new_cutoff == backend_cutoff


class TestProperExecution:
    """Test that various frontend circuits execute through
    the backend with no error"""

    def test_no_return_state(self, setup_eng):
        """Engine returns no state object when no modes are requested."""
        eng, prog = setup_eng(2)
        res = eng.run(prog, modes=[])
        assert res.state is None

    def test_return_state(self, setup_eng):
        """Engine returns a valid state object."""
        eng, prog = setup_eng(2)
        res = eng.run(prog)
        assert isinstance(res.state, BaseState)
        assert res.state.num_modes == 2

    # TODO: Some of these tests should probably check *something* after execution

    def test_regref_transform(self, setup_eng):
        """Test circuit with regref transforms doesn't raise any errors"""
        eng, prog = setup_eng(2)

        with prog.context as q:
            ops.MeasureX | q[0]
            ops.Sgate(q[0]) | q[1]
            # symbolic hermitian conjugate together with register reference
            ops.Dgate(q[0]).H | q[1]
            ops.Sgate(ops.RR(q[0], lambda x: x ** 2)) | q[1]
            ops.Dgate(ops.RR(q[0], lambda x: -x)).H | q[1]

        eng.run(prog)

    def test_homodyne_measurement_vacuum(self, setup_eng, tol):
        """MeasureX and MeasureP leave the mode in the vacuum state"""
        eng, prog = setup_eng(2)
        with prog.context as q:
            ops.Coherent(a, c) | q[0]
            ops.Coherent(b, c) | q[1]
            ops.MeasureX | q[0]
            ops.MeasureP | q[1]

        eng.run(prog)
        assert np.all(eng.backend.is_vacuum(tol))

    def test_homodyne_measurement_vacuum_phi(self, setup_eng, tol):
        """Homodyne measurements leave the mode in the vacuum state"""
        eng, prog = setup_eng(2)
        with prog.context as q:
            ops.Coherent(a, b) | q[0]
            ops.MeasureHomodyne(c) | q[0]

        eng.run(prog)
        assert np.all(eng.backend.is_vacuum(tol))

    def test_program_subroutine(self, setup_eng, tol):
        """Simple quantum program with a subroutine and references."""
        eng, prog = setup_eng(2)

        # define some gates
        D = ops.Dgate(0.5)
        BS = ops.BSgate(0.7 * np.pi, np.pi / 2)
        R = ops.Rgate(np.pi / 3)

        def subroutine(a, b):
            """Subroutine for the quantum program"""
            R | a
            BS | (a, b)
            R.H | a

        # main program
        with prog.context as q:
            # get register references
            alice, bob = q
            ops.All(ops.Vacuum()) | (alice, bob)
            D | alice
            subroutine(alice, bob)
            BS | (alice, bob)
            subroutine(bob, alice)

        state = eng.run(prog).state

        # state norm must be invariant
        if isinstance(eng.backend, BaseFock):
            assert np.allclose(state.trace(), 1, atol=tol, rtol=0)

    def test_subsystems(self, setup_eng, tol):
        """Check that the backend keeps in sync with the program when creating and deleting modes."""
        null = sf.Program(2)  # empty program
        eng, prog = setup_eng(2)

        # define some gates
        D = ops.Dgate(0.5)
        BS = ops.BSgate(2 * np.pi, np.pi / 2)
        R = ops.Rgate(np.pi)

        with prog.context as q:
            alice, bob = q
            D | alice
            BS | (alice, bob)
            ops.Del | alice
            R | bob
            charlie, = ops.New(1)
            BS | (bob, charlie)
            ops.MeasureX | bob
            ops.Del | bob
            D.H | charlie
            ops.MeasureX | charlie

        def check_reg(p, expected_n=None):
            """Compare Program.register with the mode list returned by the backend.
            They should always be in agreement after Engine.run() and Engine.reset().
            """
            rr = p.register
            modes = eng.backend.get_modes()
            # number of elements
            assert len(rr) == len(modes)

            if expected_n is not None:
                assert len(rr) == expected_n

            # check indices match
            assert np.all([r.ind for r in rr] == modes)
            # activity
            assert np.all([r.active for r in rr])

        state = eng.run(null)
        check_reg(null, 2)
        state = eng.run(prog).state
        check_reg(prog, 1)

        # state norm must be invariant
        if isinstance(eng.backend, BaseFock):
            assert np.allclose(state.trace(), 1, atol=tol, rtol=0)

        # check that reset() works
        eng.reset()
        # the regrefs are reset as well
        assert np.all([r.val is None for r in prog.register])

    def test_empty_program(self, setup_eng):
        """Empty programs do not change the state of the backend."""
        eng, p1 = setup_eng(2)
        a = 0.23
        r = 0.1
        with p1.context as q:
            ops.Dgate(a) | q[0]
            ops.Sgate(r) | q[1]
        state1 = eng.run(p1).state

        # empty program
        p2 = sf.Program(p1)
        state2 = eng.run(p2).state
        assert state1 == state2

        p3 = sf.Program(p2)
        with p3.context as q:
            ops.Rgate(r) | q[0]
        state3 = eng.run(p3).state
        assert not state1 == state3

        state4 = eng.run(p2).state
        assert state3 == state4

    # TODO: when ``shots`` is incorporated into other backends, unmark this test
    @pytest.mark.backends("gaussian")
    def test_results_measure_fock_shots(self, setup_eng):
        """Tests that passing shots with a program containing MeasureFock
           returns a result whose entries have the right shapes and values"""
        shots = 5
        expected = np.zeros(dtype=int, shape=(shots,))

        # all modes
        eng, p1 = setup_eng(3)
        with p1.context as q:
            ops.MeasureFock() | q

        samples = eng.run(p1, shots=shots).samples

        assert type(samples) == dict
        assert len(samples) == 3
        assert np.all(samples[0] == expected)
        assert np.all(samples[1] == expected)
        assert np.all(samples[2] == expected)

        # some modes
        eng, p2 = setup_eng(3)
        with p2.context as q:
            ops.MeasureFock() | (q[0], q[2])

        samples = eng.run(p2, shots=shots).samples

        assert type(samples) == dict
        assert len(samples) == 2
        assert np.all(samples[0] == expected)
        assert 1 not in samples
        assert np.all(samples[2] == expected)

        # one mode
        eng, p3 = setup_eng(3)
        with p3.context as q:
            ops.MeasureFock() | q[0]

        samples = eng.run(p3, shots=shots).samples

        assert type(samples) == dict
        assert len(samples) == 1
        assert np.all(samples[0] == expected)
        assert 1 not in samples
        assert 2 not in samples

    # TODO: when ``shots`` is incorporated into other backends, delete this test
    @pytest.mark.backends("tf", "fock")
    def test_measure_fock_shots_exception(self, setup_eng):
        shots = 5
        eng, p1 = setup_eng(3)
        with p1.context as q:
            ops.MeasureFock() | q

        backend_name = eng.backend.__str__()
        with pytest.raises(NotImplementedError,
                           match=r"""(Measure|MeasureFock) has not been implemented in {} """
                                  """for the arguments {{'shots': {}}}""".format(backend_name, shots)):
            eng.run(p1, shots=shots).samples


class TestResults:
    """Integration tests for the Results class"""

    def test_results_no_meas(self, setup_eng):
        """Tests the Results object when a program containing no measurements is run."""
        eng, p = setup_eng(3)
        with p.context as q:
            ops.Dgate(0.1) | q[0]
        res = eng.run(p)

        assert type(res.samples) == dict
        assert len(res.samples) == 0
        assert type(res.measured_modes) == list
        assert len(res.measured_modes) == 0
        assert type(res.samples_array) == np.ndarray
        assert res.samples_array.shape == (0,)  # no entries or axes yet

    def test_return_samples(self, setup_eng):
        """Engine returns measurement samples."""
        eng, prog = setup_eng(2)
        with prog.context as q:
            ops.MeasureX | q[0]

        res = eng.run(prog)
        # one entry for each mode
        assert len(res.samples) == 1
        # the same samples can also be found in the regrefs
        assert prog.register[0].val == res.samples[0]
        # first mode was measured
        assert res.samples[0].dtype == float
        # second mode was not measured
        assert 1 not in res.samples

    def test_results_all_measure_fock_no_shots(self, setup_eng):
        """Tests the Results object when all modes are measured in the Fock basis
            and no value for ``shots`` is given."""

        # measured in canonical order
        expected_samples = {0: [0], 1: [0], 2: [0]}
        expected_measured_modes = [0, 1, 2]
        expected_samples_array = np.array([[0], [0], [0]])  # shape = (3,1)

        eng, p1 = setup_eng(3)

        with p1.context as q:
            ops.Measure | q

        res = eng.run(p1)

        assert res.samples == expected_samples
        assert res.measured_modes == expected_measured_modes
        assert np.all(res.samples_array == expected_samples_array)

        # TODO: refactor returned measurements
        # so that they are in the same order as requested
        # by the user

        # # measured in non-canonical order
        # eng, p2 = setup_eng(3)

        # with p2.context as q:
        #     ops.Measure | (q[2], q[1], q[0])

        # res = eng.run(p2)

        # perm = [2, 1, 0]
        # assert res.samples == expected_samples
        # assert res.measured_modes == [expected_measured_modes[i] for i in perm]
        # assert np.all(res.samples_array == expected_samples_array[perm])

    def test_results_subset_measure_fock_no_shots(self, setup_eng):
        """Tests the Results object when a subset of modes are measured in the Fock basis
            and no value for ``shots`` is given."""
        expected_samples = {0: [0], 2: [0]}
        expected_measured_modes = [0, 2]
        expected_samples_array = np.array([[0], [0]])

        # measured in canonical order
        eng, p1 = setup_eng(3)

        with p1.context as q:
            ops.Measure | (q[0], q[2])

        res = eng.run(p1)

        assert res.samples == expected_samples
        assert res.measured_modes == expected_measured_modes
        assert np.all(res.samples_array == expected_samples_array)

        # TODO: refactor returned measurements
        # so that they are in the same order as requested
        # by the user

        # # measured in non-canonical order
        # eng, p2 = setup_eng(3)

        # with p2.context as q:
        #     ops.Measure | (q[2], q[0])

        # res = eng.run(p2)

        # expected_measured_modes = [2, 0]
        # assert res.samples == expected_samples
        # assert res.measured_modes == expected_measured_modes
        # assert np.all(res.samples_array == expected_samples_array)

    # TODO: when ``shots`` is incorporated into other backends, add here
    @pytest.mark.backends("gaussian")
    def test_results_all_measure_fock_with_shots(self, setup_eng):
        """Tests the Results object when all modes are measured in the Fock basis
            and a value for ``shots`` is given."""

        shots = 5
        zeros = [0] * shots

        # measured in canonical order
        expected_samples = {0: zeros, 1: zeros, 2: zeros}
        expected_measured_modes = [0, 1, 2]
        expected_samples_array = np.array([zeros] * 3)  # shape = (3,5)

        eng, p1 = setup_eng(3)

        with p1.context as q:
            ops.Measure | q

        res = eng.run(p1, shots=shots)

        assert np.all(res.samples[0] == expected_samples[0])
        assert np.all(res.samples[1] == expected_samples[1])
        assert np.all(res.samples[2] == expected_samples[2])
        assert res.measured_modes == expected_measured_modes
        assert np.all(res.samples_array == expected_samples_array)

        # TODO: refactor returned measurements
        # so that they are in the same order as requested
        # by the user

        # # measured in non-canonical order
        # eng, p2 = setup_eng(3)
        # with p2.context as q:
        #     ops.Measure | (q[2], q[1], q[0])
        # res = eng.run(p2, shots=shots)

        # perm = [2, 1, 0]
        # assert res.samples == expected_samples
        # assert res.measured_modes == expected_measured_modes[perm]
        # assert res.samples_array == expected_samples_array[perm]

    # TODO: when ``shots`` is incorporated into other backends, add here
    @pytest.mark.backends("gaussian")
    def test_results_subset_measure_fock_with_shots(self, setup_eng):
        """Tests the Results object when a subset of modes are measured in the Fock basis
            and a value for ``shots`` is given."""

        shots = 5
        zeros = [0] * shots

        # measured in canonical order
        expected_samples = {0: zeros,
                            2: zeros}
        expected_measured_modes = [0, 2]
        expected_samples_array = np.array([zeros] * 2)  # shape = (2,5)

        eng, p1 = setup_eng(3)
        with p1.context as q:
            ops.Measure | (q[0], q[2])
        res = eng.run(p1, shots=shots)

        assert res.samples == expected_samples
        assert res.measured_modes == expected_measured_modes
        assert np.all(res.samples_array == expected_samples_array)

        # # measured in non-canonical order
        # eng, p2 = setup_eng(3)
        # with p2.context as q:
        #     ops.Measure | (q[2], q[0])
        # res = eng.run(p2, shots=shots)

        # perm = [2, 0]
        # assert res.samples == expected_samples
        # assert res.measured_modes == expected_measured_modes[perm]
        # assert res.samples_array == expected_samples_array[perm]

    @pytest.mark.backends("gaussian")
    def test_results_measure_heterodyne_no_shots(self, setup_eng):
        """Tests the Results object when all modes are measured with heterodyne
           and no value for ``shots`` is given"""

        eng, p = setup_eng(3)
        with p.context as q:
            ops.MeasureHeterodyne() | q[1]
        res = eng.run(p)

        assert type(res.samples) == dict
        assert len(res.samples) == 1
        assert 1 in res.samples
        assert isinstance(res.samples[1], np.complex)
        assert res.measured_modes == [1]
        assert type(res.samples_array) == np.ndarray
        assert res.samples_array.dtype == np.complex
        assert res.samples_array.shape == (1, 1)

    @pytest.mark.backends("gaussian")
    def test_results_measure_heterodyne_with_shots(self, setup_eng):
        """Tests the Results object when all modes are measured with heterodyne
           and a value for ``shots`` is given"""

        eng, p = setup_eng(3)
        # TODO: replace with proper test when implemented
        shots = 5
        with p.context as q:
            ops.MeasureHeterodyne() | q[1]
        name = eng.backend._short_name.capitalize()
        with pytest.raises(NotImplementedError,
                           match="{} backend currently does not support "
                                 "shots != 1 for heterodyne measurement".format(name)):
            res = eng.run(p, shots=shots)

    def test_results_measure_homodyne_no_shots(self, setup_eng):
        """Tests the Results object when all modes are measured with heterodyne
           and no value for ``shots`` is given"""

        eng, p = setup_eng(3)
        with p.context as q:
            ops.MeasureHomodyne(c) | q[1]
        res = eng.run(p)

        assert type(res.samples) == dict
        assert len(res.samples) == 1
        assert 1 in res.samples
        assert res.samples[1].dtype == float
        assert res.measured_modes == [1]
        assert type(res.samples_array) == np.ndarray
        assert res.samples_array.dtype == np.float
        assert res.samples_array.shape == (1, 1)

    def test_results_measure_homodyne_shots(self, setup_eng):
        """Tests the Results object when all modes are measured with homodyne
           and a value for ``shots`` is given"""

        # TODO: replace with proper test when implemented
        shots = 5
        eng, p = setup_eng(3)
        with p.context as q:
            ops.MeasureHomodyne(c) | q[1]
        name = eng.backend.__class__.__name__
        with pytest.raises(NotImplementedError,
                           match=r"The operation MeasureHomodyne(0.312) has not been implemented in"):
            res = eng.run(p, shots=shots)

    #TODO: following tests should be marked to run only when BATCHED=1

    @pytest.mark.backends("tf")
    def test_results_batched_all_measure_fock_no_shots(self, batch_size):
        """Tests the Results object when all modes are measured in the Fock basis
            in batch mode and no value for ``shots`` is given."""

        zeros = np.zeros(batch_size, dtype=int)
        expected = {0: zeros, 1: zeros, 2: zeros}

        eng = sf.Engine("tf", backend_options={"batch_size": batch_size,
                                               "cutoff_dim": 3})
        p1 = sf.Program(3)
        with p1.context as q:
            ops.Measure | q
        res = eng.run(p1)

        assert type(res.samples) == dict
        assert np.all([val == expected[k] for k,val in res.samples.items()])
        assert sorted(res.measured_modes) == list(expected.keys())
        assert res.samples_array.shape == (batch_size, 3, 1)

    @pytest.mark.backends("tf")
    def test_results_batched_subset_measure_fock_no_shots(self, batch_size):
        """Tests the Results object when a subset of modes are measured in the Fock basis
            in batch mode and no value for ``shots`` is given."""

        zeros = np.zeros(batch_size, dtype=int)
        expected = {0: zeros, 2: zeros}

        eng = sf.Engine("tf", backend_options={"batch_size": batch_size,
                                               "cutoff_dim": 3})
        p1 = sf.Program(3)
        with p1.context as q:
            ops.Measure | (q[0], q[2])
        res = eng.run(p1)

        assert type(res.samples) == dict
        assert np.all([val == expected[k] for k,val in res.samples.items()])
        assert sorted(res.measured_modes) == list(expected.keys())
        assert res.samples_array.shape == (batch_size, 2, 1)

    @pytest.mark.backends("tf")
    def test_results_batched_measure_homodyne_no_shots(self, batch_size):
        """Tests the Results object when homodyne measurement is made
            in batch mode and no value for ``shots`` is given."""

        eng = sf.Engine("tf", backend_options={"batch_size": batch_size,
                                               "cutoff_dim": 3})
        p = sf.Program(3)
        with p.context as q:
            ops.MeasureHomodyne(c) | q[1]
        res = eng.run(p)

        assert type(res.samples) == dict
        assert res.measured_modes == [1]
        assert res.samples_array.shape == (batch_size, 1, 1)

    @pytest.mark.backends("tf")
    def test_results_batched_measure_homodyne_with_shots(self, batch_size):
        """Tests the Results object when homodyne measurement is made
            in batch mode and no value for ``shots`` is given."""

        # TODO: replace with proper test when implemented
        # note that this will lead to a different test than
        # ``test_results_measure_homodyne_shots`` above,
        # even though they currently test for the same thing
        shots = 5
        eng = sf.Engine("tf", backend_options={"batch_size":batch_size,
                                               "cutoff_dim":3})
        p = sf.Program(3)
        with p.context as q:
            ops.MeasureHomodyne(c) | q[1]
        name = eng.backend._short_name.capitalize()
        with pytest.raises(NotImplementedError,
                           match="{} backend currently does not support "
                                 "shots != 1 for homodyne measurement".format(name)):
            res = eng.run(p, shots=shots)
