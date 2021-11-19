# Copyright 2019-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Unit tests for engine.py"""
import pytest
import numpy as np

tf = pytest.importorskip("tensorflow", minversion="2.0")
import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.backends.base import BaseBackend

pytestmark = pytest.mark.frontend

# pylint: disable=redefined-outer-name,no-self-use,bad-continuation,expression-not-assigned,pointless-statement


@pytest.fixture
def eng(backend):
    """Engine fixture."""
    return sf.LocalEngine(backend)


@pytest.fixture
def prog():
    """Program fixture."""
    prog = sf.Program(2)
    with prog.context as q:
        ops.Dgate(0.5, 0.0) | q[0]
    return prog


batch_engines = [
    sf.Engine("gaussian", backend_options={"batch_size": 2, "cutoff_dim": 6}),
    sf.Engine("fock", backend_options={"batch_size": 2, "cutoff_dim": 6}),
    sf.Engine("tf", backend_options={"batch_size": 2, "cutoff_dim": 6}),
    sf.Engine("tf", backend_options={"batch_size": 2, "cutoff_dim": 6, "dtype": tf.complex128}),
]

engines = [
    sf.Engine("gaussian", backend_options={"cutoff_dim": 6}),
    sf.Engine("fock", backend_options={"cutoff_dim": 6}),
    sf.Engine("tf", backend_options={"cutoff_dim": 6}),
    sf.Engine("tf", backend_options={"cutoff_dim": 6, "dtype": tf.complex128}),
]


class TestEngine:
    """Test basic engine functionality"""

    def test_load_backend(self):
        """Backend can be correctly loaded via strings"""
        eng = sf.LocalEngine("base")
        assert isinstance(eng.backend, BaseBackend)

    def test_bad_backend(self):
        """Backend must be a string or a BaseBackend instance."""
        with pytest.raises(TypeError, match="backend must be a string or a BaseBackend instance"):
            _ = sf.LocalEngine(0)


class TestEngineProgramInteraction:
    """Test the Engine class and its interaction with Program instances."""

    def test_shots_default(self):
        """Test that default shots (1) is used"""
        prog = sf.Program(1)
        eng = sf.Engine("gaussian")

        with prog.context as q:
            ops.Sgate(0.5) | q[0]
            ops.MeasureFock() | q

        results = eng.run(prog)
        assert len(results.samples) == 1

    def test_shots_run_options(self):
        """Test that run_options takes precedence over default"""
        prog = sf.Program(1)
        eng = sf.Engine("gaussian")

        with prog.context as q:
            ops.Sgate(0.5) | q[0]
            ops.MeasureFock() | q

        prog.run_options = {"shots": 5}
        results = eng.run(prog)
        assert len(results.samples) == 5

    def test_shots_passed(self):
        """Test that shots supplied via eng.run takes precedence over
        run_options and that run_options isn't changed"""
        prog = sf.Program(1)
        eng = sf.Engine("gaussian")

        with prog.context as q:
            ops.Sgate(0.5) | q[0]
            ops.MeasureFock() | q

        prog.run_options = {"shots": 5}
        results = eng.run(prog, shots=2)
        assert len(results.samples) == 2
        assert prog.run_options["shots"] == 5

    def test_history(self, eng, prog):
        """Engine history."""
        # no programs have been run
        assert not eng.run_progs
        eng.run(prog)
        # one program has been run
        assert len(eng.run_progs) == 1
        assert eng.run_progs[-1] == prog  # no compilation required with BaseBackend

    def test_reset(self, eng, prog):
        """Running independent programs with an engine reset in between."""
        assert not eng.run_progs
        eng.run(prog)
        assert len(eng.run_progs) == 1

        eng.reset()
        assert not eng.run_progs
        p2 = sf.Program(3)
        with p2.context as q:
            ops.Rgate(1.0) | q[2]
        eng.run(p2)
        assert len(eng.run_progs) == 1

    def test_regref_mismatch(self, eng):
        """Running incompatible programs sequentially gives an error."""
        p1 = sf.Program(3)
        p2 = sf.Program(p1)
        p1.locked = False
        with p1.context as q:
            ops.Del | q[0]

        with pytest.raises(RuntimeError, match="Register mismatch"):
            eng.run([p1, p2])

    def test_sequential_programs(self, eng):
        """Running several program segments sequentially."""
        D = ops.Dgate(0.2, 0.0)
        p1 = sf.Program(3)
        with p1.context as q:
            D | q[1]
            ops.Del | q[0]
        assert not eng.run_progs
        eng.run(p1)
        assert len(eng.run_progs) == 1

        # p2 succeeds p1
        p2 = sf.Program(p1)
        with p2.context as q:
            D | q[1]
        eng.run(p2)
        assert len(eng.run_progs) == 2

        # p2 does not alter the register so it can be repeated
        eng.run([p2] * 3)
        assert len(eng.run_progs) == 5

        eng.reset()
        assert not eng.run_progs

    def test_print_applied(self, eng):
        """Tests the printing of executed programs."""
        a = 0.23
        r = 0.1

        def inspect():
            res = []
            print_fn = lambda x: res.append(x.__str__())
            eng.print_applied(print_fn)
            return res

        p1 = sf.Program(2)
        with p1.context as q:
            ops.Dgate(np.abs(a), np.angle(a)) | q[1]
            ops.Sgate(r) | q[1]

        eng.run(p1)
        expected1 = [
            "Run 0:",
            "Dgate({}, 0) | (q[1])".format(a),
            "Sgate({}, 0) | (q[1])".format(r),
        ]
        assert inspect() == expected1

        # run the program again
        eng.reset()
        eng.run(p1)
        assert inspect() == expected1

        # apply more commands to the same backend
        p2 = sf.Program(2)
        with p2.context as q:
            ops.Rgate(r) | q[1]

        eng.run(p2)
        expected2 = expected1 + ["Run 1:", "Rgate({}) | (q[1])".format(r)]
        assert inspect() == expected2

        # reapply history
        eng.reset()
        eng.run([p1, p2])
        assert inspect() == expected2

    @pytest.mark.parametrize("eng", engines)
    def test_combining_samples(self, eng):
        """Check that samples are combined correctly when using multiple measurements"""

        prog = sf.Program(5)
        with prog.context as q:
            ops.MeasureX | q[2]
            ops.MeasureFock() | (q[1], q[3])
            ops.MeasureFock() | q[0]

        result = eng.run(prog)

        # check that shape is (shots, measured_modes)
        assert result.samples.shape == (1, 4)

        # check that MeasureFock measures `0` while MeasureX does NOT measure `0`.
        correct_samples = [0, 0, 1, 0]
        assert [bool(i) for i in result.samples[0]] == correct_samples

    @pytest.mark.parametrize("eng", engines)
    def test_measuring_same_modes(self, eng):
        """Check that only the last measurement is returned when measuring the same mode twice"""

        prog = sf.Program(5)
        with prog.context as q:
            ops.MeasureFock() | q[2]
            ops.MeasureFock() | (q[1], q[3])
            ops.MeasureX | q[2]

        result = eng.run(prog)

        # check that shape is (shots, measured_modes)
        assert result.samples.shape == (1, 3)

        # check that MeasureFock measures `0` while MeasureX does NOT measure `0`.
        correct_samples = [0, 1, 0]
        assert [bool(i) for i in result.samples[0]] == correct_samples

    @pytest.mark.parametrize("eng", engines)
    def test_samples_dict_one_measurement_per_mode(self, eng):
        """Test that samples are stored for the correct modes with
        a single measurement each mode."""
        prog = sf.Program(5)
        with prog.context as q:
            ops.MeasureFock() | q[2]
            ops.MeasureFock() | (q[1], q[3])

        result = eng.run(prog)

        # Check that the number of all the samples equals to the number
        # of measurements
        assert len(result.samples_dict.values()) == 3

        correct_modes = [2, 1, 3]
        correct_samples = [[0], [0], [0]]

        assert result.samples_dict[1] == [0]
        assert result.samples_dict[3] == [0]
        assert result.samples_dict[2] == [0]

    @pytest.mark.parametrize("eng", engines)
    def test_samples_dict_multiple_measurements_per_mode(self, eng):
        """Test that samples are stored for the correct modes with
        multiple measurements on certain modes."""
        prog = sf.Program(5)
        with prog.context as q:
            ops.MeasureFock() | q[2]
            ops.MeasureFock() | (q[1], q[3])
            ops.MeasureX | q[2]

        result = eng.run(prog)

        assert result.samples_dict[1] == [0]
        assert result.samples_dict[3] == [0]
        assert [bool(i) for i in result.samples_dict[2]] == [0, 1]

    @pytest.mark.parametrize("eng", engines)
    def test_samples_dict_multiple_runs(self, eng):
        """Test that consecutive engine runs reset the samples_dict
        attribute by checking the length of the attribute."""
        prog = sf.Program(5)
        with prog.context as q:
            ops.MeasureFock() | q[2]
            ops.MeasureFock() | (q[1], q[3])
            ops.MeasureX | q[2]

        result = eng.run(prog)

        # Check that the number of all the samples equals to the number
        # of measurements
        assert result.samples_dict[1] == [0]
        assert result.samples_dict[3] == [0]
        assert [bool(i) for i in result.samples_dict[2]] == [0, 1]

        prog = sf.Program(5)
        with prog.context as q:
            ops.MeasureFock() | q[0]

        result = eng.run(prog)

        # Check that samples_dict contains the same elements and new items were
        # not appended
        assert result.samples_dict[0] == [0]

    def test_samples_dict_multiple_shots(self):
        """Test the case of storing all samples for multiple shots"""
        eng = sf.Engine("gaussian", backend_options={"cutoff_dim": 6})
        shots = 5
        prog = sf.Program(5)
        with prog.context as q:
            ops.MeasureFock() | q[2]
            ops.MeasureFock() | (q[1], q[3])
            ops.MeasureFock() | q[2]

        result = eng.run(prog, shots=shots)

        assert len(result.samples_dict[1]) == 1
        assert len(result.samples_dict[3]) == 1
        assert len(result.samples_dict[2]) == 2

        assert np.array_equal(result.samples_dict[1][0], np.array([0, 0, 0, 0, 0]))
        assert np.array_equal(result.samples_dict[3][0], np.array([0, 0, 0, 0, 0]))
        assert np.array_equal(result.samples_dict[2][0], np.array([0, 0, 0, 0, 0]))
        assert np.array_equal(result.samples_dict[2][1], np.array([0, 0, 0, 0, 0]))

    def test_samples_dict_batched(self):
        """Test the case of storing all samples for batches"""
        batch_size = 2
        eng = sf.Engine("tf", backend_options={"batch_size": batch_size, "cutoff_dim": 6})
        prog = sf.Program(5)
        with prog.context as q:
            ops.MeasureFock() | q[2]
            ops.MeasureFock() | (q[1], q[3])
            ops.MeasureFock() | q[2]

        result = eng.run(prog)
        assert len(result.samples_dict[1]) == 1
        assert len(result.samples_dict[3]) == 1
        assert len(result.samples_dict[2]) == 2

        assert np.array_equal(result.samples_dict[1][0].numpy(), np.array([[0], [0]]))
        assert np.array_equal(result.samples_dict[3][0].numpy(), np.array([[0], [0]]))
        assert np.array_equal(result.samples_dict[2][0].numpy(), np.array([[0], [0]]))
        assert np.array_equal(result.samples_dict[2][1].numpy(), np.array([[0], [0]]))


class TestMultipleShotsErrors:
    """Test if errors are raised correctly when using multiple shots."""

    @pytest.mark.parametrize("meng", batch_engines)
    def test_batching_error(self, meng, prog):
        """Check that correct error is raised with batching and shots > 1."""
        with pytest.raises(
            NotImplementedError, match="Batching cannot be used together with multiple shots."
        ):
            meng.run(prog, **{"shots": 2})

    @pytest.mark.parametrize("meng", engines)
    def test_postselection_error(self, meng):
        """Check that correct error is raised with post-selection and shots > 1."""
        prog = sf.Program(2)
        with prog.context as q:
            ops.MeasureFock(select=0) | q[0]

        with pytest.raises(
            NotImplementedError, match="Post-selection cannot be used together with multiple shots."
        ):
            meng.run(prog, **{"shots": 2})

    @pytest.mark.parametrize("meng", engines)
    def test_feedforward_error(self, meng):
        """Check that correct error is raised with feed-forwarding and shots > 1."""
        prog = sf.Program(2)
        with prog.context as q:
            ops.MeasureFock() | q[0]
            ops.Dgate(q[0].par, 0) | q[1]

        with pytest.raises(
            NotImplementedError,
            match="Feed-forwarding of measurements cannot be used together with multiple shots.",
        ):
            meng.run(prog, **{"shots": 2})
