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
r"""Unit tests for program.py"""
import textwrap
import pytest

pytestmark = pytest.mark.frontend

import numpy as np

import strawberryfields as sf

from strawberryfields import program
from strawberryfields import ops

from strawberryfields import devicespecs
from strawberryfields.devicespecs.device_specs import DeviceSpecs


# make test deterministic
np.random.random(42)
A = np.random.random()


# all single-mode gates with at least one parameter
single_mode_gates = [x for x in ops.one_args_gates + ops.two_args_gates if x.ns == 1]


@pytest.fixture
def eng(backend):
    """Engine fixture."""
    return sf.LocalEngine(backend)


@pytest.fixture
def prog():
    """Program fixture."""
    return sf.Program(2)


@pytest.fixture
def permute_gates():
    """Returns a list containing an instance of each
    single mode gate, with random parameters"""
    params = np.random.random(len(single_mode_gates))
    gates = [G(p) for G, p in zip(single_mode_gates, params)]

    # random permutation
    return np.random.permutation(gates)


class TestProgram:
    """Tests the Program class."""

    def test_with_block(self, prog):
        """Gate application using a with block."""
        # command queue is empty
        assert len(prog) == 0

        with prog.context as q:
            ops.Dgate(0.5) | q[0]
        # now there is one gate in the queue
        assert len(prog) == 1

        with prog.context as q:
            ops.BSgate(0.5, 0.3) | (q[1], q[0])
        assert len(prog) == 2

    def test_identity_command(self, prog):
        """Tests that the None command acts as the identity"""
        identity = program.Command(None, prog.register[0])
        prog.circuit.append(identity)
        assert len(prog) == 1
        prog = prog.compile("base")
        assert len(prog) == 0

    def test_parent_program(self):
        """Continuing one program with another."""
        D = ops.Dgate(0.5)
        prog = sf.Program(3)
        with prog.context as q:
            D | q[1]
            ops.Del | q[0]
        cont = sf.Program(prog)
        with cont.context as q:
            D | q[0]
            r = ops.New(1)
            D | r
        assert cont.can_follow(prog)
        assert prog.reg_refs == cont.init_reg_refs
        assert prog.unused_indices == cont.init_unused_indices

    def test_print_commands(self, eng, prog):
        """Program.print and Engine.print_applied return correct strings."""
        prog = sf.Program(2)

        # store the result of the print command in list res
        res = []
        # use a print function that simply appends the operation
        # name to the results list
        print_fn = lambda x: res.append(x.__str__())

        # prog should now be empty
        prog.print(print_fn)
        assert res == []

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
            ops.Dgate(bob).H | charlie
            ops.Del | bob
            ops.MeasureX | charlie

        res = []
        prog.print(print_fn)

        expected = [
            "Dgate(0.5, 0) | (q[0])",
            "BSgate(6.283, 1.571) | (q[0], q[1])",
            "Del | (q[0])",
            "Rgate(3.142) | (q[1])",
            "New(1)",
            "BSgate(6.283, 1.571) | (q[1], q[2])",
            "MeasureX | (q[1])",
            "Dgate(RR(q[1]), 0).H | (q[2])",
            "Del | (q[1])",
            "MeasureX | (q[2])",
        ]

        assert res == expected

        # NOTE optimization can change gate order
        state = eng.run(prog, compile_options={'optimize': False})
        res = []
        eng.print_applied(print_fn)
        assert res == ["Run 0:"] + expected


class TestRegRefs:
    """Testing register references."""

    def test_corrupt_index(self, prog):
        """User messes up the RegRef indices."""
        with prog.context as q:
            q[0].ind = 1
            with pytest.raises(program.RegRefError, match="RegRef state has become inconsistent"):
                ops.Dgate(0.5) | q[0]

    def test_nonexistent_index(self, prog):
        """Acting on a non-existent mode raises an error."""
        with prog.context as q:
            with pytest.raises(IndexError):
                ops.Dgate(0.5) | q[3]
            with pytest.raises(program.RegRefError, match="does not exist"):
                ops.Dgate(0.5) | 3

    def test_deleted_index(self, prog):
        """Test that acting on a deleted mode raises an error"""
        with prog.context as q:
            ops.Del | q[0]
            with pytest.raises(program.RegRefError, match="been deleted"):
                ops.Dgate(0.5) | 0

            with pytest.raises(program.RegRefError, match="been deleted"):
                ops.Dgate(0.5) | q[0]

    def test_unknown_regref(self, prog):
        """Applying an operation on a RegRef not belonging to the program."""
        r = program.RegRef(3)
        with prog.context:
            with pytest.raises(program.RegRefError, match="Unknown RegRef."):
                ops.Dgate(0.5) | r

    def test_invalid_measurement(self, eng, prog):
        """Cannot use a measurement before it exists."""
        with prog.context as q:
            ops.Dgate(q[0]) | q[1]
        with pytest.raises(program.CircuitError, match="nonexistent measurement result"):
            eng.run(prog)

    def test_invalid_regref(self, prog):
        """Cannot refer to a register with a non-integral or RegRef object."""
        with prog.context:
            with pytest.raises(program.RegRefError, match="using integers and RegRefs"):
                ops.Dgate(0) | 1.2

    def test_regref_val(self, eng, prog):
        """RegRefs storing measured values."""
        with prog.context as q:
            ops.MeasureX | q[0]

        eng.run(prog)
        assert q[0].val is not None
        assert q[1].val is None
        eng.reset()
        # the regrefs are reset as well
        temp = [r.val is None for r in prog.register]
        assert np.all(temp)

        eng.run(prog)
        assert q[0].val is not None
        assert q[1].val is None
        prog._clear_regrefs()
        assert np.all([r.val is None for r in prog.register])


class TestOptimizer:
    """Tests for the Program optimizer"""

    @pytest.mark.parametrize("G", single_mode_gates)
    def test_merge_dagger(self, G):
        """Optimizer merging single-mode gates with their daggered versions."""
        prog = sf.Program(1)
        G = G(A)

        with prog.context:
            G | 0
            G.H | 0

        prog.optimize()
        assert len(prog) == 0

    @pytest.mark.parametrize("G", single_mode_gates)
    def test_merge_negated(self, G):
        """Optimizer merging single-mode gates with their negated versions."""
        prog = sf.Program(1)
        G1 = G(A)
        G2 = G(-A)

        with prog.context:
            G1 | 0
            G2 | 0

        prog.optimize()
        assert len(prog) == 0

    def test_merge_palindromic_cancelling(self, permute_gates):
        """Optimizer merging chains cancel out with palindromic cancelling"""
        prog = sf.Program(3)

        with prog.context:
            for G in permute_gates:
                G | 0
            for G in reversed(permute_gates):
                G.H | 0

        prog.optimize()
        assert len(prog) == 0

    def test_merge_pairwise_cancelling(self, permute_gates):
        """Optimizer merging chains cancel out with pairwise cancelling"""
        prog = sf.Program(3)

        with prog.context:
            for G in permute_gates:
                G | 0
                G.H | 0

        prog.optimize()
        assert len(prog) == 0

    def test_merge_interleaved_chains_two_modes(self, permute_gates):
        """Optimizer merging chains cancel out with palindromic
        cancelling with interleaved chains on two modes"""
        prog = sf.Program(3)

        # create a random vector of 0s and 1s, corresponding
        # to the applied mode of each gates
        modes = np.random.randint(2, size=len(permute_gates))

        with prog.context:
            for G, m in zip(permute_gates, modes):
                G | m
            for G, m in zip(reversed(permute_gates), reversed(modes)):
                G.H | m

        prog.optimize()
        assert len(prog) == 0

    def test_merge_incompatible(self):
        """Test merging of incompatible gates does nothing"""
        prog = sf.Program(3)

        with prog.context:
            ops.Xgate(0.6) | 0
            ops.Zgate(0.2) | 0

        prog.optimize()
        assert len(prog) == 2


class TestValidation:
    """Test for Program device validation within
    the compile() method."""

    def test_incorrect_modes(self, monkeypatch):
        """Test that an exception is raised if the device
        is called with the incorrect number of modes"""

        class DummyDevice(DeviceSpecs):
            """A device with 2 modes"""
            modes = 2
            remote = False
            local = True
            interactive = True
            primitives = {'S2gate', 'Interferometer'}
            decompositions = set()

        dev = DummyDevice()

        prog = sf.Program(3)

        with prog.context as q:
            ops.S2gate(0.6) | [q[0], q[1]]
            ops.S2gate(0.6) | [q[1], q[2]]

        with monkeypatch.context() as m:
            # monkeypatch our DummyDevice into the
            # backend database
            db =  {'dummy': DummyDevice}
            m.setattr("strawberryfields.devicespecs.backend_specs", db)

            with pytest.raises(program.CircuitError, match="requires 3 modes"):
                new_prog = prog.compile(backend='dummy')

    def test_no_decompositions(self, monkeypatch):
        """Test that no decompositions take
        place if the device doesn't support it."""

        class DummyDevice(DeviceSpecs):
            """A device with no decompositions"""
            modes = None
            remote = False
            local = True
            interactive = True
            primitives = {'S2gate', 'Interferometer'}
            decompositions = set()

        dev = DummyDevice()

        prog = sf.Program(3)
        U = np.array([[0, 1], [1, 0]])

        with prog.context as q:
            ops.S2gate(0.6) | [q[0], q[1]]
            ops.Interferometer(U) | [q[0], q[1]]

        with monkeypatch.context() as m:
            # monkeypatch our DummyDevice into the
            # backend database
            db =  {'dummy': DummyDevice}
            m.setattr("strawberryfields.devicespecs.backend_specs", db)
            new_prog = prog.compile(backend='dummy')

        # check compiled program only has two gates
        assert len(new_prog) == 2

        # test gates are correct
        circuit = new_prog.circuit
        assert circuit[0].op.__class__.__name__ == "S2gate"
        assert circuit[1].op.__class__.__name__ == "Interferometer"

    def test_decompositions(self, monkeypatch):
        """Test that decompositions take
        place if the device requests it."""

        class DummyDevice(DeviceSpecs):
            """A device with no decompositions"""
            modes = None
            remote = False
            local = True
            interactive = True
            primitives = {'S2gate', 'Interferometer', 'BSgate', 'Sgate'}
            decompositions = {'S2gate': {}}

        dev = DummyDevice()

        prog = sf.Program(3)
        U = np.array([[0, 1], [1, 0]])

        with prog.context as q:
            ops.S2gate(0.6) | [q[0], q[1]]
            ops.Interferometer(U) | [q[0], q[1]]

        with monkeypatch.context() as m:
            # monkeypatch our DummyDevice into the
            # backend database
            db =  {'dummy': DummyDevice}
            m.setattr("strawberryfields.devicespecs.backend_specs", db)
            new_prog = prog.compile(backend='dummy')

        # check compiled program now has 5 gates
        # the S2gate should decompose into two BS and two Sgates
        assert len(new_prog) == 5

        # test gates are correct
        circuit = new_prog.circuit
        assert circuit[0].op.__class__.__name__ == "BSgate"
        assert circuit[1].op.__class__.__name__ == "Sgate"
        assert circuit[2].op.__class__.__name__ == "Sgate"
        assert circuit[3].op.__class__.__name__ == "BSgate"
        assert circuit[4].op.__class__.__name__ == "Interferometer"

    def test_invalid_decompositions(self, monkeypatch):
        """Test that an exception is raised if the device spec
        requests a decomposition that doesn't exist"""

        class DummyDevice(DeviceSpecs):
            """A device with no decompositions"""
            modes = None
            remote = False
            local = True
            interactive = True
            primitives = {'Rgate', 'Interferometer'}
            decompositions = {'Rgate': {}}

        dev = DummyDevice()

        prog = sf.Program(3)
        U = np.array([[0, 1], [1, 0]])

        with prog.context as q:
            ops.Rgate(0.6) | q[0]
            ops.Interferometer(U) | [q[0], q[1]]

        with monkeypatch.context() as m:
            # monkeypatch our DummyDevice into the
            # backend database
            db =  {'dummy': DummyDevice}
            m.setattr("strawberryfields.devicespecs.backend_specs", db)

            with pytest.raises(NotImplementedError, match="No decomposition available: Rgate"):
                new_prog = prog.compile(backend='dummy')

    def test_invalid_primitive(self):
        """Test that an exception is raised if the program
        contains a primitive not allowed on the device.

        Here, we can simply use the guassian backend and
        the Kerr gate as an existing example.
        """
        prog = sf.Program(3)

        with prog.context as q:
            ops.Kgate(0.6) | q[0]

        with pytest.raises(program.CircuitError, match="Kgate cannot be used with the gaussian backend"):
            new_prog = prog.compile(backend='gaussian')

    def test_user_defined_decomposition_false(self):
        """Test that an operation that is both a primitive AND
        a decomposition (for instance, ops.Gaussian in the gaussian
        backend) can have it's decomposition behaviour user defined.

        In this case, the Gaussian operation should remain after compilation.
        """
        prog = sf.Program(3)
        cov = np.array([[0, 1], [1, 0]])

        with prog.context as q:
            ops.Gaussian(cov, decomp=False) | q[0]

        new_prog = prog.compile(backend='gaussian')

        assert len(new_prog) == 1
        circuit = new_prog.circuit
        assert circuit[0].op.__class__.__name__ == "Gaussian"

    def test_user_defined_decomposition_true(self):
        """Test that an operation that is both a primitive AND
        a decomposition (for instance, ops.Gaussian in the gaussian
        backend) can have it's decomposition behaviour user defined.

        In this case, the Gaussian operation should compile
        to an Sgate.
        """
        prog = sf.Program(3)
        r = 0.453
        cov = np.array([[np.exp(-2*r), 0], [0, np.exp(2*r)]])*sf.hbar/2

        with prog.context:
            ops.Gaussian(cov, decomp=True) | 0

        new_prog = prog.compile(backend='gaussian')

        assert len(new_prog) == 1

        # decomposed gate should be an Sgate
        circuit = new_prog.circuit
        assert circuit[0].op.__class__.__name__ == "Sgate"
        assert circuit[0].op.p[0] == r

    def test_topology_validation(self, monkeypatch):
        """Test compilation properly matches the device topology"""

        class DummyDevice(DeviceSpecs):
            """A device with no decompositions"""
            modes = None
            remote = False
            local = True
            interactive = True
            primitives = {'Sgate', 'BSgate', 'Dgate', 'MeasureFock'}
            decompositions = set()

            circuit = textwrap.dedent(
                """\
                name test
                version 0.0

                Sgate({sq}, 0) | 0
                Dgate(-7.123) | 1
                BSgate({theta}) | 0, 1
                MeasureFock() | 0
                MeasureFock() | 2
                """
            )

        dev = DummyDevice()

        prog = sf.Program(3)

        with prog.context as q:
            # the circuit given below is an
            # isomorphism of the one provided above
            # in circuit, so should validate.
            ops.Measure | q[2]
            ops.Dgate(-7.123) | q[1]
            ops.Sgate(0.543) | q[0]
            ops.BSgate(-0.32) | (q[0], q[1])
            ops.MeasureFock() | q[0]

        with monkeypatch.context() as m:
            # monkeypatch our DummyDevice into the
            # backend database
            db =  {'dummy': DummyDevice}
            m.setattr("strawberryfields.devicespecs.backend_specs", db)
            new_prog = prog.compile(backend='dummy')

        # no exception should be raised; topology correctly validated
        assert len(new_prog) == 5

    def test_invalid_topology(self, monkeypatch):
        """Test compilation raises exception if toplogy not matched"""

        class DummyDevice(DeviceSpecs):
            """A device with no decompositions"""
            modes = None
            remote = False
            local = True
            interactive = True
            primitives = {'Sgate', 'BSgate', 'Dgate', 'MeasureFock'}
            decompositions = set()

            circuit = textwrap.dedent(
                """\
                name test
                version 0.0

                Sgate({sq}, 0) | 0
                Dgate(-7.123) | 1
                BSgate({theta}) | 0, 1
                MeasureFock() | 0
                MeasureFock() | 2
                """
            )

        dev = DummyDevice()

        prog = sf.Program(3)

        with prog.context as q:
            # the circuit given below is NOT an
            # isomorphism of the one provided above
            # in circuit, as the Sgate
            # comes AFTER the beamsplitter.
            ops.Measure | q[2]
            ops.Dgate(-7.123) | q[1]
            ops.BSgate(-0.32) | (q[0], q[1])
            ops.Sgate(0.543) | q[0]
            ops.MeasureFock() | q[0]

        with monkeypatch.context() as m:
            # monkeypatch our DummyDevice into the
            # backend database
            db =  {'dummy': DummyDevice}
            m.setattr("strawberryfields.devicespecs.backend_specs", db)

            with pytest.raises(program.CircuitError, match="incompatible topology"):
                new_prog = prog.compile(backend='dummy')
