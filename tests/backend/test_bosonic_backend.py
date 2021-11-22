# Copyright 2021 Xanadu Quantum Technologies Inc.

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
Unit tests for backends.bosonicbackend.backend.py.
"""

import numpy as np
import strawberryfields as sf
import strawberryfields.backends.bosonicbackend.backend as bosonic
import pytest
import sympy

pytestmark = pytest.mark.bosonic

ALPHA_VALS = np.linspace(0, 1, 5)
PHI_VALS = np.linspace(0, np.pi, 3)
FOCK_VALS = np.arange(5, dtype=int)
r_fock = 0.05
EPS_VALS = np.array([0.01, 0.05, 0.1, 0.5])
R_VALS = np.linspace(-1, 1, 5)
SHOTS_VALS = np.array([1, 19, 100])


class TestKronList:
    """Test kron_list function from the bosonic backend."""

    def test_kron_list(self):
        l1 = [1, 2]
        l2 = [3, 4, 5]
        list_compare = [3, 4, 5, 6, 8, 10]
        assert np.allclose(list_compare, bosonic.kron_list([l1, l2]))


class TestParameterChecker:
    """Test parameter_checker function from the bosonic backend."""

    def test_parameter_checker(self):
        symbolic_param = sympy.Expr()
        params = []
        assert not bosonic.parameter_checker(params)

        params = [1, "real", 3.0, [4 + 1j, 5], [[3.5, 4.8, "complex"]], np.array([7, 9]), range(3)]
        assert not bosonic.parameter_checker(params)

        params = [1, "real", symbolic_param]
        assert bosonic.parameter_checker(params)

        params = [1, [3, symbolic_param]]
        assert bosonic.parameter_checker(params)


class TestBosonicCatStates:
    r"""Tests cat state method of the BosonicBackend class."""

    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    @pytest.mark.parametrize("phi", PHI_VALS)
    def test_cat_complex(self, alpha, phi):
        r"""Checks the complex cat state representation."""
        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.Catstate(alpha, phi) | q[0]

        backend = bosonic.BosonicBackend()
        backend.run_prog(prog)
        state = backend.state()

        if alpha != 0:
            # Check shapes
            assert state.num_weights == 4
            assert state.weights().shape == (4,)
            assert np.allclose(sum(state.weights()), 1)
            assert state.means().shape == (4, 2)
            assert state.covs().shape == (4, 2, 2)
            covs_compare = np.tile(np.eye(2) * sf.hbar / 2, (4, 1, 1))
            assert np.allclose(state.covs(), covs_compare)

            # Weights should be real if phi is an integer
            if phi % 1 == 0:
                assert np.isreal(state.weights()).all()
            else:
                assert np.iscomplexobj(state.weights())
            # Covs should be real, means complex
            assert np.isreal(state.covs()).all()
            assert np.iscomplexobj(state.means())
        else:
            assert state.num_weights == 1
            assert state.weights().shape == (1,)
            assert np.allclose(sum(state.weights()), 1)
            assert state.means().shape == (1, 2)
            assert state.covs().shape == (1, 2, 2)
            covs_compare = np.tile(np.eye(2) * sf.hbar / 2, (1, 1, 1))
            assert np.allclose(state.covs(), covs_compare)

    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    @pytest.mark.parametrize("phi", PHI_VALS)
    def test_cat_real(self, alpha, phi):
        r"""Checks the real cat state representation produces real weights,
        means and covariances."""
        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.Catstate(alpha, phi, representation="real", ampl_cutoff=1e-6, D=1) | q[0]

        backend = bosonic.BosonicBackend()
        backend.run_prog(prog)
        state = backend.state()
        num_weights = state.num_weights

        assert np.allclose(sum(state.weights()), 1)
        assert state.means().shape == (num_weights, 2)
        assert state.covs().shape == (num_weights, 2, 2)

        # Weights, means and covs should be real
        assert np.allclose(state.weights().real, state.weights())
        assert np.allclose(state.means().real, state.means())
        assert np.allclose(state.covs().real, state.covs())

    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    @pytest.mark.parametrize("phi", PHI_VALS)
    def test_cat_real_cutoff_and_D(self, alpha, phi):
        r"""Checks that for the real cat state representation a
        low cutoff and low D produce fewer weights when alpha !=0."""
        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.Catstate(alpha, phi, representation="real", ampl_cutoff=1e-6, D=1) | q[0]

        backend = bosonic.BosonicBackend()
        backend.run_prog(prog)
        state = backend.state()
        num_weights_low = state.num_weights

        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.Catstate(alpha, phi, representation="real", ampl_cutoff=1e-6, D=10) | q[0]

        backend = bosonic.BosonicBackend()
        backend.run_prog(prog)
        state = backend.state()
        num_weights_high = state.num_weights

        assert np.allclose(sum(state.weights()), 1)
        assert state.means().shape == (num_weights_high, 2)
        assert state.covs().shape == (num_weights_high, 2, 2)

        if alpha != 0:
            assert num_weights_low < num_weights_high
        else:
            assert num_weights_low == num_weights_high

    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("representation", ["complex", "real"])
    def test_cat_state_wigners(self, alpha, phi, representation):
        r"""Checks that the real and complex cat state representations
        have the same Wigner functions as the cat state from the Fock
        backend."""
        x = np.linspace(-2 * alpha, 2 * alpha, 100)
        p = np.linspace(-2 * alpha, 2 * alpha, 100)

        prog_complex_cat = sf.Program(1)
        with prog_complex_cat.context as qb:
            sf.ops.Catstate(alpha, phi, representation=representation) | qb[0]

        backend_bosonic = bosonic.BosonicBackend()
        backend_bosonic.run_prog(prog_complex_cat)
        wigner_bosonic = backend_bosonic.state().wigner(0, x, p)

        prog_cat_fock = sf.Program(1)
        with prog_cat_fock.context as qf:
            if alpha != 0:
                sf.ops.Catstate(alpha, phi) | qf[0]
            else:
                sf.ops.Vacuum() | qf[0]
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 20})
        results = eng.run(prog_cat_fock)
        wigner_fock = results.state.wigner(0, x, p)

        assert np.allclose(wigner_bosonic, wigner_fock, rtol=1e-3, atol=1e-6)

    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("representation", ["complex", "real"])
    @pytest.mark.parametrize("p,expected_parity", [(0, 1), (1, -1)])
    def test_cat_state_parity(self, alpha, phi, representation, p, expected_parity):
        r"""Checks that the real and complex cat state representations
        yield the correct parity."""
        prog_cat = sf.Program(1)
        with prog_cat.context as q:
            sf.ops.Catstate(alpha, phi, p, representation) | q[0]

        backend = bosonic.BosonicBackend()
        backend.run_prog(prog_cat)
        state = backend.state()
        parity = state.parity_expectation([0])

        # for p = 0, should yield parity of 1
        # for p = 1, should yield parity of -1 unless alpha == 0
        if not (p == 1 and alpha == 0):
            assert np.allclose(parity, expected_parity)

    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    @pytest.mark.parametrize("phi", PHI_VALS)
    def test_cat_marginal(self, alpha, phi):
        """Tests marginal method in BaseBosonicState."""
        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.Catstate(alpha, phi) | q

        backend = bosonic.BosonicBackend()
        backend.run_prog(prog)
        state = backend.state()

        scale = np.sqrt(sf.hbar)
        x = np.linspace(-6, 6, 1000) * scale

        marginal = state.marginal(0, x)

        # Calculate the wavefunction directly
        alpha_complex = alpha * np.exp(1j * phi)
        disp = np.sqrt(2 * sf.hbar) * alpha_complex
        norm = 1 / np.sqrt(2 + 2 * np.exp(-2 * abs(alpha_complex) ** 2))
        psi = np.exp(-((x - disp) ** 2) / (2 * sf.hbar))
        psi += np.exp(-((x + disp) ** 2) / (2 * sf.hbar))
        psi *= norm * np.exp(-alpha_complex.imag ** 2) / (np.pi * sf.hbar) ** 0.25

        assert np.allclose(marginal, abs(psi) ** 2)

    def test_cat_threshold(self):
        r"""Tests threshold measurement applied to a cat Bell state."""
        # For large alpha, a beamsplitter effectively creates a cat Bell pair.
        # The Bell pair is such that when there is vacuum in one mode, there is a
        # cat state with large alpha (i.e. non-vacuum) in the other mode.
        # This means the threshold outcomes should always be anti-correlated.
        alpha = 4
        phi = 0
        num_repeats = 50
        for _ in range(num_repeats):
            prog = sf.Program(2)
            with prog.context as q:
                sf.ops.Catstate(alpha, phi) | q[0]
                sf.ops.Catstate(alpha, phi) | q[1]
                sf.ops.BSgate() | (q[0], q[1])
                sf.ops.MeasureThreshold() | q[0]
                sf.ops.MeasureThreshold() | q[1]

            backend = bosonic.BosonicBackend()
            _, results = backend.run_prog(prog)
            res0, res1 = results[0][0][0], results[1][0][0]
            assert ([res0, res1] == [0, 1]) or ([res0, res1] == [1, 0])


class TestBosonicFockStates:
    r"""Tests fock state method of the BosonicBackend class."""

    @pytest.mark.parametrize("n", FOCK_VALS)
    def test_fock(self, n):
        r"""Checks fock states in the bosonic representation."""
        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.Fock(n) | q[0]

        backend = bosonic.BosonicBackend()
        backend.run_prog(prog)
        state = backend.state()

        # Check shapes
        assert state.num_weights == n + 1
        assert state.weights().shape == (n + 1,)
        assert np.allclose(sum(state.weights()), 1)
        assert state.means().shape == (n + 1, 2)
        assert state.covs().shape == (n + 1, 2, 2)

        # Check mean photon is close to n
        mean, var = state.mean_photon(0)
        assert np.allclose(mean, n, atol=r_fock)
        assert np.allclose(var, 0, atol=r_fock)

        # Weights, means and covs should be real
        assert np.allclose(state.weights().real, state.weights())
        assert np.allclose(state.means().real, state.means())
        assert np.allclose(state.covs().real, state.covs())

        if n == 0:
            covs_compare = np.tile(np.eye(2) * sf.hbar / 2, (1, 1, 1))
            assert np.allclose(state.covs(), covs_compare)
            assert np.allclose(state.fidelity_vacuum(), 1)

    @pytest.mark.parametrize("n", FOCK_VALS)
    def test_fock_state_wigners(self, n):
        r"""Checks that fock state Wigner functions in the bosonic and Fock
        backends match."""
        x = np.linspace(-n, n, 100)
        p = np.linspace(-n, n, 100)

        prog = sf.Program(1)
        with prog as q:
            sf.ops.Fock(n) | q[0]

        backend = bosonic.BosonicBackend()
        backend.run_prog(prog)
        wigner_bosonic = backend.state().wigner(0, x, p)

        eng = sf.Engine("fock", backend_options={"cutoff_dim": int(n + 1)})
        results = eng.run(prog)
        wigner_fock = results.state.wigner(0, x, p)

        assert np.allclose(wigner_fock, wigner_bosonic, atol=r_fock)

    @pytest.mark.parametrize("n", FOCK_VALS)
    def test_fock_state_parity(self, n):
        r"""Checks that fock states have the right parity."""
        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.Fock(n) | q[0]

        backend = bosonic.BosonicBackend()
        backend.run_prog(prog)
        state = backend.state()
        assert np.allclose(state.parity_expectation([0]), (-1.0) ** n, atol=r_fock)

    @pytest.mark.parametrize("n", FOCK_VALS)
    def test_fock_threshold(self, n):
        r"""Tests that Fock states n > 0 always yield a click."""
        num_repeats = 50
        for _ in range(num_repeats):
            prog = sf.Program(1)
            with prog.context as q:
                sf.ops.Fock(n) | q[0]
                sf.ops.MeasureThreshold() | q[0]

            backend = bosonic.BosonicBackend()
            _, results = backend.run_prog(prog)
            res0 = results[0][0][0]
            if n == 0:
                assert res0 == 0
            else:
                assert res0 == 1

    def test_hong_ou_mandel_threshold(self):
        r"""Tests Hong-Ou-Mandel interference"""
        num_repeats = 50
        for _ in range(num_repeats):
            prog = sf.Program(2)
            with prog.context as q:
                sf.ops.Fock(1) | q[0]
                sf.ops.Fock(1) | q[1]
                sf.ops.BSgate() | (q[0], q[1])
                sf.ops.MeasureThreshold() | (q[0], q[1])

            backend = bosonic.BosonicBackend()
            results = backend.run_prog(prog)
            _, results = backend.run_prog(prog)
            res0, res1 = results[0][0][0], results[1][0][0]
            assert ([res0, res1] == [0, 1]) or ([res0, res1] == [1, 0])

    def test_g2_threshold(self):
        r"""Tests that the g^2 of a single photon is zero"""
        num_repeats = 50
        for _ in range(num_repeats):
            prog = sf.Program(2)
            with prog.context as q:
                sf.ops.Fock(1) | q[0]
                sf.ops.BSgate() | (q[0], q[1])
                sf.ops.MeasureThreshold() | (q[0], q[1])

            backend = bosonic.BosonicBackend()
            results = backend.run_prog(prog)
            _, results = backend.run_prog(prog)
            res0, res1 = results[0][0][0], results[1][0][0]
            assert ([res0, res1] == [0, 1]) or ([res0, res1] == [1, 0])


class TestBosonicGKPStates:
    r"""Tests the gkp method of the BosonicBackend class."""

    @pytest.mark.parametrize("eps", EPS_VALS)
    def test_gkp(self, eps):
        r"""Checks the GKP state weights, means and covariances are correct."""
        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.GKP(epsilon=eps) | q[0]

        backend = bosonic.BosonicBackend()
        backend.run_prog(prog)
        state = backend.state()
        num_weights = state.num_weights
        assert state.weights().shape == (num_weights,)
        assert np.allclose(sum(state.weights()), 1)
        assert state.means().shape == (num_weights, 2)
        assert state.covs().shape == (num_weights, 2, 2)

        # Weights, means and covs should be real
        assert np.allclose(state.weights().real, state.weights())
        assert np.allclose(state.means().real, state.means())
        assert np.allclose(state.covs().real, state.covs())

        # Covariance should be diagonal with these entries
        cov_val = (1 - np.exp(-2 * eps)) / (1 + np.exp(-2 * eps)) * sf.hbar / 2
        covs_compare = np.tile(cov_val * np.eye(2), (num_weights, 1, 1))
        assert np.allclose(state.covs(), covs_compare)

        # Means should be integer multiples of sqrt(pi*hbar)/2 times a damping
        damping = 2 * np.exp(-eps) / (1 + np.exp(-2 * eps))
        mean_ints = state.means() / (damping * np.sqrt(np.pi * sf.hbar) / 2)

        # Round to make sure numerical errors are washed out
        mean_ints = np.round(np.real_if_close(mean_ints), 10)
        assert np.allclose(mean_ints % 1, np.zeros(mean_ints.shape))

    @pytest.mark.parametrize("eps", EPS_VALS)
    def test_gkp_logical(self, eps):
        r"""Checks that logically equivalent GKP states have
        Wigner functions that agree."""
        x = np.linspace(-3 * np.sqrt(np.pi), 3 * np.sqrt(np.pi), 40)
        p = np.copy(x)

        # Prepare GKP 0 and apply Hadamard
        prog_0H = sf.Program(1)
        with prog_0H.context as q0:
            sf.ops.GKP(epsilon=eps) | q0[0]
            sf.ops.Rgate(np.pi / 2) | q0[0]

        backend_0H = bosonic.BosonicBackend()
        backend_0H.run_prog(prog_0H)
        state_0H = backend_0H.state()
        wigner_0H = state_0H.wigner(0, x, p)

        # Prepare GKP +
        prog_plus = sf.Program(1)
        with prog_plus.context as qp:
            sf.ops.GKP(state=[np.pi / 2, 0], epsilon=eps) | qp[0]

        backend_plus = bosonic.BosonicBackend()
        backend_plus.run_prog(prog_plus)
        state_plus = backend_plus.state()
        wigner_plus = state_plus.wigner(0, x, p)

        assert np.allclose(wigner_0H, wigner_plus)

    @pytest.mark.parametrize("eps", EPS_VALS)
    def test_gkp_state_parity(self, eps):
        r"""Checks that GKP states yield the right parity."""
        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.GKP(epsilon=eps) | q[0]

        backend = bosonic.BosonicBackend()
        backend.run_prog(prog)
        state = backend.state()
        assert np.allclose(state.parity_expectation([0]), 1, atol=r_fock)

    def test_gkp_complex(self):
        r"""Checks that trying to call a complex representation
        raises an error.

        This test can be updated once the complex representation is
        implemented."""
        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.GKP(representation="complex") | q[0]

        backend = bosonic.BosonicBackend()
        with pytest.raises(NotImplementedError):
            backend.run_prog(prog)

    @pytest.mark.parametrize("fock_eps", [0.1, 0.2, 0.3])
    def test_gkp_wigner_compare_backends(self, fock_eps):
        """Test correct Wigner function are generated by comparing fock and bosonic outputs.
        Since the fock backend cannot simulate very small epsilon, we restrict to fock_eps>0.1."""
        theta = np.pi / 8
        phi = np.pi / 6
        ket = [theta, phi]
        xvec = np.linspace(-1, 1, 200)
        prog_fock = sf.Program(1)
        with prog_fock.context as q:
            sf.ops.GKP(epsilon=fock_eps, state=ket) | q
        cutoff = 100
        hbar = 2
        eng_fock = sf.Engine("fock", backend_options={"cutoff_dim": cutoff, "hbar": hbar})
        gkp_fock = eng_fock.run(prog_fock).state
        W_fock = gkp_fock.wigner(0, xvec, xvec)

        prog_bosonic = sf.Program(1)

        with prog_bosonic.context as q:
            sf.ops.GKP(epsilon=fock_eps, state=ket) | q
        hbar = 2
        eng_bosonic = sf.Engine("bosonic", backend_options={"hbar": hbar})
        gkp_bosonic = eng_bosonic.run(prog_bosonic).state
        W_bosonic = gkp_bosonic.wigner(0, xvec, xvec)
        assert np.allclose(W_fock, W_bosonic)


class TestBosonicUserSpecifiedState:
    """Checks the Bosonic preparation method."""

    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("p", [0, 1])
    def test_complex_weight(self, alpha, phi, p):
        r"""Checks that Bosonic creates a state with user-specifed weights, means
        and covariances."""
        dummy_backend = bosonic.BosonicBackend()
        dummy_backend.begin_circuit(1)
        # Get weights, means and covs for a cat state
        weights, means, covs = dummy_backend.prepare_cat(alpha, phi, p, "complex", 0, 2)

        # Initiate the state, but use Bosonic method
        backend = bosonic.BosonicBackend()
        prog = sf.Program(1)
        with prog.context as q1:
            sf.ops.Bosonic(weights, means, covs) | q1[0]
        backend.run_prog(prog)
        state = backend.state()

        # Prepare using the Catstate method
        backend2 = bosonic.BosonicBackend()
        prog2 = sf.Program(1)
        with prog2.context as q2:
            sf.ops.Catstate(alpha, phi, p, D=2) | q2[0]
        backend2.run_prog(prog2)
        state2 = backend2.state()

        # Check the two states are equal
        assert state.__eq__(state2)


class TestBosonicPrograms:
    """Tests that small programs run and return the correct output."""

    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    def test_init_circuit(self, alpha, phi, r):
        """Checks init_circuit only prepares non-Gaussian states and vacuum."""
        prog = sf.Program(3)
        with prog.context as q:
            sf.ops.Catstate(alpha, phi) | q[0]
            # this line should have no effect since it is a Gaussian prep
            # that would be picked up in run_prog, but not init_circuit
            sf.ops.Squeezed(r) | q[1]

        backend = bosonic.BosonicBackend()
        backend.init_circuit(prog)
        state = backend.state()

        # Check that second and third mode are still in vacuum
        weights, means, covs = state.reduced_bosonic([1, 2])

        mean_compare = np.zeros(4)
        means_compare = np.tile(mean_compare, (4, 1))
        assert np.allclose(means, means_compare)

        cov_compare = np.diag([1, 1, 1, 1])
        covs_compare = np.tile(cov_compare * sf.hbar / 2, (4, 1, 1))
        assert np.allclose(covs, covs_compare)

    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    def test_different_preparations(self, alpha, phi, r):
        """Runs a program with non-Gaussian and Gaussian preparations."""
        prog = sf.Program(3)

        with prog.context as q:
            sf.ops.Catstate(alpha, phi) | q[0]
            sf.ops.Squeezed(r) | q[1]

        backend = bosonic.BosonicBackend()
        backend.run_prog(prog)
        state = backend.state()

        # Check output shapes
        if alpha != 0:
            assert state.weights().shape == (4,)
            assert np.allclose(sum(state.weights()), 1)
            assert state.means().shape == (4, 6)
            assert state.covs().shape == (4, 6, 6)

        else:
            assert state.weights().shape == (1,)
            assert np.allclose(sum(state.weights()), 1)
            assert state.means().shape == (1, 6)
            assert state.covs().shape == (1, 6, 6)

        # Check covariance is correct
        cov_compare = np.diag([1, 1, np.exp(-2 * r), np.exp(2 * r), 1, 1])
        covs_compare = np.tile(cov_compare * sf.hbar / 2, (4, 1, 1))
        assert np.allclose(state.covs(), covs_compare)

    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    def test_measurement(self, alpha, phi, r):
        """Runs a program with measurements."""
        prog = sf.Program(2)
        with prog.context as q:
            sf.ops.Catstate(alpha, phi) | q[0]
            sf.ops.Squeezed(r) | q[1]
            sf.ops.MeasureX | q[0]
            sf.ops.MeasureHD | q[1]
        backend = bosonic.BosonicBackend()
        applied, samples_dict = backend.run_prog(prog)
        state = backend.state()

        # Check output is vacuum since everything was measured
        assert np.allclose(state.fidelity_vacuum(), 1)

        # Check samples
        for i in range(2):
            assert i in samples_dict.keys()
            assert samples_dict[i][0].shape == (1,)

    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    @pytest.mark.parametrize("phi", PHI_VALS)
    @pytest.mark.parametrize("shots", SHOTS_VALS)
    def test_measurement_many_shots(self, alpha, phi, shots):
        """Runs a program with measurements."""
        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.Catstate(alpha, phi) | q[0]
            sf.ops.MeasureHomodyne(0) | q[0]

        backend = bosonic.BosonicBackend()
        applied, samples_dict = backend.run_prog(prog, shots=shots)
        state = backend.state()

        # Check output is vacuum since everything was measured
        assert np.allclose(state.fidelity_vacuum(), 1)

        # Check samples
        assert 0 in samples_dict.keys()
        assert samples_dict[0][0].shape == (int(shots),)

    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    @pytest.mark.parametrize("r", R_VALS)
    def test_mb_gates(self, alpha, r):
        """Runs a program with measurement-based gates."""
        prog = sf.Program(2)
        with prog.context as q:
            sf.ops.MSgate(1, 0, 1, 1, avg=True) | q[0]
            sf.ops.MSgate(1, 0, 1, 1, avg=False) | q[1]
            sf.ops.MSgate(1, 0, 1, 1, avg=False) | q[1]
        backend = bosonic.BosonicBackend()
        applied, samples_dict = backend.run_prog(prog)

        # Check number of active modes did not change
        assert backend.get_modes() == [0, 1]

        # Check samples for the data modes are empty
        assert samples_dict == {}

        # Check ancilla samples exist for the second mode
        ancilla_samples = backend.ancillae_samples_dict
        assert 1 in ancilla_samples.keys()
        assert len(ancilla_samples[1]) == 2

    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    @pytest.mark.parametrize("phi", PHI_VALS)
    def test_add_new_mode(self, alpha, phi):
        """Tests adding a new mode in a program context."""
        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.Vacuum() | q[0]
            q = q + (sf.ops.New(n=1),)
            sf.ops.Catstate(alpha, phi) | q[1]
        backend = bosonic.BosonicBackend()
        backend.run_prog(prog)
        state = backend.state()
        # Check output shapes
        if alpha != 0:
            assert state.weights().shape == (4,)
            assert np.allclose(sum(state.weights()), 1)
            assert state.means().shape == (4, 4)
            assert state.covs().shape == (4, 4, 4)

        else:
            assert state.weights().shape == (1,)
            assert np.allclose(sum(state.weights()), 1)
            assert state.means().shape == (1, 4)
            assert state.covs().shape == (1, 4, 4)

    def test_raised_errors(self):
        """Runs programs with operations that are not applicable
        or have not been implemented."""
        from strawberryfields.backends.base import NotApplicableError

        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.Kgate(1) | q[0]
        backend = bosonic.BosonicBackend()
        with pytest.raises(NotApplicableError):
            backend.run_prog(prog)

    def test_non_initial_prep_error(self):
        """Tests that more than one non-Gaussian state preparations in the same mode
        raises an error."""
        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.Catstate() | q[0]
            sf.ops.GKP() | q[0]
        backend = bosonic.BosonicBackend()
        with pytest.raises(NotImplementedError):
            backend.run_prog(prog)
