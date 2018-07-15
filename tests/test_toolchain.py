"""
Unit tests for the :mod:`strawberryfields` full toolchain.
"""

import logging
logging.getLogger()

import unittest
import inspect
import itertools

import numpy as np
from numpy.random import (randn, uniform, randint)
from numpy import pi

import tensorflow as tf

# NOTE: strawberryfields must be imported from defaults
from defaults import BaseTest, FockBaseTest, GaussianBaseTest, strawberryfields as sf
from strawberryfields.engine import *
from strawberryfields.ops import *
from strawberryfields.backends import BaseFock


class BasicTests(BaseTest):
    """Implementation-independent tests."""
    num_subsystems = 2
    def setUp(self):
        super().setUp()
        # construct a fresh compiler engine
        self.eng = Engine(num_subsystems=self.num_subsystems, hbar=self.hbar)
        # attach the backend (NOTE: self.backend is shared between the tests, but the parent class re-initializes it in its setUp() method)
        self.eng.backend = self.backend

    def test_identity_command(self):
        """Tests that the None command acts as the identity"""
        q = self.eng.register
        Identity = Command(None, q[0])
        self.eng.cmd_queue = [Identity]
        self.eng.run()
        res = self.eng._cmd_applied_all()
        self.assertEqual(res, [])

    def test_engine(self):
        "Basic engine tests."
        self.logTestName()

        q = self.eng.register

        # backed is initialized to vacuum state
        self.assertAllTrue(self.backend.is_vacuum(self.tol))

        D = Dgate(0.5)
        #D = Sgate(0.543)
        with self.eng:
            # trying to act on a nonexistent mode
            with self.assertRaisesRegex(RegRefError, "does not exist"):
                D | self.eng.num_subsystems

            Del | q[0]
            # trying to act on a deleted mode
            with self.assertRaisesRegex(RegRefError, "been deleted"):
                D | 0

        self.eng.reset()
        qNew = RegRef(3)
        with self.eng:
            # trying to act on RegRef that is not part of the engine
            with self.assertRaisesRegex(RegRefError, "Unknown RegRef."):
                D | qNew

        self.eng.reset()
        # successful gate application
        with self.eng:
            D | 0
        self.assertEqual(len(self.eng.cmd_queue), 1)

        # print the queue contents
        self.eng.print_queue(print_fn=logging.debug)

        # clear the command queue
        self.eng.reset_queue()
        self.assertEqual(len(self.eng.cmd_queue), 0)

        # fill it again
        with self.eng:
            D | 0

        # run engine
        self.eng.run()
        self.assertEqual(len(self.eng.cmd_queue), 0)
        self.assertEqual(len(self.eng.cmd_applied), 1)
        self.assertEqual(len(self.eng.cmd_applied[0]), 1)

        # see what the state looks like
        temp = self.backend.state()

        # reset the backend state to vacuum
        self.eng.reset()
        self.assertAllTrue(self.backend.is_vacuum(self.tol))

        # Should never happen!
        q[0].ind = 1
        with self.eng:
            with self.assertRaisesRegex(RegRefError, "Should never happen!"):
                D | q[0]

    def test_gate_dagger(self):
        "Dagger functionality of the gates."
        self.logTestName()

        def test_gate(G):
            self.eng.reset()
            q = self.eng.register
            with self.eng:
                if G.ns == 1:
                    G   | q[0]
                    G.H | q[0]
                else:
                    G   | (q[0],q[1])
                    G.H | (q[0],q[1])
            state = self.eng.run()
            # state norm must be invariant
            if isinstance(self.eng.backend, BaseFock):
                self.assertAllAlmostEqual(state.trace(), 1, delta=self.tol,
                                       msg="Trace after G.H * G is not 1 for gate G={}".format(G))
            # we must end up back in vacuum since G and G.H cancel each other
            self.assertAllTrue(self.backend.is_vacuum(self.tol),
                                   msg="State is not vacuum after G.H * G for gate G={}".format(G))

        for G in two_args_gates:
            # construct a random gate
            G = G(uniform(high=0.1), uniform(high=2 * pi))
            test_gate(G)

        for G in one_args_gates:
            if G in (Vgate, Kgate, CKgate) and not self.args.fock_support:
                continue  # the V gate cannot be used on Gaussian backends
            # construct a random gate
            G = G(uniform(high=0.25))
            test_gate(G)


    def test_regrefs(self):
        """Testing register references."""
        self.logTestName()
        q = self.eng.register
        # using a measurement result before it exists
        with self.eng:
            Dgate(q[0]) | q[1]

        with self.assertRaisesRegex(CircuitError, "Trying to use a nonexistent measurement result"):
            self.eng.run()

        self.eng.reset()

        # referring to a register with a non-integral or RegRef object
        with self.eng:
            with self.assertRaisesRegex(RegRefError, "using integers and RegRefs"):
                Dgate(0) | 1.2

        self.eng.reset()

        # proper use
        with self.eng:
            MeasureX | q[0]
            Sgate(q[0])   | q[1]
            Dgate(q[0]).H | q[1]  # symbolic hermitian conjugate together with register reference
            Sgate(RR(q[0], lambda x: x**2)) | q[1]
            Dgate(RR(q[0], lambda x: -x)).H | q[1]
        self.eng.run()


    def test_homodyne_measurement(self):
        """Homodyne measurements."""
        self.logTestName()
        q = self.eng.register
        with self.eng:
            Coherent(*randn(2)) | q[0]
            Coherent(*randn(2)) | q[1]
            MeasureX | q[0]
            MeasureP | q[1]
        self.eng.run()
        # homodyne measurements leave the mode in vacuum state
        self.assertAllTrue(self.backend.is_vacuum(self.tol))

        with self.eng:
            Coherent(*randn(2)) | q[0]
            MeasureHomodyne(*randn(1)) | q[0]
        self.eng.run()
        self.assertAllTrue(self.backend.is_vacuum(self.tol))


    def test_program_subroutine(self):
        """Simple quantum program with a subroutine and references."""
        self.logTestName()
        # define some gates
        D = Dgate(0.5)
        BS = BSgate(0.7*pi, pi/2)
        R = Rgate(pi/3)
        # get register references
        alice, bob = self.eng.register

        def subroutine(a, b):
            "Subroutine for the quantum program"
            R   | a
            BS  | (a,b)
            R.H | a

        # main program
        with self.eng:
            All(Vacuum()) | (alice, bob)
            D   | alice
            subroutine(alice, bob)
            BS  | (alice, bob)
            subroutine(bob, alice)

        state = self.eng.run()
        # state norm must be invariant
        if isinstance(self.eng.backend, BaseFock):
            self.assertAllAlmostEqual(state.trace(), 1, delta=self.tol)


    def _check_reg(self, expected_n=None):
        """Compare Engine.register with the mode list returned by the backend.

        They should always be in agreement after Engine.run(), Engine.reset_queue() and Engine.reset().
        """
        rr = self.eng.register
        modes = self.eng.backend.get_modes()
        # number of elements
        self.assertEqual(len(rr), len(modes))
        if expected_n is not None:
            self.assertEqual(len(rr), expected_n)
        # indices match
        self.assertAllEqual([r.ind for r in rr], modes)
        # activity
        self.assertAllTrue([r.active for r in rr])


    def test_create_delete(self):
        """Creating and deleting modes."""
        self.logTestName()
        # define some gates
        D = Dgate(0.5)
        BS = BSgate(2*pi, pi/2)
        R = Rgate(pi)

        # get register references
        reg = self.eng.register
        alice, bob = reg
        with self.eng:
            D        | alice
            BS       | (alice, bob)
            Del      | alice
            R        | bob
            charlie, = New(1)
            BS       | (bob, charlie)
            MeasureX | bob
            Del      | bob
            D.H      | charlie
            MeasureX | charlie
            # trying to act on a deleted mode
            with self.assertRaisesRegex(RegRefError, "been deleted"):
                D | alice
            with self.assertRaisesRegex(RegRefError, "been deleted"):
                D | bob

        #self.eng.print_queue()
        self.eng.optimize()
        state = self.eng.run()

        # state norm must be invariant
        if isinstance(self.eng.backend, BaseFock):
            self.assertAllAlmostEqual(state.trace(), 1, delta=self.tol)

        ## check that reset_queue() restores the latest RegRef checkpoint (created by eng.run() above)
        self.assertTrue(not alice.active)
        self.assertTrue(charlie.active)
        self.eng.reset_queue()
        self.assertTrue(not alice.active)
        self.assertTrue(charlie.active)
        with self.eng:
            diana, = New(1)
            Del | charlie
        self.assertTrue(not charlie.active)
        self.assertTrue(diana.active)
        self.eng.reset_queue()
        self.assertTrue(charlie.active)
        self.assertTrue(not diana.active)

        ## check that reset() works
        self._check_reg(1)
        self.eng.reset()
        new_reg = self.eng.register
        # original number of modes
        self.assertEqual(len(new_reg), len(reg))
        # the regrefs are reset as well
        self.assertAllTrue([r.val is None for r in new_reg])
        self._check_reg(2)


    def test_create_delete_reset(self):
        """Test various use cases creating and deleting modes, together with engine and backend resets."""
        self.logTestName()
        # define some gates
        X = Xgate(0.5)

        reg = self.eng.register
        alice, bob = reg
        eng = self.eng
        state = np.full((len(reg), 2), True)  # states of the subsystems: active, not_measured

        def prog1(q):
            X        | q
            MeasureX | q

        def prog2(q):
            X        | q
            MeasureX | q
            Del      | q
            state[q.ind, 0] = False  # no longer active

        def reset():
            eng.reset()
            state[:] = True

        def check():
            """Check the activity and measurement state of subsystems.

            Activity state is changed right away when a Del command is applied to Engine,
            but a measurement result only becomes available during Engine.run().
            """
            for r, (a, nm) in zip(reg, state):
                self.assertTrue(r.active == a)
                self.assertTrue((r.val is None) == nm)

        def input(p, reg):
            with eng:
                p(reg)
            check()

        def input_and_run(p, reg):
            with eng:
                p(reg)
            check()
            eng.run()
            state[reg.ind, 1] = False  # now measured (assumes p will always measure reg!)
            check()

        reset()
        ## (1) run several independent programs, resetting everything in between
        input_and_run(prog1, alice)
        reset()
        input_and_run(prog2, bob)
        reset()

        ## (2) interactive state evolution in multiple runs
        input_and_run(prog2, alice)
        input_and_run(prog1, bob)
        reset()

        ## (3) repeat a (possibly extended) program fragment (the fragment must not create/delete subsystems!)
        input_and_run(prog1, alice)
        eng.cmd_queue.extend(eng.cmd_applied[-1])
        check()
        input_and_run(prog1, bob)
        reset()

        ## (4) interactive use, "changed my mind"
        input_and_run(prog1, alice)
        input(prog2, bob)
        eng.reset_queue()  # scratch that, back to last checkpoint
        state[bob.ind, 0] = True  # bob should be active again
        check()
        input_and_run(prog1, bob)
        reset()

        ## (5) reset the state, run the same program again to get new measurement samples
        input_and_run(prog1, alice)
        input(prog2, bob)
        eng.reset(keep_history=True)
        state[alice.ind, 1] = True  # measurement result was erased, activity did not change
        state[bob.ind, 1] = True
        check()
        eng.run()
        state[alice.ind, 1] = False
        state[bob.ind, 1] = False
        check()


    def test_eng_reset(self):
        """Test the Engine.reset() features."""
        self.logTestName()

        state = self.eng.run()
        if self.args.fock_support:
            # change the cutoff dimension
            old_cutoff = self.eng.backend.get_cutoff_dim()
            self.assertEqual(state._cutoff, old_cutoff)
            self.assertEqual(self.D, old_cutoff)
            new_cutoff = old_cutoff+1
            self.eng.reset(cutoff_dim=new_cutoff)
            state = self.eng.run()
            temp = self.eng.backend.get_cutoff_dim()
            self.assertEqual(temp, new_cutoff)
            self.assertEqual(state._cutoff, new_cutoff)

    def test_parameters(self):
        """Test using different types of Parameters with different classes of ParOperations."""
        self.logTestName()
        @sf.convert
        def func1(x):
            return abs(2*x**2 -3*x +1)

        @sf.convert
        def func2(x,y):
            return abs(2*x*y -y**2 +0.5)

        kwargs = {}
        r = self.eng.register

        # RegRefTransforms for deferred measurements (note that some operations expect nonnegative parameter values)
        rr_inputs = [RR(r[0], lambda x: x**2), func1(r[0]), func2(*r)]
        rr_pars = tuple(Parameter(k) for k in rr_inputs)

        # other types of parameters
        other_inputs = [0.14]
        if isinstance(self.backend, sf.backends.TFBackend):
            # HACK: add placeholders for tf.Variables
            other_inputs.append(np.inf)
            if self.bsize > 1:
                # test batched input
                other_inputs.append(uniform(size=(self.bsize,)))

        other_pars = tuple(Parameter(k) for k in other_inputs)

        def check(G, par, measure=False):
            "Check a ParOperation/Parameters combination"
            # construct the op using the given tuple of Parameters as args
            G = G(*par)
            with self.eng:
                if measure:
                    # RR parameters require measurements
                    #MeasureX  | r[0]
                    #MeasureP  | r[1]
                    # postselection is much faster
                    #MeasureHomodyne(0, select=0.1)     | r[0]
                    #MeasureHomodyne(pi/2, select=0.2)  | r[1]
                    # faking the measurements entirely is faster still
                    r[0].val = 0.1
                    r[1].val = 0.2
                if G.ns == 1:
                    G | r[0]
                else:
                    G | (r[0], r[1])
            logging.debug(G)
            self.eng.optimize()
            try:
                self.eng.run(**kwargs)
            except NotApplicableError as err:
                # catch unapplicable op/backend combinations here
                logging.debug(err)
                self.eng.reset_queue()  # unsuccessful run means the queue was not emptied.

        scalar_arg_preparations = (Coherent, Squeezed, DisplacedSqueezed, Thermal, Catstate)  # Fock requires an integer parameter
        testset = one_args_gates +two_args_gates +channels +scalar_arg_preparations

        for G in testset:
            sig = inspect.signature(G.__init__)
            n_args = len(sig.parameters) -1  # number of parameters __init__ takes, minus self
            if n_args < 1:
                logging.debug('Unexpected number of args ({}) for {}, check the testset.'.format(n_args, G))

            #n_args = min(n_args, 2)   # shortcut, only test cartesian products up to two parameter types
            # check all combinations of Parameter types
            for p in itertools.product(rr_pars+other_pars, repeat=n_args):
                if isinstance(self.backend, sf.backends.TFBackend):
                    # declare tensorflow Variables here (otherwise they will be on a different graph when we do backend.reset below)
                    # replace placeholders with tf.Variable instances
                    def f(q):
                        if np.any(q.x == np.inf):  # detect the HACK
                            if f.v is None:  # only construct one instance, only when needed
                                f.v = Parameter(tf.Variable(0.8, name='sf_parameter'))
                            return f.v
                        return q
                    f.v = None  # kind of like a static variable in a function in C
                    p = tuple(map(f, p))
                    # create new tf.Session (which will be associated with the default graph) and initialize variables
                    session = tf.Session()
                    kwargs['session'] = session
                    session.run(tf.global_variables_initializer())
                    check(G, p, measure=True)
                    session.close()
                else:
                    check(G, p, measure=True)
                self.eng.backend.reset()

    def test_print_commands(self):
        """Test print_queue and print_applied returns correct strings."""
        self.logTestName()
        # define some gates
        D = Dgate(0.5)
        BS = BSgate(2*pi, pi/2)
        R = Rgate(pi)

        # get register references
        reg = self.eng.register
        alice, bob = reg
        with self.eng:
            D        | alice
            BS       | (alice, bob)
            Del      | alice
            R        | bob
            charlie, = New(1)
            BS       | (bob, charlie)
            MeasureX | bob
            Dgate(bob).H      | charlie
            Del      | bob
            MeasureX | charlie

        res = []
        print_fn = lambda x: res.append(x.__str__())
        self.eng.print_queue(print_fn)

        expected = ['Dgate(0.5, 0) | (q[0])',
                    'BSgate(6.283, 1.571) | (q[0], q[1])',
                    'Del | (q[0])',
                    'Rgate(3.142) | (q[1])',
                    'New(1) ',
                    'BSgate(6.283, 1.571) | (q[1], q[2])',
                    'MeasureX | (q[1])',
                    'Dgate(RR(q[1]), 0).H | (q[2])',
                    'Del | (q[1])',
                    'MeasureX | (q[2])']

        self.assertEqual(res, expected)
        state = self.eng.run()

        # queue should now be empty
        res = []
        print_fn = lambda x: res.append(x.__str__())
        self.eng.print_queue(print_fn)
        self.assertEqual(res, [])

        # print_applied should now not be empty
        res = []
        print_fn = lambda x: res.append(x.__str__())
        self.eng.print_applied(print_fn)
        self.assertEqual(res, ['Run 0:'] + expected)

    def test_no_return_state(self):
        """Tests that engine returns None when no state is requested"""
        self.logTestName()

        q = self.eng.register
        with self.eng:
            Dgate(0.34) | q[0]
        res = self.eng.run(return_state=False)
        self.assertEqual(res, None)

    def test_apply_history(self):
        """Tests the reapply history argument"""
        self.logTestName()

        a = 0.23
        r = 0.1

        def inspect():
            res = []
            print_fn = lambda x: res.append(x.__str__())
            self.eng.print_applied(print_fn)
            return res

        q = self.eng.register
        with self.eng:
            Dgate(a) | q[0]
            Sgate(r) | q[1]

        state1 = self.eng.run()
        expected = ['Run 0:',
                    'Dgate({}, 0) | (q[0])'.format(a),
                    'Sgate({}, 0) | (q[1])'.format(r)]
        self.assertEqual(inspect(), expected)

        # reset backend, but reapply history
        self.backend.reset(pure=self.kwargs["pure"])
        state2 = self.eng.run(backend=self.backend, apply_history=True)
        self.assertEqual(inspect(), expected)
        self.assertEqual(state1, state2)

        # append more commands to the same backend
        with self.eng:
            Rgate(r) | q[0]

        state3 = self.eng.run()
        expected = ['Run 0:',
                     'Dgate({}, 0) | (q[0])'.format(a),
                     'Sgate({}, 0) | (q[1])'.format(r),
                     'Run 1:',
                     'Rgate({}) | (q[0])'.format(r),
                    ]
        self.assertEqual(inspect(), expected)
        self.assertNotEqual(state2, state3)

        # reset backend, but reapply history
        self.backend.reset(pure=self.kwargs["pure"])
        state4 = self.eng.run(backend=self.backend, apply_history=True)
        expected = ['Run 0:',
                     'Dgate({}, 0) | (q[0])'.format(a),
                     'Sgate({}, 0) | (q[1])'.format(r),
                     'Rgate({}) | (q[0])'.format(r),
                    ]
        self.assertEqual(inspect(), expected)
        self.assertEqual(state3, state4)


class FockBasisTests(FockBaseTest):
    """Fock-basis dependent tests."""
    num_subsystems = 2
    def setUp(self):
        super().setUp()
        # construct a compiler engine
        self.eng = Engine(num_subsystems=self.num_subsystems, hbar=self.hbar)
        # attach the backend (NOTE: self.backend is shared between the tests, make sure it is reset)
        self.eng.backend = self.backend
        self.backend.reset()

    def test_fock_measurement(self):
        """Fock measurements."""
        self.logTestName()
        q = self.eng.register
        s = randint(0, self.D, (2,))
        with self.eng:
            Fock(s[0]) | q[0]
            Fock(s[1]) | q[1]
            Measure | q
        self.eng.run()
        # measurement result must be equal to preparation
        for a,b in zip(q, s):
            self.assertAllEqual(a.val, b)
        # Fock measurements put the modes into vacuum state
        self.assertAllTrue(self.backend.is_vacuum(self.tol))



class GaussianTests(GaussianBaseTest):
    """Gaussian-only tests."""
    num_subsystems = 2
    def setUp(self):
        super().setUp()
        # construct a compiler engine
        self.eng = Engine(num_subsystems=self.num_subsystems, hbar=self.hbar)
        # attach the backend (NOTE: self.backend is shared between the tests, make sure it is reset)
        self.eng.backend = self.backend
        self.backend.reset()

    def test_gaussian_measurement(self):
        """Gaussian-only measurements."""
        self.logTestName()
        q = self.eng.register
        with self.eng:
            Coherent(*randn(2)) | q[0]
            Coherent(*randn(2)) | q[1]
            MeasureHD | q[0]
            MeasureHD | q[1]
        self.eng.run()
        # heterodyne measurements leave the mode in vacuum state
        self.assertAllTrue(self.backend.is_vacuum(self.tol))


if __name__ == '__main__':
    print('Testing Strawberry Fields version ' + sf.version() + ', full toolchain.')

    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTests, FockBasisTests, GaussianTests):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
