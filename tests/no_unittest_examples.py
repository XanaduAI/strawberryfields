"""
Unit tests for the :mod:`strawberryfields` example programs.
"""

import os
import re
import unittest
from subprocess import Popen, PIPE, check_call, CalledProcessError

from defaults import TFBackendTest, FockBackendTest, GaussianBaseTest, strawberryfields as sf
from strawberryfields import backends


current_dir = os.path.dirname(os.path.abspath(__file__))
examples_folder = os.path.join(current_dir, os.pardir, 'examples/')
float_regex = r'(-)?(\d+\.\d+)'


class GaussianExamples(GaussianBaseTest):
    num_subsystems = 1

    def setUp(self):
        super().setUp()

    def test_teleportation(self):
        filename = os.path.join(examples_folder,"teleportation.py")
        # tests whether the output fidelity is almost 1
        p = Popen(["python3", filename], stdout=PIPE, 
                       preexec_fn=os.setsid)
        output = float(p.stdout.readlines()[-1])

        self.assertAllAlmostEqual(output, 1, delta=0.06)
            
        p.terminate()

    def test_gate_teleportation(self):
        filename = os.path.join(examples_folder,"gate_teleportation.py")
        # tests whether the initial state and the teleported state
        # are the same in each element up to the tolerance
        p = Popen(["python3", filename], stdout=PIPE, 
                       preexec_fn=os.setsid)

        out = []
        for line in p.stdout.readlines():
            tmp = re.findall(float_regex, str(line))
            out.append([float("".join(i)) for i in tmp])

        results1 = out[0] + out[1]
        results2 = out[2] + out[3]

        for i,j in zip(results1,results2):  
            self.assertAllAlmostEqual(i,j, delta=0.3)
            
        p.terminate()

    def test_gaussian_boson_sampling(self):
        filename = os.path.join(examples_folder,"gaussian_boson_sampling.py")
        soln = [
            0.176378447614135,
            0.0685595637122246,
            0.002056097258977398,
            0.00834294639986785,
            0.01031294525345511
        ]
        # tests whether the output fock probability are as
        # expected, as written in soln
        p = Popen(["python3", filename], stdout=PIPE, 
                       preexec_fn=os.setsid)

        out = []
        for line in p.stdout.readlines():
            tmp = re.findall(float_regex, str(line))
            out.append([float("".join(i)) for i in tmp][0])

        for i,j in zip(out,soln):  
            self.assertAllAlmostEqual(i,j, delta=self.tol)
            
        p.terminate()

    def test_gaussian_cloning(self):
        filename = os.path.join(examples_folder,"gaussian_cloning.py")
        # tests whether the file runs without error
        try:
            check_call(["python3", filename])
        except CalledProcessError:
            assert False, "Gaussian cloning failed"

class FockExamples(FockBackendTest):
    num_subsystems = 1

    def setUp(self):
        super().setUp()

    def test_boson_sampling(self):
        filename = os.path.join(examples_folder,"boson_sampling.py")
        soln = [
            0.174689160486,
            0.106441927246
        ]
        # tests whether the output fock probability are as
        # expected, as written in soln
        p = Popen(["python3", filename], stdout=PIPE, 
                       preexec_fn=os.setsid)

        out = []
        for line in p.stdout.readlines():
            out.append(float(line))

        for i,j in zip(out,soln):  
            self.assertAllAlmostEqual(i,j, delta=self.tol)

        p.terminate()

    def test_IQP(self):
        filename = os.path.join(examples_folder,"IQP.py")

        # tests whether the file runs without error
        try:
            check_call(["python3", filename])
        except CalledProcessError:
            assert False, "IQP failed"

    def test_hamiltonian_simulation(self):
        filename = os.path.join(examples_folder,"hamiltonian_simulation.py")
        soln = [
            0.52240124572,
            0.235652876857,
            0.241945877423
        ]
        # tests whether the output fock probability are as
        # expected, as written in soln
        p = Popen(["python3", filename], stdout=PIPE, 
                       preexec_fn=os.setsid)

        out = []
        for line in p.stdout.readlines():
            out.append(float(line))

        for i,j in zip(out,soln):  
            self.assertAllAlmostEqual(i,j, delta=self.tol)

        p.terminate()

class TFExamples(TFBackendTest):
    num_subsystems = 1

    def setUp(self):
        super().setUp()

    def test_optimization(self):
        filename = os.path.join(examples_folder,"optimization.py")
        soln = b"Value at step 49: 0.36787864565849304\n"
        # tests whether the output fock probability are as
        # expected, as written in soln
        p = Popen(["python3", filename], stdout=PIPE, 
                       preexec_fn=os.setsid)

        out = p.stdout.readlines()[-1]
        self.assertEqual(out, soln)

        p.terminate()


if __name__ == '__main__':
    print('Testing Strawberry Fields version ' + sf.version() + ', examples.')

    # run the tests in this file
    suite = unittest.TestSuite()
    tests = [
        GaussianExamples,
        FockExamples,
        TFExamples
    ]
    for t in tests:
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
