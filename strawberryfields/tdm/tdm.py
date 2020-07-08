import strawberryfields as sf
from strawberryfields import ops
import numpy as np
from matplotlib import pyplot as plt
import time
from pathlib import Path

class TDMstate:
    # def __init__(self):
    #   pass

    r = 2.3
    eta = 1
    xi = 0

    def assignsetup(self, r, eta, xi):
        '''
        Assigns squeezing parameter, total transmission and excess noise xi. If a TDM state class is initialzied without these parameters, the default values defines on TDMstate class level are used.
        '''

        if r is not None:
            self.r = r
        if eta is not None:
            self.eta = eta
        if xi is not None:
            self.xi = xi

    @property
    def tdm_program(self):
        '''
        Runs a sequence of Gaussian gates with compiled lists of phase settings. Returns a SF Program object.
        '''

        prog = sf.Program(3)
        with prog.context as q:

            for a, p, t in zip(self.alpha_, self.phi_, self.theta_):
                ops.Sgate(self.r, 0) | q[2]
                ops.ThermalLossChannel(self.eta, self.nbar) | q[2]
                ops.BSgate(a) | (q[2], q[1])
                ops.Rgate(p) | q[2]
                ops.MeasureHomodyne(t) | q[0]
                q = q[1:] + q[:1]

        return prog

    def compile(self):
        '''
        - Converts excess noise xi to mean photon number nbar. xi is the excess quadrature variance on top of quantum noise: V = V_qu + xi. nbar is the mean photon number of a thermal state mixed with a quantum mode yielding a quadrature variance of V = V_qu + 2*nbar.
        - Converts list with beamsplitter settings (intensity transmission coefficients), phase gate and measurement gate (phase angles in multiples of pi) to radians.
        - Extends gate sequences to repetitive sequences according to the number of copies.
        '''

        self.nbar = self.xi/2

        # convert from intensity transmission T to phase angle
        alpha = np.arccos(np.sqrt(np.array(self.BS)))

        # convert from multiples of pi to angle
        phi = np.pi*np.array(self.R)

        # convert from multiples of pi to angle
        theta = np.pi*np.array(self.M)

        self.alpha_ = np.tile(alpha, self.copies)
        self.phi_ = np.tile(phi, self.copies)
        self.theta_ = np.tile(theta, self.copies)
    # def printgates(self):
    #     self.prog.print()
    
    def run_local(self):
        '''
        Run local SF engine. The result list X_vals will be created as object parameter to perform statistical evaluations and plots on.
        '''

        eng = sf.Engine('gaussian')
        result = eng.run(self.tdm_program) # local simulation

        dd = eng.all_samples
        self.X_vals = []
        for i in range(len(dd[len(dd)-1])):
            for j in range(len(dd)):
                self.X_vals.append(float(dd[j][i]))

    def run_remote(self):
        '''
        Run remote SF engine. The result list X_vals will be created as object parameter to perform statistical evaluations and plots on.
        '''

        eng = sf.Engine('tdm')
        result = eng.run(self.tdm_program) # local simulation

    def saveresults(self, path=None):
        '''
        Saves list of quadrature samples X_vals to file in user-defined or default path.
        '''

        if path is None:
            path = Path.home() / 'tdm_data' / 'results'
        Path(path).mkdir(parents=True, exist_ok=True)

        f = open(path / 'results.txt', 'w+')
        f.write('QUADRATURE SAMPLES\n\n')
        for x in self.X_vals:
            f.write(str(x)+'\n')
        f.close()

    def savesettings(self, path=None):
        '''
        Saves compiled lists of gate settings to file in user-defined or default path. All gate settings are given in radians.
        '''

        if path is None:
            path = Path.home() / 'tdm_data' / 'waveforms'
        Path(path).mkdir(parents=True, exist_ok=True)

        start = time.time()

        f = open(path / 'beamsplitter.txt', 'w+')
        f.write('BEAMSPLITTER SETTINGS\n\n')
        for x in self.alpha_:
            f.write(str(x)+'\n')
        f.close()
               
        f = open(path / 'phase_gate.txt', 'w+')
        f.write('PHASE-GATE SETTINGS\n\n')
        for x in self.phi_:
            f.write(str(x)+'\n')
        f.close()
        
        f = open(path / 'homodyne_measurement.txt', 'w+')
        f.write('MEASUREMENT SETTINGS\n\n')
        for x in self.theta_:
            f.write(str(x)+'\n')  
        f.close()
                
        duration = time.time() - start
        print('Took me '+str(duration)+' sec.')

    def histogram(self, X_vals=None):
        '''
        Plots a histogram of quadrature samples X_vals. X_vals can either be an object parameter or an external list passed to this function.
        '''
        
        if X_vals is None:
            X_vals = self.X_vals

        fig, ax = plt.subplots()
        fig.suptitle('Sample histogram', fontsize=18)
        
        ax.set_xlabel('$X$', fontsize=18)
        ax.set_ylabel('occurrence', fontsize=18)
        ax.grid(True)
        plt.hist(self.X_vals,50)

    def nearest_neighbour_corr(self, X_vals=None):
        '''
        Plots a 2x2 grid of nearest neighbour correlations between subsequent quadrature samples XX, XP, PX, PP. X_vals can either be an object parameter or an external list passed to this function.
        '''

        if X_vals is None:
            X_vals = self.X_vals
        
        X0 = X_vals[0::8]
        X1 = X_vals[1::8]

        X2 = X_vals[2::8]
        P0 = X_vals[3::8]

        P1 = X_vals[4::8]
        X3 = X_vals[5::8]

        P2 = X_vals[6::8]
        P3 = X_vals[7::8]   
                   
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Nearest-neighbour correlations', fontsize=18)

        axs[0,0].scatter(X1, X0)
        axs[0,1].scatter(P0, X2)
        axs[1,0].scatter(X3, P1)
        axs[1,1].scatter(P3, P2)

        axs[0,0].set_ylabel('$X_{i-1}$', fontsize=18)
        axs[1,0].set_xlabel('$X_{i}$', fontsize=18)
        axs[1,0].set_ylabel('$P_{i-1}$', fontsize=18)
        axs[1,1].set_xlabel('$P_{i}$', fontsize=18)

        for ax in axs.flat:
            ax.grid(True)


class EPRstate(TDMstate):
    def __init__(self, copies=1, r=None, eta=None, xi=None):
        super().__init__()

        self.copies = copies

        self.gatesettings()
        self.assignsetup(r, eta, xi)

    def gatesettings(self):
        '''
        Pre-compiled settings for two-mode-squeezed states (EPR states)
        '''

        # dictionary for four EPR states to accomodate all X-P-measurement permutations
        self.BS = [1,1/2]*4 # BS intensity transmission coefficients
        self.R = [1/2,0]*4 # phase angles in multiples of pi
        self.M = [0, 0]+[0, 1/2]+[1/2, 0]+[1/2, 1/2] # measurement angles in multiples of pi
                
        # compensating for list multiplication
        self.copies = int(np.ceil(self.copies/4))

        self.compile()

    def insep_param(self, X_vals=None):
        '''
        Computes the inseparability parameter of EPR states. It is obtained by

        < (X_(i)-X_(i+1))**2 > + < (P_(i)+P_(i+1))**2 >.

        Reference: https://advances.sciencemag.org/content/advances/5/5/eaaw4530.full.pdf

        X_vals can either be an object parameter or an external list passed to this function.
        '''
        
        if X_vals is None:
            X_vals = self.X_vals
        
        X0 = X_vals[0::8]
        X1 = X_vals[1::8]
        
        P2 = X_vals[6::8]
        P3 = X_vals[7::8] 
        
        k = 1
        DeltaX = np.array(X0) - np.array(X1)*(-1)**k
        DeltaP = np.array(P2) - np.array(P3)*(-1)**(k+1)

        insep_param = 1/len(X0)*np.linalg.norm(DeltaX)**2 + 1/len(P2)*np.linalg.norm(DeltaP)**2
        print('Estimated inseparability parameter: '+str(insep_param))


class GHZstate(TDMstate):
    def __init__(self, copies, r=None, eta=None, xi=None):
        super().__init__()

        self.copies = copies

        self.gatesettings()
        self.assignsetup(r, eta, xi)
        self.compile()

    def gatesettings(self):
        '''
        Pre-compiled settings for 3-mode GHZ states
        '''
        self.BS = [1, 1/3, 1/2] # BS intensity transmission coefficients
        self.R = [1/2, 1, 0] # phase angles in multiples of pi
        self.M = [0, 0, 0] # measurement angles in multiples of pi

    def insep_param(self, X_vals):
        '''
        Computes the inseparability parameter of three-mode GHZ states. X_vals can either be an object parameter or an external list passed to this function.

        +++ COMING SOON +++

        '''
        if X_vals is None:
            X_vals = self.X_vals
        pass


class Clusterstate(TDMstate):
    def __init__(self, modes, copies, r=None, eta=None, xi=None):
        super().__init__()

        self.modes = modes
        self.copies = copies

        self.gatesettings()
        self.assignsetup(r, eta, xi)
        self.compile()

    def gatesettings(self):
        '''
        Pre-compiled settings for 1D cluster state
        '''
        self.BS = [1/2]*self.modes # BS intensity transmission coefficients
        self.R = [1/2]*self.modes # phase angles in multiples of pi
        self.M = [0]*self.modes # measurement angles in multiples of pi

    def insep_param(self, X_vals):
        '''
        Computes the inseparability parameter of one-dimensional cluster states. X_vals can either be an object parameter or an external list passed to this function.

        +++ COMING SOON +++

        '''

        if X_vals is None:
            X_vals = self.X_vals
        pass


class Randomnumbers(TDMstate):
    def __init__(self, copies=1, r=None, eta=None, xi=None):
        super().__init__()

        self.copies = copies

        self.gatesettings()
        self.assignsetup(r, eta, xi)
        self.compile()

    def gatesettings(self):
        '''
        Pre-compiled settings for a Gaussian-distributed quantum random number
        '''
        self.BS = [0] # BS intensity transmission coefficients
        self.R = [0] # phase angles in multiples of pi
        self.M = [0] # measurement angles in multiples of pi
    

class Randomstate(TDMstate):
    def __init__(self, modes, copies=1, r=None, eta=None, xi=None):
        super().__init__()

        self.modes = modes
        self.copies = copies

        self.gatesettings()
        self.assignsetup(r, eta, xi)

    def gatesettings(self):        
        '''
        Pre-compiled settings for n-mode random states
        '''
        self.BS = np.random.rand(self.modes) # BS intensity transmission coefficients
        self.R = np.random.rand(self.modes) # phase angles in multiples of pi
        self.M = np.random.rand(self.modes) # measurement angles in multiples of pi
    

class Customstate(TDMstate):
    def __init__(self, BS, R, M, copies = 1, r=None, eta=None, xi=None):
        super().__init__()

        self.BS = BS
        self.R = R
        self.M = M
        self.copies = copies

        self.check_input()
        self.assignsetup(r, eta, xi)
        self.compile()

    def check_input(self):
        '''
        Check user input here:
        - all lists of equal length?
        - all T in [0,1]?
        - phase settings in allowed set?
        - etc.
        '''     
        pass