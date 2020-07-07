import strawberryfields as sf
from strawberryfields import ops
import numpy as np
from matplotlib import pyplot as plt
import time

class TDMstate:
    # def __init__(self):
    #   pass

    r = 2.3
    eta = 1
    xi = 0

    def assignsetup(self, r, eta, xi):

        if r is not None:
            self.r = r
        if eta is not None:
            self.eta = eta
        if xi is not None:
            self.xi = xi

    @property
    def tdm_program(self):
        """docstring"""

        nbar = self.xi/2

        # convert from intensity transmission T to phase angle
        alpha = np.arccos(np.sqrt(np.array(self.BS)))

        # convert from multiples of pi to angle
        phi = np.pi*np.array(self.R)

        # convert from multiples of pi to angle
        theta = np.pi*np.array(self.M)

        alpha_ = np.tile(alpha, self.copies)
        phi_ = np.tile(phi, self.copies)
        theta_ = np.tile(theta, self.copies)

        prog = sf.Program(3)
        with prog.context as q:

            for a, p, t in zip(alpha_, phi_, theta_):
                ops.Sgate(self.r, 0) | q[2]
                ops.ThermalLossChannel(self.eta, nbar) | q[2]
                ops.BSgate(a) | (q[2], q[1])
                ops.Rgate(p) | q[2]
                ops.MeasureHomodyne(t) | q[0]
                q = q[1:] + q[:1]

        return prog

    # def printgates(self):
    #     self.prog.print()
    
    def run_local(self):

        eng = sf.Engine('gaussian')
        result = eng.run(self.tdm_program) # local simulation

        dd = eng.all_samples
        self.X_vals = []
        for i in range(len(dd[len(dd)-1])):
            for j in range(len(dd)):
                self.X_vals.append(float(dd[j][i]))

    def run_remote(self):
        eng = sf.Engine('tdm')
        result = eng.run(self.tdm_program) # local simulation

    def saveresults(self):
        f = open('results/results.txt', 'w+')
        f.write('QUADRATURE SAMPLES\n\n')
        for x in self.X_vals:
            f.write(str(x)+'\n')
        f.close()

    def savesettings(self):
        
        start = time.time()

        f = open('waveforms/beamsplitter.txt', 'w+')
        f.write('BEAMSPLITTER SETTINGS\n\n')
        for x in self.BS_:
            f.write(str(x)+'\n')
        f.close()
               
        f = open('waveforms/phase_gate.txt', 'w+')
        f.write('PHASE-GATE SETTINGS\n\n')
        for x in self.R_:
            f.write(str(x)+'\n')
        f.close()
        
        f = open('waveforms/homodyne_measurement.txt', 'w+')
        f.write('MEASUREMENT SETTINGS\n\n')
        for x in self.M_:
            f.write(str(x)+'\n')  
        f.close()
                
        duration = time.time() - start
        print('Took me '+str(duration)+' sec.')

    def histogram(self):
        
        fig, ax = plt.subplots()
        fig.suptitle('Sample histogram', fontsize=18)
        
        ax.set_xlabel('$X$', fontsize=18)
        ax.set_ylabel('occurrence', fontsize=18)
        ax.grid(True)
        plt.hist(self.X_vals,50)


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

    def nearest_neighbour_corr(self, X_vals=None):

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

    def insep_param(self, X_vals=None):

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

    def gatesettings(self):
        '''
        Pre-compiled settings for 3-mode GHZ states
        '''
        self.BS = [1, 1/3, 1/2] # BS intensity transmission coefficients
        self.R = [1/2, 1, 0] # phase angles in multiples of pi
        self.M = [0, 0, 0] # measurement angles in multiples of pi

    def insep_param(self, X_vals):

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

    def gatesettings(self):
        '''
        Pre-compiled settings for 1D cluster state
        '''
        self.BS = [1/2]*self.modes # BS intensity transmission coefficients
        self.R = [1/2]*self.modes # phase angles in multiples of pi
        self.M = [0]*self.modes # measurement angles in multiples of pi
    
    def insep_param(self, X_vals):

        if X_vals is None:
            X_vals = self.X_vals
        pass


class Randomnumbers(TDMstate):
    def __init__(self, copies=1, r=None, eta=None, xi=None):
        super().__init__()

        self.copies = copies

        self.gatesettings()
        self.assignsetup(r, eta, xi)

    def gatesettings(self):
        '''
        Pre-compiled settings for 1D cluster state
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

        self.compiler()
        self.assignsetup(r, eta, xi)

    def compiler(self):
        '''
        Check user input here:
        - all lists of equal length?
        - all T in [0,1]?
        - phase settings in allowed set?
        - etc.
        '''     
        pass