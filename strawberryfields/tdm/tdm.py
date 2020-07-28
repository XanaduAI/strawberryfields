import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.parameters import par_is_symbolic
import numpy as np
from matplotlib import pyplot as plt
import time
from pathlib import Path

class TDMProgram(sf.Program):
    def __init__(self, N, name=None):
        self.concurr_modes = N
        self.total_timebins = 0
        self.measurement_order = None
        super().__init__(num_subsystems=N, name=name)

    def context(self, *args, copies=1):
        self.input_check(args, copies)
        self.copies = copies
        self.tdm_params = args
        self.loop_vars = self.params(*[f"p{i}" for i in range(len(args))])
        return self

    def __enter__(self):
        super().__enter__()
        return self.loop_vars, self.register

    def __exit__(self, ex_type, ex_value, ex_tb):
        super().__exit__(ex_type, ex_value, ex_tb)
        
        cmds = self.circuit.copy()
        self.circuit = []
        
        timebins = len(self.tdm_params[0])
        self.total_timebins = timebins * self.copies
        
        self.measurement_order = []
        
        for i in range(self.total_timebins):
            for cmd in cmds:
                
                params = cmd.op.p.copy()
                
                if par_is_symbolic(params[0]):
                    arg_index = int(params[0].name[1:])
                    params[0] = self.tdm_params[arg_index][i % timebins]
                    
                reg = [(r.ind + i) % self.concurr_modes for r in cmd.reg]
                self.append(cmd.op.__class__(*params), reg)
                
                if isinstance(cmd.op, ops.Measurement):
                    self.measurement_order.extend(reg)
                    
        

    def __str__(self):
        return f"<TDMProgram: concurrent modes={self.concurr_modes}, time bins={self.total_timebins}>"

    def input_check(self, args, copies=1):
        
        # check if all lists of equal length
        param_lengths = [len(param) for param in args]
        if param_lengths.count(param_lengths[0]) is not len(param_lengths):
            raise ValueError(
                "Gate-parameter lists must be of equal length."
            )
            
        # check if lists contain nothing but real numbers
        for params in args:
            for x in params:
                if isinstance(x, (int, float)) is False or isinstance(x, bool):
                    raise TypeError(
                    "Gate parameters must be real numbers."
                    )
               
        # check copies
        if isinstance(copies, int) is False or isinstance(copies, bool) or copies < 1:
            raise TypeError(
               "Number of copies must be a real positive integer."
            )




# def create_program(BS, R, M, copies):
#     '''
#     Runs a sequence of Gaussian gates with compiled lists of phase settings. Returns a SF Program object.
#     '''
#     r = 2
#     eta = 0.9
#     xi = 0.05

#     nbar = xi/2

#     BS, R, M, copies = check_input(BS, R, M, copies)
#     alpha, phi, theta = compile(BS, R, M, copies)

#     prog = sf.Program(3)
#     with prog.context as q:

#         for a, p, t in zip(alpha, phi, theta):
#             ops.Sgate(r, 0) | q[2]
#             ops.ThermalLossChannel(eta, nbar) | q[2]
#             ops.BSgate(a) | (q[2], q[1])
#             ops.Rgate(p) | q[2]
#             ops.MeasureHomodyne(t) | q[0]
#             q = q[1:] + q[:1]
#     return prog


# def check_input(BS, R, M, copies):
#     '''
#     Check user input here:
#     - all lists of equal length?
#     - all T in [0,1]?
#     - phase settings in allowed set?
#     - etc.
#     '''     
#     return BS, R, M, copies


# def compile(BS, R, M, copies):
#     '''
#     - Converts excess noise xi to mean photon number nbar. xi is the excess quadrature variance on top of quantum noise: V = V_qu + xi. nbar is the mean photon number of a thermal state mixed with a quantum mode yielding a quadrature variance of V = V_qu + 2*nbar.
#     - Converts list with beamsplitter settings (intensity transmission coefficients), phase gate and measurement gate (phase angles in multiples of pi) to radians.
#     - Extends gate sequences to repetitive sequences according to the number of copies.
#     '''

#     # convert from intensity transmission T to phase angle
#     alpha_ = np.arccos(np.sqrt(np.array(BS)))

#     # convert from multiples of pi to angle
#     phi_ = np.pi*np.array(R)

#     # convert from multiples of pi to angle
#     theta_ = np.pi*np.array(M)

#     alpha = np.tile(alpha_, copies)
#     phi = np.tile(phi_, copies)
#     theta = np.tile(theta_, copies)

#     return alpha, phi, theta



# def histogram(X_vals):
#     '''
#     Plots a histogram of quadrature samples X_vals. X_vals can either be an object parameter or an external list passed to this function.
#     '''

#     fig, ax = plt.subplots()
#     fig.suptitle('Sample histogram', fontsize=18)
    
#     ax.set_xlabel('$X$', fontsize=18)
#     ax.set_ylabel('occurrence', fontsize=18)
#     ax.grid(True)
#     plt.hist(self.X_vals,50)

# def nearest_neighbour_corr(X_vals):
#     '''
#     Plots a 2x2 grid of nearest neighbour correlations between subsequent quadrature samples XX, XP, PX, PP. X_vals can either be an object parameter or an external list passed to this function.
#     '''
    
#     X0 = X_vals[0::8]
#     X1 = X_vals[1::8]

#     X2 = X_vals[2::8]
#     P0 = X_vals[3::8]

#     P1 = X_vals[4::8]
#     X3 = X_vals[5::8]

#     P2 = X_vals[6::8]
#     P3 = X_vals[7::8]   
               
#     fig, axs = plt.subplots(2, 2, figsize=(10, 10))
#     fig.suptitle('Nearest-neighbour correlations', fontsize=18)

#     axs[0,0].scatter(X1, X0)
#     axs[0,1].scatter(P0, X2)
#     axs[1,0].scatter(X3, P1)
#     axs[1,1].scatter(P3, P2)

#     axs[0,0].set_ylabel('$X_{i-1}$', fontsize=18)
#     axs[1,0].set_xlabel('$X_{i}$', fontsize=18)
#     axs[1,0].set_ylabel('$P_{i-1}$', fontsize=18)
#     axs[1,1].set_xlabel('$P_{i}$', fontsize=18)

#     for ax in axs.flat:
#         ax.grid(True)


# class TDMstate:
#     def create_program(self, copies):
#         prog = create_program(self.BS, self.R, self.M, copies)
#         return prog


# class EPRstate(TDMstate):
#     '''
#     Pre-compiled settings for two-mode-squeezed states (EPR states)
#     '''
#     # dictionary for four EPR states to accomodate all X-P-measurement permutations
#     BS = [1,1/2]*4 # BS intensity transmission coefficients
#     R = [1/2,0]*4 # phase angles in multiples of pi
#     M = [0, 0]+[0, 1/2]+[1/2, 0]+[1/2, 1/2] # measurement angles in multiples of pi
            
#     ## compensating for list multiplication
#     # copies = int(np.ceil(self.copies/4))

#     def __init__(self):
#         super().__init__()

#     def insep_param(self, X_vals):
#         '''
#         Computes the inseparability parameter of EPR states. It is obtained by

#         < (X_(i)-X_(i+1))**2 > + < (P_(i)+P_(i+1))**2 >.

#         Reference: https://advances.sciencemag.org/content/advances/5/5/eaaw4530.full.pdf

#         X_vals can either be an object parameter or an external list passed to this function.
#         '''
        
#         X0 = X_vals[0::8]
#         X1 = X_vals[1::8]
        
#         P2 = X_vals[6::8]
#         P3 = X_vals[7::8] 
        
#         k = 1
#         DeltaX = np.array(X0) - np.array(X1)*(-1)**k
#         DeltaP = np.array(P2) - np.array(P3)*(-1)**(k+1)

#         insep_param = 1/len(X0)*np.linalg.norm(DeltaX)**2 + 1/len(P2)*np.linalg.norm(DeltaP)**2
#         print('Estimated inseparability parameter: '+str(insep_param))



# class GHZstate(TDMstate):
#     '''
#     Pre-compiled settings for 3-mode GHZ states
#     '''
#     BS = [1, 1/3, 1/2] # BS intensity transmission coefficients
#     R = [1/2, 1, 0] # phase angles in multiples of pi
#     M = [0, 0, 0] # measurement angles in multiples of pi
    
#     def __init__(self):
#         super().__init__()

#     def insep_param(self, X_vals):
#         '''
#         Computes the inseparability parameter of three-mode GHZ states. X_vals can either be an object parameter or an external list passed to this function.

#         +++ COMING SOON +++

#         '''


# class Clusterstate(TDMstate):

#     BS = [1/2] # BS intensity transmission coefficients
#     R = [1/2] # phase angles in multiples of pi
#     M = [0] # measurement angles in multiples of pi

#     def __init__(self, modes):
#         super().__init__()

#         self.BS = self.BS*modes
#         self.R = self.R*modes
#         self.M = self.M*modes

#     def insep_param(X_vals):
#         '''
#         Computes the inseparability parameter of one-dimensional cluster states. X_vals can either be an object parameter or an external list passed to this function.

#         +++ COMING SOON +++

#         '''
#         pass


# class Randomnumbers(TDMstate):
#     '''
#     Pre-compiled settings for a Gaussian-distributed quantum random number
#     '''
#     BS = [0] # BS intensity transmission coefficients
#     R = [0] # phase angles in multiples of pi
#     M = [0] # measurement angles in multiples of pi
    
#     def __init__(self):
#         super().__init__()


# class Randomstate(TDMstate):
#     def __init__(self, modes):
#         super().__init__()
#         '''
#         Pre-compiled settings for n-mode random states
#         '''
#         self.BS = np.random.rand(modes) # BS intensity transmission coefficients
#         self.R = np.random.rand(modes) # phase angles in multiples of pi
#         self.M = np.random.rand(modes) # measurement angles in multiples of pi


    




    # def assignsetup(self, r, eta, xi):
    #     '''
    #     Assigns squeezing parameter, total transmission and excess noise xi. If a TDM state class is initialzied without these parameters, the default values defines on TDMstate class level are used.
    #     '''

    #     if r is not None:
    #         self.r = r
    #     if eta is not None:
    #         self.eta = eta
    #     if xi is not None:
    #         self.xi = xi


    # # def printgates(self):
    # #     self.prog.print()
    
    # def run_local(self):
    #     '''
    #     Run local SF engine. The result list X_vals will be created as object parameter to perform statistical evaluations and plots on.
    #     '''

    #     eng = sf.Engine('gaussian')
    #     result = eng.run(self.tdm_program) # local simulation

    #     dd = eng.all_samples
    #     self.X_vals = []
    #     for i in range(len(dd[len(dd)-1])):
    #         for j in range(len(dd)):
    #             self.X_vals.append(float(dd[j][i]))

    # def run_remote(self):
    #     '''
    #     Run remote SF engine. The result list X_vals will be created as object parameter to perform statistical evaluations and plots on.
    #     '''

    #     eng = sf.Engine('tdm')
    #     result = eng.run(self.tdm_program) # local simulation

    # def saveresults(self, path=None):
    #     '''
    #     Saves list of quadrature samples X_vals to file in user-defined or default path.
    #     '''

    #     if path is None:
    #         path = Path.home() / 'tdm_data' / 'results'
    #     Path(path).mkdir(parents=True, exist_ok=True)

    #     f = open(path / 'results.txt', 'w+')
    #     f.write('QUADRATURE SAMPLES\n\n')
    #     for x in self.X_vals:
    #         f.write(str(x)+'\n')
    #     f.close()

    # def savesettings(self, path=None):
    #     '''
    #     Saves compiled lists of gate settings to file in user-defined or default path. All gate settings are given in radians.
    #     '''

    #     if path is None:
    #         path = Path.home() / 'tdm_data' / 'waveforms'
    #     Path(path).mkdir(parents=True, exist_ok=True)

    #     start = time.time()

    #     f = open(path / 'beamsplitter.txt', 'w+')
    #     f.write('BEAMSPLITTER SETTINGS\n\n')
    #     for x in self.alpha_:
    #         f.write(str(x)+'\n')
    #     f.close()
               
    #     f = open(path / 'phase_gate.txt', 'w+')
    #     f.write('PHASE-GATE SETTINGS\n\n')
    #     for x in self.phi_:
    #         f.write(str(x)+'\n')
    #     f.close()
        
    #     f = open(path / 'homodyne_measurement.txt', 'w+')
    #     f.write('MEASUREMENT SETTINGS\n\n')
    #     for x in self.theta_:
    #         f.write(str(x)+'\n')  
    #     f.close()
                
    #     duration = time.time() - start
    #     print('Took me '+str(duration)+' sec.')