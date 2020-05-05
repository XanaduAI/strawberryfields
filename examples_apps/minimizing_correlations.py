"""
Minimizing the amount of correlations
=====================================

*Author: Nicolas Quesada*

"""


######################################################################
# In `this paper <https://doi.org/10.1103/PhysRevA.88.044301>`__ by Jiang,
# Lang, and Caves [1], the authors show that if one has two qumodes states
# :math:`\left|\psi \right\rangle` and :math:`\left|\phi \right\rangle`
# and a beamsplitter :math:`\text{BS}(\theta)`, then the only way no
# entanglement is generated when the beamsplitter acts on the product of
# the two states
#
# .. math:: \left|\Psi  \right\rangle = \text{BS}(\theta) \ \left|\psi \right\rangle \otimes \left|\phi \right\rangle,
#
# is if the states :math:`\left|\psi \right\rangle` and
# :math:`\left|\phi \right\rangle` are squeezed states along the same
# quadrature and by the same amount.
#
# Now imagine the following task: > Given an input state
# :math:`\left|\psi \right\rangle`, which is not necessarily a squeezed
# state, what is the optimal state :math:`\left|\phi \right\rangle`
# incident on a beamsplitter :math:`\text{BS}(\theta)` together with
# :math:`\left|\psi \right\rangle` such that the resulting entanglement is
# minimized?
#
# In our `paper <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.98.043813>`__ we showed that if
# :math:`\theta \ll 1` the optimal state :math:`\left|\phi \right\rangle`,
# for any input state :math:`\left|\psi \right\rangle`, is always a
# squeezed state. We furthermore conjectured that this holds for any value
# of :math:`\theta`.
#
# Here, we numerically explore this question by performing numerical
# minimization over :math:`\left|\phi \right\rangle` to find the state
# that minimizes the entanglement between the two modes.
#
# First, we import the libraries required for this analysis; NumPy, SciPy,
# TensorFlow, and StrawberryFields.
#

import numpy as np
from scipy.linalg import expm
import tensorflow as tf

import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.backends.tfbackend.ops import partial_trace


# set the random seed
tf.random.set_seed(137)
np.random.seed(42)


######################################################################
# Now, we set the Fock basis truncation; in this case, we choose
# :math:`cutoff=30`:
#

cutoff = 30


######################################################################
# Creating the initial states
# ---------------------------
#
# We define our input state
# :math:`\newcommand{ket}[1]{\left|#1\right\rangle}\ket{\psi}`, an equal
# superposition of :math:`\ket{0}` and :math:`\ket{1}`:
#
# .. math:: \ket{\psi}=\frac{1}{\sqrt{2}}\left(\ket{0}+\ket{1}\right)
#

psi = np.zeros([cutoff], dtype=np.complex128)
psi[0] = 1.0
psi[1] = 1.0
psi /= np.linalg.norm(psi)


######################################################################
# We can now define our initial random guess for the second state
# :math:`\ket{\phi}`:
#

phi = np.random.random(size=[cutoff]) + 1j*np.random.random(size=[cutoff])
phi[10:] = 0.
phi /= np.linalg.norm(phi)


######################################################################
# Next, we define the creation operator :math:`\hat{a}`,
#

a = np.diag(np.sqrt(np.arange(1, cutoff)), k=1)


######################################################################
# the number operator :math:`\hat{n}=\hat{a}^\dagger \hat{a}`,
#

n_opt = a.T @ a


######################################################################
# and the quadrature operators :math:`\hat{x}=a+a^\dagger`,
# :math:`\hat{p}=-i(a-a^\dagger)`.
#

# Define quadrature operators
x = a + a.T
p = -1j*(a-a.T)


######################################################################
# We can now calculate the displacement of the states in the phase space,
# :math:`\alpha=\langle \psi \mid\hat{a}\mid\psi\rangle`. The following
# function calculates this displacement, and then displaces the state by
# :math:`-\alpha` to ensure it has zero displacement.
#

def recenter(state):
    alpha = state.conj() @ a @ state
    disp_alpha = expm(alpha.conj()*a - alpha*a.T)
    out_state = disp_alpha @ state
    return out_state


######################################################################
# First, let’s have a look at the displacement of state :math:`\ket{\psi}`
# and state :math:`\ket{\phi}`:
#

print(psi.conj().T @ a @ psi)
print(phi.conj().T @ a @ phi)


######################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     (0.4999999999999999+0j)
#     (1.4916658938055996+0.20766920902420394j)

######################################################################
# Now, let’s center them in the phase space:
#

psi = recenter(psi)
phi = recenter(phi)


######################################################################
# Checking they now have zero displacement:
#

print(np.round(psi.conj().T @ a @ psi, 8))
print(np.round(phi.conj().T @ a @ phi, 8))

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     0j
#     0j

######################################################################
# Performing the optimization
# ---------------------------
#
# We can construct the variational quantum circuit cost function, using Strawberry
# Fields:
#

penalty_strength = 10

prog = sf.Program(2)
eng = sf.Engine("tf", backend_options={"cutoff_dim": cutoff})

psi = tf.cast(psi, tf.complex64)
phi_var_re = tf.Variable(phi.real)
phi_var_im = tf.Variable(phi.imag)


with prog.context as q:
    Ket(prog.params("input_state")) | q
    BSgate(np.pi/4, 0) | q



######################################################################
# Here, we are initializing a TensorFlow variable ``phi_var`` representing
# the initial state of mode ``q[1]``, which we will optimize over. Note
# that we take the outer product
# :math:`\ket{in}=\ket{\psi}\otimes\ket{\phi}`, and use the ``Ket``
# operator to initialise the circuit in this initial multimode pure state.


def cost(phi_var_re, phi_var_im):
    phi_var = tf.cast(phi_var_re, tf.complex64) + 1j*tf.cast(phi_var_im, tf.complex64)

    in_state = tf.einsum('i,j->ij', psi, phi_var)

    result = eng.run(prog, args={"input_state": in_state}, modes=[1])
    state = result.state

    rhoB = state.dm()
    cost = tf.cast(tf.math.real((tf.linalg.trace(rhoB @ rhoB)
                                -penalty_strength*tf.linalg.trace(rhoB @ x)**2
                                -penalty_strength*tf.linalg.trace(rhoB @ p)**2)
                           /(tf.linalg.trace(rhoB))**2), tf.float64)

    return cost


######################################################################
# The cost function runs the engine, and we use the argument ``modes=[1]`` to
# return the density matrix of mode ``q[1]``.
#
# The returned cost contains the purity of the reduced density matrix
#
# .. math:: \text{Tr}(\rho_B^2),
#
# and an extra penalty that forces the optimized state to have zero
# displacement; that is, we want to minimise the value
#
# .. math:: \langle \hat{x}\rangle=\text{Tr}(\rho_B\hat{x}).
#
# Finally, we divide by the :math:`\text{Tr}(\rho_B)^2` so that the state
# is always normalized.
#
# We can now set up the optimization, to minimise the cost function.
# Running the optimization process for 200 reps:
#

opt = tf.keras.optimizers.Adam(learning_rate=0.007)

reps = 200

cost_progress = []

for i in range(reps):
    # reset the engine if it has already been executed
    if eng.run_progs:
        eng.reset()

    with tf.GradientTape() as tape:
        loss = -cost(phi_var_re, phi_var_im)

    # Stores cost at each step
    cost_progress.append(loss.numpy())

    # one repetition of the optimization
    gradients = tape.gradient(loss, [phi_var_re, phi_var_im])
    opt.apply_gradients(zip(gradients, [phi_var_re, phi_var_im]))

    # Prints progress
    if i % 1 == 0:
        print("Rep: {} Cost: {}".format(i, loss))

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Rep: 0 Cost: -0.4766758978366852
#     Rep: 1 Cost: -0.4223320484161377
#     Rep: 2 Cost: -0.4963147044181824
#     Rep: 3 Cost: -0.45958423614501953
#     Rep: 4 Cost: -0.4818280339241028
#     Rep: 5 Cost: -0.5147174596786499
#     Rep: 6 Cost: -0.5162041783332825
#     Rep: 7 Cost: -0.5110473036766052
#     Rep: 8 Cost: -0.5209271907806396
#     Rep: 9 Cost: -0.5389845371246338
#     Rep: 10 Cost: -0.5502253770828247
#     Rep: 11 Cost: -0.5520217418670654
#     Rep: 12 Cost: -0.5553423762321472
#     Rep: 13 Cost: -0.5651862621307373
#     Rep: 14 Cost: -0.5767862796783447
#     Rep: 15 Cost: -0.5858915448188782
#     Rep: 16 Cost: -0.592574417591095
#     Rep: 17 Cost: -0.5986008644104004
#     Rep: 18 Cost: -0.6053515076637268
#     Rep: 19 Cost: -0.6134768724441528
#     Rep: 20 Cost: -0.6225422620773315
#     Rep: 21 Cost: -0.631293535232544
#     Rep: 22 Cost: -0.6387172937393188
#     Rep: 23 Cost: -0.6450194716453552
#     Rep: 24 Cost: -0.6514304876327515
#     Rep: 25 Cost: -0.6588233113288879
#     Rep: 26 Cost: -0.666886568069458
#     Rep: 27 Cost: -0.6747294068336487
#     Rep: 28 Cost: -0.6818798184394836
#     Rep: 29 Cost: -0.688532292842865
#     Rep: 30 Cost: -0.6950534582138062
#     Rep: 31 Cost: -0.7016120553016663
#     Rep: 32 Cost: -0.7082613110542297
#     Rep: 33 Cost: -0.715013861656189
#     Rep: 34 Cost: -0.7217401266098022
#     Rep: 35 Cost: -0.7281915545463562
#     Rep: 36 Cost: -0.7342450618743896
#     Rep: 37 Cost: -0.7400316596031189
#     Rep: 38 Cost: -0.7457355856895447
#     Rep: 39 Cost: -0.7513759136199951
#     Rep: 40 Cost: -0.7568838596343994
#     Rep: 41 Cost: -0.7622535824775696
#     Rep: 42 Cost: -0.7675091028213501
#     Rep: 43 Cost: -0.7726098895072937
#     Rep: 44 Cost: -0.7774857878684998
#     Rep: 45 Cost: -0.7821434140205383
#     Rep: 46 Cost: -0.7866566181182861
#     Rep: 47 Cost: -0.7910702228546143
#     Rep: 48 Cost: -0.7953729629516602
#     Rep: 49 Cost: -0.7995505332946777
#     Rep: 50 Cost: -0.8036108613014221
#     Rep: 51 Cost: -0.8075516223907471
#     Rep: 52 Cost: -0.8113498091697693
#     Rep: 53 Cost: -0.8149938583374023
#     Rep: 54 Cost: -0.8185041546821594
#     Rep: 55 Cost: -0.82191002368927
#     Rep: 56 Cost: -0.8252224326133728
#     Rep: 57 Cost: -0.8284385204315186
#     Rep: 58 Cost: -0.8315609693527222
#     Rep: 59 Cost: -0.8345984220504761
#     Rep: 60 Cost: -0.8375518321990967
#     Rep: 61 Cost: -0.840415358543396
#     Rep: 62 Cost: -0.8431857228279114
#     Rep: 63 Cost: -0.8458709716796875
#     Rep: 64 Cost: -0.8484804034233093
#     Rep: 65 Cost: -0.8510205149650574
#     Rep: 66 Cost: -0.8534920811653137
#     Rep: 67 Cost: -0.8558962941169739
#     Rep: 68 Cost: -0.8582370281219482
#     Rep: 69 Cost: -0.8605165481567383
#     Rep: 70 Cost: -0.8627358078956604
#     Rep: 71 Cost: -0.8648967146873474
#     Rep: 72 Cost: -0.8670018315315247
#     Rep: 73 Cost: -0.8690568208694458
#     Rep: 74 Cost: -0.8710666298866272
#     Rep: 75 Cost: -0.8730334043502808
#     Rep: 76 Cost: -0.874958336353302
#     Rep: 77 Cost: -0.8768417239189148
#     Rep: 78 Cost: -0.8786842226982117
#     Rep: 79 Cost: -0.8804850578308105
#     Rep: 80 Cost: -0.8822449445724487
#     Rep: 81 Cost: -0.8839642405509949
#     Rep: 82 Cost: -0.8856424689292908
#     Rep: 83 Cost: -0.8872807025909424
#     Rep: 84 Cost: -0.8888780474662781
#     Rep: 85 Cost: -0.8904346823692322
#     Rep: 86 Cost: -0.8919486999511719
#     Rep: 87 Cost: -0.8934181928634644
#     Rep: 88 Cost: -0.8948409557342529
#     Rep: 89 Cost: -0.8962130546569824
#     Rep: 90 Cost: -0.8975301384925842
#     Rep: 91 Cost: -0.8987880349159241
#     Rep: 92 Cost: -0.899979829788208
#     Rep: 93 Cost: -0.9011000990867615
#     Rep: 94 Cost: -0.9021427631378174
#     Rep: 95 Cost: -0.9031027555465698
#     Rep: 96 Cost: -0.9039766192436218
#     Rep: 97 Cost: -0.9047646522521973
#     Rep: 98 Cost: -0.9054698348045349
#     Rep: 99 Cost: -0.9061003923416138
#     Rep: 100 Cost: -0.9066662192344666
#     Rep: 101 Cost: -0.9071784019470215
#     Rep: 102 Cost: -0.9076470732688904
#     Rep: 103 Cost: -0.9080801606178284
#     Rep: 104 Cost: -0.9084805846214294
#     Rep: 105 Cost: -0.9088495373725891
#     Rep: 106 Cost: -0.9091846346855164
#     Rep: 107 Cost: -0.9094842672348022
#     Rep: 108 Cost: -0.9097475409507751
#     Rep: 109 Cost: -0.9099751710891724
#     Rep: 110 Cost: -0.9101698398590088
#     Rep: 111 Cost: -0.9103357195854187
#     Rep: 112 Cost: -0.9104783535003662
#     Rep: 113 Cost: -0.9106017351150513
#     Rep: 114 Cost: -0.9107118844985962
#     Rep: 115 Cost: -0.9108109474182129
#     Rep: 116 Cost: -0.9109019637107849
#     Rep: 117 Cost: -0.9109862446784973
#     Rep: 118 Cost: -0.9110654592514038
#     Rep: 119 Cost: -0.9111390709877014
#     Rep: 120 Cost: -0.91120845079422
#     Rep: 121 Cost: -0.9112733602523804
#     Rep: 122 Cost: -0.9113341569900513
#     Rep: 123 Cost: -0.9113907814025879
#     Rep: 124 Cost: -0.9114443063735962
#     Rep: 125 Cost: -0.9114949107170105
#     Rep: 126 Cost: -0.9115434288978577
#     Rep: 127 Cost: -0.9115898013114929
#     Rep: 128 Cost: -0.9116342067718506
#     Rep: 129 Cost: -0.9116772413253784
#     Rep: 130 Cost: -0.9117189645767212
#     Rep: 131 Cost: -0.9117587804794312
#     Rep: 132 Cost: -0.9117969274520874
#     Rep: 133 Cost: -0.9118331670761108
#     Rep: 134 Cost: -0.9118679165840149
#     Rep: 135 Cost: -0.9118999242782593
#     Rep: 136 Cost: -0.9119301438331604
#     Rep: 137 Cost: -0.9119581580162048
#     Rep: 138 Cost: -0.9119839072227478
#     Rep: 139 Cost: -0.9120074510574341
#     Rep: 140 Cost: -0.9120291471481323
#     Rep: 141 Cost: -0.91204833984375
#     Rep: 142 Cost: -0.9120661020278931
#     Rep: 143 Cost: -0.9120829105377197
#     Rep: 144 Cost: -0.9120978713035583
#     Rep: 145 Cost: -0.9121111631393433
#     Rep: 146 Cost: -0.9121232628822327
#     Rep: 147 Cost: -0.9121342897415161
#     Rep: 148 Cost: -0.9121443033218384
#     Rep: 149 Cost: -0.9121529459953308
#     Rep: 150 Cost: -0.9121609926223755
#     Rep: 151 Cost: -0.9121681451797485
#     Rep: 152 Cost: -0.9121744632720947
#     Rep: 153 Cost: -0.9121802449226379
#     Rep: 154 Cost: -0.9121854305267334
#     Rep: 155 Cost: -0.9121901988983154
#     Rep: 156 Cost: -0.9121944904327393
#     Rep: 157 Cost: -0.912198543548584
#     Rep: 158 Cost: -0.912202000617981
#     Rep: 159 Cost: -0.912205159664154
#     Rep: 160 Cost: -0.9122082591056824
#     Rep: 161 Cost: -0.9122109413146973
#     Rep: 162 Cost: -0.9122134447097778
#     Rep: 163 Cost: -0.9122157692909241
#     Rep: 164 Cost: -0.9122180342674255
#     Rep: 165 Cost: -0.912219762802124
#     Rep: 166 Cost: -0.912221372127533
#     Rep: 167 Cost: -0.9122230410575867
#     Rep: 168 Cost: -0.912224292755127
#     Rep: 169 Cost: -0.9122257828712463
#     Rep: 170 Cost: -0.9122268557548523
#     Rep: 171 Cost: -0.9122276306152344
#     Rep: 172 Cost: -0.9122287034988403
#     Rep: 173 Cost: -0.9122294187545776
#     Rep: 174 Cost: -0.9122301340103149
#     Rep: 175 Cost: -0.9122307896614075
#     Rep: 176 Cost: -0.9122319221496582
#     Rep: 177 Cost: -0.9122322201728821
#     Rep: 178 Cost: -0.9122328758239746
#     Rep: 179 Cost: -0.9122334718704224
#     Rep: 180 Cost: -0.9122337698936462
#     Rep: 181 Cost: -0.9122341275215149
#     Rep: 182 Cost: -0.9122346043586731
#     Rep: 183 Cost: -0.9122347235679626
#     Rep: 184 Cost: -0.9122350215911865
#     Rep: 185 Cost: -0.9122350811958313
#     Rep: 186 Cost: -0.9122355580329895
#     Rep: 187 Cost: -0.9122354388237
#     Rep: 188 Cost: -0.9122354984283447
#     Rep: 189 Cost: -0.9122358560562134
#     Rep: 190 Cost: -0.9122357964515686
#     Rep: 191 Cost: -0.9122359156608582
#     Rep: 192 Cost: -0.9122360944747925
#     Rep: 193 Cost: -0.9122361540794373
#     Rep: 194 Cost: -0.912236213684082
#     Rep: 195 Cost: -0.9122362732887268
#     Rep: 196 Cost: -0.912236213684082
#     Rep: 197 Cost: -0.9122360348701477
#     Rep: 198 Cost: -0.9122362732887268
#     Rep: 199 Cost: -0.912236213684082


######################################################################
# We can see that the optimization converges to the optimum purity value
# of 0.9122365.
#
# Visualising the optimum state
# -----------------------------
#
# We can now calculate the density matrix of the input state
# :math:`\ket{\phi}` which minimises entanglement:
#
# .. math:: \rho_{\phi} = \ket{\phi}\left\langle \phi\right|
#

ket_val = phi_var_re.numpy() + phi_var_im.numpy()*1j
out_rhoB = np.outer(ket_val, ket_val.conj())
out_rhoB /= np.trace(out_rhoB)


######################################################################
# Note that the optimization is not guaranteed to keep the state normalized,
# so we renormalize the minimized state.
#
# Next, we can use the following function to plot the Wigner function of
# this density matrix:
#

import copy
def wigner(rho, xvec, pvec):
    # Modified from qutip.org
    Q, P = np.meshgrid(xvec, pvec)
    A = (Q + P * 1.0j) / (2 * np.sqrt(2 / 2))

    Wlist = np.array([np.zeros(np.shape(A), dtype=complex) for k in range(cutoff)])

    # Wigner function for |0><0|
    Wlist[0] = np.exp(-2.0 * np.abs(A) ** 2) / np.pi

    # W = rho(0,0)W(|0><0|)
    W = np.real(rho[0, 0]) * np.real(Wlist[0])

    for n in range(1, cutoff):
        Wlist[n] = (2.0 * A * Wlist[n - 1]) / np.sqrt(n)
        W += 2 * np.real(rho[0, n] * Wlist[n])

    for m in range(1, cutoff):
        temp = copy.copy(Wlist[m])
        # Wlist[m] = Wigner function for |m><m|
        Wlist[m] = (2 * np.conj(A) * temp - np.sqrt(m)
                    * Wlist[m - 1]) / np.sqrt(m)

        # W += rho(m,m)W(|m><m|)
        W += np.real(rho[m, m] * Wlist[m])

        for n in range(m + 1, cutoff):
            temp2 = (2 * A * Wlist[n - 1] - np.sqrt(m) * temp) / np.sqrt(n)
            temp = copy.copy(Wlist[n])
            # Wlist[n] = Wigner function for |m><n|
            Wlist[n] = temp2

            # W += rho(m,n)W(|m><n|) + rho(n,m)W(|n><m|)
            W += 2 * np.real(rho[m, n] * Wlist[n])

    return Q, P, W / 2

# Import plotting
from matplotlib import pyplot as plt
# %matplotlib inline

x = np.arange(-5, 5, 0.1)
p = np.arange(-5, 5, 0.1)
X, P, W = wigner(out_rhoB, x, p)
plt.contourf(X, P, np.round(W,3), cmap="PiYG")
plt.show()

######################################################################
# .. image:: /_static/images/sphx_glr_run_minimizing_correlations_001.png
#     :class: sphx-glr-single-img

######################################################################
# We see that the optimal state is indeed a (mildly) squeezed state. This
# can be confirmed by visuallising the Fock state probabilities of state
# :math:`\ket{\phi}`:
#

plt.bar(np.arange(cutoff), height=np.diag(out_rhoB).real)
plt.show()


######################################################################
# .. image:: /_static/images/sphx_glr_run_minimizing_correlations_002.png
#     :class: sphx-glr-single-img


######################################################################
# Finally, printing out the mean number of photons
# :math:`\bar{n} = \left\langle \phi \mid \hat{n} \mid \phi\right\rangle = \text{Tr}(\hat{n}\rho)`,
# as well as the squeezing magnitude
# :math:`r=\sinh^{-1}\left(\sqrt{\bar{n}}\right)` of this state:
#

nbar = np.trace(n_opt @ out_rhoB).real
print("mean number of photons =", nbar)
print("squeezing parameter =", np.arcsinh(np.sqrt(nbar)))

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     mean number of photons = 0.08352929059995343
#     squeezing parameter = 0.28513493641217097

######################################################################
# References
# ----------
#
# [1] Jiang, Z., Lang, M. D., & Caves, C. M. (2013). Mixing nonclassical
# pure states in a linear-optical network almost always generates modal
# entanglement. *Physical Review A*, 88(4), 044301.
#
# [2] Quesada, N., & Brańczyk, A. M. (2018). Gaussian functions are
# optimal for waveguided nonlinear-quantum-optical processes.
# *Physical Review A*, 98, 043813.
#
