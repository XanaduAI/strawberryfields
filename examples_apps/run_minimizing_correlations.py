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
# In our `paper <https://arxiv.org/abs/1805.06868>`__ we showed that if
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
# Now, let’s center them in the phase space:
#

psi = recenter(psi)
phi = recenter(phi)


######################################################################
# Checking they now have zero displacement:
#

print(np.round(psi.conj().T @ a @ psi, 9))
print(np.round(phi.conj().T @ a @ phi, 9))


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


######################################################################
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
# We see that the optimal state is indeed a (mildly) squeezed state. This
# can be confirmed by visuallising the Fock state probabilities of state
# :math:`\ket{\phi}`:
#

plt.bar(np.arange(cutoff), height=np.abs(ket_val)**2)
plt.show()


######################################################################
# Finally, printing out the mean number of photons
# :math:`\bar{n} = \left\langle \phi \mid \hat{n} \mid \phi\right\rangle`,
# as well as the squeezing magnitude
# :math:`r=\sinh^{-1}\left(\sqrt{\bar{n}}\right)` of this state:
#

nbar = ((ket_val.conj()).T @ n_opt @ ket_val).real
print("mean number of photons =", nbar)
print("squeezing parameter =", np.arcsinh(np.sqrt(nbar)))


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
# *arXiv:1805.06868*.
#