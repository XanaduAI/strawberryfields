# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 15:18:59 2021

@author: Xanadu
"""

import strawberryfields as sf
import numpy as np
import pylab as plt

fock_eps = 0.2
alpha = 1.2
xvec = np.linspace(-5, 5, 200)

prog_fock = sf.Program(1)
with prog_fock.context as q:
    sf.ops.GKP(epsilon=fock_eps, state=[np.pi/2,0], shape="rectangular", alpha=alpha) | q
cutoff = 40
sf.hbar = 1
eng_fock = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
gkp_fock = eng_fock.run(prog_fock).state
W_fock = gkp_fock.wigner(0, xvec, xvec)

prog_bosonic = sf.Program(1)

with prog_bosonic.context as q:
    sf.ops.GKP(epsilon=fock_eps, state = [np.pi/2,0], shape="rectangular", alpha=alpha) | q
sf.hbar = 1
eng_bosonic = sf.Engine("bosonic")
gkp_bosonic = eng_bosonic.run(prog_bosonic).state
W_bosonic = gkp_bosonic.wigner(0, xvec, xvec)


plt.contourf(xvec/np.sqrt(np.pi*sf.hbar*2),xvec/np.sqrt(np.pi*sf.hbar*2),W_fock,levels=50)
plt.show()

plt.contourf(xvec/np.sqrt(np.pi*sf.hbar*2),xvec/np.sqrt(np.pi*sf.hbar*2),W_bosonic,levels=50)
plt.show()

plt.contourf(xvec/np.sqrt(np.pi*sf.hbar*2),xvec/np.sqrt(np.pi*sf.hbar*2),np.log(abs(W_fock-W_bosonic)),levels=50)
plt.colorbar()
plt.show()