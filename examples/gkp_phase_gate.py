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

"""A comparison of the Wigner function for a finite energy GKP |+i> state
to the output of a GKP qubit phase gate applied to a finite energy |+> state.
The qubit phase gate is applied using measurement-based squeezing."""

import strawberryfields as sf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, colorbar


sf.hbar = 1
x = np.linspace(-4, 4, 400) * np.sqrt(np.pi)
p = np.linspace(-4, 4, 400) * np.sqrt(np.pi)
wigners = []

# Finite energy GKP |+i> state
prog_y = sf.Program(1)
with prog_y.context as q:
    sf.ops.GKP([np.pi / 2, -np.pi / 2], epsilon=0.1) | q[0]
eng = sf.Engine("bosonic")
result = eng.run(prog_y)
wigners.append(result.state.wigner(0, x, p))

# Squeezing values for measurement-based squeezing ancillae
rdB = np.array([14, 10, 6])
r_anc = np.log(10) * rdB / 20

# Detection efficiencies
eta_anc = np.array([1, 0.98, 0.95])

# Target squeezing and rotation angles to apply a GKP phase gate
r = np.arccosh(np.sqrt(5 / 4))
theta = np.arctan(0.5)
phi = -np.pi / 2 * np.sign(0.5) - theta

# Apply a GKP phase gate to |+> for three different qualities of
# measurement-based squeezing.
for i in range(3):
    prog = sf.Program(1)
    with prog.context as q:
        sf.ops.GKP([np.pi / 2, 0], epsilon=0.1) | q[0]
        sf.ops.MSgate(r, phi, r_anc[i], eta_anc[i], avg=True) | q[0]
        sf.ops.Rgate(theta) | q[0]
    eng = sf.Engine("bosonic")
    result = eng.run(prog)
    wigners.append(result.state.wigner(0, x, p))


# Plotting
cmax = np.real_if_close(np.amax(np.array(wigners)))
cmin = np.real_if_close(np.amin(np.array(wigners)))
if abs(cmin) < cmax:
    cmin = -cmax
else:
    cmax = -cmin

fig, axs = plt.subplots(1, 5, figsize=(16, 4), gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.05]})
cmap = plt.cm.RdBu
norm = colors.Normalize(vmin=cmin, vmax=cmax)
cb1 = colorbar.ColorbarBase(axs[4], cmap=cmap, norm=norm, orientation="vertical")
ims = np.empty(4, dtype=object)
axs[0].set_ylabel(r"p (units of $\sqrt{\hbar\pi}$)", fontsize=15)
axs[0].set_title(r"$|+i^\epsilon\rangle_{gkp}$, $\epsilon$=0.1 (10 dB)", fontsize=15)
for i in range(4):
    ims[i] = axs[i].contourf(
        x / np.sqrt(np.pi),
        p / np.sqrt(np.pi),
        wigners[i],
        levels=60,
        cmap=plt.cm.RdBu,
        vmin=cmin,
        vmax=cmax,
    )
    axs[i].set_xlabel(r"q (units of $\sqrt{\hbar\pi}$)", fontsize=15)
    if i > 0:
        axs[i].set_title(
            r"$r_{anc}=$" + str(rdB[i - 1]) + r"dB, $\sqrt{\eta}=$" + str(eta_anc[i - 1]),
            fontsize=15,
        )
    for c in ims[i].collections:
        c.set_edgecolor("face")
    axs[i].tick_params(labelsize=14)
cb1.set_label("Wigner function", fontsize=15)
fig.tight_layout()
plt.show()
