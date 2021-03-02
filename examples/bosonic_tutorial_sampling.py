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

"""A comparison of the asymptotic homodyne distributions and sampled values,
for GKP and cat states."""

import strawberryfields as sf
import numpy as np
import matplotlib.pyplot as plt

# Create cat state
prog_cat = sf.Program(1)
with prog_cat.context as q:
    sf.ops.Catstate(alpha=2) | q
eng = sf.Engine("bosonic")
cat = eng.run(prog_cat).state

# Calculate the asymptotic quadrature distributions
scale = np.sqrt(sf.hbar)
quad = np.linspace(-6, 6, 1000) * scale
cat_prob_x = cat.marginal(0, quad)
cat_prob_p = cat.marginal(0, quad, phi=np.pi / 2)

# Run the program again, collecting x samples this time
prog_cat_x = sf.Program(1)
with prog_cat_x.context as q:
    sf.ops.Catstate(alpha=2) | q
    sf.ops.MeasureX | q
eng = sf.Engine("bosonic")
cat_samples_x = eng.run(prog_cat_x, shots=2000).samples[:, 0]

# Run the program again, collecting p samples this time
prog_cat_p = sf.Program(1)
with prog_cat_p.context as q:
    sf.ops.Catstate(alpha=2) | q
    sf.ops.MeasureP | q
eng = sf.Engine("bosonic")
cat_samples_p = eng.run(prog_cat_p, shots=2000).samples[:, 0]

# Plot the results
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(r"$|0^{\alpha}\rangle_{cat}$, $\alpha=2$", fontsize=18)

axs[0].hist(cat_samples_x / scale, bins=100, density=True, label="Samples", color="cornflowerblue")
axs[0].plot(quad / scale, cat_prob_x * scale, "--", label="Ideal", color="tab:red")
axs[0].set_xlabel(r"q (units of $\sqrt{\hbar}$)", fontsize=15)
axs[0].set_ylabel("pdf(q)", fontsize=15)

axs[1].hist(cat_samples_p / scale, bins=100, density=True, label="Samples", color="cornflowerblue")
axs[1].plot(quad / scale, cat_prob_p * scale, "--", label="Ideal", color="tab:red")
axs[1].set_xlabel(r"p (units of $\sqrt{\hbar}$)", fontsize=15)
axs[1].set_ylabel("pdf(p)", fontsize=15)

axs[1].legend()
axs[0].tick_params(labelsize=13)
axs[1].tick_params(labelsize=13)
plt.show()


# Create GKP state
prog_gkp = sf.Program(1)
with prog_gkp.context as q:
    sf.ops.GKP(epsilon=0.1) | q
eng = sf.Engine("bosonic")
gkp = eng.run(prog_gkp).state

# Calculate the asymptotic quadrature distributions
scale = np.sqrt(sf.hbar * np.pi)
quad = np.linspace(-6, 6, 1000) * scale
gkp_prob_x = gkp.marginal(0, quad)
gkp_prob_p = gkp.marginal(0, quad, phi=np.pi / 2)

# Run the program again, collecting x samples this time
prog_gkp_x = sf.Program(1)
with prog_gkp_x.context as q:
    sf.ops.GKP(epsilon=0.1) | q
    sf.ops.MeasureX | q
eng = sf.Engine("bosonic")
gkp_samples_x = eng.run(prog_gkp_x, shots=2000).samples[:, 0]

# Run the program again, collecting p samples this time
prog_gkp_p = sf.Program(1)
with prog_gkp_p.context as q:
    sf.ops.GKP(epsilon=0.1) | q
    sf.ops.MeasureP | q
eng = sf.Engine("bosonic")
gkp_samples_p = eng.run(prog_gkp_p, shots=2000).samples[:, 0]

# Plot the results
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(r"$|0^\epsilon\rangle_{GKP}$, $\epsilon=0.1$ (10 dB)", fontsize=18)

axs[0].hist(gkp_samples_x / scale, bins=100, density=True, label="Samples", color="cornflowerblue")
axs[0].plot(quad / scale, gkp_prob_x * scale, "--", label="Ideal", color="tab:red")
axs[0].set_xlabel(r"q (units of $\sqrt{\pi\hbar}$)", fontsize=15)
axs[0].set_ylabel("pdf(q)", fontsize=15)

axs[1].hist(gkp_samples_p / scale, bins=100, density=True, label="Samples", color="cornflowerblue")
axs[1].plot(quad / scale, gkp_prob_p * scale, "--", label="Ideal", color="tab:red")
axs[1].set_xlabel(r"p (units of $\sqrt{\pi\hbar}$)", fontsize=15)
axs[1].set_ylabel("pdf(p)", fontsize=15)

axs[1].legend()
axs[0].tick_params(labelsize=13)
axs[1].tick_params(labelsize=13)
plt.show()
