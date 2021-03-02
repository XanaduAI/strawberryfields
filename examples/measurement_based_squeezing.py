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

"""A comparison of ideal squeezing, and measurement-based squeezing applied using 
the average and single shot maps."""

import strawberryfields as sf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Values of target squeezing
r = [0.3, 1, 2]

# Style choices for plots
color = ["tab:red", "seagreen", "cornflowerblue"]
style = [":", "--", "-"]
label = ["Ideal", "Average map", "Single shot"]
e = [0, 1, 2]

# Create plot
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

# Loop over target r values
for j in range(3):

    prog = sf.Program(3)
    with prog.context as q:
        # Ideal squeezing
        sf.ops.Sgate(r[j], 0) | q[0]
        # Average map of measurement-based squeezing
        sf.ops.MSgate(r[j], 0, r_anc=1.2, eta_anc=0.99, avg=True) | q[1]
        # Single-shot map of measurement-based squeezing
        sf.ops.MSgate(r[j], 0, r_anc=1.2, eta_anc=0.99, avg=False) | q[2]
    eng = sf.Engine("bosonic")
    state = eng.run(prog).state

    # Quadrature values
    x = np.linspace(-4, 4, 1000) * np.sqrt(sf.hbar) * r[j]
    p = np.linspace(-4, 4, 1000) * np.sqrt(sf.hbar) * r[j]

    # Plot ellipses for each squeezed state
    for i in range(3):
        _, mean, cov = state.reduced_bosonic(i)
        width = 2 * np.sqrt(cov[0, 0, 0])
        height = 2 * np.sqrt(cov[0, 1, 1])
        e[i] = Ellipse(
            xy=mean[0],
            width=width,
            height=height,
            facecolor="none",
            edgecolor=color[i],
            linestyle=style[i],
        )
        ax[j].add_artist(e[i])

    ax[j].set_title(r"$r_{target}=$" + str(r[j]))
    ax[j].set_xlabel(r"q (units of $\sqrt{\hbar}$)", fontsize=12)
    ax[j].set_xlim(x[0], x[-1])
    ax[j].set_ylim(p[0], p[-1])

ax[0].set_ylabel(r"p (units of $\sqrt{\hbar}$)", fontsize=12)
ax[1].legend(e, label, ncol=3, bbox_to_anchor=(0.5, -0.3), loc="lower center")
plt.show()
