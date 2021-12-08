# %%
import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.utils.program_functions import extract_unitary
from strawberryfields.utils import random_interferometer
from strawberryfields.compilers.compiler import Compiler
from strawberryfields.decompositions import sun_compact

# %%
n_modes = 3
cutoff = n_modes+1
prog = sf.Program(n_modes)
eng = sf.Engine('fock', backend_options={'cutoff_dim': cutoff,  "pure": True})
u1 = random_interferometer(n_modes)

# %%
G = ops.Interferometer(u1, mesh="sun_compact")
cmds = G.decompose(prog.register)

# %%
for cmd in cmds: print(cmd.op, [reg.par for reg in cmd.reg])

# %%
parameters, global_phase = sun_compact(u1)
print(parameters)
print(global_phase)
# %%
