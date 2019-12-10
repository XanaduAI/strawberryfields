.. code-block:: python

   prog = sf.Program(num_subsystems)
   with prog.context as q:
       Coherent(0.5)  | q[0]
       Vac            | q[1]
       Sgate(2)       | q[1]
       BSgate(1)      | q[0:2]
       Rgate(0.5)     | q[0]
       Dgate(0.5).H   | q[1]
       MeasureFock()  | q

   eng = sf.LocalEngine(backend='fock', backend_options={'cutoff_dim': 5})
   result = eng.run(prog)
   assert prog.register[0].val == result.samples[0]
