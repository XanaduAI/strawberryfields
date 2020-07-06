

def tdm_program(r, phi, theta, homodyne, loss=None, shots=1):
    """docstring"""
    prog = sf.Program(3)

    r = np.tile(r, shots)
    phi = np.tile(phi, shots)
    theta = np.tile(theta, shots)

    with prog.context as q:

        for p, t, h in zip(phi, theta, homodyne):
            ops.Sgate(r, 0) | q[2]

            if loss is not None:
                ops.ThermalLossChannel(loss[0], loss[1]) | q[2]

            ops.Rgate(p) | q[1]
            ops.BSgate(t) | (q[2], q[1])
            ops.MeasureHomodyne(h) | q[0]
            q = q[1:] + q[:1]

    return prog