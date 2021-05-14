import numpy as np
from collections import defaultdict

#############Utils##############
def M(n, sigma, delta, m):
    r"""To generate the sMZI matrix.
    
    The Bell M matrix from Eq 1 of the paper (arXiv:2104.0756).
    
    Args:
        n (int): the starting mode of sMZI
        sigma (complex): parameter of the sMZI :math:`\frac{(\theta_1+\theta_2)}{2}`
        delta (complex): parameter of the sMZI :math:`\frac{(\theta_1-\theta_2)}{2}`
        m (int): the length of the unitary matrix to be decomposed
        
    Returns:
        array[complex,complex]: the sMZI matrix between n-th and (n+1)-th mode
    """
    mat = np.identity(m, dtype=np.complex128)
    mat[n, n] = np.exp(1j * sigma) * np.sin(delta)
    mat[n, n+1] = np.exp(1j * sigma) * np.cos(delta)
    mat[n+1, n] = np.exp(1j * sigma) * np.cos(delta)
    mat[n+1, n+1] = -np.exp(1j * sigma) * np.sin(delta)
    return mat

def P(j, phi, m):
    r"""To generate the phase shifter matrix.

    Args:
        j (int): the starting mode of phase-shifter
        phi (complex): parameter of the phase-shifter
        m (int): the length of the unitary matrix to be decomposed
        
    Returns:
        array[complex,complex]: the phase-shifter matrix on the j-th mode
    """
    mat = np.identity(m, dtype=np.complex128)
    mat[j,j] = np.exp(1j * phi)
    return mat
################################

def reck_decompose(U):
    r"""Decomposition of a unitary into the Reck scheme with sMZIs and phase-shifters.

    Args:
        U (array): unitary matrix

    Returns:
        dict[]: returns a dictionary contains all parameters
            where the keywords:
            * ``m``: the length of the matrix
            * ``phi_ins``: parameter of the phase-shifter
            * ``sigmas``: parameter of the sMZI :math:`\frac{(\theta_1+\theta_2)}{2}`
            * ``deltas``: parameter of the sMZI :math:`\frac{(\theta_1-\theta_2)}{2}`
            * ``zetas``: parameter of the phase-shifter

    """
    
    if not U.shape[0] == U.shape[1]:
        raise Exception('Matrix is not square')
    
    V = U.conj()
    m = U.shape[0]
    
    phases = dict()
    phases['m'] = m
    phases['phi_ins'] = dict() # mode : phi
    phases['deltas'] = dict() # (mode, layer) : delta
    phases['sigmas'] = dict() # (mode, layer) : sigma
    phases['zetas'] = dict() # mode : zeta

    for j in range(m-1):
        x = m - 1
        y = j
        phi_j = - np.angle(V[x, y+1]) + np.angle(V[x, y])
        Pj = P(j+1, phi_j, m)
        phases['phi_ins'][j] = phi_j
        V = V @ Pj
        for k in range(j+1):
            n = j - k
            delta = np.arctan(-V[x,y+1] / V[x,y]).real
            V_temp = V @ M(n, 0, delta, m)
            sigma = np.angle(V_temp[x-1, y-1]) - np.angle(V_temp[x-1,y])
            phases['deltas'][n,k] = delta
            phases['sigmas'][n,k] = sigma
            V = V @ M(n, sigma, delta, m)
            x -= 1
            y -= 1
        
    # these next two lines are just to remove a global phase
    zeta = - np.angle(V[0,0])
    phases['zetas'][0] = zeta
    V = V @ P(0, zeta, m)

    for j in range(1,m):
        zeta = np.angle(V[0,0]) - np.angle(V[j,j])
        phases['zetas'][j] = zeta
        V = V @ P(j, zeta, m)
        
    if not np.allclose(V, np.eye(m)):
        raise Exception('decomposition failed')
        
    return phases
    
def reck_recompose(phases):
    m = phases['m']
    U = np.identity(m, dtype=np.complex128)
    for j in range(m-1):
        phi_j = phases['phi_ins'][j]
        U = P(j+1, phi_j, m) @ U
        for k in range(j+1):
            n = j - k
            delta = phases['deltas'][n,k]
            sigma = phases['sigmas'][n,k]
            U = M(n, sigma, delta, m) @ U
    for j in range(m):
        zeta = phases['zetas'][j]
        U = P(j, zeta, m) @ U
    return U

def clement_decompose(U):
    r"""Decomposition of a unitary into the Clement scheme with sMZIs and phase-shifters.

    Args:
        U (array): unitary matrix

    Returns:
        dict[]: returns a dictionary contains all parameters
            where the keywords:
            * ``m``: the length of the matrix
            * ``phi_ins``: parameter of the phase-shifter
            * ``sigmas``: parameter of the sMZI :math:`\frac{(\theta_1+\theta_2)}{2}`
            * ``deltas``: parameter of the sMZI :math:`\frac{(\theta_1-\theta_2)}{2}`
            * ``zetas``: parameter of the phase-shifter
            * ``phi_outs``: parameter of the phase-shifter

    """
    if not U.shape[0] == U.shape[1]:
        raise Exception('Matrix is not square')
            
    V = U.conj()
    
    m = U.shape[0]
        
    phases = dict()
    phases['m'] = m
    phases['phi_ins'] = dict() # mode : phi
    phases['deltas'] = dict() # (mode, layer) : delta
    phases['sigmas'] = dict() # (mode, layer) : sigma
    phases['zetas'] = dict() # mode : zeta
    phases['phi_outs'] = dict() # mode : phi
        
    for j in range(m-1):
        if j % 2 == 0:
            x = m - 1
            y = j
            phi_j = np.angle(V[x, y+1]) - np.angle(V[x, y]) # reversed order from paper
            V = V @ P(j, phi_j, m)
            phases['phi_ins'][j] = phi_j
            for k in range(j+1):
                if V[x,y] == 0:
                    delta = 0.5 * np.pi
                else:
                    delta = np.arctan2(-abs(V[x,y+1]), abs(V[x,y]))
                n = j - k
                V_temp = V @ M(n, 0, delta, m)
                sigma = np.angle(V_temp[x-1, y-1]) - np.angle(V_temp[x-1,y])
                V = V @ M(n, sigma, delta, m)
                phases['deltas'][n,k] = delta
                phases['sigmas'][n,k] = sigma
                x -= 1
                y -= 1
        else:
            x = m - j - 1
            y = 0
            phi_j = np.angle(V[x-1,y]) - np.angle(V[x,y])
            V = P(x, phi_j, m) @ V
            phases['phi_outs'][x] = phi_j
            for k in range(j+1):
                if V[x,y] == 0.:
                    delta = 0.5 * np.pi
                else:
                    delta = np.arctan2(abs(V[x-1,y]), abs(V[x,y]))
                V_temp = M(x-1, 0, delta, m) @ V
                n = m + k - j - 2
                if j != k:
                    sigma = (np.angle(V_temp[x+1, y+1]) - np.angle(V_temp[x,y+1]))
                else:
                    sigma = 0
                phases['deltas'][n,m-k-1] = delta
                phases['sigmas'][n,m-k-1] = sigma
                V = M(n, sigma, delta, m) @ V
                x += 1
                y += 1

    # these next two lines are just to remove a global phase
    zeta = - np.angle(V[0,0])
    V = V @ P(0, zeta, m)
    phases['zetas'][0] = zeta

    for j in range(1,m):
        zeta = np.angle(V[0,0]) - np.angle(V[j,j])
        V = V @ P(j, zeta, m)
        phases['zetas'][j] = zeta
        
    if not np.allclose(V, np.eye(m)): #is_unitary(V):
        raise Exception('decomposition failed')
    return phases
    
def clement_recompose(phases):
    m = phases['m']
    U = np.eye(m, dtype=np.complex128)
    
    #upper left of interferometer
    for j in range(0,m-1,2):
        phi_j = phases['phi_ins'][j]
        U = P(j, phi_j, m) @ U
        for k in range(j+1):
            n = j - k
            delta = phases['deltas'][n,k]
            sigma = phases['sigmas'][n,k]
            U = M(n, sigma, delta, m) @ U
            
    #diagonal phases
    for j in range(m):
        zeta = phases['zetas'][j]
        U = P(j, zeta, m) @ U
    
    #lower right of interferometer
    for j in reversed(range(1,m-1,2)):
        for k in reversed(range(j+1)):
            n = m + k - j - 2
            delta = phases['deltas'][n,m-k-1]
            sigma = phases['sigmas'][n,m-k-1]
            U = M(n, sigma, delta, m) @ U
            
    for j in range(1,m-1,2):
        x = m - j - 1
        phi_j = phases['phi_outs'][x]
        U = P(x, phi_j, m) @ U
    return U
    
def absorb_zeta(phases):
    m = phases['m']
    new_phases = phases.copy()
    del new_phases['zetas']
    new_phases['phi_edges'] = defaultdict(float) # (mode, layer) : phi
    
    if m % 2 == 0:
        new_phases['phi_outs'][0] = phases['zetas'][0]
        for j in range(1,m):
            zeta = phases['zetas'][j]
            layer = m - j
            for mode in range(j,m-1,2):
                new_phases['sigmas'][mode, layer] += zeta
            for mode in range(j+1,m-1,2):
                new_phases['sigmas'][mode, layer-1] -= zeta
            if layer % 2 == 1:
                new_phases['phi_edges'][m-1, layer] += zeta
            else:
                new_phases['phi_edges'][m-1, layer-1] -= zeta
    else:
        for j in range(m):
            zeta = phases['zetas'][j]
            layer =  m - j - 1
            for mode in range(j,m-1,2):
                new_phases['sigmas'][mode, layer] += zeta
            for mode in range(j+1,m-1,2):
                new_phases['sigmas'][mode, layer-1] -= zeta
            if layer % 2 == 0:
                new_phases['phi_edges'][m-1, layer] += zeta
            else:
                new_phases['phi_edges'][m-1, layer-1] -= zeta
    return new_phases
    
def rectangle_compact_decompose(U):
    r"""Decomposition of a unitary into the Clement scheme with sMZIs and phase-shifters and absorb the phase-shifter between the sMZIs.

    Args:
        U (array): unitary matrix

    Returns:
        dict[]: returns a dictionary contains all parameters
            where the keywords:
            * ``m``: the length of the matrix
            * ``phi_ins``: parameter of the phase-shifter
            * ``sigmas``: parameter of the sMZI :math:`\frac{(\theta_1+\theta_2)}{2}`
            * ``deltas``: parameter of the sMZI :math:`\frac{(\theta_1-\theta_2)}{2}`
            * ``phi_outs``: parameter of the phase-shifter
            * ``phi_edges``: parameter of the phase-shifter

    """
    phases_temp = clement_decompose(U)
    return absorb_zeta(phases_temp)

def rectangle_compact_recompose(phases):
    m = phases['m']
    U = np.eye(m, dtype=np.complex128)
    for j in range(0,m-1,2):
        phi = phases['phi_ins'][j]
        U = P(j, phi, m) @ U
    for layer in range(m):
        if (layer + m + 1) % 2 == 0:
            phi_bottom = phases['phi_edges'][m-1, layer]
            U = P(m-1, phi_bottom, m) @ U
        for mode in range(layer % 2, m-1, 2):
            delta = phases['deltas'][mode, layer]
            sigma = phases['sigmas'][mode, layer]
            U = M(mode, sigma, delta, m) @ U
    for j, phi_j in phases['phi_outs'].items():
        U = P(j, phi_j, m) @ U
    return U
