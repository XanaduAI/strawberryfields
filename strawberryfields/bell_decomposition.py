import numpy as np 

def M(j, k, sigma, delta, m):
    r"""The Bell M matrix from Eq 1 of the paper"""
    mat = np.identity(m, dtype=np.complex128)
    mat[j, j] = np.exp(1j * sigma) * np.sin(delta)
    mat[j, k] = np.exp(1j * sigma) * np.cos(delta)
    mat[k, j] = np.exp(1j * sigma) * np.cos(delta)
    mat[k, k] = -np.exp(1j * sigma) * np.sin(delta)
    return mat

def P(j, phi, m):
    mat = np.identity(m, dtype=np.complex128)
    mat[j,j] = np.exp(1j * phi)
    return mat

def null_M_odd(j, k, x, y, V):
    if V[x-1,y-1] == 0:
        delta = 0.5 * np.pi
    elif y == j:
        r = - V[x-1,k-1] / V[x-1,j-1]
        delta = np.arctan(r)
    elif y == k:
        if V[x-1,k-1] == 0:
            delta = 0.
        else:
            r = V[x-1,j-1] / V[x-1,k-1]
            delta = np.arctan(r)
    else:
        raise ValueError('invalid nulled element for the chosen MZI')

    sigma = np.angle(V[x-2,y-2]) - np.angle(V[x-2,y-1])
    return delta, sigma

def rectangular_compact(U, atol=1e-11, rtol=1e-11):

    V = U.conj()
    m = localV.shape[0]

    if not np.allclose(V @ V.conj().T, np.identity(nsize), atol=tol, rtol=0):
        raise ValueError("The input matrix is not unitary")

    phi_list = []
    M_odd_list = []
    M_even_list = []

    for j in range(1,m):
        if j % 2 == 1:
            x = m 
            y = j 
            phi_j = np.angle(V[x-1,y-1]) - np.angle(V[x-1,y])
            phi_list.append((j, phi_j))
            V = V @ P(j, phi_j, m)
            for k in range(1, j+1):
                sigma_jk, delta_jk = null_M_odd(j, k, x, y, V)
                V = V @ M(j, k, sigma_jk, delta_jk)
                assert np.isclose(V[x-1,y-1], 0, atol=atol, rtol=rtol)
