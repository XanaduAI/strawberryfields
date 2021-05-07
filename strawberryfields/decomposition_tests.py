import numpy as np 

def M(n, sigma, delta, m):
    r"""The Bell M matrix from Eq 1 of the paper
    n here represents the starting mode of MZI, and n = k-j
    """
    mat = np.identity(m, dtype=np.complex128)
    mat[n, n] = np.exp(1j * sigma) * np.sin(delta)
    mat[n, n+1] = np.exp(1j * sigma) * np.cos(delta)
    mat[n+1, n] = np.exp(1j * sigma) * np.cos(delta)
    mat[n+1, n+1] = -np.exp(1j * sigma) * np.sin(delta)
    return mat

def P(j, phi, m):
    mat = np.identity(m, dtype=np.complex128)
    mat[j,j] = np.exp(1j * phi)
    return mat
    
def is_unitary(U):
    return np.allclose(U @ U.conj().T, np.eye(U.shape[0]))

def Reck_decomposition(U):
    V = U.conj()
    #TODO : Check if it is a square matrix or not
    
    m = U.shape[0]
    
    #{"P":[P1,...Pj],
    #"Q":[Q1,...Qj],
    #"j=0":[[delta_0,sigma_0],],
    #"j=1":[[delta_0,sigma_0],[delta_1,sigma_1]]}
    #...
    params_list = {}
    params_list["P_"] = []
    params_list["Q_"] = []
    params_list["global_zeta"] = []
    for j in range(m-1):
        params_list[j] = []

    for j in range(m-1):
        x = m - 1
        y = j
        phi_j = np.angle(V[x, y+1]) - np.angle(V[x, y])
        Pj = P(j, phi_j, m)
        params_list["P_"].append(phi_j)
        V = V @ Pj
        for k in range(j+1):
            delta = np.arctan(-V[x,y+1] / V[x,y]).real
            V_temp = V @ M(j-k, 0, delta, m)
            sigma = np.angle(V_temp[x-1, y-1]) - np.angle(V_temp[x-1,y])
            params_list[j].append([delta,sigma])
            n = j - k
            V = V @ M(n, sigma, delta, m)
            x -= 1
            y -= 1
        
    # these next two lines are just to remove a global phase
    zeta = - np.angle(V[0,0])
    params_list["global_zeta"].append(zeta)
    V = V @ P(0, zeta, m)

    for j in range(1,m):
        zeta = np.angle(V[0,0]) - np.angle(V[j,j])
        params_list["Q_"].append(zeta)
        V = V @ P(j, zeta, m)
        
    assert is_unitary(V), "Wrong decomposition!"
    print("V matrix is unitary:",V)
    return params_list
    
    
def Reck_reconstruction(m,params_list):
    U = np.identity(m, dtype=np.complex128)
    for j in range(m-1):
        phi = params_list["P_"][j]
        U = P(j,phi,m) @ U
        for k in range(len(params_list[j])):
            delta = params_list[j][k][0]
            sigma = params_list[j][k][1]
            U = M(j-k, sigma, delta, m) @ U
    U = P(0, params_list["global_zeta"][0], m) @ U
    for j in range(1,m):
        zeta = params_list["Q_"][j-1]
        U = P(j, zeta, m) @ U
    return U

def Clement_decomposition(U):
    V = U.conj()
    
    m = U.shape[0]
        
    params_list = {}
    params_list["P_"] = []
    params_list["Q_"] = []
    params_list["global_zeta"] = []
    for j in range(m-1):
        params_list[j] = []
        
    for j in range(m-1):
        #odd case in paper, because we index from 0 not 1
        if j % 2 == 0:
            x = m - 1
            y = j
            phi_j = np.angle(V[x, y+1]) - np.angle(V[x, y]) # reversed order from paper
            params_list["P_"].append(phi_j)
            V = V @ P(j, phi_j, m)
            for k in range(j+1):
                delta = np.arctan(-V[x,y+1] / V[x,y]) # flipped from paper
                n = j - k
                V_temp = V @ M(n, 0, delta, m)
                sigma = np.angle(V_temp[x-1, y-1]) - np.angle(V_temp[x-1,y])
                params_list[j].append([delta,sigma])
                V = V @ M(n, sigma, delta, m)
                x -= 1
                y -= 1
        else:
            x = m - j - 1
            y = 0
            phi_j = np.angle(V[x-1,y]) - np.angle(V[x,y])
            params_list["P_"].append(phi_j)
            V = P(x, phi_j, m) @ V
            for k in range(j+1):
                delta = np.arctan(V[x-1,y] / V[x,y]) # flipped from paper
                V_temp = M(x-1, 0, delta, m) @ V
                n = m + k - j - 2
                if j != k:
                    sigma = (np.angle(V_temp[x+1, y+1]) - np.angle(V_temp[x,y+1]))
                else:
                    sigma = 0
                params_list[j].append([delta,sigma])
                V = M(n, sigma, delta, m) @ V
                x += 1
                y += 1

    # these next two lines are just to remove a global phase
    zeta = - np.angle(V[0,0])
    params_list["global_zeta"].append(zeta)
    V = V @ P(0, zeta, m)

    for j in range(1,m):
        zeta = np.angle(V[0,0]) - np.angle(V[j,j])
        params_list["Q_"].append(zeta)
        V = V @ P(j, zeta, m)
        
        assert is_unitary(V), "Wrong decomposition!"
    print("V matrix is unitary:",V)
    return params_list
            
def Clement_reconstruction(m,params_list):
    U_Q = np.identity(m, dtype=np.complex128)
    U_odd = np.identity(m, dtype=np.complex128)
    U_even = np.identity(m, dtype=np.complex128)

    for j in range(m-1):
        if j % 2 == 0:
            
            phi = params_list["P_"][j]
            U_odd =  U_odd@P(j,phi,m)
            for k in range(len(params_list[j])):
                delta = params_list[j][k][0]
                sigma = params_list[j][k][1]
                U_odd = U_odd@M(j-k, sigma, delta, m)
        else:
            phi = params_list["P_"][j]
            U_even =    P(m - j - 1,phi,m)@U_even
            for k in range(len(params_list[j])):
                delta = params_list[j][k][0]
                sigma = params_list[j][k][1]
                U_even = M(m + k - j - 2, sigma, delta, m)@U_even
            

    for j in range(1,m):
        zeta = params_list["Q_"][j-1]
        U_Q = P(j, zeta, m) @ U_Q
        
    phase = P(0, params_list["global_zeta"][0], m)

    U_right = U_odd@phase@U_Q
    U_left = U_even
    return (U_right@U_left).T
