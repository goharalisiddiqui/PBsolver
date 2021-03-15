import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import scipy.linalg
import matrixGenerator as mg


## Physical Constants
kb = 1.38064852e-23
el_c = 1.60217662e-19
perm_0 = 8.854178176e-12
Na = 6.022140857e23
Far = Na*el_c


def phSolver(space_grid, perm_grid, NPSD, tol, T, Kon1, Kon2, Koff1, Koff2, Bulk_ionic, Bulk_H, C_max):
    
    V_T = (kb*T)/el_c

    ## Matric Setup
    cap_mat = mg.get_capMat_1D_cartesian(space_grid,perm_grid,'N','D')
    h = np.append(space_grid[1:] - space_grid[:-1],[0.0])
    


    ## Electric Potential
    V_grid = np.zeros(NPSD)

    ## Ionic Charges
    rho_pos_ion = np.zeros(NPSD)
    rho_pos_H = np.zeros(NPSD)
    rho_neg_ion = np.zeros(NPSD)
    rho_neg_H = np.zeros(NPSD)

    ## Trap states
    rho_ct1 = np.zeros(NPSD)
    rho_ct2 = np.zeros(NPSD)
    rho_ct0 = np.zeros(NPSD)

    delta = 1.0
    while delta > tol:
        
        ## Positive Charges
        rho_pos_ion = Bulk_ionic*np.exp(-V_grid/V_T)
        rho_pos_H = Bulk_H*np.exp(-V_grid/V_T)
        

        ## Negative Charges
        rho_neg_ion = -Bulk_ionic*np.exp(V_grid/V_T)
        rho_neg_H = -Bulk_H*np.exp(V_grid/V_T)


        K = np.array([[Kon1*rho_pos_H[0],-Koff1,0],[0,Kon2*rho_pos_H[0],-Koff2],[1,1,1]])
        C = np.array([0,0,C_max])
        X = scipy.linalg.solve(K,C)
        rho_ct0[0] = X[0]
        rho_ct1[0] = X[1]
        rho_ct2[0] = X[2]

        ##Total charge
        rho_total = Far*(rho_pos_H + rho_pos_ion + rho_neg_H + rho_neg_ion + (rho_ct2/(h[0]/2)) - (rho_ct0/(h[0]/2)))
        
        ##Boundary conditions for Bulk   
        rho_total[-1] = 0.0                                                     #Dirichlet Condition for right boundary
        

        #Total Function
        F_tot = (cap_mat.dot(V_grid)) - rho_total

        #Derivative of trap states
        K2 = np.array([[Kon1*rho_pos_H[0],-Koff1,0],[0,Kon2*rho_pos_H[0],-Koff2],[1,1,1]])
        C2 = np.array([-(Kon1*(-1/V_T)*rho_pos_H[0]*rho_ct0[0]),-(Kon2*(-1/V_T)*rho_pos_H[0]*rho_ct1[0]),0])
        X2 = scipy.linalg.solve(K2,C2)
        rho_V_ct0 = X2[0]
        rho_V_ct1 = X2[1]
        rho_V_ct2 = X2[2]
        
        
        #Jacobian
        P = Far*(((-1/V_T)*(rho_pos_H+rho_pos_ion)) + ((1/V_T)*(rho_neg_H+rho_neg_ion)))
        P[0] = P[0] - ((Far*rho_V_ct0)/(h[0]/2)) + ((Far*rho_V_ct2)/(h[0]/2))
        Jac = cap_mat - (sparse.dia_matrix((P, [0]), shape=(NPSD, NPSD)))
        
        y = scipy.sparse.linalg.spsolve(Jac, F_tot)
        
        ## Dampen large errors
        for m,j in enumerate(y):
            if j>V_T:
                y[m] = V_T
            elif j<-V_T:
                y[m] = -V_T

        
        ## Update potential
        V_grid = V_grid - y
        
        ## Calculate norm
        #print(scipy.linalg.norm(y))
        delta = scipy.linalg.norm(y)

    ## Final Positive Charges
    rho_pos_ion = Bulk_ionic*np.exp(-V_grid/V_T)
    rho_pos_H = Bulk_H*np.exp(-V_grid/V_T)


    ## Final Negative Charges
    rho_neg_ion = Bulk_ionic*np.exp(V_grid/V_T)
    rho_neg_H = Bulk_H*np.exp(V_grid/V_T)

    return V_grid,rho_pos_ion,rho_neg_ion, rho_pos_H, rho_neg_H, rho_ct0, rho_ct1, rho_ct2