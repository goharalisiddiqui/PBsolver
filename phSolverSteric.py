import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import scipy.linalg
import matrixGenerator as mg
import math


## Physical Constants
kb = 1.38064852e-23
el_c = 1.60217662e-19
perm_0 = 8.854178176e-12
Na = 6.022140857e23
Far = Na*el_c


def phSolver(space_grid, perm_grid, NPSD, tol, T, Kon1, Kon2, Koff1, Koff2, Bulk_ionic, Bulk_H, C_max, eff_sizes, stern_size):
    
    V_T = (kb*T)/el_c
    C_max = C_max/Na
    
    ## Matric Setup
    cap_mat = mg.get_capMat_1D_cartesian(space_grid,perm_grid,'N','D')
    h = np.append(space_grid[1:] - space_grid[:-1],[0.0])
    
    ## Potential Overflow flags
    overflow1 = False
    overflow2 = False

    ## Electric Potential
    V_grid = np.zeros(NPSD)

    ## Surface states Concentration
    ct1 = np.zeros(NPSD)        #[SiO-]
    ct2 = np.zeros(NPSD)        #[SiOH]
    ct0 = np.zeros(NPSD)        #[SiOH2+]


    ## Size effect Parameters
    eff_size_H = eff_sizes[2]*1e-10 #0.311e-10
    eff_size_OH = eff_sizes[3]*1e-10 #0.311e-10
    eff_size_posI = eff_sizes[0]*1e-10 #5e-10
    eff_size_negI = eff_sizes[1]*1e-10 #5e-10
    ref_activity_H = 1.0
    ref_activity_OH = 1.0
    ref_activity_posI = 1.0
    ref_activity_negI = 1.0


    ## Stern Layer
    stern_fac = np.ones(NPSD)
    for ind, point in enumerate(space_grid):
        if point<=stern_size:
            stern_fac[ind] = 0.0


    ## Derived Parameters
    molar_vol_H = Na*math.pow(eff_size_H,3)
    molar_vol_OH = Na*math.pow(eff_size_OH,3)
    molar_vol_posI = Na*math.pow(eff_size_posI,3)
    molar_vol_negI = Na*math.pow(eff_size_negI,3)
    
    c_bar_H = 1.0/molar_vol_H
    c_bar_OH = 1.0/molar_vol_OH
    c_bar_posI = 1.0/molar_vol_posI
    c_bar_negI = 1.0/molar_vol_negI
        
    

    PHI = ((1.0/c_bar_H)*Bulk_H) + ((1.0/c_bar_OH)*Bulk_H) + ((1.0/c_bar_posI)*Bulk_ionic) + ((1.0/c_bar_negI)*Bulk_ionic)
    try:
            q_fermi_H = V_T*(math.log((ref_activity_H*Bulk_H)/(c_bar_H*(1.0-PHI))))
            q_fermi_OH = -V_T*(math.log( (ref_activity_OH*Bulk_H) / (c_bar_OH*(1.0-PHI)) ) )
            q_fermi_posI = V_T*(math.log((ref_activity_posI*Bulk_ionic)/(c_bar_posI*(1.0-PHI))))
            q_fermi_negI = -V_T*(math.log( (ref_activity_negI*Bulk_ionic) / (c_bar_negI*(1.0-PHI)) ) )
    
    except ValueError:
            print("Value error encountered. This is probably because some bulk concentration is defined higher\
            than the maximum value corresponding to the size. Exiting....")
            exit()
   
    delta = 1.0
    while delta > tol:
   
        if overflow1 == False and any(abs(V_grid) > 9.2):
                overflow1 = True
                print("Surface potential overflown first critical of 9.2 Volts. Check the parameters. Results will be less accurate")

        if overflow2 == False and any(abs(V_grid) > 18.3):
                overflow2 = True
                print("Surface potential overflown second critical of 18.4 Volts. Cannot Continue now. Exiting......")
                exit()



        H_f = np.exp((q_fermi_H-V_grid)/V_T)
        OH_f = np.exp(-(q_fermi_OH-V_grid)/V_T)
        posI_f = stern_fac*np.exp((q_fermi_posI-V_grid)/V_T)
        negI_f = stern_fac*np.exp(-(q_fermi_negI-V_grid)/V_T)

        ## Steric Effect
        steric_f = (1.0 + (H_f/ref_activity_H) + (OH_f/ref_activity_OH) + (posI_f/ref_activity_posI) + (negI_f/ref_activity_negI))
        steric_H = ref_activity_H * steric_f
        steric_OH = ref_activity_OH * steric_f
        steric_posI = ref_activity_posI * steric_f
        steric_negI = ref_activity_negI * steric_f



        H_c = c_bar_H * (H_f/steric_H)
        OH_c = c_bar_OH * (OH_f/steric_OH)
        posI_c = c_bar_posI * (posI_f/steric_posI)
        negI_c =  c_bar_negI * (negI_f/steric_negI)    
       
        
        # Charge densities
        rho_pos_H = Far*H_c
        rho_neg_H = -Far*OH_c
        rho_pos_ion = Far*posI_c
        rho_neg_ion = -Far*negI_c



        ## Surface States
        K = np.array([[Kon1*H_c[0],-Koff1,0],[0,Kon2*H_c[0],-Koff2],[1,1,1]])
        C = np.array([0,0,C_max])
        X = scipy.linalg.solve(K,C)
        ct0[0] = X[0]
        ct1[0] = X[1]
        ct2[0] = X[2]

        rho_ct0 = -Far*ct0/(h[0]/2.0)
        rho_ct2 = Far*ct2/(h[0]/2.0)

        ##Total charge
        rho_total = rho_pos_H + rho_pos_ion + rho_neg_H + rho_neg_ion + rho_ct2 + rho_ct0
        
        ##Boundary conditions for Bulk   
        rho_total[-1] = 0.0                                                     #Dirichlet Condition for right boundary
        

        #Total Function
        F_tot = cap_mat.dot(V_grid) - rho_total

        
        
        
        #Jacobian
    
        P_H = ((-Far*c_bar_H)/V_T) * (ref_activity_H/np.power(np.minimum(steric_H,1e154),2))\
                                * (H_f + ((2.0/ref_activity_OH)*np.exp((q_fermi_H-q_fermi_OH)/V_T)) \
                                    + ((2.0/ref_activity_negI)*np.exp((q_fermi_H-q_fermi_negI)/V_T)))

        P_posI = ((-Far*c_bar_posI)/V_T) * (ref_activity_posI/np.power(np.minimum(steric_posI,1e154),2))\
                                * (posI_f + ((2.0/ref_activity_OH)*np.exp((q_fermi_posI-q_fermi_OH)/V_T)) \
                                    + ((2.0/ref_activity_negI)*np.exp((q_fermi_posI-q_fermi_negI)/V_T)))
    
        P_OH = ((-Far*c_bar_OH)/V_T) * (ref_activity_OH/np.power(np.minimum(steric_OH,1e154),2))\
                                * (OH_f + ((2.0/ref_activity_H)*np.exp((q_fermi_H-q_fermi_OH)/V_T)) \
                                    + ((2.0/ref_activity_posI)*np.exp((q_fermi_posI-q_fermi_OH)/V_T)))
        
        P_negI = ((-Far*c_bar_negI)/V_T) * (ref_activity_negI/np.power(np.minimum(steric_negI,1e154),2))\
                                * (negI_f + ((2.0/ref_activity_H)*np.exp((q_fermi_H-q_fermi_negI)/V_T)) \
                                    + ((2.0/ref_activity_posI)*np.exp((q_fermi_posI-q_fermi_negI)/V_T)))
        

        

        P = P_H + P_OH + P_posI + P_negI


        if overflow1 == True:
                for i,p in enumerate(V_grid):
                        if abs(p) >= 9.3:
                                P[i] = 0.0 
                                

        #Derivative of trap states
        K2 = np.array([[Kon1*H_c[0],-Koff1,0],[0,Kon2*H_c[0],-Koff2],[1,1,1]])
        C2 = np.array([-Kon1*(P_H[0]/Far)*ct0[0],-(Kon2*(P_H[0]/Far)*ct1[0]),0])
        X2 = scipy.linalg.solve(K2,C2)
        ct0_dV = X2[0]
        ct1_dV = X2[1]
        ct2_dV = X2[2]

        P[0] = P[0] - ((Far*ct0_dV)/(h[0]/2)) + ((Far*ct2_dV)/(h[0]/2))


        
        Jac = cap_mat - sparse.diags(P, 0, shape=(NPSD, NPSD))
        
        y = scipy.sparse.linalg.spsolve(Jac, F_tot)
        
        
        ## Step size cut-off
        for m,j in enumerate(y):
            if j>(V_T):
                y[m] = V_T
            elif j<-(V_T):
                y[m] = -V_T

        

        
        ## Update potential
        V_grid = V_grid - y
        
        ## Calculate norm
        
        delta = scipy.linalg.norm(y)
        #print("Current Residual=",delta)
        
        

    ## Final Positive Charges
    posI_c = stern_fac*c_bar_posI * (np.exp((q_fermi_posI-V_grid)/V_T)/steric_posI)
    H_c = c_bar_H * (np.exp((q_fermi_H-V_grid)/V_T)/steric_H)




    ## Final Negative Charges
    negI_c = stern_fac*c_bar_negI * (np.exp(-(q_fermi_negI-V_grid)/V_T)/steric_negI)
    OH_c = c_bar_OH * (np.exp(-(q_fermi_OH-V_grid)/V_T)/steric_OH)

    return V_grid,posI_c,negI_c, H_c, OH_c, ct0, ct1, ct2
