import matrixGenerator as mg
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as pylab
from scipy.sparse.linalg import spsolve
import scipy.linalg 
import shelve
import math
from datetime import datetime
import sys
import os
import progressbar




## Physical Constants
kb = 1.38064852e-23
el_c = 1.60217662e-19
perm_0 = 8.854178176e-12
Na = 6.022140857e23
Far = Na*el_c



def steadyState_1D(NPSD ,L ,tol ,T ,rel_perm ,pos_ions ,neg_ions ,NeuBC, grid = 'S'):


    ## Physical Parameters
    perm = rel_perm*perm_0
    V_T = (kb*T)/el_c


    ##Space Grid
    x = np.linspace(0.0, L, NPSD)
    h = x[1] - x[0]

    ##Permitivity Grid
    perm_grid = np.zeros(NPSD)
    perm_grid.fill(perm)

    
    
    ##Potential grid
    V_grid = np.zeros(NPSD)

    

    delta = 1.0
    
    while delta > tol:
        
        ## Ionic Charges
        rho_pos_boltz = np.zeros(NPSD)
        rho_neg_boltz = np.zeros(NPSD)
        rho_pos_const = np.zeros(NPSD)
        rho_neg_const = np.zeros(NPSD)
        
        #Positive ion charge
        for index,data in enumerate(pos_ions):
            for i,p in enumerate(x):
                if p>(data[1]*L) and p<=(data[2]*L):
                    if data[3] == True:
                        rho_pos_boltz[i] = rho_pos_boltz[i] + (data[0]*math.exp(-V_grid[i]/V_T))
                    elif data[3] == False:
                        rho_pos_const[i] = rho_pos_const[i] + data[0]




        ##Negative ion charge
        for index,data in enumerate(neg_ions):
            for i,p in enumerate(x):
                if p>(data[1]*L) and p<=(data[2]*L):
                    if data[3] == True:
                        rho_neg_boltz[i] = rho_neg_boltz[i] + (data[0]*math.exp(V_grid[i]/V_T))
                    elif data[3] == False:
                        rho_neg_const[i] = rho_neg_const[i] + data[0]

        ##Total charge
        rho_total = Far*(rho_pos_boltz + rho_pos_const - rho_neg_boltz - rho_neg_const)
        
        ##Boundary conditions for charge density
        rho_total[0] = NeuBC                                                      #Neumann condition for left boundary    
        rho_total[-1] = 0.0                                                     #Dirichlet Condition for right boundary
        
        ## Matrix setup
        if grid == 'C':
            cap_mat = mg.get_capMat_1D_cartesian(x, perm_grid,"N","D")
        else:
            cap_mat = mg.get_capMat_1D_spherical(x, perm_grid,"N","D")


        #Total Function
        F_tot = cap_mat.dot(V_grid) - rho_total

        #Jacobian 
        P = Far*(((-1/V_T)*(rho_pos_boltz)) - ((1/V_T)*(rho_neg_boltz)))
        Jac = cap_mat - (sparse.dia_matrix((P, [0]), shape=(NPSD, NPSD)))
        
        ##solve y
        y = spsolve(Jac, F_tot)


        ## Dampen large errors
        for m,j in enumerate(y):
            if j>V_T:
                y[m] = V_T
            elif j<-V_T:
                y[m] = -V_T
        
        ## Update potential
        V_grid = V_grid - y
        
        ## Calculate norm
        delta = np.linalg.norm(y)
    
        
    #Electric field
    E = -((V_grid[1:] - V_grid[:-1])/h)

    
    ##plot
    x = x/L
    pylab.subplot(3, 1, 1)
    pylab.title("Double Layer Model")
    pylab.ylabel("Potential (V)")
    pylab.xlim(x[0],x[-1])
    pylab.plot(x, V_grid, label="V")
    pylab.legend()
    pylab.subplot(3, 1, 2)
    pylab.ylabel("Electric Field (V/m)")
    pylab.xlim(x[0],x[-1])
    pylab.plot(x[:-1], E, label="E")
    pylab.legend()
    pylab.subplot(3, 1, 3)
    pylab.xlabel("Distance (fraction of system Length)")
    pylab.ylabel("Density of Species (moles/m^3)")
    pylab.xlim(x[0],x[-1])
    for i,p in enumerate(rho_neg_boltz):
        if p > 0.0:
            spind = i
            break
    pylab.plot(x[spind:], rho_pos_boltz[spind:], label="C+")
    pylab.plot(x[spind:], rho_neg_boltz[spind:], label="C-")
    pylab.legend()
    pylab.show()
    
    
    #return V_grid,E,rho_pos_boltz,rho_neg_boltz


def steadyState_1D_silanoDL(NPSD, L, tol, T,rel_perm, ion_conc, pH, surf_states, K1, K2, shelf_loc="./", \
                                    title = "Untitled", steric=True, eff_sizes=[5.0,5.0,0.3,0.3], force_new_calc=False, plot=True, stern_size = 0.0):

    import phPlotter as php

    # shelf_loc is a directory
    if shelf_loc[-1] != '/':
        shelf_loc = shelf_loc+"/"


    # Check if calculations with same name already exist
    filename = shelf_loc+"shelve_"+title+".out"
    if force_new_calc == False and os.path.isfile(filename):
        print("Shelf file Already exist. Not calculating again")
        if plot==True:
            php.steadyState_1D_silanoDL_plot(filename,title)
        exit()

    
    # Timing the execution
    startTime = datetime.now()
    try:
        os.stat(shelf_loc)
    except:
        os.mkdir(shelf_loc)
      


                                                                
    Bulk_ionic_list = np.array(ion_conc)                                             # Bulk Ionic concentration [moles/m^3]
    pH_list = np.array(pH)                                                           # pH range to calculate
    Kon1 = K1*(1e-3)                                                                 # Kon for trap state SiOH [m^6/moles^2.sec]
    Kon2 = K2*(1e-3)                                                                 # Kon for trap state SiOH2+ [m^6/moles^2.sec]
    Koff1 = 1.0                                                                      # Koff for trap state SiOH [m^3/moles.sec]
    Koff2 = 1.0                                                                      # Koff for trap state SiOH2+ [m^3/moles.sec]
    C_max_list = np.array(surf_states)                                               # Density of Surface States [moles/m^2]


    ## Derived Parameters
    perm = rel_perm*perm_0
    Bulk_H_list = 1e3*np.power(10.0,-pH_list)

    ### Initializations

    ## Log Space Grid
    space_grid = np.logspace(math.log(1e-10), math.log(L), num=NPSD,base=math.e)

    ## Linear Space Grid
    #space_grid = np.linspace(0.0,L,NPSD)
  
   

    ##Permitivity Grid
    perm_grid = np.zeros(NPSD)
    perm_grid.fill(perm)


    ## Shelving Variables Initialization used for storage
    tup1 = (Bulk_ionic_list.size,C_max_list.size,pH_list.size,space_grid.size)
    tup2 = (Bulk_ionic_list.size,C_max_list.size,pH_list.size)

    rho_pos_ion_shelve = np.zeros(tup1)
    rho_pos_H_shelve = np.zeros(tup1) 
    rho_neg_ion_shelve = np.zeros(tup1) 
    rho_neg_H_shelve = np.zeros(tup1) 
    V_grid_shelve = np.zeros(tup1) 

    srf_pot_shelve = np.zeros(tup2) 
    ct0_shelve = np.zeros(tup2) 
    ct1_shelve = np.zeros(tup2) 
    ct2_shelve = np.zeros(tup2)


    for k,Bulk_ionic in enumerate(Bulk_ionic_list):
        
        for a,C_max in enumerate(C_max_list):
            
            for i,Bulk_H in enumerate(Bulk_H_list):  
                
                if steric==True:
                    import phSolverSteric as phss
                    V_grid,rho_pos_ion,rho_neg_ion, rho_pos_H, rho_neg_H, rho_ct0, rho_ct1, rho_ct2 = phss.phSolver(space_grid, perm_grid, NPSD, tol, T, Kon1, Kon2, Koff1, Koff2, Bulk_ionic, Bulk_H, C_max, eff_sizes, stern_size*1e-10)
                else:
                    import phSolver as phs
                    V_grid,rho_pos_ion,rho_neg_ion, rho_pos_H, rho_neg_H, rho_ct0, rho_ct1, rho_ct2 = phs.phSolver(space_grid, perm_grid, NPSD, tol, T, Kon1, Kon2, Koff1, Koff2, Bulk_ionic, Bulk_H, C_max)
                
                ## Saving Results in Shelving Variables
                rho_pos_ion_shelve[k][a][i] = rho_pos_ion
                rho_pos_H_shelve[k][a][i] = rho_pos_H
                rho_neg_ion_shelve[k][a][i] = rho_neg_ion
                rho_neg_H_shelve[k][a][i] = rho_neg_H
                V_grid_shelve[k][a][i] = V_grid
                ct0_shelve[k][a][i] = rho_ct0[0]
                ct1_shelve[k][a][i] = rho_ct1[0]
                ct2_shelve[k][a][i] = rho_ct2[0]
                srf_pot_shelve[k][a][i] = V_grid[0]


                ## Status Printing
                print("Done with calculation for pH="+str(pH_list[i])+" SD =  "+str(C_max_list[a])+" 1/m^2"+" IC =  "+str(Bulk_ionic_list[k])+" moles/m^3. "+str((pH_list.size*C_max_list.size*Bulk_ionic_list.size)-(i+1)-(a*pH_list.size)-(k*pH_list.size*C_max_list.size))+" calculations remaining")
                print("Surface Potential = "+str(V_grid[0]*1e3)+" mV \n")





    ## Saving Date File
    try:
        my_shelf = shelve.open(filename,'n')
    except:
        os.remove(filename)
        my_shelf = shelve.open(filename,'n')

    for key in ['ct0_shelve','ct1_shelve','ct2_shelve','srf_pot_shelve','rho_neg_H_shelve','rho_pos_H_shelve','rho_neg_ion_shelve','rho_pos_ion_shelve','V_grid_shelve','pH_list','C_max_list','Bulk_ionic_list','space_grid']:
        try:
            my_shelf[key] = locals()[key]
        except TypeError:
            print('ERROR shelving: {0}'.format(key))
    my_shelf.close()
    

    ## Plotting Results
    if plot==True:
        php.steadyState_1D_silanoDL_plot(filename,title)

    print("Total Execution Time :"+str(datetime.now() - startTime))

    return space_grid, V_grid_shelve,  rho_pos_ion_shelve, rho_pos_H_shelve, rho_neg_ion_shelve, rho_neg_H_shelve, ct0_shelve, ct1_shelve, ct2_shelve
                



def timeInt_1D_sensor(Length,Duration,Kon,Koff,Ntotal,A_total, c_bulk, D_const = 1e-9,Ng = 10,points_per_m = 20000,points_per_sec = 30,Kp = 1.0):

    import biosensorSolver as bss


    NPSD = int(points_per_m*Length)                         # Spacial Resolution (Number of Discrete Points in Space)
    NPTD = int(points_per_sec*Duration)                     # Time Resolution (Number of Discrete Points in Time)




    Space_grid, Time_grid, Sol, Conc_grid = bss.biosensorSolver(NPSD,Length,NPTD,Duration,Kon,Koff,Kp,Ntotal,A_total, c_bulk, D_const)


    ## Plotting
    pylab.subplot(2,1,1)
    pylab.title("Biosensor Model")
    pylab.ylabel("Concentration (mole/m^3)")
    pylab.xlabel("Distance (m)")
    for i in range(0,NPTD-1,int(NPTD/Ng)):
        tsec = i*(Duration/NPTD)
        pylab.plot(Space_grid,Conc_grid[i],label = "t="+str(tsec)+" sec")
    pylab.legend()
    pylab.subplot(2,1,2)
    pylab.ylabel("Surface Density of Occupied Sensor States (m^-2)")
    pylab.xlabel("Time (sec)")
    pylab.plot(Time_grid, Sol,label = "C_T")
    pylab.legend()
    pylab.show()


