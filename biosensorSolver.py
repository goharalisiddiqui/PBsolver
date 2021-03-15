import numpy as np
import pylab
import matrixGenerator as mg
import scipy.sparse as sparse
from datetime import datetime
import progressbar

Na = 6.022e23                   # Avagadro's Number







def biosensorSolver(NPSD,Length,NPTD,Duration,Kon,Koff,Kp,Ntotal,A_total, c_bulk, D_const):
    

    maxval = NPTD-1
    bar = progressbar.ProgressBar(maxval=maxval, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    startTime = datetime.now()
    ##Space Grid
    Space_grid = np.linspace(0.0, Length, NPSD)
    hx = Space_grid[1] - Space_grid[0]

    ##Time Grid
    Time_grid = np.linspace(0.0, Duration, NPTD)
    ht = Time_grid[1] - Time_grid[0]

    ##Trap Grid
    Trap_grid = np.zeros(NPTD)

    ##Concentration Grid
    Conc_grid = np.zeros((NPTD,NPSD))
    Conc_grid[0].fill(c_bulk)

    ##Diffusion coefficient grid
    D_grid = np.zeros((NPTD,NPSD))
    #D_grid[0] = k_u/np.power(Conc_grid[0],2)
    D_grid[0].fill(D_const) 

    ##Nfree/Ntotal grid
    freef = np.zeros(NPTD)
    freef[0] = 1.0

    ## Time Integration
    for t in range(NPTD-1):

        ## Matrix setup
        DiM = -((2.0*D_grid[t])/(hx*hx))*ht                                         #main diagonal
        DiL = ((D_grid[t]/(hx*hx))*ht)#[1:]                                          #left diagonal
        DiR = ((D_grid[t]/(hx*hx))*ht)#[:-1]                                         #right diagonal

        DiM[-1] = DiM[-1]/2.0
        
        data = np.array([DiL, DiM, DiR])
        offsets = np.array([-1,0,1])
        cap_mat = sparse.dia_matrix((data, offsets), shape=(NPSD, NPSD))
        #cap_mat = mg.get_capMat_1D_cartesian(Space_grid,-D_grid[t]*ht,'N','N')


        G = Kon*Conc_grid[t][0]*freef[t]                                            # Adsoption rate
        R = Koff*(1-freef[t])#Trap_grid[t]                                           # Desoption rate
        Conc_grid[t+1][0] = ((2.0*D_grid[t][0]*ht*(Conc_grid[t][1]-Conc_grid[t][0]))/(hx*hx)) \
                            +((2.0*(R-G)*ht)/(Na*A_total*hx))+Conc_grid[t][0]

        y = cap_mat.dot(Conc_grid[t])

        Conc_grid[t+1][1:] = y[1:] + Conc_grid[t][1:]
        Trap_grid[t+1] = ((G-R)*(ht/A_total)*Kp) + Trap_grid[t]
        freef[t+1] = max(1-(Trap_grid[t+1]*(A_total/(Ntotal*Kp))),0.0)
        #D_grid[0] = k_u/np.power(Conc_grid[0],2)
        D_grid[t+1].fill(D_const)
        bar.update(t)
    bar.finish()
    print("Total Execution Time :"+str(datetime.now() - startTime))
    return Space_grid, Time_grid, Trap_grid, Conc_grid 