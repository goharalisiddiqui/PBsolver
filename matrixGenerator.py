#! /usr/bin/python3
import numpy as np
from scipy.sparse import diags




def get_capMat_1D_spherical(space_grid, perm_grid, BC_left, BC_right):
    
    space_grid = np.array(space_grid)
    perm_grid = np.array(perm_grid)

    h = np.append(space_grid[1:] - space_grid[:-1],[0.0])
    h_prev = np.append([0.0],h[:-1])
    h_avg = (h+h_prev)/2.0

    p_par = (np.power(space_grid+(h/2),3) - np.power(space_grid-(h_prev/2),3))/3.0


    ## Main Diagonal
    Di0_1 = np.power(space_grid-(h_prev/2),2)
    Di0_2 = np.power(space_grid+(h/2),2)
    Di0_3 = Di0_1 + Di0_2
    Di0_4 = (Di0_3*perm_grid)/h_avg
    DiM   = np.divide(Di0_4,p_par)

                                                                                                    

    ## Right diagonal
    DiR = -np.divide(((perm_grid/h_avg)*Di0_2),p_par)[:-1]



    ## right diagonal BC
    if BC_left == 'D':
        DiR[0] = 0.0                                                            #Dirichlet for the left boundary
        DiM[0] = 1.0                                                            



    ## Left diagonal
    DiL = -np.divide(((perm_grid/h_avg)*Di0_1),p_par)[1:] 



    ## left diagonal BC
    if BC_right == 'D':
        DiL[-1] = 0.0                                                          #Dirichlet condition for right boundary
        DiM[-1] = 1.0

    cap_mat = diags([DiL,DiM,DiR], [-1,0,1], shape=(space_grid.size, space_grid.size))

    return cap_mat


def get_capMat_1D_cartesian(space_grid, perm_grid, BC_left, BC_right):
    
    space_grid = np.array(space_grid)
    perm_grid = np.array(perm_grid)

    h = np.append(space_grid[1:] - space_grid[:-1],[0.0])
    h_prev = np.append([0.0],h[:-1])
    h_avg = (h+h_prev)/2.0

   


    ## Main Diagonal
    DiM = perm_grid[1:-1]*((1.0/(h_prev[1:-1]*h_avg[1:-1]))+(1.0/(h[1:-1]*h_avg[1:-1])))
    DiM = np.append([perm_grid[0]*(1/(h[0]*h_avg[0]))],np.append(DiM,[perm_grid[-1]*(1/(h_prev[-1]*h_avg[-1]))]))

                                                                                                    

    ## Right diagonal
    DiR = -perm_grid[:-1]*(1.0/(h[:-1]*h_avg[:-1]))



    ## right diagonal BC
    if BC_left == 'D':
        DiR[0] = 0.0                                                            #Dirichlet for the left boundary
        DiM[0] = 1.0                                                            



    ## Left diagonal
    DiL = -perm_grid[1:]*(1.0/(h_prev[1:]*h_avg[1:])) 



    ## left diagonal BC
    if BC_right == 'D':
        DiL[-1] = 0.0                                                          #Dirichlet condition for right boundary
        DiM[-1] = 1.0


    cap_mat = diags([DiL,DiM,DiR], [-1,0,1], shape=(space_grid.size, space_grid.size))

    return cap_mat
