#! /usr/bin/python3

import matplotlib.pyplot as pylab
from datetime import datetime
import os
import sys
import shelve
import numpy as np
import progressbar
import matplotlib.ticker as ticker


def steadyState_1D_silanoDL_plot(shelf_file,title):

    
    startTime = datetime.now()

    ## Some parameters
    Na = 6.022140857e23
    img_dpi = 400

    print("\n\n Plotting Now ...... \n")
    
    

    out_folder = os.path.dirname(os.path.realpath(shelf_file)) + "/Results_"+title


    try:
        os.stat(out_folder)
    except:
        os.mkdir(out_folder) 


    ## Retrieving Shelved Variables

    filename = shelf_file
    
    my_shelf = shelve.open(filename)
    for key in my_shelf:
        ct2_shelve = np.array(my_shelf["ct2_shelve"])
        ct1_shelve = np.array(my_shelf["ct1_shelve"])
        ct0_shelve = np.array(my_shelf["ct0_shelve"])
        srf_pot_shelve = np.array(my_shelf["srf_pot_shelve"])
        V_grid_shelve = np.array(my_shelf["V_grid_shelve"])
        x = np.array(my_shelf["space_grid"])
        rho_pos_ion = np.array(my_shelf["rho_pos_ion_shelve"])
        rho_neg_ion = np.array(my_shelf["rho_neg_ion_shelve"])
        rho_pos_H = np.array(my_shelf["rho_pos_H_shelve"])
        rho_neg_H = np.array(my_shelf["rho_neg_H_shelve"])
        Bulk_ionic_list = my_shelf["Bulk_ionic_list"]
        pH_list = my_shelf["pH_list"]
        C_max_list = my_shelf["C_max_list"]
    my_shelf.close()

    
    

    

    ## Progress Bar
    maxval = int(len(pH_list)*len(C_max_list)*len(Bulk_ionic_list))
    bar = progressbar.ProgressBar(maxval=maxval, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()


    pH_plot = np.unique(pH_list.astype(int))   #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    ICfig = []
    ICfigaxes = []
    for u in range(C_max_list.size):
        fig, ax = pylab.subplots()
        ICfig.append(fig)
        ICfigaxes.append(ax)


    for i in range(Bulk_ionic_list.size):
        
        ndir = out_folder+"/"+"IC_"+str(Bulk_ionic_list[i]*1e-3)+"[M]"
        try:
            os.stat(ndir)
        except:
            os.mkdir(ndir)


        ## Initializing plots
        fig3, f3ax = pylab.subplots(nrows=3,ncols=1)
        fig1, f1ax = pylab.subplots()
        fig4, f4ax = pylab.subplots()
        fig5, f5ax = pylab.subplots()








        for j in range(C_max_list.size):
            
            ###### Plotting Surface States
            f3ax[0].plot(pH_list, ct2_shelve[i][j]*Na*1e-4, label="SD = "+str(C_max_list[j]*1e-4)+" cm^-2")
            f3ax[1].plot(pH_list, ct1_shelve[i][j]*Na*1e-4, label="SD = "+str(C_max_list[j]*1e-4)+" cm^-2")
            f3ax[2].plot(pH_list, ct0_shelve[i][j]*Na*1e-4, label="SD = "+str(C_max_list[j]*1e-4)+" cm^-2")
            

            ###### Plotting Surface pH vs Bulk pH
            
            #delta_srf_pH = []
            #for ind,yy in enumerate(pH_list):
            #    delta_srf_pH.append((-np.log10(rho_pos_H[i][j][ind][0]*1e-3))-yy)
            
            f4ax.plot(pH_list,(-np.log10(rho_pos_H[i,j,:,0]*1e-3)), label="SD = "+str(C_max_list[j]*1e-4)+" cm^-2")
            


            ###### Plotting Surface Potential Change vs pH for different SS
            delta_srf_pot = ((srf_pot_shelve[i][j][1:]-srf_pot_shelve[i][j][:-1])*1e3)/(pH_list[1:]-pH_list[:-1])
            f5ax.plot(pH_list[1:], delta_srf_pot, label="SD = "+str(C_max_list[j]*1e-4)+" cm^-2")
            

            ###### Plotting Surface Potential Change vs pH for different IC
            ICfigaxes[j].plot(pH_list[1:], delta_srf_pot, label="IC = "+str(Bulk_ionic_list[i]*1e-3)+" M")
            
            
            
            ###### Plotting Surface Potential vs pH
            f1ax.plot(pH_list, srf_pot_shelve[i][j]*1e3, label="SD = "+str(C_max_list[j]*1e-4)+" cm^-2")
            
            

            ##### Making directories to store plots
            ndir = out_folder+"/"+"IC_"+str(Bulk_ionic_list[i]*1e-3)+"[M]"+"/"+"SD_"+str(C_max_list[j]*1e-4)+"[cm^-2]"
            try:
                os.stat(ndir)
            except:
                os.mkdir(ndir)
            
            
            
            for l in range(pH_list.size):
                
                if pH_list[l] in pH_plot:

                    ##### Searching for a suitable length to plot the Electrostatics
                    midpointind = int(x.size)
                    for ind, point in enumerate(V_grid_shelve[i][j][l]):
                        if abs(point) < 1e-7 and ind < x.size:
                            midpointind = ind
                            break
                    
                    ## Plots Initialization
                    fig2, f2ax = pylab.subplots(nrows=3, ncols=1)
                    fig2.tight_layout()


                    ###### Plotting Electrostatics
                    f2ax[0].plot(x[:(midpointind)]*1e9, V_grid_shelve[i][j][l][:(midpointind)]*1e3, label="Electrostatic Potential")

                    f2ax[1].semilogy(x[:(midpointind)]*1e9, rho_pos_ion[i][j][l][:(midpointind)]*1e-3, label="pos_ion")
                    f2ax[1].semilogy(x[:(midpointind)]*1e9, rho_neg_ion[i][j][l][:(midpointind)]*1e-3, label="neg_ion")

                    
                    f2ax[2].plot(x[:(midpointind)]*1e9, -np.log10(rho_pos_H[i][j][l][:(midpointind)]*1e-3), label="H+")
                    f2ax[2].plot(x[:(midpointind)]*1e9, -np.log10(rho_neg_H[i][j][l][:(midpointind)]*1e-3), label="OH-")

                    ##### Setting plot attributes for Electrostatics
                    f2ax[0].set_title("PHsensor model Electrostatics plot for pH = "+str(pH_list[l])+", SD = "+str(C_max_list[j]*1e-4)+" cm^-2  and IC = "+str(Bulk_ionic_list[i]*1e-3)+" M ",fontsize='small')
                    f2ax[0].set_ylabel("Potential (mV)",fontsize='small')
                    f2ax[0].legend()
                    
                    f2ax[1].set_ylabel("Concentration (M)",fontsize='small')
                    f2ax[1].legend()
                    
                    f2ax[2].set_xlabel("Distance (nm)")
                    f2ax[2].set_ylabel("pH/pOH",fontsize='small')
                    #f2ax[2].yaxis.set_major_locator(ticker.MultipleLocator(1.0))
                    f2ax[2].legend()
                    

                    ##### Saving and closing Electrostatics plots
                    fig2.savefig(out_folder+"/"+"IC_"+str(Bulk_ionic_list[i]*1e-3)+"[M]/"+"SD_"+str(C_max_list[j]*1e-4)+"[cm^-2]/Electrostatics_pH_"+str(pH_list[l])+".jpg",dpi=img_dpi,bbox_inches="tight",quality=95)
                    pylab.close(fig2)
                
                bar.update((l+1)+(j*len(pH_list))+(i*len(pH_list)*len(C_max_list)))


        ##### Setting plot attributes for the rest of the plots
        f1ax.set_xlabel("pH",fontsize=14)
        f1ax.set_ylabel("Surface Potential  (mV)",fontsize=14)
        f1ax.set_title("Surface Potential for IC = "+str(Bulk_ionic_list[i]*1e-3)+" M ")
        f1ax.set_xticks(pH_plot)
        f1ax.tick_params(labelsize=14)
        f1ax.minorticks_on()
        f1ax.grid(True)
        f1ax.legend()
        
        f3ax[0].set_title("Surface density of SiOH2+ (cm^-2)",fontsize=14)
        f3ax[1].set_title("Surface denisty of SiOH (cm^-2)",fontsize=14)
        f3ax[2].set_title("Surface density of SiO- (cm^-2)",fontsize=14)
        f3ax[2].set_xlabel("pH")
        
        for axx in f3ax:
            axx.set_xticks(pH_plot)
            axx.grid(True)
            axx.tick_params(labelsize=14)
            axx.minorticks_on()
            axx.legend(prop={'size': 8})
        
        
        
        f4ax.plot(pH_list,pH_list, label="Reference line x=y", linestyle='dashed')
        f4ax.set_xlabel("Bulk pH",fontsize=14)
        f4ax.set_ylabel("Surface_pH",fontsize=14)
        f4ax.set_title("Surface pH for IC = "+str(Bulk_ionic_list[i]*1e-3)+" M ")
        f4ax.set_xticks(pH_plot)
        f4ax.set_yticks(pH_plot)
        f4ax.tick_params(labelsize=14)
        f4ax.minorticks_on()
        f4ax.grid(True)
        f4ax.legend()

        f5ax.set_xlabel("pH",fontsize=14)
        f5ax.set_ylabel("dV/d(pH) (mV/pH)",fontsize=14)
        f5ax.set_title("Surface Potential Change for IC = "+str(Bulk_ionic_list[i]*1e-3)+" M ")
        f5ax.set_xticks(pH_plot)
        f5ax.tick_params(labelsize=14)
        f5ax.minorticks_on()
        f5ax.grid(True)
        f5ax.legend()

        
        ##### Saving and closing the remaining plots
        fig1.tight_layout()
        fig1.savefig(out_folder+"/"+"IC_"+str(Bulk_ionic_list[i]*1e-3)+"[M]/"+"Surface_Potential.jpg",dpi=img_dpi,bbox_inches="tight",quality=95)
        pylab.close(fig1)
        fig3.tight_layout()
        fig3.savefig(out_folder+"/"+"IC_"+str(Bulk_ionic_list[i]*1e-3)+"[M]/"+"Surface_States.jpg",dpi=img_dpi,bbox_inches="tight",quality=95)
        pylab.close(fig3)
        fig4.tight_layout()
        fig4.savefig(out_folder+"/"+"IC_"+str(Bulk_ionic_list[i]*1e-3)+"[M]/"+"Surface_pH.jpg",dpi=img_dpi,bbox_inches="tight",quality=95)
        pylab.close(fig4)
        fig5.tight_layout()
        fig5.savefig(out_folder+"/"+"IC_"+str(Bulk_ionic_list[i]*1e-3)+"[M]/"+"Surface_pot_Change.jpg",dpi=img_dpi,bbox_inches="tight",quality=95)
        pylab.close(fig5)

        
        
    for figax in ICfigaxes:
        #### Plot attributes for the sensing change with IC plot
        figax.set_xlabel("pH",fontsize=14)
        figax.set_ylabel("dV/d(pH) (mV/pH)",fontsize=14)
        figax.set_title("Surface Potential Change for different IC")
        figax.set_xticks(pH_plot)
        figax.tick_params(labelsize=14)
        figax.minorticks_on()
        figax.grid(True)
        figax.legend()

    for i,fig in enumerate(ICfig):
        ##### Saving and Closing the last plot
        fig.tight_layout()
        fig.savefig(out_folder+"/"+"SD_"+str(C_max_list[i]*1e-4)+"cm^-2"+"_Surface_pot_Change.jpg",dpi=img_dpi,bbox_inches="tight",quality=95)
        pylab.close(fig)

    bar.finish()
    print("Plotting Time :"+str(datetime.now() - startTime))
