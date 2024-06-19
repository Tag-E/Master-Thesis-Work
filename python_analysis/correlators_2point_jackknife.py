##### imports #####
import sys #for command line arguments reading
import numpy as np #to manage data
import matplotlib.pyplot as plt #to plot data
import struct #to read binary file
from pathlib import Path #to handle dir creation
from os import listdir #to list png files to show
from os.path import isfile, join #to list png files to show
import os #to show png files



###### global variables #####
#conversion dictionaries
noise_dict={0:"Z2",1:"Gauss",2:"U1",3:"One Component"}
dirac_dict={0:"G0",1:"G1",2:"G2",3:"G3",5:"G5",6:"ONE",7:"G0G1",8:"G0G2",9:"G0G3",10:"G0G5",11:"G1G2",12:"G1G3",13:"G1G5",14:"G2G3",15:"G2G5",16:"G3G5"}

latex_dirac_dict={0:r'$\gamma_0$',1:r'$\gamma_1$',2:r'$\gamma_2$',3:r'$\gamma_3$',5:r'$\gamma_5$',6:"1",7:r'$\gamma_0\gamma_1$',8:r'$\gamma_0\gamma_2$',
                  9:r'$\gamma_0\gamma_3$',10:r'$\gamma_0\gamma_5$',11:r'$\gamma_1\gamma_2$',12:r'$\gamma_1\gamma_3$',13:r'$\gamma_1\gamma_5$',
                  14:r'$\gamma_2\gamma_3$',15:r'$\gamma_2\gamma_5$',16:r'$\gamma_3\gamma_5$'}

#the correlators array will be stored in a dict having as keys the configurations
conf_dict = {}
conf_dict_2p = {}

#list with configurations number
conf_num_list = []

#number of operators
noperators = 5

#filename
fileName = ''
runName = ''

#initialization of correlators' variables
k1=['']
k2=['']
k3=['']
k4=['']
mu1=['']
mu2=['']
mu3=['']
mu4=['']
typeA=['']
typeB=['']
x0=['']
z0=['']

ncorr=0
nnoise=0
tvals=0
noise_type=0
check_gauge_inv=0
random_conf=0
csw=0.0
cf=0.0

#global variables used for the jackknife
conf_names = []
conf_list = []
nconf = 0
all_correlators = np.array((),dtype=complex)

#variable used to skip some configurations
confSTEP = 1


##### function printing info to terminal #####
def print_info():
    #Header print
    print("\n[File Header]\n")
    print(f"- ncorr           = {ncorr}\n")
    print(f"- nnoise          = {nnoise}\n")
    print(f"- tvals           = {tvals}\n")
    print(f"- noise_type      = {noise_dict[noise_type]}\n")
    print(f"- check_gauge_inv = {check_gauge_inv}\n")
    print(f"- random_conf     = {random_conf}\n")
    print(f"- csw             = {csw}\n")
    print(f"- cF              = {cf}\n\n")

    #configurations details print
    print("\n[Configurations]\n")
    print(f"- nconf               = {nconf}\n")
    print(f"- conf_step           = {confSTEP}\n")

    #Correlators Header print
    for i in range(ncorr):
        print(f"[Correlator {i}]\n")
        print(f" - k1 = {k1[i]}\n")
        print(f" - k2 = {k2[i]}\n")
        print(f" - k3 = {k3[i]}\n")
        print(f" - k4 = {k4[i]}\n\n")
        print(f" - mu1 = {mu1[i]}\n")
        print(f" - mu2 = {mu2[i]}\n")
        print(f" - mu3 = {mu3[i]}\n")
        print(f" - mu4 = {mu4[i]}\n\n")
        print(f" - typeA = {dirac_dict[typeA[i]]}\n")
        print(f" - typeB = {dirac_dict[typeB[i]]}\n\n")
        print(f" - x0 = {x0[i]}\n")
        print(f" - z0 = {z0[i]}\n\n\n")


##### function to plot correlators #####
def plotCorr(confNumb,corrNumb,name,save=True,show=False):

    #global variables
    global conf_dict,conf_name,fileName,runName
    global k1,k2,k3,k4,mu1,mu2,mu3,mu4,typeA,typeB,x0,z0
    global ncorr, nnoise, tvals, noise_type, check_gauge_inv, csw, cf
    
    #names of 5 operators
    op_names = ["VA","AV","SP","PS",r'T $\mathbf{\~{T}}$']

    #name of dir for plots
    plot_base_dir = "plots/"
    plot_dir=plot_base_dir+'plot_'+name.split('.')[0]
    Path(plot_dir).mkdir(parents=True, exist_ok=True)


    #times on x axis
    times = np.arange(0,tvals)

    #create figure and axis
    fig, ax_list = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=False, figsize=(32, 14))



    #loop over plot, one for each of the 5 operators
    for iop,op_name in enumerate(op_names):

        #compute connected, disconnected and total operatpr
        conn_corr = conf_dict[confNumb][0][corrNumb][iop] #conf - piece - corr - op
        disc_corr = conf_dict[confNumb][1][corrNumb][iop] 
        tot_corr = conn_corr+disc_corr

        #array for connected,disconnected,total correaltors
        corr_list = [conn_corr,disc_corr,tot_corr]
        corr_lab = ["Connected","Disconnected","Total"]
        corr_colors = ["red","blue","purple"]

    

        #take average over noise
        avg_list = [corr.mean(axis=-1).mean(axis=-1) for corr in corr_list]
        #take modulus
        #Gt_list = [np.abs(avg) for avg in avg_list]
        Gt_list_re = [avg.real for avg in avg_list]
        Gt_list_im = [avg.imag for avg in avg_list]

        #plot 3 pieces
        #for ipiece,Gt in enumerate(Gt_list):
            #ax_list[iop].plot(times,Gt,'-o',label=corr_lab[ipiece],color=corr_colors[ipiece],markersize=10,linewidth=0.5)
        for ipiece,Gt in enumerate(Gt_list_re):
            #ax_list[iop].plot(times,Gt,'-*',label=corr_lab[ipiece]+" re",color=corr_colors[ipiece],markersize=10,linewidth=0.5)
            ax_list[iop].plot(times,Gt_list_im[ipiece],'-o',label=corr_lab[ipiece]+" im",color=corr_colors[ipiece],markersize=10,linewidth=0.5)

        #enable grid
        ax_list[iop].grid()

        #set title
        ax_list[iop].set_title(op_name,fontsize=15,weight="bold")

        #set y label
        ax_list[iop].set_ylabel("G(t)",rotation=0,labelpad=20,fontsize=16)

        #set legend
        ax_list[iop].legend(loc='right')

    
    #set x ticks to be all time values
    #plt.xticks(times)

    #adjust subplot spacing
    plt.subplots_adjust(left=0.04,
                        bottom=0.05, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.6)

    #set x label
    #fig.supylabel("G(t)",rotation=0,fontsize=20)
    plt.xlabel('Time [lattice units]',fontsize=16)

    #set title
    plt.suptitle(f'G(t) for parity odd operators - (Configuration {confNumb}, Correlator {corrNumb})', fontsize=25,y=0.98,)

    #Display text box with frelevant parameters outside the plot
    textstr = '\n'.join((
         'Correlator %d parameters:' % (corrNumb),
         '           ',
        r'$k_1$=%.9f ' % (k1[corrNumb] ),
        r'$k_2$=%.9f ' % (k2[corrNumb] ),
        r'$k_3$=%.9f ' % (k3[corrNumb] ),
        r'$k_4$=%.9f ' % (k4[corrNumb] ),
         '           ',
        r'$\mu_1$=%.9f ' % (mu1[corrNumb] ),
        r'$\mu_2$=%.9f ' % (mu2[corrNumb] ),
        r'$\mu_3$=%.9f ' % (mu3[corrNumb] ),
        r'$\mu_4$=%.9f ' % (mu4[corrNumb] ),
         '           ',
        r'$\Gamma_A$=' + latex_dirac_dict[typeA[corrNumb]],
        r'$\Gamma_B$=' + latex_dirac_dict[typeB[corrNumb]],
         '           ',
        r'$x_0$=%d' % x0[corrNumb],
        r'$z_0$=%d' % z0[corrNumb],
         '           ',
         '           ',
         '           ',
         '           ',
         'Simulation parameters:',
         '           ',
        r'$N_{NOISE}$=%d' % nnoise,
         'Noise Type=%s' % noise_dict[noise_type],
         'Random Conf=%d' % random_conf,
        r'$T$=%d' % tvals,
         r'$c_{SW}$=%.9f ' % csw,
         r'$c_F$=%.9f' % cf))


    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place the text box in upper left in axes coords
    plt.text(1.01, 0.95, textstr, transform=ax_list[0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    #save figure
    if save:
        fig_name = f"plot_conf{confNumb}_corr{corrNumb}_{runName}.png"
        plt.savefig(plot_dir+"/"+fig_name)

    #show figure
    if show:
        plt.show()


#### function to plot correlators using the jackknife method ####
def jackknife_plots(corrNumb,name,save=True,show=False):

    #global variables
    global conf_dict,conf_name,fileName,runName
    global k1,k2,k3,k4,mu1,mu2,mu3,mu4,typeA,typeB,x0,z0
    global ncorr, nnoise, tvals, noise_type, check_gauge_inv, csw, cf
    #for jackknife
    global conf_names, conf_list, nconf, all_correlators
    #for 2 points
    global conf_dict_2p, all_2point_x, all_2point_z

    #names of 5 operators
    op_names = ["VA","AV","SP","PS",r'T $\mathbf{\~{T}}$']

    #name of dir for plots
    plot_base_dir = "plots/"
    plot_dir=plot_base_dir+'plot_jack2_'+name.split('.')[0]
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    #runName to be shown in plots
    runName = name.split('.')[0]


    ### average over noise ###
    all_correlators_navg = all_correlators.mean(axis=-1).mean(axis=-1)
    all_correlators_navg_tot = all_correlators_navg[:,0,:,:,:] + all_correlators_navg[:,1,:,:,:] #used to plot all configurations

    ### creation of jack replicates ###
    jack_replicates = np.asarray( [np.delete(all_correlators_navg,iconf,axis=0).mean(axis=0) for iconf in range(nconf)] )
    jack_replicates_tot = jack_replicates[:,0,:,:,:] + jack_replicates[:,1,:,:,:] #sum of conneced and disconnected diagram

    #jacknife estimator of mean
    jack_mean = jack_replicates.mean(axis=0)
    jack_mean_tot = jack_replicates_tot.mean(axis=0)

    #jacknife estimator of std, looking at re and im part
    jack_std_imag = np.sqrt(nconf-1) * np.std(jack_replicates.imag,axis=0)
    jack_std_re = np.sqrt(nconf-1) * np.std(jack_replicates.real,axis=0)



    #### jackknife with 2 point functions ####
    
    #noise avg
    all_2point_x_navg = all_2point_x.mean(axis=-1)
    all_2point_z_navg = all_2point_z.mean(axis=-1)

    #creation of jack replicates
    delta = 1
    jack_replicates_x = np.asarray( [np.delete(all_2point_x_navg, list(range(iconf,min(iconf+delta,nconf))) ,axis=0).mean(axis=0) for iconf in range(0,nconf,delta)] )
    jack_replicates_z = np.asarray( [np.delete(all_2point_z_navg, list(range(iconf,min(iconf+delta,nconf))) ,axis=0).mean(axis=0) for iconf in range(0,nconf,delta)] )
    jack_replicates_3 = np.asarray( [np.delete(all_correlators_navg_tot, list(range(iconf,min(iconf+delta,nconf))) ,axis=0).mean(axis=0) for iconf in range(0,nconf,delta)] )

    normalized_replicates = np.empty(shape=np.shape(jack_replicates_3),dtype=complex)

    for irep in range(np.shape(jack_replicates_3)[0]):
        for ic in range(ncorr):
            for iop in range(noperators):
                normalized_replicates[irep,ic,iop,:] = jack_replicates_3[irep,ic,iop,:]/(jack_replicates_x[irep,ic,:]*jack_replicates_z[irep,ic,:])


    norm_mean = normalized_replicates.mean(axis=0)

    norm_std_imag = np.sqrt(np.shape(normalized_replicates)[0]-1) * np.std(normalized_replicates.imag,axis=0)





    #times on x axis
    times = np.arange(0,tvals)

    ##############  plot number 1: comparison of connected and disconnected piece (by looking at imag part) #####################ààà

    #create figure and axis
    fig, ax_list = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=False, figsize=(32, 14))

    #loop over plot, one for each of the 5 operators
    for iop,op_name in enumerate(op_names):

        #compute connected, disconnected and total correlator
        conn_corr = jack_mean[0,corrNumb,iop,:].imag #piece - corr - op - t
        disc_corr = jack_mean[1,corrNumb,iop,:].imag 
        tot_corr = conn_corr+disc_corr

        #compute variance
        conn_std = jack_std_imag[0,corrNumb,iop,:]
        disc_std = jack_std_imag[1,corrNumb,iop,:]
        tot_std = np.sqrt(nconf-1) * np.std(jack_replicates_tot.imag,axis=0)[corrNumb,iop,:]

        #array for connected,disconnected,total correaltors
        corr_list = [conn_corr,disc_corr,tot_corr]
        std_list = [conn_std,disc_std,tot_std]
        corr_lab = ["Connected","Disconnected","Total"]
        corr_colors = ["red","blue","purple"]

        #plot 3 pieces
        for ipiece,corr_piece in enumerate(corr_list):
            ax_list[iop].errorbar(times,corr_piece,yerr=std_list[ipiece],marker='o',linestyle='solid',label=corr_lab[ipiece],color=corr_colors[ipiece],markersize=10,linewidth=0.5,elinewidth=2)

        #enable grid
        ax_list[iop].grid()

        #set title
        ax_list[iop].set_title(op_name,fontsize=15,weight="bold")

        #set y label
        ax_list[iop].set_ylabel("G(t)",rotation=0,labelpad=20,fontsize=16)

        #set legend
        ax_list[iop].legend(loc='right')

    
    #set x ticks to be all time values
    #plt.xticks(times)

    #adjust subplot spacing
    plt.subplots_adjust(left=0.04,
                        bottom=0.05, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.6)

    #set x label
    #fig.supylabel("G(t)",rotation=0,fontsize=20)
    plt.xlabel('Time [lattice units]',fontsize=16)

    #set title
    plt.suptitle(f'Im[G(t)] for parity odd operators - Correlator {corrNumb} - Connected, Disconnected and Total using Jackknife Method', fontsize=25,y=0.98,)

    #Display text box with frelevant parameters outside the plot
    textstr = '\n'.join((
         'Correlator %d parameters:' % (corrNumb),
         '           ',
        r'$k_1$=%.9f ' % (k1[corrNumb] ),
        r'$k_2$=%.9f ' % (k2[corrNumb] ),
        r'$k_3$=%.9f ' % (k3[corrNumb] ),
        r'$k_4$=%.9f ' % (k4[corrNumb] ),
         '           ',
        r'$\mu_1$=%.9f ' % (mu1[corrNumb] ),
        r'$\mu_2$=%.9f ' % (mu2[corrNumb] ),
        r'$\mu_3$=%.9f ' % (mu3[corrNumb] ),
        r'$\mu_4$=%.9f ' % (mu4[corrNumb] ),
         '           ',
        r'$\Gamma_A$=' + latex_dirac_dict[typeA[corrNumb]],
        r'$\Gamma_B$=' + latex_dirac_dict[typeB[corrNumb]],
         '           ',
        r'$x_0$=%d' % x0[corrNumb],
        r'$z_0$=%d' % z0[corrNumb],
         '           ',
         '           ',
         '           ',
         '           ',
         'Simulation parameters:',
         '           ',
        r'$N_{NOISE}$=%d' % nnoise,
         'Noise Type=%s' % noise_dict[noise_type],
         'Random Conf=%d' % random_conf,
        r'$T$=%d' % tvals,
        r'$c_{SW}$=%.9f ' % csw,
        r'$c_F$=%.9f' % cf,
         '           ',
         '           ',
         '           ',
         'Configurations:',
         '           ',
        r'$N_{CONF}$=%d' % nconf,))


    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place the text box in upper left in axes coords
    plt.text(1.01, 0.95, textstr, transform=ax_list[0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    #save figure
    if save:
        fig_name = f"plot_conn-disc_corr{corrNumb}_{runName}.png"
        plt.savefig(plot_dir+"/"+fig_name)

    #show figure
    if show:
        plt.show()

    

    ##############  plot number 2: comparison of real and imaginary part (by looking at total correlator) #####################ààà

    #create figure and axis
    fig, ax_list = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=False, figsize=(32, 14))

    #loop over plot, one for each of the 5 operators
    for iop,op_name in enumerate(op_names):

        #real imag and modulus of total correlator
        re_corr = jack_mean_tot[corrNumb,iop,:].real #corr - op - t
        imag_corr = jack_mean_tot[corrNumb,iop,:].imag 
        mod_corr = np.abs(jack_mean_tot[corrNumb,iop,:])

        #compute variance
        re_std = np.sqrt(nconf-1) * np.std(jack_replicates_tot.real,axis=0)[corrNumb,iop,:]
        im_std = np.sqrt(nconf-1) * np.std(jack_replicates_tot.imag,axis=0)[corrNumb,iop,:]
        mod_std = np.sqrt(nconf-1) * np.std(np.abs(jack_replicates_tot),axis=0)[corrNumb,iop,:]

        #array for connected,disconnected,total correaltors
        corr_list = [re_corr,imag_corr,mod_corr]
        std_list = [re_std,im_std,mod_std]
        corr_lab = ["Real","Imaginary","Modulus"]
        corr_colors = ["red","blue","purple"]

        #plot 3 pieces
        for ipiece,corr_piece in enumerate(corr_list):
            ax_list[iop].errorbar(times,corr_piece,yerr=std_list[ipiece],marker='o',label=corr_lab[ipiece],color=corr_colors[ipiece],markersize=10,linewidth=0.5,elinewidth=2)

        #enable grid
        ax_list[iop].grid()

        #set title
        ax_list[iop].set_title(op_name,fontsize=15,weight="bold")

        #set y label
        ax_list[iop].set_ylabel("G(t)",rotation=0,labelpad=20,fontsize=16)

        #set legend
        ax_list[iop].legend(loc='right')

    
    #set x ticks to be all time values
    #plt.xticks(times)

    #adjust subplot spacing
    plt.subplots_adjust(left=0.04,
                        bottom=0.05, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.6)

    #set x label
    #fig.supylabel("G(t)",rotation=0,fontsize=20)
    plt.xlabel('Time [lattice units]',fontsize=16)

    #set title
    plt.suptitle(f'Total G(t) for parity odd operators - Correlator {corrNumb} - Real, Imaginary and Modulus using Jackknife Method', fontsize=25,y=0.98,)

    #Display text box with frelevant parameters outside the plot
    
    #textstr is defined above

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place the text box in upper left in axes coords
    plt.text(1.01, 0.95, textstr, transform=ax_list[0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    #save figure
    if save:
        fig_name = f"plot_re-im_corr{corrNumb}_{runName}.png"
        plt.savefig(plot_dir+"/"+fig_name)

    #show figure
    if show:
        plt.show()


    
    ##############  plot number 3: comparison of jacknife replicates #####################ààà

    #create figure and axis
    fig, ax_list = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=False, figsize=(32, 14))

    #loop over plot, one for each of the 5 operators
    for iop,op_name in enumerate(op_names):

        for iconf in range(nconf):
            lbl = None
            if iconf == nconf-1:
                lbl = "Jackknife Replicates"
            ax_list[iop].plot(times,jack_replicates_tot[iconf,corrNumb,iop,:].imag,'-o',markersize=7,linewidth=0.5,alpha=0.4,color="red",label=lbl)

        #mean and std with jackknife method
        mean_corr = jack_mean_tot[corrNumb,iop,:].imag 
        std_corr = np.sqrt(nconf-1) * np.std(jack_replicates_tot.imag,axis=0)[corrNumb,iop,:]


        ax_list[iop].errorbar(times,mean_corr,yerr=std_corr,marker='o',label=r"Jackknife Mean $\pm$ std",color="black",markersize=10,linewidth=0.8,elinewidth=2)

        #enable grid
        ax_list[iop].grid()

        #set title
        ax_list[iop].set_title(op_name,fontsize=15,weight="bold")

        #set y label
        ax_list[iop].set_ylabel("G(t)",rotation=0,labelpad=20,fontsize=16)

        #set legend
        ax_list[iop].legend(loc='right')

    
    #set x ticks to be all time values
    #plt.xticks(times)

    #adjust subplot spacing
    plt.subplots_adjust(left=0.04,
                        bottom=0.05, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.6)

    #set x label
    #fig.supylabel("G(t)",rotation=0,fontsize=20)
    plt.xlabel('Time [lattice units]',fontsize=16)

    #set title
    plt.suptitle(f'Total Im[G(t)] for parity odd operators - Correlator {corrNumb} - Jackknife Replicates and Jackknife Mean', fontsize=25,y=0.98,)

    #Display text box with frelevant parameters outside the plot
    
    #textstr is defined above

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place the text box in upper left in axes coords
    plt.text(1.01, 0.95, textstr, transform=ax_list[0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    #save figure
    if save:
        fig_name = f"plot_jack_corr{corrNumb}_{runName}.png"
        plt.savefig(plot_dir+"/"+fig_name)

    #show figure
    if show:
        plt.show()



    ##############  plot number 4: comparison of configuration with jackknife mean #####################ààà

    #create figure and axis
    fig, ax_list = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=False, figsize=(32, 14))

    #loop over plot, one for each of the 5 operators
    for iop,op_name in enumerate(op_names):

        for iconf in range(nconf):
            lbl = None
            if iconf == nconf-1:
                lbl = "Configurations (no Jackknife)"
            ax_list[iop].plot(times,all_correlators_navg_tot[iconf,corrNumb,iop,:].imag,'-o',markersize=7,linewidth=0.5,alpha=0.4,color="red",label=lbl)

        #mean and std with jackknife method
        mean_corr = jack_mean_tot[corrNumb,iop,:].imag 
        std_corr = np.sqrt(nconf-1) * np.std(jack_replicates_tot.imag,axis=0)[corrNumb,iop,:]


        ax_list[iop].errorbar(times,mean_corr,yerr=std_corr,marker='o',label=r"Jackknife Mean $\pm$ std",color="black",markersize=10,linewidth=0.8,elinewidth=2)

        #enable grid
        ax_list[iop].grid()

        #set title
        ax_list[iop].set_title(op_name,fontsize=15,weight="bold")

        #set y label
        ax_list[iop].set_ylabel("G(t)",rotation=0,labelpad=20,fontsize=16)

        #set legend
        ax_list[iop].legend(loc='right')

    
    #set x ticks to be all time values
    #plt.xticks(times)

    #adjust subplot spacing
    plt.subplots_adjust(left=0.04,
                        bottom=0.05, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.6)

    #set x label
    #fig.supylabel("G(t)",rotation=0,fontsize=20)
    plt.xlabel('Time [lattice units]',fontsize=16)

    #set title
    plt.suptitle(f'Total Im[G(t)] for parity odd operators - Correlator {corrNumb} - All Configurations and Jackknife Mean', fontsize=25,y=0.98,)

    #Display text box with frelevant parameters outside the plot

    #textstr is defined above    

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place the text box in upper left in axes coords
    plt.text(1.01, 0.95, textstr, transform=ax_list[0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    #save figure
    if save:
        fig_name = f"plot_jack_corr{corrNumb}_allconf_{runName}.png"
        plt.savefig(plot_dir+"/"+fig_name)

    #show figure
    if show:
        plt.show()





    ##############  plot number 5: correlators normalized to 2 points func #####################

    #create figure and axis
    fig, ax_list = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=False, figsize=(32, 14))

    #loop over plot, one for each of the 5 operators
    for iop,op_name in enumerate(op_names):

        ax_list[iop].errorbar(times,norm_mean[corrNumb,iop,:].imag,yerr=norm_std_imag[corrNumb,iop,:],marker='o',label=r"Jackknife Mean $\pm$ std",color="black",markersize=10,linewidth=0.8,elinewidth=2)

        #enable grid
        ax_list[iop].grid()

        #set title
        ax_list[iop].set_title(op_name,fontsize=15,weight="bold")

        #set y label
        ax_list[iop].set_ylabel("G(t)",rotation=0,labelpad=20,fontsize=16)

        #set legend
        ax_list[iop].legend(loc='right')

        #ax_list[iop].set_yscale('log')

    
    #set x ticks to be all time values
    #plt.xticks(times)

    #adjust subplot spacing
    plt.subplots_adjust(left=0.04,
                        bottom=0.05, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.6)

    #set x label
    #fig.supylabel("G(t)",rotation=0,fontsize=20)
    plt.xlabel('Time [lattice units]',fontsize=16)

    #set title
    plt.suptitle(f'Total Im[G(t)] for parity odd operators - Correlator {corrNumb} - Normalized to 2 point functions', fontsize=25,y=0.98,)

    #Display text box with frelevant parameters outside the plot #texstr defined before
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place the text box in upper left in axes coords
    plt.text(1.01, 0.95, textstr, transform=ax_list[0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    #save figure
    if save:
        fig_name = f"plot_normjack_corr{corrNumb}_{runName}.png"
        plt.savefig(plot_dir+"/"+fig_name)

    #show figure
    if show:
        plt.show()

    

    ##############  plot number 6: 2 points func #####################

    #create figure and axis
    fig, ax_list = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=(32, 14))


    xmean = jack_replicates_x.mean(axis=0)
    xstd_re = np.std(jack_replicates_x.real,axis=0)

    zmean = jack_replicates_z.mean(axis=0)
    zstd_re = np.std(jack_replicates_z.real,axis=0)


    ax_list[0].errorbar(times,xmean[corrNumb,:].real,yerr=xstd_re[corrNumb,:],marker='o',label=r"Jackknife Mean $\pm$ std",color="black",markersize=10,linewidth=0.8,elinewidth=2)
    #enable grid
    ax_list[0].grid()
    #set title
    ax_list[0].set_title(r"With source in $x_0$",fontsize=15,weight="bold")
    #set y label
    ax_list[0].set_ylabel("G(t)",rotation=0,labelpad=20,fontsize=16)
    #set legend
    ax_list[0].legend(loc='right')

    ax_list[1].errorbar(times,zmean[corrNumb,:].real,yerr=zstd_re[corrNumb,:],marker='o',label=r"Jackknife Mean $\pm$ std",color="black",markersize=10,linewidth=0.8,elinewidth=2)
    #enable grid
    ax_list[1].grid()
    #set title
    ax_list[1].set_title(r"With source in $z_0$",fontsize=15,weight="bold")
    #set y label
    ax_list[1].set_ylabel("G(t)",rotation=0,labelpad=20,fontsize=16)
    #set legend
    ax_list[1].legend(loc='right')

    
    #set x ticks to be all time values
    #plt.xticks(times)

    #adjust subplot spacing
    plt.subplots_adjust(left=0.04,
                        bottom=0.05, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.6)

    #set x label
    #fig.supylabel("G(t)",rotation=0,fontsize=20)
    plt.xlabel('Time [lattice units]',fontsize=16)

    #set title
    plt.suptitle(f'Re[G(t)] for 2 point correlator - Correlator {corrNumb}', fontsize=25,y=0.98,)

    #Display text box with frelevant parameters outside the plot #texstr defined before
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place the text box in upper left in axes coords
    plt.text(1.01, 0.95, textstr, transform=ax_list[0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    #save figure
    if save:
        fig_name = f"plot_2point_corr{corrNumb}_{runName}.png"
        plt.savefig(plot_dir+"/"+fig_name)

    #show figure
    if show:
        plt.show()




##### main function #####
def main():

    args = sys.argv[1:] #read command line argument

    #check on correct number of arguments
    if len(args)<1:
        print("usage: python3 correlators_plot.py path/to/filename.dat")
        exit()

    #global variables
    global conf_dict,conf_name,fileName,runName
    global k1,k2,k3,k4,mu1,mu2,mu3,mu4,typeA,typeB,x0,z0
    global ncorr, nnoise, tvals, noise_type, check_gauge_inv, random_conf, csw, cf
    #for jackknife
    global conf_names, conf_list, nconf, all_correlators
    #for 2 point
    global conf_dict_2p, all_2point_x, all_2point_z
    global confSTEP


    #read log file name from command line
    fileName = args[0]

    #set verbose option
    verbose=False
    if "-verbose" in args:
        verbose=True

    #set show plot option
    show=False
    if "-show" in args:
        show=True

    #decide the step at which configurations are read
    if "-step" in args:
        confSTEP = int(args[args.index("-step")+1])


    ##### reading data from binary dat file #####
    with open(fileName, mode='rb') as file: # b is important -> binary
        fileContent = file.read()

        #header is made up of 5 integers, 5x4=20byte, and 2 double, 2x8=16byte
        header_size= 6*4 +2*8

        #first 16 byte are four 4-byte integers
        ncorr, nnoise, tvals, noise_type, check_gauge_inv, random_conf, csw, cf = struct.unpack("iiiiiidd", fileContent[:header_size])

        #initialization of correlators' variables
        k1=['']*ncorr
        k2=['']*ncorr
        k3=['']*ncorr
        k4=['']*ncorr
        mu1=['']*ncorr
        mu2=['']*ncorr
        mu3=['']*ncorr
        mu4=['']*ncorr
        typeA=['']*ncorr
        typeB=['']*ncorr
        x0=['']*ncorr
        z0=['']*ncorr

    
        #each correlator has an header of size given by 8x8 + 4x4 = 80byte
        corr_header_size = 8*8 + 4*4
    
        #then there are ncorr block, 8x8 + 4x4 (8 double and 4 int) with the following structure 
        for i in range(ncorr):
            k1[i], k2[i], k3[i], k4[i], mu1[i], mu2[i], mu3[i], mu4[i], typeA[i], typeB[i], x0[i], z0[i] = struct.unpack("ddddddddiiii",fileContent[header_size+corr_header_size*i:header_size+corr_header_size*(i+1)])



        #initialization of correlators array
        conn_corr = np.empty(shape=(ncorr,noperators,tvals,nnoise,nnoise),dtype=complex) #connected correlators
        disc_corr = np.empty(shape=(ncorr,noperators,tvals,nnoise,nnoise),dtype=complex) #disconnected correlators

        x_corr = np.empty(shape=(ncorr,tvals,nnoise),dtype=complex)
        z_corr = np.empty(shape=(ncorr,tvals,nnoise),dtype=complex)



        #offset due to 3 points (see comment below)
        offset_3point = nnoise    *   nnoise   *  tvals                  * noperators        * 2                                    * 2         * 8 
        offset_2point = nnoise * tvals * 2                   * 2         * 8

        

        #configuration start right after the header
        first_conf = header_size+corr_header_size*ncorr
    
        #we compute the lenght of the data block for each configuration
        #          nnoise_A  *  nnoise_B  *  time lenght of lattice * noperators   * ncorrelators * 2 (diagrams=connected+disconnected)  * 2 (re+im) * 8 (sizeof(double))  + sizeof(int) (= conf_number)
        conf_len = nnoise    *   nnoise   *  tvals                  * noperators   * ncorr        * 2                                    * 2         * 8                    + 4
        #                     ncorr * nnoise * tvaks * 2 (x+z 2point corr) * 2 (re+im) * 8 (sizeof(double))
        conf_len = conf_len + ncorr * nnoise * tvals * 2                   * 2         * 8

        #starting right after the header we read each configuration block
        for start_conf in range(first_conf, len(fileContent), conf_len*confSTEP):

            #breaks loop if the file does not contain the whole correlator for the given configuration
            if start_conf+conf_len > len(fileContent) :
                break

            #reading of the configuration number
            conf_number = struct.unpack("i",fileContent[start_conf:start_conf+4])[0]
            if verbose==True:
                print(f"Reading: Gauge Conf = {conf_number}\n")

            #store of conf num
            if conf_number not in conf_num_list:
                conf_num_list.append(conf_number)

            #initialize to 0 the array storing the correlators
            conn_corr.fill(complex(0,0))
            disc_corr.fill(complex(0,0))
            #same for 2 point
            x_corr.fill(complex(0,0))
            z_corr.fill(complex(0,0))
            

            #loop over the correlators (ncorr blocks of...)
            for ic in range(ncorr):

                #reading of chunk with conn corr
            
                #loop over the operators (...noperators blocs of...)
                for op in range(noperators):
                    #loop over time values (...tvals blocks of...)
                    for t in range(tvals):
                        #loop over noiseB vectors (...nnoise blocks of...)
                        for noiseB in range(nnoise):
                            #loop over noiseA vectors (...nnoise blocks of...)
                            for noiseA in range(nnoise):
                        
                                #reading of re and im part of correlators   (...complex numbers)
                            

                                #connected correlators

                                #begin of reading point (8=sizeof(double)) (last 2 is number of diagrams)
                                start_reading = start_conf+4 + 2*8*noiseA+2*8*nnoise*(noiseB+nnoise*(t+tvals*(op+noperators*2*ic)))

                                #reading of real and complex for connected corr
                                re,im = struct.unpack("dd",fileContent[start_reading:start_reading+16])

                                #store complex number
                                conn_corr[ic][op][t][noiseB][noiseA] = complex(re,im)

                                #disconnected correlators

                                #begin of reading point (8=sizeof(double)) (last 2 is number of diagrams)
                                start_reading = start_conf+4 + 2*8*noiseA+2*8*nnoise*(noiseB+nnoise*(t+tvals*(op+noperators*(2*ic+1))))

                                #reading of real and complex for connected corr
                                re,im = struct.unpack("dd",fileContent[start_reading:start_reading+16])

                                #store complex number
                                disc_corr[ic][op][t][noiseB][noiseA] = complex(re,im)

                                ############# new reading #########

                                #start_reading  = start_conf + 4 + 2 * 2*8 * (noiseA + nnoise*(noiseB + nnoise*(t + tvals*(op + noperators*ic ) ) ) )

                                #re_con, im_con, re_disc, im_disc = struct.unpack("dddd",fileContent[start_reading:start_reading+2*2*8])
                                #conn_corr[ic,op,t,noiseB,noiseA] = complex(re_con,im_con)
                                #disc_corr[ic,op,t,noiseB,noiseA] = complex(re_disc,im_disc)

                #reading of 2 point functions
                for t in range(tvals):
                    for inoise in range(nnoise):
                        start_reading = start_conf + 4 + offset_3point + 2*8*(inoise+nnoise*(t+tvals*2*ic)) #last 2 is for 2point in x and z
                        re,im = struct.unpack("dd",fileContent[start_reading:start_reading+16])
                        x_corr[ic][t][inoise] = complex(re,im)
                        start_reading = start_conf + 4 + offset_3point + 2*8*(inoise+nnoise*(t+tvals*(2*ic+1))) 
                        re,im = struct.unpack("dd",fileContent[start_reading:start_reading+16])
                        z_corr[ic][t][inoise] = complex(re,im)


            #store of correlators associated to the given configuration
            if str(conf_number) not in conf_dict.keys():
                conf_dict[str(conf_number)] = (conn_corr.copy(),disc_corr.copy())
                conf_dict_2p[str(conf_number)] = (x_corr.copy(),z_corr.copy())
            else:
                conf_dict[str(conf_number)+"_GaugeInvCheck"] = (conn_corr.copy(),disc_corr.copy())
                conf_dict_2p[str(conf_number)+"_GaugeInvCheck"] = (x_corr.copy(),z_corr.copy())


    #construction of array with names and numbers of configurations
    conf_names = list(conf_dict.keys())[0::1+check_gauge_inv]
    conf_list = [int(conf) for conf in conf_names ]
    nconf = len(conf_list)

    #creation of a numpy array with all the correlators
    all_correlators = np.empty(shape=(nconf,2,ncorr,noperators,tvals,nnoise,nnoise),dtype=complex) #2 is for connected and disconnected
    for iconf,nameconf in enumerate(conf_names):
        all_correlators[iconf] = conf_dict[nameconf]

    #analogous for 2 points func
    all_2point_x = np.empty(shape=(nconf,ncorr,tvals,nnoise),dtype=complex)
    all_2point_z = np.empty(shape=(nconf,ncorr,tvals,nnoise),dtype=complex)
    for iconf,nameconf in enumerate(conf_names):
        all_2point_x[iconf] = conf_dict_2p[nameconf][0]
        all_2point_z[iconf] = conf_dict_2p[nameconf][1]


    #print info if requested from command line
    if verbose==True:
        print_info()



    
    #list with all configuration names
    #conf_list = list(conf_dict.keys())

    name = fileName.split('/')[-1][:-4]
    #print(name)

    #plot everything
    for corrNumb in range(ncorr):
        jackknife_plots(corrNumb,name,save=True,show=False)

    #if show is given open every png inside the dir
    if show==True:
        #name of dir for plots
        plot_base_dir = "plots/"
        plot_dir=plot_base_dir+'plot_jack2_'+name.split('.')[0]
        png_list = [f for f in listdir(plot_dir) if f.endswith('png') and isfile(join(plot_dir, f) )]
        for png in png_list[:1]:
            os.system("xdg-open "+plot_dir+'/'+png)
            




#if the file is called as main then the main function is executed
if __name__ == "__main__":
    main()
