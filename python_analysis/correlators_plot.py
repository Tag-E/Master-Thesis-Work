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


##### function printing info to terminal #####
def print_info():
    #Header print
    print("\n[File Header]\n")
    print(f"- ncorr           = {ncorr}\n")
    print(f"- nnoise          = {nnoise}\n")
    print(f"- tvals           = {tvals}\n")
    print(f"- noise_type      = {noise_dict[noise_type]}\n")
    print(f"- check_gauge_inv = {check_gauge_inv}\n")
    print(f"- random_conf     = {random_conf}\n\n")

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
    global ncorr, nnoise, tvals, noise_type, check_gauge_inv
    
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
    plt.suptitle(f'|G(t)| for parity odd operators - (Configuration {confNumb}, Correlator {corrNumb})', fontsize=25,y=0.98,)

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
        r'$T$=%d' % tvals))


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
    global ncorr, nnoise, tvals, noise_type, check_gauge_inv, random_conf


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


    ##### reading data from binary dat file #####
    with open(fileName, mode='rb') as file: # b is important -> binary
        fileContent = file.read()

        #header is made up of 6 integers, 5x4=20byte
        header_size= 6*4

        #first 16 byte are four 4-byte integers
        ncorr, nnoise, tvals, noise_type, check_gauge_inv, random_conf = struct.unpack("iiiiii", fileContent[:header_size])

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

        

        #configuration start right after the header
        first_conf = header_size+corr_header_size*ncorr
    
        #we compute the lenght of the data block for each configuration
        #          nnoise_A  *  nnoise_B  *  time lenght of lattice * noperators   * ncorrelators * 2 (diagrams=connected+disconnected)  * 2 (re+im) * 8 (sizeof(double))  + sizeof(int) (= conf_number)
        conf_len = nnoise    *   nnoise   *  tvals                  * noperators   * ncorr        * 2                                    * 2         * 8                    + 4

        #starting right after the header we read each configuration block
        for start_conf in range(first_conf, len(fileContent), conf_len):

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

            #store of correlators associated to the given configuration
            if str(conf_number) not in conf_dict.keys():
                conf_dict[str(conf_number)] = (conn_corr,disc_corr)
            else:
                conf_dict[str(conf_number)+"_GaugeInvCheck"] = (conn_corr,disc_corr)


    if verbose==True:
        print_info()

    
    #list with all configuration names
    conf_list = list(conf_dict.keys())

    name = fileName.split('/')[-1][:-4]
    #print(name)

    #plot everything
    for confNumb in conf_list:
        for corrNumb in range(ncorr):
            plotCorr(confNumb,corrNumb,name,save=True,show=False)

    #if show is given open every png inside the dir
    if show==True:
        #name of dir for plots
        plot_base_dir = "plots/"
        plot_dir=plot_base_dir+'plot_'+name.split('.')[0]
        png_list = [f for f in listdir(plot_dir) if f.endswith('png') and isfile(join(plot_dir, f) )]
        for png in png_list[:1]:
            os.system("xdg-open "+plot_dir+'/'+png)
            




#if the file is called as main then the main function is executed
if __name__ == "__main__":
    main()
