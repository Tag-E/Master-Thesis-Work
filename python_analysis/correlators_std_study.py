##### imports #####
import sys #for command line arguments reading
import numpy as np #to manage data
import matplotlib.pyplot as plt #to plot data
import struct #to read binary file
from pathlib import Path #to handle dir creation
from os import listdir #to list png files to show
from os.path import isfile, join #to list png files to show
import os #to show png files

from astropy.stats import jackknife_resampling
from astropy.stats import jackknife_stats
from scipy.stats import bootstrap

from sympy import divisors


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
confSTOP = 0


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


#### function producing the plot for the std vs binning study ####
def std_study(corrNumb,name,save=True,show=False):

    ## variables ##

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
    plot_dir=plot_base_dir+'plot_stdStudy_'+name.split('.')[0]
    #create plot directory if not already there
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    #runName to be shown in plots
    runName = name.split('.')[0]

    #the correlator chose is the corrNumb-th one
    chosen_corr = all_correlators[:,:,corrNumb,:,:,:,:]


    #average over noise
    corr_navg = chosen_corr.mean(axis=-1).mean(axis=-1)
    #get total correlator summing connected and disconnected part
    corr_navg_tot = corr_navg[:,0,:,:] + corr_navg[:,1,:,:]

    #array with a value of normalized std for each delta
    std_list_jack = []
    std_list_boot = []     
    std_list_simple = []

    #choice of binning: divisors of number of configurations
    deltaList = [delta for delta in divisors(nconf) if delta < nconf/10]

    #study of the std for different delta (= binsize = number of deleted configurations)

    #loop over the possible deltas (size of deleted elements)
    for delta in deltaList:

        #the axis with the configuration is now replaced with an axis with averages of configurations: 
        #   - the lenght of the axis passes from nconf to nconf/delta (that is an int by construction)
        #   - the i-th element will be the average of the configuration in the range (i*delta, (i+1)*delta)

        corr_binned = np.array([np.mean(corr_navg_tot[i*delta:(i+1)*delta],axis=0) for i in range(int(nconf/delta))])

        #now we compute mean and std using the jackknife method (implementd in the astropy library)

        #observable we're interested in
        test_statistic = np.mean

        #array where we will store the mean and std
        mean_array_jack = np.empty(shape=(noperators,tvals),dtype=float)
        std_array_jack = np.empty(shape=(noperators,tvals),dtype=float)

        mean_array_boot = np.empty(shape=(noperators,tvals),dtype=float)
        std_array_boot = np.empty(shape=(noperators,tvals),dtype=float)

        #loop in which we use the jackknife and the bootstrap (over the configurations for fixed operator and time)

        for iop in range(noperators): #for each operator
            for t in range(tvals): #and for each time

                #jackknife with astropy

                #we choose as array the one spanning over the configurations (binned)
                data = corr_binned[:,iop,t].imag
            
                #we compute mean and std using the jackknife
                estimate, _, stderr, _ = jackknife_stats(data, test_statistic, 0.95)
            
                mean_array_jack[iop,t] = estimate
                std_array_jack[iop,t] = stderr

                #bootstrap with scipy

                #we choose as array the one spanning over the configurations (binned)
                data = (corr_binned[:,iop,t].imag,)

                #we compute mean and sd using the bootstrap
                res = bootstrap(data, test_statistic,  n_resamples=9999, confidence_level=0.9)
                        
                mean_array_boot[iop,t] = res.bootstrap_distribution.mean()
            
                std_array_boot[iop,t] = res.standard_error


        
        #we now append to the std list the mean relative error
        std_list_jack.append(np.mean( (std_array_jack/np.abs(mean_array_jack))[:,1:-2] ))
        std_list_boot.append(np.mean( (std_array_boot/np.abs(mean_array_boot))[:,1:-2] ))

        #computation of mean and std without using the jackknife or the bootstrap (simple mean estimation)
        mean_array_simple = np.mean(corr_binned,axis=0) 
        std_array_simple = np.std(corr_binned,axis=0) /np.sqrt(np.shape(corr_binned)[0]-1)

        #we now append to the std list the mean relative error
        std_list_simple.append(np.mean( (std_array_simple/np.abs(mean_array_simple))[:,1:-2] ))


    ### make plot ###

    #create figure and axis
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

     #adjust subplot spacing
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.87, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.6)


    ax.plot(deltaList,std_list_jack,'-o',linewidth=0.1,color='blue',label='Jackknife Estimate')
    ax.plot(deltaList,std_list_boot,'-o',linewidth=0.1,color='green',label='Bootstrap Estimate')
    ax.plot(deltaList,std_list_simple,'-o',linewidth=0.1,color='red',label='Simple Estimate')
    ax.set_xticks(deltaList)
    ax.tick_params(axis='both', which='major', labelsize=8)

    ax.set_xlabel(r"$\Delta$",fontsize=12)
    ax.set_ylabel(r"$\sigma$ / $|\mu|$",fontsize=12)

    ax.legend()

    #set title
    ax.set_title(r"Normalized Standard Deviation as a function of the binsize $\Delta$" + f' - Correlator {corrNumb}', fontsize=15,y=1.03)
    

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
    plt.text(1.01, 0.95, textstr, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', bbox=props)
    
    
    
    
    #save figure
    if save:
        fig_name = f"stdVSbin_corr{corrNumb}_{runName}.png"
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
    global confSTEP, confSTOP


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

    #decide how many configuration to skip at the end
    if "-stop" in args:
        confSTOP = int(args[args.index("-stop")+1])


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

        x_corr = np.empty(shape=(ncorr,tvals,nnoise),dtype=complex) #2 point with source in x0
        z_corr = np.empty(shape=(ncorr,tvals,nnoise),dtype=complex) #2 point with source in z0


        #chunk size of the data due to a 3 point func and to a 2 point func
        offset_diag3 = nnoise    *   nnoise   *  tvals                  * noperators        * 2         * 8
        offset_diag2 = nnoise * tvals                   * 2         * 8
        

        #configuration start right after the header
        first_conf = header_size+corr_header_size*ncorr
    
        ########## we compute the lenght of the data block for each configuration:
        #          nnoise_A  *  nnoise_B  *  time lenght of lattice * noperators   * ncorrelators * 2 (diagrams=connected+disconnected)  * 2 (re+im) * 8 (sizeof(double))  + sizeof(int) (= conf_number)
        conf_len = nnoise    *   nnoise   *  tvals                  * noperators   * ncorr        * 2                                    * 2         * 8                    + 4
        #
        #                     ncorr * nnoise * tvaks * 2 (x+z 2point corr) * 2 (re+im) * 8 (sizeof(double))
        conf_len = conf_len + ncorr * nnoise * tvals * 2                   * 2         * 8
        ##########


        #starting right after the header we read each configuration block
        for start_conf in range(first_conf, len(fileContent)-conf_len*confSTOP, conf_len*confSTEP):

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

            #update reading pointer after the reading of the conf number
            start_reading = start_conf+4

            #loop over the correlators (ncorr blocks of...)
            for ic in range(ncorr):

                #reading of 3 point correlators (connected and disconnected)
            
                #loop over the operators (...noperators blocks of...)
                for op in range(noperators):
                    #loop over time values (...tvals blocks of...)
                    for t in range(tvals):
                        #loop over noiseB vectors (...nnoise blocks of...)
                        for noiseB in range(nnoise):
                            #loop over noiseA vectors (...nnoise blocks of...)
                            for noiseA in range(nnoise):
                        
                                #reading of re and im part of correlators   (...complex numbers)

                                #reading of real and complex for connected corr
                                re,im = struct.unpack("dd",fileContent[start_reading:start_reading+16])

                                #store complex number
                                conn_corr[ic,op,t,noiseB,noiseA] = complex(re,im)                                

                                #reading of real and complex for disconnected corr
                                re,im = struct.unpack("dd",fileContent[start_reading+offset_diag3:start_reading+offset_diag3+16])

                                #store complex number
                                disc_corr[ic][op][t][noiseB][noiseA] = complex(re,im)

                                #update start reading (only for the part due to the connected correlator)
                                start_reading = start_reading+16


                #update start reading for the disconnected part
                start_reading = start_reading + offset_diag3 #offset_diag3 is the chunk size of the disconnected part (generically of a 3point correlator)


                #reading of 2 point correlators

                #loop over time value (...tvals blocks of...)
                for t in range(tvals):
                    #loop over noise vectores (...nnoise blocks of...)
                    for inoise in range(nnoise):

                        #reading of re and im part of correlators   (...complex numbers)

                        #read real and imag part for 2point with source in x0
                        re,im = struct.unpack("dd",fileContent[start_reading:start_reading+16])
                        #store them
                        x_corr[ic,t,inoise] = complex(re,im)

                        #read real and imag part for 2point with source in z0
                        re,im = struct.unpack("dd",fileContent[start_reading+offset_diag2:start_reading+offset_diag2+16])
                        #store them
                        z_corr[ic,t,inoise] = complex(re,im)
                        
                        #update start reading (only for part due to x_corr)
                        start_reading = start_reading+16

                #update start reading for the 2point with source in z0
                start_reading = start_reading + offset_diag2 #offset _diag2 is the chunk size for the 2point correlator


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
        std_study(corrNumb,name,save=True,show=False)

    #if show is given open every png inside the dir
    if show==True:
        #name of dir for plots
        plot_base_dir = "plots/"
        plot_dir=plot_base_dir+'plot_stdStudy_'+name.split('.')[0]
        png_list = [f for f in listdir(plot_dir) if f.endswith('png') and isfile(join(plot_dir, f) )]
        for png in png_list[:1]:
            os.system("xdg-open "+plot_dir+'/'+png)
            




#if the file is called as main then the main function is executed
if __name__ == "__main__":
    main()