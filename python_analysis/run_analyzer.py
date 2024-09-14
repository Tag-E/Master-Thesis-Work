######################################################
## run_analyzer_tmdf24fop.py                        ##
## created by Emilio Taggi - 2024/06/27             ##
######################################################

#########################################################################
# This program is free software: you can redistribute it and/or modify  #
# it under the terms of the GNU General Public License as published by  #
# the Free Software Foundation, either version 3 of the License, or     #
# (at your option) any later version.                                   #
#                                                                       #
# This program is distributed in the hope that it will be useful,       #
# but WITHOUT ANY WARRANTY; without even the implied warranty of        #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         #
# GNU General Public License for more details.                          #
#                                                                       #
# You should have received a copy of the GNU General Public License     #
# along with this program.  If not, see <http://www.gnu.org/licenses/>. #
#########################################################################

########################## Program Usage ################################
#
# ... externaly accesible classes are ...
# ... etc etc ...
# ...
#
#########################################################################



######################## Library Imports ################################

import struct #utility to read binary file
import numpy as np #for data handling
from tqdm import tqdm #for the progress bar in loops
from pathlib import Path #to handle directory creation
from sympy import divisors #to get a list of divisor of an int
from astropy.stats import jackknife_stats #for the data analysis using the jackknife method
from scipy.stats import bootstrap #for the data analysis using the bootstrap method
import matplotlib.pyplot as plt #for the plots
from os import listdir #to list png files to show
from os.path import isfile, join #to list png files to show
import os #to show png files
from math import log10, floor, sqrt #per arrotondare i risultati
from scipy.optimize import curve_fit #to fit data to functions




########################## Main Class ###################################

#classes with all the data analysis tools for the run (related to the file odd_df2_4fop)
class run:

    '''
    Create once class instance for a given run and analyze the
    results using the bult-in methods (for the run of the type odd_df2_4fop)
    - accesible methods are: ...
    - accesible variables are: ...
    '''

    #global variables shared by all the runs (class instances)
    
    #number of operators (VA,AV,SP,PS,TT^~)
    noperators = 5

    #names of 5 operators
    op_names = ["VA","AV","SP","PS",r'T $\mathbf{\~{T}}$']
    op_names_simple = ["VA","AV","SP","PS",'TTtilda']
    #names of 5 operators in the right basis
    op_names_rot = ["VA+AV","VA-AV","-SP+PS","SP+PS",r'T $\mathbf{\~{T}}$']
    op_names_rot_simple = ["VA+AV","VA-AV","-SP+PS","SP+PS",'TTtilda']

    #rotation matrix to go into the right basis (VA,AV,SP,PS,T~T) -> (VA+AV,VA-AV,-SP+PS,SP+PS,T~T)
    rot_mat = np.array([[1,  1,  0, 0, 0],
                        [1, -1,  0, 0, 0],
                        [0,  0, -1, 1, 0],
                        [0,  0,  1, 1, 0],
                        [0,  0,  0, 0, 1]],dtype=float)

    #conversion dictionaries
    noise_dict={0:"Z2",1:"Gauss",2:"U1",3:"One Component"}
    dirac_dict={0:"G0",1:"G1",2:"G2",3:"G3",5:"G5",6:"ONE",7:"G0G1",8:"G0G2",9:"G0G3",10:"G0G5",11:"G1G2",12:"G1G3",13:"G1G5",14:"G2G3",15:"G2G5",16:"G3G5"}

    latex_dirac_dict={0:r'$\gamma_0$',1:r'$\gamma_1$',2:r'$\gamma_2$',3:r'$\gamma_3$',5:r'$\gamma_5$',6:"1",7:r'$\gamma_0\gamma_1$',8:r'$\gamma_0\gamma_2$',
                      9:r'$\gamma_0\gamma_3$',10:r'$\gamma_0\gamma_5$',11:r'$\gamma_1\gamma_2$',12:r'$\gamma_1\gamma_3$',13:r'$\gamma_1\gamma_5$',
                      14:r'$\gamma_2\gamma_3$',15:r'$\gamma_2\gamma_5$',16:r'$\gamma_3\gamma_5$'}
    
    #name of dir for plots
    plot_base_dir = "plots/"

    #specifics of the data file

    #header is made up of 5 integers, 5x4=20byte, and 2 double, 2x8=16byte
    header_size= 6*4 +2*8

    #each correlator has an header of size given by 8x8 + 4x4 = 80byte
    corr_header_size = 8*8 + 4*4





    #standard constructor that requires the path to the run
    def __init__(self,filePath,verbose=True):

        #name of dir for plots
        name = filePath.split('/')[-1][:-4]
        self.run_name = name.split('.')[0]
        self.plot_dir=self.plot_base_dir+self.run_name

        #create plot directory if not already there
        Path(self.plot_dir).mkdir(parents=True, exist_ok=True)

        if verbose:
            print("\nPlots and relevant info concerning the run will be stored in "+self.plot_dir)
        
        #initialization of relevant struct storing data
        self.conf_dict_3p = {} #dict with the 3 points correlators (the keys are the configuration numbers)
        self.conf_dict_2p = {} #dict with the 2 points correlators (the keys are the configuration numbers)


        #to inizialize the instance of the class we read from the file passed from input

        ##### reading data from binary dat file #####
        with open(filePath, mode='rb') as file: # b is important -> binary

            #get all file content
            fileContent = file.read()


            ## read the header ##

            if verbose:
                print("\nReading the Header...\n")

            #first general information

            #reading: 6 int (4 byte) and 2 double (8 byte)
            self.ncorr, self.nnoise, self.tvals, self.noise_type, self.check_gauge_inv, self.random_conf, self.csw, self.cf = struct.unpack("iiiiiidd", fileContent[:self.header_size])

            #then the information regarding each correlator

            #initialization of correlators' variables
            self.k1=['']*self.ncorr
            self.k2=['']*self.ncorr
            self.k3=['']*self.ncorr
            self.k4=['']*self.ncorr
            self.mu1=['']*self.ncorr
            self.mu2=['']*self.ncorr
            self.mu3=['']*self.ncorr
            self.mu4=['']*self.ncorr
            self.typeA=['']*self.ncorr
            self.typeB=['']*self.ncorr
            self.x0=['']*self.ncorr
            self.z0=['']*self.ncorr

            #reading: there are ncorr block, 8x8 + 4x4 (8 double and 4 int) with the following structure 
            for i in range(self.ncorr):
                self.k1[i], self.k2[i], self.k3[i], self.k4[i], self.mu1[i], self.mu2[i], self.mu3[i], self.mu4[i], self.typeA[i], self.typeB[i], self.x0[i], self.z0[i] = struct.unpack("ddddddddiiii",fileContent[self.header_size+self.corr_header_size*i:self.header_size+self.corr_header_size*(i+1)])


            ## read the content of the file ##

            #first the initialization of the relevant arrays with the correlators (temporary arrays used to read the correlators for a given configuration)
            conn_corr = np.empty(shape=(self.ncorr,self.noperators,self.tvals,self.nnoise,self.nnoise),dtype=complex) #connected 3point correlators
            disc_corr = np.empty(shape=(self.ncorr,self.noperators,self.tvals,self.nnoise,self.nnoise),dtype=complex) #disconnected 3point correlators
            x_corr = np.empty(shape=(self.ncorr,self.tvals,self.nnoise),dtype=complex)                                #2point correlators with source in x0 
            z_corr = np.empty(shape=(self.ncorr,self.tvals,self.nnoise),dtype=complex)                                #2point correlators with source in z0

            #data chunk size for a given correlator (diag3 for a 3 points diagram (so connected or disconnected), diag2 for the 2point correlator)
            offset_diag3 = self.nnoise * self.nnoise * self.tvals * self.noperators * 2 * 8 #2 is for real+imaginary part
            offset_diag2 = self.nnoise * self.tvals * 2 * 8                                 #8 is for sizeof(double)  

            #pointer to begin of first configuration (right after the header)
            first_conf = self.header_size+self.corr_header_size*self.ncorr

            #compute the size of the data chunk for a given configuration
            #          4(=sizeof(int)=conf id) + ( connected 3point+ disconnected 3point  + 2point in x0 + 2point in z0    ) * number of correlators
            conf_len = 4                       + (           offset_diag3*2               +        offset_diag2*2          ) * self.ncorr

            #we now start with the actual reading

            if verbose:
                print("Reading the data for each configuration...\n")

            #starting right after the header we read each configuration block
            for start_conf in tqdm(range(first_conf, len(fileContent), conf_len)):

                #security check for run that were killed before finishing:
                #breaks loop if the file does not contain the whole correlator for the given configuration
                if start_conf+conf_len > len(fileContent) :
                    break

                #initialize to 0 the arrays with the correlators
                conn_corr.fill(complex(0,0))
                disc_corr.fill(complex(0,0))
                x_corr.fill(complex(0,0))
                z_corr.fill(complex(0,0))

                #reading of the configuration number
                conf_number = struct.unpack("i",fileContent[start_conf:start_conf+4])[0]

                #we set the pointer to the data we have to read right after the configuration number
                start_reading = start_conf+4

                #then we start to read the data associated with the given configuration

                #loop over the correlators                         (for each conf we have ncorr blocks of...) 
                for ic in range(self.ncorr):

                    #reading of 3point correlator                   (concerning the 3point correlator)

                    #loop over the operators                           (...noperators blocks of...)
                    for op in range(self.noperators):
                        #loop over time values                             (...tvals blocks of...)
                        for t in range(self.tvals):
                            #loop over noiseB vectors                          (...nnoise blocks of...)
                            for noiseB in range(self.nnoise):
                                #loop over noiseA vectors                          (...nnoise blocks of...)
                                for noiseA in range(self.nnoise):

                                    #reading of re and im part of 3point correlators   (...complex numbers)

                                    #first we read the connected diagram of the 3 point correlators

                                    #reading of real and imaginary part for connected diagram
                                    re,im = struct.unpack("dd",fileContent[start_reading:start_reading+16])

                                    #we store the complex number
                                    conn_corr[ic,op,t,noiseB,noiseA] = complex(re,im)

                                    #then we read the disconnected diagram (that is offset_diag3 bytes after the connected diagram)

                                    #reading of real and imaginary part for disconnected diagram
                                    re,im = struct.unpack("dd",fileContent[start_reading+offset_diag3:start_reading+offset_diag3+16])

                                    #store complex number
                                    disc_corr[ic,op,t,noiseB,noiseA] = complex(re,im)

                                    #we update start reading (for the part that concerns the connected diagram (we are reading connected and disconnected in parallel))
                                    start_reading = start_reading+16

                    #then we also update start reading for the disconnected part
                    start_reading = start_reading + offset_diag3 #offset_diag3 is the chunk size of the disconnected part

                    #reading of 2point correlators

                    #loop over the time values                         (...tvals blocks of..)
                    for t in range(self.tvals):
                        #loop over the noise vectors                       (...nnoise blocks of...)
                        for inoise in range(self.nnoise):

                            #reading of re and im part of 2point correlators   (...complex numbers)

                            #first we read the two point with source in x0

                            #reading of real and imaginary part for the 2point corr with source in x0
                            re,im = struct.unpack("dd",fileContent[start_reading:start_reading+16])

                            #store complex number
                            x_corr[ic,t,inoise] = complex(re,im)

                            #then we read the two point with source in z0 (that is offset_diag2 bytes after the 2point with source in x0)

                            #reading of real and imaginary part for the 2point corr with source in z0
                            re,im = struct.unpack("dd",fileContent[start_reading+offset_diag2:start_reading+offset_diag2+16])

                            #store complex number 
                            z_corr[ic][t][inoise] = complex(re,im)

                            ##we update start reading (for the part that concerns the 2point with source in x0 (we are reading both 2point correlators in parallel))
                            start_reading = start_reading+16

                    #then we also update start reading for the 2point with source in z0
                    start_reading = start_reading + offset_diag2

                #once we read all the data of the given configuration we store it in a dictionary
                if str(conf_number) not in self.conf_dict_3p.keys():                           #if the key is new then we are reading a new configuration, so we just store it
                    self.conf_dict_3p[str(conf_number)] = (conn_corr.copy(),disc_corr.copy()) #(.copy() is mandatory, without it the reference to the array passed)
                    self.conf_dict_2p[str(conf_number)] = (x_corr.copy(),z_corr.copy())
                else:                                                                  #else if the conf number is the same is because the check_gauge_inv flag is on
                    self.conf_dict_3p[str(conf_number)+"_GaugeInvCheck"] = (conn_corr.copy(),disc_corr.copy())
                    self.conf_dict_2p[str(conf_number)+"_GaugeInvCheck"] = (x_corr.copy(),z_corr.copy())

        #data file completely read
        #we now initialize the relevant arrays that store the data in the class

        if verbose:
            print("\nInitializing the data arrays...\n")

        #construction of array with names and numbers of configurations
        self.conf_names = list(self.conf_dict_3p.keys())[0::1+self.check_gauge_inv] #(we skip the gauge transformed configuration if there are)
        self.nconf = len(self.conf_names)

        #creation of a numpy array with all the correlators
        self.all_3pCorr = np.empty(shape=(self.nconf,2,self.ncorr,self.noperators,self.tvals,self.nnoise,self.nnoise),dtype=complex) #the 2 is for connected and disconnected
        self.all_2pCorr_x = np.empty(shape=(self.nconf,self.ncorr,self.tvals,self.nnoise),dtype=complex)
        self.all_2pCorr_z = np.empty(shape=(self.nconf,self.ncorr,self.tvals,self.nnoise),dtype=complex)
        for iconf,nameconf in enumerate(self.conf_names):
            self.all_3pCorr[iconf] = self.conf_dict_3p[nameconf]
            self.all_2pCorr_x[iconf] = self.conf_dict_2p[nameconf][0]
            self.all_2pCorr_z[iconf] = self.conf_dict_2p[nameconf][1]


        #creation of the info box that will be displayed on all plots
        self.text_infobox = []
        for icorr in range(self.ncorr): #a different infobox for each correlator
            self.text_infobox.append( '\n'.join((
                 'Correlator %d parameters:' % (icorr),
                 '           ',
                r'$k_1$=%.9f ' % (self.k1[icorr] ),
                r'$k_2$=%.9f ' % (self.k2[icorr] ),
                r'$k_3$=%.9f ' % (self.k3[icorr] ),
                r'$k_4$=%.9f ' % (self.k4[icorr] ),
                    '           ',
                r'$\mu_1$=%.9f ' % (self.mu1[icorr] ),
                r'$\mu_2$=%.9f ' % (self.mu2[icorr] ),
                r'$\mu_3$=%.9f ' % (self.mu3[icorr] ),
                r'$\mu_4$=%.9f ' % (self.mu4[icorr] ),
                    '           ',
                r'$\Gamma_A$=' + self.latex_dirac_dict[self.typeA[icorr]],
                r'$\Gamma_B$=' + self.latex_dirac_dict[self.typeB[icorr]],
                    '           ',
                r'$x_0$=%d' % self.x0[icorr],
                r'$z_0$=%d' % self.z0[icorr],
                 '           ',
                 '           ',
                 '           ',
                 '           ',
                 'Simulation parameters:',
                 '           ',
                r'$N_{NOISE}$=%d' % self.nnoise,
                 'Noise Type=%s' % self.noise_dict[self.noise_type],
                 'Random Conf=%d' % self.random_conf,
                r'$T$=%d' % self.tvals,
                r'$c_{SW}$=%.9f ' % self.csw,
                r'$c_F$=%.9f' % self.cf,
                 '           ',
                 '           ',
                 '           ',
                 'Configurations:',
                 '           ',
                r'$N_{CONF}$=%d' % self.nconf,)) )


        #we also initilize the array that will be storing the final result about the matrix element
        #..for Q+ operators
        self.matrix_element = np.zeros(shape=(self.ncorr,self.noperators),dtype=float)
        self.matrix_element_std = np.zeros(shape=(self.ncorr,self.noperators),dtype=float)
        #..for Q- operators
        self.matrix_elementM = np.zeros(shape=(self.ncorr,self.noperators),dtype=float)
        self.matrix_element_stdM = np.zeros(shape=(self.ncorr,self.noperators),dtype=float)


        #initialization completed

        if verbose:
            print("Initialization complete\n")


    #utility function that prints all the info of the run
    def print_info(self):
        
        #Header print
        print("\n[File Header]\n")
        print(f"- ncorr           = {self.ncorr}\n")
        print(f"- nnoise          = {self.nnoise}\n")
        print(f"- tvals           = {self.tvals}\n")
        print(f"- noise_type      = {self.noise_dict[self.noise_type]}\n")
        print(f"- check_gauge_inv = {self.check_gauge_inv}\n")
        print(f"- random_conf     = {self.random_conf}\n")
        print(f"- csw             = {self.csw}\n")
        print(f"- cF              = {self.cf}\n\n")

        #configurations details print
        print("\n[Configurations]\n")
        print(f"- nconf               = {self.nconf}\n\n")
        #print(f"- conf_step           = {confSTEP}\n")

        #Correlators Header print
        for i in range(self.ncorr):
            print(f"[Correlator {i}]\n")
            print(f" - k1 = {self.k1[i]}\n")
            print(f" - k2 = {self.k2[i]}\n")
            print(f" - k3 = {self.k3[i]}\n")
            print(f" - k4 = {self.k4[i]}\n\n")
            print(f" - mu1 = {self.mu1[i]}\n")
            print(f" - mu2 = {self.mu2[i]}\n")
            print(f" - mu3 = {self.mu3[i]}\n")
            print(f" - mu4 = {self.mu4[i]}\n\n")
            print(f" - typeA = {self.dirac_dict[self.typeA[i]]}\n")
            print(f" - typeB = {self.dirac_dict[self.typeB[i]]}\n\n")
            print(f" - x0 = {self.x0[i]}\n")
            print(f" - z0 = {self.z0[i]}\n\n\n")

    #method for the analysis of the std vs the binsize (TO DO: HANDLE WARNINGS !!!)
    def std_study(self,first_conf=0,last_conf=None,step_conf=1,times=None,
                  show=False,save=True,verbose=True,subdir_name="stdAnalysis_plots"):

        #creation of subdir where to save plots
        subdir = self.plot_dir+"/" + subdir_name
        Path(subdir).mkdir(parents=True, exist_ok=True)

        #by default, if the times chosen to be ploted are not specified, three of them are chosen accordingly to tvals
        if times is None:
            times = [t for t in range(5,self.tvals,int(self.tvals/3))]

        #by default the last_configuration considered is the nconf-th one
        if last_conf is None:
            last_conf = self.nconf

        #we take the selected slice of correlator data
        correlators = self.all_3pCorr[first_conf:last_conf:step_conf]

        #current nconf considered
        new_nconf = np.shape(correlators)[0]

        #we perform the noise average
        correlators_navg = correlators.mean(axis=-1).mean(axis=-1)

        #we consider the total correlator by summing the connected and the disconnected part
        correlators_navg = correlators_navg[:,0,:,:,:] + correlators_navg[:,1,:,:,:]

        #choice of binning: divisors of number of configurations
        deltaList = [delta for delta in divisors(new_nconf) if delta < new_nconf/10]

        #output info
        if verbose:
            print("\nMaking std analysis plots for each correlator...\n")

        #we now loop over all the correlators, for each one we compute the std using the jackknife method and then we make a plot

        #loop over the correlators
        for icorr in range(self.ncorr):

            #we initialize the array in which we store the normalized std

            #array with a value of normalized std for each delta
            std_list_jack = []
            std_list_boot = []     
            std_list_simple = []

            #array with all the std for each delta
            allstd_jack = np.empty(shape=(len(deltaList),self.noperators,self.tvals),dtype=float)
            allstd_boot = np.empty(shape=(len(deltaList),self.noperators,self.tvals),dtype=float)
            allstd_simple = np.empty(shape=(len(deltaList),self.noperators,self.tvals),dtype=float)

            #loop over the possible deltas (size of deleted elements)
            for i_d,delta in enumerate(deltaList):

                #the axis with the configuration is now replaced with an axis with averages of configurations: 
                #   - the lenght of the axis passes from nconf to nconf/delta (that is an int by construction)
                #   - the i-th element will be the average of the configuration in the range (i*delta, (i+1)*delta)

                corr_binned = np.array([np.mean(correlators_navg[i*delta:(i+1)*delta,icorr,:,:],axis=0) for i in range(int(new_nconf/delta))])

                #now we compute mean and std using the jackknife method (implementd in the astropy library)

                #observable we're interested in
                test_statistic = np.mean

                #array where we will store the mean and std
                mean_array_jack = np.empty(shape=(self.noperators,self.tvals),dtype=float)
                std_array_jack = np.empty(shape=(self.noperators,self.tvals),dtype=float)

                mean_array_boot = np.empty(shape=(self.noperators,self.tvals),dtype=float)
                std_array_boot = np.empty(shape=(self.noperators,self.tvals),dtype=float)

                #loop in which we use the jackknife and the bootstrap (over the configurations for fixed operator and time)

                for iop in range(self.noperators): #for each operator
                    for t in range(self.tvals): #and for each time
                    
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
                std_array_simple = np.std(corr_binned.imag,axis=0) /np.sqrt(np.shape(corr_binned)[0]-1)

                #we now append to the std list the mean relative error
                std_list_simple.append(np.mean( (std_array_simple/np.abs(mean_array_simple.imag))[:,1:-2] ))

                allstd_jack[i_d] = (std_array_jack/np.abs(mean_array_jack))[:]
                allstd_boot[i_d] = (std_array_boot/np.abs(mean_array_boot))[:]
                allstd_simple[i_d] = (std_array_simple/np.abs(mean_array_simple.imag))[:]

            #now that we have all the std for each size of the bin (delta) we can make the plot of std vs delta

            ### make plot ###


            #output info
            if verbose:
                print(f"\nMaking plots for the correlator number {icorr} ...\n")


            #create figure and axis
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

            #adjust subplot spacing
            plt.subplots_adjust(left=0.1,
                                bottom=0.1, 
                                right=0.87, 
                                top=0.9, 
                                wspace=0.4, 
                                hspace=0.6)


            ax.plot(deltaList,std_list_jack,'-o',linewidth=0.4,alpha=0.7,markersize=9.5,color='blue',label='Jackknife Estimate')
            ax.plot(deltaList,std_list_boot,'-o',linewidth=0.25,alpha=0.7,markersize=8,color='green',label='Bootstrap Estimate')
            ax.plot(deltaList,std_list_simple,'-o',linewidth=0.1,alpha=0.7,markersize=6.5,color='red',label='Simple Estimate')
            ax.set_xticks(deltaList)
            ax.tick_params(axis='both', which='major', labelsize=8)

            ax.set_xlabel(r"$\Delta$",fontsize=12)
            ax.set_ylabel(r"$\sigma$ / $|\mu|$",fontsize=12)

            ax.legend()

            #set title
            ax.set_title(r"Normalized Standard Deviation as a function of the binsize $\Delta$" + f' - Correlator {icorr}', fontsize=15,y=1.03)
            

            #Display text box with frelevant parameters outside the plot
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place the text box in upper left in axes coords
            plt.text(1.01, 0.95, self.text_infobox[icorr], transform=ax.transAxes, fontsize=7,
                    verticalalignment='top', bbox=props)
            
            
            #save figure
            if save:
                fig_name = f"stdVSbin_corr{icorr}_{self.run_name}.png"
                plt.savefig(subdir+"/"+fig_name)


            # make other plots

            #loop over the operators
            for iop in tqdm(range(self.noperators)):

                #name of iop (enumerate is not used such that the loading bar is correctly visualized)
                op_name = self.op_names[iop]

                #create the figure for the given icorr and iop
                fig, ax_list = plt.subplots(nrows=len(times), ncols=1, sharex=True, sharey=False, figsize=(32, 14))

                #adjust subplot spacing
                plt.subplots_adjust(left=0.1,
                                    bottom=0.1, 
                                    right=0.87, 
                                    top=0.9, 
                                    wspace=0.4, 
                                    hspace=0.6)
    

                #for each time we have a different subplot
                for i,t in enumerate(times):

                    #plot the data
                    ax_list[i].plot(deltaList,allstd_jack[:,iop,t],'-o',linewidth=0.4,alpha=0.7,markersize=9.5,color='blue',label='Jackknife Estimate')
                    ax_list[i].plot(deltaList,allstd_boot[:,iop,t],'-o',linewidth=0.25,alpha=0.7,markersize=8,color='green',label='Bootstrap Estimate')
                    ax_list[i].plot(deltaList,allstd_simple[:,iop,t],'-o',linewidth=0.1,alpha=0.7,markersize=6.5,color='red',label='Simple Estimate')




                    #set title
                    ax_list[i].set_title(f"t = {t}",fontsize=15,weight="bold")

                    #set y label
                    ax_list[i].set_ylabel(r"$\sigma$ / $|\mu|$",rotation=90,labelpad=23,fontsize=18)

                    #set x ticks
                    ax_list[i].set_xticks(deltaList)
                    ax_list[i].tick_params(axis='both', which='major', labelsize=12)

                    #show legend
                    ax_list[i].legend()

                plt.xlabel(r"Binsize $\Delta$",fontsize=18,labelpad=23)

                plt.suptitle(r"Normalized Standard Deviation as a function of the binsize $\Delta$" + f' - Correlator {icorr} - Operator {op_name}', fontsize=23)


            
            

                #Display text box with frelevant parameters outside the plot
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                # place the text box in upper left in axes coords
                plt.text(1.01, 0.95, self.text_infobox[icorr], transform=ax_list[0].transAxes, fontsize=14,verticalalignment='top', bbox=props)
            
            
                #save figure
                if save:
                    fig_name = f"stdVSbin_jbs_corr{icorr}_{self.op_names_simple[iop]}_{self.run_name}.png"
                    plt.savefig(subdir+"/"+fig_name)



        #output info
        if verbose:
            print("\nAll plots done!\n")
        
        #if show is given open one png inside the dir
        if show==True:
        
            png_list = [f for f in listdir(subdir) if f.endswith('png') and isfile(join(subdir, f) )]
            for png in png_list[:1]:
                os.system("xdg-open "+subdir+'/'+png)    

    

    #method to plot the five correlators for each configuration
    def allConf_plot(self,first_conf=0,last_conf=None,step_conf=1,
                     first_time=0,last_time=None,
                     show=False,save=True,verbose=True,subdir_name="all_configurations"):
        
        #creation of subdir where to save plots
        subdir = self.plot_dir+"/"+subdir_name
        Path(subdir).mkdir(parents=True, exist_ok=True)

        #by default the last_configuration considered is the nconf-th one
        if last_conf is None:
            last_conf = self.nconf

        #by default the last time on the plot (on the x axis) is the number of tvals
        if last_time is None:
            last_time = self.tvals

        #creation of array of x values (for each plot)
        if last_time<0:
            times = np.arange(first_time,self.tvals+last_time)
        else:
            times = np.arange(first_time,last_time)

        #we take the selected slice of correlator data
        correlators = self.all_3pCorr[:,:,:,:,first_time:last_time]

        #we perform the noise average
        correlators_navg = correlators.mean(axis=-1).mean(axis=-1)

        #arrays used in the plots
        corr_lab = ["Connected","Disconnected","Total"]
        corr_colors = ["red","blue","purple"]


        #printo status info if verbose
        if verbose:
            print("\nMaking plot for each configuration...\n")

        #loop over the configurations to be plotted
        for iconf in tqdm(range(first_conf,last_conf,step_conf)):

            #loop over all correlators
            for icorr in range(self.ncorr):

                #create figure and axis
                fig, ax_list = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=False, figsize=(32, 14))

                #loop over plot, one for each of the 5 operators
                for iop,op_name in enumerate(self.op_names):

                    #the correlators we want to plot are connected, disconnected and total
                    conn_corr = correlators_navg[iconf,0,icorr,iop].imag #conf - piece - corr - op
                    disc_corr = correlators_navg[iconf,1,icorr,iop].imag
                    tot_corr = conn_corr+disc_corr
                    #we put them in a list
                    corr_list = [conn_corr,disc_corr,tot_corr]

                    #we plot everything
                    for i_plot,corr in enumerate(corr_list):
                        ax_list[iop].plot(times,corr,'-o',label=corr_lab[i_plot]+" im",color=corr_colors[i_plot],markersize=10,linewidth=0.5)

                    #enable grid
                    ax_list[iop].grid()

                    #set title
                    ax_list[iop].set_title(op_name,fontsize=15,weight="bold")

                    #set y label
                    ax_list[iop].set_ylabel("G(t)",rotation=0,labelpad=20,fontsize=16)

                    #set legend
                    ax_list[iop].legend(loc='right')

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
                plt.suptitle(f'G(t) for parity odd operators - (Configuration {iconf}, Correlator {icorr})', fontsize=25,y=0.98,)

                #Display text box with frelevant parameters outside the plot
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                # place the text box in upper left in axes coords
                plt.text(1.01, 0.95, self.text_infobox[icorr], transform=ax_list[0].transAxes, fontsize=14,
                        verticalalignment='top', bbox=props)

                #save figure
                if save:
                    fig_name = f"plot_conf{iconf}_corr{icorr}_{self.run_name}.png"
                    plt.savefig(subdir+"/"+fig_name)                    

                #once the plot is created terminate it
                plt.close()

        if verbose:
            print("\nAll plots done!\n")

        #if show is given open one png inside the dir
        if show==True:
        
            png_list = [f for f in listdir(subdir) if f.endswith('png') and isfile(join(subdir, f) )]
            for png in png_list[:1]:
                os.system("xdg-open "+subdir+'/'+png)


    #method to plot the five correlators for each configuration
    def preliminary_plots(self,first_time=0,last_time=None,
                          first_conf=0,last_conf=None,binsize=1,
                          show=False,save=True,verbose=True,subdir_name="preliminary_plots"):
        
        #creation of subdir where to save plots
        subdir = self.plot_dir+"/"+subdir_name
        Path(subdir).mkdir(parents=True, exist_ok=True)

        #by default the last_configuration considered is the nconf-th one
        if last_conf is None:
            last_conf = self.nconf

        #check that the parameters 
        if (last_conf-first_conf)%binsize != 0:
            print("\nlast_conf-first_conf should be a multiple integer of binsize!\n")
            return

        #by default the last time on the plot (on the x axis) is the number of tvals
        if last_time is None:
            last_time = self.tvals

        

        #creation of array of x values (for each plot)
        if last_time<0:
            times = np.arange(first_time,self.tvals+last_time)
        else:
            times = np.arange(first_time,last_time)

        #we take the selected slice of correlator data
        correlators3p = self.all_3pCorr[:,:,:,:,first_time:last_time] #conf - piece - corr - op - tval - noise - noise
        correlators2px = self.all_2pCorr_x[:,:,first_time:last_time] #conf - corr - tval - noise
        correlators2pz = self.all_2pCorr_z[:,:,first_time:last_time]

        #we perform the noise average
        correlators3p_navg = correlators3p.mean(axis=-1).mean(axis=-1)
        correlators2px_navg = correlators2px.mean(axis=-1)
        correlators2pz_navg = correlators2pz.mean(axis=-1)

        #for the 3 point corr we consider also the total correlator
        correlators3p_navg_tot = correlators3p_navg[:,0,:,:,:] + correlators3p_navg[:,1,:,:,:]

        #creation of jack replicates
        jack_replicates_x = np.asarray( [np.delete(correlators2px_navg, list(range(iconf,min(iconf+binsize,last_conf))) ,axis=0).mean(axis=0) for iconf in range(first_conf,last_conf,binsize)] )
        jack_replicates_z = np.asarray( [np.delete(correlators2pz_navg, list(range(iconf,min(iconf+binsize,last_conf))) ,axis=0).mean(axis=0) for iconf in range(first_conf,last_conf,binsize)] )
        jack_replicates_3 = np.asarray( [np.delete(correlators3p_navg, list(range(iconf,min(iconf+binsize,last_conf))) ,axis=0).mean(axis=0) for iconf in range(first_conf,last_conf,binsize)] )
        jack_replicates_3_tot = np.asarray( [np.delete(correlators3p_navg_tot, list(range(iconf,min(iconf+binsize,last_conf))) ,axis=0).mean(axis=0) for iconf in range(first_conf,last_conf,binsize)] )

        #jacknife estimator of mean
        jack_mean_x = jack_replicates_x.mean(axis=0)
        jack_mean_z = jack_replicates_z.mean(axis=0)
        jack_mean_3 = jack_replicates_3.mean(axis=0)
        jack_mean_3_tot = jack_replicates_3_tot.mean(axis=0)

        #number of replicates
        nrepl = np.shape(jack_replicates_3)[0]

        #jacknife estimator of std, looking at re and im part
        jack_std_x_re = np.sqrt(nrepl-1) * np.std(jack_replicates_x.real,axis=0)
        jack_std_z_re = np.sqrt(nrepl-1) * np.std(jack_replicates_z.real,axis=0)
        jack_std_3_re = np.sqrt(nrepl-1) * np.std(jack_replicates_3.real,axis=0)
        jack_std_3_im = np.sqrt(nrepl-1) * np.std(jack_replicates_3.imag,axis=0)
        jack_std_3_tot_re = np.sqrt(nrepl-1) * np.std(jack_replicates_3_tot.real,axis=0)
        jack_std_3_tot_im = np.sqrt(nrepl-1) * np.std(jack_replicates_3_tot.imag,axis=0)
        jack_std_3_tot_abs = np.sqrt(nrepl-1) * np.std(np.abs(jack_replicates_3_tot),axis=0)

        #output info
        if verbose:
            print("\nMaking preliminary plots for each correlator...\n")

        #we now loop over each correlator and for each we do a series of plots

        #loop over correlators
        for icorr in tqdm(range(self.ncorr)):

            ##############  plot number 1: comparison of connected and disconnected piece (by looking at imag part) ########################

            #create figure and axis
            fig, ax_list = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=False, figsize=(32, 14))

            #loop over plot, one for each of the 5 operators
            for iop,op_name in enumerate(self.op_names):
            
                #compute connected, disconnected and total correlator
                conn_corr = jack_mean_3[0,icorr,iop,:].imag #piece - corr - op - t
                disc_corr = jack_mean_3[1,icorr,iop,:].imag
                tot_corr = jack_mean_3_tot[icorr,iop,:].imag #corr - op - t

                #compute variance
                conn_std = jack_std_3_im[0,icorr,iop,:]
                disc_std = jack_std_3_im[1,icorr,iop,:]
                tot_std = jack_std_3_tot_im[icorr,iop,:].imag #corr - op - t

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

            #adjust subplot spacing
            plt.subplots_adjust(left=0.04,
                                bottom=0.05, 
                                right=0.9, 
                                top=0.9, 
                                wspace=0.4, 
                                hspace=0.6)

            #set x label
            plt.xlabel('Time [lattice units]',fontsize=16)

            #set title
            plt.suptitle(f'Im[G(t)] for parity odd operators - Correlator {icorr} - Connected, Disconnected and Total using Jackknife Method', fontsize=25,y=0.98)

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place the text box in upper left in axes coords
            plt.text(1.01, 0.95, self.text_infobox[icorr], transform=ax_list[0].transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)

            #save figure
            if save:
                fig_name = f"plot_conn-disc_corr{icorr}_{self.run_name}.png"
                plt.savefig(subdir+"/"+fig_name)

            #once the plot is created terminate it
            plt.close()


            ##############  plot number 2: comparison of real and imaginary part (by looking at total correlator) #####################ààà

            #create figure and axis
            fig, ax_list = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=False, figsize=(32, 14))

            #loop over plot, one for each of the 5 operators
            for iop,op_name in enumerate(self.op_names):
            
                #real imag and modulus of total correlator
                re_corr = jack_mean_3_tot[icorr,iop,:].real #corr - op - t
                imag_corr = jack_mean_3_tot[icorr,iop,:].imag 
                mod_corr = np.abs(jack_mean_3_tot[icorr,iop,:])

                #compute variance
                re_std = jack_std_3_tot_re[icorr,iop,:]
                im_std = jack_std_3_tot_im[icorr,iop,:]
                mod_std = jack_std_3_tot_abs[icorr,iop,:]

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
            plt.suptitle(f'Total G(t) for parity odd operators - Correlator {icorr} - Real, Imaginary and Modulus using Jackknife Method', fontsize=25,y=0.98)

            #Display text box with frelevant parameters outside the plot

            #textstr is defined above

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place the text box in upper left in axes coords
            plt.text(1.01, 0.95, self.text_infobox[icorr], transform=ax_list[0].transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)

            #save figure
            if save:
                fig_name = f"plot_re-im_corr{icorr}_{self.run_name}.png"
                plt.savefig(subdir+"/"+fig_name)

            #once the plot is created terminate it
            plt.close()


            ##############  plot number 3: comparison of jacknife replicates ########################

            #create figure and axis
            fig, ax_list = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=False, figsize=(32, 14))

            #loop over plot, one for each of the 5 operators
            for iop,op_name in enumerate(self.op_names):

                for irep in range(nrepl):
                    lbl = None
                    if irep == nrepl-1:
                        lbl = "Jackknife Replicates"
                    ax_list[iop].plot(times,jack_replicates_3_tot[irep,icorr,iop,:].imag,'-o',markersize=7,linewidth=0.5,alpha=0.4,color="red",label=lbl)

                #mean and std with jackknife method
                mean_corr = jack_mean_3_tot[icorr,iop,:].imag 
                std_corr = jack_std_3_tot_im[icorr,iop,:]

                #make the plot
                ax_list[iop].errorbar(times,mean_corr,yerr=std_corr,marker='o',label=r"Jackknife Mean $\pm$ std",color="black",markersize=10,linewidth=0.8,elinewidth=2)

                #enable grid
                ax_list[iop].grid()

                #set title
                ax_list[iop].set_title(op_name,fontsize=15,weight="bold")

                #set y label
                ax_list[iop].set_ylabel("G(t)",rotation=0,labelpad=20,fontsize=16)

                #set legend
                ax_list[iop].legend(loc='right')

            #adjust subplot spacing
            plt.subplots_adjust(left=0.04,
                                bottom=0.05, 
                                right=0.9, 
                                top=0.9, 
                                wspace=0.4, 
                                hspace=0.6)

            #set x label
            plt.xlabel('Time [lattice units]',fontsize=16)

            #set title
            plt.suptitle(f'Total Im[G(t)] for parity odd operators - Correlator {icorr} - Jackknife Replicates and Jackknife Mean', fontsize=25,y=0.98)

            #Display text box with frelevant parameters outside the plot
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place the text box in upper left in axes coords
            plt.text(1.01, 0.95, self.text_infobox[icorr], transform=ax_list[0].transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)

            #save figure
            if save:
                fig_name = f"plot_jack_corr{icorr}_{self.run_name}.png"
                plt.savefig(subdir+"/"+fig_name)

            #once the plot is created terminate it
            plt.close()



            ##############  plot number 4: comparison of configuration with jackknife mean #####################ààà

            #create figure and axis
            fig, ax_list = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=False, figsize=(32, 14))

            #loop over plot, one for each of the 5 operators
            for iop,op_name in enumerate(self.op_names):

                for iconf in range(self.nconf):
                    lbl = None
                    if iconf == self.nconf-1:
                        lbl = "Configurations (no Jackknife)"
                    ax_list[iop].plot(times,correlators3p_navg_tot[iconf,icorr,iop,:].imag,'-o',markersize=7,linewidth=0.5,alpha=0.4,color="red",label=lbl)

                #mean and std with jackknife method
                mean_corr = jack_mean_3_tot[icorr,iop,:].imag 
                std_corr = jack_std_3_tot_im[icorr,iop,:]

                #plot the mean
                ax_list[iop].errorbar(times,mean_corr,yerr=std_corr,marker='o',label=r"Jackknife Mean $\pm$ std",color="black",markersize=10,linewidth=0.8,elinewidth=2)

                #enable grid
                ax_list[iop].grid()

                #set title
                ax_list[iop].set_title(op_name,fontsize=15,weight="bold")

                #set y label
                ax_list[iop].set_ylabel("G(t)",rotation=0,labelpad=20,fontsize=16)

                #set legend
                ax_list[iop].legend(loc='right')

            #adjust subplot spacing
            plt.subplots_adjust(left=0.04,
                                bottom=0.05, 
                                right=0.9, 
                                top=0.9, 
                                wspace=0.4, 
                                hspace=0.6)

            #set x label
            plt.xlabel('Time [lattice units]',fontsize=16)

            #set title
            plt.suptitle(f'Total Im[G(t)] for parity odd operators - Correlator {icorr} - All Configurations and Jackknife Mean', fontsize=25,y=0.98)

            #Display text box with frelevant parameters outside the plot

            #textstr is defined above    

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place the text box in upper left in axes coords
            plt.text(1.01, 0.95, self.text_infobox[icorr], transform=ax_list[0].transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)

            #save figure
            if save:
                fig_name = f"plot_jack_corr{icorr}_allconf_{self.run_name}.png"
                plt.savefig(subdir+"/"+fig_name)

            #once the plot is created terminate it
            plt.close()


            

            ##############  plot number 5: 2 points func #####################

            #create figure and axis
            fig, ax_list = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=(32, 14))

            #plot the 2 point with source in x0

            ax_list[0].errorbar(times,np.abs(jack_mean_x[icorr,:].real),yerr=jack_std_x_re[icorr,:],marker='o',label=r"Jackknife Mean $\pm$ std",color="black",markersize=10,linewidth=0.8,elinewidth=2)
            #enable grid
            ax_list[0].grid()
            #set title
            ax_list[0].set_title(r"With source in $x_0$",fontsize=15,weight="bold")
            #set y label
            ax_list[0].set_ylabel("G(t)",rotation=0,labelpad=20,fontsize=16)
            #plot all conf
            for iconf in range(self.nconf):
                lbl = None
                if iconf == self.nconf-1:
                    lbl = "Configurations (no Jackknife)"
                ax_list[0].plot(times,np.abs(correlators2px_navg[iconf,icorr,:].real),'-o',markersize=7,linewidth=0.5,alpha=0.4,color="red",label=lbl)
            #set legend
            ax_list[0].legend(loc='right')

            #plot the 2 point with source in z0

            ax_list[1].errorbar(times,np.abs(jack_mean_z[icorr,:].real),yerr=jack_std_z_re[icorr,:],marker='o',label=r"Jackknife Mean $\pm$ std",color="black",markersize=10,linewidth=0.8,elinewidth=2)
            #enable grid
            ax_list[1].grid()
            #set title
            ax_list[1].set_title(r"With source in $z_0$",fontsize=15,weight="bold")
            #set y label
            ax_list[1].set_ylabel("G(t)",rotation=0,labelpad=20,fontsize=16)
            #plot all conf
            for iconf in range(self.nconf):
                lbl = None
                if iconf == self.nconf-1:
                    lbl = "Configurations (no Jackknife)"
                ax_list[1].plot(times,np.abs(correlators2pz_navg[iconf,icorr,:].real),'-o',markersize=7,linewidth=0.5,alpha=0.4,color="red",label=lbl)
            #set legend
            ax_list[1].legend(loc='right')

            #set log scale (to observe exponential behaviour)
            ax_list[1].set_yscale('log')
            ax_list[0].set_yscale('log')

            #adjust subplot spacing
            plt.subplots_adjust(left=0.04,
                                bottom=0.05, 
                                right=0.9, 
                                top=0.9, 
                                wspace=0.4, 
                                hspace=0.6)

            #set x label
            plt.xlabel('Time [lattice units]',fontsize=16)

            #set title
            plt.suptitle(f'|Re[G(t)]| for 2 point correlator - Correlator {icorr}', fontsize=25,y=0.98)

            #Display text box with frelevant parameters outside the plot #texstr defined before
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place the text box in upper left in axes coords
            plt.text(1.01, 0.95, self.text_infobox[icorr], transform=ax_list[0].transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)

            #save figure
            if save:
                fig_name = f"plot_2point_corr{icorr}_{self.run_name}.png"
                plt.savefig(subdir+"/"+fig_name)

            #once the plot is created terminate it
            plt.close()

        #output info
        if verbose:
            print("\nAll plots done!\n")


        #if show is given open one png inside the dir
        if show==True:
        
            png_list = [f for f in listdir(subdir) if f.endswith('png') and isfile(join(subdir, f) )]
            for png in png_list[:1]:
                os.system("xdg-open "+subdir+'/'+png)



    #method to plot separately the five operators for each correlator
    def detailed_plots(self,first_time=0,last_time=None,
                          first_conf=0,last_conf=None,binsize=1,
                          show=False,save=True,verbose=True,subdir_name="detailed_plots"):
        
        #creation of subdir where to save plots
        subdir = self.plot_dir+"/"+subdir_name
        Path(subdir).mkdir(parents=True, exist_ok=True)

        #by default the last_configuration considered is the nconf-th one
        if last_conf is None:
            last_conf = self.nconf

        #check that the parameters 
        if (last_conf-first_conf)%binsize != 0:
            print("\nlast_conf-first_conf should be a multiple integer of binsize!\n")
            return

        #by default the last time on the plot (on the x axis) is the number of tvals
        if last_time is None:
            last_time = self.tvals

        #creation of array of x values (for each plot)
        if last_time<0:
            times = np.arange(first_time,self.tvals+last_time)
        else:
            times = np.arange(first_time,last_time)

        #we take the selected slice of correlator data
        corr = self.all_3pCorr[:,:,:,:,first_time:last_time] #conf - piece - corr - op - tvals - noise - noise

        #we perform the noise average
        corr_navg = corr.mean(axis=-1).mean(axis=-1)

        #we consider the total correlator
        corr_navg_tot = corr_navg[:,0,:,:,:] + corr_navg[:,1,:,:,:]

        #we bin bin the correlators by averaging over an interval with size binsize
        corr_binned = np.array([np.mean(corr_navg_tot[i*binsize:(i+1)*binsize],axis=0) for i in range(int((last_conf-first_conf)/binsize))])

        #observable we're interested in when using the jackknife
        test_statistic = np.mean

        #we now loop over the correlators, over the 5 operators and each time we do a plot using the jackknife method

        #loop over the correlators
        for icorr in range(self.ncorr):


            #output info
            if verbose:
                print(f"\nMaking plots for the correlator number {icorr} ...\n")


            #loop over the operators
            for iop in tqdm(range(self.noperators)):

                #name of iop (enumerate is not used such that the loading bar is correctly visualized)
                op_name = self.op_names[iop]

                #first let's compute mean and std using the jackknife

                mean_array = np.empty(shape=np.shape(times),dtype=float)
                std_array = np.empty(shape=np.shape(times),dtype=float)

                #for each time we have a 1D array and we use the jackknife implementation of astropy
                for t in range(len(times)):

                    #jackknife with astropy

                    #we choose as array the one spanning over the configurations (binned)
                    data = corr_binned[:,icorr,iop,t].imag
                
                    #we compute mean and std using the jackknife
                    estimate, _, stderr, _ = jackknife_stats(data, test_statistic, 0.95)
                
                    mean_array[t] = estimate
                    std_array[t] = stderr

                #now we do the plot

                #create figure and axis
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(32, 14))

                #we plot the correlator
                ax.errorbar(times,mean_array,yerr=std_array,marker='o',label=r"Jackknife Mean $\pm$ std",color="black",markersize=10,linewidth=0.8,elinewidth=2)

                #enable grid
                ax.grid()

                #set y label
                ax.set_ylabel("G(t)",rotation=0,labelpad=20,fontsize=16)

                #set legend
                ax.legend(loc='right')


                #adjust subplot spacing
                plt.subplots_adjust(left=0.04,
                                    bottom=0.05, 
                                    right=0.9, 
                                    top=0.9, 
                                    wspace=0.4, 
                                    hspace=0.6)

                #set x label
                plt.xlabel('Time [lattice units]',fontsize=16)

                #set title
                plt.suptitle(f'Total Im[G(t)] for {op_name} operator - Correlator {icorr} - Zoom on Plateau', fontsize=25,y=0.98)

                #Display text box with frelevant parameters outside the plot
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                # place the text box in upper left in axes coords
                plt.text(1.01, 0.95, self.text_infobox[icorr], transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', bbox=props)

                #save figure
                if save:
                    fig_name = f"plot_plateau_{self.op_names_simple[iop]}_corr{icorr}_{self.run_name}.png"
                    plt.savefig(subdir+"/"+fig_name)


        #output info
        if verbose:
            print("\nAll plots done!\n")


        #if show is given open one png inside the dir
        if show==True:
        
            png_list = [f for f in listdir(subdir) if f.endswith('png') and isfile(join(subdir, f) )]
            for png in png_list[:1]:
                os.system("xdg-open "+subdir+'/'+png)



    #method to plot separately the five operators for each correlator
    def mat_ele_extraction(self,first_conf=0,last_conf=None,binsize=1,zoom_out=0,digits=1,max_chi2=1.0,
                          show=False,save=True,verbose=True,result_save=True,y_min=None,subdir_name="mat_ele_extraction"):
        
        #creation of subdir where to save plots
        subdir = self.plot_dir+"/"+subdir_name
        Path(subdir).mkdir(parents=True, exist_ok=True)

        #by default the last_configuration considered is the nconf-th one
        if last_conf is None:
            last_conf = self.nconf

        #check that the parameters 
        if (last_conf-first_conf)%binsize != 0:
            print("\nlast_conf-first_conf should be a multiple integer of binsize!\n")
            return
        
        #time array for the plot
        times=np.arange(self.tvals)
        
        #we now choose the arrays that we have to use
        
        #for the 3 point the dimensions are: conf - piece - corr - op - tval - noise - noise
        corr_3p = self.all_3pCorr[:,0,:,:,:,:,:] +  self.all_3pCorr[:,1,:,:,:,:,:] #we sum disconnected and connected piece (Q+ operators)
        corrM_3p = self.all_3pCorr[:,0,:,:,:,:,:] -  self.all_3pCorr[:,1,:,:,:,:,:] #we sum disconnected and connected piece with a minus (Q- operators)


        #for the 2 point the dimensions are: conf - corr - tvals - noise
        corr_x = self.all_2pCorr_x[:,:,:,:] 
        corr_z = self.all_2pCorr_z[:,:,:,:]

        #then we take the average over the noise vectors
        corr_3p_navg_before_rot = corr_3p.mean(axis=-1).mean(axis=-1)
        corrM_3p_navg_before_rot = corrM_3p.mean(axis=-1).mean(axis=-1)
        corr_x_navg = corr_x.mean(axis=-1)
        corr_z_navg = corr_z.mean(axis=-1)

        #rotate the navg 3p corr into the right basis
        corr_3p_navg = np.einsum('ij,lmjn->lmin',self.rot_mat,corr_3p_navg_before_rot)
        corrM_3p_navg = np.einsum('ij,lmjn->lmin',self.rot_mat,corrM_3p_navg_before_rot)


        #output info
        if verbose:
            print(f"\nExtracting Matrix Elements using the Jackknife Method ...\n")


        #we now implement the jackknife technique

        #1) first the creation of the resamples
        corr_3p_navg_resamp = np.asarray( [np.delete(corr_3p_navg, list(range(iconf,min(iconf+binsize,last_conf))) ,axis=0) for iconf in range(first_conf,last_conf,binsize)] )
        corrM_3p_navg_resamp = np.asarray( [np.delete(corrM_3p_navg, list(range(iconf,min(iconf+binsize,last_conf))) ,axis=0) for iconf in range(first_conf,last_conf,binsize)] )
        corr_x_navg_resamp = np.asarray( [np.delete(corr_x_navg, list(range(iconf,min(iconf+binsize,last_conf))) ,axis=0) for iconf in range(first_conf,last_conf,binsize)] )
        corr_z_navg_resamp = np.asarray( [np.delete(corr_z_navg, list(range(iconf,min(iconf+binsize,last_conf))) ,axis=0) for iconf in range(first_conf,last_conf,binsize)] )
        
        #the number of resamples is
        nresamples = int((last_conf-first_conf)/binsize)

        #2) then with each resample we compute the matrix element

        #we average over the gauge configurations
        corr_3p_navg_resamp_gavg = corr_3p_navg_resamp.mean(axis=1)
        corrM_3p_navg_resamp_gavg = corrM_3p_navg_resamp.mean(axis=1)
        corr_x_navg_resamp_gavg = corr_x_navg_resamp.mean(axis=1)
        corr_z_navg_resamp_gavg = corr_z_navg_resamp.mean(axis=1)

        #with these arrays we can compute the replica of the matrix element for each subsample...
        matele_replicas = np.empty(shape=(nresamples,self.ncorr,self.noperators,self.tvals),dtype=float)
        mateleM_replicas = np.empty(shape=(nresamples,self.ncorr,self.noperators,self.tvals),dtype=float)
        #...with the following loop
        #then we estimate the mass
        for ires in range(nresamples): #for each resample
            for icorr in range(self.ncorr): #for each correlator
                for iop in range(self.noperators): #for each operator
                    for t in range(self.tvals): #and for each time
                        #we compute the matrix element using the formula
                        matele_replicas[ires,icorr,iop,t] = np.sqrt( ( corr_3p_navg_resamp_gavg[ires,icorr,iop,t] * np.conjugate( corr_3p_navg_resamp_gavg[ires,icorr,iop,self.tvals-1-t] ) / ( corr_z_navg_resamp_gavg[ires,icorr,1] * corr_x_navg_resamp_gavg[ires,icorr,self.tvals-2] ) ).real )
                        mateleM_replicas[ires,icorr,iop,t] = np.sqrt( ( corrM_3p_navg_resamp_gavg[ires,icorr,iop,t] * np.conjugate( corrM_3p_navg_resamp_gavg[ires,icorr,iop,self.tvals-1-t] ) / ( corr_z_navg_resamp_gavg[ires,icorr,1] * corr_x_navg_resamp_gavg[ires,icorr,self.tvals-2] ) ).real )
        
        #3) we then compute the matrix element also on the whole dataset (non on the subsamples)

        #to do so we compute first the gauge averages on the whole dataset
        corr_3p_navg_gavg = corr_3p_navg.mean(axis=0)
        corrM_3p_navg_gavg = corrM_3p_navg.mean(axis=0)
        corr_x_navg_gavg = corr_x_navg.mean(axis=0)
        corr_z_navg_gavg = corr_z_navg.mean(axis=0)

        #the matrix element on the whole dataset is
        matele_total = np.empty(shape=(self.ncorr,self.noperators,self.tvals),dtype=float)
        mateleM_total = np.empty(shape=(self.ncorr,self.noperators,self.tvals),dtype=float)
        #..and we compute it with the following loop
        for icorr in range(self.ncorr): #for each correlator
            for iop in range(self.noperators): #for each operator
                for t in range(self.tvals): #and for each time
                    #we use the formula of the matrix element
                    matele_total[icorr,iop,t] = np.sqrt( ( corr_3p_navg_gavg[icorr,iop,t] * np.conjugate( corr_3p_navg_gavg[icorr,iop,self.tvals-1-t] ) / ( corr_z_navg_gavg[icorr,1] * corr_x_navg_gavg[icorr,self.tvals-2] ) ).real )
                    mateleM_total[icorr,iop,t] = np.sqrt( ( corrM_3p_navg_gavg[icorr,iop,t] * np.conjugate( corrM_3p_navg_gavg[icorr,iop,self.tvals-1-t] ) / ( corr_z_navg_gavg[icorr,1] * corr_x_navg_gavg[icorr,self.tvals-2] ) ).real )

        #4) then we compute the estimate, the bias and the std according to the jackknife method

        #the estimate is the average over the resamples
        matele_estimate = np.mean(matele_replicas,axis=0)
        mateleM_estimate = np.mean(mateleM_replicas,axis=0)

        #the bias is the following difference between the mean of the replicates and the mean on the whole dataset
        bias = (nresamples-1) * (matele_estimate-matele_total)
        biasM = (nresamples-1) * (mateleM_estimate-mateleM_total)

        #the std is given by the following formula (variance of replicates times n-1)
        matele_std = np.sqrt( (nresamples-1)/nresamples * np.sum( (matele_replicas - matele_estimate)**2,axis=0 ) )
        mateleM_std = np.sqrt( (nresamples-1)/nresamples * np.sum( (mateleM_replicas - mateleM_estimate)**2,axis=0 ) )

        #then we correct the estimate for the bias
        matele = matele_estimate-bias
        mateleM = mateleM_estimate-biasM

        #compute the covariance matrix
        cov_mat = np.empty(shape=(self.ncorr,self.noperators,self.tvals,self.tvals),dtype=float)
        cov_matM = np.empty(shape=(self.ncorr,self.noperators,self.tvals,self.tvals),dtype=float)
        for t1 in range(self.tvals):
            for t2 in range(self.tvals):
                cov_mat[:,:,t1,t2] = (nresamples-1)/nresamples * np.sum( (matele_replicas[:,:,:,t1] - matele_estimate[:,:,t1]) * (matele_replicas[:,:,:,t2] - matele_estimate[:,:,t2]),axis=0 )
                cov_matM[:,:,t1,t2] = (nresamples-1)/nresamples * np.sum( (mateleM_replicas[:,:,:,t1] - mateleM_estimate[:,:,t1]) * (mateleM_replicas[:,:,:,t2] - mateleM_estimate[:,:,t2]),axis=0 )



        #we now loop over the correlators, over the 5 operators and each time we do a plot using the jackknife method

        #loop over the correlators
        for icorr in range(self.ncorr):


            #output info
            if verbose:
                print(f"\nMaking plots for the correlator number {icorr} ...\n")

            
            #first for the Q+ operators


            chosen_cuts = []

            #first we determine which is the plateau region for each operator
            for iop in range(self.noperators):
                for icut in range(1,int(self.tvals/2)):
                    #if (chi2(matele[icorr,:,icut:-icut],matele_std[icorr,:,icut:-icut],axis=1) < np.shape(matele[icorr,:,icut:-icut])[1]).all():
                    if reduced_cov_chi2(matele[icorr,:,icut:-icut],cov_mat[icorr,:,icut:-icut,icut:-icut],axis=1)[iop] < max_chi2:
                        chosen_cut = icut
                        chosen_cuts.append(chosen_cut)
                        break
                #then we average the data points on the plateau to find the matrix element
                self.matrix_element[icorr,iop] = np.mean(matele[icorr,iop,chosen_cut:-chosen_cut]) 
                self.matrix_element_std[icorr,iop] = np.sqrt( np.mean( matele_std[icorr,iop,chosen_cut:-chosen_cut]**2 ) ) 


            #one plot with all the operators together

            #output info
            if verbose:
                print(f"\nPlotting all the Q+ matrix elements in one graph ...\n")

            #create figure and axis                 
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(32, 14))

            colors = ["red","blue","orange","green","purple"]

            #loop over the operators
            for iop in tqdm(range(self.noperators)):

                #name of iop (enumerate is not used such that the loading bar is correctly visualized)
                op_name = self.op_names_rot[iop]

                #now we do the plot

                #for the plateau region the cut is the one found by the chi2 method
                cut = chosen_cuts[iop]

                #we plot first the hlines representing the plateau region
                ax.hlines(self.matrix_element[icorr,iop],cut,self.tvals-1-cut,color=colors[iop],linewidth=4,alpha=0.8)
                ax.hlines(self.matrix_element[icorr,iop]+self.matrix_element_std[icorr,iop],cut,self.tvals-1-cut,linestyle='dashed',linewidth=3,color=colors[iop],alpha=0.5)
                ax.hlines(self.matrix_element[icorr,iop]-self.matrix_element_std[icorr,iop],cut,self.tvals-1-cut,linestyle='dashed',linewidth=3,color=colors[iop],alpha=0.5)


                #for the other data we can use a wider range in the plot
                cut = np.min(np.array(chosen_cuts)-zoom_out)

                
    
                #we plot the atrix element obtained from the jackknife
                ax.errorbar(times[cut:-cut],matele[icorr,iop,cut:-cut],yerr=matele_std[icorr,iop,cut:-cut],
                             marker='o',linestyle='--',color=colors[iop],markersize=10,linewidth=0.6,alpha=0.8,elinewidth=2,label=op_name)
                
            #with all the operators together the x axis is shown by default
            ax.set_ylim(0,np.max(matele[icorr,:,cut:-cut]+matele_std[icorr,:,cut:-cut]))

            #enable grid
            ax.grid()

            #set y label
            ax.set_ylabel(r'$\left|\left<\widetilde{ps}|O|PS\right>\right|$',rotation=90,labelpad=20,fontsize=16)

            #set legend
            ax.legend(loc='right')


            #adjust subplot spacing
            plt.subplots_adjust(left=0.04,
                                bottom=0.05, 
                                right=0.9, 
                                top=0.9, 
                                wspace=0.4, 
                                hspace=0.6)

            #set x label
            plt.xlabel('Time [lattice units]',fontsize=16)

            #set title
            plt.suptitle(r"Matrix Element - $\mathcal{Q}^+$"+f' Operators - Correlator {icorr}', fontsize=25,y=0.98)

            #Display text box with frelevant parameters outside the plot
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place the text box in upper left in axes coords
            plt.text(1.01, 0.95, self.text_infobox[icorr], transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)

            #save figure
            if save:
                fig_name = f"plot_matrixelement_alloperators_corr{icorr}_{self.run_name}.png"
                plt.savefig(subdir+"/"+fig_name)



            #5 plots for the 5 operators

            #output info
            if verbose:
                print(f"\nMaking one plot for each Q+ matrix element ...\n")

            #loop over the operators
            for iop in tqdm(range(self.noperators)):

                #name of iop (enumerate is not used such that the loading bar is correctly visualized)
                op_name = self.op_names_rot[iop]


                #create figure and axis
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(32, 14))


                #now we do the plot

                #for the plateau region the cut is the one found by the chi2 method
                cut = chosen_cuts[iop]

                #we plot first the hlines representing the plateau region
                ax.hlines(self.matrix_element[icorr,iop],cut,self.tvals-1-cut,color='red',label='Average',linewidth=4)
                ax.hlines(self.matrix_element[icorr,iop]+self.matrix_element_std[icorr,iop],cut,self.tvals-1-cut,color='orange',linestyle='dashed',linewidth=3)
                ax.hlines(self.matrix_element[icorr,iop]-self.matrix_element_std[icorr,iop],cut,self.tvals-1-cut,color='orange',linestyle='dashed',label=r'Average $\pm$ Standard Deviation',linewidth=3)


                #for the other data we can use a wider range in the plot
                cut = chosen_cuts[iop]-zoom_out

                #show x axis
                if y_min is not None:
                    ax.set_ylim(y_min*np.min(matele[icorr,iop,cut:-cut]+matele_std[icorr,iop,cut:-cut]),np.max(matele[icorr,iop,cut:-cut]+matele_std[icorr,iop,cut:-cut]))
                
    
                #we plot the atrix element obtained from the jackknife
                ax.errorbar(times[cut:-cut],matele[icorr,iop,cut:-cut],yerr=matele_std[icorr,iop,cut:-cut],
                             marker='o',linestyle='solid',markersize=10,linewidth=0.8,elinewidth=2,label='Jackknife Estimates')

                #enable grid
                ax.grid()

                #set y label
                ax.set_ylabel(r'$\left|\left<\widetilde{ps}|O|PS\right>\right|$',rotation=90,labelpad=20,fontsize=16)

                #set legend
                ax.legend(loc='right')


                #adjust subplot spacing
                plt.subplots_adjust(left=0.04,
                                    bottom=0.05, 
                                    right=0.9, 
                                    top=0.9, 
                                    wspace=0.4, 
                                    hspace=0.6)

                #set x label
                plt.xlabel('Time [lattice units]',fontsize=16)

                #set title
                plt.suptitle(r'$\mathcal{Q}^+$'+f' Matrix Element - Operator {op_name} - Correlator {icorr}', fontsize=25,y=0.98)

                #Display text box with frelevant parameters outside the plot
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                # place the text box in upper left in axes coords
                plt.text(1.01, 0.95, self.text_infobox[icorr], transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', bbox=props)

                #save figure
                if save:
                    fig_name = f"plot_matrixelement_{self.op_names_rot_simple[iop]}_corr{icorr}_{self.run_name}.png"
                    plt.savefig(subdir+"/"+fig_name)




            #then for the Q- operators

            chosen_cuts=[]

            #first we determine which is the plateau region for each operator
            for iop in range(self.noperators):
                for icut in range(1,int(self.tvals/2)):
                    #if (chi2(matele[icorr,:,icut:-icut],matele_std[icorr,:,icut:-icut],axis=1) < np.shape(matele[icorr,:,icut:-icut])[1]).all():
                    if reduced_cov_chi2(mateleM[icorr,:,icut:-icut],cov_matM[icorr,:,icut:-icut,icut:-icut],axis=1)[iop] < max_chi2:
                        chosen_cut = icut
                        chosen_cuts.append(chosen_cut)
                        break
                #then we average the data points on the plateau to find the matrix element
                self.matrix_elementM[icorr,iop] = np.mean(mateleM[icorr,iop,chosen_cut:-chosen_cut]) 
                self.matrix_element_stdM[icorr,iop] = np.sqrt( np.mean( mateleM_std[icorr,iop,chosen_cut:-chosen_cut]**2 ) ) 


            #one plot with all the operators together

            #output info
            if verbose:
                print(f"\nPlotting all the Q- matrix elements in one graph ...\n")

            #create figure and axis                 
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(32, 14))

            colors = ["red","blue","orange","green","purple"]

            #loop over the operators
            for iop in tqdm(range(self.noperators)):

                #name of iop (enumerate is not used such that the loading bar is correctly visualized)
                op_name = self.op_names_rot[iop]

                #now we do the plot

                #for the plateau region the cut is the one found by the chi2 method
                cut = chosen_cuts[iop]

                #we plot first the hlines representing the plateau region
                ax.hlines(self.matrix_elementM[icorr,iop],cut,self.tvals-1-cut,color=colors[iop],linewidth=4,alpha=0.8)
                ax.hlines(self.matrix_elementM[icorr,iop]+self.matrix_element_stdM[icorr,iop],cut,self.tvals-1-cut,linestyle='dashed',linewidth=3,color=colors[iop],alpha=0.5)
                ax.hlines(self.matrix_elementM[icorr,iop]-self.matrix_element_stdM[icorr,iop],cut,self.tvals-1-cut,linestyle='dashed',linewidth=3,color=colors[iop],alpha=0.5)


                #for the other data we can use a wider range in the plot
                cut = np.min(np.array(chosen_cuts)-zoom_out)

                
    
                #we plot the atrix element obtained from the jackknife
                ax.errorbar(times[cut:-cut],mateleM[icorr,iop,cut:-cut],yerr=mateleM_std[icorr,iop,cut:-cut],
                             marker='o',linestyle='--',color=colors[iop],markersize=10,linewidth=0.6,alpha=0.8,elinewidth=2,label=op_name)
                
            #with all the operators together the x axis is shown by default
            ax.set_ylim(0,np.max(mateleM[icorr,:,cut:-cut]+mateleM_std[icorr,:,cut:-cut]))

            #enable grid
            ax.grid()

            #set y label
            ax.set_ylabel(r'$\left|\left<\widetilde{ps}|O|PS\right>\right|$',rotation=90,labelpad=20,fontsize=16)

            #set legend
            ax.legend(loc='right')


            #adjust subplot spacing
            plt.subplots_adjust(left=0.04,
                                bottom=0.05, 
                                right=0.9, 
                                top=0.9, 
                                wspace=0.4, 
                                hspace=0.6)

            #set x label
            plt.xlabel('Time [lattice units]',fontsize=16)

            #set title
            plt.suptitle(r"Matrix Element - $\mathcal{Q}^-$"+f' Operators - Correlator {icorr}', fontsize=25,y=0.98)

            #Display text box with frelevant parameters outside the plot
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place the text box in upper left in axes coords
            plt.text(1.01, 0.95, self.text_infobox[icorr], transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)

            #save figure
            if save:
                fig_name = f"plot_matrixelement_alloperatorsM_corr{icorr}_{self.run_name}.png"
                plt.savefig(subdir+"/"+fig_name)



            #5 plots for the 5 operators

            #output info
            if verbose:
                print(f"\nMaking one plot for each Q- matrix element ...\n")

            #loop over the operators
            for iop in tqdm(range(self.noperators)):

                #name of iop (enumerate is not used such that the loading bar is correctly visualized)
                op_name = self.op_names_rot[iop]


                #create figure and axis
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(32, 14))


                #now we do the plot

                #for the plateau region the cut is the one found by the chi2 method
                cut = chosen_cuts[iop]

                #we plot first the hlines representing the plateau region
                ax.hlines(self.matrix_elementM[icorr,iop],cut,self.tvals-1-cut,color='red',label='Average',linewidth=4)
                ax.hlines(self.matrix_elementM[icorr,iop]+self.matrix_element_stdM[icorr,iop],cut,self.tvals-1-cut,color='orange',linestyle='dashed',linewidth=3)
                ax.hlines(self.matrix_elementM[icorr,iop]-self.matrix_element_stdM[icorr,iop],cut,self.tvals-1-cut,color='orange',linestyle='dashed',label=r'Average $\pm$ Standard Deviation',linewidth=3)


                #for the other data we can use a wider range in the plot
                cut = chosen_cuts[iop]-zoom_out

                #show x axis
                if y_min is not None:
                    ax.set_ylim(y_min*np.min(mateleM[icorr,iop,cut:-cut]+mateleM_std[icorr,iop,cut:-cut]),np.max(mateleM[icorr,iop,cut:-cut]+mateleM_std[icorr,iop,cut:-cut]))
                
    
                #we plot the atrix element obtained from the jackknife
                ax.errorbar(times[cut:-cut],mateleM[icorr,iop,cut:-cut],yerr=mateleM_std[icorr,iop,cut:-cut],
                             marker='o',linestyle='solid',markersize=10,linewidth=0.8,elinewidth=2,label='Jackknife Estimates')

                #enable grid
                ax.grid()

                #set y label
                ax.set_ylabel(r'$\left|\left<\widetilde{ps}|O|PS\right>\right|$',rotation=90,labelpad=20,fontsize=16)

                #set legend
                ax.legend(loc='right')


                #adjust subplot spacing
                plt.subplots_adjust(left=0.04,
                                    bottom=0.05, 
                                    right=0.9, 
                                    top=0.9, 
                                    wspace=0.4, 
                                    hspace=0.6)

                #set x label
                plt.xlabel('Time [lattice units]',fontsize=16)

                #set title
                plt.suptitle(r'$\mathcal{Q}^-$'+f' Matrix Element - Operator {op_name} - Correlator {icorr}', fontsize=25,y=0.98)

                #Display text box with frelevant parameters outside the plot
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                # place the text box in upper left in axes coords
                plt.text(1.01, 0.95, self.text_infobox[icorr], transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', bbox=props)

                #save figure
                if save:
                    fig_name = f"plot_matrixelementM_{self.op_names_rot_simple[iop]}_corr{icorr}_{self.run_name}.png"
                    plt.savefig(subdir+"/"+fig_name)
    



        #output info
        if verbose:
            print("\nAll plots done!\n")
            print("\nThe final result for the matrix elements are:\n")
            for icorr in range(self.ncorr):
                print(f"-Correlator {icorr}:\n")
                print("  Q+ :\n")
                for iop in range(self.noperators):
                    print(present_result(f"-{self.op_names_rot_simple[iop]}:{' '*(12-len(self.op_names_rot_simple[iop]))}",self.matrix_element[icorr,iop],self.matrix_element_std[icorr,iop],digits,'')+"\n")
                print("  Q- :\n")
                for iop in range(self.noperators):
                    print(present_result(f"-{self.op_names_rot_simple[iop]}:{' '*(12-len(self.op_names_rot_simple[iop]))}",self.matrix_elementM[icorr,iop],self.matrix_element_stdM[icorr,iop],digits,'')+"\n")
                print(f"\n")


        #if show is given open one png inside the dir
        if show==True:
        
            png_list = [f for f in listdir(subdir) if f.endswith('png') and isfile(join(subdir, f) )]
            for png in png_list[:1]:
                os.system("xdg-open "+subdir+'/'+png)


        #the final result of the matrix element gets store in a txt file
        if result_save:
            with open(self.plot_dir+"matrix_element_result.txt","w") as file:
                #first we write and header explaining how to read the file
                file.write("#The file is structured as follows:\n")
                file.write(f"#ncorr={self.ncorr} blocks, each with the {self.noperators}x2 operators, in the order {','.join(self.op_names_rot_simple)}, first all the Q+, then all the Q-\n")
                file.write("#column 0 is the mean, column 1 is the std\n")
                #then loop over the correlators and the operators and we print all the matrix elements computed
                for icorr in range(self.ncorr):
                    for iop in range(self.noperators):
                        file.write(f"{self.matrix_element[icorr,iop]} {self.matrix_element_std[icorr,iop]}\n")
                    for iop in range(self.noperators):
                        file.write(f"{self.matrix_elementM[icorr,iop]} {self.matrix_element_stdM[icorr,iop]}\n")


    #method for the analysis of the std vs the binsize (TO DO: HANDLE WARNINGS !!!)
    def std_study_detailed(self,first_conf=0,last_conf=None,step_conf=1,times=None,
                           show=False,save=True,verbose=True,subdir_name="stdAnalysis_plots"):

        #creation of subdir where to save plots
        subdir = self.plot_dir+"/" + subdir_name
        Path(subdir).mkdir(parents=True, exist_ok=True)

        #by default the last_configuration considered is the nconf-th one
        if last_conf is None:
            last_conf = self.nconf

        #by default, if the times chosen to be ploted are not specified, three of them are chosen accordingly to tvals
        if times is None:
            times = [t for t in range(5,self.tvals,int(self.tvals/3))]

        #we take the selected slice of correlator, we sum connected and disconnected part and we look only at the imaginary part
        corr_3p = ( self.all_3pCorr[first_conf:last_conf:step_conf,0,:,:,:,:,:] + self.all_3pCorr[first_conf:last_conf:step_conf,1,:,:,:,:,:] ).imag

        #current nconf considered
        new_nconf = np.shape(corr_3p)[0]

        #we perform the noise average
        corr_3p_navg = corr_3p.mean(axis=-1).mean(axis=-1)

        #choice of binning: divisors of number of configurations
        deltaList = [delta for delta in divisors(new_nconf) if delta < new_nconf/10]

        #output info
        if verbose:
            print("\nMaking std analysis plots for each correlator...\n")


        #we now loop over the available binsize and we compute the std according to the jackknife method

        #we initialize the array where the std will be stored
        std_array = np.empty(shape=(len(deltaList),self.ncorr,self.noperators,self.tvals),dtype=float)

        #loop over the available binsizes
        for i,delta in enumerate(deltaList):
    
            #creation of list of elements to be deleted
            delete_list = [list(range(iconf,min(iconf+delta,new_nconf))) for iconf in range(0,new_nconf,delta)]

            #first the creation of the subsamples
            corr_3p_navg_resamp = np.asarray( [np.delete(corr_3p_navg, ith_delete ,axis=0) for ith_delete in delete_list] )

            #the number of resamples is
            nresamples = int(new_nconf/delta) # = np.shape(corr_3p_navg_resamp)[0]
    
            #we average over the gauge configurations
            corr_3p_navg_resamp_gavg = corr_3p_navg_resamp.mean(axis=1)
    

            #to do so we compute also the gauge averages on the whole dataset
            corr_3p_navg_gavg = corr_3p_navg.mean(axis=0)

            #then we compute estimate and bias with the jackknife, for each operator and for each time

            #the estimate is the average over the resamples
            estimate_biased = np.mean(corr_3p_navg_resamp_gavg,axis=0) #the mean is computed along the replicas axis

            #the bias is the following difference between the mean of the replicates and the mean on the whole dataset
            bias = (nresamples-1) * (estimate_biased-corr_3p_navg_gavg)

            #the std is given by the following formula (variance of replicates times n-1)
            std = np.sqrt( (nresamples-1)/nresamples * np.sum( (corr_3p_navg_resamp_gavg - estimate_biased)**2,axis=0 ) )

            #then we correct the estimate for the bias
            estimate = estimate_biased-bias

            #we store the std
            std_array[i] = std / np.abs(estimate)
        

        #now that we have all the data, we loop over all the correlators, and the operators and for each we make a plot

        #loop over the correlators
        for icorr in range(self.ncorr):

            #output info
            if verbose:
                print(f"\nMaking plots for the correlator number {icorr} ...\n")

            #loop over the operators
            for iop in tqdm(range(self.noperators)):

                #name of iop (enumerate is not used such that the loading bar is correctly visualized)
                op_name = self.op_names[iop]


                #create the figure for the given icorr and iop
                fig, ax_list = plt.subplots(nrows=len(times), ncols=1, sharex=True, sharey=False, figsize=(32, 14))

                #adjust subplot spacing
                plt.subplots_adjust(left=0.1,
                                    bottom=0.1, 
                                    right=0.87, 
                                    top=0.9, 
                                    wspace=0.4, 
                                    hspace=0.6)
    

                #for each time we have a different subplot
                for i,t in enumerate(times):

                    #plot the data
                    ax_list[i].plot(deltaList,std_array[:,icorr,iop,t],'-o',linewidth=0.1,color='blue')

                    #set title
                    ax_list[i].set_title(f"t = {t}",fontsize=15,weight="bold")

                    #set y label
                    ax_list[i].set_ylabel(r"$\sigma$ / $|\mu|$",rotation=90,labelpad=23,fontsize=18)

                    #set x ticks
                    ax_list[i].set_xticks(deltaList)
                    ax_list[i].tick_params(axis='both', which='major', labelsize=12)

                plt.xlabel(r"Binsize $\Delta$",fontsize=18,labelpad=23)

                plt.suptitle(r"Normalized Standard Deviation as a function of the binsize $\Delta$" + f' - Correlator {icorr} - Operator {op_name}', fontsize=23)


            
            

                #Display text box with frelevant parameters outside the plot
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                # place the text box in upper left in axes coords
                plt.text(1.01, 0.95, self.text_infobox[icorr], transform=ax_list[0].transAxes, fontsize=14,verticalalignment='top', bbox=props)
            
            
                #save figure
                if save:
                    fig_name = f"stdVSbin_corr{icorr}_{self.op_names_simple[iop]}_{self.run_name}.png"
                    plt.savefig(subdir+"/"+fig_name)


        #output info
        if verbose:
            print("\nAll plots done!\n")
        
        #if show is given open one png inside the dir
        if show==True:
        
            png_list = [f for f in listdir(subdir) if f.endswith('png') and isfile(join(subdir, f) )]
            for png in png_list[:1]:
                os.system("xdg-open "+subdir+'/'+png) 






########################## Main Class ###################################

#classes with all the data analysis tools for the run of the two point correlator (related to the file tm_mesons)
class run_2p:

    '''
    Create once class instance for a given run and analyze the
    results using the bult-in methods (for the run of the type tm_mesons)
    - accesible methods are: ...
    - accesible variables are: ...
    '''

    #global variables shared by all the runs (class instances)

    #conversion dictionaries
    noise_dict={0:"Z2",1:"Gauss",2:"U1",3:"One Component"}
    dirac_dict={0:"G0",1:"G1",2:"G2",3:"G3",5:"G5",6:"ONE",7:"G0G1",8:"G0G2",9:"G0G3",10:"G0G5",11:"G1G2",12:"G1G3",13:"G1G5",14:"G2G3",15:"G2G5",16:"G3G5"}

    latex_dirac_dict={0:r'$\gamma_0$',1:r'$\gamma_1$',2:r'$\gamma_2$',3:r'$\gamma_3$',5:r'$\gamma_5$',6:"1",7:r'$\gamma_0\gamma_1$',8:r'$\gamma_0\gamma_2$',
                      9:r'$\gamma_0\gamma_3$',10:r'$\gamma_0\gamma_5$',11:r'$\gamma_1\gamma_2$',12:r'$\gamma_1\gamma_3$',13:r'$\gamma_1\gamma_5$',
                      14:r'$\gamma_2\gamma_3$',15:r'$\gamma_2\gamma_5$',16:r'$\gamma_3\gamma_5$'}
    
    #name of dir for plots
    plot_base_dir = "plots/"

    #specifics of the data file

    #header is made up of 4 integers, 4x4=16byte
    header_size= 4*4

    #each correlator has an header of size given by 4x8 + 4x4 = 48byte
    corr_header_size = 4*8 + 4*4



    #standard constructor that requires the path to the run
    def __init__(self,filePath,corr_to_mu_ratio,verbose=True):

        #name of dir for plots
        name = filePath.split('/')[-1][:-4]
        self.run_name = name.split('.')[0]
        self.plot_dir=self.plot_base_dir+self.run_name

        #create plot directory if not already there
        Path(self.plot_dir).mkdir(parents=True, exist_ok=True)

        #store to variable how many different types of correlators there are
        self.corr_to_mu_ratio = corr_to_mu_ratio #this can be either 1,2 or 3

        if verbose:
            print("\nPlots and relevant info concerning the run will be stored in "+self.plot_dir)
        
        #initialization of relevant struct storing data
        self.conf_dict = {} #dict with the 2 points correlators (the keys are the configuration numbers)


        #to inizialize the instance of the class we read from the file passed from input

        ##### reading data from binary dat file #####
        with open(filePath, mode='rb') as file: # b is important -> binary

            #get all file content
            fileContent = file.read()


            ## read the header ##

            if verbose:
                print("\nReading the Header...\n")

            #first general information

            #reading: 4 int (4 byte each)
            self.ncorr, self.nnoise, self.tvals, self.noise_type = struct.unpack("iiii", fileContent[:self.header_size])

            #then the information regarding each correlator

            #initialization of correlators' variables
            self.k1=['']*self.ncorr
            self.k2=['']*self.ncorr
            self.mu1=['']*self.ncorr
            self.mu2=['']*self.ncorr
            self.type1=['']*self.ncorr
            self.type2=['']*self.ncorr
            self.x0=['']*self.ncorr
            self.isreal=['']*self.ncorr

            #reading: there are ncorr block, 8x8 + 4x4 (8 double and 4 int) with the following structure 
            for i in range(self.ncorr):
                self.k1[i], self.k2[i], self.mu1[i], self.mu2[i], self.type1[i], self.type2[i], self.x0[i], self.isreal[i] = struct.unpack("ddddiiii",fileContent[self.header_size+self.corr_header_size*i:self.header_size+self.corr_header_size*(i+1)])


            ## read the content of the file ##

            #first the initialization of the array with all the correlators (temporary arrays used to read the correlators for a given configuration)
            corr = np.empty(shape=(self.ncorr,self.tvals,self.nnoise),dtype=complex) #correlators


            #pointer to begin of first configuration (right after the header)
            first_conf = self.header_size+self.corr_header_size*self.ncorr


            #compute the size of the data chunk for one configuration
            #          sizeof(int) (=confNumber)           ncorr * nnoise * tvals * 2 (re+im) * 8 (sizeof(double))  --> (the term with isreal is there because for real correlators only the real part gets stored)
            conf_len = 4                                 +  self.nnoise * self.tvals * (2*self.ncorr - np.sum(self.isreal))  * 8 

            #we now start with the actual reading

            if verbose:
                print("Reading the data for each configuration...\n")

            #starting right after the header we read each configuration block
            for start_conf in tqdm(range(first_conf, len(fileContent), conf_len)):

                #security check for run that were killed before finishing:
                #breaks loop if the file does not contain the whole correlator for the given configuration
                if start_conf+conf_len > len(fileContent) :
                    break

                #initialize to 0 the array with the correlators
                corr.fill(complex(0,0))

                #reading of the configuration number
                conf_number = struct.unpack("i",fileContent[start_conf:start_conf+4])[0]

                #we set the pointer to the data we have to read right after the configuration number
                start_reading = start_conf+4

                #then we start to read the data associated with the given configuration

                #loop over the correlators                         (for each conf we have ncorr blocks of...) 
                for ic in range(self.ncorr):

                    #loop over time values                             (...tvals blocks of...)
                    for t in range(self.tvals):

                        #loop over noise vectors                          (...nnoise blocks of...)
                        for inoise in range(self.nnoise):

                            #reading of re and im part of 2point correlators   (...complex numbers)

                            #the reading has to be done differently depending whether the correlator is real or not

                            #complex correlator case: we read both the real and the imaginary part
                            if self.isreal[ic]==0:
                                re,im = struct.unpack("dd",fileContent[start_reading:start_reading+16])
                                #store them
                                corr[ic][t][inoise] = complex(re,im)
                                #update start reading
                                start_reading = start_reading+16

                            #real correlator case: we read only the real part and the imaginary one we set to 0
                            if self.isreal[ic]==1:
                                #read re
                                re = struct.unpack("d",fileContent[start_reading:start_reading+8])[0]
                                #store it
                                corr[ic][t][inoise] = complex(re,0.0)
                                #update start reading
                                start_reading = start_reading+8

                                                                

                #once we read all the data of the given configuration we store it in a dictionary
                self.conf_dict[str(conf_number)] = corr.copy()


        #data file completely read
        #we now initialize the relevant arrays that store the data in the class

        if verbose:
            print("\nInitializing the data arrays...\n")

        #construction of array with names and numbers of configurations
        self.conf_names = list(self.conf_dict.keys()) #(we skip the gauge transformed configuration if there are)
        self.nconf = len(self.conf_names)

        #creation of a numpy array with all the correlators
        self.all_corr = np.empty(shape=(self.nconf,self.ncorr,self.tvals,self.nnoise),dtype=complex)
        for iconf,nameconf in enumerate(self.conf_names):
            self.all_corr[iconf] = self.conf_dict[nameconf]


        #creation of the info box that will be displayed on all plots
        self.text_infobox = []
        for icorr in range(self.ncorr): #a different infobox for each correlator
            self.text_infobox.append( '\n'.join((
                 'Correlator %d parameters:' % (icorr),
                 '           ',
                r'$k_1$=%.9f ' % (self.k1[icorr] ),
                r'$k_2$=%.9f ' % (self.k2[icorr] ),
                    '           ',
                r'$\mu_1$=%.9f ' % (self.mu1[icorr] ),
                r'$\mu_2$=%.9f ' % (self.mu2[icorr] ),
                    '           ',
                r'$\Gamma_1$=' + self.latex_dirac_dict[self.type1[icorr]],
                r'$\Gamma_2$=' + self.latex_dirac_dict[self.type2[icorr]],
                    '           ',
                r'$x_0$=%d' % self.x0[icorr],
                '           ',
                r'$isreal$=%d' % self.isreal[icorr],
                 '           ',
                 '           ',
                 '           ',
                 'Simulation parameters:',
                 '           ',
                r'$N_{NOISE}$=%d' % self.nnoise,
                 'Noise Type=%s' % self.noise_dict[self.noise_type],
                r'$T$=%d' % self.tvals,
                 '           ',
                 '           ',
                 '           ',
                 'Configurations:',
                 '           ',
                r'$N_{CONF}$=%d' % self.nconf,)) )


        #we also initilize the array that will be storing the final result about the masses (squared) of the two point functions
        #..determined with the effective formula
        #self.m2_eff = np.zeros(shape=(self.ncorr),dtype=float)
        #self.m2_eff_std = np.zeros(shape=(self.ncorr),dtype=float)
        #..determined from the fit
        #self.m2_fit = np.zeros(shape=(self.ncorr),dtype=float)
        #self.m2_fit_std = np.zeros(shape=(self.ncorr),dtype=float)


        #varialbe storing the value of the true masses
        self.true_masses = np.asarray([0.2150,0.2449,0.3401,0.4182,0.4873,0.5512,0.1811])
        self.mus = np.array(self.mu1[::corr_to_mu_ratio])


        #initialization completed
        if verbose:
            print("Initialization complete\n")



    #utility function that prints all the info of the run
    def print_info(self):
        
        #Header print
        print("\n[File Header]\n")
        print(f"- ncorr           = {self.ncorr}\n")
        print(f"- nnoise          = {self.nnoise}\n")
        print(f"- tvals           = {self.tvals}\n")
        print(f"- noise_type      = {self.noise_dict[self.noise_type]}\n")

        #configurations details print
        print("\n[Configurations]\n")
        print(f"- nconf               = {self.nconf}\n\n")
        #print(f"- conf_step           = {confSTEP}\n")

        #Correlators Header print
        for i in range(self.ncorr):
            print(f"[Correlator {i}]\n")
            print(f" - k1 = {self.k1[i]}\n")
            print(f" - k2 = {self.k2[i]}\n\n")
            print(f" - mu1 = {self.mu1[i]}\n")
            print(f" - mu2 = {self.mu2[i]}\n\n")
            print(f" - type1 = {self.dirac_dict[self.type1[i]]}\n")
            print(f" - type2 = {self.dirac_dict[self.type2[i]]}\n\n")
            print(f" - x0 = {self.x0[i]}\n\n")
            print(f" - isreal = {self.isreal[i]}\n\n\n")



    #function used to extract the mass of the correlator ad to make fancy plots
    def mass_extraction(self,first_conf=0,last_conf=None,binsize=1,
                        l_cut=0,r_cut=0,max_chi2=1.0,zoom_out=0,
                        verbose=True,show=True,save=True,result_save=True,subdir_name="mass_extraction"):

        #0) initialization of variables used in the function

        #creation of subdir where to save plots
        subdir = self.plot_dir+"/"+subdir_name
        Path(subdir).mkdir(parents=True, exist_ok=True)

        #by default the last_configuration considered is the nconf-th one
        if last_conf is None:
            last_conf = self.nconf

        #check that the parameters 
        if (last_conf-first_conf)%binsize != 0:
            print("\nlast_conf-first_conf should be a multiple integer of binsize!\n")
            return
        
        #time array for the plot
        times=np.arange(self.tvals)


        #1) we adjust the dimension depending on how many different correlator there are (just π+-, or also 1 or 2 π0)
        

        #we use here the actual number of correlators depending on how many different types there are
        if self.corr_to_mu_ratio == 3:
            ncorr = int(self.ncorr/3 * 2)
        elif self.corr_to_mu_ratio == 1:
            ncorr = self.ncorr

        #creation of an array with all the correlators
        correlators = np.empty(shape=(self.nconf,ncorr,self.tvals,self.nnoise),dtype=complex)

        #in the case we have three correlators for each value of mu we sum the π0 correlators together
        if self.corr_to_mu_ratio == 3:
            for icorr in range(ncorr):
                if icorr%2==0:
                    correlators[:,icorr] =  self.all_corr[:,int(icorr*3/2)]
                else:
                    correlators[:,icorr] = self.all_corr[:,int(icorr*3/2)] + self.all_corr[:,int(icorr*3/2)+1]


        #2) we cast the correlators to real
        correlators = correlators.real


        #3) we perform the noise average
        corr_navg = correlators.mean(axis=-1)


        #output info
        if verbose:
            print(f"\nExtracting Masses using a Jackknife Analysis ...\n")


        #4) we now implement the jackknife technique for the effective mass

        #output info
            if verbose:
                print(f"\n... first with the effective formula ...\n")

        #4.1) first the creation of the resamples
        corr_navg_resamp = np.asarray( [np.delete(corr_navg, list(range(iconf,min(iconf+binsize,last_conf))) ,axis=0) for iconf in range(first_conf,last_conf,binsize)] )

        #the number of resamples is
        nresamples = int((last_conf-first_conf)/binsize)

        #4.2) with each resample we compute the mass using the effective formula and the fit

        #we average over the gauge configurations
        corr_navg_resamp_gavg = corr_navg_resamp.mean(axis=1)

        #maximum number of iterations used in the effective formula
        chosen_maxit=2000

        #with these arrays we can compute the replica of the mass for each subsample...
        mass_eff_replicas = np.empty(shape=(nresamples,ncorr,self.tvals-1),dtype=float)
        #...with the following loop
        #then we estimate the mass
        for ires in range(nresamples): #for each resample
            for icorr in range(ncorr): #for each correlator
                
                for t in range(self.tvals-1): #and for each time
                    #we compute the mass using the effective formula
                    mass_eff_replicas[ires,icorr,t] = self.eff_mass_func( corr_navg_resamp_gavg[ires,icorr,t] /  corr_navg_resamp_gavg[ires,icorr,t+1] , t,max_it=chosen_maxit)

        #4.3) mass estimation on the whole dataset

        #gauge average of whole dateset
        corr_navg_gavg = corr_navg.mean(axis=0)

        #we initialize the eff mass array computed from the whole dataset
        mass_eff_total = np.empty(shape=(ncorr,self.tvals-1),dtype=float)
        #mass_fit_total = np.empty(shape=(ncorr,self.tvals),dtype=float)
        #..and we compute it with the following loop
        for icorr in range(ncorr): #for each correlator
            for t in range(self.tvals-1): #and for each time
                #we use the formula of the matrix element
                    mass_eff_total[icorr,t] = self.eff_mass_func( corr_navg_gavg[icorr,t] /  corr_navg_gavg[icorr,t+1] , t,max_it=chosen_maxit)

        #4.5) then we compute the estimate, the bias and the std according to the jackknife method

        #the estimate is the average over the resamples
        mass_eff_estimate = np.mean(mass_eff_replicas,axis=0)
        #mass_fit_estimate = np.mean(mass_eff_replicas,axis=0)

        #the bias is the following difference between the mean of the replicates and the mean on the whole dataset
        bias_eff = (nresamples-1) * (mass_eff_estimate-mass_eff_total)
        #bias_fit = (nresamples-1) * (mass_fit_estimate-mass_fit_total)

        #the std is given by the following formula (variance of replicates times n-1)
        mass_eff_std = np.sqrt( (nresamples-1)/nresamples * np.sum( (mass_eff_replicas - mass_eff_estimate)**2,axis=0 ) )
        #mass_fit_std = np.sqrt( (nresamples-1)/nresamples * np.sum( (mass_fit_replicas - mass_fit_estimate)**2,axis=0 ) )
        
        #then we correct the estimate for the bias
        mass_eff = mass_eff_estimate-bias_eff
        #mass_fit = mass_fit_estimate-bias_fit

        #compute the covariance matrix
        cov_mat_eff = np.empty(shape=(ncorr,self.tvals-1,self.tvals-1),dtype=float)
        #cov_mat_fit = np.empty(shape=(ncorr,self.tvals,self.tvals),dtype=float)
        for t1 in range(self.tvals-1):
            for t2 in range(self.tvals-1):
                cov_mat_eff[:,t1,t2] = (nresamples-1)/nresamples * np.sum( (mass_eff_replicas[:,:,t1] - mass_eff_estimate[:,t1]) * (mass_eff_replicas[:,:,t2] - mass_eff_estimate[:,t2]),axis=0 )
                #cov_mat_eff[:,:,t1,t2] = (nresamples-1)/nresamples * np.sum( (mass_fit_replicas[:,:,:,t1] - mass_fit_estimate[:,:,t1]) * (mass_fit_replicas[:,:,:,t2] - mass_fit_estimate[:,:,t2]),axis=0 )

        
        #5) now that we have a value of effective mass for each time we compute the plateaux

        #we initialize the array we use to store the range of the plateaux (used also later during the fit)
        chosen_cuts = np.empty(shape=(ncorr),dtype=int)

        #we also initialize the array with the masses
        m_eff = np.empty(shape=(ncorr),dtype=float)
        m_eff_std = np.empty(shape=(ncorr),dtype=float)


        #we loop over the correlators and for each we determine the plateaux
        for icorr in range(ncorr):

            #at the iteration 0 the plataux is the smallest possible
            #chosen_cut = int(self.tvals/2)-1

            #we loop over the possible cuts and see the biggest whose chi2 is ok
            for icut in range(1,int(self.tvals/2)):
                
                #value of the plateaux with the given cut (plus left and right offsets)
                plateaux_value = np.mean(mass_eff[icorr,icut+l_cut:-icut-r_cut])

                #if the reduced chi2 is smaller than the reference value (~1) we select the cut for the plateaux
                if reduced_cov_chi2(mass_eff[:,icut+l_cut:-icut-r_cut],cov_mat_eff[:,icut+l_cut:-icut-r_cut,icut+l_cut:-icut-r_cut],axis=1)[icorr] < max_chi2:
                #if chi2(mass_eff[icorr,icut+l_cut:-icut-r_cut],mass_eff_std[icorr,icut+l_cut:-icut-r_cut],axis=0) < max_chi2*np.shape(mass_eff[icorr,icut+l_cut:-icut-r_cut])[0]:
                    chosen_cut = icut
                    chosen_cuts[icorr] = chosen_cut
                    break

            #once the plataux has been found we store the plateaux value
            m_eff[icorr] = np.mean(mass_eff[icorr,chosen_cut+l_cut:-chosen_cut-r_cut])
            #print(mass_eff[icorr])
            #print(f"{icorr}  {chosen_cut}    {m_eff[icorr]}\n")
            m_eff_std[icorr] = np.sqrt( np.mean( mass_eff_std[icorr,chosen_cut+l_cut:-chosen_cut-r_cut]**2 ) )


        #6) now that we have determined the plateaux range we repeat the jacknife analysis with the for the fit mass

        #output info
        if verbose:
            print(f"\n... then from the fit to the sinh function ...\n")

        #6.1) we compute the fit mass (and ampitude) for each replica

        #mass replica array initialization
        mass_fit_replicas = np.empty(shape=(nresamples,ncorr),dtype=float)
        amp_fit_replicas = np.empty(shape=(nresamples,ncorr),dtype=float) #to plot the correlator obtained from the fit we have to perform the same analysis on the amplitude also

        #for each correlator
        for icorr in range(ncorr):
                
            #preparation for fit
            guess_mass=self.true_masses[int(icorr/2)]
            cut=chosen_cuts[icorr]
            fit_times=times[cut+l_cut:-cut-r_cut]

            #for each replica
            for ires in range(nresamples):

                #fit
                guess_amp=corr_navg_resamp_gavg[ires,icorr,int(self.tvals/2)] / np.sinh( (self.tvals-1-self.tvals/2) * guess_mass )
                guess=[guess_amp,guess_mass]
                fit_data=corr_navg_resamp_gavg[ires,icorr,cut+l_cut:-cut-r_cut]
                popt_x,pcov_x = curve_fit(self.fit_sinh_x, fit_times, fit_data, p0=guess,maxfev = 1300)
                mass_fit_replicas[ires,icorr] = popt_x[1]
                amp_fit_replicas[ires,icorr] = popt_x[0]

        #6.2) we compute fit mass and amplitude on the whole dataset


        #we initialize the fit mass array computed from the whole dataset
        mass_fit_total = np.empty(shape=(ncorr),dtype=float)
        amp_fit_total = np.empty(shape=(ncorr),dtype=float)
        #..and we compute it with the following loop
        for icorr in range(ncorr): #for each correlator

            #preparation for fit
            guess_mass=self.true_masses[int(icorr/2)]
            cut=chosen_cuts[icorr]
            fit_times=times[cut+l_cut:-cut-r_cut]
            guess_amp=corr_navg_gavg[icorr,int(self.tvals/2)] / np.sinh( (self.tvals-1-self.tvals/2) * guess_mass )
            guess=[guess_amp,guess_mass]
            fit_data=corr_navg_gavg[icorr,cut+l_cut:-cut-r_cut]

            #fit
            popt_x,pcov_x = curve_fit(self.fit_sinh_x, fit_times, fit_data, p0=guess,maxfev = 1300)
            mass_fit_total[icorr] = popt_x[1]
            amp_fit_total[icorr] = popt_x[0]


        #6.3) then we compute the estimate, the bias and the std according to the jackknife method

        #the estimate is the average over the resamples
        mass_fit_estimate = np.mean(mass_fit_replicas,axis=0)
        amp_fit_estimate = np.mean(amp_fit_replicas,axis=0)

        #the bias is the following difference between the mean of the replicates and the mean on the whole dataset
        bias_fit = (nresamples-1) * (mass_fit_estimate-mass_fit_total)
        bias_fit_amp = (nresamples-1) * (amp_fit_estimate-amp_fit_total)


        #the std is given by the following formula (variance of replicates times n-1)
        mass_fit_std = np.sqrt( (nresamples-1)/nresamples * np.sum( (mass_fit_replicas - mass_fit_estimate)**2,axis=0 ) )
        amp_fit_std = np.sqrt( (nresamples-1)/nresamples * np.sum( (amp_fit_replicas - amp_fit_estimate)**2,axis=0 ) )

        #then we correct the estimate for the bias
        mass_fit = mass_fit_estimate-bias_fit
        amp_fit = amp_fit_estimate-bias_fit_amp

        #7) we do the plots of the plateaux determination

        #output info
        if verbose:
            print(f"\nMaking the plots concerning the plataux determination ...\n")

        #loop over the correlators
        for icorr in tqdm(range(0,ncorr,2)):
                
            #create figure and axis
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(26, 11.5))


            #we plot the hlines of the plateaux for the eff mass for the non twisted corr
            cut = chosen_cuts[icorr]
            ax.hlines(m_eff[icorr],cut+l_cut,self.tvals-1-cut-r_cut,color='blue',linewidth=4,alpha=0.8)
            ax.hlines(m_eff[icorr]+m_eff_std[icorr],cut+l_cut,self.tvals-1-cut-r_cut,linestyle='dashed',linewidth=3,color='blue',alpha=0.5)
            ax.hlines(m_eff[icorr]-m_eff_std[icorr],cut+l_cut,self.tvals-1-cut-r_cut,linestyle='dashed',linewidth=3,color='blue',alpha=0.5)

            #we plot the hlines of the plateaux for the eff mass for the twisted corr
            cut = chosen_cuts[icorr+1]
            ax.hlines(m_eff[icorr+1],cut+l_cut,self.tvals-1-cut-r_cut,color='red',linewidth=4,alpha=0.8)
            ax.hlines(m_eff[icorr+1]+m_eff_std[icorr+1],cut+l_cut,self.tvals-1-cut-r_cut,linestyle='dashed',linewidth=3,color='red',alpha=0.5)
            ax.hlines(m_eff[icorr+1]-m_eff_std[icorr+1],cut+l_cut,self.tvals-1-cut-r_cut,linestyle='dashed',linewidth=3,color='red',alpha=0.5)

                



            #now we plot the errorbar with all the other data

            #now the cut can be smaller
            cut = np.min(np.array(chosen_cuts[icorr],chosen_cuts[icorr+1]))-zoom_out

            #times on the x axis
            mass_times = times[cut+l_cut:-cut-1-r_cut]+0.5

        
            #we plot the mass for the non twisted corr
            ax.errorbar(mass_times,mass_eff[icorr,cut+l_cut:-cut-r_cut],yerr=mass_eff_std[icorr,cut+l_cut:-cut-r_cut],
                             marker='o',linestyle='--',color='blue',markersize=10,linewidth=0.6,alpha=0.8,elinewidth=2,label=r'$m_{eff}$')

            #we plot the mass for the twisted corr
            ax.errorbar(mass_times,mass_eff[icorr+1,cut+l_cut:-cut-r_cut],yerr=mass_eff_std[icorr+1,cut+l_cut:-cut-r_cut],
                             marker='o',linestyle='--',color='red',markersize=10,linewidth=0.6,alpha=0.8,elinewidth=2,label=r'$\widetilde{m}_{eff}$')


            #enable grid
            ax.grid()

            #set y label
            ax.set_ylabel('Mass',rotation=90,labelpad=20,fontsize=16)

            #set legend
            ax.legend(fontsize=16)


            #adjust subplot spacing
            plt.subplots_adjust(left=0.04,
                                bottom=0.05, 
                                right=0.9, 
                                top=0.9, 
                                wspace=0.4, 
                                hspace=0.6)

            #set x label
            plt.xlabel('Time [lattice units]',fontsize=16)

            #set title
            plt.suptitle(f' Effective Mass - Correlator {icorr}', fontsize=25,y=0.98)

            #Display text box with frelevant parameters outside the plot
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place the text box in upper left in axes coords
            plt.text(1.01, 0.95, self.text_infobox[int(icorr*3/2)], transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)

            #save figure
            if save:
                fig_name = f"plot_massplateaux_corr{icorr}_{self.run_name}.png"
                plt.savefig(subdir+"/"+fig_name)


        #9) before making the plot with the fit we compute the correlators with the jackknife analysis

        #the estimate is the mean of the replicates
        corr_estimate = np.mean(corr_navg_resamp_gavg,axis=0) #(the replicate axis)
        #the bias is the following difference between the mean of the replicates and the mean on the whole dataset
        corr_bias = (nresamples-1) * (corr_estimate-corr_navg_gavg)
        #the std is given by the following formula (variance of replicates times n-1)
        corr_std = np.array( [np.sqrt( (nresamples-1)/nresamples * np.sum( (corr_navg_resamp_gavg[:,icorr] - corr_estimate[icorr])**2,axis=0 ) ) for icorr in range(ncorr)]   ) 
        #then we correct the estimate for the bias
        corr_estimate_biascorr = corr_estimate-corr_bias



        #9) we do the plots of the fit

        #output info
        if verbose:
            print(f"\nMaking the plots concerning the plataux determination ...\n")

        #loop over the correlators
        for icorr in tqdm(range(ncorr)):

            #cut to use
            cut = chosen_cuts[icorr]
            
            #data used in the fit
            fit_data = corr_estimate_biascorr[icorr,cut+l_cut:-cut-r_cut]
            fit_std = corr_std[icorr,cut+l_cut:-cut-r_cut]
            fit_result = self.fit_sinh_x(times[cut+l_cut:-cut-r_cut],amp_fit[icorr],mass_fit[icorr])
            #fit result
            npoints = 100 #number of points drawn in the fit
            fit_times = np.linspace(cut+l_cut,self.tvals-cut-r_cut,npoints)
            fit_plot = self.fit_sinh_x(fit_times,amp_fit[icorr],mass_fit[icorr])

            #create figure and axis
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(26, 11.5))

            #plot the fit
            ax.plot(fit_times,fit_plot,color='red',linewidth=4,alpha=0.8,label=r"fit ($\chi^2 = $ "+f"{round(chi2_fit(fit_result,fit_data,fit_std),2)})")

            #plot the correlator
            cut = chosen_cuts[icorr] - zoom_out #(include zoom out in this plot)
            ax.errorbar(times[cut+l_cut:-cut-r_cut],corr_estimate_biascorr[icorr,cut+l_cut:-cut-r_cut],yerr=corr_std[icorr,cut+l_cut:-cut-r_cut],
                             marker='o',linestyle='',color='blue',markersize=10,linewidth=0.6,alpha=0.8,elinewidth=2,label="Data")
            
            #enable grid
            ax.grid()

            #set y label
            ax.set_ylabel('G(t)',rotation=90,labelpad=20,fontsize=16)

            #set legend
            ax.legend(fontsize=16)


            #adjust subplot spacing
            plt.subplots_adjust(left=0.04,
                                bottom=0.05, 
                                right=0.9, 
                                top=0.9, 
                                wspace=0.4, 
                                hspace=0.6)

            #set x label
            plt.xlabel('Time [lattice units]',fontsize=16)

            #set title
            plt.suptitle(f' Correlator Sinh Fit - Correlator {icorr}', fontsize=25,y=0.98)

            #Display text box with frelevant parameters outside the plot
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place the text box in upper left in axes coords
            plt.text(1.01, 0.95, self.text_infobox[int(icorr*3/2)], transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)

            #save figure
            if save:
                fig_name = f"plot_fit_corr{icorr}_{self.run_name}.png"
                plt.savefig(subdir+"/"+fig_name)


        #10) now the last plot, the mass squared vs mu scaling

        #output info
        if verbose:
            print(f"\nMaking the plot of the mass squared vs mu scaling ...\n")

        #10.1) error propagation to mass squared

        mass2_eff = m_eff**2
        mass2_eff_std = 2 * np.abs(m_eff)*m_eff_std

        mass2_fit = mass_fit**2
        mass2_fit_std = 2 * np.abs(mass_fit)*mass_fit_std

        m2_true = self.true_masses**2

        #10.2) then the m-mtilda division

        m2_eff = mass2_eff[::2]
        m2_eff_std = mass2_eff_std[::2]

        m2t_eff = mass2_eff[1::2]
        m2t_eff_std = mass2_eff_std[1::2]

        m2_fit = mass2_fit[::2]
        m2_fit_std = mass2_fit_std[::2]

        m2t_fit = mass2_fit[1::2]
        m2t_fit_std = mass2_fit_std[1::2]

        #10.3) linear fit

        guess_par = [ ( m2_eff[-1]-m2_eff[0])/(self.mus[-1]-self.mus[0]) , 0.0]
        popt,pcov = curve_fit(fit_lin, self.mus, m2_eff, sigma= m2_eff_std, p0=guess_par)
        M2_eff_linfit = fit_lin(np.linspace(0,np.max(self.mu1)*1.1,100),*popt)
        result_string = "\nM2 eff"
        result_string += str(popt)
        result_string += str(np.sqrt(np.diag(pcov)))

        guess_par = [ ( m2t_eff[-1]-m2t_eff[0])/(self.mus[-1]-self.mus[0]) , 0.0]
        popt,pcov = curve_fit(fit_lin, self.mus, m2t_eff, sigma= m2t_eff_std, p0=guess_par)
        M2t_eff_linfit = fit_lin(np.linspace(0,np.max(self.mu1)*1.1,100),*popt)
        result_string += "\nM2t eff"
        result_string += str(popt)
        result_string += str(np.sqrt(np.diag(pcov)))


        guess_par = [ ( m2_fit[-1]-m2_fit[0])/(self.mus[-1]-self.mus[0]) , 0.0]
        popt,pcov = curve_fit(fit_lin, self.mus, m2_fit, sigma= m2_fit_std, p0=guess_par)
        M2_fit_linfit = fit_lin(np.linspace(0,np.max(self.mu1)*1.1,100),*popt)
        result_string += "\nM2 fit"
        result_string += str(popt)
        result_string += str(np.sqrt(np.diag(pcov)))

        guess_par = [ ( m2t_fit[-1]-m2t_fit[0])/(self.mus[-1]-self.mus[0]) , 0.0]
        popt,pcov = curve_fit(fit_lin, self.mus, m2t_fit, sigma= m2t_fit_std, p0=guess_par)
        M2t_fit_linfit = fit_lin(np.linspace(0,np.max(self.mu1)*1.1,100),*popt)
        result_string += "\nM2t fit"
        result_string += str(popt)
        result_string += str(np.sqrt(np.diag(pcov)))


        guess_par = [ ( m2_true[-1]-m2_true[0])/(self.mus[-1]-self.mus[0]) , 0.0]
        popt,pcov = curve_fit(fit_lin, self.mus, m2_true, p0=guess_par)
        fit_reference = fit_lin(np.linspace(0,np.max(self.mu1)*1.1,100),*popt)
        result_string += "\nM2 true"
        result_string += str(popt)
        result_string += str(np.sqrt(np.diag(pcov)))

        #10.4) we do the actual plot

        #create figure and axis
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(23, 11.5))

        #range of the plot of linfit
        mu_range = np.linspace(0,np.max(self.mu1)*1.1,100)

        #m2 and m2t from effective formula
        ax.errorbar(self.mus,m2_eff,yerr=m2_eff_std,marker='o',linewidth=0,elinewidth=6.0,label=r"$m^2$ (effective mass)",color='royalblue',alpha=0.6,markersize=10)
        ax.plot(mu_range,M2_eff_linfit,linewidth =2.7,linestyle='dashed',color='royalblue',alpha=0.6)
        ax.errorbar(self.mus,m2t_eff,yerr=m2t_eff_std,marker='s',linewidth=0,elinewidth=6.0,label=r"$\widetilde{m^2}$ (effective mass)",color='darkblue',alpha=0.6,markersize=10)
        ax.plot(mu_range,M2t_eff_linfit,linewidth =2.7,linestyle='dashed',color='blue',alpha=0.6)

        #m2 and m2t from fit
        ax.errorbar(self.mus,m2_fit,m2_fit_std,marker='o',linewidth=0,elinewidth=6.0,label=r"$m^2$ (from sinh fit)",color='limegreen',alpha=0.6,markersize=10)
        ax.plot(mu_range,M2_fit_linfit,linewidth =2.7,linestyle='dashed',color='limegreen',alpha=0.6)
        ax.errorbar(self.mus,m2t_fit,m2t_fit_std,marker='s',linewidth=0,elinewidth=6.0,label=r"$\widetilde{m^2}$ (from sinh fit)",color='darkgreen',alpha=0.6,markersize=10)
        ax.plot(mu_range,M2t_fit_linfit,linewidth =2.7,linestyle='dashed',color='darkgreen',alpha=0.6)

        ax.plot(self.mus,m2_true,marker='o',linewidth=0,label=r"$m^2$ from reference",color='red',alpha=0.6,markersize=4)
        ax.plot(mu_range,fit_reference,linewidth =2.7,linestyle='dashed',color='red',alpha=0.6)

         #enable grid
        ax.grid()

        #set y label
        ax.set_ylabel(r"$m^2$",rotation=90,labelpad=20,fontsize=16)
        
        #set legend
        ax.legend(fontsize=16)
        
        #adjust subplot spacing
        plt.subplots_adjust(left=0.04,
                            bottom=0.05, 
                            right=0.9, 
                            top=0.9, 
                            wspace=0.4, 
                            hspace=0.6)
        
        #set x label
        plt.xlabel(r'$\mu$ [lattice units]',fontsize=16)
        
        #set title
        plt.suptitle(r"$m^2$ vs $\mu$ Scaling", fontsize=25,y=0.98)
        
        #save figure
        if save:
            fig_name = f"plot_m2scaling_{self.run_name}.png"
            plt.savefig(subdir+"/"+fig_name)

        


        #output info
        if verbose:
            print("\nAll plots done!\n")
            print("\nThe final result for the matrix elements are (slope-offset, value and then std):\n")
            print(result_string)


        #if show is given open one png inside the dir
        if show==True:
            png_list = [f for f in listdir(subdir) if f.endswith('png') and isfile(join(subdir, f) )]
            for png in png_list[:1]:
                os.system("xdg-open "+subdir+'/'+png)


        #the final result of the matrix element gets store in a txt file
        if result_save:
            with open(self.plot_dir+"mass_result.txt","w") as file:
                #first we write and header explaining how to read the file
                file.write("#The file is structured as follows:\n")
                file.write("#- name of the mass\n")
                file.write("#- [slope, offset]\n")
                file.write("#- [slop_std, offsets_std]\n\n")
                file.write(result_string)
             


            







    #auxiliary functions used  in the extraction of the mass

    #function used to fit the correlators
    def fit_sinh_x(self,t,amp,mass):
        T=self.tvals-1 #tvals-1 because the direction of the lattice range from 0 to tvals-1
        return amp*np.sinh(mass*(T-t))
    
    def eff_mass_func(self,corr_ratio,t,max_it=150,verb=False):

        #resolution parameter for an early stopping
        eps=3e-7

        #maximum time in the lattice
        T = self.tvals-1 #tvals-1 because the direction of the lattice range from 0 to tvals-1

        #starting value of the mass 
        m0 = np.log(corr_ratio)

        #value of the mass after 0 iterations
        massa = m0 #at iteration 0 the sinh is an exp and the mass is simply the log of the ration of correlators

        #early stopping if ratio of correlators is negative
        if corr_ratio <1.0:
            massa = 0
            return massa

        #loop where the iterative effective formula is implemented
        for it in range(max_it):
            
            #computes new mass from previous one
            
            den = 1 - np.exp( -2*massa * ( T-t  ) )
            num = 1 - np.exp( -2*massa * ( T-t-1) )
        
            new_massa = np.log( corr_ratio * num/den )

            
            #print some output if verbose
            if it%5 == 0 and verb==True:
                print(f"x t{t} it{it} delta = {np.abs((new_massa-massa)/massa)} \n")
        

            #early stopping if the mass is changing no more
            if np.abs(massa-new_massa)/massa < eps:
                massa=new_massa
                break
        

            massa = new_massa

        #return the value of the mass obtained from the effective iterative formula
        return massa







#auxiliary function to compute the chi2 of a fit
def chi2(array,std_array,axis):
    avg = np.mean(array,axis=axis,keepdims=True)
    return np.sum( ((array-avg)/std_array)**2 , axis=axis)


def chi2_fit(result,data,data_std):
    return np.sum( ((result-data)/data_std)**2 )/len(data_std)


#auxiliary function to compute chi2 of a fit taking into account the covariant matrix
def reduced_cov_chi2(array,cov_array,axis):
    cov_inv = np.linalg.inv(cov_array)
    avg = np.mean(array,axis=axis,keepdims=True)
    deltas = array-avg

    plateau_T = np.shape(array)[axis]

    return np.einsum( 'ij,ij->i' , deltas, np.einsum('ijk,ik->ij',cov_inv,deltas) ) / plateau_T


#function used to fit to a linear function
def fit_lin(x,m,q):
    return m * x + q


#auxiliary function to compute chi2 of a fit taking into account the covariant matrix (without the operator dimension)
def reduced_cov_chi2_single(array,cov_array,axis):
    cov_inv = np.linalg.inv(cov_array)
    avg = np.mean(array,axis=axis,keepdims=True)
    deltas = array-avg

    plateau_T = np.shape(array)[axis]

    return np.einsum( 'j,j->' , deltas, np.einsum('jk,k->j',cov_inv,deltas) ) / plateau_T


#Print a result in a nice format
def present_result(name,mean,sigma,digits,unit):
    #(digits is the number of digits we want after the point in the sigma)
    #mantissa here is not actually the mantissa, is the int part plus the mantissa

    #we extract the mantissa and the power of the exponential from he mean and the std
    mean_mant = float(str(mean).split('e')[0])
    mean_pow = int(str(mean).split('e')[1])
    sigma_mant = float(str(sigma).split('e')[0])
    sigma_pow = int(str(sigma).split('e')[1])

    #we compute the difference in the powers of ten
    diff_pow = mean_pow-sigma_pow

    #then we compute the mean and the sigma to be printed
    mean_print = round(mean_mant,digits+diff_pow)
    sigma_print = round(sigma_mant* 10**(-diff_pow),digits+diff_pow)
    
    #we return the string with mean +- std, unit of measure and relative error in %
    return name + " ( {0} +- {1} ) 10^({2}) {3} [{4:.2f}%]".format(mean_print,sigma_print,mean_pow,unit,sigma/mean*100)