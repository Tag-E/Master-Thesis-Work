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

                #initialize to 0 the array with the correlators
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
    def std_study(self,first_conf=0,last_conf=None,step_conf=1,
                  show=False,save=True,verbose=True,subdir_name="stdAnalysis_plots"):

        #creation of subdir where to save plots
        subdir = self.plot_dir+"/" + subdir_name
        Path(subdir).mkdir(parents=True, exist_ok=True)

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
        for icorr in tqdm(range(self.ncorr)):

            #we initialize the array in which we store the normalized std

            #array with a value of normalized std for each delta
            std_list_jack = []
            std_list_boot = []     
            std_list_simple = []

            #loop over the possible deltas (size of deleted elements)
            for delta in deltaList:

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
                std_array_simple = np.std(corr_binned,axis=0) /np.sqrt(np.shape(corr_binned)[0]-1)

                #we now append to the std list the mean relative error
                std_list_simple.append(np.mean( (std_array_simple/np.abs(mean_array_simple))[:,1:-2] ))

            #now that we have all the std for each size of the bin (delta) we can make the plot of std vs delta

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
        corr = self.all_3pCorr[:,:,:,:,first_time:last_time] #conf - piece - corr - op - tval - noise - noise

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