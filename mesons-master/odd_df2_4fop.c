/*******************************************************************************
*
* File odd_df2_4fop.c
*
* Copyright (C) 2024 Emilio Taggi
*
* Based on tm_mesons
* Copyright (C) 2016 David Preti
*
* Based on mesons 
* Copyright (C) 2013, 2014 Tomasz Korzec
*
* Based on openQCD, ms1 and ms4 
* Copyright (C) 2012 Martin Luescher and Stefan Schaefer
*
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*******************************************************************************
*
* Computation of quark propagators
*
* Syntax: odd_df2_4fop -i <input file> [-noexp] [-a]
*
* For usage instructions see the file README.deltaF2_4fop
*
*******************************************************************************/

#define MAIN_PROGRAM



/*******************************************************************************/
/******************************** Includes *************************************/
/*******************************************************************************/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "archive.h"
#include "sflds.h"
#include "linalg.h"
#include "dirac.h"
#include "sap.h"
#include "dfl.h"
#include "forces.h"
#include "version.h"
#include "global.h"
#include "mesons.h"
#include "su3fcts.h" /*for the random_su3_dble() function*/
#include "uflds.h" /*for the udfld() function*/



/*******************************************************************************/
/******************************** Defines **************************************/
/*******************************************************************************/


/*Number of processes in each spacetime direction*/
#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)


/*macro that sets n to be the greatest between n and m*/
#define MAX(n,m) \
   if ((n)<(m)) \
      (n)=(m)


/*noise type for the creation of spinors with just one component filled*/
#define ONE_COMPONENT 3


/*number of operators to be computed for each correlator (VA, AV, PS, SP TT~)*/
#define noperator 5


/*******************************************************************************/
/******************* Declaration of Global Variables ***************************/
/*******************************************************************************/

/*** declaration of custom made structures ***/

/*struct containing all the parameters of the input file related to the correlators*/
static struct
{
   int ncorr; /*number of correlators to be computed*/
   int nnoise; /*number of noise vector for each configuration*/
   int tvals; /*= size of time lattice spaces times number of time processes = number of time intervals*/
   int noisetype; /*type of noise vectors, either U1, Z2 or GAUSS*/

   double *kappa1; /*kappa of the first kind of quark, one for each correlator*/
   double *kappa2; /*kappa of the second kind of quark, one for each correlator*/
   double *kappa3; /*kappa of the third kind of quark, one for each correlator*/
   double *kappa4; /*kappa of the fourth kind of quark, one for each correlator*/

   double *mus1; /*twisted mass of the first kind of quark, one for each correlator*/
   double *mus2; /*twisted mass of the second kind of quark, one for each correlator*/
   double *mus3; /*twisted mass of the third kind of quark, one for each correlator*/
   double *mus4; /*twisted mass of the fourth kind of quark, one for each correlator*/

   int *typeA; /*array containing the Dirac structure of the first meson in each correlator*/
   int *typeB; /*array containing the Dirac structure of the second meson in each correlator*/

   int *x0; /*array containing the first source time slice of each correlator*/
   int *z0; /*array containing the second source time slice of each correlator*/
} file_head;

/*structure containing the correlators, it has:
   - an array corr (of complex doubles) with the complete values of all the correlators at all times,
   - a related array corr_tmp used as temporary copy to store the partial values of the correlators computed by a single process,
   - an index nc labelling the gauge configuration used to compute the correlator*/
static struct
{
   complex_dble *corrConn; /*complete value of the connected part of the correlators*/
   complex_dble *corrConn_tmp; /*partial value of the connected part of the correlators computed by a single process*/
   complex_dble *corrDisc; /*complete value of the disconnected part of the correlators*/
   complex_dble *corrDisc_tmp; /*partial value of the discconnected part of the correlators computed by a single process*/
   int nc; /*index of the gauge configuration the correlator is related to*/
} data;

/*structure containing the list of GAMMA1 and GAMMA2 structures appearing in each deltaF2 4f operator*/
static struct
{
   int ngammas; /*number of Dirac structures in each operator to be summed over*/
   int *type1; /*array containing the Dirac structure of the GAMMA_1 in each piece of the operator*/
   int *type2; /*array containing the Dirac structure of the GAMMA_2 in each piece of the operator*/
   double *weights; /*weights appearing in the sum over the pieces in the operator*/
} operator_info[noperator];


/*** variables for input reading ***/

static char line[NAME_SIZE+1]; /*auxiliary variable used to store temporarily strings of characters*/

static int ipgrd[2]; /*variable used to keep track of changes in the number of processes between runs*/
                     /*if ipgrd[0]!=0 then the process grid changed, if ipgrd[1]!=0 then the process
                       block size changed*/


/*** variables for the random generator ***/

static int *rlxs_state=NULL; /*state of the random number generator rlxs*/
static int *rlxd_state=NULL; /*state of the random number generator rlxd*/


/*** variables for openMPI ***/

static int my_rank; /*rank of the process (unique identifier of the process inside the communicator group)*/


/*** variables read from input file ***/

static int noexp;  /*True if the option -noexp is set by command line, False(0) otherwise*/
static int append; /*True if the option -a is set by command line, False(0) otherwise*/
static int norng; /*True if the option -norng is set by command line, False(0) otherwise*/
static int endian; /*BIG_ENDIAN, LITTLE_ENDIAN or UNKOWN_ENDIAN depending on the machine*/

static int first; /*index of the first gauge configuration*/
static int last; /*index of the last gauge configuration*/
static int step; /*step used in the scanning of the gauge configurations*/

static int level; /*parameter of the random number generator*/
static int seed; /*seed of the random number generator*/

static int random_conf; /*if set to 1 the gauge configurations are randomly generated*/
static int check_gauge_inv; /*if set to 1 each gauge configuration is used again for computation after a gauge transformation*/

static char nbase[NAME_SIZE]; /*name of the run specified in the .in file (used to set the name of the cnfg file)*/
static char outbase[NAME_SIZE]; /*name of the output file specified in the .in file*/


/*** specifics of the run read from input file ***/

static int nprop; /*number of different quark lines (of different types of quarks)*/
static int ncorr; /*number of different correlators to be computed*/
static int nnoise; /*number of noise vectors used for each gauge configuration*/
static int noisetype; /*either U1, Z2 or GAUSS (expand to something like 1,2,3)*/

static int tvals; /*number of time intervals = time size of local lattice times number of lattices*/


/*** arrays storing the quarks' information ***/

static double *kappas; /*array containing the value of kappa (hopping parameter->mass) for each propagator*/
static double *mus; /*array containing the value of mu (twisted mass) for each propagator*/
static int *isps; /*array containing the solver id for each propagator*/


/*** arrays storing the information regarding the correlators to be computed ***/

static int *props1; /*array containing the type of the first quark appearing in each correlator*/
static int *props2; /*array containing the type of the second quark appearing in each correlator*/
static int *props3; /*array containing the type of the third quark appearing in each correlator*/
static int *props4; /*array containing the type of the fourth quark appearing in each correlator*/

static int *typeA; /*array containing the dirac structures GAMMA_A (first meson) appearing in each correlator*/
static int *typeB; /*array containing the dirac structures GAMMA_B (second meson) appearing in each correlator*/

static int *x0s; /*array containing the time slice of the first source (x0) of each correlator*/
static int *z0s; /*array containing the time slice of the second source (z0) of each correlator*/


/*** variables used to gauge transform the gauge configuration ***/
static su3_dble *g; /*su3 matrix containing the random transformation to be applied on the gauge configuration*/
static su3_dble *gbuf; /*auxiliary su3 matrix used*/
static int nfc[8],ofs[8]; /*arrays used for random g generation*/


/*** variables with the directories' names (paths) ***/

static char log_dir[NAME_SIZE]; /*path of the directory where log files are stored*/
static char loc_dir[NAME_SIZE]; /*path of the directory where configurations in the imported format are stored*/
                                /*(loc stands for local, since imported configuration have to be used only
                                   if each process reads the configuration locally)*/
static char cnfg_dir[NAME_SIZE]; /*path of the directory where configurations in the exported format are stored*/
static char dat_dir[NAME_SIZE]; /*path of the directory where various data files (.dat, .par and .rng) are stored*/


/*** variables with files' names ***/

/*
   - in the _file variable the name of the file is stored,
   - in the _save variable the same name but with a "~" at the end is stored (_save is the file used for backup purpouses)
*/

/*
the maximum lenght of the various files' names is specified by NAME_SIZE,
with some files being slightly longer (by the below offsets) and by a factor of 2 due to the way their name is constructed
*/
#define len_filename 20 /* = lenght of the string given by "/" + ".odd_df2_4fop.log"*/
#define len_offset 10 /* = lenght of the string given by "/" + "n%d_%d" */
#define len_tilda 1 /* = lenght of the string "~" */

static char log_file[NAME_SIZE*2+len_filename]; /*name of the .log file (used by the program as stdout)*/
static char log_save[NAME_SIZE*2+len_filename+len_tilda]; /*name of the .log~ file used as backup file for the .log file*/

static char par_file[NAME_SIZE*2+len_filename]; /*name of the .par file containing the (lattice) parameters of the simulation*/
static char par_save[NAME_SIZE*2+len_filename+len_tilda]; /*name of the .par~ file used as backup for the .par file*/

static char dat_file[NAME_SIZE*2+len_filename]; /*name of the .dat file where the values of the correlators get stored*/
static char dat_save[NAME_SIZE*2+len_filename+len_tilda]; /*name of the .dat~ file used as backup for the .dat file*/

static char rng_file[NAME_SIZE*2+len_filename]; /*name of the .rng file where the state of the random number generator is stored*/
static char rng_save[NAME_SIZE*2+len_filename+len_tilda]; /*name of .rng~ file used as backup for the .rng file*/

static char end_file[NAME_SIZE*2+len_filename]; /*name of the file (same as run name) with .end extension used to signal early termination*/
                                                /*(creating a file with this name, in the correct directory (??) will cause
                                                   the program to stop as soon as the correlator with the current gauge
                                                   configuration is computed)*/

static char cnfg_file[NAME_SIZE*2+len_offset]; /*name given to the .cnfg file where the configurations are stored*/
                                               /*(the name is different in the configuration are read in imported
                                                   (i.e. local) or exported file format)*/


/*** pointers used to open files ***/

static FILE *fin=NULL; /*input file where the specifics of the simulation are written*/
static FILE *flog=NULL; /*log file used as stdout where execution errors get written*/
static FILE *fend=NULL; /*(??)*/
static FILE *fdat=NULL; /*used for data file and also for binary parameters file (containing the lat structure)*/



/*******************************************************************************/
/************************** Definition of Functions ****************************/
/*******************************************************************************/


/*** input reading functions ***/

/*function reading directories' names and other inputs from input file
(function called by read_infile)*/
static void read_dirs(void)
{

   /*reading from input file is done only on process 0*/

   if (my_rank==0)
   {

      /*reading of the "run name" section:
         - nbase : set to the string written after "name"
         - outbase : set to be the string written after "output", being an optional
           parameter if nothing is written then it is set to be equal to nbase 
      */

      find_section("Run name"); /*pointer reading from input file is set the line after the string"[Run name]"*/
      read_line("name","%s",nbase); /*nbase is set to the name of the run*/
      read_line_opt("output",nbase,"%s",outbase); /*outbase is set to be the name used for output files*/

      /*reading of "Directories section":
        - log and dat dir are always read
        - if noexp is set loc dir is read while cnfg is not
        - if noexp is not set the opposite is true
      */

      find_section("Directories"); /*pointer reading from input file is set after the string "[Directories]"*/
      read_line("log_dir","%s",log_dir); /*log_dir is set to the string written after "log_dir"*/

      if (noexp) /*if configurations are in the imported file format they have to be read from the local directory*/
      {
         read_line("loc_dir","%s",loc_dir); /*loc_dir is set to the string written after "loc_dir"*/
         cnfg_dir[0]='\0'; /*cnfg_dir is set to '\0' (is not read)*/
      }
      else /*if configurations are in the usual exported file format then they are in the cnfg directory*/
      {
         read_line("cnfg_dir","%s",cnfg_dir); /*cnfg_dir is set to the string written after "cnfg_dir"*/
         loc_dir[0]='\0'; /*loc_dir is set to '\0' (is not read)*/
      }

      read_line("dat_dir","%s",dat_dir); /*dat_dir is set to the string written after "dat_dir"*/

      /*reading of "Configurations" section:
        - first is set to the index of the firt configuration
        - last is set to the index of the last configuration
        - step is set to the step at which configurations are scanned
      */

      find_section("Configurations"); /*pointer reading from input file is set after the string "Configurations"*/
      read_line("first","%d",&first); /*first is set to the integer written after "first"*/
      read_line("last","%d",&last); /*last is set to the integer written after "last"*/
      read_line("step","%d",&step); /*step is set to the integer written after "step"*/

      /*reading of "Random number generator" section:
        -level and seed are set to the specified integers
      */

      find_section("Random number generator"); /*pointer reading from input file is set after the string "Random number generator"*/
      read_line("level","%d",&level); /*level is set t the specified integer*/
      read_line("seed","%d",&seed); /*seed is set t the specified integer*/

      /*an error is raised if first, last and step are not valid:
        last-first should be non negative and an integer multiple of step*/
      error_root((last<first)||(step<1)||(((last-first)%step)!=0),1,
                 "read_dirs [odd_df2_4fop.c]","Improper configuration range");
   }

   /*all the parameters read during process 0 are broadcasted
     to all the other processes of the group*/

   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(outbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);

   MPI_Bcast(log_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(loc_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(cnfg_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(dat_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);

   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);

   MPI_Bcast(&level,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&seed,1,MPI_INT,0,MPI_COMM_WORLD);

} /*end of read_dirs function*/


/*function for files inizialization according to input file specifications
(function called by read_infile)*/
static void setup_files(void)
{
   /*lenght check of the string loc_dir or cnfg_dir*/
   if (noexp)
      error_root(name_size("%s/%sn%d_%d",loc_dir,nbase,last,NPROC-1)>=NAME_SIZE,
                 1,"setup_files [odd_df2_4fop.c]","loc_dir name is too long");
   else
      error_root(name_size("%s/%sn%d",cnfg_dir,nbase,last)>=NAME_SIZE,
                 1,"setup_files [odd_df2_4fop.c]","cnfg_dir name is too long");

   /*check on accessibility (only on process 0) and name lenght of dat_dir*/
   check_dir_root(dat_dir);
   error_root(name_size("%s/%s.odd_df2_4fop.dat~",dat_dir,outbase)>=NAME_SIZE,
              1,"setup_files [odd_df2_4fop.c]","dat_dir name is too long");

   /*check on accessibility (only on process 0) and name lenght of log_dir*/
   check_dir_root(log_dir);
   error_root(name_size("%s/%s.odd_df2_4fop.log~",log_dir,outbase)>=NAME_SIZE,
              1,"setup_files [odd_df2_4fop.c]","log_dir name is too long");

   /*assignment of files' names based on input file specifications*/

   sprintf(log_file,"%s/%s.odd_df2_4fop.log",log_dir,outbase);
   sprintf(end_file,"%s/%s.odd_df2_4fop.end",log_dir,outbase);
   sprintf(par_file,"%s/%s.odd_df2_4fop.par",dat_dir,outbase);
   sprintf(dat_file,"%s/%s.odd_df2_4fop.dat",dat_dir,outbase);
   sprintf(rng_file,"%s/%s.odd_df2_4fop.rng",dat_dir,outbase);
   sprintf(log_save,"%s~",log_file);
   sprintf(par_save,"%s~",par_file);
   sprintf(dat_save,"%s~",dat_file);
   sprintf(rng_save,"%s~",rng_file);

} /*end of setup_files function*/


/*function used to set the entry type[ic] to an integer depending on the Dirac structure written in tmpstring
(function called by read_lat_parms)*/
void setTypeFromTmpString(char *tmpstring, int *type, int icorr) {
    if(strncmp(tmpstring,"1",1)==0)
        type[icorr]=ONE_TYPE;
    else if(strncmp(tmpstring,"G0G1",4)==0)
        type[icorr]=GAMMA0GAMMA1_TYPE;
    else if(strncmp(tmpstring,"G0G2",4)==0)
        type[icorr]=GAMMA0GAMMA2_TYPE;
    else if(strncmp(tmpstring,"G0G3",4)==0)
        type[icorr]=GAMMA0GAMMA3_TYPE;
    else if(strncmp(tmpstring,"G0G5",4)==0)
        type[icorr]=GAMMA0GAMMA5_TYPE;
    else if(strncmp(tmpstring,"G1G2",4)==0)
        type[icorr]=GAMMA1GAMMA2_TYPE;
    else if(strncmp(tmpstring,"G1G3",4)==0)
        type[icorr]=GAMMA1GAMMA3_TYPE;
    else if(strncmp(tmpstring,"G1G5",4)==0)
        type[icorr]=GAMMA1GAMMA5_TYPE;
    else if(strncmp(tmpstring,"G2G3",4)==0)
        type[icorr]=GAMMA2GAMMA3_TYPE;
    else if(strncmp(tmpstring,"G2G5",4)==0)
        type[icorr]=GAMMA2GAMMA5_TYPE;
    else if(strncmp(tmpstring,"G3G5",4)==0)
        type[icorr]=GAMMA3GAMMA5_TYPE;
    else if(strncmp(tmpstring,"G0",2)==0)
        type[icorr]=GAMMA0_TYPE;
    else if(strncmp(tmpstring,"G1",2)==0)
        type[icorr]=GAMMA1_TYPE;
    else if(strncmp(tmpstring,"G2",2)==0)
        type[icorr]=GAMMA2_TYPE;
    else if(strncmp(tmpstring,"G3",2)==0)
        type[icorr]=GAMMA3_TYPE;
    else if(strncmp(tmpstring,"G5",2)==0)
        type[icorr]=GAMMA5_TYPE;
}


/*function to read lattice parameters from input file and assignin them to global variables
(function called by read_infile)*/
static void read_lat_parms(void)
{
   /*declaration of temporary variables used for reading parameters*/

   double csw; /*coefficient of sw term*/
   double cF; /*coefficient of the Fermion O(a) boundary counterterm*/

   char tmpstring[NAME_SIZE]; /*temporary string used for reading*/
   char tmpstring2[NAME_SIZE]; /*temporary string used for reading*/
   
   int iprop; /*index running over the different propagators (quark types)*/
   int icorr; /*index running over the different correlators to be computed*/

   int eoflg; /*twisted mass flag read from input file*/

   /*on process 0 reads parameters from file*/

   if (my_rank==0)
   {

      /*reading of the [Measurements] section from input file*/

      find_section("Measurements"); /*reading pointer set the line after the string "[Measurements]"*/
      read_line("nprop","%d",&nprop); /*nprop is set to the number of different quark lines written in the input file*/
      read_line("ncorr","%d",&ncorr); /*ncorr is set to the number of different correlators written in the input file*/
      read_line("nnoise","%d",&nnoise); /*nnoise is set to the number of noise vector for each configuration*/
      read_line("noise_type","%s",tmpstring); /*noise_type set to U1, Z2 or GAUSS according to input file*/
      read_line("csw","%lf",&csw); /*csw coefficient read from input file*/
      read_line("cF","%lf",&cF); /*cF coefficient read from input file*/
      read_line("eoflg","%d",&eoflg); /*eoflg read from input file*/

      read_line("random_conf","%d",&random_conf); /*random_conf read from input file*/
      read_line("check_gauge_inv","%d",&check_gauge_inv); /*check_gauge_inv read from input file*/


      /*check on the validity of the parameters read from input file*/

      /*nprop, ncorr and nnoise must be positive integers*/
      error_root(nprop<1,1,"read_lat_parms [odd_df2_4fop.c]",
                 "Specified nprop must be larger than zero");
      error_root(ncorr<1,1,"read_lat_parms [odd_df2_4fop.c]",
                 "Specified ncorr must be larger than zero");
      error_root(nnoise<1,1,"read_lat_parms [odd_df2_4fop.c]",
                 "Specified nnoise must be larger than zero");
      
      /*eoflg must be either 0 or 1*/
      error_root((eoflg<0)||(eoflg>1),1,"read_lat_parms [odd_df2_4fop.c]",
		 "Specified eoflg must be 0,1");

      /*random_conf and check_gauge_inv must be either 0 or 1*/
      error_root((random_conf<0)||(random_conf>1),1,"read_lat_parms [odd_df2_4fop.c]",
		 "Specified random_conf must be 0,1");
       error_root((check_gauge_inv<0)||(check_gauge_inv>1),1,"read_lat_parms [odd_df2_4fop.c]",
		 "Specified check_gauge_inv must be 0,1");

      /*noise_type must be either U1, Z2 or GAUSS or ONE_COMPONENT*/
      noisetype=-1;
      if(strcmp(tmpstring,"Z2")==0)
         noisetype=Z2_NOISE;
      if(strcmp(tmpstring,"GAUSS")==0)
         noisetype=GAUSS_NOISE;
      if(strcmp(tmpstring,"U1")==0)
         noisetype=U1_NOISE;
      if(strcmp(tmpstring,"ONE_COMPONENT")==0)
         noisetype=ONE_COMPONENT;
      error_root(noisetype==-1,1,"read_lat_parms [mesons.c]",
                 "Unknown noise type");
   }

   /*broadcast of parameters read on process 0 to
   all other proceses of the communicator group*/

   MPI_Bcast(&nprop,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&ncorr,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nnoise,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&noisetype,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&eoflg,1,MPI_INT,0,MPI_COMM_WORLD);

   MPI_Bcast(&random_conf,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&check_gauge_inv,1,MPI_INT,0,MPI_COMM_WORLD);

   /*memory allocation for all the parameters needed
   by the various propagators and correlator*/

   kappas=malloc(nprop*sizeof(double)); /*one kappa for each propagator*/
   mus=malloc(nprop*sizeof(double)); /*one mu for each propagator*/
   isps=malloc(nprop*sizeof(int)); /*one solver id for each propagator*/

   props1=malloc(ncorr*sizeof(int)); /*type of first quark, one for each correlator*/
   props2=malloc(ncorr*sizeof(int)); /*type of second quark, one for each correlator*/
   props3=malloc(ncorr*sizeof(int)); /*type of third quark, one for each correlator*/
   props4=malloc(ncorr*sizeof(int)); /*type of fourth quark, one for each correlator*/

   typeA=malloc(ncorr*sizeof(int)); /*Dirac structure of first meson, one for each correlator*/
   typeB=malloc(ncorr*sizeof(int)); /*Dirac structure of second meson, one for each correlator*/

   x0s=malloc(ncorr*sizeof(int)); /*time slice of the first source, one for each correlator*/
   z0s=malloc(ncorr*sizeof(int)); /*time slice of the second source, one for each correlator*/
   
   file_head.kappa1=malloc(ncorr*sizeof(double)); /*kappa of the first kind of quark, one for each correlator*/
   file_head.kappa2=malloc(ncorr*sizeof(double)); /*kappa of the second kind of quark, one for each correlator*/
   file_head.kappa3=malloc(ncorr*sizeof(double)); /*kappa of the third kind of quark, one for each correlator*/
   file_head.kappa4=malloc(ncorr*sizeof(double)); /*kappa of the fourth kind of quark, one for each correlator*/
   
   file_head.mus1=malloc(ncorr*sizeof(double)); /*twisted mass of the first kind of quark, one for each correlator*/
   file_head.mus2=malloc(ncorr*sizeof(double)); /*twisted mass of the second kind of quark, one for each correlator*/
   file_head.mus3=malloc(ncorr*sizeof(double)); /*twisted mass of the third kind of quark, one for each correlator*/
   file_head.mus4=malloc(ncorr*sizeof(double)); /*twisted mass of the fourth kind of quark, one for each correlator*/


   file_head.typeA=typeA; /*Dirac structure of first meson, one for each correlator*/
   file_head.typeB=typeB; /*Dirac structure of second meson, one for each correlator*/

   file_head.x0=x0s; /*time slice of the first source, one for each correlator*/
   file_head.z0=z0s; /*time slice of the first source, one for each correlator*/

   /*check of successful memory allocation*/
   error((kappas==NULL)||(mus==NULL)||(isps==NULL)||
         (props1==NULL)||(props2==NULL)||(props3==NULL)||(props4==NULL)||
         (typeA==NULL)||(typeB==NULL)||
         (x0s==NULL)||(z0s==NULL)||
         (file_head.kappa1==NULL)||(file_head.kappa2==NULL)||(file_head.kappa3==NULL)||(file_head.kappa4==NULL)||
         (file_head.mus1==NULL)||(file_head.mus2==NULL)||(file_head.mus3==NULL)||(file_head.mus4==NULL),
         1,"read_lat_parms [odd_df2_4fop.c]","Out of memory");


   /*on process 0 reads from the input file the parameters
   related to each of the nprop propagators and each of the ncorr correlators*/

   if (my_rank==0)
   {
      /*loop over the different propagators*/
      for(iprop=0; iprop<nprop; iprop++)
      {
         sprintf(tmpstring,"Propagator %i",iprop); /*temporary string set to the propagator identifier*/
         find_section(tmpstring); /*reading pointer set in the section of the iprop-th propagator*/
         read_line("kappa","%lf",&kappas[iprop]); /*for the given propagator kappa is read from input file*/
         read_line("isp","%d",&isps[iprop]); /*for the given propagator the solver id is read from input file*/
	      read_line("mus","%lf",&mus[iprop]); /*for the given propagator the twisted mass is read from input file*/
      }

      /*loop over the different correlators*/
      for(icorr=0; icorr<ncorr; icorr++)
      {
         sprintf(tmpstring,"Correlator %i",icorr); /*temporary string set to the correlator identifier*/
         find_section(tmpstring); /*reading pointer set in the section of the icorr-th correlator*/

         /*the types of the first and the second quarks are read from the input file and
         the validity of the input parameters is checked (they must range from 0 to to nprop-1)*/
         read_line("iprop","%d %d %d %d", &props1[icorr], &props2[icorr], &props3[icorr], &props4[icorr]);
         error_root((props1[icorr]<0)||(props1[icorr]>=nprop),1,"read_lat_parms [odd_df2_4fop.c]",
                 "Propagator index out of range");
         error_root((props2[icorr]<0)||(props2[icorr]>=nprop),1,"read_lat_parms [odd_df2_4fop.c]",
                 "Propagator index out of range");
         error_root((props3[icorr]<0)||(props3[icorr]>=nprop),1,"read_lat_parms [odd_df2_4fop.c]",
                 "Propagator index out of range");
         error_root((props4[icorr]<0)||(props4[icorr]>=nprop),1,"read_lat_parms [odd_df2_4fop.c]",
                 "Propagator index out of range");

         /*the two temporary strings are used to read from the input file the Dirac
         structures appearing in the given correlator, then the string read
         from file is converted to an integer identifier*/

         read_line("gamma_A_B","%s %s",tmpstring,tmpstring2); /*reading of GAMMA_A and GAMMA_B*/
         
         typeA[icorr]=-1; /*inizialization of GAMMA_A (its integer identifier)*/
         typeB[icorr]=-1; /*inizialization of GAMMA_B (its integer identifier)*/
         
         setTypeFromTmpString(tmpstring,typeA,icorr); /*conversion of tmpstring to an integer identifier*/
         setTypeFromTmpString(tmpstring2,typeB,icorr); /*conversion of tmpstring2 to an integer identifier*/

         /*validity check of type1 and type2 read from the input file*/
         error_root((typeA[icorr]==-1)||(typeB[icorr]==-1),1,"read_lat_parms [odd_df2_4fop.c]",
                 "Unknown or unsupported Dirac structure");

         /*source time slice is read for each correlator and its validity checked
         (the timeslice must be inside the previously specified time boundaries)*/
         read_line("x0","%d",&x0s[icorr]);
         error_root((x0s[icorr]<=0)||(x0s[icorr]>=(NPROC0*L0-1)),1,"read_lat_parms [odd_df2_4fop.c]",
                 "Specified time x0 is out of range");
         read_line("z0","%d",&z0s[icorr]);
         error_root((z0s[icorr]<=0)||(z0s[icorr]>=(NPROC0*L0-1)),1,"read_lat_parms [odd_df2_4fop.c]",
                 "Specified time z0 is out of range");

      }

   }

   /*broadcast of parameters read on process 0 to
   all other proceses of the communicator group*/

   MPI_Bcast(kappas,nprop,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(mus,nprop,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(isps,nprop,MPI_INT,0,MPI_COMM_WORLD);

   MPI_Bcast(props1,ncorr,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(props2,ncorr,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(props3,ncorr,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(props4,ncorr,MPI_INT,0,MPI_COMM_WORLD);

   MPI_Bcast(typeA,ncorr,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(typeB,ncorr,MPI_INT,0,MPI_COMM_WORLD);

   MPI_Bcast(x0s,ncorr,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(z0s,ncorr,MPI_INT,0,MPI_COMM_WORLD);

   /*lattice parameters are saved in global structure:
      - lat : set_lat_parms specifies the parameters of the lat structure, and below
              the inverse bare coupling is set to 0, the coefficient of the plaquette
              loops in the gauge action is set to 1, kappa for u and d is set to the
              specified value, for s and c is set to 0, csw and cF are set to the 
              specified values, cG (coefficient of gauge O(a) counter term) is set to 1
      - sw  : structure containing cG, cF and the bare quark mass m0,
              with set_sw_parms m0 is set to the bare quark mass of the up quark
              (sea_quark_mass turns 0,1,2 to m0u,m0s,m0c)
      - tm  : structue containing the twisted mass flag eoflg,
              if eoflg=1 then twisted mass, SAP preconditioner and little Dirac operator
              is turned off on odd sites
      - file_head : structure containing details of correlators
   */

   set_lat_parms(0.0,1.0,kappas[0],0.0,0.0,csw,1.0,cF); /*parameters of the global structure lat are set*/
   set_sw_parms(sea_quark_mass(0)); /*parameters of the global structure sw are set*/
   set_tm_parms(eoflg); /*eoflg in the global structure tm is set to the read value*/

   file_head.ncorr = ncorr; /*number of correlators saved to global structure*/
   file_head.nnoise = nnoise; /*number of noise vectors saved to global structure*/
   file_head.noisetype = noisetype; /*noisetype saved to global structure*/
   file_head.tvals = NPROC0*L0; /*tvals saved to global structure*/

   tvals = NPROC0*L0; /*tvals saved to global variable*/
   
   for(icorr=0; icorr<ncorr; icorr++) /*for each correlator the related parameters are saved in file_head*/
   {
      file_head.kappa1[icorr]=kappas[props1[icorr]]; /*kappa of first quark saved to global structure*/
      file_head.kappa2[icorr]=kappas[props2[icorr]]; /*kappa of second quark saved to global structure*/
      file_head.kappa3[icorr]=kappas[props3[icorr]]; /*kappa of third quark saved to global structure*/
      file_head.kappa4[icorr]=kappas[props4[icorr]]; /*kappa of fourth quark saved to global structure*/

      file_head.mus1[icorr]=mus[props1[icorr]]; /*twisted mass of first quark saved to global structure*/
      file_head.mus2[icorr]=mus[props2[icorr]]; /*twisted mass of second quark saved to global structure*/
      file_head.mus3[icorr]=mus[props3[icorr]]; /*twisted mass of third quark saved to global structure*/
      file_head.mus4[icorr]=mus[props4[icorr]]; /*twisted mass of fourth quark saved to global structure*/
   }

   /*the parameters in the lat structure get saved in the fdat file,
   if the option -a is given the consistency with the previously
   written parameters is checked*/

   if (append) /*if the current simulation is the continuation of a previous run ...*/
      check_lat_parms(fdat); /*... the match between the previous and the current lattice parameters is checked*/
   else /*if the current simulation is a new run ...*/
      write_lat_parms(fdat); /*...the lat structure gets written to the fdat file*/
}


/*function that reads the section "SAP" from the input file
(this function is called by the read_solver function)*/
static void read_sap_parms(void)
{
   int bs[4];

   if (my_rank==0)
   {
      find_section("SAP");
      read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   set_sap_parms(bs,1,4,5);
}


/*function that reads the sections "Deflation subspace",
"Deflation subspace generation" and "Deflation projection" from the input file
(this function is called by the read_solver function)*/
static void read_dfl_parms(void)
{
   int bs[4],Ns;
   int ninv,nmr,ncy,nkv,nmx;
   double kappa,mudfl,res;

   if (my_rank==0)
   {
      find_section("Deflation subspace");
      read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
      read_line("Ns","%d",&Ns);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&Ns,1,MPI_INT,0,MPI_COMM_WORLD);
   set_dfl_parms(bs,Ns);

   if (my_rank==0)
   {
      find_section("Deflation subspace generation");
      read_line("kappa","%lf",&kappa);
      read_line("mu","%lf",&mudfl);
      read_line("ninv","%d",&ninv);
      read_line("nmr","%d",&nmr);
      read_line("ncy","%d",&ncy);
   }

   MPI_Bcast(&kappa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&mudfl,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&ninv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmr,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&ncy,1,MPI_INT,0,MPI_COMM_WORLD);
   set_dfl_gen_parms(kappa,mudfl,ninv,nmr,ncy);

   if (my_rank==0)
   {
      find_section("Deflation projection");
      read_line("nkv","%d",&nkv);
      read_line("nmx","%d",&nmx);
      read_line("res","%lf",&res);
   }

   MPI_Bcast(&nkv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   set_dfl_pro_parms(nkv,nmx,res);
}


/*function reading the solvers' parameters from the input file
(function called by the read_infile function)*/
static void read_solvers(void)
{
   solver_parms_t sp;
   int i,j;
   int isap=0,idfl=0;

   for (i=0;i<nprop;i++)
   {
      j=isps[i];
      sp=solver_parms(j);
      if (sp.solver==SOLVERS)
      {
         read_solver_parms(j);
         sp=solver_parms(j);
         if (sp.solver==SAP_GCR)
            isap=1;
         if (sp.solver==DFL_SAP_GCR)
         {
            isap=1;
            idfl=1;
         }
      }
   }

   if (isap)
      read_sap_parms();

   if (idfl)
      read_dfl_parms();
}


/*function determining the global variables of the simulatiom from the input file
(function called by the main)*/
static void read_infile(int argc,char *argv[])
{
   int ifile; /*position in argv[] of the -ifile option*/

   /*reading of the input from command line during the 
     first process*/
   
   if (my_rank==0) /*--> true on the first process*/
   {
      /*setting the file STARTUP_ERROR as the place where
        execution output (errors) gets written, i.e. setting it as stdout*/
      flog=freopen("STARTUP_ERROR","w",stdout);

      /*reading input from command line*/
 
      ifile=find_opt(argc,argv,"-i"); /*option to specify input file*/
      endian=endianness(); /*endianness of the machine*/

      /*gives an error if input file not specified*/
      error_root((ifile==0)||(ifile==(argc-1)),1,"read_infile [odd_df2_4fop.c]",
                 "Syntax: odd_df2_4fop -i <input file> [-noexp] [-a [-norng]]");

      /*gives an error if the machine has unkown endianness*/
      error_root(endian==UNKNOWN_ENDIAN,1,"read_infile [odd_df2_4fop.c]",
                 "Machine has unknown endianness");

      noexp=find_opt(argc,argv,"-noexp"); /*option to specify configurations reading*/
      append=find_opt(argc,argv,"-a"); /*option to specify output appending*/
      norng=find_opt(argc,argv,"-norng"); /*option to specify generator initialization*/

      /*opening the input file*/

      /*setting stdin to be the input file,
        gives an error if the input file cannot be open*/
      fin=freopen(argv[ifile+1],"r",stdin);
      error_root(fin==NULL,1,"read_infile [odd_df2_4fop.c]",
                 "Unable to open input file");
   }

   /*broadcast from process 0 to all the other processes
     in the communicator of the input parameters just read*/

   MPI_Bcast(&endian,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&noexp,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&append,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&norng,1,MPI_INT,0,MPI_COMM_WORLD);

   /*reading of parameters from the input file (done only in process 0)*/

   read_dirs(); /*reads input file and set global variables*/
   setup_files(); /*initialization of files specified in the input file*/

   /*on process 0 open binary parameters file*/
   if (my_rank==0)
   {
      if (append)
         fdat=fopen(par_file,"rb");
      else
         fdat=fopen(par_file,"wb");

      error_root(fdat==NULL,1,"read_infile [odd_df2_4fop.c]",
                 "Unable to open parameter file");
   }

   read_lat_parms(); /*reads lattice parameters from input file*/
   read_solvers(); /*reads solver parameters from input file*/

   /*closing opened files*/

   /*on process 0 closes input and parameter file*/
   if (my_rank==0)
   {
      fclose(fin);
      fclose(fdat);

      if (append==0)
         copy_file(par_file,par_save); /*save binary parameter file to par_save file*/
   }
}


/*** functions handling the data structure ***/

/*function used to allocate the structure data
(function called by the main)*/
static void alloc_data(void)
{
   /*to store all the values of the correlators the number of complex double needed is equal 
     to the number of correlators times the number of time intervals timse the number of noise vectors squared time number of operators
     (such a quantity is needed both for data.corr and for the temporary counterpart data.corr_tmp,
     and both for the connected and the disconnected counterpart)
   */

   /*memory allocation*/
   data.corrConn=malloc(file_head.nnoise*file_head.nnoise*file_head.ncorr*file_head.tvals*noperator*sizeof(complex_dble));
   data.corrConn_tmp=malloc(file_head.nnoise*file_head.nnoise*file_head.ncorr*file_head.tvals*noperator*sizeof(complex_dble));
   data.corrDisc=malloc(file_head.nnoise*file_head.nnoise*file_head.ncorr*file_head.tvals*noperator*sizeof(complex_dble));
   data.corrDisc_tmp=malloc(file_head.nnoise*file_head.nnoise*file_head.ncorr*file_head.tvals*noperator*sizeof(complex_dble));
   
   /*check on correct memory allocation*/
   error((data.corrConn==NULL)||(data.corrConn_tmp==NULL)||
         (data.corrDisc==NULL)||(data.corrDisc_tmp==NULL),1,"alloc_data [odd_df2_4fop.c]","Unable to allocate data arrays");
}


/*function used to read the data stucture from the .dat file,
returns 1 if something has been read, 0 if there is nothing to read
(this function is called by the check_old_dat function)*/
static int read_data(void)
{
   int ir; /*index used for the reading count*/
   int nr; /*total readings to be done*/
   int chunk; /*size of the chunk written on the file*/
   int icorr; /*index used in the function*/

   /*first we read nc*/

   nr=1;
   ir=fread(&(data.nc),sizeof(int),1,fdat);

   /*then we read the data of each correlator*/

   for (icorr=0;icorr<file_head.ncorr;icorr++)
   {
      chunk=file_head.nnoise*file_head.nnoise*file_head.tvals*2; /*size of the chunk to be read */
      nr+=chunk; /*count update*/
      
      /*reading*/
      ir+=fread(&(data.corrConn[icorr*file_head.tvals*file_head.nnoise*file_head.nnoise*noperator]),
                    sizeof(double),chunk,fdat);
      
      nr+=chunk; /*count update*/
      
      /*reading*/
      ir+=fread(&(data.corrDisc[icorr*file_head.tvals*file_head.nnoise*file_head.nnoise*noperator]),
                    sizeof(double),chunk,fdat);
   }

   /*if nothing has been read the function returns 0*/
   if (ir==0)
      return 0;

   /*check on correct reading count*/
   error_root(ir!=nr,1,"read_data [odd_df2_4fop.c]",
                 "Read error or incomplete data record");
   
   /*if the machine is big endian swaps the bit of the read input (that is always little endian)*/
   if(endian==BIG_ENDIAN)
   {
      bswap_double(nr,data.corrConn);
      bswap_double(nr,data.corrDisc);
      bswap_int(1,&(data.nc));
   }

   return 1;

}


/*function that writes on the .dat file the values of the correlators
(and the index of the gauge configuration they correspond to)
(function called by the main)*/
static void write_data(void)
{
   int iw; /*counter used to write*/
   int nw; /*total number of elements to be written*/
   int chunk; /*size of each chunk that is written to file (??)*/
   int icorr; /*index used in the function*/

   /*the data is written only on process 0*/
   if (my_rank==0)
   {
      /*open data file and check on correct opening procedure*/
      fdat=fopen(dat_file,"ab");
      error_root(fdat==NULL,1,"write_data [odd_df2_4fop.c]",
                 "Unable to open dat file");

      /*first we write the index of the gauge configuration and the 4fop complete correlators*/

      nw = 1; /*total number of writing so far = 1 (after we write nc just below)*/

      /*swap of bit before writing if big endian*/
      if(endian==BIG_ENDIAN)
      {
         bswap_double(file_head.nnoise*file_head.nnoise*file_head.tvals*file_head.ncorr*noperator*2,data.corrConn);
         bswap_double(file_head.nnoise*file_head.nnoise*file_head.tvals*file_head.ncorr*noperator*2,data.corrDisc);
         bswap_int(1,&(data.nc));
      }

      /*writing of nc*/
      iw=fwrite(&(data.nc),sizeof(int),1,fdat);

      /*then we write the data for each 4fop corr*/

      for (icorr=0;icorr<file_head.ncorr;icorr++)
      {
         chunk=file_head.nnoise*file_head.nnoise*file_head.tvals*noperator*2; /*size of the chunk to write*/
         nw+=chunk; /*update the writing count*/
         
         /*writing of the correlator*/
         iw+=fwrite(&(data.corrConn[icorr*file_head.tvals*file_head.nnoise*file_head.nnoise*noperator]),sizeof(double),chunk,fdat);

         nw+=chunk; /*update the writing count*/
         
         /*writing of the correlator*/
         iw+=fwrite(&(data.corrDisc[icorr*file_head.tvals*file_head.nnoise*file_head.nnoise*noperator]),sizeof(double),chunk,fdat);
         
      }

      /*swap of bit after writing if big endian to restore initial situation*/
      if(endian==BIG_ENDIAN)
      {
         bswap_double(file_head.nnoise*file_head.nnoise*file_head.tvals*file_head.ncorr*noperator*2,data.corrConn);
         bswap_double(file_head.nnoise*file_head.nnoise*file_head.tvals*file_head.ncorr*noperator*2,data.corrDisc);
         bswap_int(1,&(data.nc));
      }

      /*check on correct writing count*/
      error_root(iw!=nw,1,"write_data [odd_df2_4fop.c]",
                 "Incorrect write count");

      /*close data file*/
      fclose(fdat);
   }
}


/*** functions handling the file_head structure ***/

/*function used to check the compatibility of the current global file_head structure
with what is written in the .dat file
(function called by the check_old_dat function)*/
static void check_file_head(void)
{
   int i,ir,ie;
   stdint_t istd[1];
   double dbl[1];

   ir=fread(istd,sizeof(stdint_t),1,fdat);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   ie=(istd[0]!=(stdint_t)(file_head.ncorr));

   ir+=fread(istd,sizeof(stdint_t),1,fdat);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   ie+=(istd[0]!=(stdint_t)(file_head.nnoise));

   ir+=fread(istd,sizeof(stdint_t),1,fdat);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   ie+=(istd[0]!=(stdint_t)(file_head.tvals));

   ir+=fread(istd,sizeof(stdint_t),1,fdat);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   ie+=(istd[0]!=(stdint_t)(file_head.noisetype));

   error_root(ir!=4,1,"check_file_head [odd_df2_4fop.c]",
              "Incorrect read count");
   error_root(ie!=0,1,"check_file_head [odd_df2_4fop.c]",
              "Unexpected value of ncorr, nnoise, tvals or noisetype");

   
   for (i=0;i<file_head.ncorr;i++)
   {
      ir=fread(dbl,sizeof(double),1,fdat);
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      ie=(dbl[0]!=file_head.kappa1[i]);

      ir+=fread(dbl,sizeof(double),1,fdat);
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      ie+=(dbl[0]!=file_head.kappa2[i]);

      ir+=fread(dbl,sizeof(double),1,fdat);
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      ie+=(dbl[0]!=file_head.kappa3[i]);

      ir+=fread(dbl,sizeof(double),1,fdat);
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      ie+=(dbl[0]!=file_head.kappa4[i]);


      ir+=fread(dbl,sizeof(double),1,fdat);
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      ie+=(dbl[0]!=file_head.mus1[i]);

      ir+=fread(dbl,sizeof(double),1,fdat);
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      ie+=(dbl[0]!=file_head.mus2[i]);

      ir+=fread(dbl,sizeof(double),1,fdat);
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      ie+=(dbl[0]!=file_head.mus3[i]);

      ir+=fread(dbl,sizeof(double),1,fdat);
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      ie+=(dbl[0]!=file_head.mus4[i]);


      ir+=fread(istd,sizeof(stdint_t),1,fdat);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      ie+=(istd[0]!=(stdint_t)(file_head.typeA[i]));

      ir+=fread(istd,sizeof(stdint_t),1,fdat);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      ie+=(istd[0]!=(stdint_t)(file_head.typeB[i]));


      ir+=fread(istd,sizeof(stdint_t),1,fdat);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      ie+=(istd[0]!=(stdint_t)(file_head.x0[i]));

      ir+=fread(istd,sizeof(stdint_t),1,fdat);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      ie+=(istd[0]!=(stdint_t)(file_head.z0[i]));


      error_root(ir!=12,1,"check_file_head [odd_df2_4fop.c]",
              "Incorrect read count");
      error_root(ie!=0,1,"check_file_head [odd_df2_4fop.c]",
              "Unexpected value of kappa, type, x0 or isreal");
   }
}


/*function used to save the global file_head structure containing
the correlators' information on the binary .dat file
(function called by the check_files function)*/
static void write_file_head(void)
{
   stdint_t istd[1];
   int iw=0;
   int i;
   double dbl[1];

   istd[0]=(stdint_t)(file_head.ncorr);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   iw=fwrite(istd,sizeof(stdint_t),1,fdat);

   istd[0]=(stdint_t)(file_head.nnoise);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

   istd[0]=(stdint_t)(file_head.tvals);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

   istd[0]=(stdint_t)(file_head.noisetype);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

   istd[0]=(stdint_t)(check_gauge_inv);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

   istd[0]=(stdint_t)(random_conf);
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

   error_root(iw!=6,1,"write_file_head [odd_df2_4fop.c]",
              "Incorrect write count");

   
   for (i=0;i<file_head.ncorr;i++)
   {
      dbl[0] = file_head.kappa1[i];
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      iw=fwrite(dbl,sizeof(double),1,fdat);

      dbl[0] = file_head.kappa2[i];
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      iw+=fwrite(dbl,sizeof(double),1,fdat);

      dbl[0] = file_head.kappa3[i];
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      iw+=fwrite(dbl,sizeof(double),1,fdat);

      dbl[0] = file_head.kappa4[i];
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      iw+=fwrite(dbl,sizeof(double),1,fdat);


      dbl[0] = file_head.mus1[i];
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      iw+=fwrite(dbl,sizeof(double),1,fdat);

      dbl[0] = file_head.mus2[i];
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      iw+=fwrite(dbl,sizeof(double),1,fdat);

      dbl[0] = file_head.mus3[i];
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      iw+=fwrite(dbl,sizeof(double),1,fdat);

      dbl[0] = file_head.mus4[i];
      if (endian==BIG_ENDIAN)
      bswap_double(1,dbl);
      iw+=fwrite(dbl,sizeof(double),1,fdat);


      istd[0]=(stdint_t)(file_head.typeA[i]);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

      istd[0]=(stdint_t)(file_head.typeB[i]);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      iw+=fwrite(istd,sizeof(stdint_t),1,fdat);


      istd[0]=(stdint_t)(file_head.x0[i]);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

      istd[0]=(stdint_t)(file_head.z0[i]);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      iw+=fwrite(istd,sizeof(stdint_t),1,fdat);


      error_root(iw!=12,1,"write_file_head [odd_df2_4fop.c]",
              "Incorrect write count");

   }

}


/*** compatibility checking functions ***/

/*function used to read the old log file. fts, lst and stp are assigned respectively to: 
   -fts, number of the first configuration of the previous run
   -lst, number of the last configuration of the previous run
   -stp, step between each configuration of the previous run
(function called by the check_files function)*/
static void check_old_log(int *fst,int *lst,int *stp)
{
   int ie,ic,isv;
   int fc,lc,dc,pc;
   int np[4],bp[4];

   fend=fopen(log_file,"r");
   error_root(fend==NULL,1,"check_old_log [odd_df2_4fop.c]",
              "Unable to open log file");

   fc=0;
   lc=0;
   dc=0;
   pc=0;

   ie=0x0;
   ic=0;
   isv=0;

   while (fgets(line,NAME_SIZE,fend)!=NULL)
   {
      if (strstr(line,"process grid")!=NULL)
      {
         if (sscanf(line,"%dx%dx%dx%d process grid, %dx%dx%dx%d",
                    np,np+1,np+2,np+3,bp,bp+1,bp+2,bp+3)==8)
         {
            ipgrd[0]=((np[0]!=NPROC0)||(np[1]!=NPROC1)||
                      (np[2]!=NPROC2)||(np[3]!=NPROC3));
            ipgrd[1]=((bp[0]!=NPROC0_BLK)||(bp[1]!=NPROC1_BLK)||
                      (bp[2]!=NPROC2_BLK)||(bp[3]!=NPROC3_BLK));
         }
         else
            ie|=0x1;
      }
      
      if (strstr(line,"fully processed")!=NULL)
      {
         pc=lc;

         if (sscanf(line,"Configuration no %d",&lc)==1)
         {
            ic+=1;
            isv=1;
         }
         else
            ie|=0x1;

         if (ic==1)
            fc=lc;
         else if (ic==2)
            dc=lc-fc;
         else if ((ic>2)&&(lc!=(pc+dc)))
            ie|=0x2;
      }
      else if (strstr(line,"Configuration no")!=NULL)
         isv=0;
   }

   fclose(fend);

   error_root((ie&0x1)!=0x0,1,"check_old_log [odd_df2_4fop.c]",
              "Incorrect read count");
   error_root((ie&0x2)!=0x0,1,"check_old_log [odd_df2_4fop.c]",
              "Configuration numbers are not equally spaced");
   error_root(isv==0,1,"check_old_log [odd_df2_4fop.c]",
              "Log file extends beyond the last configuration save");

   (*fst)=fc;
   (*lst)=lc;
   (*stp)=dc;
}


/*function used to check that the first, last and step of the configuration scan
reported in the .dat file are the ones passed as inputs - the inputs should be
the first, last and step read from the .log file
(function called by the check_files function)*/
static void check_old_dat(int fst,int lst,int stp)
{
   int ie,ic;
   int fc,lc,dc,pc;

   fdat=fopen(dat_file,"rb");
   error_root(fdat==NULL,1,"check_old_dat [odd_df2_4fop.c]",
              "Unable to open data file");

   check_file_head();

   fc=0;
   ic=0;
   lc=0;
   dc=0;
   pc=0;
   ie=0x0;

   while (read_data()==1)
   {
      pc=lc;
      lc=data.nc;
      ic+=1;

      if (ic==1)
         fc=lc;
      else if (ic==2)
         dc=lc-fc;
      else if ((ic>2)&&(lc!=(pc+dc)))
         ie|=0x1;
   }

   fclose(fdat);

   error_root(ic==0,1,"check_old_dat [odd_df2_4fop.c]",
              "No data records found");
   error_root((ie&0x1)!=0x0,1,"check_old_dat [odd_df2_4fop.c]",
              "Configuration numbers are not equally spaced");
   error_root((fst!=fc)||(lst!=lc)||(stp!=dc),1,"check_old_dat [odd_df2_4fop.c]",
              "Configuration range is not as reported in the log file");
}


/*function used to check compatibility with the log and dat files already written
(as safety measure if -a is not given but the log and dat file are present, they won't be
overwritten but instead an error will be raised)
(function called by the main)*/
static void check_files(void)
{
   int fst,lst,stp; /*local variables to check first, last and step of the  previous configurations scan*/

   ipgrd[0]=0; /*ipgrd[0]=0 means that the process grid has not changed (true by default)*/
   ipgrd[1]=0; /*ipgrd[1]=0 means that the process block size has not changed (true by default)*/
   
   /*the check is done only on the first process*/

   if (my_rank==0)
   {
      if (append) /*if the run is a continuation of a previous run the compatibility with old files is checked*/
      {
         check_old_log(&fst,&lst,&stp); /*fst, lst, stp are read from the .log file of the previous run*/
         check_old_dat(fst,lst,stp); /*compatibility check between the old .log file and the old .dat file*/

         /*raise an error if the previous and the current step of the scan are different
         (except for the case in which in the previous scan there was only one configuration,
         i.e. the case in which first=last)*/
         error_root((fst!=lst)&&(stp!=step),1,"check_files [odd_df2_4fop.c]",
                    "Continuation run:\n"
                    "Previous run had a different configuration separation");
         
         /*raise an error if the current scan does not continue the previous one*/
         error_root(first!=lst+step,1,"check_files [odd_df2_4fop.c]",
                    "Continuation run:\n"
                    "Configuration range does not continue the previous one");
      }
      else /*if the run is a new run an error is raised to avoid overwriting existing files*/
      {
         /*attempt to read log_file: if that is possible an error is raised
         as to avoid overwriting the .log file*/
         fin=fopen(log_file,"r");
         error_root(fin!=NULL,1,"check_files [odd_df2_4fop.c]",
                    "Attempt to overwrite old *.log file");
         
         /*attempt to read dat_file: if that is possible an error is raised
         as to avoid overwriting the .dat file*/
         fdat=fopen(dat_file,"r");
         error_root(fdat!=NULL,1,"check_files [odd_df2_4fop.c]",
                    "Attempt to overwrite old *.dat file");
         
         /*creates of the .dat file and checks whether the operation was successful*/
         fdat=fopen(dat_file,"wb");
         error_root(fdat==NULL,1,"check_files [odd_df2_4fop.c]",
                    "Unable to open data file");
         
         /*the structure file_head containing correlators info is written to the .dat file,
         then the .dat file is closed*/
         write_file_head(); /*global file_head structure saved on the .dat file*/
         fclose(fdat); /*.dat file closed*/
      }
   }
}


/*** output functions ***/

/*function that sets the .log file as stdout and prints there all the information
related to the simulation (parameters, hardware specifics ecc.)
(function called by the main)*/
static void print_info(void)
{
   int i,isap,idfl;
   long ip;
   lat_parms_t lat;

   tm_parms_t tm; /*local structure containing the copy of the global tm structure*/

   tm = tm_parms(); /*tm gets assigned to the global structure tm with the twisted mass flag*/

   if (my_rank==0)
   {
      ip=ftell(flog);
      fclose(flog);

      if (ip==0L)
         remove("STARTUP_ERROR");

      if (append)
         flog=freopen(log_file,"a",stdout);
      else
         flog=freopen(log_file,"w",stdout);

      error_root(flog==NULL,1,"print_info [odd_df2_4fop.c]",
                 "Unable to open log file");
      printf("\n");

      if (append)
         printf("Continuation run\n\n");
      else
      {
         printf("Computation of meson correlators with insertion of a deltaF=2 4 fermions operator\n");
         printf("--------------------------------\n\n");
         printf("cnfg   base name: %s\n",nbase);
         printf("output base name: %s\n\n",outbase);
      }

      printf("openQCD version: %s, meson version: %s\n",openQCD_RELEASE,
                                                      mesons_RELEASE);
      if (endian==LITTLE_ENDIAN)
         printf("The machine is little endian\n");
      else
         printf("The machine is big endian\n");
      if (noexp)
         printf("Configurations are read in imported file format\n\n");
      else
         printf("Configurations are read in exported file format\n\n");

      if ((ipgrd[0]!=0)&&(ipgrd[1]!=0))
         printf("Process grid and process block size changed:\n");
      else if (ipgrd[0]!=0)
         printf("Process grid changed:\n");
      else if (ipgrd[1]!=0)
         printf("Process block size changed:\n");

      if ((append==0)||(ipgrd[0]!=0)||(ipgrd[1]!=0))
      {
         printf("%dx%dx%dx%d lattice, ",N0,N1,N2,N3);
         printf("%dx%dx%dx%d local lattice\n",L0,L1,L2,L3);
         printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
         printf("%dx%dx%dx%d process block size\n",
                NPROC0_BLK,NPROC1_BLK,NPROC2_BLK,NPROC3_BLK);

         if (append)
            printf("\n");
         else
            printf("Open boundary conditions on the quark fields\n\n"); 
      }
      
      
      if (append)
      {
         printf("Random number generator:\n");

         if (norng)
            printf("level = %d, seed = %d, effective seed = %d\n\n",
                   level,seed,seed^(first-step));
         else
         {
            printf("State of ranlxs and ranlxd reset to the\n");
            printf("last exported state\n\n");
         }
      }
      else
      {
         printf("Random number generator:\n");
         printf("level = %d, seed = %d\n\n",level,seed);

         lat=lat_parms();

         printf("Measurements:\n");
         printf("nprop     = %i\n",nprop);
         printf("ncorr     = %i\n",ncorr);
         printf("nnoise    = %i\n",nnoise);
         if (noisetype==Z2_NOISE)
            printf("noisetype = Z2\n");
         if (noisetype==GAUSS_NOISE)
            printf("noisetype = GAUSS\n");
         if (noisetype==U1_NOISE)
            printf("noisetype = U1\n");
         if (noisetype==ONE_COMPONENT)
            printf("noisetype = ONE_COMPONENT\n");
         printf("csw       = %.6f\n",lat.csw);
         printf("cF        = %.6f\n",lat.cF);
	      printf("eoflg     = %i\n\n",tm.eoflg); /*print the twisted mass flag eoflag to the .log file*/

         printf("random_conf     = %i\n",random_conf);
         printf("check_gauge_inv = %i\n\n",check_gauge_inv);

         for (i=0; i<nprop; i++)
         {
            printf("Propagator %i:\n",i);
            printf("kappa  = %.6f\n",kappas[i]);
            printf("isp    = %i\n",isps[i]);
            printf("mu     = %.6f\n\n",mus[i]);
         }

         for (i=0; i<ncorr; i++)
         {
            printf("Correlator %i:\n",i);
            printf("iprop  = %i %i %i %i\n",props1[i],props2[i],props3[i],props4[i]);
            printf("gamma_A_B   = %i %i\n",typeA[i],typeB[i]);
            printf("x0     = %i\n",x0s[i]);
            printf("z0     = %i\n\n",z0s[i]);
         }

         printf("Odd DeltaF=2 4 Fermions operators: VA, AV, SP, PS, TT~\n\n");
        
      }

      print_solver_parms(&isap,&idfl);

      if (isap)
         print_sap_parms(0);

      if (idfl)
         print_dfl_parms(0);

      printf("Configurations no %d -> %d in steps of %d\n\n",
             first,last,step);
      fflush(flog);
   }
}


/*** functions handling the operator_info struct ***/

/*function used to initialize the structure with the GAMMA1 and GAMMA2 of the 4 fermions operators*/
static void init_4foperators(void)
{
   /*first for each of the 5 odd operators we allocate the needed memory*/

   /*0 is related to the VA operator*/
   operator_info[0].ngammas = 4;  /*the sum is over mu, so over 4 terms*/
   operator_info[0].type1 = malloc(4*sizeof(int)); /*array needed to store the GAMMA1 of each piece of the sum*/
   operator_info[0].type2 = malloc(4*sizeof(int)); /*array needed to store the GAMMA2 of each piece of the sum*/
   operator_info[0].weights = malloc(4*sizeof(double)); /*array of weights in the sum*/

   /*1 is related to the AV operator*/
   operator_info[1].ngammas = 4;  /*the sum is over mu, so over 4 terms*/
   operator_info[1].type1 = malloc(4*sizeof(int)); /*array needed to store the GAMMA1 of each piece of the sum*/
   operator_info[1].type2 = malloc(4*sizeof(int)); /*array needed to store the GAMMA2 of each piece of the sum*/
   operator_info[1].weights = malloc(4*sizeof(double)); /*array of weights in the sum*/

   /*2 is related to the SP operator*/
   operator_info[2].ngammas = 1;  /*the sum is over mu, so over 4 terms*/
   operator_info[2].type1 = malloc(1*sizeof(int)); /*array needed to store the GAMMA1 of each piece of the sum*/
   operator_info[2].type2 = malloc(1*sizeof(int)); /*array needed to store the GAMMA2 of each piece of the sum*/
   operator_info[2].weights = malloc(1*sizeof(double)); /*array of weights in the sum*/

   /*3 is related to the PS operator*/
   operator_info[3].ngammas = 1;  /*the sum is over mu, so over 4 terms*/
   operator_info[3].type1 = malloc(1*sizeof(int)); /*array needed to store the GAMMA1 of each piece of the sum*/
   operator_info[3].type2 = malloc(1*sizeof(int)); /*array needed to store the GAMMA2 of each piece of the sum*/
   operator_info[3].weights = malloc(1*sizeof(double)); /*array of weights in the sum*/

   /*4 is related to the TT~ operator*/
   operator_info[4].ngammas = 6;  /*the sum is over mu, so over 4 terms*/
   operator_info[4].type1 = malloc(6*sizeof(int)); /*array needed to store the GAMMA1 of each piece of the sum*/
   operator_info[4].type2 = malloc(6*sizeof(int)); /*array needed to store the GAMMA2 of each piece of the sum*/
   operator_info[4].weights = malloc(6*sizeof(double)); /*array of weights in the sum*/

   /*correct memory allocation is now checked*/
   error((operator_info[0].type1==NULL)||(operator_info[0].type2==NULL)||(operator_info[0].weights==NULL)||
         (operator_info[1].type1==NULL)||(operator_info[1].type2==NULL)||(operator_info[1].weights==NULL)||
         (operator_info[2].type1==NULL)||(operator_info[2].type2==NULL)||(operator_info[2].weights==NULL)||
         (operator_info[3].type1==NULL)||(operator_info[3].type2==NULL)||(operator_info[3].weights==NULL)||
         (operator_info[4].type1==NULL)||(operator_info[4].type2==NULL)||(operator_info[4].weights==NULL),
         1,"init_4foperators [odd_df2_4fop.c]","Out of memory");

   /*the array are now initialized with the correct structures*/

   /*for VA the sum is over gamma_mu * gamma_mu gamma5*/
   operator_info[0].type1[0] = GAMMA0_TYPE;     /*these 4 are gammma_mu*/
   operator_info[0].type1[1] = GAMMA1_TYPE;
   operator_info[0].type1[2] = GAMMA2_TYPE;
   operator_info[0].type1[3] = GAMMA3_TYPE;

   operator_info[0].type2[0] = GAMMA0GAMMA5_TYPE; /*these 4 are gamma_mu gamma5*/
   operator_info[0].type2[1] = GAMMA1GAMMA5_TYPE;
   operator_info[0].type2[2] = GAMMA2GAMMA5_TYPE;
   operator_info[0].type2[3] = GAMMA3GAMMA5_TYPE;

   operator_info[0].weights[0] = 1.0; /*all terms summed with +*/
   operator_info[0].weights[1] = 1.0;
   operator_info[0].weights[2] = 1.0;
   operator_info[0].weights[3] = 1.0;

   /*for AV the sum is over gamma_mu gamma5 * gamma_mu*/
   operator_info[1].type1[0] = GAMMA0GAMMA5_TYPE;     /*these 4 are gammma_mu gamma5*/
   operator_info[1].type1[1] = GAMMA1GAMMA5_TYPE;
   operator_info[1].type1[2] = GAMMA2GAMMA5_TYPE;
   operator_info[1].type1[3] = GAMMA3GAMMA5_TYPE;

   operator_info[1].type2[0] = GAMMA0_TYPE; /*these 4 are gamma_mu gamma5*/
   operator_info[1].type2[1] = GAMMA1_TYPE;
   operator_info[1].type2[2] = GAMMA2_TYPE;
   operator_info[1].type2[3] = GAMMA3_TYPE;

   operator_info[1].weights[0] = 1.0; /*all terms summed with +*/
   operator_info[1].weights[1] = 1.0;
   operator_info[1].weights[2] = 1.0;
   operator_info[1].weights[3] = 1.0;

   /*for SP there's just the piece identity * gamma5*/
   operator_info[2].type1[0] = ONE_TYPE; /*identity*/
   operator_info[2].type2[0] = GAMMA5_TYPE; /*gamma5*/
   operator_info[2].weights[0] = 1.0;

   /*for PS there's just the piece gamma5 * identity*/
   operator_info[3].type1[0] = GAMMA5_TYPE; /*gamma5*/
   operator_info[3].type2[0] = ONE_TYPE; /*identity*/
   operator_info[3].weights[0] = 1.0;

   /*for TT~ the sum is over gamma_mu gamma_nu * gamma_nu gamma_mu, with mu < nu*/
   operator_info[4].type1[0] = GAMMA0GAMMA1_TYPE;     /*these 4 are gammma_mu gamma_nu*/
   operator_info[4].type1[1] = GAMMA0GAMMA2_TYPE;
   operator_info[4].type1[2] = GAMMA0GAMMA3_TYPE;
   operator_info[4].type1[3] = GAMMA1GAMMA2_TYPE;
   operator_info[4].type1[4] = GAMMA1GAMMA3_TYPE;
   operator_info[4].type1[5] = GAMMA2GAMMA3_TYPE;

   operator_info[4].type2[0] = GAMMA2GAMMA3_TYPE; /*these 4 are gamma5 gamma_mu gamma_vu*/
   operator_info[4].type2[1] = GAMMA1GAMMA3_TYPE;
   operator_info[4].type2[2] = GAMMA1GAMMA2_TYPE;
   operator_info[4].type2[3] = GAMMA0GAMMA3_TYPE;
   operator_info[4].type2[4] = GAMMA0GAMMA2_TYPE;
   operator_info[4].type2[5] = GAMMA0GAMMA1_TYPE;

   operator_info[4].weights[0] = 1.0;
   operator_info[4].weights[1] = -1.0;
   operator_info[4].weights[2] = -1.0;
   operator_info[4].weights[3] = 1.0;
   operator_info[4].weights[4] = -1.0;
   operator_info[4].weights[5] = 1.0;

   /* T = i/2 times commutator and T~ = i/2 gamma5 times commutator,
   the commutator cancels the two, so there is an overall minus and relative minuses  coming from the
   product of gamma5 and gamma_mu gamma_nu*/

   if (my_rank==0)
      printf("VA, AV, SP, PS, TT~ operators initialized\n\n");

}


/*** functions handling the random number generator (rng) ***/

/*function used to initialize the random number generator (rng)
(function called by the main)*/
static void init_rng(void)
{
   int ic;

   if (append) /*if the run is a continuation of a previous run ...*/
   {
      if (norng) /*... but if the option -norng is given ...*/
         start_ranlux(level,seed^(first-step)); /*... traditional generator initialization*/
      else /*... if instead -norng is not given the run start from the saved state of the random generator*/
      {
         /*the random generator is initialized importing the saved state of the random generator from
         the previous run, then the compatibility with the parameters of the current run is checked*/
         ic=import_ranlux(rng_file);
         error_root(ic!=(first-step),1,"init_rng [odd_df2_4fop.c]",
                    "Configuration number mismatch (*.rng file)");
      }
   }
   else /*if the run is a new run ...*/
      start_ranlux(level,seed); /*...traditional generator initialization*/
}


/*function used to save the current state of the random number generators rlxs and rlxd
(and initialize them if needed)
(function called by the main)*/
static void save_ranlux(void)
{
   int nlxs,nlxd; /*number of integers required to save the state of the generators*/

   /*initialize the generators if needed*/

   if (rlxs_state==NULL) /*if the state of the generator is uninitializied (as it is by default)*/
   {
      nlxs=rlxs_size(); /*get the number of integers required to save the state of the rlxs generator*/
      nlxd=rlxd_size(); /*get the number of integers required to save the state of the rlxd generator*/

      /*allocate memory for the two generators states*/
      rlxs_state=malloc((nlxs+nlxd)*sizeof(int)); /*memory allocation for the two arrays*/
      rlxd_state=rlxs_state+nlxs; /*set rlxd_state memory right after rlxs_state memory*/

      /*check on the correct memory allocation*/
      error(rlxs_state==NULL,1,"save_ranlux [odd_df2_4fop.c]",
            "Unable to allocate state arrays");
   }

   /*save the state of the generators*/

   rlxs_get(rlxs_state); /*store the state of the rlxs generator in rlsxs_state*/
   rlxd_get(rlxd_state); /*store the state of the rlxd generator in rlsxd_state*/
}


/*function that sets the states of the rlxs and rlxd generators to the
states currently saved in the globl variables rlxs_state and rlxd_state
(function called by the main)*/
static void restore_ranlux(void)
{
   rlxs_reset(rlxs_state); /*sets the rlxs_generator to the state rlxs_state*/
   rlxd_reset(rlxd_state); /*sets the rlxd_generator to the state rlxd_state*/
}


/*** solver related functions ***/

/*function used to increase nws, nwv, nwvd if they are smaller than
the minimum value allowed (specified in the parameters of the deflation subspace)
(function called by the wsize function)*/
static void dfl_wsize(int *nws,int *nwv,int *nwvd)
{
   dfl_parms_t dp; /*Parameters of Deflation subspace*/
   dfl_pro_parms_t dpp; /*Parameters of Projectors of Deflation subspace*/

   /*dp and dpp are now set to two global structures:
      - dp to dfl, the structure containing the parameters of the deflation subspace
      - dpp to dfl_pro,the structure containing the parameters of projector on the deflation subspace
   */

   dp=dfl_parms(); /*dp is set to the global structure dfl*/
   dpp=dfl_pro_parms(); /*dp is set to the global structure dfl_pro*/

   /*if the content of nws, nwv, nwvd is smaller than the related parameters of the deflation subspace,
   then they are set to the parameter taken from the deflation subspace (??)
   ( MAX(n,m) sets n to be the greatest between n and m) */

   MAX(*nws,dp.Ns+2); /*nws set to dp.Ns+2 if dp.Ns+2 is bigger*/
   MAX(*nwv,2*dpp.nkv+2); /*nwv set to 2*dpp.nkv+2 if 2*dpp.nkv+2 is bigger */
   MAX(*nwvd,4); /*nwvd set to 4 if nwvd<4*/
}


/*function that sets nws, nwsd, nwv, nwvd according to the
solver method chosen in the input file
(function called by the main)*/
static void wsize(int *nws,int *nwsd,int *nwv,int *nwvd)
{
   int nsd; /*number of double-precision spinor fields to be allocated*/
   solver_parms_t sp; /*local variable with the solver parameters*/

   (*nws)=0;
   (*nwsd)=0;
   (*nwv)=0;
   (*nwvd)=0;

   /*nws, nwsd, nwv, nwvd are initialized to 0*/

   sp=solver_parms(0); /*set sp to the global structure with the solver parameters*/
   nsd=2*file_head.nnoise+2+1+2; /*nsd set to 2 times the number of noise vectors +2 +1 (xiA,zetaA x nnoise + xiB,zetaB + a tmp spinor+2tmp spinors) */

   /*depending on the solver method nws, nwsd, nwv, nwvd get modified
   (they increase if they are smaller than some minimum)*/

   if (sp.solver==CGNE)
   {
      MAX(*nws,5);
      MAX(*nwsd,nsd+3);
   }
   else if (sp.solver==SAP_GCR)
   {
      MAX(*nws,2*sp.nkv+1);
      MAX(*nwsd,nsd+2);
   }
   else if (sp.solver==DFL_SAP_GCR)
   {
      MAX(*nws,2*sp.nkv+2);
      MAX(*nwsd,nsd+4);
      dfl_wsize(nws,nwv,nwvd); /*compatibility check with the specifics of the deflation subspace (??)*/
   }
   else /*raises an error if the solver method is not one of the allowed ones*/
      error_root(1,1,"wsize [odd_df2_4fop.c]",
                 "Unknown or unsupported solver");
}


/*** early termination functions ***/

/*function that sets the endflag if in the log directory there is a file
with the .end extension and with the same name of the run
(that's a gentle way to kill the program execution from terminal)
(function called by the main)*/
static void check_endflag(int *iend)
{
   /*only on process 0 checks if the endflag has to be set*/
   if (my_rank==0)
   {
      fend=fopen(end_file,"r"); /*tries to open the .end file*/

      if (fend!=NULL) /*if .end file has been opened succesfully ...*/
      {
         fclose(fend); /*... closes the .end file*/
         remove(end_file); /*... removes the .end file*/
         (*iend)=1; /*... sets the end flag on (i.e. to 1)*/
         printf("End flag set, run stopped\n\n"); /*... writes about the early termination in the .log file*/
      }
      else /*if instead there is not a .end file ...*/
         (*iend)=0; /*... the endflag stays off (i.e. to 0) and the run continues*/
   }

   /*the endflag is then broadcasted to all other processes*/
   MPI_Bcast(iend,1,MPI_INT,0,MPI_COMM_WORLD);
}


/*** computation functions ***/

/*function used to set to 1 only one component of a spinor
(function called by the random_spinor function)*/
void fill_one_component(spinor_dble *sd)
{
   static int counter = 0; /*counter used to discriminate which component to fill*/
   const int max_counter = 12; /*the maximum number of components is 12*/

   /*each time the fuction is called counter is increased by 1, from 0 up to 11, then it starts
   again from 0, and depending on counter a different component is initialized*/

   /*
   printf("\n counter = %i \n",counter);
   fflush(flog);
   */

  /*CHECK --->  NNOISE=12 !!!!!!!!*/

   if (counter==0)
   {
      (*sd).c1.c1.re=1.0; /*dirac 1, color 1 set to 1.0*/
   }
   else if (counter==1)
   {
      (*sd).c1.c2.re=1.0; /*dirac 1, color 2 set to 1.0*/
   }
   else if (counter==2)
   {
      (*sd).c1.c3.re=1.0; /*dirac 1, color 3 set to 1.0*/
   }
   else if (counter==3)
   {
      (*sd).c2.c1.re=1.0; /*dirac 2, color 1 set to 1.0*/
   }
   else if (counter==4)
   {
      (*sd).c2.c2.re=1.0; /*dirac 2, color 2 set to 1.0*/
   }
   else if (counter==5)
   {
      (*sd).c2.c3.re=1.0; /*dirac 2, color 3 set to 1.0*/
   }
   else if (counter==6)
   {
      (*sd).c3.c1.re=1.0; /*dirac 3, color 1 set to 1.0*/
   }
   else if (counter==7)
   {
      (*sd).c3.c2.re=1.0; /*dirac 3, color 2 set to 1.0*/
   }
   else if (counter==8)
   {
      (*sd).c3.c3.re=1.0; /*dirac 3, color 3 set to 1.0*/
   }
   else if (counter==9)
   {
      (*sd).c4.c1.re=1.0; /*dirac 4, color 1 set to 1.0*/
   }
   else if (counter==10)
   {
      (*sd).c4.c2.re=1.0; /*dirac 4, color 2 set to 1.0*/
   }
   else if (counter==11)
   {
      (*sd).c4.c3.re=1.0; /*dirac 4, color 3 set to 1.0*/
   }

   counter = (counter+1)%max_counter;

}


/*function used to create a random spinor:
the array eta is first set to 0 on the whole lattice, then at the timeslice x0 
it gets filled with random doubles according to the random method specified (globally)
(function called by the propagators function)*/
static void random_spinor(spinor_dble *eta, int x0)
{
   
   int y0; /* = x0 after a change of variable where the center is in the cartesian coordinate of the local process*/
   int iy; /*index running on the time extent of the lattice*/
   int ix; /*index of the point on the local lattice*/

   set_sd2zero(VOLUME,eta); /*eta is set to 0 on the whole lattice volume*/

   y0=x0-cpr[0]*L0;/*y0 is set to the distance between x0 and the center of the local process*/
   /*(cpr are the coordinates of the local process)*/

   /*if x0 lies inside the block of the current local process
   then the random vector should be generated in this process*/

   if ((y0>=0)&&(y0<L0)) /*i.e. if x0 inside the block of the current process ...*/
   {

      /*... then in the x0 timeslice eta is generated randomly*/

      if (noisetype==Z2_NOISE) /*if  the random generation is of type Z2*/
      {
         for (iy=0;iy<(L1*L2*L3);iy++) /*loop over the timeslice*/
         {
            ix=ipt[iy+y0*L1*L2*L3]; /*index of the point on the local lattice*/
            random_Z2_sd(1,eta+ix); /*random Z2 generation of the spinor entry (just 1) at position ix*/
         }
      }
      else if (noisetype==GAUSS_NOISE) /*if the random generation is gaussian*/
      {
         for (iy=0;iy<(L1*L2*L3);iy++) /*loop over the timeslice*/
         {
            ix=ipt[iy+y0*L1*L2*L3]; /*index of the point on the local lattice*/
            random_sd(1,eta+ix,1.0); /*random gaussian generation of the spinor entry (just 1) at position ix*/
         }
      }
      else if (noisetype==U1_NOISE) /*if the random generation U1*/
      {
         for (iy=0;iy<(L1*L2*L3);iy++) /*loop over the timeslice*/
         {
            ix=ipt[iy+y0*L1*L2*L3]; /*index of the point on the local lattice*/
            random_U1_sd(1,eta+ix); /*random U1 generation of the spinor entry (just 1) at position ix*/
         }
      }
      else if (noisetype==ONE_COMPONENT) /*if only one component of the spinor has to be filled*/
      {
         if ( (cpr[1]==0) && (cpr[2]==0) && (cpr[3]==0) ) /*the component to be filled is on the process timeslice with 0 space components*/
         {
            iy=0;
            ix=ipt[iy+y0*L1*L2*L3];
            fill_one_component(eta+ix);
         }
      }

      /*(the above random generations are done entry by entry,
      they can't be done in one shot like random_sd(L1*L2*L3, eta)
      because the entries are on a timeslice, so they are not contiguous)*/

      /*add another if with:
      cpr[1]=cpr[2]=cpr[3]=0
      iy = 0
      y0 lo stesso
      poi inizializzaione secondo indici di dirac*/

   }

}


/* out_spinor = gamma5 * GAMMA^dagger * eta
function that constructs the source term of the Dirac equation from the random spinor eta according to
what specified in the documentation (inverting the Dirac equation with this source zeta and xi can be found):
   - eta        : is the random spinor passed as input (it is not modified) 
   - out_spinor : it is set to be gamma5 * GAMMA^dagger * eta
                  that according to the documentation is the source for the Dirac equation whose 
                  inversion gives us zeta and xi, 
   - type       : it is what speicifies GAMMA, and it should be either type_A or type_B specified in the
                  input file (the product of gamma matrices is done according to the table 1 of the documentation)
(function called by the propagators function)*/
void mul_g5_GAMMAdag(spinor_dble *eta, int type, spinor_dble *out_spinor)
{

   /*
   - assign_msd2sd : sets the second spinor to be equal to minus the first one
   - assign_sd2sd : sets the second spinor to be equal to the first one
   - mulgigj : multiplies the spinor by gamma_i gamma_j

   VOLUME means that these operation are done on the whole lattice
   */

   switch (type)
   {
      case GAMMA0_TYPE:
         assign_msd2sd(VOLUME,eta,out_spinor);
         mulg0g5_dble(VOLUME,out_spinor);
         break;
      case GAMMA1_TYPE:
         assign_msd2sd(VOLUME,eta,out_spinor);
         mulg1g5_dble(VOLUME,out_spinor);
         break;
      case GAMMA2_TYPE:
         assign_msd2sd(VOLUME,eta,out_spinor);
         mulg2g5_dble(VOLUME,out_spinor);
         break;
      case GAMMA3_TYPE:
         assign_msd2sd(VOLUME,eta,out_spinor);
         mulg3g5_dble(VOLUME,out_spinor);
         break;
      case GAMMA5_TYPE:
         assign_sd2sd(VOLUME,eta,out_spinor);
         break;
      case GAMMA0GAMMA1_TYPE:
         assign_sd2sd(VOLUME,eta,out_spinor);
         mulg2g3_dble(VOLUME,out_spinor);
         break;
      case GAMMA0GAMMA2_TYPE:
         assign_msd2sd(VOLUME,eta,out_spinor);
         mulg1g3_dble(VOLUME,out_spinor);
         break;
      case GAMMA0GAMMA3_TYPE:
         assign_sd2sd(VOLUME,eta,out_spinor);
         mulg1g2_dble(VOLUME,out_spinor);
         break;
      case GAMMA0GAMMA5_TYPE:
         assign_sd2sd(VOLUME,eta,out_spinor);
         mulg0_dble(VOLUME,out_spinor);
         break;
      case GAMMA1GAMMA2_TYPE:
         assign_sd2sd(VOLUME,eta,out_spinor);
         mulg0g3_dble(VOLUME,out_spinor);
         break;
      case GAMMA1GAMMA3_TYPE:
         assign_msd2sd(VOLUME,eta,out_spinor);
         mulg0g2_dble(VOLUME,out_spinor);
         break;
      case GAMMA1GAMMA5_TYPE:
         assign_sd2sd(VOLUME,eta,out_spinor);
         mulg1_dble(VOLUME,out_spinor);
         break;
      case GAMMA2GAMMA3_TYPE:
         assign_sd2sd(VOLUME,eta,out_spinor);
         mulg0g1_dble(VOLUME,out_spinor);
         break;
      case GAMMA2GAMMA5_TYPE:
         assign_sd2sd(VOLUME,eta,out_spinor);
         mulg2_dble(VOLUME,out_spinor);
         break;
      case GAMMA3GAMMA5_TYPE:
         assign_sd2sd(VOLUME,eta,out_spinor);
         mulg3_dble(VOLUME,out_spinor);
         break;
      case ONE_TYPE:
         assign_sd2sd(VOLUME,eta,out_spinor);
         mulg5_dble(VOLUME,out_spinor);
         break;
      default:
         error_root(1,1,"mul_g5_GAMMAdag [odd_df2_4fop.c]",
                 "Unknown or unsupported type");
   }
}


/* out_spinor = (Dw + i mu gamma5)^(-1) * source_spinor  (??)
function that inverts the Dirac equation and sets the output spinor to be like the xi of eq 6 in the documentatation:
   - prop          : is the index of the quark whose Dirac operator gets inverted
   - source_spinor : is the source for the Dirac equation that gets inverted
   - out_spinor    : where the result is stored
   - status        : where the status of the inversion gets stored
(function called by the correlators function)*/
static void solve_dirac(int prop, spinor_dble *source_spinor, spinor_dble *out_spinor,
                        int *status)
{
   solver_parms_t sp; /*local structure with the solver parameters*/
   sap_parms_t sap; /*structure related to the sap solver*/

   sp=solver_parms(isps[prop]); /*sp assigned to global solver structure depending on solver id of the propagator*/
   set_sw_parms(0.5/kappas[prop]-4.0); /*sets the bare quark mass to that of the propagator with index prop*/

   /*depending on the chosen solver a different method is used*/

   if (sp.solver==CGNE) /*if the solver is CGNE*/
   {
      mulg5_dble(VOLUME,source_spinor); /*multiplies source_spinor by gamma5*/

      tmcg(sp.nmx,sp.res,mus[prop],source_spinor,source_spinor,status); /*source_spinor is set to be the solution of the Wilson-Dirac equation (with a twisted mass term)*/

      /*on process 0 writes on the log file the status of the solver*/
      if (my_rank==0)
         printf("%i\n",status[0]);
      
      /*raises an error if the solver was unsuccesful (i.e. status<0)*/
      error_root(status[0]<0,1,"solve_dirac [odd_df2_4fop.c]",
                 "CGNE solver failed (status = %d)",status[0]);

      Dw_dble(-mus[prop],source_spinor,out_spinor); /*out_spinor = (Dw + i (-mu) gamma5) source_spinor*/

      /*after the above function out_spinor becomes
      out_spinor = (Dw^dagger - i mu gamma5)^(-1) gamma5 * (the original source_spinor) */

      mulg5_dble(VOLUME,out_spinor); /*out_spinor gets multiplied by gamma5*/

      /*after this last function out_spinor in now equal to the xi of equation 6
      of the documentation -> with maybe the twisted mass as extra (??)*/

   }
   else if (sp.solver==SAP_GCR) /*if the solver is SAP_GCR*/
   {

      /*initialization of sap solver*/
      sap=sap_parms(); /*sap set to the global structure with the parameters of the SAP preconditioner*/
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy); /*set the parameters of the sap preconditioner*/

      /*out_spinor is set as the solution of the dirac equation with source source_spinor (using sap_gcr solver)*/
      sap_gcr(sp.nkv,sp.nmx,sp.res,mus[prop],source_spinor,out_spinor,status);

      /*on process 0 writes on the log file the status of the solver*/
      if (my_rank==0)
         printf("%i\n",status[0]);
      
      /*raises an error if the solver was unsuccesful (i.e. status<0)*/
      error_root(status[0]<0,1,"solve_dirac [odd_df2_4fop.c]",
                 "SAP_GCR solver failed (status = %d)",status[0]);
      
   }
   else if (sp.solver==DFL_SAP_GCR) /*if the solver is DFL_SAP_GCR*/
   {
      /*initialization of sap solver*/
      sap=sap_parms(); /*sap set to the global structure with the parameters of the SAP preconditioner*/
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy); /*set the parameters of the sap preconditioner*/

      /*out_spinor is set as the solution of the dirac equation with source source_spinor (using dfl_sap_gcr solver)*/
      dfl_sap_gcr2(sp.nkv,sp.nmx,sp.res,mus[prop],source_spinor,out_spinor,status);

      /*on process 0 writes on the log file the status of the solver*/
      if (my_rank==0)
         printf("%i %i\n",status[0],status[1]);
      
      /*raises an error if the solver was unsuccesful (i.e. status<0)*/
      error_root((status[0]<0)||(status[1]<0),1,
                 "solve_dirac [odd_df2_4fop.c]","DFL_SAP_GCR solver failed "
                 "(status = %d,%d)",status[0],status[1]);
      
   }
   else /*if the specified solver is unknown raises an error*/
      error_root(1,1,"solve_dirac [odd_df2_4fop.c]",
                 "Unknown or unsupported solver");
}


/* out_spinor = GAMMA^dagger *gamma5 * xi
function that constructs the product that appears inside the correlator according to
what specified in the documentation:
   - xi         : is the xi of the documentation (either xi_A or xi_B)
   - out_spinor : it is set to be GAMMA^dagger *gamma5 * xi
                  that according to the documentation should later be multiplied to zeta, 
   - type       : it is what speicifies GAMMA, and it should be either type_1 or type_2 specified in the
                  input file (the product of gamma matrices is done according to the table 1 of the documentation)
(function called by the propagators function)*/
void mul_GAMMAdag_g5(spinor_dble *xi,int type,spinor_dble *out_spinor)
{
   /*
   - assign_msd2sd : sets the second spinor to be equal to minus the first one
   - assign_sd2sd : sets the first spinor to be equal to the first one
   - mulgigj : multiplies the spinor by gamma_i gamma_j

   VOLUME means that these operation are done on the whole lattice
   */

   switch (type)
   {
      case GAMMA0_TYPE:
         assign_sd2sd(VOLUME,xi,out_spinor);
         mulg0g5_dble(VOLUME,out_spinor);
         break;
      case GAMMA1_TYPE:
         assign_sd2sd(VOLUME,xi,out_spinor);
         mulg1g5_dble(VOLUME,out_spinor);
         break;
      case GAMMA2_TYPE:
         assign_sd2sd(VOLUME,xi,out_spinor);
         mulg2g5_dble(VOLUME,out_spinor);
         break;
      case GAMMA3_TYPE:
         assign_sd2sd(VOLUME,xi,out_spinor);
         mulg3g5_dble(VOLUME,out_spinor);
         break;
      case GAMMA5_TYPE:
         assign_sd2sd(VOLUME,xi,out_spinor);
         break;
      case GAMMA0GAMMA1_TYPE:
         assign_sd2sd(VOLUME,xi,out_spinor);
         mulg2g3_dble(VOLUME,out_spinor);
         break;
      case GAMMA0GAMMA2_TYPE:
         assign_msd2sd(VOLUME,xi,out_spinor);
         mulg1g3_dble(VOLUME,out_spinor);
         break;
      case GAMMA0GAMMA3_TYPE:
         assign_sd2sd(VOLUME,xi,out_spinor);
         mulg1g2_dble(VOLUME,out_spinor);
         break;
      case GAMMA0GAMMA5_TYPE:
         assign_msd2sd(VOLUME,xi,out_spinor);
         mulg0_dble(VOLUME,out_spinor);
         break;
      case GAMMA1GAMMA2_TYPE:
         assign_sd2sd(VOLUME,xi,out_spinor);
         mulg0g3_dble(VOLUME,out_spinor);
         break;
      case GAMMA1GAMMA3_TYPE:
         assign_msd2sd(VOLUME,xi,out_spinor);
         mulg0g2_dble(VOLUME,out_spinor);
         break;
      case GAMMA1GAMMA5_TYPE:
         assign_msd2sd(VOLUME,xi,out_spinor);
         mulg1_dble(VOLUME,out_spinor);
         break;
      case GAMMA2GAMMA3_TYPE:
         assign_sd2sd(VOLUME,xi,out_spinor);
         mulg0g1_dble(VOLUME,out_spinor);
         break;
      case GAMMA2GAMMA5_TYPE:
         assign_msd2sd(VOLUME,xi,out_spinor);
         mulg2_dble(VOLUME,out_spinor);
         break;
      case GAMMA3GAMMA5_TYPE:
         assign_msd2sd(VOLUME,xi,out_spinor);
         mulg3_dble(VOLUME,out_spinor);
         break;
      case ONE_TYPE:
         assign_sd2sd(VOLUME,xi,out_spinor);
         mulg5_dble(VOLUME,out_spinor);
         break;
      default:
         error_root(1,1,"mul_GAMMAdag_g5 [odd_df2_4fop.c]",
                 "Unknown or unsupported type");
   }
}


/*core function used to compute the correlator once all the input
variable have been specified in the corresponding global structures
(function called by the set_data function)*/
static void correlators(void)
{

   /** declaration of local variables **/

   int icorr; /*index running over the correlators to be computed*/
   int iop; /*index running over the 5 different odd 4f operators*/
   int ipiece; /*index running over the pieces to be summed over in each of the 5 operators*/
   int inoise_A, inoise_B; /*indeces running over the noise vectors to be generated*/

   int stat[4]; /*status array used in the Dirac inversion*/

   int y0; /*index running on the local time extent (from 0 to L0)*/
   int l; /*index running on the local space extent (from 0 to L1*L2*L3) (and also used to fill the correlators)*/

   int iy; /*index on the (global) lattice obtained from the cartesian coordinates of the  (local) point y*/

   spinor_dble **wsd; /*workspace of double-precision spinor fields */
   spinor_dble **zeta_A; /*array with nnoise different zeta_A, as in the documentation*/
   spinor_dble **xi_A; /*array with nnoise different xi_A ,as in the documentation*/
   spinor_dble *zeta_B; /*zeta_B as in the documentation*/
   spinor_dble *xi_B; /*xi_B as in the documentation*/
   spinor_dble *source_spinor; /*spinor used as source for the Dirac equation, inverting it zeta and xi can be found*/

   spinor_dble *G1_g5_xi_A; /*spinor that will be set to be GAMMA_1 gamma5 xi_A*/
   spinor_dble *G2_g5_xi_B; /*spinor that will be set to be GAMMA_2 gamma5 xi_B*/


   complex_dble tmp1, tmp2; /*temporary complex variables used to compute the values of the correlators*/

   /** allocation of the spinor fields **/

   /*the spinor fields needed for the computation are here allocated,
   then the correct memory allocation is checked*/

   wsd=reserve_wsd(2*nnoise+2+1+2); /*a workspace with 2xnnoise+2+2 spinor (double) fields is allocated*/

   zeta_B=wsd[0]; /*the first array (field) in wsd is assigned to zeta_B*/
   xi_B=wsd[1]; /*the second array (field) in wsd is assigned to xi_B*/

   source_spinor=wsd[2]; /*the third array (field) in wsd is assigned to source_spinor*/

   G1_g5_xi_A=wsd[3]; /*fourth array (field) assigned to G1_g5_xi_A*/
   G2_g5_xi_B=wsd[4]; /*fifth array (field) assigned to G1_g5_xi_A*/

   zeta_A=malloc(nnoise*sizeof(spinor_dble*)); /*allocation of nnoise spinors for zeta_A (why needed ?? )*/
   xi_A=malloc(nnoise*sizeof(spinor_dble*)); /*allocation of nnoise spinors for xi_A (why needed ?? )*/
   error((zeta_A==NULL)||(xi_A==NULL),1,"correlators [odd_df2_4fop.c]","Out of memory"); /*check on successful allocation*/

   /*zeta_A and xi_A are now set to be the remaining spinors already reserved in wsd*/
   for (inoise_A=0;inoise_A<nnoise;inoise_A++)
   {
      zeta_A[inoise_A]=wsd[5+2*inoise_A];
      xi_A[inoise_A]=wsd[5+2*inoise_A+1];
   }

   /*why is the zeta_A and xi_A allocation needed (??)
   could

   zeta_A = wsd[2]
   xi_A = wsd[2+nnoise]
   
   do the trick ??
   */

   /** initialization of the correlators (tmp counterpars used by local processes) **/

   for (l=0;l<nnoise*nnoise*ncorr*tvals*noperator;l++)
   {
      data.corrConn_tmp[l].re=0.0;
      data.corrConn_tmp[l].im=0.0;
      data.corrDisc_tmp[l].re=0.0;
      data.corrDisc_tmp[l].im=0.0;
   }

   /** loop over the correlators to be computed **/

   for (icorr=0;icorr<ncorr;icorr++) /*icorr ranges from 0 to the number of correlators to be computed*/
   {

      /*there will be a print on the log file prior to every Dirac inversion*/

      /*informative print on process 0*/
      if (my_rank==0)
      {
         printf("Inversions for xi_A and zeta_A :\n");
         printf("   icorr=%i , x0=%i\n",icorr,x0s[icorr]); /*the value of the x0 being used is printed on the .log file*/
      }

      /** loop over the noise vectors to generate nnoise zeta_A and xi_A **/

      for (inoise_A=0;inoise_A<nnoise;inoise_A++) /*inoise_A ranges from 0 to the number of noise vectors*/
      {

         /*informative print on process 0*/
         if (my_rank==0)
            printf("      noise vector A %i\n",inoise_A); /*the index of the noise vector being generated is written on the .log file*/

         /*generation of the random source:
         the eta_A of the documentation is generated: first it set to 0 on the whole space time volume,
         then at the timeslice specified in the input file (that is at x0[icorr]) eta is filled with
         random complex numbers according to the chosen random method (U1, Z2 or GAUSS);
         the eta_A of the documentation is here stored inside source_spinor*/

         random_spinor(source_spinor,x0s[icorr]); /*source_spinor = eta_A (it gets filled at the timeslice x0 with random numbers)*/

         /*computation of zeta_A : using eta_A as source we invert the Dirac equation with he right propagator
         (the one related to the first quark appearing in the correlation function)*/

         /*informative print on process 0*/
            if (my_rank==0)
               printf("         type=%i, prop=%i, status:",ONE_TYPE, props1[icorr]); /*type and index of prop printed on .log*/

         solve_dirac(props1[icorr],source_spinor,zeta_A[inoise_A],stat); /* zeta_A = (D_2 +i mu_2 gamma5)^-1 source_spinor */

         /*analogously we now compute xi_A : first we compute gamma5 GAMMA_A^dag eta_A, then we invert the
         Dirac equation related to the second quark appearing in the correlation function*/

         mul_g5_GAMMAdag(source_spinor,typeA[icorr],source_spinor); /* source_spinor =  gamma5 GAMMA_A^dag eta_A*/

         /*informative print on process 0*/
            if (my_rank==0)
               printf("         type=%i, prop=%i, status:",typeA[icorr], props2[icorr]); /*type and index of prop printed on .log*/

         solve_dirac(props2[icorr], source_spinor, xi_A[inoise_A],stat); /*xi_A = (D_1 +i mu_1 gamma5)^-1 source_spinor */

         /*then since what matters is GAMMA_1^dag gamma5 xi_A that is what we store inside the array xi_A*/

         /*mul_GAMMAdag_g5(xi_A[inoise_A],type1[icorr],xi_A[inoise_A]);*/ /*xi_A set to be GAMMA_1^dag gamma5 xi_A*/

      } /*end of noise loop, nnoise zeta_A and xi_A produced*/


      /*now that we have nnoise zeta_A and nnoise xi_A we loop again over the noise vector, each time we compute
      zeta_B and xi_B and then we compute the connected and disconnected part of the correlators*/

      /*informative print on process 0*/
      if (my_rank==0)
      {
         printf("Inversions for xi_B and zeta_B :\n");
         printf("   icorr=%i , z0=%i\n",icorr,z0s[icorr]); /*the value of the z0 being used is printed on the .log file*/
      }

      for (inoise_B=0;inoise_B<nnoise;inoise_B++) /*inoise ranges from 0 to the number of noise vectors*/
      {

         /*informative print on process 0*/
         if (my_rank==0)
            printf("      noise vector B %i\n",inoise_B); /*the index of the noise vector being generated is written on the .log file*/

         /*following the same procedure as before we now compute zeta_B and xi_B*/

         random_spinor(source_spinor,z0s[icorr]); /*source_spinor = eta_B(it gets filled at the timeslice z0 with random numbers)*/

         /*informative print on process 0*/
         if (my_rank==0)
            printf("         type=%i, prop=%i, status:",ONE_TYPE, props3[icorr]); /*type and index of prop printed on .log*/

         solve_dirac(props3[icorr],source_spinor,zeta_B,stat); /* zeta_B = (D_4 +i mu_4 gamma5)^-1 source_spinor */

         mul_g5_GAMMAdag(source_spinor,typeB[icorr],source_spinor); /* source_spinor =  gamma5 GAMMA_B^dag eta_B*/

         /*informative print on process 0*/
         if (my_rank==0)
            printf("         type=%i, prop=%i, status:",typeB[icorr], props4[icorr]); /*type and index of prop printed on .log*/

         solve_dirac(props4[icorr], source_spinor, xi_B,stat); /*xi_B = (D_3 +i mu_3 gamma5)^-1 source_spinor */

         /*mul_GAMMAdag_g5(xi_B,type2[icorr],xi_B);*/ /*xi_B set to be GAMMA_2^dag gamma5 xi_B*/

         /*now that we have xi_B and zeta_B we loop over all the xi_A and zeta_A already computed and then
         compute nnoise x nnoise correlators given by all the combination of xi_A,zeta_A and xi_B,zeta_B*/

         for (inoise_A=0;inoise_A<nnoise;inoise_A++)
         {

            /*loop over the 5 odd 4f operators*/

            for (iop=0;iop<noperator;iop++)
            {
               
               /*loop over the pieces to be summed over in each operator*/

               for (ipiece=0;ipiece<operator_info[iop].ngammas;ipiece++)
               {
                  /*computation*/

                  /*each piece of the sum has its own GAMMA1 and GAMMA2, so first we compute the following*/

                  mul_GAMMAdag_g5(xi_A[inoise_A],operator_info[iop].type1[ipiece],G1_g5_xi_A); /*G1_g5_xi_A set to be GAMMA_1^dag gamma5 xi_A*/

                  mul_GAMMAdag_g5(xi_B,operator_info[iop].type2[ipiece],G2_g5_xi_B); /*G2_g5_xi_B set to be GAMMA_2^dag gamma5 xi_B*/


                  /*loop over the space time to compute the correlators*/

                  for (y0=0;y0<L0;y0++) /*loop over the time values y0*/
                  {
                     /*code optimization that can be made here:
                        int temp_index = y0*L1*L2*L3;
                        int temp_data_index = inoise+nnoise*(cpr[0]*L0+y0+file_head.tvals*icorr);
                     */
            
                     for (l=0;l<L1*L2*L3;l++) /*sum over the space index l*/
                     {
                        iy = ipt[l+y0*L1*L2*L3]; /*index of the point on the global (??) lattice*/

                        /** Computation of Disconnected Part **/

                        /*first we compute in the given space-time point the two pieces appearing in the disconnected correlator*/
                        tmp1 = spinor_prod_dble(1,0,G1_g5_xi_A+iy,zeta_A[inoise_A]+iy);  /*tmp1 = ( GAMMA_1^dag g5 xi_A )^dag zeta_A*/
                        tmp2 = spinor_prod_dble(1,0,G2_g5_xi_B+iy,zeta_B+iy);  /*tmp2 = ( GAMMA_2^dag g5 xi_B )^dag zeta_B*/
                        /*code optimization: this tmp2 could be brought outside the noise_A loop*/

                        /*then sum their product to the disconnected correlator at y0 (tmp because is only on the local process)*/
                        data.corrDisc_tmp[inoise_A +nnoise*(inoise_B + nnoise*(cpr[0]*L0+y0 + tvals*(iop + noperator*icorr)))].re += (tmp1.re*tmp2.re - tmp1.im*tmp2.im) * operator_info[iop].weights[ipiece];
                        data.corrDisc_tmp[inoise_A +nnoise*(inoise_B + nnoise*(cpr[0]*L0+y0 + tvals*(iop + noperator*icorr)))].im += (tmp1.re*tmp2.im + tmp1.im*tmp2.re) * operator_info[iop].weights[ipiece];

                        /** Computation of Connected Part**/

                        /*first we compute in the given space-time point the two pieces appearing in the connected correlator*/
                        tmp1 = spinor_prod_dble(1,0,G1_g5_xi_A+iy,zeta_B+iy);  /*tmp1 = ( GAMMA_1^dag g5 xi_A )^dag zeta_B*/
                        tmp2 = spinor_prod_dble(1,0,G2_g5_xi_B+iy,zeta_A[inoise_A]+iy);  /*tmp1 = ( GAMMA_2^dag g5 xi_B )^dag zeta_A*/

                        /*then sum their product to the connected correlator at y0 (tmp because is only on the local process)*/
                        data.corrConn_tmp[inoise_A +nnoise*(inoise_B + nnoise*(cpr[0]*L0+y0 + tvals*(iop + noperator*icorr)))].re += (tmp1.re*tmp2.re - tmp1.im*tmp2.im) * operator_info[iop].weights[ipiece];
                        data.corrConn_tmp[inoise_A +nnoise*(inoise_B + nnoise*(cpr[0]*L0+y0 + tvals*(iop + noperator*icorr)))].im += (tmp1.re*tmp2.im + tmp1.im*tmp2.re) * operator_info[iop].weights[ipiece];

                        /*the array with the correlator is indexed in the following way:
                        data.corr[inoise_A,inoise_B,t,iop, icorr]
                        so that for each correlator, for each of the 5 odd 4f operators, and at each time, nnoise squared configurations are saved*/

                     } /*end space loop*/

                  } /*end y0 loop*/

               }/*end of sum over pieces in each operator*/

            } /*end of loop over 5 different operators*/

         } /*end of A noise loop*/

      } /*end of B noise loop, sum of correlators computed*/

   } /*end of loop over the correlators*/


   /*each process computes a part of the correlator and stores it inside data.corr_tmp, 
   with the following function the information coming from all the processes gets
   combined (summed, since MPI_SUM) into data.corr, that hence now stores the
   complete results of all the correlators*/

   MPI_Allreduce(data.corrConn_tmp,data.corrConn,nnoise*nnoise*ncorr*tvals*noperator*2,MPI_DOUBLE,MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(data.corrDisc_tmp,data.corrDisc,nnoise*nnoise*ncorr*tvals*noperator*2,MPI_DOUBLE,MPI_SUM, MPI_COMM_WORLD);

   /** memory deallocation**/

   free(zeta_A); /*deallocation of memory used for zeta_A*/
   free(xi_A); /*deallocation of memory used for xi_A*/
   release_wsd(); /*release of the workspace allocated for all the spinors*/

   /*can I add intermediate memory deallocation ??*/

}


/*computes the correlator related to the gauge configuration with index nc
by calling the correlators() function
(function called by the main)*/
static void set_data(int nc)
{

   data.nc=nc; /*sets data.nc to be te index of the gauge configuration passed as input*/
   correlators(); /*computes the correlator with the gauge configuration nc*/

   /*on process 0 prints to the .log file information regarding the correlator*/
   if (my_rank==0)
   {
      printf("G_conn(t) =  %.4e%+.4ei",data.corrConn[0].re,data.corrConn[0].im); /*prints the correlator at the first time,...*/
      printf(",%.4e%+.4ei,...",data.corrConn[1].re,data.corrConn[1].im); /*...at the second time ...*/
      printf(",%.4e%+.4ei",data.corrConn[file_head.tvals-1].re,
                           data.corrConn[file_head.tvals-1].im); /*... and at the last time ...*/
      printf("\n");
      printf("G_disc(t) =  %.4e%+.4ei",data.corrDisc[0].re,data.corrDisc[0].im); /*prints the correlator at the first time,...*/
      printf(",%.4e%+.4ei,...",data.corrDisc[1].re,data.corrDisc[1].im); /*...at the second time ...*/
      printf(",%.4e%+.4ei",data.corrDisc[file_head.tvals-1].re,
                           data.corrDisc[file_head.tvals-1].im); /*... and at the last time ...*/
      printf("\n");
      fflush(flog); /*the output (that is directed on the log file) is flushed*/
   }

}


/*** function transforming the gauge configurations ***/


/*function used to allocate g
(function called by the main)*/
static void allocate_g(void)
{
   g=amalloc(NSPIN*sizeof(su3_dble),4);

   if (BNDRY>0)
      gbuf=amalloc((BNDRY/2)*sizeof(su3_dble),4);

   error((g==NULL)||((BNDRY>0)&&(gbuf==NULL)),1,"allocate_g [odd_df2_4fop.c]",
         "Unable to allocate auxiliary arrays");
}


/*(function called by random_g)*/
static void pack_gbuf(void)
{
   int n,ix,iy,io;

   nfc[0]=FACE0/2;
   nfc[1]=FACE0/2;
   nfc[2]=FACE1/2;
   nfc[3]=FACE1/2;
   nfc[4]=FACE2/2;
   nfc[5]=FACE2/2;
   nfc[6]=FACE3/2;
   nfc[7]=FACE3/2;

   ofs[0]=0;
   ofs[1]=ofs[0]+nfc[0];
   ofs[2]=ofs[1]+nfc[1];
   ofs[3]=ofs[2]+nfc[2];
   ofs[4]=ofs[3]+nfc[3];
   ofs[5]=ofs[4]+nfc[4];
   ofs[6]=ofs[5]+nfc[5];
   ofs[7]=ofs[6]+nfc[6];

   for (n=0;n<8;n++)
   {
      io=ofs[n];

      for (ix=0;ix<nfc[n];ix++)
      {
         iy=map[io+ix];
         gbuf[io+ix]=g[iy];
      }
   }
}


/*(function called by random_g)*/
static void send_gbuf(void)
{
   int n,mu,np,saddr,raddr;
   int nbf,tag;
   su3_dble *sbuf,*rbuf;
   MPI_Status stat;

   for (n=0;n<8;n++)
   {
      nbf=18*nfc[n];

      if (nbf>0)
      {
         tag=mpi_tag();
         mu=n/2;
         np=cpr[mu];

         if (n==(2*mu))
         {
            saddr=npr[n+1];
            raddr=npr[n];
         }
         else
         {
            saddr=npr[n-1];
            raddr=npr[n];
         }

         sbuf=gbuf+ofs[n];
         rbuf=g+ofs[n]+VOLUME;

         if ((np|0x1)!=np)
         {
            MPI_Send((double*)(sbuf),nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD);
            MPI_Recv((double*)(rbuf),nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,
                     &stat);
         }
         else
         {
            MPI_Recv((double*)(rbuf),nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,
                     &stat);
            MPI_Send((double*)(sbuf),nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD);
         }
      }
   }
}


/*function used to create a random su3 transformation g to be applied on the gauge configuration
(function called by the main)*/
static void random_g(void)
{
   su3_dble *gx,*gm;

   gm=g+VOLUME;

   for (gx=g;gx<gm;gx++)
      random_su3_dble(gx);

   if (BNDRY>0)
   {
      pack_gbuf();
      send_gbuf();
   }
}


/*function used to perform a random gauge transformation on the current gauge configuration
(function called by the main)*/
static void transform_ud(void)
{

   /*(function taken by openQCD-1.2 > devel > dirac > check4.c -> transform_ud)*/

   int ix,iy,mu;
   su3_dble *ub,u,v,w;
   su3_dble gx,gxi,gy,gyi;

   ub=udfld();
   
   for (ix=(VOLUME/2);ix<VOLUME;ix++)
   {
      gx=g[ix];

      for (mu=0;mu<4;mu++)
      {
         iy=iup[ix][mu];
         gy=g[iy];
         u=ub[2*mu];
         _su3_dagger(gyi,gy);
         _su3_times_su3(v,u,gyi);
         _su3_times_su3(w,gx,v);
         ub[2*mu]=w;

         iy=idn[ix][mu];
         gy=g[iy];
         u=ub[2*mu+1];
         _su3_dagger(gxi,gx);
         _su3_times_su3(v,u,gxi);
         _su3_times_su3(w,gy,v);
         ub[2*mu+1]=w;
      }

      ub+=8;
   }

   set_flags(UPDATED_UD);
}



/*******************************************************************************/
/***************************** Main of the Program *****************************/
/*******************************************************************************/


int main(int argc,char *argv[])
{
   /** definition of variables **/

   int nc; /*index of the gauge configuration (index used in the loop over gauge configurations)*/
   int iend; /*end flag (1 set, 0 not set) of the configuration loop*/
   int status[4]; /*status of the deflation subspace*/

   int nws; /*number of workspace spinor fields*/
   int nwsd; /*number of workspace spinor fields, in double precision*/
   int nwv; /*number of workspace vector fields*/
   int nwvd; /*number of workspace vector fields, in double precision*/

   double wt1; /*time measured before processing a propagator*/
   double wt2; /*time measured after processing a propagator*/
   double wtavg; /*variable used to store the average time needed for the computation of a propagator*/

   dfl_parms_t dfl; /*structure with the parameters of the deflation subspace*/

   /** openMPI inizialization **/

   MPI_Init(&argc,&argv); /*Inizialization of MPI execution environment*/
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank); /*setting my_rank to the rank of the calling process*/

   /** file and simulation parameters initialization **/

   read_infile(argc,argv); /*read input from command line and input file*/
   alloc_data(); /*allocate memory for the data structure*/
   check_files(); /*check compatibility with .dat and .log files already written*/
   print_info(); /*write all the variables and parameters of the simulation to the .log file*/
   dfl=dfl_parms(); /*get the parameters of the deflation subspace from global structure*/

   init_4foperators(); /*initialization of the gamma structure of the operators VA, AV, SP, PS, TT~*/

   geometry(); /*compute global arrays related to MPI process grid and to indexes of lattice grid*/
   init_rng(); /*initialization of the random number generator*/

   wsize(&nws,&nwsd,&nwv,&nwvd); /*set nws, nwsd, nwv, nwvd according to the solver method chosen in input file*/
   alloc_ws(nws); /*allocates a workspace of nws single-precision spinor fields*/
   alloc_wsd(nwsd); /*allocates a workspace of nwsd double-precision spinor fields*/
   alloc_wv(nwv); /*allocates a workspace of nws single-precision vector fields*/
   alloc_wvd(nwvd); /*allocates a workspace of nwsd double-precision vector fields*/

   /*if the gauge invariance check is required then g is allocated*/
   if (check_gauge_inv==1)
      allocate_g();

   /** loop over the gauge field configurations **/

   iend=0; /*end flag initialized to 0 (= flag not set)*/
   wtavg=0.0; /*average waiting time initialized to 0*/

   for (nc=first;(iend==0)&&(nc<=last);nc+=step) /*loop scanning over the specified configuration range*/
   {
      MPI_Barrier(MPI_COMM_WORLD); /*synchronization between all the MPI processes in the group*/
      wt1=MPI_Wtime(); /*time measured before the nc-th configuration is processed*/

      /*on process 0 the number of the processed configuration is printed on the .log file*/
      if (my_rank==0)
         printf("Configuration no %d\n",nc);
      
      /*gauge configurations are read according to what specified in the input file*/
      if (noexp) /*if -noexp is set the configurations are read in imported file format locally*/
      {
         save_ranlux(); /*save to global variables the current states of the random number generators rlxs and rlxd*/
         sprintf(cnfg_file,"%s/%sn%d_%d",loc_dir,nbase,nc,my_rank); /*get the name of the configuration file from input parameters*/
         /*read_cnfg(cnfg_file);*/ /*reads the configurations from the cnfg_file, saves them on global struct and resets the generators*/
         restore_ranlux(); /*set the states of the generators to the saved values (saved before the reset due to read_cnfg)*/
      }
      else /*if instead -noexp is not set the configurations are read in exported file format*/
      {
         sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,nc); /*get the name of the configuration file from input parameters*/
         /*import_cnfg(cnfg_file);*/ /*reads the configurations from the cnfg_file, saves them on global structure*/
      }

      /*if the random gauge configuration flag is set then a random configuration is used*/
      if(random_conf==1) /*if the flag is set...*/
      {
         random_ud(); /*...the gauge configuration is chesen randomly (previous reading gets overwritten)*/
      }

      /*the deflation subspace is generated*/
      if (dfl.Ns) /*if the number of deflation mode is different from 0...*/
      {
         /*... the deflation subspace is initialized*/

         /*initialize deflation subspace and its compute basis vectors,
         an error is raised if the operation fails*/
         dfl_modes(status);
         error_root(status[0]<0,1,"main [odd_df2_4fop.c]",
                    "Deflation subspace generation failed (status = %d)",
                    status[0]);

         /*on process 0 the (succesful) status of the deflation subspace generation is written to the .log file*/
         if (my_rank==0)
            printf("Deflation subspace generation: status = %d\n",status[0]);
      }

      /*the actual computation of the correlators is done here*/ 
      set_data(nc); /*the correlator corresponding to the gauge configuration nc is computed and stored inside the structure data*/

      /*the computed correlators then gets stored*/
      write_data(); /*writes on the .dat files the values of the correlators*/

      /*some more info are written in the rng file*/
      export_ranlux(nc,rng_file); /*the tag (nc) and the state of the random generator is written on the rng_file*/

      /*erros are checked*/
      error_chk(); /*checks the status of the data and aborts if an error is detected*/

      /*then some estimate of the time required for the computation is done*/
      
      MPI_Barrier(MPI_COMM_WORLD); /*synchronization between all the MPI processes in the group*/
      wt2=MPI_Wtime();  /*time measured after the nc-th configuration is processed*/
      wtavg+=(wt2-wt1); /*the time needed to processes the nc-th configuration is added to the total time needed*/

      /*on process 0 the time needed to process the nc-th configuration and the exstimate of the average time
      needed to process one configuration are printed (to stdout that now is the .log file)*/
      if (my_rank==0)
      {
         printf("Configuration no %d fully processed in %.2e sec ",
                nc,wt2-wt1);
         printf("(average = %.2e sec)\n\n",
                wtavg/(double)((nc-first)/step+1));
      }


      /*then if the check_invariance flag has been set the computation is repeated with a gauge transformed configuraiton*/

      if (check_gauge_inv==1) /*if the flag is set...*/
      {
         if (my_rank==0)
            printf("Gauge transforming configuration no %d and repeating the computation\n",nc);

         MPI_Barrier(MPI_COMM_WORLD); /*synchronization between all the MPI processes in the group*/
         wt1=MPI_Wtime(); /*time measured before the nc-th configuration is processed*/

         random_g(); /*... we choose randomly the transformation to be applied on the gauge configuration*/
         transform_ud(); /*...the gauge configuration gets gauge transformed (according to g, chosen randomly)*/

         /*the deflation subspace is generated*/
         if (dfl.Ns) /*if the number of deflation mode is different from 0...*/
         {
            /*... the deflation subspace is initialized*/

            /*initialize deflation subspace and its compute basis vectors,
            an error is raised if the operation fails*/
            dfl_modes(status);
            error_root(status[0]<0,1,"main [odd_df2_4fop.c]",
                       "Deflation subspace generation failed (status = %d)",
                       status[0]);

            /*on process 0 the (succesful) status of the deflation subspace generation is written to the .log file*/
            if (my_rank==0)
               printf("Deflation subspace generation: status = %d\n",status[0]);
         }

         
         set_data(nc); /*... the computation is repeated*/
         write_data(); /*... the data is saved again*/

         MPI_Barrier(MPI_COMM_WORLD); /*synchronization between all the MPI processes in the group*/
         wt2=MPI_Wtime();  /*time measured after the nc-th configuration is processed*/

         if (my_rank==0)
            printf("Configuration no %d (gauge transformed) fully processed in %.2e sec \n", nc,wt2-wt1);
      }
      

      /*once the correlator corresponding to the current gauge configuration has been computed
      the program checks if its execution has to be terminated early by looking for the endflag
      (the user can kill the program gently by creating a .end file with the same name of the run in the
      log directory, if such a file is found the function below sets the endflag to true)*/

      check_endflag(&iend);  /*check if the endlfag has been raised by the user (if so this is the last loop iteration)*/

      /*on process 0 before a new loop iteration the output gets flushed and the file gets saved*/
      if (my_rank==0)
      {
         fflush(flog); /*flush of printf to the .log file*/
         copy_file(log_file,log_save); /*.log file saved to .log~ file for backup*/
         copy_file(dat_file,dat_save); /*.dat file saved to .dat~ file for backup*/
         copy_file(rng_file,rng_save); /*.rng file saved to .rng~ file for backup*/
      }

   } /*end of loop over the configurations */

   /** program ending **/

   /*on process 0 the .log file is closed (it was still open since it served as stdout)*/
   if (my_rank==0)
   {
      fclose(flog);
   }

   MPI_Finalize(); /*MPI execution environmment is terminated*/
   
   exit(0); /*termination of main*/

} /*end of Main*/