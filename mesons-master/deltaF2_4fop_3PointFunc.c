/*******************************************************************************
*
* File deltaF2_4fop_3PointFunc.c
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
* Syntax: deltaF2_4fop_3PointFunc -i <input file> [-noexp] [-a]
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



/*******************************************************************************/
/******************* Declaration of Global Variables ***************************/
/*******************************************************************************/


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

static char nbase[NAME_SIZE]; /*name of the run specified in the .in file (used to set the name of the cnfg file)*/
static char outbase[NAME_SIZE]; /*name of the output file specified in the .in file*/

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
#define len_filename 30 /* = lenght of the string given by "/" + ".deltaF2_4fop_3PointFunc.log"*/
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



/*******************************************************************************/
/************************** Definition of Functions ****************************/
/*******************************************************************************/


/*** input reading functions ***/

/*function reading directories' names and other inputs from input file (function called by read_infile)*/
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
                 "read_dirs [deltaF2_4fop_3PointFunc.c]","Improper configuration range");
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


/*function for files inizialization according to input file specifications (function called by read_infile)*/
static void setup_files(void)
{
   /*lenght check of the string loc_dir or cnfg_dir*/
   if (noexp)
      error_root(name_size("%s/%sn%d_%d",loc_dir,nbase,last,NPROC-1)>=NAME_SIZE,
                 1,"setup_files [deltaF2_4fop_3PointFunc.c]","loc_dir name is too long");
   else
      error_root(name_size("%s/%sn%d",cnfg_dir,nbase,last)>=NAME_SIZE,
                 1,"setup_files [deltaF2_4fop_3PointFunc.c]","cnfg_dir name is too long");

   /*check on accessibility (only on process 0) and name lenght of dat_dir*/
   check_dir_root(dat_dir);
   error_root(name_size("%s/%s.deltaF2_4fop_3PointFunc.dat~",dat_dir,outbase)>=NAME_SIZE,
              1,"setup_files [deltaF2_4fop_3PointFunc.c]","dat_dir name is too long");

   /*check on accessibility (only on process 0) and name lenght of log_dir*/
   check_dir_root(log_dir);
   error_root(name_size("%s/%s.deltaF2_4fop_3PointFunc.log~",log_dir,outbase)>=NAME_SIZE,
              1,"setup_files [deltaF2_4fop_3PointFunc.c]","log_dir name is too long");

   /*assignment of files' names based on input file specifications*/

   sprintf(log_file,"%s/%s.deltaF2_4fop_3PointFunc.log",log_dir,outbase);
   sprintf(end_file,"%s/%s.deltaF2_4fop_3PointFunc.end",log_dir,outbase);
   sprintf(par_file,"%s/%s.deltaF2_4fop_3PointFunc.par",dat_dir,outbase);
   sprintf(dat_file,"%s/%s.deltaF2_4fop_3PointFunc.dat",dat_dir,outbase);
   sprintf(rng_file,"%s/%s.deltaF2_4fop_3PointFunc.rng",dat_dir,outbase);
   sprintf(log_save,"%s~",log_file);
   sprintf(par_save,"%s~",par_file);
   sprintf(dat_save,"%s~",dat_file);
   sprintf(rng_save,"%s~",rng_file);

} /*end of setup_files function*/


/*function to read lattice parameters from input file and assignin them to global variables (function called by read_infile)*/
static void read_lat_parms(void)
{

   /*declaration of temporary variables used for reading parameters*/

   double csw,cF; /*coefficient of sw term (csw) and of Fermion O(a) boundary counterterm (cF)*/
   char tmpstring[NAME_SIZE]; /*temporary string used for reading*/
   char tmpstring2[NAME_SIZE]; /*temporary string used for reading*/
   int iprop,icorr,eoflg; /*index running on propagators (iprop), correlators (icorr), twisted mass flag (eoflg)*/
   int i_4fop; /*ET: index running of 4fop corr*/

   /*on process 0 reads parameters from file*/

   if (my_rank==0)
   {

      /*reading of the [Measurements] section from input file*/

      find_section("Measurements"); /*reading pointer set the line after the string "[Measurements]"*/
      read_line("nprop","%d",&nprop); /*nprop is set to the number of different quark lines written in the input file*/
      read_line("ncorr","%d",&ncorr); /*ncorr is set to the number of different correlators written in the input file*/
      read_line("n4fop","%d",&n4fop); /*ET: n4fop is set to the number of different corr with 4fop written in the input file*/
      read_line("nnoise","%d",&nnoise); /*nnoise is set to the number of noise vector for each configuration*/
      read_line("noise_type","%s",tmpstring); /*noise_type set to U1, Z2 or GAUSS according to input file*/
      read_line("csw","%lf",&csw); /*csw coefficient read from input file*/
      read_line("cF","%lf",&cF); /*cF coefficient read from input file*/
/* DP */      
      read_line("eoflg","%d",&eoflg); /*eoflg read from input file*/
/* DP */

      /*check on the validity of the parameters read from input file*/

      /*nprop, ncorr and nnoise must be positive integers*/
      error_root(nprop<1,1,"read_lat_parms [mesons.c]",
                 "Specified nprop must be larger than zero");
      error_root(ncorr<1,1,"read_lat_parms [mesons.c]",
                 "Specified ncorr must be larger than zero");
      error_root(nnoise<1,1,"read_lat_parms [mesons.c]",
                 "Specified nnoise must be larger than zero");
      /*ET: also n4fop must be*/
      error_root(n4fop<1,1,"read_lat_parms [mesons.c]",
                 "Specified n4fop must be larger than zero");

/* DP */
      /*eoflg must be either 0 or 1*/
      error_root(eoflg<0,1,"read_lat_parms [mesons.c]",
		 "Specified eoflg must be 0,1");
      error_root(eoflg>1,1,"read_lat_parms [mesons.c]",
		 "Specified eoflg must be 0,1");
/* DP */

       /*noise_type must be either U1, Z2 or GAUSS*/
      noisetype=-1;
      if(strcmp(tmpstring,"Z2")==0)
         noisetype=Z2_NOISE;
      if(strcmp(tmpstring,"GAUSS")==0)
         noisetype=GAUSS_NOISE;
      if(strcmp(tmpstring,"U1")==0)
         noisetype=U1_NOISE;
      error_root(noisetype==-1,1,"read_lat_parms [mesons.c]",
                 "Unknown noise type");
   }

   /*broadcast of parameters read on process 0 to
   all other proceses of the communicator group*/

   MPI_Bcast(&nprop,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&ncorr,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&n4fop,1,MPI_INT,0,MPI_COMM_WORLD); /*ET: broadcast of n4fop*/
   MPI_Bcast(&nnoise,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&noisetype,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
/* DP */
   MPI_Bcast(&eoflg,1,MPI_INT,0,MPI_COMM_WORLD);
/* DP */

   /*memory allocation for all the parameters needed
   by the various propagators and correlator*/

   kappas=malloc(nprop*sizeof(double)); /*one kappa for each propagator*/
   mus=malloc(nprop*sizeof(double)); /*one mus for each propagator*/
   isps=malloc(nprop*sizeof(int)); /*one solver id for each propagator*/
   props1=malloc(ncorr*sizeof(int)); /*type of first quark, one for each correlator*/
   props2=malloc(ncorr*sizeof(int)); /*type of second quark, one for each correlator*/
   type1=malloc(ncorr*sizeof(int)); /*Dirac structure of first meson, one for each correlator*/
   type2=malloc(ncorr*sizeof(int)); /*Dirac structure of second meson, one for each correlator*/
   x0s=malloc(ncorr*sizeof(int)); /*time slice of the source, one for each correlator*/
   file_head.kappa1=malloc(ncorr*sizeof(double)); /*kappa of the first kind of quark, one for each correlator*/
   file_head.kappa2=malloc(ncorr*sizeof(double)); /*kappa of the second kind of quark, one for each correlator*/
/* DP */   
   file_head.mus1=malloc(ncorr*sizeof(double)); /*twisted mass of the first kind of quark, one for each correlator*/
   file_head.mus2=malloc(ncorr*sizeof(double)); /*twisted mass of the second kind of quark, one for each correlator*/
/* DP */

   file_head.type1=type1; /*Dirac structure of first meson, one for each correlator*/
   file_head.type2=type2; /*Dirac structure of second meson, one for each correlator*/
   file_head.x0=x0s; /*time slice of the source, one for each correlator*/
   file_head.isreal=malloc(ncorr*sizeof(int)); /*array with either 1 (pion-pion case) or 0 for each correlator (??)*/

   /*check of successful memory allocation*/
   error((kappas==NULL)||(mus==NULL)||(isps==NULL)||(props1==NULL)||
         (props2==NULL)||(type1==NULL)||(type2==NULL)||(x0s==NULL)||
         (file_head.kappa1==NULL)||(file_head.kappa2==NULL)||
         (file_head.mus1==NULL)||(file_head.mus2==NULL)||
         (file_head.isreal==NULL),
         1,"read_lat_parms [mesons.c]","Out of memory");

   /*ET: allocation of the two array needed for a 4fop corr*/
   file_head.corr1=malloc(n4fop*sizeof(int)); /*indices of the correlator1 in the 4fop correlation functions*/
   file_head.corr2=malloc(n4fop*sizeof(int)); /*indices of the correlator1 in the 4fop correlation functions*/
   error((file_head.corr1==NULL)||(file_head.corr2==NULL),
          1,"read_lat_parms [mesons.c]","Out of memory");
   /*ET: also allocation of array needed for 4fop*/
   corrs1=malloc(n4fop*sizeof(int));
   corrs2=malloc(n4fop*sizeof(int));
   error((corrs1==NULL)||(corrs2==NULL),
          1,"read_lat_parms [mesons.c]","Out of memory");

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
         read_line("iprop","%d %d",&props1[icorr],&props2[icorr]);
         error_root((props1[icorr]<0)||(props1[icorr]>=nprop),1,"read_lat_parms [mesons.c]",
                 "Propagator index out of range");
         error_root((props2[icorr]<0)||(props2[icorr]>=nprop),1,"read_lat_parms [mesons.c]",
                 "Propagator index out of range");

         /*the two temporary strings are used to read from the input file the type
         of the two mesons appearing in the given correlator, then the string read
         from file is converted to an integer identifier*/

         read_line("type","%s %s",tmpstring,tmpstring2);
         type1[icorr]=-1;
         type2[icorr]=-1;
         
         /*conversion of type1 to an integer identifier*/
         if(strncmp(tmpstring,"1",1)==0)
            type1[icorr]=ONE_TYPE;
         else if(strncmp(tmpstring,"G0G1",4)==0)
            type1[icorr]=GAMMA0GAMMA1_TYPE;
         else if(strncmp(tmpstring,"G0G2",4)==0)
            type1[icorr]=GAMMA0GAMMA2_TYPE;
         else if(strncmp(tmpstring,"G0G3",4)==0)
            type1[icorr]=GAMMA0GAMMA3_TYPE;
         else if(strncmp(tmpstring,"G0G5",4)==0)
            type1[icorr]=GAMMA0GAMMA5_TYPE;
         else if(strncmp(tmpstring,"G1G2",4)==0)
            type1[icorr]=GAMMA1GAMMA2_TYPE;
         else if(strncmp(tmpstring,"G1G3",4)==0)
            type1[icorr]=GAMMA1GAMMA3_TYPE;
         else if(strncmp(tmpstring,"G1G5",4)==0)
            type1[icorr]=GAMMA1GAMMA5_TYPE;
         else if(strncmp(tmpstring,"G2G3",4)==0)
            type1[icorr]=GAMMA2GAMMA3_TYPE;
         else if(strncmp(tmpstring,"G2G5",4)==0)
            type1[icorr]=GAMMA2GAMMA5_TYPE;
         else if(strncmp(tmpstring,"G3G5",4)==0)
            type1[icorr]=GAMMA3GAMMA5_TYPE;
         else if(strncmp(tmpstring,"G0",2)==0)
            type1[icorr]=GAMMA0_TYPE;
         else if(strncmp(tmpstring,"G1",2)==0)
            type1[icorr]=GAMMA1_TYPE;
         else if(strncmp(tmpstring,"G2",2)==0)
            type1[icorr]=GAMMA2_TYPE;
         else if(strncmp(tmpstring,"G3",2)==0)
            type1[icorr]=GAMMA3_TYPE;
         else if(strncmp(tmpstring,"G5",2)==0)
            type1[icorr]=GAMMA5_TYPE;
         
         /*conversion of type2 to an integer identifier*/
         if(strncmp(tmpstring2,"1",1)==0)
            type2[icorr]=ONE_TYPE;
         else if(strncmp(tmpstring2,"G0G1",4)==0)
            type2[icorr]=GAMMA0GAMMA1_TYPE;
         else if(strncmp(tmpstring2,"G0G2",4)==0)
            type2[icorr]=GAMMA0GAMMA2_TYPE;
         else if(strncmp(tmpstring2,"G0G3",4)==0)
            type2[icorr]=GAMMA0GAMMA3_TYPE;
         else if(strncmp(tmpstring2,"G0G5",4)==0)
            type2[icorr]=GAMMA0GAMMA5_TYPE;
         else if(strncmp(tmpstring2,"G1G2",4)==0)
            type2[icorr]=GAMMA1GAMMA2_TYPE;
         else if(strncmp(tmpstring2,"G1G3",4)==0)
            type2[icorr]=GAMMA1GAMMA3_TYPE;
         else if(strncmp(tmpstring2,"G1G5",4)==0)
            type2[icorr]=GAMMA1GAMMA5_TYPE;
         else if(strncmp(tmpstring2,"G2G3",4)==0)
            type2[icorr]=GAMMA2GAMMA3_TYPE;
         else if(strncmp(tmpstring2,"G2G5",4)==0)
            type2[icorr]=GAMMA2GAMMA5_TYPE;
         else if(strncmp(tmpstring2,"G3G5",4)==0)
            type2[icorr]=GAMMA3GAMMA5_TYPE;
         else if(strncmp(tmpstring2,"G0",2)==0)
            type2[icorr]=GAMMA0_TYPE;
         else if(strncmp(tmpstring2,"G1",2)==0)
            type2[icorr]=GAMMA1_TYPE;
         else if(strncmp(tmpstring2,"G2",2)==0)
            type2[icorr]=GAMMA2_TYPE;
         else if(strncmp(tmpstring2,"G3",2)==0)
            type2[icorr]=GAMMA3_TYPE;
         else if(strncmp(tmpstring2,"G5",2)==0)
            type2[icorr]=GAMMA5_TYPE;

         /*validity check of type1 and type2 read from the input file*/
         error_root((type1[icorr]==-1)||(type2[icorr]==-1),1,"read_lat_parms [mesons.c]",
                 "Unknown or unsupported Dirac structure");

         /*source time slice is read for each correlator and its validity checked
         (the timeslice must be inside the previously specified time boundaries)*/
         read_line("x0","%d",&x0s[icorr]);
         error_root((x0s[icorr]<=0)||(x0s[icorr]>=(NPROC0*L0-1)),1,"read_lat_parms [mesons.c]",
                 "Specified time x0 is out of range");
      }

      /*ET: loop over the 4fop corr */
      for(i_4fop=0; i_4fop<n4fop; i_4fop++)
      {
         sprintf(tmpstring,"Corr4fop %i",i_4fop); /*temporary string set to the 4fop correlator identifier*/
         find_section(tmpstring); /*reading pointer set in the section of the icorr-th correlator*/

         /*the types of the first and the second quarks are read from the input file and
         the validity of the input parameters is checked (they must range from 0 to to nprop-1)*/
         read_line("icorr","%d %d",&corrs1[i_4fop],&corrs2[i_4fop]);
         error_root((corrs1[i_4fop]<0)||(corrs1[i_4fop]>=ncorr),1,"read_lat_parms [mesons.c]",
                 "4fop index out of range");
         error_root((corrs2[i_4fop]<0)||(corrs2[i_4fop]>=ncorr),1,"read_lat_parms [mesons.c]",
                 "4fop index out of range");
      }

   }

   /*broadcast of parameters read on process 0 to
   all other proceses of the communicator group*/

   MPI_Bcast(kappas,nprop,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(mus,nprop,MPI_DOUBLE,0,MPI_COMM_WORLD);
   /*MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);*/ /*commented becouse broadcasted twice (to be removed)*/
   /*MPI_Bcast(&cF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);*/ /*commented becouse broadcasted twice (to be removed)*/
/* DP */
   /*MPI_Bcast(&eoflg,1,MPI_INT,0,MPI_COMM_WORLD);*/ /*commented becouse broadcasted twice (to be removed)*/
/* DP */
   MPI_Bcast(isps,nprop,MPI_INT,0,MPI_COMM_WORLD);

   MPI_Bcast(props1,ncorr,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(props2,ncorr,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(type1,ncorr,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(type2,ncorr,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(x0s,ncorr,MPI_INT,0,MPI_COMM_WORLD);

   /*ET: bcast of corrs1 and corrs2*/
   MPI_Bcast(corrs1,n4fop,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(corrs2,n4fop,MPI_INT,0,MPI_COMM_WORLD);

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
/* DP */
   set_tm_parms(eoflg); /*eoflg in the global structure tm is set to the read value*/
/* DP */

   file_head.ncorr = ncorr; /*number of correlators saved to global structure*/
   file_head.n4fop = n4fop; /*ET: number of 4fop correlators saved to global structure*/
   file_head.nnoise = nnoise; /*number of noise vectors saved to global structure*/
   file_head.tvals = NPROC0*L0; /*tvals saved to global structure*/
   tvals = NPROC0*L0; /*tvals saved to global variable*/
   file_head.noisetype = noisetype; /*noisetype saved to global structure*/
   for(icorr=0; icorr<ncorr; icorr++) /*for each correlator the related parameters are saved in file_head*/
   {
      file_head.kappa1[icorr]=kappas[props1[icorr]]; /*kappa of first quark saved to global structure*/
      file_head.kappa2[icorr]=kappas[props2[icorr]]; /*kappa of second quark saved to global structure*/

/* DP */
      file_head.mus1[icorr]=mus[props1[icorr]]; /*twisted mass of first quark saved to global structure*/
      file_head.mus2[icorr]=mus[props2[icorr]]; /*twisted mass of second quark saved to global structure*/
/* DP */

      /*in the pion-pion case isreal is set to 1, in any other case to 0*/
      if ((type1[icorr]==GAMMA5_TYPE)&&(type2[icorr]==GAMMA5_TYPE)&&
          (props1[icorr]==props2[icorr]))
         file_head.isreal[icorr]=1;
      else
         file_head.isreal[icorr]=0;
   }

   /*ET: global corrs1 and corrs2 copied inside the global file_head struct*/
   for(i_4fop=0; i_4fop<n4fop; i_4fop++)
   {
      file_head.corr1[i_4fop] = corrs1[i_4fop];
      file_head.corr2[i_4fop] = corrs2[i_4fop];
   }

   /*the parameters in the lat structure get saved in the fdat file,
   if the option -a is given the consistency with the previously
   written parameters is checked*/

   if (append) /*if the current simulation is the continuation of a previous run ...*/
      check_lat_parms(fdat); /*... the match between the previous and the current lattice parameters is checked*/
   else /*if the current simulation is a new run ...*/
      write_lat_parms(fdat); /*...the lat structure gets written to the fdat file*/
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

} /*end of Main*/