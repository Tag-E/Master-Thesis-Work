/*******************************************************************************
*
* File mesons.c
*
* Copyright (C) 2013, 2014 Tomasz Korzec
*
* Based on openQCD, ms1 and ms4
* Copyright (C) 2012 Martin Luescher and Stefan Schaefer and David Preti
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
* Syntax: mesons -i <input file> [-noexp] [-a]
*
* For usage instructions see the file README.mesons
*
*******************************************************************************/

#define MAIN_PROGRAM

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

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)


/**declaration of global variables**/

static char line[NAME_SIZE+1];

/*this macro sets n to be the greatest between n and m*/
#define MAX(n,m) \
   if ((n)<(m)) \
      (n)=(m)

/*struct containing the parameters of the input file related to the correlators*/
static struct
{
   int ncorr; /*number of correlators*/
   int nnoise; /*number of noise vector for each configuration*/
   int tvals; /*= size of time lattice spaces times number of time processes (??)*/
   int noisetype; /*type of noise vectors, either U1, Z2 or GAUSS*/
   double *kappa1; /*kappa of the first kind of quark, one for each correlator*/
   double *kappa2; /*kappa of the second kind of quark, one for each correlator*/
   int *type1; /*array containing the Dirac structure of the first meson in each correlator*/
   int *type2; /*array containing the Dirac structure of the second meson in each correlator*/
   int *x0; /*array containing the source time slice of each correlator*/
   int *isreal; /*array containing 1 for pion-pion correlators, 0 otherwise (??)*/
} file_head; 

/*structure containing the correlators: it has an array corr (of complex doubles) with the
complete values of all the correlators at all times, a related array corr_tmp used as temporary
copy to store the partial values of the correlators computed by a single process, an index nc
laballing the gauge configuration used to compute the correlator*/
static struct
{
   complex_dble *corr; /*complete value of the correlators*/
   complex_dble *corr_tmp; /*partial value of the correlators computed by a single process*/
   int nc; /*index of the gauge configuration the correlator is related to*/
} data;

/*structure containing the list with all the propagators and their information*/
static struct
{
   int nux0;     /* number of unique x0 values */
   int *ux0;     /* unique x0 values */
   int *nprop;   /* number of propagators at each x0 */
   int nmax;     /* max(nprop) */
   int **prop;   /* propagator index of each x0 and propagator */
   int **type;   /* type index of each x0 and propagator  */
} proplist;


/*
   - my_rank : rank of the process (unique identifier of the process inside the communicator group)
   - noexp : True if the option -noexp is set by command line, False(0) otherwise 
   - append :  True if the option -a is set by command line, False(0) otherwise
   - norng : True if the option -norng is set by command line, False(0) otherwise
   - endian : BIG_ENDIAN, LITTLE_ENDIAN or UNKOWN_ENDIAN depending on the machine
*/
static int my_rank,noexp,append,norng,endian;
/*
   - first : index of the first configuration
   - last : inde of the last configuration
   - step : step used in the scanning of configurations (??)
*/
static int first,last,step;
/*
   - level, seed : parameters of the random generator
   - nprop, ncorr : number of different quark lines and of different correlators
   - nnoise : number of noise vector for each configuration
   - noisetype : either U1, Z2 or GAUSS (expand to something like 1,2,3)
   - tvals : size of time lattice spaces times number of time processes (??)
*/
static int level,seed,nprop,ncorr,nnoise,noisetype,tvals;
/*
   - isps : array containing the solver id for each propagator
   - props1 : array containing the type of the first quark appearing in each correlator
   - props2 : array containing the type of the second quark appearing in each correlator
   - type1 : array containing the Dirac structure of the first meson in each correlator
   - type2 : array containing the Dirac structure of the second meson in each correlator
   - x0s : array containing the time slice of the source of each correlator
*/
static int *isps,*props1,*props2,*type1,*type2,*x0s;
/*
   - ipgrd : variable used to keep track of changes in the number of processes between runs,
             if ipgrd[0]!=0 then the process grid changed, if ipgrd[1]!=0 then the process
             block size changed
   - rlxs_state : state of the random number generator rlxs
   - rlxd_state : state of the random number generator rlxd
*/
static int ipgrd[2],*rlxs_state=NULL,*rlxd_state=NULL;
/*
   - kappas : array containing the value of kappa for each propagator
   - mus : array containing the value of mus for each propagator (??)
*/
static double *kappas,*mus;

/*names of directories and files used:
   - _dir are the names of the directories used
   - _file and the names of the files produced
   - _save the names of the related backup files
*/

static char log_dir[NAME_SIZE],loc_dir[NAME_SIZE];
static char cnfg_dir[NAME_SIZE],dat_dir[NAME_SIZE];
/*
   - log_file : name of the .log file used as stdout
   - log_save : name of the backup file of the .log file
   - end_file : name of the file (same as run name) with .extension used to signal early termination
*/
static char log_file[NAME_SIZE],log_save[NAME_SIZE],end_file[NAME_SIZE];
static char dat_file[NAME_SIZE],dat_save[NAME_SIZE];
static char par_file[NAME_SIZE],par_save[NAME_SIZE];
static char rng_file[NAME_SIZE],rng_save[NAME_SIZE];
/*
   - cnfg_file : name given to various files where the configurations are stored
   - nbase : name given to the run
   - outbase : name given to output files (=nbase by default)
*/
static char cnfg_file[NAME_SIZE],nbase[NAME_SIZE],outbase[NAME_SIZE];

/*
   - fin : input file where the specifics of the simulation are written
   - flog : log file used as stdout where execution errors get written
   - fdat : binary parameters file (containing the lat structure)
*/
static FILE *fin=NULL,*flog=NULL,*fend=NULL,*fdat=NULL;

/*************************** Definition of Functions ***********************************/


/*function used to allocate the structure data*/
static void alloc_data(void)
{

   /*the number of complex double needed is equal to the number of correlators
   times the number of noise vectors per configuration times the number of
   time intervals (such a quantity for data.corr and for the temporary counterpart
   data.corr_tmp)*/

   /*memory allocation*/
   data.corr=malloc(file_head.ncorr*file_head.nnoise*file_head.tvals*
                                                          sizeof(complex_dble));
   data.corr_tmp=malloc(file_head.ncorr*file_head.nnoise*file_head.tvals*
                                                      sizeof(complex_dble));
   
   /*check on correct memory allocation*/
   error((data.corr==NULL)||(data.corr_tmp==NULL),1,"alloc_data [mesons.c]",
         "Unable to allocate data arrays");
}


/*function used to save the global file_head structure containing
the correlators' information on the binary .dat file*/
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

   error_root(iw!=4,1,"write_file_head [mesons.c]",
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

      istd[0]=(stdint_t)(file_head.type1[i]);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

      istd[0]=(stdint_t)(file_head.type2[i]);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

      istd[0]=(stdint_t)(file_head.x0[i]);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

      istd[0]=(stdint_t)(file_head.isreal[i]);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

      error_root(iw!=6,1,"write_file_head [mesons.c]",
              "Incorrect write count");
   }
}

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

   error_root(ir!=4,1,"check_file_head [mesons.c]",
              "Incorrect read count");
   error_root(ie!=0,1,"check_file_head [mesons.c]",
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

      ir+=fread(istd,sizeof(stdint_t),1,fdat);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      ie+=(istd[0]!=(stdint_t)(file_head.type1[i]));

      ir+=fread(istd,sizeof(stdint_t),1,fdat);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      ie+=(istd[0]!=(stdint_t)(file_head.type2[i]));

      ir+=fread(istd,sizeof(stdint_t),1,fdat);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      ie+=(istd[0]!=(stdint_t)(file_head.x0[i]));

      ir+=fread(istd,sizeof(stdint_t),1,fdat);
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);
      ie+=(istd[0]!=(stdint_t)(file_head.isreal[i]));

      error_root(ir!=6,1,"check_file_head [mesons.c]",
              "Incorrect read count");
      error_root(ie!=0,1,"check_file_head [mesons.c]",
              "Unexpected value of kappa, type, x0 or isreal");
   }
}

static void write_data(void)
{
   int iw;
   int nw;
   int chunk;
   int icorr,i;

   if (my_rank==0)
   {
      fdat=fopen(dat_file,"ab");
      error_root(fdat==NULL,1,"write_data [mesons.c]",
                 "Unable to open dat file");

      nw = 1;
      if(endian==BIG_ENDIAN)
      {
         bswap_double(file_head.nnoise*file_head.tvals*file_head.ncorr*2,
                      data.corr);
         bswap_int(1,&(data.nc));
      }
      iw=fwrite(&(data.nc),sizeof(int),1,fdat);
      for (icorr=0;icorr<file_head.ncorr;icorr++)
      {
         chunk=file_head.nnoise*file_head.tvals*(2-file_head.isreal[icorr]);
         nw+=chunk;
         if (file_head.isreal[icorr])
         {
            for (i=0;i<chunk;i++)
               iw+=fwrite(&(data.corr[icorr*file_head.tvals*file_head.nnoise+i]),
                       sizeof(double),1,fdat);
         }else
         {
            iw+=fwrite(&(data.corr[icorr*file_head.tvals*file_head.nnoise]),
                       sizeof(double),chunk,fdat);
         }
      }
      if(endian==BIG_ENDIAN)
      {
         bswap_double(file_head.nnoise*file_head.tvals*file_head.ncorr*2,
                      data.corr);
         bswap_int(1,&(data.nc));
      }
      error_root(iw!=nw,1,"write_data [mesons.c]",
                 "Incorrect write count");
      fclose(fdat);
   }
}

static int read_data(void)
{
   int ir;
   int nr;
   int chunk;
   int icorr,i;
   double zero;

   zero=0;
   if(endian==BIG_ENDIAN)
      bswap_double(1,&zero);
   nr=1;
   ir=fread(&(data.nc),sizeof(int),1,fdat);

   for (icorr=0;icorr<file_head.ncorr;icorr++)
   {
      chunk=file_head.nnoise*file_head.tvals*(2-file_head.isreal[icorr]);
      nr+=chunk;
      if (file_head.isreal[icorr])
      {
         for (i=0;i<chunk;i++)
         {
            ir+=fread(&(data.corr[icorr*file_head.tvals*file_head.nnoise+i]),
                    sizeof(double),1,fdat);
            data.corr[icorr*file_head.tvals*file_head.nnoise+i].im=zero;
         }
      }else
      {
         ir+=fread(&(data.corr[icorr*file_head.tvals*file_head.nnoise]),
                    sizeof(double),chunk,fdat);
      }
   }

   if (ir==0)
      return 0;

   error_root(ir!=nr,1,"read_data [mesons.c]",
                 "Read error or incomplete data record");
   if(endian==BIG_ENDIAN)
   {
      bswap_double(nr,data.corr);
      bswap_int(1,&(data.nc));
   }
   return 1;
}


/*function reading directories' names and other inputs from input file*/
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

      find_section("Run name"); /*pointer reading from input file is set after the line after string "[Run name]"*/
      read_line("name","%s",nbase); /*nbase is set to the name of the run*/
      read_line_opt("output",nbase,"%s",outbase); /*outbase is set to be the name used for output files*/

      /*reading of "Directories" section:
        - log and dat dir are always read
        - if noexp is set loc dir is read while cnfg is not
        - if noexp is not set the opposite is true
      */

      find_section("Directories"); /*pointer reading from input file is set after the string "[Directories]"*/
      read_line("log_dir","%s",log_dir); /*log_dir is set to the string written after "log_dir"*/

      if (noexp) /*if configurations are in the imported file format...*/
      {
         read_line("loc_dir","%s",loc_dir); /*loc_dir is set to the string written after "loc_dir"*/
         cnfg_dir[0]='\0'; /*cnfg_dir is set to '\0' (is not read)*/
      }
      else /*if configuration are in the usual exported file format...*/
      {
         read_line("cnfg_dir","%s",cnfg_dir); /*cnfg_dir is set to the string written after "cnfg_dir"*/
         loc_dir[0]='\0'; /*loc_dir is set to '\0' (is not read)*/
      }

      read_line("dat_dir","%s",dat_dir); /*dat_dir is set to the string written after "dat_dir"*/

      /*reading of "Configurations" section:
        - first is set to the index of the firt configuration
        - last is set to the index of the last configuration
        - step is set to the step at which configurations are
          scanned (?)
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
      read_line("seed","%d",&seed); /*seed is set to the specified integer*/

      /*an error is raised if first, last and step are not valid:
        last-first should be non negative and an integer multiple of step*/
      error_root((last<first)||(step<1)||(((last-first)%step)!=0),1,
                 "read_dirs [mesons.c]","Improper configuration range");
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
}


/*function for files inizialization according to input file specifications*/
static void setup_files(void)
{
   /*lenght check of the string loc_dir or cnfg_dir*/
   if (noexp)
      error_root(name_size("%s/%sn%d_%d",loc_dir,nbase,last,NPROC-1)>=NAME_SIZE,
                 1,"setup_files [mesons.c]","loc_dir name is too long");
   else
      error_root(name_size("%s/%sn%d",cnfg_dir,nbase,last)>=NAME_SIZE,
                 1,"setup_files [mesons.c]","cnfg_dir name is too long");

   /*check on accessibility (only on process 0) and name lenght of dat_dir*/
   check_dir_root(dat_dir);
   error_root(name_size("%s/%s.mesons.dat~",dat_dir,outbase)>=NAME_SIZE,
              1,"setup_files [mesons.c]","dat_dir name is too long");

   /*check on accessibility (only on process 0) and name lenght of log_dir*/
   check_dir_root(log_dir);
   error_root(name_size("%s/%s.mesons.log~",log_dir,outbase)>=NAME_SIZE,
              1,"setup_files [mesons.c]","log_dir name is too long");

   /*assignment of files' names based on input file specifications*/

   sprintf(log_file,"%s/%s.mesons.log",log_dir,outbase);
   sprintf(end_file,"%s/%s.mesons.end",log_dir,outbase);
   sprintf(par_file,"%s/%s.mesons.par",dat_dir,outbase);
   sprintf(dat_file,"%s/%s.mesons.dat",dat_dir,outbase);
   sprintf(rng_file,"%s/%s.mesons.rng",dat_dir,outbase);
   sprintf(log_save,"%s~",log_file);
   sprintf(par_save,"%s~",par_file);
   sprintf(dat_save,"%s~",dat_file);
   sprintf(rng_save,"%s~",rng_file);
}


/*function to read lattice parameters from input file
and assignin them to global variables*/
static void read_lat_parms(void)
{

   /*declaration of temporary variables used for reading parameters*/

   double csw,cF; /*coefficient of sw term (csw) and of Fermion O(a) boundary counterterm (cF)*/
   char tmpstring[NAME_SIZE]; /*temporary string used for reading*/
   char tmpstring2[NAME_SIZE]; /*temporary string used for reading*/
   int iprop,icorr,eoflg; /*index running on propagators (iprop), correlators (icorr), eoflg ??*/

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
/* DP */      
      read_line("eoflg","%d",eoflg); /*eoflg read from input file ---> ??????*/
/* DP */

      /*check on the validity of the parameters read from input file*/

      /*nprop, ncorr and nnoise must be positive integers*/
      error_root(nprop<1,1,"read_lat_parms [mesons.c]",
                 "Specified nprop must be larger than zero");
      error_root(ncorr<1,1,"read_lat_parms [mesons.c]",
                 "Specified ncorr must be larger than zero");
      error_root(nnoise<1,1,"read_lat_parms [mesons.c]",
                 "Specified nnoise must be larger than zero");

/* DP */
      /*eoflg must be either 0 or 1 ----> ?????*/
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
   file_head.type1=type1; /*Dirac structure of first meson, one for each correlator*/
   file_head.type2=type2; /*Dirac structure of second meson, one for each correlator*/
   file_head.x0=x0s; /*time slice of the source, one for each correlator*/
   file_head.isreal=malloc(ncorr*sizeof(int)); /*array with either 1 (pion-pion case) or 0 for each correlator (??)*/

   /*check of successful memory allocation*/
   error((kappas==NULL)||(mus==NULL)||(isps==NULL)||(props1==NULL)||
         (props2==NULL)||(type1==NULL)||(type2==NULL)||(x0s==NULL)||
         (file_head.kappa1==NULL)||(file_head.kappa2==NULL)||
         (file_head.isreal==NULL),
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
	      read_line("mus","%lf",&mus[iprop]); /*for the given propagator mus(??) is read from input file*/
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

         read_line("type","%s %s",tmpstring,tmpstring2); /*reading of mesons type from input*/
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
   }

   /*broadcast of parameters read on process 0 to
   all other proceses of the communicator group*/

   MPI_Bcast(kappas,nprop,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(mus,nprop,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(isps,nprop,MPI_INT,0,MPI_COMM_WORLD);

   MPI_Bcast(props1,ncorr,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(props2,ncorr,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(type1,ncorr,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(type2,ncorr,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(x0s,ncorr,MPI_INT,0,MPI_COMM_WORLD);

   /*lattice parameters are saved in global structure:
      - lat : set_lat_parms specifies the parameters of the lat structure, and below
              the inverse bare coupling is set to 0, the coefficient of the plaquette
              loops in the gauge action is set to 1, kappa for u and d is set to the
              specified value, for s and c is set to 0, csw and cF are set to the 
              specified values, cG (coefficient of gauge O(a) counter term) is set to 1
      - sw  : structure containing cG, cF and the bare quark mass m0,
              with set_sw_parms m0 is set to the bare quark mass of the up quark
              (sea_quark_mass turns 0,1,2 to m0u,m0s,m0c)
      - file_head : structure containing details of correlators
   */

   set_lat_parms(0.0,1.0,kappas[0],0.0,0.0,csw,1.0,cF); /*parameters of the global structure lat are set*/
   set_sw_parms(sea_quark_mass(0)); /*parameters of the global structure sw are set*/
   file_head.ncorr = ncorr; /*number of correlators saved to global structure*/
   file_head.nnoise = nnoise; /*number of noise vectors saved to global structure*/
   file_head.tvals = NPROC0*L0; /*tvals saved to global structure*/
   tvals = NPROC0*L0; /*tvals saved to global variable*/
   file_head.noisetype = noisetype; /*noisetype saved to global structure*/
   for(icorr=0; icorr<ncorr; icorr++) /*for each correlator the related parameters are saved in file_head*/
   {
      file_head.kappa1[icorr]=kappas[props1[icorr]]; /*kappa of first quark saved to global structure*/
      file_head.kappa2[icorr]=kappas[props2[icorr]]; /*kappa of second quark saved to global structure*/
      /*in the pion-pion case isreal is set to 1, in any other case to 0*/
      if ((type1[icorr]==GAMMA5_TYPE)&&(type2[icorr]==GAMMA5_TYPE)&&
          (props1[icorr]==props2[icorr]))
         file_head.isreal[icorr]=1;
      else
         file_head.isreal[icorr]=0;
   }

   /*the parameters in the lat structure get saved in the fdat file,
   if the option -a is given the consistency with the previously
   written parameters is checked*/

   if (append) /*if the current simulation is the continuation of a previous run ...*/
      check_lat_parms(fdat); /*... the match between the previous and the current lattice parameters is checked*/
   else /*if the current simulation is a new run ...*/
      write_lat_parms(fdat); /*...the lat structure gets written to the fdat file*/
}


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


/*function reading the solvers' parameters from the input file*/
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


/*function to determine the global variables of the
  simulatiom from input*/
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
      error_root((ifile==0)||(ifile==(argc-1)),1,"read_infile [mesons.c]",
                 "Syntax: mesons -i <input file> [-noexp] [-a [-norng]]");

      /*gives an error if the machine has unkown endianness*/
      error_root(endian==UNKNOWN_ENDIAN,1,"read_infile [mesons.c]",
                 "Machine has unknown endianness");

      noexp=find_opt(argc,argv,"-noexp"); /*option to specify configurations reading*/
      append=find_opt(argc,argv,"-a"); /*option to specify output appending*/
      norng=find_opt(argc,argv,"-norng"); /*option to specify generator initialization*/

      /*opening the input file*/

      /*setting stdin to be the input file,
        gives an error if the input file cannot be open*/
      fin=freopen(argv[ifile+1],"r",stdin);
      error_root(fin==NULL,1,"read_infile [mesons.c]",
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

      error_root(fdat==NULL,1,"read_infile [mesons.c]",
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


/*function used to read the old log file. fts, lst and stp are assigned respectively to: 
   -fts, number of the first configuration of the previous run
   -lst, number of the last configuration of the previous run
   -stp, step between each configuration of the previous run*/
static void check_old_log(int *fst,int *lst,int *stp)
{
   int ie,ic,isv;
   int fc,lc,dc,pc;
   int np[4],bp[4];

   fend=fopen(log_file,"r");
   error_root(fend==NULL,1,"check_old_log [mesons.c]",
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

   error_root((ie&0x1)!=0x0,1,"check_old_log [mesons.c]",
              "Incorrect read count");
   error_root((ie&0x2)!=0x0,1,"check_old_log [mesons.c]",
              "Configuration numbers are not equally spaced");
   error_root(isv==0,1,"check_old_log [mesons.c]",
              "Log file extends beyond the last configuration save");

   (*fst)=fc;
   (*lst)=lc;
   (*stp)=dc;
}


/*function used to check that the first, last and step of the configuration scan
reported in the .dat file are the ones passed as inputs - the inputs should be
the first, last and step read from the .log file*/
static void check_old_dat(int fst,int lst,int stp)
{
   int ie,ic;
   int fc,lc,dc,pc;

   fdat=fopen(dat_file,"rb");
   error_root(fdat==NULL,1,"check_old_dat [mesons.c]",
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

   error_root(ic==0,1,"check_old_dat [mesons.c]",
              "No data records found");
   error_root((ie&0x1)!=0x0,1,"check_old_dat [mesons.c]",
              "Configuration numbers are not equally spaced");
   error_root((fst!=fc)||(lst!=lc)||(stp!=dc),1,"check_old_dat [mesons.c]",
              "Configuration range is not as reported in the log file");
}


/*function used to check compatibility with the log and dat files already written
(as safety measure if -a is not given but the log and dat file are present, they won't be
overwritten but instead an error will be raised)*/
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
         error_root((fst!=lst)&&(stp!=step),1,"check_files [mesons.c]",
                    "Continuation run:\n"
                    "Previous run had a different configuration separation");
         
         /*raise an error if the current scan does not continue the previous one*/
         error_root(first!=lst+step,1,"check_files [mesons.c]",
                    "Continuation run:\n"
                    "Configuration range does not continue the previous one");
      }
      else /*if the run is a new run an error is raised to avoid overwriting existing files*/
      {
         /*attempt to read log_file: if that is possible an error is raised
         as to avoid overwriting the .log file*/
         fin=fopen(log_file,"r");
         error_root(fin!=NULL,1,"check_files [mesons.c]",
                    "Attempt to overwrite old *.log file");
         
         /*attempt to read dat_file: if that is possible an error is raised
         as to avoid overwriting the .dat file*/
         fdat=fopen(dat_file,"r");
         error_root(fdat!=NULL,1,"check_files [mesons.c]",
                    "Attempt to overwrite old *.dat file");

         /*creates of the .dat file and checks whether the operation was successful*/
         fdat=fopen(dat_file,"wb");
         error_root(fdat==NULL,1,"check_files [mesons.c]",
                    "Unable to open data file");
         
         /*the structure file_head containing correlators info is written to the .dat file,
         then the .dat file is closed*/
         write_file_head(); /*global file_head structure saved on the .dat file*/
         fclose(fdat); /*.dat file closed*/
      }
   }
}


/*function that sets the .log file as stdout and prints there all the information
related to the simulation (parameters, hardware specifics ecc.)*/
static void print_info(void)
{
   int i,isap,idfl;
   long ip;
   lat_parms_t lat;

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

      error_root(flog==NULL,1,"print_info [mesons.c]",
                 "Unable to open log file");
      printf("\n");

      if (append)
         printf("Continuation run\n\n");
      else
      {
         printf("Computation of meson correlators\n");
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
            printf("SF boundary conditions on the quark fields\n\n");
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
         printf("csw       = %.6f\n",lat.csw);
         printf("cF        = %.6f\n\n",lat.cF);
/* DP */
	 /*printf("eoflg     = %i\n",eoflg);*/ /*had to comment this to avoid problems*/
/* DP */
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
            printf("iprop  = %i %i\n",props1[i],props2[i]);
            printf("type   = %i %i\n",type1[i],type2[i]); /*TODO: strings*/
            printf("x0     = %i\n\n",x0s[i]);
         }
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


/*function used to increase nws, nwv, nwvd if they are smaller than
the minimum value allowed (specified in the parameters of the deflation subspace)*/
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


/*function that allocates memory for the global proplist structure and
fills it with the relevant parameters taken from input*/
static void make_proplist(void)
{
   int i,j,k,iprop,icorr; /*indices used locally*/
   char *kappatype; /*local array containing the Dirac structures of the propagator*/

   proplist.nux0=0; /*number of unique x0 values is set by default to 0*/

   /*memory for the array with unique x0 values is allocated, then correct allocation is checked */
   proplist.ux0=malloc(NPROC0*L0*sizeof(int));
   error(proplist.ux0==NULL,1,"make_proplist [mesons.c]","Out of memory");

   /* unique x0 values */
   /*the following loop determines the number of unique x0 values x0
   and fills the array ux0 with the number of different x0 values*/

   for (icorr=0;icorr<ncorr;icorr++) /*loop over the correlators*/
   {
      for (j=0;j<proplist.nux0;j++) /*loop over the number of different x0 values (= 0 at the first iteration)*/
      {
         if (proplist.ux0[j]==x0s[icorr]) /*the loop breaks if the x0 of the icorr-th correlator is already registered*/
            break;
         
         /*when this loop end there are two cases :
            - j<nux0 if the icorr-th correlator has an x0 at the source already registered in ux0 
            - j=nux0 if the icorr-th correlator has an x0 at the source that is a new value
         */

      }
      if (j==proplist.nux0) /*if j = nux0 then a new different value of x0 has to be registered*/
      {
         proplist.ux0[j]=x0s[icorr]; /*x0s of the current correlator added to the array of unique x0 values*/
         proplist.nux0++; /*number of unique x0 values increased by 1*/
      }
   }

   /*memory allocation for the other attributes of the structure proplist
   and consequent correct memory allocation check*/
   proplist.nprop=malloc(proplist.nux0*sizeof(int));
   proplist.prop=malloc(proplist.nux0*sizeof(int*));
   proplist.type=malloc(proplist.nux0*sizeof(int*));
   kappatype=malloc(MAX_TYPE*nprop*sizeof(char)); /*auxiliary array used to determine the Dirac structure of the propagators*/
   error((proplist.nprop==NULL)||(proplist.prop==NULL)||(proplist.type==NULL)
               ||(kappatype==NULL),
               1,"make_proplist [mesons.c]","Out of memory");
   
   proplist.nmax=0; /*maximum number of propagators set by default to 0 (and it gets updated by the following loop)*/

   for (i=0;i<proplist.nux0;i++) /*loop over the unique x0 values*/
   {
      /*the auxiliary array kappatype is initialized with zeros
      (a 0 in the j-th position means that there is not the j-th Dirac structure for the i-th x0)*/
      for (j=0;j<MAX_TYPE*nprop;j++)
         kappatype[j]=0;

      /*with the following loop the number of propagators at the i-th x0 value is determined*/

      proplist.nprop[i]=0; /*the number of propagators at the i-th value of x0 is initialized to 0*/
      for (icorr=0;icorr<ncorr;icorr++) /*loop over the correlators*/
      {
         if (x0s[icorr]==proplist.ux0[i]) /*if the x0 of the correlator is the i-th one...*/
         {
            /*... and if the gamma structure is not yet registered
            (this meaning that the kappatype specified below is 0) ...*/

            if (!kappatype[type1[icorr]+MAX_TYPE*props2[icorr]]) /*(true(!0) if the gamma structure is new)*/
            {
               kappatype[type1[icorr]+MAX_TYPE*props2[icorr]]=1; /*... gamma structure gets registered*/
               proplist.nprop[i]++; /*... number of propagators with the i-th x0 is increased by 1*/
            }
            if (!kappatype[GAMMA5_TYPE+MAX_TYPE*props1[icorr]]) /*(true(!0) if the gamma structure is new)*/
            {
               kappatype[GAMMA5_TYPE+MAX_TYPE*props1[icorr]]=1; /*... gamma structure gets registered*/
               proplist.nprop[i]++; /*... number of propagators with the i-th x0 is increased by 1*/
            }
         }
      }

      /*the maximum number of propagators (that there can be in a given x0) gets updated*/

      if (proplist.nprop[i]>proplist.nmax) /*if the number of propagator at the i-th x0 is greater than the max ...*/
         proplist.nmax=proplist.nprop[i]; /*... the max number of propagators gets updated*/

      /*once the number of propagators with a given x0 is known the following arrays can be allocated*/

      /*memory allocation for the other attributes of the structure proplist
      and consequent correct memory allocation check*/
      proplist.prop[i]=malloc(proplist.nprop[i]*sizeof(int));
      proplist.type[i]=malloc(proplist.nprop[i]*sizeof(int));
      error((proplist.prop[i]==NULL)||(proplist.type[i]==NULL),
                 1,"make_proplist [mesons.c]","Out of memory");
      

      /*with the following loop the arrays prop and type get filled:
         - prop[i][j] contains the indicex of the j-th propagator positioned at the i-th x0 value
         - type[i][j] contains the Dirac structure of the j-th propagator positioned at the i-th x0 value
      */

      j=0; /*index counting the propagators founded at the i-th x0*/
      for (k=0;k<MAX_TYPE;k++) /*loop over the type of Dirac structures*/
      {
         for (iprop=0;iprop<nprop;iprop++) /*loop over the number of propagators*/
         {
            if (kappatype[k+MAX_TYPE*iprop]) /*when the Dirac structure is one of the registered ones*/
            {
               proplist.prop[i][j]=iprop; /*the iprop-th propagator gets registered*/
               proplist.type[i][j]=k; /*the k-th Dirac structure gets registered*/
               j++; /*the number of propagators at the i-th x0 position increases*/
            }
         }
      }
   }
   free(kappatype); /*the memory for auxiliary array kappatype gets freed*/
}


/*function that sets nws, nwsd, nwv, nwvd according to the
solver method chosen in the input file*/
static void wsize(int *nws,int *nwsd,int *nwv,int *nwvd)
{
   int nsd; /*??*/
   solver_parms_t sp; /*local variable with the solver parameters*/

   /*nws, nwsd, nwv, nwvd are initialized to 0*/

   (*nws)=0;
   (*nwsd)=0;
   (*nwv)=0;
   (*nwvd)=0;

   sp=solver_parms(0); /*set sp to the global structure with the solver parameters*/
   nsd=proplist.nmax+2; /*nsd set to maximum number of propagators +2 (??) */

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
      error_root(1,1,"wsize [mesons.c]",
                 "Unknown or unsupported solver");
}



/*function used to create a random spinor:
the array eta is first set to 0 on the whole lattice, then at the timeslice x0 
it gets filled with random doubles according to the random method specified (globally)*/
static void random_source(spinor_dble *eta, int x0)
{

   /*
      - y0 : x0 after a change of variable where the center is in the cartesian
            coordinate of the local process
      - iy : index running on the time extent of the lattice
      - ix : index of the point on the local lattice
   */
   int y0,iy,ix; 

   set_sd2zero(VOLUME,eta); /*eta is set to 0 on the whole volume*/

   y0=x0-cpr[0]*L0; /*y0 is set to the distance between x0 and the center of the local process*/
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
      else if (noisetype==GAUSS_NOISE) /*if  the random generation is of type GAUSS*/
      {
         for (iy=0;iy<(L1*L2*L3);iy++) /*loop over the timeslice*/
         {
            ix=ipt[iy+y0*L1*L2*L3]; /*index of the point on the local lattice*/
            random_sd(1,eta+ix,1.0); /*random gaussian generation of the spinor entry (just 1) at position ix*/
         }
      }
      else if (noisetype==U1_NOISE) /*if  the random generation is of type U1*/
      {
         for (iy=0;iy<(L1*L2*L3);iy++) /*loop over the timeslice*/
         {
            ix=ipt[iy+y0*L1*L2*L3]; /*index of the point on the local lattice*/
            random_U1_sd(1,eta+ix); /*random U1 generation of the spinor entry (just 1) at position ix*/
         }
      }

      /*the above random generations are done entry by entry,
      they can't be done in one shot like random_sd(L1*L2*L3, eta)
      because the entries are on a timeslice, so they are not contiguous*/

   }

}


/*function that sets psi to be like the xi of eq 6 in the documentatation;
in terms of the input of this function:
psi = (Dw + i mu gamma5)^(-1) * eta --> right ??? */
static void solve_dirac(int prop, spinor_dble *eta, spinor_dble *psi,
                        int *status)
{
   solver_parms_t sp; /*local structure with the solver parameters*/
   sap_parms_t sap; /*structure related to the sap solver*/

   sp=solver_parms(isps[prop]); /*sp assigned to global solver structure depending on solver id of the propagator*/
   set_sw_parms(0.5/kappas[prop]-4.0); /*sets the bare quark mass to that of the propagator with index prop*/

   /*depending on the chosen solver a different method is used*/

   if (sp.solver==CGNE) /*if the solver is CGNE*/
   {
      mulg5_dble(VOLUME,eta); /*multiplies eta by gamma5*/

      tmcg(sp.nmx,sp.res,mus[prop],eta,eta,status); /*eta is set to be the solution of the Wilson-Dirac equation (with a twisted mass term)*/

      /*on process 0 writes on the log file the status of the solver*/
      if (my_rank==0)
         printf("%i\n",status[0]);
      
      /*raises an error if the solver was unsuccesful (i.e. status<0)*/
      error_root(status[0]<0,1,"solve_dirac [mesons.c]",
                 "CGNE solver failed (status = %d)",status[0]);

      Dw_dble(-mus[prop],eta,psi); /*psi = (Dw + i (-mu) gamma5) eta*/

      /*after the above function psi becomes
      psi = (Dw^dagger - i mu gamma5)^(-1) gamma5 * (the original eta) */

      mulg5_dble(VOLUME,psi); /*psi gets multiplied by gamma5*/

      /*after this last function psi in now equal to the xi of equation 6
      of the documentation -> with maybe the twisted mass as extra(??)*/

   }
   else if (sp.solver==SAP_GCR) /*if the solver is SAP_GCR*/
   {

      /*initialization of sap solvers*/
      sap=sap_parms(); /*sap set to the global structure with the parameters of the SAP preconditioner*/
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy); /*set the parameters of the sap preconditioner*/

      /*psi is set as the solution of the dirac equation with source eta (using sap_gcr solver)*/
      sap_gcr(sp.nkv,sp.nmx,sp.res,mus[prop],eta,psi,status); 

      /*on process 0 writes on the log file the status of the solver*/
      if (my_rank==0)
         printf("%i\n",status[0]);
      
      /*raises an error if the solver was unsuccesful (i.e. status<0)*/
      error_root(status[0]<0,1,"solve_dirac [mesons.c]",
                 "SAP_GCR solver failed (status = %d)",status[0]);

   }
   else if (sp.solver==DFL_SAP_GCR) /*if the solver is DFL_SAP_GCR*/
   {

      /*initialization of sap solvers*/
      sap=sap_parms(); /*sap set to the global structure with the parameters of the SAP preconditioner*/
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy); /*set the parameters of the sap preconditioner*/

      /*psi is set as the solution of the dirac equation with source eta (using dfl_sap_gcr solver)*/
      dfl_sap_gcr2(sp.nkv,sp.nmx,sp.res,mus[prop],eta,psi,status);

      /*on process 0 writes on the log file the status of the solver*/
      if (my_rank==0)
         printf("%i %i\n",status[0],status[1]);

      /*raises an error if the solver was unsuccesful (i.e. status<0)*/
      error_root((status[0]<0)||(status[1]<0),1,
                 "solve_dirac [mesons.c]","DFL_SAP_GCR solver failed "
                 "(status = %d,%d)",status[0],status[1]);
   }
   else /*if the specified solver is unknown raises an error*/
      error_root(1,1,"solve_dirac [mesons.c]",
                 "Unknown or unsupported solver");
}


/* xi = \gamma_5 Gamma^\dagger eta (comment to be removed)*/

/*function that construct xi (source term of Dirac equation) from eta according to what specified in the documentation:
   - eta : is the source for zeta and stay as it is 
   - xi : is the source of the xi specified in the documentation so it must be equal to
          gamma5 * GAMMA^dagger * eta, where GAMMA is given by the type specified in the
          input file (such product is done according to the table 1 of the documentation)
*/
void make_source(spinor_dble *eta, int type, spinor_dble *xi)
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
         assign_msd2sd(VOLUME,eta,xi);
         mulg0g5_dble(VOLUME,xi);
         break;
      case GAMMA1_TYPE:
         assign_msd2sd(VOLUME,eta,xi);
         mulg1g5_dble(VOLUME,xi);
         break;
      case GAMMA2_TYPE:
         assign_msd2sd(VOLUME,eta,xi);
         mulg2g5_dble(VOLUME,xi);
         break;
      case GAMMA3_TYPE:
         assign_msd2sd(VOLUME,eta,xi);
         mulg3g5_dble(VOLUME,xi);
         break;
      case GAMMA5_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         break;
      case GAMMA0GAMMA1_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg2g3_dble(VOLUME,xi);
         break;
      case GAMMA0GAMMA2_TYPE:
         assign_msd2sd(VOLUME,eta,xi);
         mulg1g3_dble(VOLUME,xi);
         break;
      case GAMMA0GAMMA3_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg1g2_dble(VOLUME,xi);
         break;
      case GAMMA0GAMMA5_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg0_dble(VOLUME,xi);
         break;
      case GAMMA1GAMMA2_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg0g3_dble(VOLUME,xi);
         break;
      case GAMMA1GAMMA3_TYPE:
         assign_msd2sd(VOLUME,eta,xi);
         mulg0g2_dble(VOLUME,xi);
         break;
      case GAMMA1GAMMA5_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg1_dble(VOLUME,xi);
         break;
      case GAMMA2GAMMA3_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg0g1_dble(VOLUME,xi);
         break;
      case GAMMA2GAMMA5_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg2_dble(VOLUME,xi);
         break;
      case GAMMA3GAMMA5_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg3_dble(VOLUME,xi);
         break;
      case ONE_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg5_dble(VOLUME,xi);
         break;
      default:
         error_root(1,1,"make_source [mesons.c]",
                 "Unknown or unsupported type");
   }
}


/*function to construct xi from eta and type as :
    xi = - GammaBar^dagger gamma5 eta
where GammaBar is a gamma structure determined from type;
the references in the documentation are equation 6 and table 1
*/
void make_xi(spinor_dble *eta,int type,spinor_dble *xi)
{
   /* xi = -\bar Gamma^\dagger \gamma_5 eta  (comment to be removed)*/

   /*
   - assign_msd2sd : sets the second spinor to be equal to minus the first one
   - assign_sd2sd : sets the first spinor to be equal to the first one
   - mulgigj : multiplies the spinor by gamma_i gamma_j

   VOLUME means that these operation are done on the whole lattice
   */

   switch (type)
   {
      case GAMMA0_TYPE:
         assign_msd2sd(VOLUME,eta,xi);
         mulg0g5_dble(VOLUME,xi);
         break;
      case GAMMA1_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg1g5_dble(VOLUME,xi);
         break;
      case GAMMA2_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg2g5_dble(VOLUME,xi);
         break;
      case GAMMA3_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg3g5_dble(VOLUME,xi);
         break;
      case GAMMA5_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         break;
      case GAMMA0GAMMA1_TYPE:
         assign_msd2sd(VOLUME,eta,xi);
         mulg2g3_dble(VOLUME,xi);
         break;
      case GAMMA0GAMMA2_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg1g3_dble(VOLUME,xi);
         break;
      case GAMMA0GAMMA3_TYPE:
         assign_msd2sd(VOLUME,eta,xi);
         mulg1g2_dble(VOLUME,xi);
         break;
      case GAMMA0GAMMA5_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg0_dble(VOLUME,xi);
         break;
      case GAMMA1GAMMA2_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg0g3_dble(VOLUME,xi);
         break;
      case GAMMA1GAMMA3_TYPE:
         assign_msd2sd(VOLUME,eta,xi);
         mulg0g2_dble(VOLUME,xi);
         break;
      case GAMMA1GAMMA5_TYPE:
         assign_msd2sd(VOLUME,eta,xi);
         mulg1_dble(VOLUME,xi);
         break;
      case GAMMA2GAMMA3_TYPE:
         assign_sd2sd(VOLUME,eta,xi);
         mulg0g1_dble(VOLUME,xi);
         break;
      case GAMMA2GAMMA5_TYPE:
         assign_msd2sd(VOLUME,eta,xi);
         mulg2_dble(VOLUME,xi);
         break;
      case GAMMA3GAMMA5_TYPE:
         assign_msd2sd(VOLUME,eta,xi);
         mulg3_dble(VOLUME,xi);
         break;
      case ONE_TYPE:
         assign_msd2sd(VOLUME,eta,xi);
         mulg5_dble(VOLUME,xi);
         break;
      default:
         error_root(1,1,"make_xi [mesons.c]",
                 "Unknown or unsupported type");
   }
}


/*core function used to compute the correlator once all the input
variable have been specified in the corresponding global structure*/
static void correlators(void)
{

   /** declaration of local variables **/

   /*
      - ix0 : index running over the different unique values of x0
      - inoise : index running over the random generated noise vectors
      - iprop : index running over the different propagators with a given ix0
      - icorr : index running over the different correlators
      - ip1,ip2 : auxialiary variables to index the propagators
      - l : auxiliary variable used as a generic index
      - stat : status array used in the Dirac inversion
      - y0 : index running on the time extent of the lattice
      - iy : index on the lattice obtained from the cartesian coordinates of the point y
   */
   int ix0,inoise,iprop,icorr,ip1,ip2,l,stat[4],y0,iy;
   /*
      - eta : random noise spinors used to perform the Dirac inversion
      - xi : spinor used as source for the Dirac equation
      - zeta : spinor used as solution to the Dirac equation
      - wsd : array containing (number of propagators +2) spinor fields,
              that are eta, xi and other n fields contained in zeta
              (wsd = Workspace of Spinor Double)
   */
   spinor_dble *eta,*xi,**zeta,**wsd;
   complex_dble tmp; /*temporary complex variable*/


   /** allocation of the spinor fields **/

   /*the spinor fields needed for the computation are here allocated,
   then the correct memory allocation is checked*/

   wsd=reserve_wsd(proplist.nmax+2); /*a workspace with nprop+2 spinor fields is allocated*/
   eta=wsd[0]; /*first array (field) in wsd is assigned to eta*/
   xi=wsd[1]; /*second array (field) in wsd is assigned to xi*/
   zeta=malloc(proplist.nmax*sizeof(spinor_dble*)); /*allocation of nprop spinors for zeta (why needed ??)*/
   error(zeta==NULL,1,"correlators [mesons.c]","Out of memory"); /*check on allocation*/

   /*zeta is now set to be the remaining spinors already reserved in wsd*/
   for (l=0;l<proplist.nmax;l++)
      zeta[l]=wsd[l+2];

   /*why is the zeta allocation needed ?
   could

   zeta = wsd[2]
   
   do the trick ??
   */


   /** initialization of other temporary variables **/

   /*the total number of values of the correlators needed is number of noise vectors times number of correlators
   times number of time intervals, they are all initialized to 0*/
   for (l=0;l<nnoise*ncorr*tvals;l++)
   {
      data.corr_tmp[l].re=0.0;
      data.corr_tmp[l].im=0.0;
   }


   /** begin of the actual computation of the correlator **/

   /*informative print on process 0*/
   if (my_rank==0)
      printf("Inversions:\n");
   
   
   /*loop over the different unique x0 values present in the correlators*/
   for (ix0=0;ix0<proplist.nux0;ix0++) /*ix0 ranges from 0 to the total number of different x0s*/
   {


      /*informative print on process 0*/
      if (my_rank==0)
         printf("   x0=%i\n",proplist.ux0[ix0]); /*the value of the x0 being used is printed on the .log file*/
      

      /*loop over random the noise vectors generated*/
      for (inoise=0;inoise<nnoise;inoise++) /*i noise range from 0 to the total number of noise vectors*/
      {


         /*informative print on process 0*/
         if (my_rank==0)
            printf("      noise vector %i\n",inoise); /*the number of noise vectors being generated is written on the .log file*/
         

         /*generation of the random source:
         eta is first set to 0 on the whole space time volume, then at the timeslice specified in the input file
         (that is at x0 = ux0[ix0]) eta is filled with random complex numbers according to the chosen
         random method (U1, Z2 or GAUSS)*/
         random_source(eta,proplist.ux0[ix0]); /*eta gets filled at the timeslice ux0[ix0] with random numbers*/

         /*with the following loop for each propagator first a source is constructed, then the solution
         to the Dirac equation (with the propagator's parameters) in the presence of the source is found
         (equation 6 of the documentation)*/

         /*loop over the number of propagators (quarks) to be computed in the x0 specified by ix0*/
         for (iprop=0;iprop<proplist.nprop[ix0];iprop++) /*iprop ranges from 0 to the number of quark to be computex in ux0[ix0]*/
         {

            /*informative print on process 0*/
            if (my_rank==0)
               printf("         type=%i, prop=%i, status:",
                   proplist.type[ix0][iprop], proplist.prop[ix0][iprop]); /*type and index of prop printed on .log*/
            
            /*the stochastic sources needed for the invarision of the Dirac operator are now constructed:
               - zeta (in documentation) : is constructed from eta
               - xi (in documentation) : is constructerd from the xi obtained by the following function
            */
            make_source(eta,proplist.type[ix0][iprop],xi); /*sets xi to be gamma_5 GAMMA^dagger eta */

            solve_dirac(proplist.prop[ix0][iprop],xi,zeta[iprop],stat); /*zeta is set to be the solution of the Dirac equation with source xi*/

            /*after the the call to solve_dirac zeta becomes the xi of eq 6 of the documentation:
               zeta = (Dw + i mu gamma5)^(-1) gamma5 GAMMA^dagger eta */
         }

         /* combine propagators to correlators (comment to be removed)*/

         /*loop over the correlators to be computed*/
         for (icorr=0;icorr<ncorr;icorr++) /*icorr ranges from 0 to the number of correlators*/
         {

            /*the computation is done only if the current x0 is the x0 of the correlator*/
            if (x0s[icorr]==proplist.ux0[ix0])
            {
               /* find the two propagators that are needed for this icorr (comment to be removed)*/

               /*first the two propagators that appear in the icorr-th correlator need to be found,
               these two propagators will be indexed by ip1 and ip2*/

               /*ip1 and ip2 are initialized to 0*/
               ip1=0;
               ip2=0;

               /*to find ip1,ip2 we loop over all the propagators having as x0 the current x0*/
               for (iprop=0;iprop<proplist.nprop[ix0];iprop++) /*iprop ranges from 0 to nprop with x0 identified by ix0*/
               {
                  /*if the Dirac structure (type) and the type of quark (props) match
                  then the index of the first propagator is found*/
                  if ((type1[icorr]==proplist.type[ix0][iprop])&&
                      (props2[icorr]==proplist.prop[ix0][iprop]))
                     ip1=iprop;
                  
                  /*if the Dirac structure (type) and the type of quark (props) match
                  then the index of the second propagator is found*/
                  if ((GAMMA5_TYPE==proplist.type[ix0][iprop])&&
                      (props1[icorr]==proplist.prop[ix0][iprop]))
                     ip2=iprop;

                  /*the fact that for the second propagator the gamma structure required is GAMMA5 is
                  because in this way zeta[ip2] is exactly the solution given in eq5 of the documentation*/
               }

               /*to compute the correlator the reference equation is equation 7 in the documentation*/

               /*with the following function the piece inside curve brackets in equation 7 is computed*/

               make_xi(zeta[ip1],type2[icorr],xi); /*xi is set to be -GammaBar^dagger gamam5 zeta*/

               /*to perform the computation we have to sum over y*/

               for (y0=0;y0<L0;y0++) /*sum over the time values y0*/
               {
                  for (l=0;l<L1*L2*L3;l++) /*sum over the space index l*/
                  {
                     iy = ipt[l+y0*L1*L2*L3]; /*index of the point on the local lattice*/

                     /*first we compute the contribution from the given space-time point...*/

                     tmp = spinor_prod_dble(1,0,xi+iy,zeta[ip2]+iy); /*tmp set to be the scalar product between xi and eta at position iy*/
                     /*(this tmp is a piece (at position laballed by iy) of the sum in equation 7,
                     indeed we have that xi is the piece inside curve brackets, and zeta[ip2] what
                     we get from equation 5 of the documentation)*/

                     /*...then we sum it to the total correlator*/

                     data.corr_tmp[inoise+nnoise*(cpr[0]*L0+y0
                        +file_head.tvals*icorr)].re += tmp.re; /*real part gets updated*/
                     data.corr_tmp[inoise+nnoise*(cpr[0]*L0+y0
                        +file_head.tvals*icorr)].im += tmp.im; /*imaginary part gets updated*/

                     /*in data.corr_tmp all the computed correlator get stored,
                     and they are indexed as shown in the two lines above
                     (cpr is the cartesian coordinate of the local process)*/
                  }
               }

            }

         } /*end of loop over i corr*/

      } /*end of loop over inoise*/

   } /*end of loop over ix0*/


   /** final result**/

   /*each process computes a part of the correlator and stores it inside data.corr_tmp, 
   with the following function the information coming from all the processes gets
   combined (summed, since MPI_SUM) into data.corr, that hence now stores the
   complete results of all the correlators*/
   MPI_Allreduce(data.corr_tmp,data.corr,nnoise*ncorr*file_head.tvals*2
      ,MPI_DOUBLE,MPI_SUM, MPI_COMM_WORLD);

   /** memory deallocation**/

   free(zeta); /*deallocation of memory used for zeta*/
   release_wsd(); /*release of the workspace allocated for all the spinors*/
}


/*computes the correlator related to the gauge configuration with index nc
by calling the correlators() function*/
static void set_data(int nc)
{
   data.nc=nc; /*sets data.nc to be te index of the gauge configuration passed as input*/
   correlators(); /*computes the correlator with the gauge configuration nc*/

   /*on process 0 prints to the .log file information regarding the correlator*/
   if (my_rank==0)
   {
      printf("G(t) =  %.4e%+.4ei",data.corr[0].re,data.corr[0].im); /*prints the correlator at the firsttime,...*/
      printf(",%.4e%+.4ei,...",data.corr[1].re,data.corr[1].im); /*...at the second time ...*/
      printf(",%.4e%+.4ei",data.corr[file_head.tvals-1].re,
                           data.corr[file_head.tvals-1].im); /*... and at the last time ...*/
      printf("\n");
      fflush(flog); /*the output (that is directed on the log file) is flushed*/
   }
}


/*function used to initialize the random number generator (rng)*/
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
         error_root(ic!=(first-step),1,"init_rng [mesons.c]",
                    "Configuration number mismatch (*.rng file)");
      }
   }
   else /*if the run is a new run ...*/
      start_ranlux(level,seed); /*...traditional generator initialization*/
}


/*function used to save the current state of the random number generators rlxs and rlxd
(and initialize them if needed)*/
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
      error(rlxs_state==NULL,1,"save_ranlux [mesons.c]",
            "Unable to allocate state arrays");
   }

   /*save the state of the generators*/

   rlxs_get(rlxs_state); /*store the state of the rlxs generator in rlsxs_state*/
   rlxd_get(rlxd_state); /*store the state of the rlxd generator in rlxd_state*/
}


/*function that sets the states of the rlxs and rlxd generators to the
states currently saved in the globl variables rlxs_state and rlxd_state*/
static void restore_ranlux(void)
{
   rlxs_reset(rlxs_state); /*sets the rlxs_generator to the state rlxs_state*/
   rlxd_reset(rlxd_state); /*sets the rlxd_generator to the state rlxd_state*/
}


/*function that sets the endflag if in the log directory there is a file
with the .end extension and with the same name of the run
(that's a gentle way to kill the program execution from terminal)*/
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


/***** main of the program *****/

int main(int argc,char *argv[])
{

   /** definition of variables **/

   int nc,iend,status[4]; /*nc=index and iend=end flag (1 set, 0 not set) of the configuration loop, status=status of deflation subspace*/
   int nws,nwsd,nwv,nwvd; /*number of workspace fields (s=spinor, v=vector, d=double precision)*/
   double wt1,wt2,wtavg; /*time variables for time estimates in the loop over configurations*/
   dfl_parms_t dfl; /*structure with parameters of the deflation subspace*/

   /** openMPI inizialization **/

   MPI_Init(&argc,&argv); /*Inizialization of MPI execution environment*/
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank); /*setting my_rank to the rank of the calling process*/
   
   /** file and simulation parameters initialization **/

   read_infile(argc,argv); /*read input from command line and input file*/
   alloc_data(); /*allocate memory for the data structure*/
   check_files(); /*check compatibility with .dat and .log files already written*/
   print_info(); /*write all the variables and parameters of the simulation to the .log file*/
   dfl=dfl_parms(); /*get the parameters of the deflation subspace from global structure*/

   geometry(); /*compute global arrays related to MPI process grid and to indexes of lattice grid*/
   init_rng(); /*initialization of the random number generator*/

   make_proplist(); /*construction of the global proplist structure from parameters read from input*/
   wsize(&nws,&nwsd,&nwv,&nwvd); /*set nws, nwsd, nwv, nwvd according to the solver method chosen in input file*/
   alloc_ws(nws); /*allocates a workspace of nws single-precision spinor fields*/
   alloc_wsd(nwsd); /*allocates a workspace of nwsd double-precision spinor fields*/
   alloc_wv(nwv); /*allocates a workspace of nws single-precision vector fields*/
   alloc_wvd(nwvd); /*allocates a workspace of nwsd double-precision spinor fields*/

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
         read_cnfg(cnfg_file); /*reads the configuration from the cnfg_file, saves it (where ??) and resets the generators*/
         restore_ranlux(); /*set the states of the generators to the saved values (saved before the reset due to read_cnfg)*/
      }
      else /*if instead -noexp is not set the configurations are read in exported file format*/
      {
         sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,nc); /*get the name of the configuration file from input parameters*/
         import_cnfg(cnfg_file); /*reads the configuration from the cnfg_file, saves it (where ??)*/
      }

      /*the deflation subspace is generated*/
      if (dfl.Ns) /*if the number of deflation mode is different from 0...*/
      {
         /*... the deflation subspace is initialized*/

         /*initialize deflation subspace and its compute basis vectors,
         an error is raised if the operation fails*/
         dfl_modes(status); 
         error_root(status[0]<0,1,"main [mesons.c]",
                    "Deflation subspace generation failed (status = %d)",
                    status[0]);

         /*on process 0 the (succesful) status of the deflation subspace generation is written to the .log file*/
         if (my_rank==0)
            printf("Deflation subspace generation: status = %d\n",status[0]);
      }

      set_data(nc);
      write_data();

      export_ranlux(nc,rng_file);
      error_chk();
      
      MPI_Barrier(MPI_COMM_WORLD); /*synchronization between all the MPI processes in the group*/
      wt2=MPI_Wtime(); /*time measured after the nc-th configuration is processed*/
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

      /*once the current configuration has been computed the program checks if its execution has to be
      terminated early by looking for the endflag
      (the user can kill the program gently by creating a .end file with the same name of the run in the
      log directory, if such a file is found the function below sets the endflag to true)*/

      check_endflag(&iend); /*check if the endlfag has been raised by the user (if so that's the last loop iteration)*/

      /*on process 0 before a new loop iteration the output gets flushed and the file gets saved*/
      if (my_rank==0)
      {
         fflush(flog); /*flush of printf to the .log file*/
         copy_file(log_file,log_save); /*.log file saved to .log~ file for backup*/
         copy_file(dat_file,dat_save); /*.dat file saved to .dat~ file for backup*/
         copy_file(rng_file,rng_save); /*.rng file saved to .rng~ file for backup*/
      }
   }

   /** program ending **/

   /*on process 0 the .log file is closed (it was still open since it served as stdout)*/
   if (my_rank==0)
   {
      fclose(flog);
   }

   MPI_Finalize(); /*MPI execution environmment is terminated*/
   exit(0); /*termination of main*/
}
