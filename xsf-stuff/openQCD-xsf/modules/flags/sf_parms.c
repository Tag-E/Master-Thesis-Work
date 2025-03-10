
/*******************************************************************************
*
* File sf_parms.c
*
* Copyright (C) 2012 John Bulava, Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Schroedinger functional boundary values
*
* The externally accessible functions are
*
*   sf_parms_t set_sf_parms(double *phi,double *phi_prime)
*     Sets the parameters of the boundary fields in the Schroedinger
*     functional. The parameters are
*
*       phi           Angles phi[0],phi[1] at time 0.
*
*       phi_prime     Angles phi[0],phi[1] at time T.
*
*     See the notes for further explanations. This program may only be
*     called once.
*
*   sf_parms_t sf_parms(void)
*     Returns a structure containing the parameters of the boundary
*     fields. 
*
*   void print_sf_parms(void)
*     Prints the parameters of the boundary fields to stdout on MPI
*     process 0.
*
*   void write_sf_parms(FILE *fdat)
*     Writes the parameters of the boundary fields to the file fdat on
*     MPI process 0.
*
*   void check_sf_parms(FILE *fdat)
*     Compares the parameters of the boundary fields with the values
*     stored on the file fdat on MPI process 0, assuming the latter were
*     written to the file by the program write_sf_parms().
*
*   int sf_flg(void)
*     Returns 1 if the Schroedinger functional boundary values have been
*     set by calling set_sf_parms() and 0 otherwise.
*
* Notes:
*
* The boundary values of the field variables on the spatial links at time
* x0=0 are
*
*  U(x,k)=diag{exp(i*phi[0]/N1),{exp(i*phi[1]/N2),{exp(i*phi[2]/N3)},
*
* where
*
*  phi[2]=-phi[0]-phi[1],  N1=NPROC1*L1, N2=NPROC2*L2, N1=NPROC3*L3.
*
* The same formulae hold at time x0=NPROC0*L0-1 except that phi[k] is
* replaced by phi_prime[k].
*
* The type sf_parms_t is defined in flags.h. Its elements are
*
*  flg           Indicates whether the boundary values are set (flg=1)
*                or not (flg=0). The flag is initialized to 0 and is
*                set to 1 when set_sf_parms() is called.
*
*  phi[3]        Parameters phi[k] (k=0,1,2).
*
*  phi_prime[3]  Parameters phi_prime[k].
*
* When set_sf_parms() is called, SF boundary conditions are implied and
* all programs sensitive to the boundary conditions will act accordingly.
* It is, however, up to the programmer to ensure that the gauge fields
* have the specified boundary values by invoking the programs sfbc() and
* sfbcd() (see lattice/bcnds.c). 
*
* To ensure the consistency of the data base, the parameters must be set
* simultaneously on all processes. Note that the boundary values stored
* in the data base cannot be changed once they were set by set_sf_parms().
*
*******************************************************************************/

#define SF_PARMS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "global.h"

static sf_parms_t sf={0,{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};


sf_parms_t set_sf_parms(double *phi,double *phi_prime,double *theta)
{
   double dprms[7];
   int i;

   error_root(sf.flg!=0,1,"set_sf_parms [sf_parms.c]",
              "Attempt to reset SF boundary values");
   
   if (NPROC>1)
   {
      dprms[0]=phi[0];
      dprms[1]=phi[1];
      dprms[2]=phi_prime[0];
      dprms[3]=phi_prime[1];
      dprms[4]=theta[0];
      dprms[5]=theta[1];
      dprms[6]=theta[2];

      MPI_Bcast(dprms,7,MPI_DOUBLE,0,MPI_COMM_WORLD);

      error((dprms[0]!=phi[0])||(dprms[1]!=phi[1])||
            (dprms[2]!=phi_prime[0])||(dprms[3]!=phi_prime[1])||
            (dprms[4]!=theta[0])||(dprms[5]!=theta[1])||
            (dprms[6]!=theta[2]),1,
            "set_sf_parms [sf_parms.c]","Parameters are not global");
   }

   sf.flg=1;
   sf.phi[0]=phi[0];
   sf.phi[1]=phi[1];
   sf.phi[2]=-phi[0]-phi[1];
   sf.phi_prime[0]=phi_prime[0];
   sf.phi_prime[1]=phi_prime[1];
   sf.phi_prime[2]=-phi_prime[0]-phi_prime[1];   
   for (i=0;i<3;i++) sf.theta[i]=theta[i];
   
   return sf;
}


sf_parms_t sf_parms(void)
{
   return sf;
}


void print_sf_parms(void)
{
   int my_rank;
   
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      printf("Schroedinger functional boundary values:\n");
      printf("phi = %.4e,%.4e,%.4e\n",
             sf.phi[0],sf.phi[1],sf.phi[2]);
      printf("phi' = %.4e,%.4e,%.4e\n",
             sf.phi_prime[0],sf.phi_prime[1],sf.phi_prime[2]);
      printf("theta = %.4e,%.4e,%.4e\n\n",
             sf.theta[0],sf.theta[1],sf.theta[2]);

   }
}


void write_sf_parms(FILE *fdat)
{
   int my_rank,endian;
   int iw;
   stdint_t istd[1];
   double dstd[9];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();
   
   if (my_rank==0)
   {
      istd[0]=(stdint_t)(sf.flg);
      dstd[0]=sf.phi[0];
      dstd[1]=sf.phi[1];
      dstd[2]=sf.phi[2];
      dstd[3]=sf.phi_prime[0];
      dstd[4]=sf.phi_prime[1];
      dstd[5]=sf.phi_prime[2];
      dstd[6]=sf.theta[0];
      dstd[7]=sf.theta[1];
      dstd[8]=sf.theta[2];
      
      if (endian==BIG_ENDIAN)
      {
         bswap_int(1,istd);
         bswap_double(9,dstd);
      }
      
      iw=fwrite(istd,sizeof(stdint_t),1,fdat);
      iw+=fwrite(dstd,sizeof(double),9,fdat);

      error_root(iw!=10,1,"write_sf_parms [sf_parms.c]",
                 "Incorrect write count");
   }
}


void check_sf_parms(FILE *fdat)
{
   int my_rank,endian;
   int i,ir,ie;
   stdint_t istd[1];
   double dstd[9];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();
   
   if (my_rank==0)
   {
      ir=fread(istd,sizeof(stdint_t),1,fdat);         
      ir+=fread(dstd,sizeof(double),9,fdat);

      error_root(ir!=10,1,"check_sf_parms [sf_parms.c]",
                 "Incorrect read count");         

      if (endian==BIG_ENDIAN)
      {
         bswap_int(1,istd);
         bswap_double(9,dstd);
      }
      
      ie=(istd[0]!=(stdint_t)(sf.flg));
      
      for (i=0;i<3;i++)
      {
         ie|=(dstd[i]!=sf.phi[i]);
         ie|=(dstd[3+i]!=sf.phi_prime[i]);
         ie|=(dstd[6+i]!=sf.theta[i]);
      }
         
      error_root(ie!=0,1,"check_sf_parms [sf_parms.c]",
                 "Parameters do not match");
   }
}

int sf_flg(void)
{
   return sf.flg;
}
