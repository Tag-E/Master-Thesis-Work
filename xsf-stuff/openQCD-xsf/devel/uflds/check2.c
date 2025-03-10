
/*******************************************************************************
*
* File check2.c
*
* Copyright (C) 2010, 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Renormalization of the link variables
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "global.h"

#define N0 (NPROC0*L0)


static int is_zero_dble(su3_dble *ud)
{
   int i;
   double r[18];

   r[ 0]=(*ud).c11.re;
   r[ 1]=(*ud).c11.im;
   r[ 2]=(*ud).c12.re;
   r[ 3]=(*ud).c12.im;
   r[ 4]=(*ud).c13.re;
   r[ 5]=(*ud).c13.im;

   r[ 6]=(*ud).c21.re;
   r[ 7]=(*ud).c21.im;
   r[ 8]=(*ud).c22.re;
   r[ 9]=(*ud).c22.im;
   r[10]=(*ud).c23.re;
   r[11]=(*ud).c23.im;

   r[12]=(*ud).c31.re;
   r[13]=(*ud).c31.im;
   r[14]=(*ud).c32.re;
   r[15]=(*ud).c32.im;
   r[16]=(*ud).c33.re;
   r[17]=(*ud).c33.im;

   for (i=0;i<18;i++)
   {
      if (r[i]!=0.0)
         return 0;
   }

   return 1;
}


static complex_dble det_dble(su3_dble *u)
{
   complex_dble det1,det2,det3,detu;

   det1.re=
      ((*u).c22.re*(*u).c33.re-(*u).c22.im*(*u).c33.im)-
      ((*u).c23.re*(*u).c32.re-(*u).c23.im*(*u).c32.im);
   det1.im=
      ((*u).c22.re*(*u).c33.im+(*u).c22.im*(*u).c33.re)-
      ((*u).c23.re*(*u).c32.im+(*u).c23.im*(*u).c32.re);
   det2.re=
      ((*u).c21.re*(*u).c33.re-(*u).c21.im*(*u).c33.im)-
      ((*u).c23.re*(*u).c31.re-(*u).c23.im*(*u).c31.im);
   det2.im=
      ((*u).c21.re*(*u).c33.im+(*u).c21.im*(*u).c33.re)-
      ((*u).c23.re*(*u).c31.im+(*u).c23.im*(*u).c31.re);
   det3.re=
      ((*u).c21.re*(*u).c32.re-(*u).c21.im*(*u).c32.im)-
      ((*u).c22.re*(*u).c31.re-(*u).c22.im*(*u).c31.im);
   det3.im=
      ((*u).c21.re*(*u).c32.im+(*u).c21.im*(*u).c32.re)-
      ((*u).c22.re*(*u).c31.im+(*u).c22.im*(*u).c31.re);

   detu.re=
      ((*u).c11.re*det1.re-(*u).c11.im*det1.im)-
      ((*u).c12.re*det2.re-(*u).c12.im*det2.im)+
      ((*u).c13.re*det3.re-(*u).c13.im*det3.im);
   detu.im=
      ((*u).c11.re*det1.im+(*u).c11.im*det1.re)-
      ((*u).c12.re*det2.im+(*u).c12.im*det2.re)+
      ((*u).c13.re*det3.im+(*u).c13.im*det3.re);

   return detu;
}


static double dev_uudag_dble(su3_dble *u,su3_dble *v)
{
   int i;
   double r[18],d,dmax;
   su3_dble vdag,w;

   _su3_dagger(vdag,(*v));
   _su3_times_su3(w,(*u),vdag);

   w.c11.re-=1.0;
   w.c22.re-=1.0;
   w.c33.re-=1.0;

   r[ 0]=w.c11.re;
   r[ 1]=w.c11.im;
   r[ 2]=w.c12.re;
   r[ 3]=w.c12.im;
   r[ 4]=w.c13.re;
   r[ 5]=w.c13.im;

   r[ 6]=w.c21.re;
   r[ 7]=w.c21.im;
   r[ 8]=w.c22.re;
   r[ 9]=w.c22.im;
   r[10]=w.c23.re;
   r[11]=w.c23.im;

   r[12]=w.c31.re;
   r[13]=w.c31.im;
   r[14]=w.c32.re;
   r[15]=w.c32.im;
   r[16]=w.c33.re;
   r[17]=w.c33.im;

   dmax=0.0;

   for (i=0;i<18;i++)
   {
      d=fabs(r[i]);
      if (d>dmax)
         dmax=d;
   }

   return dmax;
}


static double dev_detu_dble(su3_dble *u)
{
   double d,dmax;
   complex_dble detu;

   detu=det_dble(u);
   dmax=0.0;

   d=fabs(1.0-detu.re);
   if (d>dmax)
      dmax=d;
   d=fabs(detu.im);
   if (d>dmax)
      dmax=d;

   return dmax;
}


static void check_ud(double *dev1,double *dev2)
{
   int iu,ix,ifc,x0;
   double d1,d2,dmax1,dmax2;
   su3_dble *u,*ub,*um;
   
   ub=udfld();
   um=ub+4*VOLUME;
   dmax1=0.0;
   dmax2=0.0;

   for (u=ub;u<um;u++)
   {
      iu=(u-ub);
      ix=iu/8+VOLUME/2;
      ifc=iu%8;
      x0=global_time(ix);

      if (((x0==0)&&(ifc==1))||((x0==(N0-1))&&(ifc==0)))
      {
         d1=(double)(1-is_zero_dble(u));
         d2=d1;
      }
      else
      {
         d1=dev_uudag_dble(u,u);
         d2=dev_detu_dble(u);
      }
      
      if (d1>dmax1)
         dmax1=d1;
      if (d2>dmax2)
         dmax2=d2;      
   }

   MPI_Reduce(&dmax1,dev1,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
   MPI_Reduce(&dmax2,dev2,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
   MPI_Bcast(dev1,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(dev2,1,MPI_DOUBLE,0,MPI_COMM_WORLD);   
}


static double cmp_ud(su3_dble *usv)
{
   int ix,mu,x0;   
   double d1,dmax1;
   su3_dble *ub,*u,*v,*um;

   ub=udfld();
   um=ub+4*VOLUME;
   v=usv;
   dmax1=0.0;

   for (u=ub;u<um;u++)
   {
      ix=(u-ub);
      mu=(ix%8);
      ix=(ix/8)+(VOLUME/2);
      x0=global_time(ix);

      if (((x0==0)&&(mu==1))||((x0==(NPROC0*L0-1))&&(mu==0)))
      {
         error_loc((is_zero_dble(u)!=1)||(is_zero_dble(v)!=1),1,
                   "cmp_ud [check2.c]","Field does not satisfy open bc");
      }
      else
      {
         d1=dev_uudag_dble(u,v);

         if (d1>dmax1)
            dmax1=d1;
      }
      
      v+=1;
   }

   error_chk();
   
   d1=dmax1;
   MPI_Reduce(&d1,&dmax1,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
   MPI_Bcast(&dmax1,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   return dmax1;
}


static void tilt_ud(double eps)
{
   double r[18];
   su3_dble *ud,*um;
   
   ud=udfld();
   um=ud+4*VOLUME;

   for (;ud<um;ud++)
   {
      gauss_dble(r,18);

      (*ud).c11.re+=eps*r[ 0];
      (*ud).c11.im+=eps*r[ 1];      
      (*ud).c12.re+=eps*r[ 2];
      (*ud).c12.im+=eps*r[ 3];
      (*ud).c13.re+=eps*r[ 4];
      (*ud).c13.im+=eps*r[ 5];

      (*ud).c21.re+=eps*r[ 6];
      (*ud).c21.im+=eps*r[ 7];      
      (*ud).c22.re+=eps*r[ 8];
      (*ud).c22.im+=eps*r[ 9];
      (*ud).c23.re+=eps*r[10];
      (*ud).c23.im+=eps*r[11];

      (*ud).c31.re+=eps*r[12];
      (*ud).c31.im+=eps*r[13];      
      (*ud).c32.re+=eps*r[14];
      (*ud).c32.im+=eps*r[15];
      (*ud).c33.re+=eps*r[16];
      (*ud).c33.im+=eps*r[17];      
   }

   openbcd();
}


int main(int argc,char *argv[])
{
   int my_rank;
   double d1,d2,d3,d4,d5,p1,p2;
   su3_dble *udb,**usv;
   FILE *flog=NULL;
   double phi[7]={0.0,0.0,0.0,0.0,1.2,-3.1,1.7};

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   set_sf_parms(phi,phi+2,phi+4);

   if (my_rank==0)
   {
      flog=freopen("check2.log","w",stdout);

      printf("\n");
      printf("Renormalization of the link variables\n");
      printf("-------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }

   start_ranlux(0,123456);
   geometry();
   alloc_wud(1);
   usv=reserve_wud(1);
   udb=udfld();
   
   random_ud();
   check_ud(&d1,&d2);
   
   if (my_rank==0)
   {
      printf("Random double-precision gauge field:\n");
      printf("|u^dag*u-1| = %.2e\n",d1);
      printf("|det{u}-1| = %.2e\n\n",d2);      
   }

   cm3x3_assign(4*VOLUME,udb,usv[0]);   
   tilt_ud(50.0*DBL_EPSILON);
   check_ud(&d1,&d2);
   renormalize_ud();
   d3=cmp_ud(usv[0]);
   check_ud(&d4,&d5);   
   
   if (my_rank==0)
   {
      printf("Tilt double-precision gauge field:\n");
      printf("|u^dag*u-1| = %.2e\n",d1);
      printf("|det{u}-1| = %.2e\n",d2);
      printf("Difference after renormalization = %.2e\n",d3);            
      printf("|u^dag*u-1| = %.2e\n",d4);
      printf("|det{u}-1| = %.2e\n\n",d5);
   }

   random_ud();
   cm3x3_assign(4*VOLUME,udb,usv[0]); 
   renormalize_ud();
   d1=cmp_ud(usv[0]);
   
   if (my_rank==0)
   {
      printf("Renormalization of a fresh random double-precision field:\n");
      printf("Difference after renormalization = %.2e\n\n",d1);       
   }

   cm3x3_assign(4*VOLUME,udb,usv[0]); 
   mult_phase(+1);
   p1=plaq_wsum_dble(1);
   d1=cmp_ud(usv[0]);
   mult_phase(-1);
   d2=cmp_ud(usv[0]);
   p2=plaq_wsum_dble(1);

   if (my_rank==0)
   {
      printf("Rotation forth/back of double-precision field::\n");
      printf("Difference after one rotation  = %.2e %15.10e\n\n",d1,p1);       
      printf("Difference after two rotations = %.2e %15.10e\n\n",d2,p2);       
      fclose(flog);
   }
   
   MPI_Finalize();
   exit(0);
}
