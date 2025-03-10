
/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2005, 2008, 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Hermiticity of Dw() and comparison with Dwee(),...
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "sflds.h"
#include "linalg.h"
#include "sw_term.h"
#include "dirac.h"
#include "global.h"


int main(int argc,char *argv[])
{
   int my_rank,i;
   float mu,d;
   complex z1,z2;
   spinor **ps;
   sw_parms_t swp;
   FILE *flog=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check3.log","w",stdout);
      printf("\n");
      printf("Hermiticity of Dw() and comparison with Dwee(),...\n");
      printf("--------------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }

   start_ranlux(0,12345);
   geometry();
   alloc_ws(5);
   ps=reserve_ws(5);

   set_lat_parms(5.5,1.0,0.0,0.0,0.0,0.456,1.0,1.234,0.5,1.0);
   swp=set_sw_parms(-0.0123);
   mu=0.0376;

   if (my_rank==0)
   {
      printf("m0 = %.4e, mu= %.4e, csw = %.4e, cF = %.4e\n\n",
             swp.m0,mu,swp.csw,swp.cF);
      printf("Deviations should be at most 10^(-5) or so in these tests\n\n");
   }

   random_ud();
   sw_term(NO_PTS);
   assign_ud2u();
   assign_swd2sw();

   for (i=0;i<4;i++)
      random_s(NSPIN,ps[i],1.0f);

   Dw(mu,ps[0],ps[2]);
   mulg5(VOLUME,ps[2]);
   Dw(-mu,ps[1],ps[3]);
   mulg5(VOLUME,ps[3]);

   z1=spinor_prod(VOLUME,1,ps[0],ps[3]);
   z2=spinor_prod(VOLUME,1,ps[2],ps[1]);

   d=(float)(sqrt((double)((z1.re-z2.re)*(z1.re-z2.re)+
                           (z1.im-z2.im)*(z1.im-z2.im))));
   d/=(float)(sqrt((double)(12*NPROC)*(double)(VOLUME)));
   error_chk();

   if (my_rank==0)
      printf("Deviation from gamma5-Hermiticity    = %.1e\n",d);

   for (i=0;i<4;i++)
      random_s(NSPIN,ps[i],1.0f);

   assign_s2s(VOLUME,ps[0],ps[1]);
   assign_s2s(VOLUME,ps[2],ps[3]);   
   Dwee(mu,ps[1],ps[2]);

   bnd_s2zero(EVEN_PTS,ps[0]);
   mulr_spinor_add(VOLUME,ps[1],ps[0],-1.0f);   
   d=norm_square(VOLUME,1,ps[1]);

   error(d!=0.0f,1,"main [check3.c]",
         "Dwee() changes the input field in unexpected ways");

   mulr_spinor_add(VOLUME/2,ps[2]+(VOLUME/2),ps[3]+(VOLUME/2),-1.0f);   
   assign_s2s(VOLUME/2,ps[2],ps[4]);
   bnd_s2zero(EVEN_PTS,ps[4]);   
   mulr_spinor_add(VOLUME/2,ps[2],ps[4],-1.0f);   
   d=norm_square(VOLUME,1,ps[2]);
   
   error(d!=0.0f,1,"main [check3.c]",
         "Dwee() changes the output field where it should not");

   for (i=0;i<4;i++)
      random_s(NSPIN,ps[i],1.0f);

   assign_s2s(VOLUME,ps[0],ps[1]);
   assign_s2s(VOLUME,ps[2],ps[3]);   
   Dwoo(mu,ps[1],ps[2]);

   bnd_s2zero(ODD_PTS,ps[0]);
   mulr_spinor_add(VOLUME,ps[1],ps[0],-1.0f);   
   d=norm_square(VOLUME,1,ps[1]);

   error(d!=0.0f,1,"main [check3.c]",
         "Dwoo() changes the input field in unexpected ways");

   mulr_spinor_add(VOLUME/2,ps[2],ps[3],-1.0f);   
   assign_s2s(VOLUME/2,ps[2]+(VOLUME/2),ps[4]+(VOLUME/2));
   bnd_s2zero(ODD_PTS,ps[4]);   
   mulr_spinor_add(VOLUME/2,ps[2]+(VOLUME/2),ps[4]+(VOLUME/2),-1.0f);   
   d=norm_square(VOLUME,1,ps[2]);
   
   error(d!=0.0f,1,"main [check3.c]",
         "Dwoo() changes the output field where it should not");

   for (i=0;i<4;i++)
      random_s(NSPIN,ps[i],1.0f);

   assign_s2s(VOLUME,ps[0],ps[1]);
   assign_s2s(VOLUME,ps[2],ps[3]);   
   Dwoe(ps[1],ps[2]);

   bnd_s2zero(EVEN_PTS,ps[0]);
   mulr_spinor_add(VOLUME,ps[1],ps[0],-1.0f);   
   d=norm_square(VOLUME,1,ps[1]);

   error(d!=0.0f,1,"main [check3.c]",
         "Dwoe() changes the input field in unexpected ways");

   mulr_spinor_add(VOLUME/2,ps[2],ps[3],-1.0f);   
   assign_s2s(VOLUME/2,ps[2]+(VOLUME/2),ps[4]+(VOLUME/2));
   bnd_s2zero(ODD_PTS,ps[4]);   
   mulr_spinor_add(VOLUME/2,ps[2]+(VOLUME/2),ps[4]+(VOLUME/2),-1.0f);   
   d=norm_square(VOLUME,1,ps[2]);
   
   error(d!=0.0f,1,"main [check3.c]",
         "Dwoe() changes the output field where it should not");

   for (i=0;i<4;i++)
      random_s(NSPIN,ps[i],1.0f);

   assign_s2s(VOLUME,ps[0],ps[1]);
   assign_s2s(VOLUME,ps[2],ps[3]);   
   Dweo(ps[1],ps[2]);

   bnd_s2zero(ODD_PTS,ps[0]);
   mulr_spinor_add(VOLUME,ps[1],ps[0],-1.0f);   
   d=norm_square(VOLUME,1,ps[1]);

   error(d!=0.0f,1,"main [check3.c]",
         "Dweo() changes the input field in unexpected ways");

   mulr_spinor_add(VOLUME/2,ps[2]+(VOLUME/2),ps[3]+(VOLUME/2),-1.0f);   
   assign_s2s(VOLUME/2,ps[2],ps[4]);
   bnd_s2zero(EVEN_PTS,ps[4]);   
   mulr_spinor_add(VOLUME/2,ps[2],ps[4],-1.0f);   
   d=norm_square(VOLUME,1,ps[2]);
   
   error(d!=0.0f,1,"main [check3.c]",
         "Dweo() changes the output field where it should not");
   
   for (i=0;i<4;i++)
      random_s(NSPIN,ps[i],1.0f);

   assign_s2s(VOLUME,ps[0],ps[1]);
   assign_s2s(VOLUME,ps[2],ps[3]);   
   Dwhat(mu,ps[1],ps[2]);

   bnd_s2zero(EVEN_PTS,ps[0]);
   mulr_spinor_add(VOLUME,ps[1],ps[0],-1.0f);   
   d=norm_square(VOLUME,1,ps[1]);

   error(d!=0.0f,1,"main [check3.c]",
         "Dwhat() changes the input field in unexpected ways");

   mulr_spinor_add(VOLUME/2,ps[2]+(VOLUME/2),ps[3]+(VOLUME/2),-1.0f);   
   assign_s2s(VOLUME/2,ps[2],ps[4]);
   bnd_s2zero(EVEN_PTS,ps[4]);   
   mulr_spinor_add(VOLUME/2,ps[2],ps[4],-1.0f);   
   d=norm_square(VOLUME,1,ps[2]);
   
   error(d!=0.0f,1,"main [check3.c]",
         "Dwhat() changes the output field where it should not");

   for (i=0;i<4;i++)
      random_s(NSPIN,ps[i],1.0f);

   assign_s2s(VOLUME,ps[0],ps[2]);
   Dw(mu,ps[0],ps[1]);
   Dwee(mu,ps[2],ps[3]);
   set_s2zero(VOLUME/2,ps[0]);
   mulr_spinor_add(VOLUME/2,ps[0],ps[3],-1.0f);    
   Dweo(ps[2],ps[0]);
   set_s2zero(VOLUME/2,ps[3]);
   mulr_spinor_add(VOLUME/2,ps[3],ps[0],-1.0f);

   Dwoo(mu,ps[2],ps[3]);   
   Dwoe(ps[2],ps[4]);
   mulr_spinor_add(VOLUME/2,ps[3]+(VOLUME/2),ps[4]+(VOLUME/2),1.0f);      
   mulr_spinor_add(VOLUME,ps[3],ps[1],-1.0f);   
   d=norm_square(VOLUME,1,ps[3])/norm_square(VOLUME,1,ps[1]);   
   d=(float)(sqrt((double)(d)));
   
   if (my_rank==0)
      printf("Deviation of Dw() from Dwee(),..     = %.1e\n",d);

   for (i=0;i<4;i++)
      random_s(NSPIN,ps[i],1.0f);

   assign_s2s(NSPIN,ps[0],ps[1]);
   Dwhat(mu,ps[0],ps[2]);

   Dwoe(ps[1],ps[1]);
   Dwee(mu,ps[1],ps[1]);   
   Dwoo(0.0,ps[1],ps[1]);
   Dweo(ps[1],ps[1]);
   
   mulr_spinor_add(VOLUME/2,ps[1],ps[2],-1.0f);
   d=norm_square(VOLUME/2,1,ps[1])/norm_square(VOLUME/2,1,ps[2]);
   d=(float)(sqrt((double)(d)));

   if (my_rank==0)
      printf("Deviation of Dwhat() from Dwee(),..  = %.1e\n",d);

   for (i=0;i<4;i++)
      random_s(NSPIN,ps[i],1.0f);

   assign_s2s(VOLUME,ps[0],ps[2]);

   set_tm_parms(1);
   Dw(mu,ps[0],ps[1]);
   set_tm_parms(0);
   
   Dwee(mu,ps[2],ps[3]);
   mulr_spinor_add(VOLUME/2,ps[1],ps[3],-1.0f);    
   Dweo(ps[2],ps[1]);
   Dwoe(ps[2],ps[3]);
   mulr_spinor_add(VOLUME/2,ps[1]+(VOLUME/2),ps[3]+(VOLUME/2),-1.0f);
   Dwoo(0.0f,ps[2],ps[3]);   
   mulr_spinor_add(VOLUME/2,ps[1]+(VOLUME/2),ps[3]+(VOLUME/2),-1.0f);
   d=norm_square(VOLUME,1,ps[1])/norm_square(VOLUME,1,ps[2]);   
   d=(float)(sqrt((double)(d)));
   
   error_chk();

   if (my_rank==0)
   {
      printf("Check of Dw()|eoflg=1                = %.1e\n\n",d);
      fclose(flog);
   }
   
   MPI_Finalize();
   exit(0);
}
