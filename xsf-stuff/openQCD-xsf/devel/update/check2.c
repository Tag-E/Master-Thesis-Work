
/*******************************************************************************
*
* File check2.c
*
* Copyright (C) 2005, 2007, 2009, 2010,      Martin Luescher, Filippo Palombi
*               2011, 2012                   Stefan Schaefer
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Reversibility of the MD evolution
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
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "mdflds.h"
#include "archive.h"
#include "forces.h"
#include "dfl.h"
#include "update.h"
#include "global.h"

static int my_rank;


static void read_lat_parms(void)
{
   double beta,c0,kappa[3],csw[5];

   if (my_rank==0)
   {
      find_section("Lattice parameters");
      read_line("beta","%lf",&beta);
      read_line("c0","%lf",&c0);
      read_line("kappa_u","%lf",kappa);
      read_line("kappa_s","%lf",kappa+1);
      read_line("kappa_c","%lf",kappa+2);
      read_line("csw","%lf",csw);
      read_line("cG","%lf",csw+1);
      read_line("cF","%lf",csw+2);   
      read_line("dF","%lf",csw+3);   
      read_line("zF","%lf",csw+4);   
   }

   MPI_Bcast(&beta,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&c0,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(kappa,3,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(csw,5,MPI_DOUBLE,0,MPI_COMM_WORLD);
   
   set_lat_parms(beta,c0,kappa[0],kappa[1],kappa[2],
                 csw[0],csw[1],csw[2],csw[3],csw[4]);
}


static void read_sf_parms(void) 
{
   double phi[7];

   if (my_rank==0)
   {
      find_section("Boundary values");
      read_dprms("phi",2,phi);
      read_dprms("phi'",2,phi+2);
      read_dprms("theta",3,phi+4);
   }

   MPI_Bcast(phi,7,MPI_DOUBLE,0,MPI_COMM_WORLD);   

   set_sf_parms(phi,phi+2,phi+4);
}


static void read_hmc_parms(void)
{
   int nact,*iact;
   int npf,nmu,nlv;
   double tau,*mu;
   
   if (my_rank==0)
   {
      find_section("HMC parameters");
      nact=count_tokens("actions");
      read_line("npf","%d",&npf);
      nmu=count_tokens("mu");
      read_line("nlv","%d",&nlv);      
      read_line("tau","%lf",&tau);
   }

   MPI_Bcast(&nact,1,MPI_INT,0,MPI_COMM_WORLD);   
   MPI_Bcast(&npf,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmu,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nlv,1,MPI_INT,0,MPI_COMM_WORLD);      
   MPI_Bcast(&tau,1,MPI_DOUBLE,0,MPI_COMM_WORLD);   

   if (nact>0)
   {
      iact=malloc(nact*sizeof(*iact));
      error(iact==NULL,1,"read_hmc_parms [check2.c]",
            "Unable to allocate temporary array");
      if (my_rank==0)
         read_iprms("actions",nact,iact);
      MPI_Bcast(iact,nact,MPI_INT,0,MPI_COMM_WORLD);
   }
   else
      iact=NULL;

   if (nmu>0)
   {
      mu=malloc(nmu*sizeof(*mu));
      error(mu==NULL,1,"read_hmc_parms [check2.c]",
            "Unable to allocate temporary array");
      if (my_rank==0)
         read_dprms("mu",nmu,mu);
      MPI_Bcast(mu,nmu,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }
   else
      mu=NULL;

   set_hmc_parms(nact,iact,npf,nmu,mu,nlv,tau);
   
   if (nact>0)
      free(iact);
   if (nmu>0)
      free(mu);
}


static void read_integrator(void)
{
   int nlv,i,j,k,l;
   hmc_parms_t hmc;
   mdint_parms_t mdp;
   force_parms_t fp;
   rat_parms_t rp;

   hmc=hmc_parms();
   nlv=hmc.nlv;

   for (i=0;i<nlv;i++)
   {
      read_mdint_parms(i);
      mdp=mdint_parms(i);

      for (j=0;j<mdp.nfr;j++)
      {
         k=mdp.ifr[j];
         fp=force_parms(k);

         if (fp.force==FORCES)
            read_force_parms2(k);

         fp=force_parms(k);

         if ((fp.force==FRF_RAT)||(fp.force==FRF_RAT_SDET))
         {
            l=fp.irat[0];
            rp=rat_parms(l);

            if (rp.degree==0)
               read_rat_parms(l);
         }
      }
   }
}


static void read_actions(void)
{
   int i,k,l,nact,*iact;
   hmc_parms_t hmc;
   action_parms_t ap;
   rat_parms_t rp;

   hmc=hmc_parms();
   nact=hmc.nact;
   iact=hmc.iact;

   for (i=0;i<nact;i++)
   {
      k=iact[i];
      ap=action_parms(k);

      if (ap.action==ACTIONS)
         read_action_parms(k);

      ap=action_parms(k);

      if ((ap.action==ACF_RAT)||(ap.action==ACF_RAT_SDET))
      {
         l=ap.irat[0];
         rp=rat_parms(l);

         if (rp.degree==0)
            read_rat_parms(l);
      }
   }
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
   int ninv,nmr,ncy,nkv,nmx,nsm;
   double kappa,mu,res,resd,dtau;

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
      read_line("mu","%lf",&mu);
      read_line("ninv","%d",&ninv);     
      read_line("nmr","%d",&nmr);
      read_line("ncy","%d",&ncy);
      read_line("nkv","%d",&nkv);
      read_line("nmx","%d",&nmx);           
      read_line("res","%lf",&res);
   }

   MPI_Bcast(&kappa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&mu,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&ninv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmr,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&ncy,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nkv,1,MPI_INT,0,MPI_COMM_WORLD);   
   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   set_dfl_gen_parms(kappa,mu,ninv,nmr,ncy,nkv,nmx,res);
   
   if (my_rank==0)
   {
      find_section("Deflation projectors");
      read_line("nkv","%d",&nkv);
      read_line("nmx","%d",&nmx);           
      read_line("resd","%lf",&resd);
      read_line("res","%lf",&res);
   }

   MPI_Bcast(&nkv,1,MPI_INT,0,MPI_COMM_WORLD);   
   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&resd,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   set_dfl_pro_parms(nkv,nmx,resd,res);

   if (my_rank==0)
   {
      find_section("Deflation update scheme");
      read_line("dtau","%lf",&dtau);
      read_line("nsm","%d",&nsm);           
   }

   MPI_Bcast(&dtau,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&nsm,1,MPI_INT,0,MPI_COMM_WORLD);   
   set_dfl_upd_parms(dtau,nsm);
}


static void read_solvers(void)
{
   int nact,*iact,nlv,nsp;
   int nfr,*ifr;
   int isap,idfl,i,j,k;
   hmc_parms_t hmc;
   mdint_parms_t mdp;
   action_parms_t ap;
   force_parms_t fp;
   solver_parms_t sp;

   hmc=hmc_parms();
   nact=hmc.nact;
   iact=hmc.iact;
   nlv=hmc.nlv;
   isap=0;
   idfl=0;
   
   for (i=0;i<nact;i++)
   {
      ap=action_parms(iact[i]);

      if ((ap.action==ACF_TM1)||
          (ap.action==ACF_TM1_EO)||
          (ap.action==ACF_TM1_EO_SDET)||
          (ap.action==ACF_TM2)||
          (ap.action==ACF_TM2_EO)||
          (ap.action==ACF_RAT)||
          (ap.action==ACF_RAT_SDET))
      {
         if ((ap.action==ACF_TM2)||(ap.action==ACF_TM2_EO))
            nsp=2;
         else
            nsp=1;
         
         for (k=0;k<nsp;k++)
         {
            j=ap.isp[k];
            sp=solver_parms(j);

            if (sp.solver==SOLVERS)
            {
               read_solver_parms(j);
               sp=solver_parms(j);

               if (sp.solver==SAP_GCR)
                  isap=1;
               else if (sp.solver==DFL_SAP_GCR)
               {
                  isap=1;
                  idfl=1;
               }
            }
         }
      }
   }

   for (i=0;i<nlv;i++)
   {
      mdp=mdint_parms(i);
      nfr=mdp.nfr;
      ifr=mdp.ifr;

      for (j=0;j<nfr;j++)
      {
         fp=force_parms(ifr[j]);

         if ((fp.force==FRF_TM1)||
             (fp.force==FRF_TM1_EO)||
             (fp.force==FRF_TM1_EO_SDET)||
             (fp.force==FRF_TM2)||
             (fp.force==FRF_TM2_EO)||
             (fp.force==FRF_RAT)||
             (fp.force==FRF_RAT_SDET))
         {
            k=fp.isp[0];
            sp=solver_parms(k);

            if (sp.solver==SOLVERS)
            {
               read_solver_parms(k);
               sp=solver_parms(k);

               if (sp.solver==SAP_GCR)
                  isap=1;
               else if (sp.solver==DFL_SAP_GCR)
               {
                  isap=1;
                  idfl=1;
               }
            }
         }
      }
   }
      
   if (isap)
      read_sap_parms();

   if (idfl)
      read_dfl_parms();
}


static void chk_mode_regen(int isp,int *status)
{
   solver_parms_t sp;

   sp=solver_parms(isp);

   if ((sp.solver==DFL_SAP_GCR)&&(status[3]>0))
      add2counter("modes",2,status+3);
}


static void start_hmc(double *act0,su3_dble *uold)
{
   int i,n,nact,*iact;
   int status[4];
   double *mu;
   su3_dble *udb;
   dfl_parms_t dfl;
   hmc_parms_t hmc;
   action_parms_t ap;

   clear_counters();
   
   udb=udfld();
   cm3x3_assign(4*VOLUME,udb,uold);
   random_mom();

   dfl=dfl_parms();

   if (dfl.Ns)
   {
      dfl_modes(status);
      error_root(status[0]<0,1,"start_hmc [hmc.c]",
                 "Deflation subspace generation failed (status = %d)",
                 status[0]);
      add2counter("modes",0,status);
   }

   hmc=hmc_parms();
   nact=hmc.nact;
   iact=hmc.iact;
   mu=hmc.mu;
   n=2;
   
   for (i=0;i<nact;i++)
   {
      ap=action_parms(iact[i]);

      if (ap.action==ACG)
         act0[1]=action0(0);
      else
      {
         set_sw_parms(sea_quark_mass(ap.im0));
         
         if (ap.action==ACF_TM1)
            act0[n]=setpf1(mu[ap.imu[0]],ap.ipf,0);
         else if (ap.action==ACF_TM1_EO)
            act0[n]=setpf4(mu[ap.imu[0]],ap.ipf,0,0);
         else if (ap.action==ACF_TM1_EO_SDET)
            act0[n]=setpf4(mu[ap.imu[0]],ap.ipf,1,0);
         else if (ap.action==ACF_TM2)
         {
            status[3]=0;
            act0[n]=setpf2(mu[ap.imu[0]],mu[ap.imu[1]],ap.ipf,ap.isp[1],
                           0,status);
            chk_mode_regen(ap.isp[1],status);         
            add2counter("field",ap.ipf,status);
         }
         else if (ap.action==ACF_TM2_EO)
         {
            status[3]=0;
            act0[n]=setpf5(mu[ap.imu[0]],mu[ap.imu[1]],ap.ipf,ap.isp[1],
                           0,status);
            chk_mode_regen(ap.isp[1],status);         
            add2counter("field",ap.ipf,status);
         }
         else if (ap.action==ACF_RAT)
         {
            status[3]=0;
            act0[n]=setpf3(ap.irat,ap.ipf,0,ap.isp[0],0,status);
            chk_mode_regen(ap.isp[0],status);         
            add2counter("field",ap.ipf,status);
         }
         else if (ap.action==ACF_RAT_SDET)
         {
            status[3]=0;
            act0[n]=setpf3(ap.irat,ap.ipf,1,ap.isp[0],0,status);
            chk_mode_regen(ap.isp[0],status);         
            add2counter("field",ap.ipf,status);
         }
         else
            error_root(1,1,"start_hmc [check2.c]","Unknown action");

         n+=1;
      }
   }

   act0[0]=momentum_action(0);
}


static void end_hmc(double *act1)
{
   int i,n,nact,*iact;
   int status[4];
   double *mu;
   hmc_parms_t hmc;
   action_parms_t ap;   

   hmc=hmc_parms();
   nact=hmc.nact;
   iact=hmc.iact;
   mu=hmc.mu;
   n=2;

   for (i=0;i<nact;i++)
   {
      ap=action_parms(iact[i]);

      if (ap.action==ACG)
         act1[1]=action0(0);
      else
      {
         set_sw_parms(sea_quark_mass(ap.im0));
         status[3]=0;
            
         if (ap.action==ACF_TM1)
            act1[n]=action1(mu[ap.imu[0]],ap.ipf,ap.isp[0],0,status);
         else if (ap.action==ACF_TM1_EO)
            act1[n]=action4(mu[ap.imu[0]],ap.ipf,0,ap.isp[0],0,status);
         else if (ap.action==ACF_TM1_EO_SDET)
            act1[n]=action4(mu[ap.imu[0]],ap.ipf,1,ap.isp[0],0,status);
         else if (ap.action==ACF_TM2)
            act1[n]=action2(mu[ap.imu[0]],mu[ap.imu[1]],ap.ipf,ap.isp[0],
                            0,status);
         else if (ap.action==ACF_TM2_EO)
            act1[n]=action5(mu[ap.imu[0]],mu[ap.imu[1]],ap.ipf,ap.isp[0],
                            0,status);
         else if (ap.action==ACF_RAT)
            act1[n]=action3(ap.irat,ap.ipf,0,ap.isp[0],0,status);
         else if (ap.action==ACF_RAT_SDET)
            act1[n]=action3(ap.irat,ap.ipf,1,ap.isp[0],0,status);
 
         chk_mode_regen(ap.isp[0],status);
         add2counter("action",iact[i],status);
         n+=1;      
      }
   }

   act1[0]=momentum_action(0);
}


static void flip_mom(void)
{
   int status;
   su3_alg_dble *mom,*momm;
   mdflds_t *mdfs;
   dfl_parms_t dfl;
   dfl_upd_parms_t dup;

   mdfs=mdflds();
   mom=(*mdfs).mom;
   momm=mom+4*VOLUME;

   for (;mom<momm;mom++)
   {
      (*mom).c1=-(*mom).c1;
      (*mom).c2=-(*mom).c2;
      (*mom).c3=-(*mom).c3;
      (*mom).c4=-(*mom).c4;
      (*mom).c5=-(*mom).c5;
      (*mom).c6=-(*mom).c6;
      (*mom).c7=-(*mom).c7;
      (*mom).c8=-(*mom).c8;         
   }

   dfl=dfl_parms();

   if (dfl.Ns)
   {
      dup=dfl_upd_parms();      
      dfl_update(dup.nsm,&status);
      error_root(status<0,1,"flip_mom [check2.c]",
                 "Deflation subspace update failed (status = %d)",status);
      add2counter("modes",1,&status);
   }
}


static double cmp_ud(su3_dble *u,su3_dble *v)
{
   int i;
   double r[18],dev,dmax;

   r[ 0]=(*u).c11.re-(*v).c11.re;
   r[ 1]=(*u).c11.im-(*v).c11.im;
   r[ 2]=(*u).c12.re-(*v).c12.re;
   r[ 3]=(*u).c12.im-(*v).c12.im;
   r[ 4]=(*u).c13.re-(*v).c13.re;
   r[ 5]=(*u).c13.im-(*v).c13.im;

   r[ 6]=(*u).c21.re-(*v).c21.re;
   r[ 7]=(*u).c21.im-(*v).c21.im;
   r[ 8]=(*u).c22.re-(*v).c22.re;
   r[ 9]=(*u).c22.im-(*v).c22.im;
   r[10]=(*u).c23.re-(*v).c23.re;
   r[11]=(*u).c23.im-(*v).c23.im;

   r[12]=(*u).c31.re-(*v).c31.re;
   r[13]=(*u).c31.im-(*v).c31.im;
   r[14]=(*u).c32.re-(*v).c32.re;
   r[15]=(*u).c32.im-(*v).c32.im;
   r[16]=(*u).c33.re-(*v).c33.re;
   r[17]=(*u).c33.im-(*v).c33.im;   

   dmax=0.0;
   
   for (i=0;i<18;i+=2)
   {
      dev=r[i]*r[i]+r[i+1]*r[i+1];
      if (dev>dmax)
         dmax=dev;
   }

   return dmax;
}


static double max_dev_ud(su3_dble *v)
{
   double d,dmax;
   su3_dble *u,*um;

   u=udfld();
   um=u+4*VOLUME;
   dmax=0.0;
   
   for (;u<um;u++)
   {
      d=cmp_ud(u,v);

      if (d>dmax)
         dmax=d;

      v+=1;
   }

   if (NPROC>1)
   {
      d=dmax;
      MPI_Reduce(&d,&dmax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Bcast(&dmax,1,MPI_DOUBLE,0,MPI_COMM_WORLD);   
   }
   
   return sqrt(dmax);
}


int main(int argc,char *argv[])
{
   int i,ie;
   int first,last,step;
   int nc,nsize,icnfg,nact;
   int isap,idfl;
   int nwud,nws,nwsd,nwv,nwvd;
   double *act0,*act1,*act2;
   double sm0[2],sm1[2],dud,dH;
   double dudmin,dudmax,dudavg,dHmin,dHmax,dHavg;
   su3_dble **usv;
   hmc_parms_t hmc;
   char cnfg_dir[NAME_SIZE],cnfg_file[NAME_SIZE];
   char nbase[NAME_SIZE];   
   FILE *flog=NULL,*fin=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check2.log","w",stdout);
      fin=freopen("check2.in","r",stdin);

      printf("\n");
      printf("Reversibility of the MD evolution\n");
      printf("---------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      find_section("Configurations");
      read_line("cnfg_dir","%s",cnfg_dir);
      read_line("name","%s",nbase);
      read_line("first","%d",&first);
      read_line("last","%d",&last);  
      read_line("step","%d",&step);  
   }
   
   MPI_Bcast(cnfg_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);

   read_lat_parms();
   read_sf_parms();
   read_hmc_parms();
   read_actions();
   read_integrator();
   read_solvers();

   if (my_rank==0)
      fclose(fin);
   
   hmc_wsize(&nwud,&nws,&nwsd,&nwv,&nwvd);
   alloc_wud(nwud);
   alloc_ws(nws);
   alloc_wsd(nwsd);
   alloc_wv(nwv);
   alloc_wvd(nwvd);   
   usv=reserve_wud(1);

   hmc=hmc_parms();   
   nact=hmc.nact;
   act0=malloc(3*(nact+1)*sizeof(*act0));
   act1=act0+nact+1;
   act2=act1+nact+1;
   error(act0==NULL,1,"main [check2.c]","Unable to allocate action arrays");


   print_lat_parms();
   print_sf_parms();
   print_hmc_parms();
   print_action_parms();
   print_rat_parms();
   print_mdint_parms();
   print_force_parms2();
   print_solver_parms(&isap,&idfl);
   if (isap)
      print_sap_parms(0);
   if (idfl)
      print_dfl_parms(1);
   
   if (my_rank==0)
   {
      printf("Configurations %sn%d -> %sn%d in steps of %d\n\n",
             nbase,first,nbase,last,step);      
      fflush(flog);
   }

   start_ranlux(0,1234);
   geometry();
   
   error_root(((last-first)%step)!=0,1,"main [check2.c]",
              "last-first is not a multiple of step");
   check_dir_root(cnfg_dir);   

   nsize=name_size("%s/%sn%d",cnfg_dir,nbase,last);
   error_root(nsize>=NAME_SIZE,1,"main [check2.c]",
              "Configuration file name is too long");

   hmc_sanity_check();
   set_mdsteps();
   setup_counters();
   setup_chrono();
   
   dudmin=0.0;
   dudmax=0.0;
   dudavg=0.0;
   dHmin=0.0;
   dHmax=0.0;
   dHavg=0.0;
   
   for (icnfg=first;icnfg<=last;icnfg+=step)
   {
      sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,icnfg);
      import_cnfg(cnfg_file);

      if (my_rank==0)
      {
         printf("Configuration no %d\n",icnfg);
         fflush(flog);
      } 

      ie=check_sfbcd();
      error_root(ie!=1,1,"main [check2.c]",
                 "Initial configuration has incorrect boundary values");
      
      mult_phase(1);

      start_hmc(act0,usv[0]);
      dud=max_dev_ud(usv[0]);
      run_mdint();
      end_hmc(act1);

      sm0[0]=0.0;
      sm0[1]=0.0;
      
      for (i=0;i<=nact;i++)
      {
         sm0[0]+=act0[i];
         sm0[1]+=(act1[i]-act0[i]);
      }

      MPI_Reduce(sm0,sm1,2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);      
      MPI_Bcast(sm1,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
      
      if (my_rank==0)
      {
         printf("start_hmc:\n");
         printf("max|U_ij-U'_ij| = %.1e\n",dud);      
         printf("run_mdint:\n");
         printf("H = %.6e\n",sm1[0]);         
         printf("dH = %.2e\n",sm1[1]);      
         fflush(flog);
      }

      print_all_avgstat();
      
      flip_mom();
      run_mdint();
      end_hmc(act2);

      sm0[0]=0.0;
      sm0[1]=0.0;
      
      for (i=0;i<=nact;i++)
      {
         sm0[0]+=act2[i];
         sm0[1]+=(act2[i]-act0[i]);
      }

      MPI_Reduce(sm0,sm1,2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);      
      MPI_Bcast(sm1,2,MPI_DOUBLE,0,MPI_COMM_WORLD);

      dH=fabs(sm1[1]);
      dud=max_dev_ud(usv[0]);
      error_chk();
      
      if (my_rank==0)
      {
         printf("Flip momenta and run_mdint:\n");
         printf("H  = %.6e\n",sm1[0]);
         printf("|dH| = % .2e\n",dH);
         printf("max|U_ij-U'_ij| = %.2e\n\n",dud);
         fflush(flog);
      }
      
      if (icnfg==first)
      {
         dudmin=dud;
         dudmax=dud;
         dudavg=dud;         

         dHmin=dH;
         dHmax=dH;
         dHavg=dH;
      }
      else
      {
         if (dud<dudmin)
            dudmin=dud;
         if (dud>dudmax)
            dudmax=dud;
         dudavg+=dud;

         if (dH<dHmin)
            dHmin=dH;
         if (dH>dHmax)
            dHmax=dH;
         dHavg+=dH;
      }
   }

   if (my_rank==0)
   {
      nc=(last-first)/step+1;
      
      printf("Test summary\n");
      printf("------------\n\n");

      printf("Considered %d configurations in the range %d -> %d\n\n",
             nc,first,last);

      printf("The three figures quoted in each case are the minimal,\n");
      printf("maximal and average values\n\n");

      printf("max|U_ij-U'_ij| = %.2e, %.2e, %.2e\n",
             dudmin,dudmax,dudavg/(double)(nc));
      printf("|dH|            = %.2e, %.2e, %.2e\n\n",
             dHmin,dHmax,dHavg/(double)(nc));

      fclose(flog);
   }
   
   MPI_Finalize();    
   exit(0);
}
