
/*******************************************************************************
*
* File hmc.c
*
* Copyright (C) 2005, 2007, 2009, 2010,  Martin Luescher, Filippo Palombi,
*               2011, 2012               Stefan Schaefer
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* HMC simulation algorithm
*
* The externally accessible functions are
*
*   void hmc_sanity_check(void)
*     Performs various checks on the chosen parameters for the HMC
*     algorithm and terminates with an error message if an inconsistency
*     is discovered.
* 
*   void hmc_wsize(int *nwud,int *nws,int *nwsd,int *nwv,int *nwvd)
*     Determines the minimal sizes of the workspaces required for the
*     HMC algorithm based on the information in the parameter data base.
*     On exit the program returns the numbers of double-precision gauge
*     (nwud), spinor (nwsd) and complex vector (nwvd) fields as well as
*     the numbers of single-precision spinor (nws) and complex vector
*     (nwv) fields that must be allocated.
*
*   int run_hmc(double *act0,double *act1)
*     Generates a random momentum field, integrates the MD equations and
*     applies the HMC acceptance step to the fields at the end of the MD
*     trajectory (see the notes).
*      The arrays act0 and act1 must have at least nact+1 elements, where
*     nact is the number of actions that take part in the HMC algorithm
*     (see flags/hmc_parms.c). On exit act0 and act1 contain the part of
*     the actions computed on the local lattice at the beginning and the
*     end of the MD evolution (see the notes).
*      The program returns 1 or 0 depending on whether the field generated
*     by the molecular-dynamics evolution was accepted or not. If it was
*     not accepted, the gauge field is restored to its initial value.
*
* Notes:
*
* The molecular-dynamics equations are integrated using the integrator
* specified by the list of elementary operations returned by mdsteps()
* (see update/mdsteps.c and update/mdint.c). The elements of the action
* arrays act0 and act1 are
*
*  act*[0]        Action of the momentum field
*  act*[1]        Gauge field action
*  act*[2+n]      Pseudo-fermion action number n
*
* where the pseudo-fermion actions are counted from 0 by 1 as they appear
* the action array hmc.iact returned by hmc_parms().
*
* Open boundary conditions are imposed without further notice, i.e. the
* link variables and momenta on the time-like links at time NPROC0*L0-1
* are set to zero. On the other links, the program run_hmc() renormalizes
* the gauge field to SU(3) at the end of the molecular-dynamics evolution
* (if the new configuration is accepted).
*
* The program in this module performs global communications and must be
* called simultaneously on all MPI processes.
*
*******************************************************************************/

#define HMC_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "uflds.h"
#include "mdflds.h"
#include "linalg.h"
#include "dfl.h"
#include "forces.h"
#include "update.h"
#include "lattice.h"
#include "global.h"

#define MAX(n,m) \
   if ((n)<(m)) \
      (n)=(m)

static int nrs=0,*rs;


static void init_rs(int nr)
{
   int k;
   
   if (nr>nrs)
   {
      if (nrs>0)
         free(rs);

      rs=malloc(nr*sizeof(*rs));
      error_root(rs==NULL,1,"init_rs [hmc.c]",
                 "Unable to allocate auxiliary array");
      nrs=nr;
   }

   for (k=0;k<nr;k++)
      rs[k]=0;
}


static int check_rat_actions(void)
{
   int k,l,j,ie;
   int nact,*iact,ir,nr,im0,isw;
   hmc_parms_t hmc;
   action_parms_t ap;
   rat_parms_t rp;

   hmc=hmc_parms();
   nact=hmc.nact;
   iact=hmc.iact;
   ie=0;      

   for (k=0;k<nact;k++)
   {
      ap=action_parms(iact[k]);

      if ((ap.action==ACF_RAT)||(ap.action==ACF_RAT_SDET))
      {
         ir=ap.irat[0];
         im0=ap.im0;
         rp=rat_parms(ir);
         nr=rp.degree;
         init_rs(nr);
         isw=0;

         for (l=0;l<nact;l++)
         {
            ap=action_parms(iact[l]);

            if ((ap.action==ACF_RAT)||(ap.action==ACF_RAT_SDET))
            {
               if ((ap.irat[0]==ir)&&(ap.im0==im0))
               {
                  if (ap.action==ACF_RAT_SDET)
                     isw+=1;

                  for (j=ap.irat[1];j<=ap.irat[2];j++)
                     rs[j]+=1;
               }
            }
         }

         for (l=0;l<nr;l++)
            ie|=(rs[l]!=isw);            
      }
   }

   return ie;
}
   

static int match_force(action_parms_t ap,force_parms_t fp)
{
   int ie;

   ie=1;
   
   if (ap.action==ACG)
      ie&=(fp.force==FRG);
   else if (ap.action==ACF_TM1)
   {
      ie&=(fp.force==FRF_TM1);
      ie&=(ap.ipf==fp.ipf);
      ie&=(ap.im0==fp.im0);
      ie&=(ap.imu[0]==fp.imu[0]);         
   }
   else if (ap.action==ACF_TM1_EO)
   {
      ie&=(fp.force==FRF_TM1_EO);
      ie&=(ap.ipf==fp.ipf);
      ie&=(ap.im0==fp.im0);
      ie&=(ap.imu[0]==fp.imu[0]);         
   }
   else if (ap.action==ACF_TM1_EO_SDET)
   {
      ie&=(fp.force==FRF_TM1_EO_SDET);
      ie&=(ap.ipf==fp.ipf);
      ie&=(ap.im0==fp.im0);
      ie&=(ap.imu[0]==fp.imu[0]);         
   }
   else if (ap.action==ACF_TM2)
   {
      ie&=(fp.force==FRF_TM2);
      ie&=(ap.ipf==fp.ipf);
      ie&=(ap.im0==fp.im0);
      ie&=(ap.imu[0]==fp.imu[0]);         
      ie&=(ap.imu[1]==fp.imu[1]);         
   }
   else if (ap.action==ACF_TM2_EO)
   {
      ie&=(fp.force==FRF_TM2_EO);
      ie&=(ap.ipf==fp.ipf);
      ie&=(ap.im0==fp.im0);
      ie&=(ap.imu[0]==fp.imu[0]);         
      ie&=(ap.imu[1]==fp.imu[1]);         
   }
   else if (ap.action==ACF_RAT)
   {
      ie&=(fp.force==FRF_RAT);
      ie&=(ap.ipf==fp.ipf);
      ie&=(ap.im0==fp.im0);
      ie&=(ap.irat[0]==fp.irat[0]);         
      ie&=(ap.irat[1]==fp.irat[1]);         
      ie&=(ap.irat[2]==fp.irat[2]);         
   }
   else if (ap.action==ACF_RAT_SDET)
   {
      ie&=(fp.force==FRF_RAT_SDET);
      ie&=(ap.ipf==fp.ipf);
      ie&=(ap.im0==fp.im0);
      ie&=(ap.irat[0]==fp.irat[0]);         
      ie&=(ap.irat[1]==fp.irat[1]);         
      ie&=(ap.irat[2]==fp.irat[2]);         
   }   
   else
      ie=0;

   return ie;
}


void hmc_sanity_check(void)
{
   int my_rank;
   int nlv,nact,*iact,npf,nmu;
   int iepf,iemu,iem0,ierat,iacg;
   int ie,ic;
   int nfr,*ifr,i,j,k;
   hmc_parms_t hmc;
   mdint_parms_t mdp;
   action_parms_t ap;
   force_parms_t fp;
   rat_parms_t rp;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      hmc=hmc_parms();
      nlv=hmc.nlv;
      nact=hmc.nact;
      iact=hmc.iact;
      npf=hmc.npf;
      nmu=hmc.nmu;

      error_root((nlv<1)||(nact<1),1,"hmc_sanity_check [hmc.c]",
                 "hmc.nlv or hmc.nact must be at least 1");

      iepf=0;
      iemu=0;
      iem0=0;
      ierat=0;
      iacg=0;
   
      for (i=0;i<nact;i++)
      {
         ap=action_parms(iact[i]);

         if (ap.action==ACG)
            iacg+=1;
         else if ((ap.action==ACF_TM1)||
                  (ap.action==ACF_TM1_EO)||
                  (ap.action==ACF_TM1_EO_SDET)||                  
                  (ap.action==ACF_TM2)||
                  (ap.action==ACF_TM2_EO))
         {
            iepf|=(ap.ipf<0);
            iepf|=(ap.ipf>=npf);
            iemu|=(ap.imu[0]<0);
            iemu|=(ap.imu[0]>=nmu);
            iem0|=(sea_quark_mass(ap.im0)==DBL_MAX);
         
            if ((ap.action==ACF_TM2)||
                (ap.action==ACF_TM2_EO))
            {
               iemu|=(ap.imu[1]<0);
               iemu|=(ap.imu[1]>=nmu);
            }
         }
         else if ((ap.action==ACF_RAT)||
                  (ap.action==ACF_RAT_SDET))
         {
            iepf|=(ap.ipf<0);
            iepf|=(ap.ipf>=npf);
            iem0|=(sea_quark_mass(ap.im0)==DBL_MAX);

            rp=rat_parms(ap.irat[0]);
            ierat|=(ap.irat[2]>=rp.degree);            
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
                (fp.force==FRF_TM2_EO))
            {
               iepf|=(fp.ipf<0);
               iepf|=(fp.ipf>=npf);
               iemu|=(fp.imu[0]<0);
               iemu|=(fp.imu[0]>=nmu);
               iem0|=(sea_quark_mass(fp.im0)==DBL_MAX);
         
               if ((fp.force==FRF_TM2)||
                   (fp.force==FRF_TM2_EO))
               {
                  iemu|=(fp.imu[1]<0);
                  iemu|=(fp.imu[1]>=nmu);
               }
            }
            else if ((fp.force==FRF_RAT)||
                     (fp.force==FRF_RAT_SDET))
            {
               iepf|=(fp.ipf<0);
               iepf|=(fp.ipf>=npf);
               iem0|=(sea_quark_mass(fp.im0)==DBL_MAX);

               rp=rat_parms(fp.irat[0]);
               ierat|=(fp.irat[2]>=rp.degree);
            }            
         }
      }

      error_root(iepf!=0,1,"hmc_sanity_check [hmc.c]",
                 "Some pseudo-fermion indices are out of range");
      error_root(iemu!=0,1,"hmc_sanity_check [hmc.c]",
                 "Some twisted-mass indices are out of range");
      error_root(iem0!=0,1,"hmc_sanity_check [hmc.c]",
                 "Some sea-quark mass indices are out of range");
      error_root(ierat!=0,1,"hmc_sanity_check [hmc.c]",
                 "Some rational functions are not or not correctly specified");
      error_root(iacg!=1,1,"hmc_sanity_check [hmc.c]",
                 "Gauge action is missing or occurs several times");   

      ie=0;

      for (k=0;k<nact;k++)
      {
         ap=action_parms(iact[k]);
         ic=0;
   
         for (i=0;i<nlv;i++)
         {
            mdp=mdint_parms(i);
            nfr=mdp.nfr;
            ifr=mdp.ifr;

            for (j=0;j<nfr;j++)
            {
               fp=force_parms(ifr[j]);
               ic+=match_force(ap,fp);
            }
         }

         ie|=(ic!=1);
      }

      for (i=0;i<nlv;i++)
      {
         mdp=mdint_parms(i);
         nfr=mdp.nfr;
         ifr=mdp.ifr;

         for (j=0;j<nfr;j++)
         {
            fp=force_parms(ifr[j]);
            ic=0;

            for (k=0;k<nact;k++)
            {
               ap=action_parms(iact[k]);
               ic+=match_force(ap,fp);
            }

            ie|=(ic!=1);
         }
      }
      
      error_root(ie!=0,1,"hmc_sanity_check [hmc.c]",
                 "Specified actions and forces do not match");

      ie=check_rat_actions();
      error_root(ie!=0,1,"hmc_sanity_check [hmc.c]",
                 "Inconsistent rational function actions");
   }

   error_chk();
}


static void dfl_wsize(int *nws,int *nwv,int *nwvd)
{
   dfl_parms_t dp;
   dfl_pro_parms_t dpp;
   dfl_gen_parms_t dgp;

   dp=dfl_parms();
   dpp=dfl_pro_parms();
   dgp=dfl_gen_parms();

   MAX(*nws,dp.Ns+2);
   MAX(*nwv,2*dpp.nkv+2);
   MAX(*nwv,2*dgp.nkv+2);
   MAX(*nwvd,4);
}


static void solver_wsize(int isp,int nsd,int np,
                         int *nws,int *nwsd,int *nwv,int *nwvd)
{
   solver_parms_t sp;

   sp=solver_parms(isp);

   if (sp.solver==CGNE)
   {
      MAX(*nws,5);
      MAX(*nwsd,nsd+5);
   }
   else if (sp.solver==MSCG)
   {
      if (np>1)
      {
         MAX(*nwsd,nsd+np+3);
      }
      else
      {
         MAX(*nwsd,nsd+5);
      }
   }   
   else if (sp.solver==SAP_GCR)
   {
      MAX(*nws,2*sp.nkv+1);
      MAX(*nwsd,nsd+2);
   }
   else if (sp.solver==DFL_SAP_GCR)
   {
      MAX(*nws,2*sp.nkv+1);      
      MAX(*nwsd,nsd+4);
      dfl_wsize(nws,nwv,nwvd);
   }         
}


void hmc_wsize(int *nwud,int *nws,int *nwsd,int *nwv,int *nwvd)
{
   int nlv,nact,*iact;
   int nfr,*ifr,nsd,np,i,j;
   hmc_parms_t hmc;
   mdint_parms_t mdp;
   action_parms_t ap;
   force_parms_t fp;
   solver_parms_t sp;

   (*nwud)=1;
   (*nws)=0;
   (*nwsd)=0;
   (*nwv)=0;
   (*nwvd)=0;

   hmc=hmc_parms();
   nlv=hmc.nlv;
   nact=hmc.nact;
   iact=hmc.iact;

   for (i=0;i<nact;i++)
   {
      ap=action_parms(iact[i]);

      if ((ap.action==ACF_TM1)||
          (ap.action==ACF_TM1_EO)||
          (ap.action==ACF_TM1_EO_SDET)||
          (ap.action==ACF_TM2)||
          (ap.action==ACF_TM2_EO))
      {
         nsd=1;
         solver_wsize(ap.isp[0],nsd,0,nws,nwsd,nwv,nwvd);

         if ((ap.action==ACF_TM2)||
             (ap.action==ACF_TM2_EO))
            solver_wsize(ap.isp[1],nsd,0,nws,nwsd,nwv,nwvd);
      }
      else if ((ap.action==ACF_RAT)||
               (ap.action==ACF_RAT_SDET))
      {
         np=ap.irat[2]-ap.irat[1]+1;
         sp=solver_parms(ap.isp[0]);

         if (sp.solver==MSCG)
            nsd=np;
         else
            nsd=2;

         solver_wsize(ap.isp[0],nsd,np,nws,nwsd,nwv,nwvd);         
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
             (fp.force==FRF_TM2_EO))
         {
            sp=solver_parms(fp.isp[0]);

            if (fp.icr[0]==0)
               nsd=2;
            else if (sp.solver==CGNE)
               nsd=3;
            else
               nsd=4;
            
            solver_wsize(fp.isp[0],nsd,0,nws,nwsd,nwv,nwvd);
         }
         else if ((fp.force==FRF_RAT)||
                  (fp.force==FRF_RAT_SDET))
         {
            np=fp.irat[2]-fp.irat[1]+1;
            sp=solver_parms(fp.isp[0]);

            if (sp.solver==MSCG)
               nsd=np;
            else
               nsd=3;

            solver_wsize(fp.isp[0],nsd,np,nws,nwsd,nwv,nwvd);              
         }
      }
   }
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
   int sf;

   sf=sf_flg();
   clear_counters();
   
   udb=udfld();
   cm3x3_assign(4*VOLUME,udb,uold);
   if (sf==1)
      mult_phase(1);
   random_mom();
   act0[0]=momentum_action(0);

   dfl=dfl_parms();

   if (dfl.Ns)
   {
      dfl_modes2(status);
      error_root((status[1]<0)||((status[1]==0)&&(status[0]<0)),1,
                 "start_hmc [hmc.c]","Deflation subspace generation "
                 "failed (status = %d;%d)",status[0],status[1]);

      if (status[1]==0)
         add2counter("modes",0,status);
      else
         add2counter("modes",2,status+1);
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
            error_root(1,1,"start_hmc [hmc.c]","Unknown action");

         n+=1;
      }
   }
}


static void end_hmc(double *act1)
{
   int i,n,nact,*iact;
   int status[4];
   double *mu;
   hmc_parms_t hmc;
   action_parms_t ap;   

   act1[0]=momentum_action(0);
   
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
}


static int accept_hmc(double *act0,double *act1,su3_dble *uold)
{
   int my_rank,nact,iac=0,i;
   double da,r;
   su3_dble *udb;
   hmc_parms_t hmc;
   int sf;

   sf=sf_flg();

   hmc=hmc_parms();
   nact=hmc.nact;
   da=0.0;

   for (i=0;i<=nact;i++)
      da+=(act1[i]-act0[i]);

   if (NPROC>1)
   {
      r=da;
      MPI_Reduce(&r,&da,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
   }

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   
   if (my_rank==0)
   {
      ranlxd(&r,1);

      if (da<=0.0)
         iac=1;
      else if (r<=exp(-da))
         iac=1;
      else
         iac=0;
   }

   if (NPROC>1)
      MPI_Bcast(&iac,1,MPI_INT,0,MPI_COMM_WORLD);    

   if (iac==0)
   {
      udb=udfld();
      cm3x3_assign(4*VOLUME,uold,udb);
      set_flags(UPDATED_UD);
   }
   else
   {
      if (sf==1)
      {
         mult_phase(-1);
         sfbcd();
      }
      renormalize_ud();
   }
      
   return iac;
}


int run_hmc(double *act0,double *act1)
{
   int iac;
   su3_dble **uold;

   uold=reserve_wud(1);

   start_hmc(act0,uold[0]);
   run_mdint();
   end_hmc(act1);
   iac=accept_hmc(act0,act1,uold[0]);

   release_wud();

   return iac;
}
