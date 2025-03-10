
/*******************************************************************************
*
* File flags.h
*
* Copyright (C) 2009, 2010, 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef FLAGS_H
#define FLAGS_H

#ifndef SU3_H
#include "su3.h"
#endif

#ifndef BLOCK_H
#include "block.h"
#endif

typedef enum
{
   UPDATED_U,UPDATED_UD,ASSIGNED_UD2U,
   COPIED_BND_UD,SET_BSTAP,SHIFTED_UD,COMPUTED_FTS,
   ERASED_SW,ERASED_SWD,COMPUTED_SWD,ASSIGNED_SWD2SW,
   INVERTED_SW_E,INVERTED_SW_O,
   INVERTED_SWD_E,INVERTED_SWD_O,
   ASSIGNED_U2UBGR,ASSIGNED_UD2UBGR,ASSIGNED_UD2UDBGR,
   ASSIGNED_SWD2SWBGR,ASSIGNED_SWD2SWDBGR,
   ERASED_AW,ERASED_AWHAT,COMPUTED_AW,COMPUTED_AWHAT,
   EVENTS
} event_t;

typedef enum
{
   U_MATCH_UD,UDBUF_UP2DATE,BSTAP_UP2DATE,
   FTS_UP2DATE,UBGR_MATCH_UD,UDBGR_MATCH_UD,
   SW_UP2DATE,SW_E_INVERTED,SW_O_INVERTED,
   SWD_UP2DATE,SWD_E_INVERTED,SWD_O_INVERTED,
   AW_UP2DATE,AWHAT_UP2DATE,
   QUERIES
} query_t;

typedef enum
{
   ACG,ACF_TM1,ACF_TM1_EO,ACF_TM1_EO_SDET,
   ACF_TM2,ACF_TM2_EO,ACF_RAT,ACF_RAT_SDET,
   ACTIONS
} action_t;

typedef enum
{
   LPFR,OMF2,OMF4,
   INTEGRATORS
} integrator_t;

typedef enum
{
   FRG,FRF_TM1,FRF_TM1_EO,FRF_TM1_EO_SDET,
   FRF_TM2,FRF_TM2_EO,FRF_RAT,FRF_RAT_SDET,
   FORCES
} force_t;

typedef enum
{
   RWTM1,RWTM1_EO,RWTM2,RWTM2_EO,RWRAT,
   RWFACTS
} rwfact_t;

typedef enum
{
   CGNE,MSCG,SAP_GCR,DFL_SAP_GCR,
   SOLVERS
} solver_t;

typedef struct
{
   action_t action;
   int ipf,im0;
   int irat[3],imu[4];
   int isp[4];
} action_parms_t;

typedef struct
{
   int bs[4];
   int Ns;
} dfl_parms_t;

typedef struct
{
   int nkv,nmx;
   double resd,res;
} dfl_pro_parms_t;

typedef struct
{
   int ninv,nmr,ncy;
   int nkv,nmx;
   double kappa,m0,mu,res;
} dfl_gen_parms_t;

typedef struct
{
   int nsm;
   double dtau;
} dfl_upd_parms_t;

typedef struct
{
   force_t force;
   int ipf,im0;
   int irat[3],imu[4];
   int isp[4];
   int ncr[4],icr[4];
} force_parms_t;

typedef struct
{
   int npf,nlv;
   int nact,nmu;
   int *iact;
   double tau,*mu;
} hmc_parms_t;

typedef struct
{
   double beta,c0,c1;
   double kappa_u,kappa_s,kappa_c;
   double csw,cG,cF,dF,zF;
   double m0u,m0s,m0c;
} lat_parms_t;

typedef struct
{
   integrator_t integrator;
   double lambda;
   int nstep,nfr;
   int *ifr;
} mdint_parms_t;

typedef struct
{
   int degree;
   double range[2];
} rat_parms_t;

typedef struct
{
   rwfact_t rwfact;
   int im0,irp,nsrc;
   int n,*np,*isp;
   double mu;
} rw_parms_t;

typedef struct
{
   double m0,csw,cF;
} sw_parms_t;

typedef struct
{
   int bs[4];
   int isolv;
   int nmr,ncy;
} sap_parms_t;

typedef struct
{
   int flg;
   double phi[3],phi_prime[3];
   double theta[3];
} sf_parms_t;

typedef struct
{
   solver_t solver;
   int nmx,nkv;
   int isolv,nmr,ncy;
   double res;
} solver_parms_t;

typedef struct
{
   int eoflg;
} tm_parms_t;

typedef struct
{
   int tau3;
} xsf_parms_t;

typedef struct
{
   int n;
   double eps;
} wflow_parms_t;

/* FLAGS_C */
extern void set_flags(event_t event);
extern void set_grid_flags(blk_grid_t grid,event_t event);
extern int query_flags(query_t query);
extern int query_grid_flags(blk_grid_t grid,query_t query);
extern void print_flags(void);
extern void print_grid_flags(blk_grid_t grid);

/* ACTION_PARMS_C */
extern action_parms_t set_action_parms(int iact,action_t action,int ipf,
                                       int im0,int *irat,int *imu,int *isp);
extern action_parms_t action_parms(int iact);
extern void read_action_parms(int iact);
extern void print_action_parms(void);
extern void write_action_parms(FILE *fdat);
extern void check_action_parms(FILE *fdat);

/* DFL_PARMS_C */
extern dfl_parms_t set_dfl_parms(int *bs,int Ns);
extern dfl_parms_t dfl_parms(void);
extern dfl_pro_parms_t set_dfl_pro_parms(int nkv,int nmx,
                                         double resd,double res);
extern dfl_pro_parms_t dfl_pro_parms(void);
extern dfl_gen_parms_t set_dfl_gen_parms(double kappa,double mu,
                                         int ninv,int nmr,int ncy,
                                         int nkv,int nmx,double res);
extern dfl_gen_parms_t dfl_gen_parms(void);
extern dfl_upd_parms_t set_dfl_upd_parms(double dtau,int nsm);
extern dfl_upd_parms_t dfl_upd_parms(void);
extern void print_dfl_parms(int ipr);
extern void write_dfl_parms(FILE *fdat);
extern void check_dfl_parms(FILE *fdat);

/* FORCE_PARMS_C */
extern force_parms_t set_force_parms(int ifr,force_t force,int ipf,int im0,
                                     int *irat,int *imu,int *isp,int *ncr);
extern force_parms_t force_parms(int ifr);
extern void read_force_parms(int ifr);
extern void read_force_parms2(int ifr);
extern void print_force_parms(void);
extern void print_force_parms2(void);
extern void write_force_parms(FILE *fdat);
extern void check_force_parms(FILE *fdat);

/* HMC_PARMS_C */
extern hmc_parms_t set_hmc_parms(int nact,int *iact,int npf,
                                 int nmu,double *mu,int nlv,double tau);
extern hmc_parms_t hmc_parms(void);
extern void print_hmc_parms(void);
extern void write_hmc_parms(FILE *fdat);
extern void check_hmc_parms(FILE *fdat);

/* LAT_PARMS_C */
extern lat_parms_t set_lat_parms(double beta,double c0,
                                 double kappa_u,double kappa_s,double kappa_c,
                                 double csw,double cG,double cF,double dF,double zF);
extern lat_parms_t lat_parms(void);
extern void print_lat_parms(void);
extern void write_lat_parms(FILE *fdat);
extern void check_lat_parms(FILE *fdat);
extern double sea_quark_mass(int im0);
extern sw_parms_t set_sw_parms(double m0);
extern sw_parms_t sw_parms(void);
extern tm_parms_t set_tm_parms(int eoflg);
extern tm_parms_t tm_parms(void);
extern xsf_parms_t set_xsf_parms(int tau3);
extern xsf_parms_t xsf_parms(void);

/* MDINT_PARMS_C */
extern mdint_parms_t set_mdint_parms(int ilv,
                                     integrator_t integrator,double lambda,
                                     int nstep,int nfr,int *ifr);
extern mdint_parms_t mdint_parms(int ilv);
extern void read_mdint_parms(int ilv);
extern void print_mdint_parms(void);
extern void write_mdint_parms(FILE *fdat);
extern void check_mdint_parms(FILE *fdat);

/* RAT_PARMS_C */
extern rat_parms_t set_rat_parms(int irp,int degree,double *range);
extern rat_parms_t rat_parms(int irp);
extern void read_rat_parms(int irp);
extern void print_rat_parms(void);
extern void write_rat_parms(FILE *fdat);
extern void check_rat_parms(FILE *fdat);

/* RW_PARMS_C */
extern rw_parms_t set_rw_parms(int irw,rwfact_t rwfact,int im0,double mu,
                               int irp,int n,int *np,int *isp,int nsrc);
extern rw_parms_t rw_parms(int irw);
extern void read_rw_parms(int irw);
extern void print_rw_parms(void);
extern void write_rw_parms(FILE *fdat);
extern void check_rw_parms(FILE *fdat);

/* SAP_PARMS_C */
extern sap_parms_t set_sap_parms(int *bs,int isolv,int nmr,int ncy);
extern sap_parms_t sap_parms(void);
extern void print_sap_parms(int ipr);
extern void write_sap_parms(FILE *fdat);
extern void check_sap_parms(FILE *fdat);

/* SF_PARMS_C */
extern sf_parms_t set_sf_parms(double *phi,double *phi_prime,double *theta);
extern sf_parms_t sf_parms(void);
extern void print_sf_parms(void);
extern void write_sf_parms(FILE *fdat);
extern void check_sf_parms(FILE *fdat);
extern int sf_flg(void);

/* SOLVER_PARMS_C */
extern solver_parms_t set_solver_parms(int isp,solver_t solver,
                                       int nkv,int isolv,int nmr,int ncy,
                                       int nmx,double res);
extern solver_parms_t solver_parms(int isp);
extern void read_solver_parms(int isp);
extern void print_solver_parms(int *isap,int *idfl);
extern void write_solver_parms(FILE *fdat);
extern void check_solver_parms(FILE *fdat);

/* WFLOW_PARMS_C */
extern wflow_parms_t set_wflow_parms(int n,double eps);
extern wflow_parms_t wflow_parms(void);

#endif
