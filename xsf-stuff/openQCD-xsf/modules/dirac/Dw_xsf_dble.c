
/*******************************************************************************
*
* File Dw_xsf_dble.c
*
* Copyright (C) 2013,2014 John Bulava, Mattia Dalla Brida
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Application of the O(a)-improved Wilson-Dirac operator Dw
*
* The externally accessible functions are
*
*   void Dw_xsf_dble(double mu,spinor_dble *s,spinor_dble *r)
*     Depending on whether the twisted-mass flag is set or not, this
*     program applies Dw_xsf+i*mu*gamma_5*1e or Dw_xsf+i*mu*gamma_5 to the
*     field s and assigns the result to the field r. On exit s is unchanged.
*
*   void Dwee_xsf_dble(double mu,spinor_dble *s,spinor_dble *r)
*     Applies Dw_ee_xsf+i*mu*gamma_5 to the field s on the even points of the
*     lattice and assigns the result to the field r. On exit s is unchanged.
*
*   void Dwoo_xsf_dble(double mu,spinor_dble *s,spinor_dble *r)
*     Depending on whether the twisted-mass flag is set or not, this program 
*     applies Dw_oo_xsf or Dw_oo_xsf+i*mu*gamma_5 to the field s on the odd 
*     points of the lattice and assigns the result to the field r. On exit 
*     s is unchanged.  
*
*   void Dwoe_xsf_dble(spinor_dble *s,spinor_dble *r)
*     Applies Dw_oe_xsf to the field s and assigns the result to the field r.
*     On exit s is unchanged.
*
*   void Dweo_xsf_dble(spinor_dble *s,spinor_dble *r)
*     Applies Dw_eo_xsf to the field s and *subtracts* the result from the
*     field r. On exit s is unchanged. 
*
*   void Dwhat_xsf_dble(double mu,spinor_dble *s,spinor_dble *r)
*     Applies Dwhat_xsf+i*mu*gamma_5 to the field s and assigns the result to
*     the field r. On exit s is unchanged.  
*
* Notes:
*
* The notation and normalization conventions are specified in the notes
* "Implementation of the lattice Dirac operator" (file doc/dirac.pdf).
*
* In all these programs, it is assumed that the SW term is in the proper
* condition and that the spinor fields have NSPIN elements. The programs 
* check whether the twisted-mass flag (see flags/lat_parms.c) is set and
* turn off the twisted-mass term on the odd lattice sites if it is. The
* input and output fields may not coincide in the case of the programs
* Dw_dble() and Dwhat_dble(). The program also assumes that the flavour 
* flag has been set through set_xsf_parms() to a proper values.   
*
* The block programs assume homogenous Dirichlet boundary conditions at the
* block boundaries. In addition, the boundary conditions at global time 0
* and NPROC0*L0-1 satisfied by the full-lattice Dirac operator are imposed.
* The even-odd preconditioned operator is in all cases obtained from the
* ee,eo,oe and oo parts of the un-preconditioned operator, where all parts
* respect the boundary conditions.
*
* The programs Dw_dble(),..,Dwhat_dble() perform global operations and must
* be called simultaneously on all processes.
*
*******************************************************************************/

#define DW_XSF_DBLE_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "utils.h"
#include "flags.h"
#include "lattice.h"
#include "uflds.h"
#include "sflds.h"
#include "sw_term.h"
#include "dirac.h"
#include "global.h"

typedef union
{
   spinor_dble s;
   weyl_dble w[2];
} spin_t;

static double coe,ceo;
static const spinor_dble sd0={{{0.0}}};
static spin_t rs ALIGNED16;

#if (defined x64)
#include "sse2.h"

#define _load_cst(c) \
__asm__ __volatile__ ("movddup %0, %%xmm15" \
                      : \
                      : \
                      "m" (c) \
                      : \
                      "xmm15")

#define _mul_cst() \
__asm__ __volatile__ ("mulpd %%xmm15, %%xmm0 \n\t" \
                      "mulpd %%xmm15, %%xmm1 \n\t" \
                      "mulpd %%xmm15, %%xmm2" \
                      : \
                      : \
                      : \
                      "xmm0", "xmm1", "xmm2")

#define _mul_cst_up() \
__asm__ __volatile__ ("mulpd %%xmm15, %%xmm3 \n\t" \
                      "mulpd %%xmm15, %%xmm4 \n\t" \
                      "mulpd %%xmm15, %%xmm5" \
                      : \
                      : \
                      : \
                      "xmm3", "xmm4", "xmm5")


static void doe(int *piup,int *pidn,su3_dble *u,spinor_dble *pk)
{
   spinor_dble *sp,*sm;

/******************************* direction +0 *********************************/

   sp=pk+(*(piup++));

   _sse_load_dble((*sp).c1);
   _sse_load_up_dble((*sp).c3);

   sm=pk+(*(pidn++));
   _prefetch_spinor_dble(sm);
   _sse_vector_add_dble();
   _sse_su3_multiply_dble(*u);
   _sse_store_up_dble(rs.s.c1);
   _sse_store_up_dble(rs.s.c3);

   _sse_load_dble((*sp).c2);
   _sse_load_up_dble((*sp).c4);

   u+=1;
   _prefetch_su3_dble(u);
   u-=1;
   
   _sse_vector_add_dble();
   _sse_su3_multiply_dble(*u);

   _sse_store_up_dble(rs.s.c2);
   _sse_store_up_dble(rs.s.c4);

/******************************* direction -0 *********************************/

   _sse_load_dble((*sm).c1);
   _sse_load_up_dble((*sm).c3);

   sp=pk+(*(piup++));
   _prefetch_spinor_dble(sp);
   _sse_vector_sub_dble();
   u+=1;
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble(rs.s.c1);
   _sse_vector_add_dble();
   _sse_store_dble(rs.s.c1);

   _sse_load_dble(rs.s.c3);
   _sse_vector_sub_dble();
   _sse_store_dble(rs.s.c3);

   _sse_load_dble((*sm).c2);
   _sse_load_up_dble((*sm).c4);

   u+=1;
   _prefetch_su3_dble(u);
   u-=1;
   
   _sse_vector_sub_dble();
   _sse_su3_inverse_multiply_dble(*u);
   
   _sse_load_dble(rs.s.c2);
   _sse_vector_add_dble();
   _sse_store_dble(rs.s.c2);

   _sse_load_dble(rs.s.c4);
   _sse_vector_sub_dble();
   _sse_store_dble(rs.s.c4);

/******************************* direction +1 *********************************/

   _sse_load_dble((*sp).c1);
   _sse_load_up_dble((*sp).c4);

   sm=pk+(*(pidn++));
   _prefetch_spinor_dble(sm);
   _sse_vector_i_add_dble();
   u+=1;
   _sse_su3_multiply_dble(*u);

   _sse_load_dble(rs.s.c1);
   _sse_vector_add_dble();
   _sse_store_dble(rs.s.c1);

   _sse_load_dble(rs.s.c4);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_store_dble(rs.s.c4);

   _sse_load_dble((*sp).c2);
   _sse_load_up_dble((*sp).c3);

   u+=1;
   _prefetch_su3_dble(u);
   u-=1;

   _sse_vector_i_add_dble();
   _sse_su3_multiply_dble(*u);

   _sse_load_dble(rs.s.c2);
   _sse_vector_add_dble();
   _sse_store_dble(rs.s.c2);

   _sse_load_dble(rs.s.c3);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_store_dble(rs.s.c3);

/******************************* direction -1 *********************************/

   _sse_load_dble((*sm).c1);
   _sse_load_up_dble((*sm).c4);

   sp=pk+(*(piup++));
   _prefetch_spinor_dble(sp);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   u+=1;
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble(rs.s.c1);
   _sse_vector_add_dble();
   _sse_store_dble(rs.s.c1);

   _sse_load_dble(rs.s.c4);
   _sse_vector_i_add_dble();
   _sse_store_dble(rs.s.c4);

   _sse_load_dble((*sm).c2);
   _sse_load_up_dble((*sm).c3);

   u+=1;
   _prefetch_su3_dble(u);
   u-=1;

   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble(rs.s.c2);
   _sse_vector_add_dble();
   _sse_store_dble(rs.s.c2);

   _sse_load_dble(rs.s.c3);
   _sse_vector_i_add_dble();
   _sse_store_dble(rs.s.c3);

/******************************* direction +2 *********************************/

   _sse_load_dble((*sp).c1);
   _sse_load_up_dble((*sp).c4);

   sm=pk+(*(pidn++));
   _prefetch_spinor_dble(sm);
   _sse_vector_add_dble();
   u+=1;
   _sse_su3_multiply_dble(*u);

   _sse_load_dble(rs.s.c1);
   _sse_vector_add_dble();
   _sse_store_dble(rs.s.c1);

   _sse_load_dble(rs.s.c4);
   _sse_vector_add_dble();
   _sse_store_dble(rs.s.c4);

   _sse_load_dble((*sp).c2);
   _sse_load_up_dble((*sp).c3);

   u+=1;
   _prefetch_su3_dble(u);
   u-=1;

   _sse_vector_sub_dble();
   _sse_su3_multiply_dble(*u);

   _sse_load_dble(rs.s.c2);
   _sse_vector_add_dble();
   _sse_store_dble(rs.s.c2);

   _sse_load_dble(rs.s.c3);
   _sse_vector_sub_dble();
   _sse_store_dble(rs.s.c3);

/******************************* direction -2 *********************************/

   _sse_load_dble((*sm).c1);
   _sse_load_up_dble((*sm).c4);

   sp=pk+(*(piup));
   _prefetch_spinor_dble(sp);
   _sse_vector_sub_dble();
   u+=1;
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble(rs.s.c1);
   _sse_vector_add_dble();
   _sse_store_dble(rs.s.c1);

   _sse_load_dble(rs.s.c4);
   _sse_vector_sub_dble();
   _sse_store_dble(rs.s.c4);

   _sse_load_dble((*sm).c2);
   _sse_load_up_dble((*sm).c3);

   u+=1;
   _prefetch_su3_dble(u);
   u-=1;

   _sse_vector_add_dble();
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble(rs.s.c2);
   _sse_vector_add_dble();
   _sse_store_dble(rs.s.c2);

   _sse_load_dble(rs.s.c3);
   _sse_vector_add_dble();
   _sse_store_dble(rs.s.c3);

/******************************* direction +3 *********************************/

   _sse_load_dble((*sp).c1);
   _sse_load_up_dble((*sp).c3);

   sm=pk+(*(pidn));
   _prefetch_spinor_dble(sm);
   _sse_vector_i_add_dble();
   u+=1;
   _sse_su3_multiply_dble(*u);

   _sse_load_dble(rs.s.c1);
   _sse_vector_add_dble();
   _sse_store_dble(rs.s.c1);

   _sse_load_dble(rs.s.c3);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_store_dble(rs.s.c3);

   _sse_load_dble((*sp).c2);
   _sse_load_up_dble((*sp).c4);

   u+=1;
   _prefetch_su3_dble(u);
   u-=1;

   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_su3_multiply_dble(*u);

   _sse_load_dble(rs.s.c2);
   _sse_vector_add_dble();
   _sse_store_dble(rs.s.c2);

   _sse_load_dble(rs.s.c4);
   _sse_vector_i_add_dble();
   _sse_store_dble(rs.s.c4);

/******************************* direction -3 *********************************/

   _sse_load_dble((*sm).c1);
   _sse_load_up_dble((*sm).c3);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   u+=1;
   _sse_su3_inverse_multiply_dble(*u);

   _load_cst(coe);
   _sse_load_dble(rs.s.c1);
   _sse_vector_add_dble();
   _mul_cst();
   _sse_store_dble(rs.s.c1);

   _sse_load_dble(rs.s.c3);
   _sse_vector_i_add_dble();
   _mul_cst();
   _sse_store_dble(rs.s.c3);

   _sse_load_dble((*sm).c2);
   _sse_load_up_dble((*sm).c4);

   u+=1;
   _prefetch_su3_dble(u);
   u-=1;

   _sse_vector_i_add_dble();
   _sse_su3_inverse_multiply_dble(*u);

   _load_cst(coe);
   _sse_load_dble(rs.s.c2);
   _sse_vector_add_dble();
   _mul_cst();
   _sse_store_dble(rs.s.c2);

   _sse_load_dble(rs.s.c4);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _mul_cst();
   _sse_store_dble(rs.s.c4);
}


static void deo(int *piup,int *pidn,su3_dble *u,spinor_dble *pl)
{
   spinor_dble *sp,*sm;

/******************************* direction +0 *********************************/

   sp=pl+(*(piup++));
   _prefetch_spinor_dble(sp);

   _load_cst(ceo);
   _sse_load_dble(rs.s.c1);
   _sse_load_up_dble(rs.s.c3);
   _mul_cst();
   _mul_cst_up();
   _sse_store_dble(rs.s.c1);
   _sse_store_up_dble(rs.s.c3);   

   sm=pl+(*(pidn++));
   _prefetch_spinor_dble(sm);
   _sse_vector_sub_dble();
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*sp).c1);
   _sse_vector_add_dble();
   _sse_store_dble((*sp).c1);

   _sse_load_dble((*sp).c3);
   _sse_vector_sub_dble();
   _sse_store_dble((*sp).c3);

   _load_cst(ceo);   
   _sse_load_dble(rs.s.c2);
   _sse_load_up_dble(rs.s.c4);
   _mul_cst();
   _mul_cst_up();
   _sse_store_dble(rs.s.c2);
   _sse_store_up_dble(rs.s.c4);
   
   _sse_vector_sub_dble();
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*sp).c2);
   _sse_vector_add_dble();
   _sse_store_dble((*sp).c2);

   _sse_load_dble((*sp).c4);
   _sse_vector_sub_dble();
   _sse_store_dble((*sp).c4);

/******************************* direction -0 *********************************/

   _sse_load_dble(rs.s.c1);
   _sse_load_up_dble(rs.s.c3);

   sp=pl+(*(piup++));
   _prefetch_spinor_dble(sp);
   _sse_vector_add_dble();
   u+=1;
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*sm).c1);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c1);

   _sse_load_dble((*sm).c3);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c3);

   _sse_load_dble(rs.s.c2);
   _sse_load_up_dble(rs.s.c4);

   _sse_vector_add_dble();
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*sm).c2);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c2);

   _sse_load_dble((*sm).c4);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c4);

/******************************* direction +1 *********************************/

   _sse_load_dble(rs.s.c1);
   _sse_load_up_dble(rs.s.c4);

   sm=pl+(*(pidn++));
   _prefetch_spinor_dble(sm);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   u+=1;   
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*sp).c1);
   _sse_vector_add_dble();
   _sse_store_dble((*sp).c1);

   _sse_load_dble((*sp).c4);
   _sse_vector_i_add_dble();
   _sse_store_dble((*sp).c4);

   _sse_load_dble(rs.s.c2);
   _sse_load_up_dble(rs.s.c3);

   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*sp).c2);
   _sse_vector_add_dble();
   _sse_store_dble((*sp).c2);

   _sse_load_dble((*sp).c3);
   _sse_vector_i_add_dble();
   _sse_store_dble((*sp).c3);

/******************************* direction -1 *********************************/

   _sse_load_dble(rs.s.c1);
   _sse_load_up_dble(rs.s.c4);

   sp=pl+(*(piup++));
   _prefetch_spinor_dble(sp);
   _sse_vector_i_add_dble();
   u+=1;
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*sm).c1);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c1);

   _sse_load_dble((*sm).c4);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_store_dble((*sm).c4);

   _sse_load_dble(rs.s.c2);
   _sse_load_up_dble(rs.s.c3);

   _sse_vector_i_add_dble();
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*sm).c2);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c2);

   _sse_load_dble((*sm).c3);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_store_dble((*sm).c3);

/******************************* direction +2 *********************************/

   _sse_load_dble(rs.s.c1);
   _sse_load_up_dble(rs.s.c4);

   sm=pl+(*(pidn++));
   _prefetch_spinor_dble(sm);
   _sse_vector_sub_dble();
   u+=1;
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*sp).c1);
   _sse_vector_add_dble();
   _sse_store_dble((*sp).c1);

   _sse_load_dble((*sp).c4);
   _sse_vector_sub_dble();
   _sse_store_dble((*sp).c4);

   _sse_load_dble(rs.s.c2);
   _sse_load_up_dble(rs.s.c3);

   _sse_vector_add_dble();
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*sp).c2);
   _sse_vector_add_dble();
   _sse_store_dble((*sp).c2);

   _sse_load_dble((*sp).c3);
   _sse_vector_add_dble();
   _sse_store_dble((*sp).c3);

/******************************* direction -2 *********************************/

   _sse_load_dble(rs.s.c1);
   _sse_load_up_dble(rs.s.c4);

   sp=pl+(*(piup));
   _prefetch_spinor_dble(sp);
   _sse_vector_add_dble();
   u+=1;
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*sm).c1);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c1);

   _sse_load_dble((*sm).c4);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c4);

   _sse_load_dble(rs.s.c2);
   _sse_load_up_dble(rs.s.c3);

   _sse_vector_sub_dble();
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*sm).c2);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c2);

   _sse_load_dble((*sm).c3);
   _sse_vector_sub_dble();
   _sse_store_dble((*sm).c3);

/******************************* direction +3 *********************************/

   _sse_load_dble(rs.s.c1);
   _sse_load_up_dble(rs.s.c3);

   sm=pl+(*(pidn));
   _prefetch_spinor_dble(sm);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   u+=1;
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*sp).c1);
   _sse_vector_add_dble();
   _sse_store_dble((*sp).c1);

   _sse_load_dble((*sp).c3);
   _sse_vector_i_add_dble();
   _sse_store_dble((*sp).c3);

   _sse_load_dble(rs.s.c2);
   _sse_load_up_dble(rs.s.c4);

   _sse_vector_i_add_dble();
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*sp).c2);
   _sse_vector_add_dble();
   _sse_store_dble((*sp).c2);

   _sse_load_dble((*sp).c4);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_store_dble((*sp).c4);

/******************************* direction -3 *********************************/

   _sse_load_dble(rs.s.c1);
   _sse_load_up_dble(rs.s.c3);

   _sse_vector_i_add_dble();
   u+=1;
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*sm).c1);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c1);

   _sse_load_dble((*sm).c3);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_store_dble((*sm).c3);

   _sse_load_dble(rs.s.c2);
   _sse_load_up_dble(rs.s.c4);

   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*sm).c2);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c2);

   _sse_load_dble((*sm).c4);
   _sse_vector_i_add_dble();
   _sse_store_dble((*sm).c4);
}

#else

static void doe(int *piup,int *pidn,su3_dble *u,spinor_dble *pk)
{
   spinor_dble *sp,*sm;
   su3_vector_dble psi,chi;

/******************************* direction +0 *********************************/

   sp=pk+(*(piup++));

   _vector_add(psi,(*sp).c1,(*sp).c3);
   _su3_multiply(rs.s.c1,*u,psi);
   rs.s.c3=rs.s.c1;

   _vector_add(psi,(*sp).c2,(*sp).c4);
   _su3_multiply(rs.s.c2,*u,psi);
   rs.s.c4=rs.s.c2;

/******************************* direction -0 *********************************/

   sm=pk+(*(pidn++));
   u+=1;

   _vector_sub(psi,(*sm).c1,(*sm).c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c1,chi);
   _vector_sub_assign(rs.s.c3,chi);

   _vector_sub(psi,(*sm).c2,(*sm).c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c2,chi);
   _vector_sub_assign(rs.s.c4,chi);

/******************************* direction +1 *********************************/

   sp=pk+(*(piup++));
   u+=1;

   _vector_i_add(psi,(*sp).c1,(*sp).c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c1,chi);
   _vector_i_sub_assign(rs.s.c4,chi);

   _vector_i_add(psi,(*sp).c2,(*sp).c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c2,chi);
   _vector_i_sub_assign(rs.s.c3,chi);

/******************************* direction -1 *********************************/

   sm=pk+(*(pidn++));
   u+=1;

   _vector_i_sub(psi,(*sm).c1,(*sm).c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c1,chi);
   _vector_i_add_assign(rs.s.c4,chi);

   _vector_i_sub(psi,(*sm).c2,(*sm).c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c2,chi);
   _vector_i_add_assign(rs.s.c3,chi);

/******************************* direction +2 *********************************/

   sp=pk+(*(piup++));
   u+=1;

   _vector_add(psi,(*sp).c1,(*sp).c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c1,chi);
   _vector_add_assign(rs.s.c4,chi);

   _vector_sub(psi,(*sp).c2,(*sp).c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c2,chi);
   _vector_sub_assign(rs.s.c3,chi);

/******************************* direction -2 *********************************/

   sm=pk+(*(pidn++));
   u+=1;

   _vector_sub(psi,(*sm).c1,(*sm).c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c1,chi);
   _vector_sub_assign(rs.s.c4,chi);

   _vector_add(psi,(*sm).c2,(*sm).c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c2,chi);
   _vector_add_assign(rs.s.c3,chi);

/******************************* direction +3 *********************************/

   sp=pk+(*(piup));
   u+=1;

   _vector_i_add(psi,(*sp).c1,(*sp).c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c1,chi);
   _vector_i_sub_assign(rs.s.c3,chi);

   _vector_i_sub(psi,(*sp).c2,(*sp).c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c2,chi);
   _vector_i_add_assign(rs.s.c4,chi);

/******************************* direction -3 *********************************/

   sm=pk+(*(pidn));
   u+=1;

   _vector_i_sub(psi,(*sm).c1,(*sm).c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c1,chi);
   _vector_i_add_assign(rs.s.c3,chi);

   _vector_i_add(psi,(*sm).c2,(*sm).c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c2,chi);
   _vector_i_sub_assign(rs.s.c4,chi);

   _vector_mul_assign(rs.s.c1,coe);
   _vector_mul_assign(rs.s.c2,coe);
   _vector_mul_assign(rs.s.c3,coe);
   _vector_mul_assign(rs.s.c4,coe);
}


static void deo(int *piup,int *pidn,su3_dble *u,spinor_dble *pl)
{
   spinor_dble *sp,*sm;
   su3_vector_dble psi,chi;

   _vector_mul_assign(rs.s.c1,ceo);
   _vector_mul_assign(rs.s.c2,ceo);
   _vector_mul_assign(rs.s.c3,ceo);
   _vector_mul_assign(rs.s.c4,ceo);
   
/******************************* direction +0 *********************************/

   sp=pl+(*(piup++));

   _vector_sub(psi,rs.s.c1,rs.s.c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c1,chi);
   _vector_sub_assign((*sp).c3,chi);

   _vector_sub(psi,rs.s.c2,rs.s.c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c2,chi);
   _vector_sub_assign((*sp).c4,chi);

/******************************* direction -0 *********************************/

   sm=pl+(*(pidn++));
   u+=1;

   _vector_add(psi,rs.s.c1,rs.s.c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c1,chi);
   _vector_add_assign((*sm).c3,chi);

   _vector_add(psi,rs.s.c2,rs.s.c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c2,chi);
   _vector_add_assign((*sm).c4,chi);

/******************************* direction +1 *********************************/

   sp=pl+(*(piup++));
   u+=1;

   _vector_i_sub(psi,rs.s.c1,rs.s.c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c1,chi);
   _vector_i_add_assign((*sp).c4,chi);

   _vector_i_sub(psi,rs.s.c2,rs.s.c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c2,chi);
   _vector_i_add_assign((*sp).c3,chi);

/******************************* direction -1 *********************************/

   sm=pl+(*(pidn++));
   u+=1;

   _vector_i_add(psi,rs.s.c1,rs.s.c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c1,chi);
   _vector_i_sub_assign((*sm).c4,chi);

   _vector_i_add(psi,rs.s.c2,rs.s.c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c2,chi);
   _vector_i_sub_assign((*sm).c3,chi);

/******************************* direction +2 *********************************/

   sp=pl+(*(piup++));
   u+=1;

   _vector_sub(psi,rs.s.c1,rs.s.c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c1,chi);
   _vector_sub_assign((*sp).c4,chi);

   _vector_add(psi,rs.s.c2,rs.s.c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c2,chi);
   _vector_add_assign((*sp).c3,chi);

/******************************* direction -2 *********************************/

   sm=pl+(*(pidn++));
   u+=1;

   _vector_add(psi,rs.s.c1,rs.s.c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c1,chi);
   _vector_add_assign((*sm).c4,chi);

   _vector_sub(psi,rs.s.c2,rs.s.c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c2,chi);
   _vector_sub_assign((*sm).c3,chi);

/******************************* direction +3 *********************************/

   sp=pl+(*(piup));
   u+=1;

   _vector_i_sub(psi,rs.s.c1,rs.s.c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c1,chi);
   _vector_i_add_assign((*sp).c3,chi);

   _vector_i_add(psi,rs.s.c2,rs.s.c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c2,chi);
   _vector_i_sub_assign((*sp).c4,chi);

/******************************* direction -3 *********************************/

   sm=pl+(*(pidn));
   u+=1;

   _vector_i_add(psi,rs.s.c1,rs.s.c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c1,chi);
   _vector_i_sub_assign((*sm).c3,chi);

   _vector_i_sub(psi,rs.s.c2,rs.s.c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c2,chi);
   _vector_i_add_assign((*sm).c4,chi);
}

#endif


static void doe_bnd(int t,int *piup,int *pidn,su3_dble *u,spinor_dble *pk)
{
   spinor_dble *sp,*sm;
   su3_vector_dble psi,chi;
   lat_parms_t lat;
   double dF; 

   lat=lat_parms();
   dF=lat.dF;

   if (t==0)
   {
/******************************* direction +0 *********************************/

      sp=pk+(*(piup++));

      _vector_add(psi,(*sp).c1,(*sp).c3);
      _su3_multiply(rs.s.c1,*u,psi);
      rs.s.c3=rs.s.c1;

      _vector_add(psi,(*sp).c2,(*sp).c4);
      _su3_multiply(rs.s.c2,*u,psi);
      rs.s.c4=rs.s.c2;

      pidn+=1;
      u+=1;
   }
   else 
   {
/******************************* direction -0 *********************************/

      sm=pk+(*(pidn++));
      u+=1;

      _vector_sub(psi,(*sm).c1,(*sm).c3);
      _su3_inverse_multiply(rs.s.c1,*u,psi);
      _vector_mul(rs.s.c3,-1.0,rs.s.c1);

      _vector_sub(psi,(*sm).c2,(*sm).c4);
      _su3_inverse_multiply(rs.s.c2,*u,psi);
      _vector_mul(rs.s.c4,-1.0,rs.s.c2);

      piup+=1;
   }

   _vector_mul_assign(rs.s.c1,1.0/dF);
   _vector_mul_assign(rs.s.c2,1.0/dF);
   _vector_mul_assign(rs.s.c3,1.0/dF);
   _vector_mul_assign(rs.s.c4,1.0/dF);
  
/******************************* direction +1 *********************************/

   sp=pk+(*(piup++));
   u+=1;

   _vector_i_add(psi,(*sp).c1,(*sp).c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c1,chi);
   _vector_i_sub_assign(rs.s.c4,chi);

   _vector_i_add(psi,(*sp).c2,(*sp).c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c2,chi);
   _vector_i_sub_assign(rs.s.c3,chi);

/******************************* direction -1 *********************************/

   sm=pk+(*(pidn++));
   u+=1;

   _vector_i_sub(psi,(*sm).c1,(*sm).c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c1,chi);
   _vector_i_add_assign(rs.s.c4,chi);

   _vector_i_sub(psi,(*sm).c2,(*sm).c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c2,chi);
   _vector_i_add_assign(rs.s.c3,chi);

/******************************* direction +2 *********************************/

   sp=pk+(*(piup++));
   u+=1;

   _vector_add(psi,(*sp).c1,(*sp).c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c1,chi);
   _vector_add_assign(rs.s.c4,chi);

   _vector_sub(psi,(*sp).c2,(*sp).c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c2,chi);
   _vector_sub_assign(rs.s.c3,chi);

/******************************* direction -2 *********************************/

   sm=pk+(*(pidn++));
   u+=1;

   _vector_sub(psi,(*sm).c1,(*sm).c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c1,chi);
   _vector_sub_assign(rs.s.c4,chi);

   _vector_add(psi,(*sm).c2,(*sm).c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c2,chi);
   _vector_add_assign(rs.s.c3,chi);

/******************************* direction +3 *********************************/

   sp=pk+(*(piup));
   u+=1;

   _vector_i_add(psi,(*sp).c1,(*sp).c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c1,chi);
   _vector_i_sub_assign(rs.s.c3,chi);

   _vector_i_sub(psi,(*sp).c2,(*sp).c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c2,chi);
   _vector_i_add_assign(rs.s.c4,chi);

/******************************* direction -3 *********************************/

   sm=pk+(*(pidn));
   u+=1;

   _vector_i_sub(psi,(*sm).c1,(*sm).c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c1,chi);
   _vector_i_add_assign(rs.s.c3,chi);

   _vector_i_add(psi,(*sm).c2,(*sm).c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign(rs.s.c2,chi);
   _vector_i_sub_assign(rs.s.c4,chi);

   _vector_mul_assign(rs.s.c1,dF*coe);
   _vector_mul_assign(rs.s.c2,dF*coe);
   _vector_mul_assign(rs.s.c3,dF*coe);
   _vector_mul_assign(rs.s.c4,dF*coe);

}


static void deo_bnd(int t,int *piup,int *pidn,su3_dble *u,spinor_dble *pl)
{
   spinor_dble *sp,*sm;
   su3_vector_dble psi,chi;
   lat_parms_t lat;
   double dF; 

   lat=lat_parms();
   dF=lat.dF;

   _vector_mul_assign(rs.s.c1,ceo);
   _vector_mul_assign(rs.s.c2,ceo);
   _vector_mul_assign(rs.s.c3,ceo);
   _vector_mul_assign(rs.s.c4,ceo);

   if (t==0)
   {
/******************************* direction +0 *********************************/

      sp=pl+(*(piup++));

      _vector_sub(psi,rs.s.c1,rs.s.c3);
      _su3_inverse_multiply(chi,*u,psi);
      _vector_add_assign((*sp).c1,chi);
      _vector_sub_assign((*sp).c3,chi);

      _vector_sub(psi,rs.s.c2,rs.s.c4);
      _su3_inverse_multiply(chi,*u,psi);
      _vector_add_assign((*sp).c2,chi);
      _vector_sub_assign((*sp).c4,chi);

      pidn+=1;
      u+=1;
   }   
   else 
   {
/******************************* direction -0 *********************************/

      sm=pl+(*(pidn++));
      u+=1;

      _vector_add(psi,rs.s.c1,rs.s.c3);
      _su3_multiply(chi,*u,psi);
      _vector_add_assign((*sm).c1,chi);
      _vector_add_assign((*sm).c3,chi);

      _vector_add(psi,rs.s.c2,rs.s.c4);
      _su3_multiply(chi,*u,psi);
      _vector_add_assign((*sm).c2,chi);
      _vector_add_assign((*sm).c4,chi);

      piup+=1;
   }

   _vector_mul_assign(rs.s.c1,dF);
   _vector_mul_assign(rs.s.c2,dF);
   _vector_mul_assign(rs.s.c3,dF);
   _vector_mul_assign(rs.s.c4,dF);

/******************************* direction +1 *********************************/

   sp=pl+(*(piup++));
   u+=1;

   _vector_i_sub(psi,rs.s.c1,rs.s.c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c1,chi);
   _vector_i_add_assign((*sp).c4,chi);

   _vector_i_sub(psi,rs.s.c2,rs.s.c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c2,chi);
   _vector_i_add_assign((*sp).c3,chi);

/******************************* direction -1 *********************************/

   sm=pl+(*(pidn++));
   u+=1;

   _vector_i_add(psi,rs.s.c1,rs.s.c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c1,chi);
   _vector_i_sub_assign((*sm).c4,chi);

   _vector_i_add(psi,rs.s.c2,rs.s.c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c2,chi);
   _vector_i_sub_assign((*sm).c3,chi);

/******************************* direction +2 *********************************/

   sp=pl+(*(piup++));
   u+=1;

   _vector_sub(psi,rs.s.c1,rs.s.c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c1,chi);
   _vector_sub_assign((*sp).c4,chi);

   _vector_add(psi,rs.s.c2,rs.s.c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c2,chi);
   _vector_add_assign((*sp).c3,chi);

/******************************* direction -2 *********************************/

   sm=pl+(*(pidn++));
   u+=1;

   _vector_add(psi,rs.s.c1,rs.s.c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c1,chi);
   _vector_add_assign((*sm).c4,chi);

   _vector_sub(psi,rs.s.c2,rs.s.c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c2,chi);
   _vector_sub_assign((*sm).c3,chi);

/******************************* direction +3 *********************************/

   sp=pl+(*(piup));
   u+=1;

   _vector_i_sub(psi,rs.s.c1,rs.s.c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c1,chi);
   _vector_i_add_assign((*sp).c3,chi);

   _vector_i_add(psi,rs.s.c2,rs.s.c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c2,chi);
   _vector_i_sub_assign((*sp).c4,chi);

/******************************* direction -3 *********************************/

   sm=pl+(*(pidn));
   u+=1;

   _vector_i_add(psi,rs.s.c1,rs.s.c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c1,chi);
   _vector_i_sub_assign((*sm).c3,chi);

   _vector_i_sub(psi,rs.s.c2,rs.s.c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c2,chi);
   _vector_i_add_assign((*sm).c4,chi);
}


void Dw_xsf_dble(double mu,spinor_dble *s,spinor_dble *r)
{
   int ix,t,io;
   int *piup,*pidn;
   su3_dble *u,*um;
   pauli_dble *m;
   spin_t *so,*ro;
   tm_parms_t tm;
   xsf_parms_t xsf;

   cpsd_int_bnd_xsf(0x1,s);   
   m=swdfld();
   xsf=xsf_parms();
   apply_sw_xsf_dble(VOLUME/2,xsf.tau3,mu,m,s,r);
   io=query_flags(SWD_O_INVERTED);
   set_sd2zero(BNDRY/2,r+VOLUME);
   tm=tm_parms();
   if (tm.eoflg==1)
      mu=0.0;
    
   coe=-0.5;
   ceo=-0.5;
   piup=iup[VOLUME/2];
   pidn=idn[VOLUME/2];

   so=(spin_t*)(s+(VOLUME/2));
   ro=(spin_t*)(r+(VOLUME/2));
   m+=VOLUME;
   u=udfld();
   um=u+4*VOLUME;

   if ((cpr[0]==0)||(cpr[0]==(NPROC0-1)))
   {
      ix=VOLUME/2;
      
      for (;u<um;u+=8)
      {
         t=global_time(ix);
         ix+=1;

         if ((t>0)&&(t<(NPROC0*L0-1)))
         {
            doe(piup,pidn,u,s);
      
            mul_pauli_dble(mu,m,(*so).w,(*ro).w);
            mul_pauli_dble(-mu,m+1,(*so).w+1,(*ro).w+1);      

            _vector_add_assign((*ro).s.c1,rs.s.c1);
            _vector_add_assign((*ro).s.c2,rs.s.c2);
            _vector_add_assign((*ro).s.c3,rs.s.c3);
            _vector_add_assign((*ro).s.c4,rs.s.c4);
            rs=(*so);
      
            deo(piup,pidn,u,r);
         }
         else
         {
            doe_bnd(t,piup,pidn,u,s);
      
            mul_spinor_bnd_dble(t,xsf.tau3,io,mu,&(so->s),&(ro->s));

            _vector_add_assign((*ro).s.c1,rs.s.c1);
            _vector_add_assign((*ro).s.c2,rs.s.c2);
            _vector_add_assign((*ro).s.c3,rs.s.c3);
            _vector_add_assign((*ro).s.c4,rs.s.c4);
            rs=(*so);
      
            deo_bnd(t,piup,pidn,u,r);
         }

         piup+=4;
         pidn+=4;
         so+=1;
         ro+=1;
         m+=2;
      }
   }
   else
   {
      for (;u<um;u+=8)
      {
         doe(piup,pidn,u,s);
      
         mul_pauli_dble(mu,m,(*so).w,(*ro).w);
         mul_pauli_dble(-mu,m+1,(*so).w+1,(*ro).w+1);      

         _vector_add_assign((*ro).s.c1,rs.s.c1);
         _vector_add_assign((*ro).s.c2,rs.s.c2);
         _vector_add_assign((*ro).s.c3,rs.s.c3);
         _vector_add_assign((*ro).s.c4,rs.s.c4);
         rs=(*so);
      
         deo(piup,pidn,u,r);

         piup+=4;
         pidn+=4;
         so+=1;
         ro+=1;
         m+=2;
      }
   }

   cpsd_ext_bnd_xsf(0x1,r);
}


void Dwee_xsf_dble(double mu,spinor_dble *s,spinor_dble *r)
{
   int ix,t,ie;
   pauli_dble *m,*mm;
   spin_t *se,*re;
   xsf_parms_t xsf;

   m=swdfld();
   mm=m+VOLUME;
   se=(spin_t*)(s);
   re=(spin_t*)(r); 

   xsf=xsf_parms();
   ie=query_flags(SWD_E_INVERTED);

   if ((cpr[0]==0)||(cpr[0]==(NPROC0-1)))
   {
      ix=0;

      for (;m<mm;m+=2)
      {
         t=global_time(ix);
         ix+=1;

         if ((t>0)&&(t<(NPROC0*L0-1)))
         {
            mul_pauli_dble(mu,m,(*se).w,(*re).w);
            mul_pauli_dble(-mu,m+1,(*se).w+1,(*re).w+1);             
         }
         else
            mul_spinor_bnd_dble(t,xsf.tau3,ie,mu,&(se->s),&(re->s));

         se+=1;
         re+=1;
      }
   }
   else
   {
      for (;m<mm;m+=2)
      {
         mul_pauli_dble(mu,m,(*se).w,(*re).w);
         mul_pauli_dble(-mu,m+1,(*se).w+1,(*re).w+1);             

         se+=1;
         re+=1;
      }
   }
}


/* 
   CHECK HERE:
   We have to invert it when the SW is inverted
*/


void Dwoo_xsf_dble(double mu,spinor_dble *s,spinor_dble *r)
{
   int ix,t,io;
   pauli_dble *m,*mm;
   spin_t *so,*ro;
   tm_parms_t tm;
   xsf_parms_t xsf;

   m=swdfld()+VOLUME;
   mm=m+VOLUME;
   so=(spin_t*)(s+(VOLUME/2));
   ro=(spin_t*)(r+(VOLUME/2));
   io=query_flags(SWD_O_INVERTED);
   xsf=xsf_parms();
   tm=tm_parms();
   if (tm.eoflg==1)
      mu=0.0;


   if ((cpr[0]==0)||(cpr[0]==(NPROC0-1)))
   {
      ix=VOLUME/2;

      for (;m<mm;m+=2)
      {
         t=global_time(ix);
         ix+=1;

         if ((t>0)&&(t<(NPROC0*L0-1)))
         {
            mul_pauli_dble(mu,m,(*so).w,(*ro).w);
            mul_pauli_dble(-mu,m+1,(*so).w+1,(*ro).w+1);             
         }
         else
            mul_spinor_bnd_dble(t,xsf.tau3,io,mu,&(so->s),&(ro->s));

         so+=1;
         ro+=1;
      }
   }
   else
   {
      for (;m<mm;m+=2)
      {
         mul_pauli_dble(mu,m,(*so).w,(*ro).w);
         mul_pauli_dble(-mu,m+1,(*so).w+1,(*ro).w+1);             

         so+=1;
         ro+=1;
      }
   }
}


void Dwoe_xsf_dble(spinor_dble *s,spinor_dble *r)
{
   int ix,t;
   int *piup,*pidn;
   su3_dble *u,*um;
   spin_t *ro;

   cpsd_int_bnd_xsf(0x1,s);   

   coe=-0.5;
   piup=iup[VOLUME/2];
   pidn=idn[VOLUME/2];

   ro=(spin_t*)(r+(VOLUME/2));   
   u=udfld();
   um=u+4*VOLUME;

   if ((cpr[0]==0)||(cpr[0]==(NPROC0-1)))
   {
      ix=VOLUME/2;
      
      for (;u<um;u+=8)
      {
         t=global_time(ix);
         ix+=1;

         if ((t>0)&&(t<(NPROC0*L0-1)))
         {
            doe(piup,pidn,u,s);
            (*ro)=rs;
         }
         else
         {
            doe_bnd(t,piup,pidn,u,s);
            (*ro)=rs;
         }

         piup+=4;
         pidn+=4;
         ro+=1;
      }
   }
   else
   {
      for (;u<um;u+=8)
      {
         doe(piup,pidn,u,s);
         (*ro)=rs;

         piup+=4;
         pidn+=4;
         ro+=1;
      }
   }
}


void Dweo_xsf_dble(spinor_dble *s,spinor_dble *r)
{
   int ix,t;
   int *piup,*pidn;
   su3_dble *u,*um;
   spin_t *so;

   set_sd2zero(BNDRY/2,r+VOLUME);

   ceo=0.5;
   piup=iup[VOLUME/2];
   pidn=idn[VOLUME/2];

   so=(spin_t*)(s+(VOLUME/2));
   u=udfld();
   um=u+4*VOLUME;

   if ((cpr[0]==0)||(cpr[0]==(NPROC0-1)))
   {
      ix=VOLUME/2;
      
      for (;u<um;u+=8)
      {
         t=global_time(ix);
         ix+=1;

         if ((t>0)&&(t<(NPROC0*L0-1)))
         {
            rs=(*so);
            deo(piup,pidn,u,r);
         }
         else
         {
            rs=(*so);
            deo_bnd(t,piup,pidn,u,r);
         }

         piup+=4;
         pidn+=4;
         so+=1;
      }
   }
   else
   {
      for (;u<um;u+=8)
      {
         rs=(*so);
         deo(piup,pidn,u,r);

         piup+=4;
         pidn+=4;
         so+=1;
      }
   }

   cpsd_ext_bnd_xsf(0x1,r);
}


void Dwhat_xsf_dble(double mu,spinor_dble *s,spinor_dble *r)
{
   int ix,t,io;
   int *piup,*pidn;
   su3_dble *u,*um;
   pauli_dble *m;
   xsf_parms_t xsf;

   cpsd_int_bnd_xsf(0x1,s);   
   m=swdfld();
   xsf=xsf_parms();
   apply_sw_xsf_dble(VOLUME/2,xsf.tau3,mu,m,s,r);
   io=query_flags(SWD_O_INVERTED);
   set_sd2zero(BNDRY/2,r+VOLUME);

   coe=-0.5;
   ceo=0.5;
   piup=iup[VOLUME/2];
   pidn=idn[VOLUME/2];

   m+=VOLUME;
   u=udfld();
   um=u+4*VOLUME;

   if ((cpr[0]==0)||(cpr[0]==(NPROC0-1)))
   {
      ix=VOLUME/2;
      
      for (;u<um;u+=8)
      {
         t=global_time(ix);
         ix+=1;

         if ((t>0)&&(t<(NPROC0*L0-1)))
         {
            doe(piup,pidn,u,s);
      
            mul_pauli_dble(0.0,m,rs.w,rs.w);
            mul_pauli_dble(0.0,m+1,rs.w+1,rs.w+1);      
      
            deo(piup,pidn,u,r);
         }
         else
         {
            doe_bnd(t,piup,pidn,u,s);

            mul_spinor_bnd_dble(t,xsf.tau3,io,0.0,&(rs.s),&(rs.s));

            deo_bnd(t,piup,pidn,u,r);
         }

         piup+=4;
         pidn+=4;
         m+=2;
      }
   }
   else
   {
      for (;u<um;u+=8)
      {
         doe(piup,pidn,u,s);
      
         mul_pauli_dble(0.0,m,rs.w,rs.w);
         mul_pauli_dble(0.0,m+1,rs.w+1,rs.w+1);      
      
         deo(piup,pidn,u,r);

         piup+=4;
         pidn+=4;
         m+=2;
      }
   }

   cpsd_ext_bnd_xsf(0x1,r);
}

