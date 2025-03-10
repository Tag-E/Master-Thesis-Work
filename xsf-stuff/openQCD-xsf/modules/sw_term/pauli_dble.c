
/*******************************************************************************
*
* File pauli_dble.c
*
* Copyright (C) 2005, 2009, 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Basic functions for double-precision Hermitian 6x6 matrices
*
* The externally accessible functions are
*
*   void mul_pauli_dble(double mu,pauli_dble *m,weyl_dble *s,weyl_dble *r)
*     Multiplies the Weyl spinor s by the matrix m+i*mu and assigns the 
*     result to the Weyl spinor r. The source spinor is overwritten if 
*     r=s and otherwise left unchanged.
* 
*  void mul_spinor_bnd_dble(int t,int tau3,int inv,double mu,spinor_dble *s,
*                          spinor_dble *r)
*     Multiplies the spinor s by the diagonal components of the XSF Dirac
*     operator with flavour structure given by tau3 at the boundaries t=0, 
*     and t=NPROC0*L0-1. The result is assigned to the spinor r. The source
*     spinor is overwritten if r=s and otherwise left unchanged. If inv=1 
*     the inverse boundary operator is applied.
*
*   int inv_pauli_dble(double mu,pauli_dble *m,pauli_dble *im)
*     Assigns the Hermitian part of the matrix (m+i*mu)^(-1) to im. The
*     matrix is overwritten if im=m and otherwise left unchanged. On
*     exit the program returns 0 or 1 depending on whether the inversion
*     was safe or not (in which case the calculated matrix is unusable).
*
*   complex_dble det_pauli_dble(double mu,pauli_dble *m)
*     Returns the determinant of the matrix m+i*mu.
*
*   void apply_sw_dble(int vol,double mu,pauli_dble *m,spinor_dble *s,
*                      spinor_dble *r)
*     Applies the matrix field m[2*vol]+i*mu*gamma_5 to the spinor field
*     s[vol] and assigns the result to the field r[vol]. The source field
*     is overwritten if r=s and otherwise left unchanged (the arrays may
*     not overlap in this case).
*
*  void apply_sw_xsf_dble(int vol,int tau3,double mu,pauli_dble *m,
*                         spinor_dble *s,spinor_dble *r)
*     Applies the matrix field m[2*vol]+i*mu*gamma_5 to the spinor field
*     s[vol] and assigns the result to the field r[vol]. The source field 
*     is overwritten if r=s and otherwise left unchanged (the arrays may
*     not overlap in this case). Note that the rountine only works for the
*     even points of the local lattice. At the boundaries t=0, and t=NPROC0*L0-1
*     the diagonal components of the XSF Dirac operator with flavour structure 
*     defined by tau3 are applied. These components are automatically inverted
*     if the SW term has been inverted on the even points of the lattice.
*
*   int apply_swinv_dble(int vol,double mu,pauli_dble *m,spinor_dble *s,
*                        spinor_dble *r)
*     Applies the inverse of the matrix field m[2*vol]+i*mu*gamma_5 to the
*     spinor field s[vol] and assigns the result to the field r[vol]. The 
*     source field is overwritten if r=s and otherwise left unchanged (the 
*     arrays may not overlap in this case). On exit the program returns 0 
*     or 1 depending on whether the matrix inversions were safe or not (in 
*     the latter case, the output field is unusable).
*
* Notes:
*
* The storage format for Hermitian 6x6 matrices is described in the notes
* "Implementation of the lattice Dirac operator" (file doc/dirac.pdf). As
* explained there, the inversion of a complex matrix is considered to be
* safe if and only if the Frobenius condition number of the matrix is less
* than 100. The Hermitian part of any complex matrix M is defined to be
* (M+M^dag)/2.
*
* Note that the program apply_swinv_dble() performs matrix inversions on
* the fly and is therefore much slower than apply_sw_dble(). When calling
* inv_pauli_dble() or apply_swinv_dble(), it is up to the calling program
* to check the return value and to take the appropriate action when there
* were unsafe matrix inversions.
*
* The programs perform no communications and can be called locally. If SSE
* instructions are used, the Pauli matrices and Dirac spinors fields must be
* aligned to a 16 byte boundary.
*
*******************************************************************************/

#define PAULI_DBLE_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "su3.h"
#include "linalg.h"
#include "global.h"
#include "lattice.h"
#include "sw_term.h"

#define DELTA 1.0e-04

typedef union
{
   spinor_dble s;
   weyl_dble w[2];
   complex_dble c[12];
} spin_t;

static double rr[5] ALIGNED16;
static complex_dble aa[36],cc[6],dd[6] ALIGNED16;

#if (defined x64)
#include "sse2.h"

void mul_pauli_dble(double mu,pauli_dble *m,weyl_dble *s,weyl_dble *r)
{
   m+=2;
   _prefetch_pauli_dble(m);
   m-=2;

   __asm__ __volatile__ ("movddup %0, %%xmm10 \n\t"
                         "movapd %1, %%xmm13 \n\t"
                         "movapd %%xmm10, %%xmm11 \n\t"
                         "movapd %%xmm13, %%xmm14 \n\t"
                         "movapd %%xmm10, %%xmm12 \n\t"
                         "movapd %%xmm13, %%xmm15"                         
                         :
                         :
                         "m" (mu),
                         "m" (_sse_sgn2_dble)
                         :
                         "xmm10", "xmm11", "xmm12", "xmm13",
                         "xmm14", "xmm15");

   __asm__ __volatile__ ("movapd %0, %%xmm0 \n\t"
                         "movapd %1, %%xmm1 \n\t"
                         "movapd %2, %%xmm2 \n\t"
                         "movapd %3, %%xmm3 \n\t"
                         "movapd %4, %%xmm4 \n\t"
                         "movapd %5, %%xmm5 \n\t"
                         "mulpd %%xmm10, %%xmm0 \n\t"
                         "mulpd %%xmm11, %%xmm1 \n\t"
                         "mulpd %%xmm12, %%xmm2 \n\t"
                         "mulpd %%xmm10, %%xmm3 \n\t"
                         "mulpd %%xmm11, %%xmm4 \n\t"
                         "mulpd %%xmm12, %%xmm5"                         
                         :
                         :
                         "m" ((*s).c1.c1),
                         "m" ((*s).c1.c2),
                         "m" ((*s).c1.c3),
                         "m" ((*s).c2.c1),
                         "m" ((*s).c2.c2),
                         "m" ((*s).c2.c3)                         
                         :
                         "xmm0", "xmm1", "xmm2", "xmm3",
                         "xmm4", "xmm5");
   
   __asm__ __volatile__ ("movddup %0, %%xmm6 \n\t"
                         "movddup %1, %%xmm7 \n\t"
                         "movddup %2, %%xmm8 \n\t"
                         "movddup %3, %%xmm9 \n\t"
                         "movddup %4, %%xmm10 \n\t"
                         "movddup %5, %%xmm11"
                         :
                         :
                         "m" ((*m).u[7]),
                         "m" ((*m).u[17]),
                         "m" ((*m).u[25]),
                         "m" ((*m).u[31]),
                         "m" ((*m).u[35]),
                         "m" ((*m).u[15])
                         :
                         "xmm6", "xmm7", "xmm8", "xmm9",
                         "xmm10", "xmm11");

   s+=2;
   _prefetch_weyl_dble(s);
   s-=2;
   
   __asm__ __volatile__ ("mulpd %0, %%xmm6 \n\t"
                         "mulpd %1, %%xmm7 \n\t"
                         "mulpd %2, %%xmm8 \n\t"
                         "mulpd %3, %%xmm9 \n\t"
                         "mulpd %4, %%xmm10 \n\t"
                         "mulpd %5, %%xmm11 \n\t"
                         "addpd %%xmm6, %%xmm0 \n\t"
                         "addpd %%xmm7, %%xmm1 \n\t"
                         "addpd %%xmm8, %%xmm2 \n\t"
                         "addpd %%xmm9, %%xmm3 \n\t"
                         "addpd %%xmm10, %%xmm4 \n\t"
                         "subpd %%xmm11, %%xmm5"
                         :
                         :
                         "m" ((*s).c1.c2),
                         "m" ((*s).c1.c3),
                         "m" ((*s).c2.c1),
                         "m" ((*s).c2.c2),
                         "m" ((*s).c2.c3),
                         "m" ((*s).c1.c1)
                         :
                         "xmm0", "xmm1", "xmm2", "xmm3",
                         "xmm4", "xmm5", "xmm6", "xmm7",
                         "xmm8", "xmm9", "xmm10", "xmm11");

   __asm__ __volatile__ ("movddup %0, %%xmm6 \n\t"
                         "movddup %1, %%xmm7 \n\t"
                         "movddup %2, %%xmm8 \n\t"
                         "movddup %3, %%xmm9 \n\t"
                         "movddup %4, %%xmm10 \n\t"
                         "movddup %5, %%xmm11"
                         :
                         :
                         "m" ((*m).u[9]),
                         "m" ((*m).u[19]),
                         "m" ((*m).u[27]),
                         "m" ((*m).u[33]),
                         "m" ((*m).u[13]),
                         "m" ((*m).u[23])
                         :
                         "xmm6", "xmm7", "xmm8", "xmm9",
                         "xmm10", "xmm11");

   r+=1;
   _prefetch_weyl_dble(r);
   r-=1;
   
   __asm__ __volatile__ ("mulpd %0, %%xmm6 \n\t"
                         "mulpd %1, %%xmm7 \n\t"
                         "mulpd %2, %%xmm8 \n\t"
                         "mulpd %3, %%xmm9 \n\t"
                         "mulpd %4, %%xmm10 \n\t"
                         "mulpd %5, %%xmm11 \n\t"
                         "addpd %%xmm6, %%xmm0 \n\t"
                         "addpd %%xmm7, %%xmm1 \n\t"
                         "addpd %%xmm8, %%xmm2 \n\t"
                         "addpd %%xmm9, %%xmm3 \n\t"
                         "subpd %%xmm10, %%xmm4 \n\t"
                         "subpd %%xmm11, %%xmm5"
                         :
                         :
                         "m" ((*s).c1.c3),
                         "m" ((*s).c2.c1),
                         "m" ((*s).c2.c2),
                         "m" ((*s).c2.c3),
                         "m" ((*s).c1.c1),
                         "m" ((*s).c1.c2)
                         :
                         "xmm0", "xmm1", "xmm2", "xmm3",
                         "xmm4", "xmm5", "xmm6", "xmm7",
                         "xmm8", "xmm9", "xmm10", "xmm11");

   __asm__ __volatile__ ("movddup %0, %%xmm6 \n\t"
                         "movddup %1, %%xmm7 \n\t"
                         "movddup %2, %%xmm8 \n\t"
                         "movddup %3, %%xmm9 \n\t"
                         "movddup %4, %%xmm10 \n\t"
                         "movddup %5, %%xmm11"
                         :
                         :
                         "m" ((*m).u[11]),
                         "m" ((*m).u[21]),
                         "m" ((*m).u[29]),
                         "m" ((*m).u[11]),
                         "m" ((*m).u[21]),
                         "m" ((*m).u[29])
                         :
                         "xmm6", "xmm7", "xmm8", "xmm9",
                         "xmm10", "xmm11");
   
   __asm__ __volatile__ ("mulpd %0, %%xmm6 \n\t"
                         "mulpd %1, %%xmm7 \n\t"
                         "mulpd %2, %%xmm8 \n\t"
                         "mulpd %3, %%xmm9 \n\t"
                         "mulpd %4, %%xmm10 \n\t"
                         "mulpd %5, %%xmm11 \n\t"
                         "addpd %%xmm6, %%xmm0 \n\t"
                         "addpd %%xmm7, %%xmm1 \n\t"
                         "addpd %%xmm8, %%xmm2 \n\t"
                         "subpd %%xmm9, %%xmm3 \n\t"
                         "subpd %%xmm10, %%xmm4 \n\t"
                         "subpd %%xmm11, %%xmm5"
                         :
                         :
                         "m" ((*s).c2.c1),
                         "m" ((*s).c2.c2),
                         "m" ((*s).c2.c3),
                         "m" ((*s).c1.c1),
                         "m" ((*s).c1.c2),
                         "m" ((*s).c1.c3)
                         :
                         "xmm0", "xmm1", "xmm2", "xmm3",
                         "xmm4", "xmm5", "xmm6", "xmm7",
                         "xmm8", "xmm9", "xmm10", "xmm11");
   
   __asm__ __volatile__ ("movddup %0, %%xmm6 \n\t"
                         "movddup %1, %%xmm7 \n\t"
                         "movddup %2, %%xmm8 \n\t"
                         "movddup %3, %%xmm9 \n\t"
                         "movddup %4, %%xmm10 \n\t"
                         "movddup %5, %%xmm11"
                         :
                         :
                         "m" ((*m).u[13]),
                         "m" ((*m).u[23]),
                         "m" ((*m).u[9]),
                         "m" ((*m).u[19]),
                         "m" ((*m).u[27]),
                         "m" ((*m).u[33])
                         :
                         "xmm6", "xmm7", "xmm8", "xmm9",
                         "xmm10", "xmm11");
   
   __asm__ __volatile__ ("mulpd %0, %%xmm6 \n\t"
                         "mulpd %1, %%xmm7 \n\t"
                         "mulpd %2, %%xmm8 \n\t"
                         "mulpd %3, %%xmm9 \n\t"
                         "mulpd %4, %%xmm10 \n\t"
                         "mulpd %5, %%xmm11 \n\t"
                         "addpd %%xmm6, %%xmm0 \n\t"
                         "addpd %%xmm7, %%xmm1 \n\t"
                         "subpd %%xmm8, %%xmm2 \n\t"
                         "subpd %%xmm9, %%xmm3 \n\t"
                         "subpd %%xmm10, %%xmm4 \n\t"
                         "subpd %%xmm11, %%xmm5"
                         :
                         :
                         "m" ((*s).c2.c2),
                         "m" ((*s).c2.c3),
                         "m" ((*s).c1.c1),
                         "m" ((*s).c1.c2),
                         "m" ((*s).c1.c3),
                         "m" ((*s).c2.c1)
                         :
                         "xmm0", "xmm1", "xmm2", "xmm3",
                         "xmm4", "xmm5", "xmm6", "xmm7",
                         "xmm8", "xmm9", "xmm10", "xmm11");

   __asm__ __volatile__ ("movddup %0, %%xmm6 \n\t"
                         "movddup %1, %%xmm7 \n\t"
                         "movddup %2, %%xmm8 \n\t"
                         "movddup %3, %%xmm9 \n\t"
                         "movddup %4, %%xmm10 \n\t"
                         "movddup %5, %%xmm11"
                         :
                         :
                         "m" ((*m).u[15]),
                         "m" ((*m).u[7]),
                         "m" ((*m).u[17]),
                         "m" ((*m).u[25]),
                         "m" ((*m).u[31]),
                         "m" ((*m).u[35])
                         :
                         "xmm6", "xmm7", "xmm8", "xmm9",
                         "xmm10", "xmm11");
   
   __asm__ __volatile__ ("mulpd %0, %%xmm6 \n\t"
                         "mulpd %1, %%xmm7 \n\t"
                         "mulpd %2, %%xmm8 \n\t"
                         "mulpd %3, %%xmm9 \n\t"
                         "mulpd %4, %%xmm10 \n\t"
                         "mulpd %5, %%xmm11 \n\t"
                         "addpd %%xmm6, %%xmm0 \n\t"
                         "subpd %%xmm7, %%xmm1 \n\t"
                         "subpd %%xmm8, %%xmm2 \n\t"
                         "subpd %%xmm9, %%xmm3 \n\t"
                         "subpd %%xmm10, %%xmm4 \n\t"
                         "subpd %%xmm11, %%xmm5"
                         :
                         :
                         "m" ((*s).c2.c3),
                         "m" ((*s).c1.c1),
                         "m" ((*s).c1.c2),
                         "m" ((*s).c1.c3),
                         "m" ((*s).c2.c1),
                         "m" ((*s).c2.c2)
                         :
                         "xmm0", "xmm1", "xmm2", "xmm3",
                         "xmm4", "xmm5", "xmm6", "xmm7",
                         "xmm8", "xmm9", "xmm10", "xmm11");

   __asm__ __volatile__ ("mulpd %%xmm13, %%xmm0 \n\t"
                         "mulpd %%xmm14, %%xmm1 \n\t"
                         "mulpd %%xmm15, %%xmm2 \n\t"
                         "mulpd %%xmm13, %%xmm3 \n\t"
                         "mulpd %%xmm14, %%xmm4 \n\t"
                         "mulpd %%xmm15, %%xmm5 \n\t"
                         "shufpd $0x1, %%xmm0, %%xmm0 \n\t"
                         "shufpd $0x1, %%xmm1, %%xmm1 \n\t"
                         "shufpd $0x1, %%xmm2, %%xmm2 \n\t"
                         "shufpd $0x1, %%xmm3, %%xmm3 \n\t"
                         "shufpd $0x1, %%xmm4, %%xmm4 \n\t"
                         "shufpd $0x1, %%xmm5, %%xmm5"
                         :
                         :
                         :
                         "xmm0", "xmm1", "xmm2", "xmm3",
                         "xmm4", "xmm5");

   __asm__ __volatile__ ("movddup %0, %%xmm6 \n\t"
                         "movddup %1, %%xmm7 \n\t"
                         "movddup %2, %%xmm8 \n\t"
                         "movddup %3, %%xmm9 \n\t"
                         "movddup %4, %%xmm10 \n\t"
                         "movddup %5, %%xmm11"
                         :
                         :
                         "m" ((*m).u[0]),
                         "m" ((*m).u[1]),
                         "m" ((*m).u[2]),
                         "m" ((*m).u[3]),
                         "m" ((*m).u[4]),
                         "m" ((*m).u[5])
                         :
                         "xmm6", "xmm7", "xmm8", "xmm9",
                         "xmm10", "xmm11");
   
   __asm__ __volatile__ ("mulpd %0, %%xmm6 \n\t"
                         "mulpd %1, %%xmm7 \n\t"
                         "mulpd %2, %%xmm8 \n\t"
                         "mulpd %3, %%xmm9 \n\t"
                         "mulpd %4, %%xmm10 \n\t"
                         "mulpd %5, %%xmm11 \n\t"
                         "addpd %%xmm6, %%xmm0 \n\t"
                         "addpd %%xmm7, %%xmm1 \n\t"
                         "addpd %%xmm8, %%xmm2 \n\t"
                         "addpd %%xmm9, %%xmm3 \n\t"
                         "addpd %%xmm10, %%xmm4 \n\t"
                         "addpd %%xmm11, %%xmm5"
                         :
                         :
                         "m" ((*s).c1.c1),
                         "m" ((*s).c1.c2),
                         "m" ((*s).c1.c3),
                         "m" ((*s).c2.c1),
                         "m" ((*s).c2.c2),
                         "m" ((*s).c2.c3)
                         :
                         "xmm0", "xmm1", "xmm2", "xmm3",
                         "xmm4", "xmm5", "xmm6", "xmm7",
                         "xmm8", "xmm9", "xmm10", "xmm11");

   __asm__ __volatile__ ("movddup %0, %%xmm6 \n\t"
                         "movddup %1, %%xmm7 \n\t"
                         "movddup %2, %%xmm8 \n\t"
                         "movddup %3, %%xmm9 \n\t"
                         "movddup %4, %%xmm10 \n\t"
                         "movddup %5, %%xmm11"
                         :
                         :
                         "m" ((*m).u[6]),
                         "m" ((*m).u[16]),
                         "m" ((*m).u[24]),
                         "m" ((*m).u[30]),
                         "m" ((*m).u[34]),
                         "m" ((*m).u[14])
                         :
                         "xmm6", "xmm7", "xmm8", "xmm9",
                         "xmm10", "xmm11");
   
   __asm__ __volatile__ ("mulpd %0, %%xmm6 \n\t"
                         "mulpd %1, %%xmm7 \n\t"
                         "mulpd %2, %%xmm8 \n\t"
                         "mulpd %3, %%xmm9 \n\t"
                         "mulpd %4, %%xmm10 \n\t"
                         "mulpd %5, %%xmm11 \n\t"
                         "addpd %%xmm6, %%xmm0 \n\t"
                         "addpd %%xmm7, %%xmm1 \n\t"
                         "addpd %%xmm8, %%xmm2 \n\t"
                         "addpd %%xmm9, %%xmm3 \n\t"
                         "addpd %%xmm10, %%xmm4 \n\t"
                         "addpd %%xmm11, %%xmm5"
                         :
                         :
                         "m" ((*s).c1.c2),
                         "m" ((*s).c1.c3),
                         "m" ((*s).c2.c1),
                         "m" ((*s).c2.c2),
                         "m" ((*s).c2.c3),
                         "m" ((*s).c1.c1)
                         :
                         "xmm0", "xmm1", "xmm2", "xmm3",
                         "xmm4", "xmm5", "xmm6", "xmm7",
                         "xmm8", "xmm9", "xmm10", "xmm11");

   __asm__ __volatile__ ("movddup %0, %%xmm6 \n\t"
                         "movddup %1, %%xmm7 \n\t"
                         "movddup %2, %%xmm8 \n\t"
                         "movddup %3, %%xmm9 \n\t"
                         "movddup %4, %%xmm10 \n\t"
                         "movddup %5, %%xmm11"
                         :
                         :
                         "m" ((*m).u[8]),
                         "m" ((*m).u[18]),
                         "m" ((*m).u[26]),
                         "m" ((*m).u[32]),
                         "m" ((*m).u[12]),
                         "m" ((*m).u[22])
                         :
                         "xmm6", "xmm7", "xmm8", "xmm9",
                         "xmm10", "xmm11");
   
   __asm__ __volatile__ ("mulpd %0, %%xmm6 \n\t"
                         "mulpd %1, %%xmm7 \n\t"
                         "mulpd %2, %%xmm8 \n\t"
                         "mulpd %3, %%xmm9 \n\t"
                         "mulpd %4, %%xmm10 \n\t"
                         "mulpd %5, %%xmm11 \n\t"
                         "addpd %%xmm6, %%xmm0 \n\t"
                         "addpd %%xmm7, %%xmm1 \n\t"
                         "addpd %%xmm8, %%xmm2 \n\t"
                         "addpd %%xmm9, %%xmm3 \n\t"
                         "addpd %%xmm10, %%xmm4 \n\t"
                         "addpd %%xmm11, %%xmm5"
                         :
                         :
                         "m" ((*s).c1.c3),
                         "m" ((*s).c2.c1),
                         "m" ((*s).c2.c2),
                         "m" ((*s).c2.c3),
                         "m" ((*s).c1.c1),
                         "m" ((*s).c1.c2)
                         :
                         "xmm0", "xmm1", "xmm2", "xmm3",
                         "xmm4", "xmm5", "xmm6", "xmm7",
                         "xmm8", "xmm9", "xmm10", "xmm11");

   __asm__ __volatile__ ("movddup %0, %%xmm6 \n\t"
                         "movddup %1, %%xmm7 \n\t"
                         "movddup %2, %%xmm8 \n\t"
                         "movddup %3, %%xmm9 \n\t"
                         "movddup %4, %%xmm10 \n\t"
                         "movddup %5, %%xmm11"
                         :
                         :
                         "m" ((*m).u[10]),
                         "m" ((*m).u[20]),
                         "m" ((*m).u[28]),
                         "m" ((*m).u[10]),
                         "m" ((*m).u[20]),
                         "m" ((*m).u[28])
                         :
                         "xmm6", "xmm7", "xmm8", "xmm9",
                         "xmm10", "xmm11");
   
   __asm__ __volatile__ ("mulpd %0, %%xmm6 \n\t"
                         "mulpd %1, %%xmm7 \n\t"
                         "mulpd %2, %%xmm8 \n\t"
                         "mulpd %3, %%xmm9 \n\t"
                         "mulpd %4, %%xmm10 \n\t"
                         "mulpd %5, %%xmm11 \n\t"
                         "addpd %%xmm6, %%xmm0 \n\t"
                         "addpd %%xmm7, %%xmm1 \n\t"
                         "addpd %%xmm8, %%xmm2 \n\t"
                         "addpd %%xmm9, %%xmm3 \n\t"
                         "addpd %%xmm10, %%xmm4 \n\t"
                         "addpd %%xmm11, %%xmm5"
                         :
                         :
                         "m" ((*s).c2.c1),
                         "m" ((*s).c2.c2),
                         "m" ((*s).c2.c3),
                         "m" ((*s).c1.c1),
                         "m" ((*s).c1.c2),
                         "m" ((*s).c1.c3)
                         :
                         "xmm0", "xmm1", "xmm2", "xmm3",
                         "xmm4", "xmm5", "xmm6", "xmm7",
                         "xmm8", "xmm9", "xmm10", "xmm11");

   __asm__ __volatile__ ("movddup %0, %%xmm6 \n\t"
                         "movddup %1, %%xmm7 \n\t"
                         "movddup %2, %%xmm8 \n\t"
                         "movddup %3, %%xmm9 \n\t"
                         "movddup %4, %%xmm10 \n\t"
                         "movddup %5, %%xmm11"
                         :
                         :
                         "m" ((*m).u[12]),
                         "m" ((*m).u[22]),
                         "m" ((*m).u[8]),
                         "m" ((*m).u[18]),
                         "m" ((*m).u[26]),
                         "m" ((*m).u[32])
                         :
                         "xmm6", "xmm7", "xmm8", "xmm9",
                         "xmm10", "xmm11");

   __asm__ __volatile__ ("mulpd %0, %%xmm6 \n\t"
                         "mulpd %1, %%xmm7 \n\t"
                         "mulpd %2, %%xmm8 \n\t"
                         "mulpd %3, %%xmm9 \n\t"
                         "mulpd %4, %%xmm10 \n\t"
                         "mulpd %5, %%xmm11 \n\t"
                         "addpd %%xmm6, %%xmm0 \n\t"
                         "addpd %%xmm7, %%xmm1 \n\t"
                         "addpd %%xmm8, %%xmm2 \n\t"
                         "addpd %%xmm9, %%xmm3 \n\t"
                         "addpd %%xmm10, %%xmm4 \n\t"
                         "addpd %%xmm11, %%xmm5"
                         :
                         :
                         "m" ((*s).c2.c2),
                         "m" ((*s).c2.c3),
                         "m" ((*s).c1.c1),
                         "m" ((*s).c1.c2),
                         "m" ((*s).c1.c3),
                         "m" ((*s).c2.c1)
                         :
                         "xmm0", "xmm1", "xmm2", "xmm3",
                         "xmm4", "xmm5", "xmm6", "xmm7",
                         "xmm8", "xmm9", "xmm10", "xmm11");

   __asm__ __volatile__ ("movddup %0, %%xmm6 \n\t"
                         "movddup %1, %%xmm7 \n\t"
                         "movddup %2, %%xmm8 \n\t"
                         "movddup %3, %%xmm9 \n\t"
                         "movddup %4, %%xmm10 \n\t"
                         "movddup %5, %%xmm11"
                         :
                         :
                         "m" ((*m).u[14]),
                         "m" ((*m).u[6]),
                         "m" ((*m).u[16]),
                         "m" ((*m).u[24]),
                         "m" ((*m).u[30]),
                         "m" ((*m).u[34])
                         :
                         "xmm6", "xmm7", "xmm8", "xmm9",
                         "xmm10", "xmm11");
   
   __asm__ __volatile__ ("mulpd %0, %%xmm6 \n\t"
                         "mulpd %1, %%xmm7 \n\t"
                         "mulpd %2, %%xmm8 \n\t"
                         "mulpd %3, %%xmm9 \n\t"
                         "mulpd %4, %%xmm10 \n\t"
                         "mulpd %5, %%xmm11 \n\t"
                         "addpd %%xmm6, %%xmm0 \n\t"
                         "addpd %%xmm7, %%xmm1 \n\t"
                         "addpd %%xmm8, %%xmm2 \n\t"
                         "addpd %%xmm9, %%xmm3 \n\t"
                         "addpd %%xmm10, %%xmm4 \n\t"
                         "addpd %%xmm11, %%xmm5"
                         :
                         :
                         "m" ((*s).c2.c3),
                         "m" ((*s).c1.c1),
                         "m" ((*s).c1.c2),
                         "m" ((*s).c1.c3),
                         "m" ((*s).c2.c1),
                         "m" ((*s).c2.c2)
                         :
                         "xmm0", "xmm1", "xmm2", "xmm3",
                         "xmm4", "xmm5", "xmm6", "xmm7",
                         "xmm8", "xmm9", "xmm10", "xmm11");

   __asm__ __volatile__ ("movapd %%xmm0, %0 \n\t"
                         "movapd %%xmm1, %1 \n\t"
                         "movapd %%xmm2, %2 \n\t"
                         "movapd %%xmm3, %3 \n\t"
                         "movapd %%xmm4, %4 \n\t"
                         "movapd %%xmm5, %5"
                         :
                         "=m" ((*r).c1.c1),
                         "=m" ((*r).c1.c2),
                         "=m" ((*r).c1.c3),
                         "=m" ((*r).c2.c1),
                         "=m" ((*r).c2.c2),
                         "=m" ((*r).c2.c3));
}


static int fwd_house(double eps)
{
   int i,j,k,ifail;
   double r1,r2,r3;
   complex_dble z,*ak,*aj;

   ifail=0;

   for (k=0;k<5;k++)
   {
      r1=aa[6*k+k].re*aa[6*k+k].re+aa[6*k+k].im*aa[6*k+k].im;
      r2=sqrt(r1);

      for (j=(k+1);j<6;j++)
         r1+=(aa[6*j+k].re*aa[6*j+k].re+aa[6*j+k].im*aa[6*j+k].im);

      if (r1>=eps)
         r1=sqrt(r1);
      else
      {
         ifail=1;
         r1=1.0;
      }

      if (r2>=(DBL_EPSILON*r1))
      {
         r3=1.0/r2;
         z.re=r3*aa[6*k+k].re;
         z.im=r3*aa[6*k+k].im;
      }
      else
      {
         z.re=1.0;
         z.im=0.0;
      }

      aa[6*k+k].re+=r1*z.re;
      aa[6*k+k].im+=r1*z.im;

      r3=1.0/(r1*(r1+r2));
      rr[k]=r3;
      dd[k].re=-(r1+r2)*r3*z.re;
      dd[k].im= (r1+r2)*r3*z.im;

      for (j=(k+1);j<6;j++)
      {
         __asm__ __volatile__ ("xorpd %%xmm7, %%xmm7"
                               :
                               :
                               :
                               "xmm7");

         ak=aa+6*k+k;
         aj=aa+6*k+j;

         for (i=k;i<6;i++)
         {
            __asm__ __volatile__ ("movddup %0, %%xmm0 \n\t"
                                  "movddup %1, %%xmm1 \n\t"
                                  "mulpd %2, %%xmm0 \n\t"
                                  "mulpd %2, %%xmm1 \n\t"
                                  "addpd %%xmm0, %%xmm7 \n\t"
                                  "mulpd %3, %%xmm1 \n\t"
                                  "shufpd $0x1, %%xmm1, %%xmm1 \n\t"
                                  "addpd %%xmm1, %%xmm7"
                                  :
                                  :
                                  "m" (ak[0].re),
                                  "m" (ak[0].im),
                                  "m" (aj[0]),
                                  "m" (_sse_sgn1_dble)
                                  :
                                  "xmm0", "xmm1", "xmm7");

            ak+=6;
            aj+=6;
         }

         __asm__ __volatile__ ("movddup %0, %%xmm5 \n\t"
                               "mulpd %%xmm5, %%xmm7 \n\t"
                               "movddup %%xmm7, %%xmm6 \n\t"
                               "unpckhpd %%xmm7, %%xmm7 \n\t"
                               "mulpd %1, %%xmm7"
                               :
                               :
                               "m" (rr[k]),
                               "m" (_sse_sgn1_dble)
                               :
                               "xmm5", "xmm6", "xmm7");

         ak=aa+6*k+k;
         aj=aa+6*k+j;

         for (i=k;i<6;i++)
         {
            __asm__ __volatile__ ("movapd %%xmm7, %%xmm5 \n\t"
                                  "movapd %%xmm6, %%xmm4 \n\t"
                                  "mulpd %1, %%xmm5 \n\t"
                                  "mulpd %1, %%xmm4 \n\t"
                                  "shufpd $0x1, %%xmm5, %%xmm5 \n\t"
                                  "subpd %2, %%xmm4 \n\t"
                                  "subpd %%xmm4, %%xmm5 \n\t"
                                  "movapd %%xmm5, %0"
                                  :
                                  "=m" (aj[0])
                                  :
                                  "m" (ak[0]),
                                  "m" (aj[0])
                                  :
                                  "xmm4", "xmm5");

            ak+=6;
            aj+=6;
         }
      }
   }

   r1=aa[35].re*aa[35].re+aa[35].im*aa[35].im;

   if (r1>=eps)
      r1=1.0/r1;
   else
   {
      ifail=1;
      r1=1.0;
   }

   dd[5].re= r1*aa[35].re;
   dd[5].im=-r1*aa[35].im;

   return ifail;
}


static void solv_sys(void)
{
   int i,j,k;

   for (k=5;k>0;k--)
   {
      for (i=(k-1);i>=0;i--)
      {
         __asm__ __volatile__ ("movddup %0, %%xmm6 \n\t"
                               "movddup %1, %%xmm7 \n\t"
                               "mulpd %2, %%xmm6 \n\t"
                               "mulpd %2, %%xmm7 \n\t"
                               "shufpd $0x1, %%xmm6, %%xmm6 \n\t"
                               "addsubpd %%xmm6, %%xmm7"
                               :
                               :
                               "m" (aa[6*i+k].im),
                               "m" (aa[6*i+k].re),
                               "m" (dd[k])
                               :
                               "xmm6", "xmm7");

         for (j=(k-1);j>i;j--)
         {
            __asm__ __volatile__ ("movddup %0, %%xmm0 \n\t"
                                  "movddup %1, %%xmm1 \n\t"
                                  "mulpd %2, %%xmm0 \n\t"
                                  "mulpd %2, %%xmm1 \n\t"
                                  "addpd %%xmm0, %%xmm7 \n\t"
                                  "shufpd $0x1, %%xmm1, %%xmm1 \n\t"
                                  "addsubpd %%xmm1, %%xmm7"
                                  :
                                  :
                                  "m" (aa[6*j+k].re),
                                  "m" (aa[6*j+k].im),
                                  "m" (aa[6*i+j])
                                  :
                                  "xmm0", "xmm1", "xmm7");
         }

         __asm__ __volatile__ ("movddup %%xmm7, %%xmm6 \n\t"
                               "unpckhpd %%xmm7, %%xmm7 \n\t"
                               "mulpd %1, %%xmm7 \n\t"
                               "mulpd %1, %%xmm6 \n\t"
                               "mulpd %2, %%xmm7 \n\t"
                               "shufpd $0x1, %%xmm7, %%xmm7 \n\t"
                               "subpd %%xmm6, %%xmm7 \n\t"
                               "movapd %%xmm7, %0"
                               :
                               "=m" (aa[6*i+k])
                               :
                               "m" (dd[i]),
                               "m" (_sse_sgn1_dble)
                               :
                               "xmm6", "xmm7");
      }
   }
}


static void bck_house(void)
{
   int i,j,k;
   complex_dble z,*d,*a;

   aa[35].re=dd[5].re;
   aa[35].im=dd[5].im;

   for (k=4;k>=0;k--)
   {
      z.re=dd[k].re;
      z.im=dd[k].im;
      dd[k].re=aa[6*k+k].re;
      dd[k].im=aa[6*k+k].im;
      aa[6*k+k].re=z.re;
      aa[6*k+k].im=z.im;

      for (j=(k+1);j<6;j++)
      {
         dd[j].re=aa[6*j+k].re;
         dd[j].im=aa[6*j+k].im;
         aa[6*j+k].re=0.0;
         aa[6*j+k].im=0.0;
      }

      for (i=0;i<6;i+=2)
      {
         __asm__ __volatile__ ("xorpd %%xmm6, %%xmm6 \n\t"
                               "xorpd %%xmm7, %%xmm7"
                               :
                               :
                               :
                               "xmm6", "xmm7");

         d=dd+k;
         a=aa+6*i+k;

         for (j=k;j<6;j++)
         {
            __asm__ __volatile__ ("movddup %0, %%xmm0 \n\t"
                                  "movddup %1, %%xmm1 \n\t"
                                  "movapd %%xmm0, %%xmm2 \n\t"
                                  "movapd %%xmm1, %%xmm3 \n\t"
                                  "mulpd %2, %%xmm0 \n\t"
                                  "mulpd %2, %%xmm1 \n\t"
                                  "mulpd %3, %%xmm2 \n\t"
                                  "mulpd %3, %%xmm3 \n\t"
                                  "addpd %%xmm0, %%xmm6 \n\t"
                                  "shufpd $0x1, %%xmm1, %%xmm1 \n\t"
                                  "addpd %%xmm2, %%xmm7 \n\t"
                                  "shufpd $0x1, %%xmm3, %%xmm3 \n\t"
                                  "addsubpd %%xmm1, %%xmm6 \n\t"
                                  "addsubpd %%xmm3, %%xmm7 \n\t"
                                  :
                                  :
                                  "m" (d[0].re),
                                  "m" (d[0].im),
                                  "m" (a[0]),
                                  "m" (a[6])
                                  :
                                  "xmm0", "xmm1", "xmm2", "xmm3",
                                  "xmm6", "xmm7");

            d+=1;
            a+=1;
         }

         __asm__ __volatile__ ("movddup %0, %%xmm0 \n\t"
                               "mulpd %%xmm0, %%xmm6 \n\t"
                               "mulpd %%xmm0, %%xmm7 \n\t"
                               "movddup %%xmm6, %%xmm4 \n\t"
                               "movddup %%xmm7, %%xmm5 \n\t"
                               "unpckhpd %%xmm6, %%xmm6 \n\t"
                               "unpckhpd %%xmm7, %%xmm7 \n\t"
                               "mulpd %1, %%xmm4 \n\t"
                               "mulpd %1, %%xmm5"
                               :
                               :
                               "m" (rr[k]),
                               "m" (_sse_sgn1_dble)
                               :
                               "xmm0", "xmm4", "xmm5",
                               "xmm6", "xmm7");

         d=dd+k;
         a=aa+6*i+k;

         for (j=k;j<6;j++)
         {
            __asm__ __volatile__ ("movapd %%xmm6, %%xmm2 \n\t"
                                  "movapd %%xmm7, %%xmm3 \n\t"
                                  "movapd %%xmm4, %%xmm0 \n\t"
                                  "movapd %%xmm5, %%xmm1 \n\t"
                                  "mulpd %2, %%xmm2 \n\t"
                                  "mulpd %2, %%xmm3 \n\t"
                                  "mulpd %2, %%xmm0 \n\t"
                                  "mulpd %2, %%xmm1 \n\t"
                                  "shufpd $0x1, %%xmm2, %%xmm2 \n\t"
                                  "shufpd $0x1, %%xmm3, %%xmm3 \n\t"
                                  "addpd %3, %%xmm0 \n\t"
                                  "addpd %4, %%xmm1 \n\t"
                                  "subpd %%xmm2, %%xmm0 \n\t"
                                  "subpd %%xmm3, %%xmm1 \n\t"
                                  "movapd %%xmm0, %0 \n\t"
                                  "movapd %%xmm1, %1"
                                  :
                                  "=m" (a[0]),
                                  "=m" (a[6])
                                  :
                                  "m" (d[0]),
                                  "m" (a[0]),
                                  "m" (a[6])
                                  :
                                  "xmm0", "xmm1", "xmm2", "xmm3");

            d+=1;
            a+=1;
         }
      }
   }
}

#else

static weyl_dble rs;


void mul_pauli_dble(double mu,pauli_dble *m,weyl_dble *s,weyl_dble *r)
{
   double *u;

   u=(*m).u;

   rs.c1.c1.re=
      u[ 0]*(*s).c1.c1.re-   mu*(*s).c1.c1.im+
      u[ 6]*(*s).c1.c2.re-u[ 7]*(*s).c1.c2.im+
      u[ 8]*(*s).c1.c3.re-u[ 9]*(*s).c1.c3.im+
      u[10]*(*s).c2.c1.re-u[11]*(*s).c2.c1.im+
      u[12]*(*s).c2.c2.re-u[13]*(*s).c2.c2.im+
      u[14]*(*s).c2.c3.re-u[15]*(*s).c2.c3.im;

   rs.c1.c1.im=
      u[ 0]*(*s).c1.c1.im+   mu*(*s).c1.c1.re+
      u[ 6]*(*s).c1.c2.im+u[ 7]*(*s).c1.c2.re+
      u[ 8]*(*s).c1.c3.im+u[ 9]*(*s).c1.c3.re+
      u[10]*(*s).c2.c1.im+u[11]*(*s).c2.c1.re+
      u[12]*(*s).c2.c2.im+u[13]*(*s).c2.c2.re+
      u[14]*(*s).c2.c3.im+u[15]*(*s).c2.c3.re;

   rs.c1.c2.re=
      u[ 6]*(*s).c1.c1.re+u[ 7]*(*s).c1.c1.im+
      u[ 1]*(*s).c1.c2.re-   mu*(*s).c1.c2.im+
      u[16]*(*s).c1.c3.re-u[17]*(*s).c1.c3.im+
      u[18]*(*s).c2.c1.re-u[19]*(*s).c2.c1.im+
      u[20]*(*s).c2.c2.re-u[21]*(*s).c2.c2.im+
      u[22]*(*s).c2.c3.re-u[23]*(*s).c2.c3.im;

   rs.c1.c2.im=
      u[ 6]*(*s).c1.c1.im-u[ 7]*(*s).c1.c1.re+
      u[ 1]*(*s).c1.c2.im+   mu*(*s).c1.c2.re+
      u[16]*(*s).c1.c3.im+u[17]*(*s).c1.c3.re+
      u[18]*(*s).c2.c1.im+u[19]*(*s).c2.c1.re+
      u[20]*(*s).c2.c2.im+u[21]*(*s).c2.c2.re+
      u[22]*(*s).c2.c3.im+u[23]*(*s).c2.c3.re;

   rs.c1.c3.re=
      u[ 8]*(*s).c1.c1.re+u[ 9]*(*s).c1.c1.im+
      u[16]*(*s).c1.c2.re+u[17]*(*s).c1.c2.im+
      u[ 2]*(*s).c1.c3.re-   mu*(*s).c1.c3.im+
      u[24]*(*s).c2.c1.re-u[25]*(*s).c2.c1.im+
      u[26]*(*s).c2.c2.re-u[27]*(*s).c2.c2.im+
      u[28]*(*s).c2.c3.re-u[29]*(*s).c2.c3.im;

   rs.c1.c3.im=
      u[ 8]*(*s).c1.c1.im-u[ 9]*(*s).c1.c1.re+
      u[16]*(*s).c1.c2.im-u[17]*(*s).c1.c2.re+
      u[ 2]*(*s).c1.c3.im+   mu*(*s).c1.c3.re+
      u[24]*(*s).c2.c1.im+u[25]*(*s).c2.c1.re+
      u[26]*(*s).c2.c2.im+u[27]*(*s).c2.c2.re+
      u[28]*(*s).c2.c3.im+u[29]*(*s).c2.c3.re;

   rs.c2.c1.re=
      u[10]*(*s).c1.c1.re+u[11]*(*s).c1.c1.im+
      u[18]*(*s).c1.c2.re+u[19]*(*s).c1.c2.im+
      u[24]*(*s).c1.c3.re+u[25]*(*s).c1.c3.im+
      u[ 3]*(*s).c2.c1.re-   mu*(*s).c2.c1.im+
      u[30]*(*s).c2.c2.re-u[31]*(*s).c2.c2.im+
      u[32]*(*s).c2.c3.re-u[33]*(*s).c2.c3.im;

   rs.c2.c1.im=
      u[10]*(*s).c1.c1.im-u[11]*(*s).c1.c1.re+
      u[18]*(*s).c1.c2.im-u[19]*(*s).c1.c2.re+
      u[24]*(*s).c1.c3.im-u[25]*(*s).c1.c3.re+
      u[ 3]*(*s).c2.c1.im+   mu*(*s).c2.c1.re+
      u[30]*(*s).c2.c2.im+u[31]*(*s).c2.c2.re+
      u[32]*(*s).c2.c3.im+u[33]*(*s).c2.c3.re;

   rs.c2.c2.re=
      u[12]*(*s).c1.c1.re+u[13]*(*s).c1.c1.im+
      u[20]*(*s).c1.c2.re+u[21]*(*s).c1.c2.im+
      u[26]*(*s).c1.c3.re+u[27]*(*s).c1.c3.im+
      u[30]*(*s).c2.c1.re+u[31]*(*s).c2.c1.im+
      u[ 4]*(*s).c2.c2.re-   mu*(*s).c2.c2.im+
      u[34]*(*s).c2.c3.re-u[35]*(*s).c2.c3.im;

   rs.c2.c2.im=
      u[12]*(*s).c1.c1.im-u[13]*(*s).c1.c1.re+
      u[20]*(*s).c1.c2.im-u[21]*(*s).c1.c2.re+
      u[26]*(*s).c1.c3.im-u[27]*(*s).c1.c3.re+
      u[30]*(*s).c2.c1.im-u[31]*(*s).c2.c1.re+
      u[ 4]*(*s).c2.c2.im+   mu*(*s).c2.c2.re+
      u[34]*(*s).c2.c3.im+u[35]*(*s).c2.c3.re;

   rs.c2.c3.re=
      u[14]*(*s).c1.c1.re+u[15]*(*s).c1.c1.im+
      u[22]*(*s).c1.c2.re+u[23]*(*s).c1.c2.im+
      u[28]*(*s).c1.c3.re+u[29]*(*s).c1.c3.im+
      u[32]*(*s).c2.c1.re+u[33]*(*s).c2.c1.im+
      u[34]*(*s).c2.c2.re+u[35]*(*s).c2.c2.im+
      u[ 5]*(*s).c2.c3.re-   mu*(*s).c2.c3.im;

   rs.c2.c3.im=
      u[14]*(*s).c1.c1.im-u[15]*(*s).c1.c1.re+
      u[22]*(*s).c1.c2.im-u[23]*(*s).c1.c2.re+
      u[28]*(*s).c1.c3.im-u[29]*(*s).c1.c3.re+
      u[32]*(*s).c2.c1.im-u[33]*(*s).c2.c1.re+
      u[34]*(*s).c2.c2.im-u[35]*(*s).c2.c2.re+
      u[ 5]*(*s).c2.c3.im+   mu*(*s).c2.c3.re;

   (*r)=rs;
}


static int fwd_house(double eps)
{
   int i,j,k,ifail;
   double r1,r2,r3;
   complex_dble z;

   ifail=0;

   for (k=0;k<5;k++)
   {
      r1=aa[6*k+k].re*aa[6*k+k].re+aa[6*k+k].im*aa[6*k+k].im;
      r2=sqrt(r1);

      for (j=(k+1);j<6;j++)
         r1+=(aa[6*j+k].re*aa[6*j+k].re+aa[6*j+k].im*aa[6*j+k].im);

      if (r1>=eps)
         r1=sqrt(r1);
      else
      {
         ifail=1;
         r1=1.0;
      }

      if (r2>=(DBL_EPSILON*r1))
      {
         r3=1.0/r2;
         z.re=r3*aa[6*k+k].re;
         z.im=r3*aa[6*k+k].im;
      }
      else
      {
         z.re=1.0;
         z.im=0.0;
      }

      aa[6*k+k].re+=r1*z.re;
      aa[6*k+k].im+=r1*z.im;

      r3=1.0/(r1*(r1+r2));
      rr[k]=r3;
      dd[k].re=-(r1+r2)*r3*z.re;
      dd[k].im= (r1+r2)*r3*z.im;

      for (j=(k+1);j<6;j++)
      {
         z.re=0.0;
         z.im=0.0;

         for (i=k;i<6;i++)
         {
            z.re+=(aa[6*i+k].re*aa[6*i+j].re+aa[6*i+k].im*aa[6*i+j].im);
            z.im+=(aa[6*i+k].re*aa[6*i+j].im-aa[6*i+k].im*aa[6*i+j].re);
         }

         z.re*=r3;
         z.im*=r3;

         for (i=k;i<6;i++)
         {
            aa[6*i+j].re-=(z.re*aa[6*i+k].re-z.im*aa[6*i+k].im);
            aa[6*i+j].im-=(z.re*aa[6*i+k].im+z.im*aa[6*i+k].re);
         }
      }
   }

   r1=aa[35].re*aa[35].re+aa[35].im*aa[35].im;

   if (r1>=eps)
      r1=1.0/r1;
   else
   {
      ifail=1;
      r1=1.0;
   }

   dd[5].re= r1*aa[35].re;
   dd[5].im=-r1*aa[35].im;

   return ifail;
}


static void solv_sys(void)
{
   int i,j,k;
   complex_dble z;

   for (k=5;k>0;k--)
   {
      for (i=(k-1);i>=0;i--)
      {
         z.re=aa[6*i+k].re*dd[k].re-aa[6*i+k].im*dd[k].im;
         z.im=aa[6*i+k].re*dd[k].im+aa[6*i+k].im*dd[k].re;

         for (j=(k-1);j>i;j--)
         {
            z.re+=(aa[6*i+j].re*aa[6*j+k].re-aa[6*i+j].im*aa[6*j+k].im);
            z.im+=(aa[6*i+j].re*aa[6*j+k].im+aa[6*i+j].im*aa[6*j+k].re);
         }

         aa[6*i+k].re=-dd[i].re*z.re+dd[i].im*z.im;
         aa[6*i+k].im=-dd[i].re*z.im-dd[i].im*z.re;
      }
   }
}


static void bck_house(void)
{
   int i,j,k;
   complex_dble z;

   aa[35].re=dd[5].re;
   aa[35].im=dd[5].im;

   for (k=4;k>=0;k--)
   {
      z.re=dd[k].re;
      z.im=dd[k].im;
      dd[k].re=aa[6*k+k].re;
      dd[k].im=aa[6*k+k].im;
      aa[6*k+k].re=z.re;
      aa[6*k+k].im=z.im;

      for (j=(k+1);j<6;j++)
      {
         dd[j].re=aa[6*j+k].re;
         dd[j].im=aa[6*j+k].im;
         aa[6*j+k].re=0.0;
         aa[6*j+k].im=0.0;
      }

      for (i=0;i<6;i++)
      {
         z.re=0.0;
         z.im=0.0;

         for (j=k;j<6;j++)
         {
            z.re+=(aa[6*i+j].re*dd[j].re-aa[6*i+j].im*dd[j].im);
            z.im+=(aa[6*i+j].re*dd[j].im+aa[6*i+j].im*dd[j].re);
         }

         z.re*=rr[k];
         z.im*=rr[k];

         for (j=k;j<6;j++)
         {
            aa[6*i+j].re-=(z.re*dd[j].re+z.im*dd[j].im);
            aa[6*i+j].im+=(z.re*dd[j].im-z.im*dd[j].re);
         }
      }
   }
}

#endif

/*

   CHECK HERE:
   Note that when the operator is inverted, 
   mu is not applied anymore.
   This operator now can have the same vector
   as s and r

*/

static spinor_dble rt;


void mul_spinor_bnd_dble(int t,int tau3,int inv,double mu,spinor_dble *s,spinor_dble *r)
{
   double c,tau3d;
   complex_dble z;
   sw_parms_t sw;
   lat_parms_t lat;

   sw=sw_parms();
   lat=lat_parms();
   tau3d=(double)(tau3);

   if (inv==1)
   { 
      c=lat.zF+3.0*lat.dF+sw.m0;
      z.re=1.0/c;
      c=-tau3d*0.5/(c*c);
      z.im=c;
   }
   else
   {
      c=tau3d*0.5;
      z.re=lat.zF+3.0*lat.dF+sw.m0;
      z.im=c+mu;
   }

   _vector_mulc(rt.c1,z,(*s).c1);
   _vector_mulc(rt.c2,z,(*s).c2);
   z.im=-z.im;
   _vector_mulc(rt.c3,z,(*s).c3);
   _vector_mulc(rt.c4,z,(*s).c4);

   if (t==0)
   {
      _vector_mulir_assign(rt.c1, c,(*s).c3);
      _vector_mulir_assign(rt.c2, c,(*s).c4);
      _vector_mulir_assign(rt.c3,-c,(*s).c1);
      _vector_mulir_assign(rt.c4,-c,(*s).c2);
   }
   else
   {
      _vector_mulir_assign(rt.c1,-c,(*s).c3);
      _vector_mulir_assign(rt.c2,-c,(*s).c4);
      _vector_mulir_assign(rt.c3, c,(*s).c1);
      _vector_mulir_assign(rt.c4, c,(*s).c2);
   }

   (*r)=rt;
}


static double set_aa(double mu,pauli_dble *m)
{
   int i,j;
   double sm,*u,*v;

   sm=0.0;
   u=(*m).u;
   v=u+6;

   for (i=0;i<6;i++)
   {
      sm+=u[0]*u[0]+mu*mu;
      aa[6*i+i].re=u[0];
      aa[6*i+i].im=mu;
      u+=1;

      for (j=i+1;j<6;j++)
      {
         sm+=2.0*(v[0]*v[0]+v[1]*v[1]);
         aa[6*i+j].re= v[0];
         aa[6*i+j].im= v[1];
         aa[6*j+i].re= v[0];
         aa[6*j+i].im=-v[1];
         v+=2;
      }
   }

   return sm;
}


static double norm_aa(void)
{
   double sm;
   complex_dble *z,*zm;

   sm=0.0;
   z=aa;
   zm=aa+36;

   for (;z<zm;z++)
      sm+=(*z).re*(*z).re+(*z).im*(*z).im;

   return sm;
}


static void apply_aa(complex_dble *v,complex_dble *w)
{
   cmat_vec_dble(6,aa,v,cc);   

   w[0].re=cc[0].re;
   w[0].im=cc[0].im;
   w[1].re=cc[1].re;
   w[1].im=cc[1].im;
   w[2].re=cc[2].re;
   w[2].im=cc[2].im;   

   w[3].re=cc[3].re;
   w[3].im=cc[3].im;
   w[4].re=cc[4].re;
   w[4].im=cc[4].im;
   w[5].re=cc[5].re;
   w[5].im=cc[5].im;    
}


int inv_pauli_dble(double mu,pauli_dble *m,pauli_dble *im)
{
   int i,j,ifail;
   double eps,sm,*u,*v;
   complex_dble *z,*w;

   eps=DELTA*set_aa(mu,m);
   ifail=fwd_house(eps);
   solv_sys();
   bck_house();

   sm=0.0;
   u=(*im).u;
   v=u+6;

   for (i=0;i<6;i++)
   {
      z=aa+6*i+i;
      w=z;
      sm+=(*z).re*(*z).re+(*z).im*(*z).im;
      u[0]=(*z).re;
      u+=1;

      for (j=i+1;j<6;j++)
      {
         z+=1;
         w+=6;
         sm+=(*z).re*(*z).re+(*z).im*(*z).im;
         sm+=(*w).re*(*w).re+(*w).im*(*w).im;
         v[0]=0.5*((*z).re+(*w).re);
         v[1]=0.5*((*z).im-(*w).im);
         v+=2;
      }
   }

   if ((eps*sm)>1.0)
      ifail=1;

   return ifail;
}


complex_dble det_pauli_dble(double mu,pauli_dble *m)
{
   int i,j,k;
   double eps,r1,r2,r3;
   complex_dble det,z,w;

   eps=DBL_EPSILON*sqrt(set_aa(mu,m));
   det.re=1.0;
   det.im=0.0;

   for (k=0;k<5;k++)
   {
      r1=aa[6*k+k].re*aa[6*k+k].re+aa[6*k+k].im*aa[6*k+k].im;
      r2=sqrt(r1);

      for (j=(k+1);j<6;j++)
         r1+=(aa[6*j+k].re*aa[6*j+k].re+aa[6*j+k].im*aa[6*j+k].im);

      r1=sqrt(r1);
      
      if (r1<=eps)
      {
         w.re=0.0;
         w.im=0.0;

         return w;
      }
      
      if (r2>=(DBL_EPSILON*r1))
      {
         r3=1.0/r2;
         z.re=r1*r3*aa[6*k+k].re;
         z.im=r1*r3*aa[6*k+k].im;
      }
      else
      {
         z.re=r1;
         z.im=0.0;
      }

      w.re=det.re*z.re-det.im*z.im;
      w.im=det.re*z.im+det.im*z.re;
      det.re=w.re;
      det.im=w.im;
      
      aa[6*k+k].re+=z.re;
      aa[6*k+k].im+=z.im;
      r3=1.0/(r1*(r1+r2));

      for (j=(k+1);j<6;j++)
      {
         z.re=0.0;
         z.im=0.0;

         for (i=k;i<6;i++)
         {
            z.re+=(aa[6*i+k].re*aa[6*i+j].re+aa[6*i+k].im*aa[6*i+j].im);
            z.im+=(aa[6*i+k].re*aa[6*i+j].im-aa[6*i+k].im*aa[6*i+j].re);
         }

         z.re*=r3;
         z.im*=r3;

         for (i=(k+1);i<6;i++)
         {
            aa[6*i+j].re-=(z.re*aa[6*i+k].re-z.im*aa[6*i+k].im);
            aa[6*i+j].im-=(z.re*aa[6*i+k].im+z.im*aa[6*i+k].re);
         }
      }
   }

   w.re=det.re*aa[35].re-det.im*aa[35].im;
   w.im=det.re*aa[35].im+det.im*aa[35].re;
      
   return w;
}


void apply_sw_dble(int vol,double mu,pauli_dble *m,spinor_dble *s,
                   spinor_dble *r)
{
   spin_t *ps,*pr,*pm;

   ps=(spin_t*)(s);
   pr=(spin_t*)(r);   
   pm=ps+vol;

   for (;ps<pm;ps++)
   {
      mul_pauli_dble(mu,m,(*ps).w,(*pr).w);
      m+=1;
      mul_pauli_dble(-mu,m,(*ps).w+1,(*pr).w+1);
      m+=1;
      pr+=1;
   }
}


/*
   CHECK HERE:
   This probably does not work for the Dw_blk
   it runs over all even sites of the local lattice
*/


void apply_sw_xsf_dble(int vol,int tau3,double mu,pauli_dble *m,
                   spinor_dble *s,spinor_dble *r)
{
   int ix,t,ie;
   spin_t *ps,*pr,*pm;

   ps=(spin_t*)(s);
   pr=(spin_t*)(r);   
   pm=ps+vol;

   ie=query_flags(SWD_E_INVERTED);

   if ((cpr[0]==0)||(cpr[0]==(NPROC0-1)))
   {
      ix=0;
      
      for (;ps<pm;ps++)
      {
         t=global_time(ix);
         ix+=1;

         if ((t>0)&&(t<(NPROC0*L0-1)))
         {
            mul_pauli_dble(mu,m,(*ps).w,(*pr).w);
            mul_pauli_dble(-mu,m+1,(*ps).w+1,(*pr).w+1);      
         }
         else
            mul_spinor_bnd_dble(t,tau3,ie,mu,&(ps->s),&(pr->s));

         pr+=1;
         m+=2;
      }
   }
   else
   {
      for (;ps<pm;ps++)
      {
         mul_pauli_dble(mu,m,(*ps).w,(*pr).w);
         mul_pauli_dble(-mu,m+1,(*ps).w+1,(*pr).w+1);      

         pr+=1;
         m+=2;
      }
   }
}


int apply_swinv_dble(int vol,double mu,pauli_dble *m,spinor_dble *s,
                     spinor_dble *r)
{
   int ifail;
   double eps;
   spin_t *ps,*pr,*pm;

   ifail=0;
   ps=(spin_t*)(s);
   pr=(spin_t*)(r);   
   pm=ps+vol;

   for (;ps<pm;ps++)
   {
      eps=DELTA*set_aa(mu,m);
      ifail|=fwd_house(eps);
      solv_sys();
      bck_house();      
      if ((eps*norm_aa())>1.0)
         ifail=1;
      apply_aa((*ps).c,(*pr).c);
      m+=1;

      eps=DELTA*set_aa(-mu,m);
      ifail|=fwd_house(eps);
      solv_sys();
      bck_house();      
      if ((eps*norm_aa())>1.0)
         ifail=1;
      apply_aa((*ps).c+6,(*pr).c+6);
      m+=1;
      pr+=1;
   }

   return ifail;
}
