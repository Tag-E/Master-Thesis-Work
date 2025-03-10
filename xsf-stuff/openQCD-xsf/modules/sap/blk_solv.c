
/*******************************************************************************
*
* File blk_solv.c
*
* Copyright (C) 2005, 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Solution of the Dirac equation on the blocks of the SAP_BLOCKS grid 
*
* The externally accessible functions are
*
*   void blk_mres(int n,float mu,int nmr)
*     Depending on whether the twisted-mass flag is set or not, this
*     program approximately solves (Dw+i*mu*gamma_5*1e)*b.s[0]=b.s[1] or
*     (Dw+i*mu*gamma_5)*b.s[0]=b.s[1] on the n'th block b of the SAP_BLOCKS
*     grid. The solution is obtained by applying nmr minimal residual steps,
*     using b.s[2] as workspace. On exit, the approximate solution and its
*     residue are in b.s[0] and b.s[1], respectively.
*
*   void blk_eo_mres(int n,float mu,int nmr)
*     Approximate solution of (Dwhat+i*mu*gamma_5)*b.s[0]=b.s[1] for given
*     b.s[1] on the n'th block b of the SAP_BLOCKS grid. The solution is
*     obtained by applying nmr minimal residual steps, using b.s[2] as
*     workspace. On exit, the approximate solution and its residue are in
*     b.s[0] and b.s[1], respectively, while b.s[0],b.s[1] and b.s[2] are
*     unchanged on the odd points.
*
* Notes:
*
* The twisted-mass flag is retrieved from the parameter data base (see
* flags/lat_parms.c). These programs do not perform any communications and
* can be called locally. It is taken for granted that the SAP_BLOCKS grid
* is allocated and that the gauge field and the SW term on the blocks are
* in the proper condition.
*
*******************************************************************************/

#define BLK_SOLV_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "su3.h"
#include "utils.h"
#include "sflds.h"
#include "linalg.h"
#include "block.h"
#include "dirac.h"
#include "sap.h"

static int vol;
static spinor **s;

#if (defined x64)
#include "sse2.h"

static void scalar_prods(float *r,complex *z)
{
   spinor *s1,*s2,*sm;
   
   __asm__ __volatile__ ("xorpd %%xmm12, %%xmm12 \n\t"
                         "xorpd %%xmm13, %%xmm13 \n\t"
                         "xorpd %%xmm14, %%xmm14"
                         :
                         :
                         :
                         "xmm12", "xmm13", "xmm14");

   s1=s[1];
   s2=s[2];
   sm=s1+vol;

   for (;s1<sm;s1++)
   {
      __asm__ __volatile__ ("movaps %0, %%xmm0 \n\t"
                            "movaps %2, %%xmm1 \n\t"
                            "movaps %4, %%xmm2"
                            :
                            :
                            "m" ((*s2).c1.c1),
                            "m" ((*s2).c1.c2),
                            "m" ((*s2).c1.c3),
                            "m" ((*s2).c2.c1),
                            "m" ((*s2).c2.c2),
                            "m" ((*s2).c2.c3)
                            :
                            "xmm0", "xmm1", "xmm2");

      __asm__ __volatile__ ("movsldup %0, %%xmm4 \n\t"
                            "movsldup %2, %%xmm5 \n\t"
                            "movsldup %4, %%xmm6 \n\t"
                            "movshdup %0, %%xmm8 \n\t"
                            "movshdup %2, %%xmm9 \n\t"
                            "movshdup %4, %%xmm10"
                            :
                            :
                            "m" ((*s1).c1.c1),
                            "m" ((*s1).c1.c2),
                            "m" ((*s1).c1.c3),
                            "m" ((*s1).c2.c1),
                            "m" ((*s1).c2.c2),
                            "m" ((*s1).c2.c3)
                            :
                            "xmm4", "xmm5", "xmm6", "xmm8",
                            "xmm9", "xmm10");

      __asm__ __volatile__ ("mulps %%xmm0, %%xmm4 \n\t"
                            "mulps %%xmm1, %%xmm5 \n\t"
                            "mulps %%xmm2, %%xmm6 \n\t"
                            "mulps %%xmm0, %%xmm8 \n\t"
                            "mulps %%xmm1, %%xmm9 \n\t"
                            "mulps %%xmm2, %%xmm10 \n\t"
                            "mulps %%xmm0, %%xmm0 \n\t"
                            "mulps %%xmm1, %%xmm1 \n\t"
                            "mulps %%xmm2, %%xmm2 \n\t"
                            "addps %%xmm5, %%xmm4 \n\t"
                            "addps %%xmm9, %%xmm8 \n\t"
                            "addps %%xmm1, %%xmm0 \n\t"
                            "addps %%xmm6, %%xmm4 \n\t"
                            "addps %%xmm10, %%xmm8 \n\t"
                            "addps %%xmm2, %%xmm0"
                            :
                            :
                            :
                            "xmm0", "xmm1", "xmm2", "xmm4",
                            "xmm5", "xmm6", "xmm8", "xmm9",
                            "xmm10");

      __asm__ __volatile__ ("movaps %0, %%xmm1 \n\t"
                            "movaps %2, %%xmm2 \n\t"
                            "movaps %4, %%xmm3"
                            :
                            :
                            "m" ((*s2).c3.c1),
                            "m" ((*s2).c3.c2),
                            "m" ((*s2).c3.c3),
                            "m" ((*s2).c4.c1),
                            "m" ((*s2).c4.c2),
                            "m" ((*s2).c4.c3)
                            :
                            "xmm1", "xmm2", "xmm3");

      __asm__ __volatile__ ("movsldup %0, %%xmm5 \n\t"
                            "movsldup %2, %%xmm6 \n\t"
                            "movsldup %4, %%xmm7 \n\t"
                            "movshdup %0, %%xmm9 \n\t"
                            "movshdup %2, %%xmm10 \n\t"
                            "movshdup %4, %%xmm11"                            
                            :
                            :
                            "m" ((*s1).c3.c1),
                            "m" ((*s1).c3.c2),
                            "m" ((*s1).c3.c3),
                            "m" ((*s1).c4.c1),
                            "m" ((*s1).c4.c2),
                            "m" ((*s1).c4.c3)
                            :
                            "xmm5", "xmm6", "xmm7","xmm9",
                            "xmm10", "xmm11");

      __asm__ __volatile__ ("mulps %%xmm1, %%xmm5 \n\t"
                            "mulps %%xmm2, %%xmm6 \n\t"
                            "mulps %%xmm3, %%xmm7 \n\t"
                            "mulps %%xmm1, %%xmm9 \n\t"
                            "mulps %%xmm2, %%xmm10 \n\t"
                            "mulps %%xmm3, %%xmm11 \n\t"
                            "mulps %%xmm1, %%xmm1 \n\t"
                            "mulps %%xmm2, %%xmm2 \n\t"
                            "mulps %%xmm3, %%xmm3 \n\t"
                            "addps %%xmm5, %%xmm4 \n\t"
                            "addps %%xmm9, %%xmm8 \n\t"
                            "addps %%xmm1, %%xmm0 \n\t"
                            "addps %%xmm6, %%xmm4 \n\t"
                            "addps %%xmm10, %%xmm8 \n\t"
                            "addps %%xmm2, %%xmm0 \n\t"
                            "addps %%xmm7, %%xmm4 \n\t"
                            "addps %%xmm11, %%xmm8 \n\t"
                            "addps %%xmm3, %%xmm0"
                            :
                            :
                            :
                            "xmm0", "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5", "xmm6", "xmm7",
                            "xmm8", "xmm9", "xmm10", "xmm11");
      
      __asm__ __volatile__ ("movhlps %%xmm4, %%xmm5 \n\t"
                            "movhlps %%xmm8, %%xmm9 \n\t"
                            "movhlps %%xmm0, %%xmm1 \n\t"
                            "addps %%xmm5, %%xmm4 \n\t"
                            "addps %%xmm9, %%xmm8 \n\t"
                            "addps %%xmm1, %%xmm0 \n\t"
                            "cvtps2pd %%xmm4, %%xmm6 \n\t"
                            "cvtps2pd %%xmm8, %%xmm10 \n\t"
                            "cvtps2pd %%xmm0, %%xmm2 \n\t"
                            "addpd %%xmm6, %%xmm12 \n\t"
                            "addpd %%xmm10, %%xmm13 \n\t"
                            "addpd %%xmm2, %%xmm14"
                            :
                            :
                            :
                            "xmm0", "xmm1", "xmm2", "xmm4",
                            "xmm5", "xmm6", "xmm8", "xmm9",
                            "xmm10", "xmm12", "xmm13", "xmm14");

      s2+=1;
   }

   __asm__ __volatile__ ("shufpd $0x1, %%xmm12, %%xmm12 \n\t"
                         "haddpd %%xmm14, %%xmm14 \n\t"
                         "addsubpd %%xmm12, %%xmm13 \n\t"
                         "cvtsd2ss %%xmm14, %%xmm0 \n\t"                         
                         "shufpd $0x1, %%xmm13, %%xmm13 \n\t"
                         "movss %%xmm0, %0 \n\t"                         
                         "cvtpd2ps %%xmm13, %%xmm1 \n\t"
                         "movlps %%xmm1, %1"
                         :
                         "=m" (*r),
                         "=m" (*z)
                         :
                         :
                         "xmm0", "xmm1", "xmm12", "xmm13",
                         "xmm14");
}


static void linear_cmbs(float *r,complex *z)
{
   spinor *s0,*s1,*s2,*sm;
   
   s0=s[0];
   s1=s[1];
   s2=s[2];
   sm=s0+vol;

   __asm__ __volatile__ ("movss %0, %%xmm4 \n\t"
                         "movss %1, %%xmm5 \n\t"
                         "movss %2, %%xmm6 \n\t"
                         "movss %3, %%xmm7 \n\t"
                         "divss %%xmm4, %%xmm5 \n\t"
                         "mulss %%xmm5, %%xmm6 \n\t"
                         "mulss %%xmm5, %%xmm7 \n\t"                         
                         "shufps $0x0, %%xmm6, %%xmm6 \n\t"
                         "shufps $0x0, %%xmm7, %%xmm7 \n\t"
                         "mulps %1, %%xmm7"
                         :
                         :
                         "m" (*r),
                         "m" (_sse_sgn13),
                         "m" ((*z).re),
                         "m" ((*z).im)
                         :
                         "xmm4", "xmm5", "xmm6", "xmm7");
   
   for (;s0<sm;s0++)
   {
      __asm__ __volatile__ ("movaps %0, %%xmm0 \n\t" 
                            "movaps %2, %%xmm1 \n\t" 
                            "movaps %4, %%xmm2" 
                            : 
                            : 
                            "m" ((*s2).c1.c1), 
                            "m" ((*s2).c1.c2), 
                            "m" ((*s2).c1.c3), 
                            "m" ((*s2).c2.c1), 
                            "m" ((*s2).c2.c2), 
                            "m" ((*s2).c2.c3) 
                            : 
                            "xmm0", "xmm1", "xmm2");  

      __asm__ __volatile__ ("movaps %%xmm0, %%xmm8 \n\t" 
                            "movaps %%xmm1, %%xmm9 \n\t" 
                            "movaps %%xmm2, %%xmm10 \n\t" 
                            "mulps %%xmm6, %%xmm0 \n\t" 
                            "mulps %%xmm6, %%xmm1 \n\t" 
                            "mulps %%xmm6, %%xmm2 \n\t" 
                            "shufps $0xb1, %%xmm8, %%xmm8 \n\t" 
                            "shufps $0xb1, %%xmm9, %%xmm9 \n\t" 
                            "shufps $0xb1, %%xmm10, %%xmm10 \n\t" 
                            "mulps %%xmm7, %%xmm8 \n\t" 
                            "mulps %%xmm7, %%xmm9 \n\t" 
                            "mulps %%xmm7, %%xmm10 \n\t" 
                            "addps %%xmm8, %%xmm0 \n\t" 
                            "addps %%xmm9, %%xmm1 \n\t" 
                            "addps %%xmm10, %%xmm2" 
                            : 
                            : 
                            : 
                            "xmm0", "xmm1", "xmm2", "xmm8",
                            "xmm9", "xmm10"); 

      __asm__ __volatile__ ("movaps %0, %%xmm3 \n\t" 
                            "movaps %2, %%xmm4 \n\t" 
                            "movaps %4, %%xmm5 \n\t"
                            "addps %%xmm3, %%xmm0 \n\t"
                            "addps %%xmm4, %%xmm1 \n\t"
                            "addps %%xmm5, %%xmm2"
                            : 
                            : 
                            "m" ((*s1).c1.c1), 
                            "m" ((*s1).c1.c2), 
                            "m" ((*s1).c1.c3), 
                            "m" ((*s1).c2.c1), 
                            "m" ((*s1).c2.c2), 
                            "m" ((*s1).c2.c3) 
                            : 
                            "xmm0", "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5");  

      __asm__ __volatile__ ("movaps %%xmm3, %%xmm8 \n\t" 
                            "movaps %%xmm4, %%xmm9 \n\t" 
                            "movaps %%xmm5, %%xmm10 \n\t" 
                            "mulps %%xmm6, %%xmm3 \n\t" 
                            "mulps %%xmm6, %%xmm4 \n\t" 
                            "mulps %%xmm6, %%xmm5 \n\t" 
                            "shufps $0xb1, %%xmm8, %%xmm8 \n\t" 
                            "shufps $0xb1, %%xmm9, %%xmm9 \n\t" 
                            "shufps $0xb1, %%xmm10, %%xmm10 \n\t"
                            "movaps %%xmm0, %0 \n\t"
                            "movaps %%xmm1, %2 \n\t"
                            "movaps %%xmm2, %4 \n\t"                            
                            "mulps %%xmm7, %%xmm8 \n\t" 
                            "mulps %%xmm7, %%xmm9 \n\t" 
                            "mulps %%xmm7, %%xmm10" 
                            :
                            "=m" ((*s1).c1.c1), 
                            "=m" ((*s1).c1.c2), 
                            "=m" ((*s1).c1.c3), 
                            "=m" ((*s1).c2.c1), 
                            "=m" ((*s1).c2.c2), 
                            "=m" ((*s1).c2.c3)                             
                            : 
                            : 
                            "xmm3", "xmm4", "xmm5", "xmm8",
                            "xmm9", "xmm10"); 

      __asm__ __volatile__ ("movaps %0, %%xmm11 \n\t" 
                            "movaps %2, %%xmm12 \n\t" 
                            "movaps %4, %%xmm13 \n\t"
                            "addps %%xmm8, %%xmm3 \n\t" 
                            "addps %%xmm9, %%xmm4 \n\t" 
                            "addps %%xmm10, %%xmm5 \n\t" 
                            "subps %%xmm3, %%xmm11 \n\t"
                            "subps %%xmm4, %%xmm12 \n\t"
                            "subps %%xmm5, %%xmm13"
                            : 
                            : 
                            "m" ((*s0).c1.c1), 
                            "m" ((*s0).c1.c2), 
                            "m" ((*s0).c1.c3), 
                            "m" ((*s0).c2.c1), 
                            "m" ((*s0).c2.c2), 
                            "m" ((*s0).c2.c3) 
                            : 
                            "xmm3", "xmm4", "xmm5", "xmm11",
                            "xmm12", "xmm13");  
      
      __asm__ __volatile__ ("movaps %%xmm11, %0 \n\t"
                            "movaps %%xmm12, %2 \n\t"
                            "movaps %%xmm13, %4"
                            :
                            "=m" ((*s0).c1.c1), 
                            "=m" ((*s0).c1.c2), 
                            "=m" ((*s0).c1.c3), 
                            "=m" ((*s0).c2.c1), 
                            "=m" ((*s0).c2.c2), 
                            "=m" ((*s0).c2.c3));

      __asm__ __volatile__ ("movaps %0, %%xmm0 \n\t" 
                            "movaps %2, %%xmm1 \n\t" 
                            "movaps %4, %%xmm2" 
                            : 
                            : 
                            "m" ((*s2).c3.c1), 
                            "m" ((*s2).c3.c2), 
                            "m" ((*s2).c3.c3), 
                            "m" ((*s2).c4.c1), 
                            "m" ((*s2).c4.c2), 
                            "m" ((*s2).c4.c3) 
                            : 
                            "xmm0", "xmm1", "xmm2");  

      __asm__ __volatile__ ("movaps %%xmm0, %%xmm8 \n\t" 
                            "movaps %%xmm1, %%xmm9 \n\t" 
                            "movaps %%xmm2, %%xmm10 \n\t" 
                            "mulps %%xmm6, %%xmm0 \n\t" 
                            "mulps %%xmm6, %%xmm1 \n\t" 
                            "mulps %%xmm6, %%xmm2 \n\t" 
                            "shufps $0xb1, %%xmm8, %%xmm8 \n\t" 
                            "shufps $0xb1, %%xmm9, %%xmm9 \n\t" 
                            "shufps $0xb1, %%xmm10, %%xmm10 \n\t" 
                            "mulps %%xmm7, %%xmm8 \n\t" 
                            "mulps %%xmm7, %%xmm9 \n\t" 
                            "mulps %%xmm7, %%xmm10 \n\t" 
                            "addps %%xmm8, %%xmm0 \n\t" 
                            "addps %%xmm9, %%xmm1 \n\t" 
                            "addps %%xmm10, %%xmm2" 
                            : 
                            : 
                            : 
                            "xmm0", "xmm1", "xmm2", "xmm8",
                            "xmm9", "xmm10"); 

      __asm__ __volatile__ ("movaps %0, %%xmm3 \n\t" 
                            "movaps %2, %%xmm4 \n\t" 
                            "movaps %4, %%xmm5 \n\t"
                            "addps %%xmm3, %%xmm0 \n\t"
                            "addps %%xmm4, %%xmm1 \n\t"
                            "addps %%xmm5, %%xmm2"
                            : 
                            : 
                            "m" ((*s1).c3.c1), 
                            "m" ((*s1).c3.c2), 
                            "m" ((*s1).c3.c3), 
                            "m" ((*s1).c4.c1), 
                            "m" ((*s1).c4.c2), 
                            "m" ((*s1).c4.c3) 
                            : 
                            "xmm0", "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5");  

      __asm__ __volatile__ ("movaps %%xmm3, %%xmm8 \n\t" 
                            "movaps %%xmm4, %%xmm9 \n\t" 
                            "movaps %%xmm5, %%xmm10 \n\t" 
                            "mulps %%xmm6, %%xmm3 \n\t" 
                            "mulps %%xmm6, %%xmm4 \n\t" 
                            "mulps %%xmm6, %%xmm5 \n\t" 
                            "shufps $0xb1, %%xmm8, %%xmm8 \n\t" 
                            "shufps $0xb1, %%xmm9, %%xmm9 \n\t" 
                            "shufps $0xb1, %%xmm10, %%xmm10 \n\t"
                            "movaps %%xmm0, %0 \n\t"
                            "movaps %%xmm1, %2 \n\t"
                            "movaps %%xmm2, %4 \n\t"                            
                            "mulps %%xmm7, %%xmm8 \n\t" 
                            "mulps %%xmm7, %%xmm9 \n\t" 
                            "mulps %%xmm7, %%xmm10" 
                            :
                            "=m" ((*s1).c3.c1), 
                            "=m" ((*s1).c3.c2), 
                            "=m" ((*s1).c3.c3), 
                            "=m" ((*s1).c4.c1), 
                            "=m" ((*s1).c4.c2), 
                            "=m" ((*s1).c4.c3)                             
                            : 
                            : 
                            "xmm3", "xmm4", "xmm5", "xmm8",
                            "xmm9", "xmm10"); 

      __asm__ __volatile__ ("movaps %0, %%xmm11 \n\t" 
                            "movaps %2, %%xmm12 \n\t" 
                            "movaps %4, %%xmm13 \n\t"
                            "addps %%xmm8, %%xmm3 \n\t" 
                            "addps %%xmm9, %%xmm4 \n\t" 
                            "addps %%xmm10, %%xmm5 \n\t" 
                            "subps %%xmm3, %%xmm11 \n\t"
                            "subps %%xmm4, %%xmm12 \n\t"
                            "subps %%xmm5, %%xmm13"
                            : 
                            : 
                            "m" ((*s0).c3.c1), 
                            "m" ((*s0).c3.c2), 
                            "m" ((*s0).c3.c3), 
                            "m" ((*s0).c4.c1), 
                            "m" ((*s0).c4.c2), 
                            "m" ((*s0).c4.c3) 
                            : 
                            "xmm3", "xmm4", "xmm5", "xmm11",
                            "xmm12", "xmm13");  
      
      __asm__ __volatile__ ("movaps %%xmm11, %0 \n\t"
                            "movaps %%xmm12, %2 \n\t"
                            "movaps %%xmm13, %4"
                            :
                            "=m" ((*s0).c3.c1), 
                            "=m" ((*s0).c3.c2), 
                            "=m" ((*s0).c3.c3), 
                            "=m" ((*s0).c4.c1), 
                            "=m" ((*s0).c4.c2), 
                            "=m" ((*s0).c4.c3));
      
      s1+=1;
      s2+=1;
   }
}


void blk_mres(int n,float mu,int nmr)
{
   int nb,isw,imr;
   float r;
   complex z;
   block_t *b;

   b=blk_list(SAP_BLOCKS,&nb,&isw);
   vol=(*b).vol;
   s=(*b).s;

   set_s2zero(vol,s[0]);

   for (imr=0;imr<nmr;imr++)
   {
      Dw_blk(SAP_BLOCKS,n,mu,1,2);
      scalar_prods(&r,&z);

      if (r<(2.0f*FLT_MIN))
         return;

      linear_cmbs(&r,&z); 
   }
}


void blk_eo_mres(int n,float mu,int nmr)
{
   int nb,isw,imr;
   float r;
   complex z;
   block_t *b;

   b=blk_list(SAP_BLOCKS,&nb,&isw);
   vol=(*b).vol/2;
   s=(*b).s;

   set_s2zero(vol,s[0]);

   for (imr=0;imr<nmr;imr++)
   {
      Dwhat_blk(SAP_BLOCKS,n,mu,1,2);
      scalar_prods(&r,&z);

      if (r<(2.0f*FLT_MIN))
         return;

      linear_cmbs(&r,&z);
   }
}

#else

void blk_mres(int n,float mu,int nmr)
{
   int nb,isw,imr;
   float r;
   complex z;
   block_t *b;

   b=blk_list(SAP_BLOCKS,&nb,&isw);
   vol=(*b).vol;
   s=(*b).s;

   set_s2zero(vol,s[0]);

   for (imr=0;imr<nmr;imr++)
   {
      Dw_blk(SAP_BLOCKS,n,mu,1,2);
      r=norm_square(vol,0,s[2]);

      if (r<(2.0f*FLT_MIN))
         return;

      z=spinor_prod(vol,0,s[2],s[1]);

      r=1.0f/r;
      z.re*=r;
      z.im*=r;
      mulc_spinor_add(vol,s[0],s[1],z);

      z.re=-z.re;
      z.im=-z.im;
      mulc_spinor_add(vol,s[1],s[2],z);
   }
}


void blk_eo_mres(int n,float mu,int nmr)
{
   int nb,isw,imr;
   float r;
   complex z;
   block_t *b;

   b=blk_list(SAP_BLOCKS,&nb,&isw);
   vol=(*b).vol/2;
   s=(*b).s;

   set_s2zero(vol,s[0]);

   for (imr=0;imr<nmr;imr++)
   {
      Dwhat_blk(SAP_BLOCKS,n,mu,1,2);
      r=norm_square(vol,0,s[2]);

      if (r<(2.0f*FLT_MIN))
         return;

      z=spinor_prod(vol,0,s[2],s[1]);

      r=1.0f/r;
      z.re*=r;
      z.im*=r;
      mulc_spinor_add(vol,s[0],s[1],z);

      z.re=-z.re;
      z.im=-z.im;
      mulc_spinor_add(vol,s[1],s[2],z);
   }
}

#endif
