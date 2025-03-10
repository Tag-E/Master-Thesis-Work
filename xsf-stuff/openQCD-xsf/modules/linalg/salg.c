
/*******************************************************************************
*
* File salg.c
*
* Copyright (C) 2005, 2007, 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Generic linear algebra routines for single-precision spinor fields
*
* The externally accessible functions are
*
*   complex spinor_prod(int vol,int icom,spinor *s,spinor *r)
*     Computes the scalar product of the fields s and r.
*
*   float spinor_prod_re(int vol,int icom,spinor *s,spinor *r)
*     Computes the real part of the scalar product of the fields
*     s and r.
*
*   float norm_square(int vol,int icom,spinor *s)
*     Computes the square of the norm of the field s.
*
*   void mulc_spinor_add(int vol,spinor *s,spinor *r,complex z)
*     Replaces the field s by s+z*r.
*
*   void mulr_spinor_add(int vol,spinor *s,spinor *r,float c)
*     Replaces the field s by s+c*r.
*
*   void project(int vol,int icom,spinor *s,spinor *r)
*     Replaces the field s by s-(r,s)*r.
*
*   void scale(int vol,float c,spinor *s)
*     Replaces the field s by c*s.
*
*   float normalize(int vol,int icom,spinor *s)
*     Replaces the field s by s/||s|| and returns the norm ||s||.
*
*   void rotate(int vol,int n,spinor **ppk,complex *v)
*     Replaces the fields pk[] by sum_j pj*v[n*j+k] where 0<=k,j<n
*     and pk=ppk[k].
*
*   void mulg5(int vol,spinor *s)
*     Multiplies the field s with gamma_5.
*
*   void mulmg5(int vol,spinor *s)
*     Multiplies the field s with -gamma_5.
*
* Notes:
*
* All these programs act on arrays of spinor fields whose base address
* is passed through the arguments. The length of the arrays is specified 
* by the parameter vol. Scalar products are globally summed if the 
* parameter icom is equal to 1. In this case, the calculated values are
* guaranteed to be exactly the same on all processes.
*
* The programs perform no communications except in the case of the scalar
* products if these are globally summed. If SSE instructions are used, the
* spinor fields must be aligned to a 16 byte boundary.
*
*******************************************************************************/

#define SALG_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "utils.h"
#include "sflds.h"
#include "linalg.h"
#include "global.h"

static int nrot=0,ifail=0;
static spinor *psi;


static void alloc_wrotate(int n)
{
   if (nrot>0)
      afree(psi);
   
   psi=amalloc(n*sizeof(*psi),ALIGN);

   if (psi==NULL)
   {
      error_loc(1,1,"alloc_wrotate [salg.c]",
                "Unable to allocate workspace");
      nrot=0;
      ifail=1;      
   }
   else
   {
      nrot=n;
      set_s2zero(n,psi);
   }
}

#if (defined x64)
#include "sse2.h"

complex spinor_prod(int vol,int icom,spinor *s,spinor *r)
{
   double x,y;
   complex z;
   complex_dble v,w;
   spinor *sm;

   __asm__ __volatile__ ("xorpd %%xmm10, %%xmm10 \n\t"
                         "xorpd %%xmm11, %%xmm11 \n\t"
                         "xorpd %%xmm12, %%xmm12 \n\t"
                         "xorpd %%xmm13, %%xmm13 \n\t"
                         "xorpd %%xmm14, %%xmm14 \n\t"
                         "xorpd %%xmm15, %%xmm15"
                         :
                         :
                         :
                         "xmm10", "xmm11", "xmm12",
                         "xmm13", "xmm14", "xmm15");
   
   sm=s+vol;

   for (;s<sm;s++)
   {
      _sse_spinor_load(*s);
      
      __asm__ __volatile__ ("mulps %0, %%xmm0 \n\t"
                            "mulps %2, %%xmm1 \n\t"
                            "mulps %4, %%xmm2"
                            :
                            :
                            "m" ((*r).c1.c1),
                            "m" ((*r).c1.c2),
                            "m" ((*r).c1.c3),
                            "m" ((*r).c2.c1),
                            "m" ((*r).c2.c2),
                            "m" ((*r).c2.c3)
                            :
                            "xmm0", "xmm1", "xmm2");

      __asm__ __volatile__ ("mulps %0, %%xmm3 \n\t"
                            "mulps %2, %%xmm4 \n\t"
                            "mulps %4, %%xmm5"
                            :
                            :
                            "m" ((*r).c3.c1),
                            "m" ((*r).c3.c2),
                            "m" ((*r).c3.c3),
                            "m" ((*r).c4.c1),
                            "m" ((*r).c4.c2),
                            "m" ((*r).c4.c3)
                            :
                            "xmm3", "xmm4", "xmm5");      

      __asm__ __volatile__ ("addps %%xmm0, %%xmm1 \n\t"
                            "addps %%xmm2, %%xmm3 \n\t"
                            "addps %%xmm4, %%xmm5 \n\t"
                            "cvtps2pd %%xmm1, %%xmm7 \n\t"
                            "cvtps2pd %%xmm3, %%xmm8 \n\t"
                            "cvtps2pd %%xmm5, %%xmm9 \n\t"
                            "shufps $0x4e, %%xmm1, %%xmm1 \n\t"
                            "shufps $0x4e, %%xmm3, %%xmm3 \n\t"
                            "shufps $0x4e, %%xmm5, %%xmm5 \n\t"
                            "cvtps2pd %%xmm1, %%xmm2 \n\t"
                            "cvtps2pd %%xmm3, %%xmm4 \n\t"
                            "cvtps2pd %%xmm5, %%xmm6 \n\t"
                            "addpd %%xmm7, %%xmm10 \n\t"
                            "addpd %%xmm8, %%xmm11 \n\t"
                            "addpd %%xmm9, %%xmm12 \n\t"
                            "addpd %%xmm2, %%xmm10 \n\t"
                            "addpd %%xmm4, %%xmm11 \n\t"
                            "addpd %%xmm6, %%xmm12"
                            :
                            :
                            :
                            "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5", "xmm6",
                            "xmm7", "xmm8", "xmm9",
                            "xmm10", "xmm11", "xmm12");

      _sse_spinor_load(*s);

      __asm__ __volatile__ ("shufps $0xb1, %%xmm0, %%xmm0 \n\t"
                            "shufps $0xb1, %%xmm1, %%xmm1 \n\t"
                            "shufps $0xb1, %%xmm2, %%xmm2 \n\t"
                            "shufps $0xb1, %%xmm3, %%xmm3 \n\t"
                            "shufps $0xb1, %%xmm4, %%xmm4 \n\t"
                            "shufps $0xb1, %%xmm5, %%xmm5"
                            :
                            :
                            :
                            "xmm0", "xmm1", "xmm2",
                            "xmm3", "xmm4", "xmm5");

      __asm__ __volatile__ ("mulps %0, %%xmm0 \n\t"
                            "mulps %2, %%xmm1 \n\t"
                            "mulps %4, %%xmm2"
                            :
                            :
                            "m" ((*r).c1.c1),
                            "m" ((*r).c1.c2),
                            "m" ((*r).c1.c3),
                            "m" ((*r).c2.c1),
                            "m" ((*r).c2.c2),
                            "m" ((*r).c2.c3)
                            :
                            "xmm0", "xmm1", "xmm2");

      __asm__ __volatile__ ("mulps %0, %%xmm3 \n\t"
                            "mulps %2, %%xmm4 \n\t"
                            "mulps %4, %%xmm5"
                            :
                            :
                            "m" ((*r).c3.c1),
                            "m" ((*r).c3.c2),
                            "m" ((*r).c3.c3),
                            "m" ((*r).c4.c1),
                            "m" ((*r).c4.c2),
                            "m" ((*r).c4.c3)
                            :
                            "xmm3", "xmm4", "xmm5");      

      __asm__ __volatile__ ("addps %%xmm0, %%xmm1 \n\t"
                            "addps %%xmm2, %%xmm3 \n\t"
                            "addps %%xmm4, %%xmm5 \n\t"
                            "cvtps2pd %%xmm1, %%xmm7 \n\t"
                            "cvtps2pd %%xmm3, %%xmm8 \n\t"
                            "cvtps2pd %%xmm5, %%xmm9 \n\t"
                            "shufps $0x4e, %%xmm1, %%xmm1 \n\t"
                            "shufps $0x4e, %%xmm3, %%xmm3 \n\t"
                            "shufps $0x4e, %%xmm5, %%xmm5 \n\t"
                            "cvtps2pd %%xmm1, %%xmm2 \n\t"
                            "cvtps2pd %%xmm3, %%xmm4 \n\t"
                            "cvtps2pd %%xmm5, %%xmm6 \n\t"
                            "addpd %%xmm7, %%xmm13 \n\t"
                            "addpd %%xmm8, %%xmm14 \n\t"
                            "addpd %%xmm9, %%xmm15 \n\t"
                            "addpd %%xmm2, %%xmm13 \n\t"
                            "addpd %%xmm4, %%xmm14 \n\t"
                            "addpd %%xmm6, %%xmm15"
                            :
                            :
                            :
                            "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5", "xmm6",
                            "xmm7", "xmm8", "xmm9",
                            "xmm13", "xmm14", "xmm15");
      
      r+=1;
   }

   __asm__ __volatile__ ("addpd %%xmm10, %%xmm12 \n\t"
                         "addpd %%xmm13, %%xmm15 \n\t"
                         "addpd %%xmm11, %%xmm12 \n\t"
                         "addpd %%xmm14, %%xmm15 \n\t"
                         "haddpd %%xmm12, %%xmm12 \n\t"
                         "hsubpd %%xmm15, %%xmm15 \n\t"
                         "movsd %%xmm12, %0 \n"
                         "movsd %%xmm15, %1"
                         :
                         "=m" (x),
                         "=m" (y)
                         :
                         :
                         "xmm12", "xmm15");

   if ((icom==1)&&(NPROC>1))
   {
      v.re=x;
      v.im=-y;

      MPI_Reduce(&v.re,&w.re,2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(&w.re,2,MPI_DOUBLE,0,MPI_COMM_WORLD);     

      z.re=(float)(w.re);
      z.im=(float)(w.im);
   }
   else
   {
      z.re=(float)(x);
      z.im=(float)(-y);
   }
   
   return z;  
}


float spinor_prod_re(int vol,int icom,spinor *s,spinor *r)
{
   double x,y;
   spinor *sm;

   __asm__ __volatile__ ("xorpd %%xmm10, %%xmm10 \n\t"
                         "xorpd %%xmm11, %%xmm11 \n\t"
                         "xorpd %%xmm12, %%xmm12"
                         :
                         :
                         :
                         "xmm10", "xmm11", "xmm12");
   
   sm=s+vol;

   for (;s<sm;s++)
   {
      _sse_spinor_load(*s);
      
      __asm__ __volatile__ ("mulps %0, %%xmm0 \n\t"
                            "mulps %2, %%xmm1 \n\t"
                            "mulps %4, %%xmm2"
                            :
                            :
                            "m" ((*r).c1.c1),
                            "m" ((*r).c1.c2),
                            "m" ((*r).c1.c3),
                            "m" ((*r).c2.c1),
                            "m" ((*r).c2.c2),
                            "m" ((*r).c2.c3)
                            :
                            "xmm0", "xmm1", "xmm2");

      __asm__ __volatile__ ("mulps %0, %%xmm3 \n\t"
                            "mulps %2, %%xmm4 \n\t"
                            "mulps %4, %%xmm5"
                            :
                            :
                            "m" ((*r).c3.c1),
                            "m" ((*r).c3.c2),
                            "m" ((*r).c3.c3),
                            "m" ((*r).c4.c1),
                            "m" ((*r).c4.c2),
                            "m" ((*r).c4.c3)
                            :
                            "xmm3", "xmm4", "xmm5");      

      __asm__ __volatile__ ("addps %%xmm0, %%xmm1 \n\t"
                            "addps %%xmm2, %%xmm3 \n\t"
                            "addps %%xmm4, %%xmm5 \n\t"
                            "cvtps2pd %%xmm1, %%xmm7 \n\t"
                            "cvtps2pd %%xmm3, %%xmm8 \n\t"
                            "cvtps2pd %%xmm5, %%xmm9 \n\t"
                            "shufps $0x4e, %%xmm1, %%xmm1 \n\t"
                            "shufps $0x4e, %%xmm3, %%xmm3 \n\t"
                            "shufps $0x4e, %%xmm5, %%xmm5 \n\t"
                            "cvtps2pd %%xmm1, %%xmm2 \n\t"
                            "cvtps2pd %%xmm3, %%xmm4 \n\t"
                            "cvtps2pd %%xmm5, %%xmm6 \n\t"
                            "addpd %%xmm7, %%xmm10 \n\t"
                            "addpd %%xmm8, %%xmm11 \n\t"
                            "addpd %%xmm9, %%xmm12 \n\t"
                            "addpd %%xmm2, %%xmm10 \n\t"
                            "addpd %%xmm4, %%xmm11 \n\t"
                            "addpd %%xmm6, %%xmm12"
                            :
                            :
                            :
                            "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5", "xmm6",
                            "xmm7", "xmm8", "xmm9",
                            "xmm10", "xmm11", "xmm12");

      r+=1;
   }   

   __asm__ __volatile__ ("addpd %%xmm10, %%xmm12 \n\t"
                         "addpd %%xmm11, %%xmm12 \n\t"
                         "haddpd %%xmm12, %%xmm12 \n\t"
                         "movsd %%xmm12, %0"
                         :
                         "=m" (x)
                         :
                         :
                         "xmm12"); 
   
   if ((icom==1)&&(NPROC>1))
   {
      MPI_Reduce(&x,&y,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(&y,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      return (float)(y);
   }
   else
      return (float)(x);
}


float norm_square(int vol,int icom,spinor *s)
{
   double x,y;
   spinor *sm;

   __asm__ __volatile__ ("xorpd %%xmm10, %%xmm10 \n\t"
                         "xorpd %%xmm11, %%xmm11 \n\t"
                         "xorpd %%xmm12, %%xmm12"
                         :
                         :
                         :
                         "xmm10", "xmm11", "xmm12");   

   sm=s+vol;
   
   for (;s<sm;s++)
   {
      _sse_spinor_load(*s);
      
      __asm__ __volatile__ ("mulps %%xmm0, %%xmm0 \n\t"
                            "mulps %%xmm1, %%xmm1 \n\t"
                            "mulps %%xmm2, %%xmm2 \n\t"
                            "mulps %%xmm3, %%xmm3 \n\t"
                            "mulps %%xmm4, %%xmm4 \n\t"
                            "mulps %%xmm5, %%xmm5"
                            :
                            :
                            :
                            "xmm0", "xmm1", "xmm2",
                            "xmm3", "xmm4", "xmm5");
      
      __asm__ __volatile__ ("addps %%xmm0, %%xmm1 \n\t"
                            "addps %%xmm2, %%xmm3 \n\t"
                            "addps %%xmm4, %%xmm5 \n\t"
                            "cvtps2pd %%xmm1, %%xmm7 \n\t"
                            "cvtps2pd %%xmm3, %%xmm8 \n\t"
                            "cvtps2pd %%xmm5, %%xmm9 \n\t"
                            "shufps $0x4e, %%xmm1, %%xmm1 \n\t"
                            "shufps $0x4e, %%xmm3, %%xmm3 \n\t"
                            "shufps $0x4e, %%xmm5, %%xmm5 \n\t"
                            "cvtps2pd %%xmm1, %%xmm2 \n\t"
                            "cvtps2pd %%xmm3, %%xmm4 \n\t"
                            "cvtps2pd %%xmm5, %%xmm6 \n\t"
                            "addpd %%xmm7, %%xmm10 \n\t"
                            "addpd %%xmm8, %%xmm11 \n\t"
                            "addpd %%xmm9, %%xmm12 \n\t"
                            "addpd %%xmm2, %%xmm10 \n\t"
                            "addpd %%xmm4, %%xmm11 \n\t"
                            "addpd %%xmm6, %%xmm12"
                            :
                            :
                            :
                            "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5", "xmm6",
                            "xmm7", "xmm8", "xmm9",
                            "xmm10", "xmm11", "xmm12");
   }

   __asm__ __volatile__ ("addpd %%xmm10, %%xmm12 \n\t"
                         "addpd %%xmm11, %%xmm12 \n\t"
                         "haddpd %%xmm12, %%xmm12 \n\t"
                         "movsd %%xmm12, %0"
                         :
                         "=m" (x)
                         :
                         :
                         "xmm12");  
   
   if ((icom==1)&&(NPROC>1))
   {
      MPI_Reduce(&x,&y,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(&y,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      return (float)(y);
   }
   else
      return (float)(x);
}


void mulc_spinor_add(int vol,spinor *s,spinor *r,complex z)
{
   spinor *sm;

   _sse_load_cmplx(z);
   sm=s+vol;
   
   for (;s<sm;s++)
   {
      _sse_spinor_load(*s);
      _sse_mulc_spinor_add(*r);
      _sse_spinor_store(*s);      

      r+=1;
   }
}


void mulr_spinor_add(int vol,spinor *s,spinor *r,float c)
{
   spinor *sm;

   _sse_load_real(c);
   sm=s+vol;
   
   for (;s<sm;s++)
   {
      _sse_spinor_load(*s);
      _sse_mulr_spinor_add(*r);
      _sse_spinor_store(*s);      

      r+=1;
   }
}


void scale(int vol,float c,spinor *s)
{
   spinor *sm;   

   _sse_load_real(c);
   sm=s+vol;
   
   for (;s<sm;s++)
   {
      _sse_mulr_spinor(*s);
      _sse_spinor_store(*s);
   }
}


void rotate(int vol,int n,spinor **ppk,complex *v)
{
   int k,j,ix;
   complex *z;
   spinor *pk,*pj;

   if ((n>nrot)&&(ifail==0))
      alloc_wrotate(n);

   if ((n>0)&&(ifail==0))
   {     
      for (ix=0;ix<vol;ix++)
      {
         for (k=0;k<n;k++)  
         {
            pj=ppk[0]+ix;
            z=v+k;

            _sse_load_cmplx(*z);
            _sse_mulc_spinor(*pj);

            for (j=1;j<n;j++)
            {
               pj=ppk[j]+ix;
               z+=n;
               _sse_load_cmplx(*z);
               _sse_mulc_spinor_add(*pj);
            }

            pk=psi+k;            
            _sse_spinor_store(*pk);
         }

         for (k=0;k<n;k++)
         {
            pk=psi+k;
            _sse_spinor_load(*pk);
            pj=ppk[k]+ix;
            _sse_spinor_store(*pj);
         }
      }
   }
}


void mulg5(int vol,spinor *s)
{
   spinor *sm;
   
   __asm__ __volatile__ ("movaps %0, %%xmm5 \n\t"
                         "movaps %%xmm5, %%xmm6 \n\t"
                         "movaps %%xmm5, %%xmm7"
                         :
                         :
                         "m" (_sse_sgn)
                         :
                         "xmm5", "xmm6", "xmm7");                         

   sm=s+vol;
   
   for (;s<sm;s++)
   {
      __asm__ __volatile__ ("movaps %0, %%xmm0 \n\t"
                            "movaps %2, %%xmm1 \n\t"
                            "movaps %4, %%xmm2"
                            :
                            :
                            "m" ((*s).c3.c1),
                            "m" ((*s).c3.c2),
                            "m" ((*s).c3.c3),
                            "m" ((*s).c4.c1),
                            "m" ((*s).c4.c2),
                            "m" ((*s).c4.c3)                            
                            :
                            "xmm0", "xmm1", "xmm2");

      __asm__ __volatile__ ("mulps %%xmm5, %%xmm0 \n\t"
                            "mulps %%xmm6, %%xmm1 \n\t"
                            "mulps %%xmm7, %%xmm2"
                            :
                            :
                            :
                            "xmm0", "xmm1", "xmm2");

      __asm__ __volatile__ ("movaps %%xmm0, %0 \n\t"
                            "movaps %%xmm1, %2 \n\t"
                            "movaps %%xmm2, %4"
                            :
                            "=m" ((*s).c3.c1),
                            "=m" ((*s).c3.c2),
                            "=m" ((*s).c3.c3),
                            "=m" ((*s).c4.c1),
                            "=m" ((*s).c4.c2),
                            "=m" ((*s).c4.c3));
   }
}


void mulmg5(int vol,spinor *s)
{
   spinor *sm;
   
   __asm__ __volatile__ ("movaps %0, %%xmm5 \n\t"
                         "movaps %%xmm5, %%xmm6 \n\t"
                         "movaps %%xmm5, %%xmm7"
                         :
                         :
                         "m" (_sse_sgn)
                         :
                         "xmm5", "xmm6", "xmm7");                         

   sm=s+vol;
   
   for (;s<sm;s++)
   {
      __asm__ __volatile__ ("movaps %0, %%xmm0 \n\t"
                            "movaps %2, %%xmm1 \n\t"
                            "movaps %4, %%xmm2"
                            :
                            :
                            "m" ((*s).c1.c1),
                            "m" ((*s).c1.c2),
                            "m" ((*s).c1.c3),
                            "m" ((*s).c2.c1),
                            "m" ((*s).c2.c2),
                            "m" ((*s).c2.c3)                            
                            :
                            "xmm0", "xmm1", "xmm2");

      __asm__ __volatile__ ("mulps %%xmm5, %%xmm0 \n\t"
                            "mulps %%xmm6, %%xmm1 \n\t"
                            "mulps %%xmm7, %%xmm2"
                            :
                            :
                            :
                            "xmm0", "xmm1", "xmm2");

      __asm__ __volatile__ ("movaps %%xmm0, %0 \n\t"
                            "movaps %%xmm1, %2 \n\t"
                            "movaps %%xmm2, %4"
                            :
                            "=m" ((*s).c1.c1),
                            "=m" ((*s).c1.c2),
                            "=m" ((*s).c1.c3),
                            "=m" ((*s).c2.c1),
                            "=m" ((*s).c2.c2),
                            "=m" ((*s).c2.c3));
   }
}

#else

complex spinor_prod(int vol,int icom,spinor *s,spinor *r)
{
   double x,y;
   complex z;
   complex_dble v,w;
   spinor *sm;
  
   x=0.0;
   y=0.0;
   sm=s+vol;

   for (;s<sm;s++)
   {
      x+=(double)(_vector_prod_re((*s).c1,(*r).c1))+
         (double)(_vector_prod_re((*s).c2,(*r).c2))+
         (double)(_vector_prod_re((*s).c3,(*r).c3))+
         (double)(_vector_prod_re((*s).c4,(*r).c4));

      y+=(double)(_vector_prod_im((*s).c1,(*r).c1))+
         (double)(_vector_prod_im((*s).c2,(*r).c2))+
         (double)(_vector_prod_im((*s).c3,(*r).c3))+
         (double)(_vector_prod_im((*s).c4,(*r).c4));

      r+=1;
   }

   if ((icom!=1)||(NPROC==1))
   {
      z.re=(float)(x);
      z.im=(float)(y);
   }
   else
   {
      v.re=x;
      v.im=y;

      MPI_Reduce(&v.re,&w.re,2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(&w.re,2,MPI_DOUBLE,0,MPI_COMM_WORLD);     

      z.re=(float)(w.re);
      z.im=(float)(w.im);
   }
   
   return z;  
}


float spinor_prod_re(int vol,int icom,spinor *s,spinor *r)
{
   double x,y;
   spinor *sm;
  
   x=0.0;
   sm=s+vol;
   
   for (;s<sm;s++)
   {
      x+=(double)(_vector_prod_re((*s).c1,(*r).c1))+
         (double)(_vector_prod_re((*s).c2,(*r).c2))+
         (double)(_vector_prod_re((*s).c3,(*r).c3))+
         (double)(_vector_prod_re((*s).c4,(*r).c4));

      r+=1;
   }

   if ((icom!=1)||(NPROC==1))
      return (float)(x);
   else
   {
      MPI_Reduce(&x,&y,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(&y,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      return (float)(y);
   }
}


float norm_square(int vol,int icom,spinor *s)
{
   double x,y;
   spinor *sm;

   x=0.0;
   sm=s+vol;
 
   for (;s<sm;s++)
   {
      x+=(double)(_vector_prod_re((*s).c1,(*s).c1))+
         (double)(_vector_prod_re((*s).c2,(*s).c2))+
         (double)(_vector_prod_re((*s).c3,(*s).c3))+
         (double)(_vector_prod_re((*s).c4,(*s).c4));
   }

   if ((icom!=1)||(NPROC==1))
      return (float)(x);
   else
   {
      MPI_Reduce(&x,&y,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(&y,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      return (float)(y);
   }
}


void mulc_spinor_add(int vol,spinor *s,spinor *r,complex z)
{
   spinor *sm;

   sm=s+vol;
   
   for (;s<sm;s++)
   {
      _vector_mulc_assign((*s).c1,z,(*r).c1);
      _vector_mulc_assign((*s).c2,z,(*r).c2);
      _vector_mulc_assign((*s).c3,z,(*r).c3);
      _vector_mulc_assign((*s).c4,z,(*r).c4);

      r+=1;
   }
}


void mulr_spinor_add(int vol,spinor *s,spinor *r,float c)
{
   spinor *sm;

   sm=s+vol;
   
   for (;s<sm;s++)
   {
      _vector_mulr_assign((*s).c1,c,(*r).c1);
      _vector_mulr_assign((*s).c2,c,(*r).c2);
      _vector_mulr_assign((*s).c3,c,(*r).c3);
      _vector_mulr_assign((*s).c4,c,(*r).c4);

      r+=1;
   }
}


void scale(int vol,float c,spinor *s)
{
   spinor *sm;

   sm=s+vol;
   
   for (;s<sm;s++)
   {
      _vector_mul((*s).c1,c,(*s).c1);
      _vector_mul((*s).c2,c,(*s).c2);
      _vector_mul((*s).c3,c,(*s).c3);
      _vector_mul((*s).c4,c,(*s).c4);
   }
}


void rotate(int vol,int n,spinor **ppk,complex *v)
{
   int k,j,ix;
   complex *z;
   spinor *pk,*pj;

   if ((n>nrot)&&(ifail==0))
      alloc_wrotate(n);

   if ((n>0)&&(ifail==0))
   {     
      for (ix=0;ix<vol;ix++)
      {
         for (k=0;k<n;k++)  
         {
            pk=psi+k;
            pj=ppk[0]+ix;
            z=v+k;

            _vector_mulc((*pk).c1,*z,(*pj).c1);
            _vector_mulc((*pk).c2,*z,(*pj).c2);
            _vector_mulc((*pk).c3,*z,(*pj).c3);
            _vector_mulc((*pk).c4,*z,(*pj).c4);
     
            for (j=1;j<n;j++)
            {
               pj=ppk[j]+ix;
               z+=n;

               _vector_mulc_assign((*pk).c1,*z,(*pj).c1);
               _vector_mulc_assign((*pk).c2,*z,(*pj).c2);
               _vector_mulc_assign((*pk).c3,*z,(*pj).c3);
               _vector_mulc_assign((*pk).c4,*z,(*pj).c4);            
            }
         }

         for (k=0;k<n;k++)
            ppk[k][ix]=psi[k];
      }
   }
}


void mulg5(int vol,spinor *s)
{
   spinor *sm;

   sm=s+vol;
   
   for (;s<sm;s++)
   {
      (*s).c3.c1.re=-(*s).c3.c1.re;
      (*s).c3.c1.im=-(*s).c3.c1.im;      
      (*s).c3.c2.re=-(*s).c3.c2.re;
      (*s).c3.c2.im=-(*s).c3.c2.im;      
      (*s).c3.c3.re=-(*s).c3.c3.re;
      (*s).c3.c3.im=-(*s).c3.c3.im;      
      (*s).c4.c1.re=-(*s).c4.c1.re;
      (*s).c4.c1.im=-(*s).c4.c1.im;      
      (*s).c4.c2.re=-(*s).c4.c2.re;
      (*s).c4.c2.im=-(*s).c4.c2.im;      
      (*s).c4.c3.re=-(*s).c4.c3.re;
      (*s).c4.c3.im=-(*s).c4.c3.im; 
   }
}


void mulmg5(int vol,spinor *s)
{
   spinor *sm;

   sm=s+vol;
   
   for (;s<sm;s++)
   {
      (*s).c1.c1.re=-(*s).c1.c1.re;
      (*s).c1.c1.im=-(*s).c1.c1.im;      
      (*s).c1.c2.re=-(*s).c1.c2.re;
      (*s).c1.c2.im=-(*s).c1.c2.im;      
      (*s).c1.c3.re=-(*s).c1.c3.re;
      (*s).c1.c3.im=-(*s).c1.c3.im;      
      (*s).c2.c1.re=-(*s).c2.c1.re;
      (*s).c2.c1.im=-(*s).c2.c1.im;      
      (*s).c2.c2.re=-(*s).c2.c2.re;
      (*s).c2.c2.im=-(*s).c2.c2.im;      
      (*s).c2.c3.re=-(*s).c2.c3.re;
      (*s).c2.c3.im=-(*s).c2.c3.im;      
   }
}

#endif

void project(int vol,int icom,spinor *s,spinor *r)
{
   complex z;

   z=spinor_prod(vol,icom,r,s);
   z.re=-z.re;
   z.im=-z.im;
   mulc_spinor_add(vol,s,r,z);   
}


float normalize(int vol,int icom,spinor *s)
{
   float r;

   r=norm_square(vol,icom,s);
   r=(float)(sqrt((double)(r)));

   if (r!=0.0f)
      scale(vol,1.0f/r,s);
   else
      error_loc(1,1,"normalize [salg.c]",
                "Vector has vanishing norm");

   return r;
}
