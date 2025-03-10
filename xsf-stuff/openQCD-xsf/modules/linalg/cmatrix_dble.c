
/*******************************************************************************
*
* File cmatrix_dble.c
*
* Copyright (C) 2007, 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Complex matrix algebra (double-precision version)
*
* The externally accessible functions are
*
*   void cmat_vec_dble(int n,complex_dble *a,complex_dble *v,complex_dble *w)
*     Computes w=a*v, where v and w are n-vectors and a an nxn matrix.
*
*   void cmat_vec_assign_dble(int n,complex_dble *a,complex_dble *v,
*                             complex_dble *w)
*     Adds a*v to w, where v and w are n-vectors and a an nxn matrix.
*     
*   void cmat_add_dble(int n,complex_dble *a,complex_dble *b,complex_dble *c)
*     Computes the sum c=a+b of two nxn matrices a and b.
*
*   void cmat_sub_dble(int n,complex_dble *a,complex_dble *b,complex_dble *c)
*     Computes the difference c=a-b of two nxn matrices a and b.
*
*   void cmat_mul_dble(int n,complex_dble *a,complex_dble *b,complex_dble *c)
*     Computes the product c=a*b of two nxn matrices a and b.
*
*   void cmat_dag_dble(int n,complex_dble *a,complex_dble *b)
*     Assigns the hermitian conjugate of a to b.
*
*   int cmat_inv_dble(int n,complex_dble *a,complex_dble *b,double *k)
*     Computes the inverse b of the nxn matrix a, using Householder
*     reflections. The Frobenius condition number k of a is also computed.
*     A non-zero return value indicates that the input matrix was found to
*     be singular within rounding errors and that the program terminated
*     prematurely.
*
* Notes:
*
* All of these programs can be called locally. Complex nxn matrices with
* matrix elements A_{ij} are represented by linear arrays a of complex
* numbers such that
*
*   A_{ij} = a[i*n+j]
*
* where i,j=0,1,..,n-1. It is assumed that the input and output arrays do 
* not overlap in memory (the results are otherwise unpredictable).
*
* The inverse of a given matrix computed by cmat_inv_dble() may suffer
* from significance losses on the order of its condition number.
*
* If SSE2 instructions are to be used, it is taken for granted that the
* starting addresses of the arrays are multiples of 16.
*
*******************************************************************************/

#define CMATRIX_DBLE_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "su3.h"
#include "utils.h"
#include "linalg.h"

static int nmax=0;
static double *rsv;
static complex_dble *dsv;

#if (defined x64)
#include "sse2.h"

void cmat_vec_dble(int n,complex_dble *a,complex_dble *v,complex_dble *w)
{
   complex_dble *vv,*vm,*wm;;

   if ((n&0x1)==0x0)
   {
      vm=v+n;
      wm=w+n;
   
      for (;w<wm;w++)
      {
         __asm__ __volatile__ ("xorpd %%xmm0, %%xmm0 \n\t"
                               "xorpd %%xmm1, %%xmm1 \n\t"
                               "xorpd %%xmm2, %%xmm2 \n\t"
                               "xorpd %%xmm3, %%xmm3 \n\t"
                               :
                               :
                               :
                               "xmm0", "xmm1", "xmm2", "xmm3");
      
         for (vv=v;vv<vm;vv+=2)
         {
            __asm__ __volatile__ ("movapd %0, %%xmm4 \n\t"
                                  "movapd %1, %%xmm5 \n\t"
                                  "movapd %%xmm4, %%xmm6 \n\t"
                                  "movapd %%xmm5, %%xmm7 \n\t"
                                  "shufpd $0x1, %%xmm6, %%xmm6 \n\t"
                                  "shufpd $0x1, %%xmm7, %%xmm7 \n\t"
                                  "mulpd %2, %%xmm4 \n\t"
                                  "mulpd %3, %%xmm5 \n\t"
                                  "mulpd %2, %%xmm6 \n\t"
                                  "mulpd %3, %%xmm7 \n\t"
                                  "addpd %%xmm4, %%xmm0 \n\t"
                                  "addpd %%xmm5, %%xmm1 \n\t"
                                  "addpd %%xmm6, %%xmm2 \n\t"
                                  "addpd %%xmm7, %%xmm3"
                                  :
                                  :
                                  "m" (a[0]),
                                  "m" (a[1]),
                                  "m" (vv[0]),
                                  "m" (vv[1])
                                  :
                                  "xmm0", "xmm1", "xmm2", "xmm3", 
                                  "xmm4", "xmm5", "xmm6", "xmm7");           

            a+=2;
         }

         __asm__ __volatile__ ("addpd %%xmm1, %%xmm0 \n\t"
                               "addpd %%xmm3, %%xmm2 \n\t"
                               "movapd %%xmm0, %%xmm1 \n\t"
                               "movapd %%xmm2, %%xmm3 \n\t"
                               "shufpd $0x1, %%xmm2, %%xmm0 \n\t"
                               "shufpd $0x2, %%xmm3, %%xmm1 \n\t"
                               "mulpd %1, %%xmm0 \n\t"
                               "addpd %%xmm1, %%xmm0 \n\t"
                               "movapd %%xmm0, %0"
                               :
                               "=m" (w[0])
                               :
                               "m" (_sse_sgn1_dble)
                               :
                               "xmm0", "xmm1", "xmm2", "xmm3");
      }
   }
   else
   {
      vm=v+n-1;
      wm=w+n;
   
      for (;w<wm;w++)
      {
         __asm__ __volatile__ ("xorpd %%xmm0, %%xmm0 \n\t"
                               "xorpd %%xmm1, %%xmm1 \n\t"
                               "xorpd %%xmm2, %%xmm2 \n\t"
                               "xorpd %%xmm3, %%xmm3 \n\t"
                               :
                               :
                               :
                               "xmm0", "xmm1", "xmm2", "xmm3");
      
         for (vv=v;vv<vm;vv+=2)
         {
            __asm__ __volatile__ ("movapd %0, %%xmm4 \n\t"
                                  "movapd %1, %%xmm5 \n\t"
                                  "movapd %%xmm4, %%xmm6 \n\t"
                                  "movapd %%xmm5, %%xmm7 \n\t"
                                  "shufpd $0x1, %%xmm6, %%xmm6 \n\t"
                                  "shufpd $0x1, %%xmm7, %%xmm7 \n\t"
                                  "mulpd %2, %%xmm4 \n\t"
                                  "mulpd %3, %%xmm5 \n\t"
                                  "mulpd %2, %%xmm6 \n\t"
                                  "mulpd %3, %%xmm7 \n\t"
                                  "addpd %%xmm4, %%xmm0 \n\t"
                                  "addpd %%xmm5, %%xmm1 \n\t"
                                  "addpd %%xmm6, %%xmm2 \n\t"
                                  "addpd %%xmm7, %%xmm3"
                                  :
                                  :
                                  "m" (a[0]),
                                  "m" (a[1]),
                                  "m" (vv[0]),
                                  "m" (vv[1])
                                  :
                                  "xmm0", "xmm1", "xmm2", "xmm3", 
                                  "xmm4", "xmm5", "xmm6", "xmm7");           

            a+=2;
         }

         __asm__ __volatile__ ("addpd %%xmm1, %%xmm0 \n\t"
                               "addpd %%xmm3, %%xmm2 \n\t"
                               "movapd %0, %%xmm1 \n\t"
                               "movapd %1, %%xmm4 \n\t"
                               "movapd %%xmm1, %%xmm3 \n\t"
                               "movapd %%xmm4, %%xmm5 \n\t"
                               "shufpd $0x1, %%xmm3, %%xmm3 \n\t"
                               "mulpd %%xmm4, %%xmm1 \n\t"
                               "mulpd %%xmm5, %%xmm3"
                               :
                               :
                               "m" (a[0]),
                               "m" (vv[0])
                               :
                               "xmm0", "xmm1", "xmm2", "xmm3",
                               "xmm4", "xmm5");
         
         __asm__ __volatile__ ("addpd %%xmm1, %%xmm0 \n\t"
                               "addpd %%xmm3, %%xmm2 \n\t"
                               "movapd %%xmm0, %%xmm1 \n\t"
                               "movapd %%xmm2, %%xmm3 \n\t"
                               "shufpd $0x1, %%xmm2, %%xmm0 \n\t"
                               "shufpd $0x2, %%xmm3, %%xmm1 \n\t"
                               "mulpd %1, %%xmm0 \n\t"
                               "addpd %%xmm1, %%xmm0 \n\t"
                               "movapd %%xmm0, %0"
                               :
                               "=m" (w[0])
                               :
                               "m" (_sse_sgn1_dble)
                               :
                               "xmm0", "xmm1", "xmm2", "xmm3");

         a+=1;
      }
   }
}


void cmat_vec_assign_dble(int n,complex_dble *a,complex_dble *v,complex_dble *w)
{
   complex_dble *vv,*vm,*wm;;

   if ((n&0x1)==0x0)
   {
      vm=v+n;
      wm=w+n;
   
      for (;w<wm;w++)
      {
         __asm__ __volatile__ ("movsd %0, %%xmm0 \n\t"
                               "xorpd %%xmm1, %%xmm1 \n\t"
                               "movsd %1, %%xmm2 \n\t"
                               "xorpd %%xmm3, %%xmm3 \n\t"
                               :
                               :
                               "m" (w[0].re),
                               "m" (w[0].im)
                               :
                               "xmm0", "xmm1", "xmm2", "xmm3");
      
         for (vv=v;vv<vm;vv+=2)
         {
            __asm__ __volatile__ ("movapd %0, %%xmm4 \n\t"
                                  "movapd %1, %%xmm5 \n\t"
                                  "movapd %%xmm4, %%xmm6 \n\t"
                                  "movapd %%xmm5, %%xmm7 \n\t"
                                  "shufpd $0x1, %%xmm6, %%xmm6 \n\t"
                                  "shufpd $0x1, %%xmm7, %%xmm7 \n\t"
                                  "mulpd %2, %%xmm4 \n\t"
                                  "mulpd %3, %%xmm5 \n\t"
                                  "mulpd %2, %%xmm6 \n\t"
                                  "mulpd %3, %%xmm7 \n\t"
                                  "addpd %%xmm4, %%xmm0 \n\t"
                                  "addpd %%xmm5, %%xmm1 \n\t"
                                  "addpd %%xmm6, %%xmm2 \n\t"
                                  "addpd %%xmm7, %%xmm3"
                                  :
                                  :
                                  "m" (a[0]),
                                  "m" (a[1]),
                                  "m" (vv[0]),
                                  "m" (vv[1])
                                  :
                                  "xmm0", "xmm1", "xmm2", "xmm3", 
                                  "xmm4", "xmm5", "xmm6", "xmm7");           

            a+=2;
         }

         __asm__ __volatile__ ("addpd %%xmm1, %%xmm0 \n\t"
                               "addpd %%xmm3, %%xmm2 \n\t"
                               "movapd %%xmm0, %%xmm1 \n\t"
                               "movapd %%xmm2, %%xmm3 \n\t"
                               "shufpd $0x1, %%xmm2, %%xmm0 \n\t"
                               "shufpd $0x2, %%xmm3, %%xmm1 \n\t"
                               "mulpd %1, %%xmm0 \n\t"
                               "addpd %%xmm1, %%xmm0 \n\t"
                               "movapd %%xmm0, %0"
                               :
                               "=m" (w[0])
                               :
                               "m" (_sse_sgn1_dble)
                               :
                               "xmm0", "xmm1", "xmm2", "xmm3");
      }
   }
   else
   {
      vm=v+n-1;
      wm=w+n;
   
      for (;w<wm;w++)
      {
         __asm__ __volatile__ ("movsd %0, %%xmm0 \n\t"
                               "xorpd %%xmm1, %%xmm1 \n\t"
                               "movsd %1, %%xmm2 \n\t"
                               "xorpd %%xmm3, %%xmm3 \n\t"
                               :
                               :
                               "m" (w[0].re),
                               "m" (w[0].im)
                               :
                               "xmm0", "xmm1", "xmm2", "xmm3");         
      
         for (vv=v;vv<vm;vv+=2)
         {
            __asm__ __volatile__ ("movapd %0, %%xmm4 \n\t"
                                  "movapd %1, %%xmm5 \n\t"
                                  "movapd %%xmm4, %%xmm6 \n\t"
                                  "movapd %%xmm5, %%xmm7 \n\t"
                                  "shufpd $0x1, %%xmm6, %%xmm6 \n\t"
                                  "shufpd $0x1, %%xmm7, %%xmm7 \n\t"
                                  "mulpd %2, %%xmm4 \n\t"
                                  "mulpd %3, %%xmm5 \n\t"
                                  "mulpd %2, %%xmm6 \n\t"
                                  "mulpd %3, %%xmm7 \n\t"
                                  "addpd %%xmm4, %%xmm0 \n\t"
                                  "addpd %%xmm5, %%xmm1 \n\t"
                                  "addpd %%xmm6, %%xmm2 \n\t"
                                  "addpd %%xmm7, %%xmm3"
                                  :
                                  :
                                  "m" (a[0]),
                                  "m" (a[1]),
                                  "m" (vv[0]),
                                  "m" (vv[1])
                                  :
                                  "xmm0", "xmm1", "xmm2", "xmm3", 
                                  "xmm4", "xmm5", "xmm6", "xmm7");           

            a+=2;
         }

         __asm__ __volatile__ ("addpd %%xmm1, %%xmm0 \n\t"
                               "addpd %%xmm3, %%xmm2 \n\t"
                               "movapd %0, %%xmm1 \n\t"
                               "movapd %1, %%xmm4 \n\t"
                               "movapd %%xmm1, %%xmm3 \n\t"
                               "movapd %%xmm4, %%xmm5 \n\t"
                               "shufpd $0x1, %%xmm3, %%xmm3 \n\t"
                               "mulpd %%xmm4, %%xmm1 \n\t"
                               "mulpd %%xmm5, %%xmm3"
                               :
                               :
                               "m" (a[0]),
                               "m" (vv[0])
                               :
                               "xmm0", "xmm1", "xmm2", "xmm3",
                               "xmm4", "xmm5");
         

         __asm__ __volatile__ ("addpd %%xmm1, %%xmm0 \n\t"
                               "addpd %%xmm3, %%xmm2 \n\t"
                               "movapd %%xmm0, %%xmm1 \n\t"
                               "movapd %%xmm2, %%xmm3 \n\t"
                               "shufpd $0x1, %%xmm2, %%xmm0 \n\t"
                               "shufpd $0x2, %%xmm3, %%xmm1 \n\t"
                               "mulpd %1, %%xmm0 \n\t"
                               "addpd %%xmm1, %%xmm0 \n\t"
                               "movapd %%xmm0, %0"
                               :
                               "=m" (w[0])
                               :
                               "m" (_sse_sgn1_dble)
                               :
                               "xmm0", "xmm1", "xmm2", "xmm3");

         a+=1;
      }
   }
}

#else

void cmat_vec_dble(int n,complex_dble *a,complex_dble *v,complex_dble *w)
{
   complex_dble *vv,*vm,*wm;;

   vm=v+n;
   wm=w+n;
   
   for (;w<wm;w++)
   {
      (*w).re=0.0;
      (*w).im=0.0;
         
      for (vv=v;vv<vm;vv++)
      {
         (*w).re+=((*a).re*(*vv).re-(*a).im*(*vv).im);
         (*w).im+=((*a).re*(*vv).im+(*a).im*(*vv).re);
         a+=1;
      }
   }
}


void cmat_vec_assign_dble(int n,complex_dble *a,complex_dble *v,complex_dble *w)
{
   complex_dble *vv,*vm,*wm;;

   vm=v+n;
   wm=w+n;
   
   for (;w<wm;w++)
   {
      for (vv=v;vv<vm;vv++)
      {
         (*w).re+=((*a).re*(*vv).re-(*a).im*(*vv).im);
         (*w).im+=((*a).re*(*vv).im+(*a).im*(*vv).re);
         a+=1;
      }
   }
}

#endif

void cmat_add_dble(int n,complex_dble *a,complex_dble *b,complex_dble *c)
{
   complex_dble *am;

   am=a+n*n;

   for (;a<am;a++)
   {
      (*c).re=(*a).re+(*b).re;
      (*c).im=(*a).im+(*b).im;
      b+=1;
      c+=1;
   }
}


void cmat_sub_dble(int n,complex_dble *a,complex_dble *b,complex_dble *c)
{
   complex_dble *am;

   am=a+n*n;

   for (;a<am;a++)
   {
      (*c).re=(*a).re-(*b).re;
      (*c).im=(*a).im-(*b).im;
      b+=1;
      c+=1;
   }
}


void cmat_mul_dble(int n,complex_dble *a,complex_dble *b,complex_dble *c)
{
   complex_dble *aa,*bb,*am,*bm,*bbm;

   am=a+n*n;
   bm=b+n;
   bbm=b+n*n;
   
   for (;a<am;a+=n)
   {
      for (;b<bm;b++)
      {
         (*c).re=0.0;
         (*c).im=0.0;
         aa=a;

         for (bb=b;bb<bbm;bb+=n)
         {
            (*c).re+=((*aa).re*(*bb).re-(*aa).im*(*bb).im);
            (*c).im+=((*aa).re*(*bb).im+(*aa).im*(*bb).re);
            aa+=1;
         }

         c+=1;
      }

      b-=n;
   }
}


void cmat_dag_dble(int n,complex_dble *a,complex_dble *b)
{
   complex_dble *bb,*am,*bbm;

   am=a+n*n;
   bbm=b+n*n;
   
   for (;a<am;)
   {
      for (bb=b;bb<bbm;bb+=n)
      {
         (*bb).re=(*a).re;
         (*bb).im=-(*a).im;
         a+=1;
      }

      b+=1;
   }
}


static int alloc_arrays(int n)
{
   if (nmax>0)
   {
      nmax=0;
      afree(rsv);
      afree(dsv);
      rsv=NULL;
      dsv=NULL;
   }
   
   if (n>0)
   {
      rsv=amalloc(n*sizeof(*rsv),4);
      dsv=amalloc(n*sizeof(*dsv),4);

      if (error_loc((rsv==NULL)||(dsv==NULL),1,"alloc_arrays [cmatrix_dble.c]",
                    "Unable to allocate auxiliary arrays")==0)
      {
         nmax=n;
         return 0;
      }
      else
      {
         if (rsv!=NULL)
            afree(rsv);
         if (dsv!=NULL)
            afree(dsv);
         rsv=NULL;
         dsv=NULL;

         return 1;
      }
   }

   return 0;
}


static int fwd_house(int n,complex_dble *a,complex_dble *b,double *fnsq)
{
   int i,j,k;
   double eps,r1,r2,r3;
   complex_dble z,*bb,*bm,*bk,*bj;

   *fnsq=0.0;
   bm=b+n*n;

   for (bb=b;bb<bm;bb++)
   {
      (*bb).re=(*a).re;
      (*bb).im=(*a).im;
      *fnsq+=((*a).re*(*a).re+(*a).im*(*a).im);
      a+=1;
   }

   eps=(double)(n)*DBL_EPSILON*(*fnsq);
   if (eps==0.0)
      return 2;

   for (k=0;k<(n-1);k++)
   {
      r1=b[n*k+k].re*b[n*k+k].re+b[n*k+k].im*b[n*k+k].im;
      r2=sqrt(r1);

      for (j=(k+1);j<n;j++)
         r1+=(b[n*j+k].re*b[n*j+k].re+b[n*j+k].im*b[n*j+k].im);

      if (r1>=eps)
         r1=sqrt(r1);
      else
         return 3;

      if (r2>=(DBL_EPSILON*r1))
      {
         r3=1.0/r2;
         z.re=r3*b[n*k+k].re;
         z.im=r3*b[n*k+k].im;
      }
      else
      {
         z.re=1.0;
         z.im=0.0;
      }

      b[n*k+k].re+=r1*z.re;
      b[n*k+k].im+=r1*z.im;

      r3=1.0/(r1*(r1+r2));
      rsv[k]=r3;
      dsv[k].re=-(r1+r2)*r3*z.re;
      dsv[k].im= (r1+r2)*r3*z.im;

      for (j=(k+1);j<n;j++)
      {
         z.re=0.0;
         z.im=0.0;
         bk=b+n*k+k;
         bj=b+n*k+j;

         for (i=k;i<n;i++)
         {
            z.re+=(*bk).re*(*bj).re+(*bk).im*(*bj).im;
            z.im+=(*bk).re*(*bj).im-(*bk).im*(*bj).re;
            bk+=n;
            bj+=n;
         }

         z.re*=r3;
         z.im*=r3;
         bk=b+n*k+k;
         bj=b+n*k+j;         

         for (i=k;i<n;i++)
         {
            (*bj).re-=(z.re*(*bk).re-z.im*(*bk).im);
            (*bj).im-=(z.re*(*bk).im+z.im*(*bk).re);
            bk+=n;
            bj+=n;
         }
      }
   }

   bb=bm-1;
   r1=(*bb).re*(*bb).re+(*bb).im*(*bb).im;

   if (r1>=eps)
      r1=1.0/r1;
   else
      return 3;

   dsv[n-1].re= r1*(*bb).re;
   dsv[n-1].im=-r1*(*bb).im;

   return 0;
}


static void solv_sys(int n,complex_dble *b)
{
   int i,j,k;
   complex_dble *bi,*bk,z;

   for (k=(n-1);k>0;k--)
   {
      for (i=(k-1);i>=0;i--)
      {
         bi=b+n*i+k;
         bk=b+n*k-n+k;
         z.re=(*bi).re*dsv[k].re-(*bi).im*dsv[k].im;
         z.im=(*bi).re*dsv[k].im+(*bi).im*dsv[k].re;

         for (j=(k-1);j>i;j--)
         {
            bi-=1;
            z.re+=((*bi).re*(*bk).re-(*bi).im*(*bk).im);
            z.im+=((*bi).re*(*bk).im+(*bi).im*(*bk).re);
            bk-=n;
         }

         (*bk).re=-dsv[i].re*z.re+dsv[i].im*z.im;
         (*bk).im=-dsv[i].re*z.im-dsv[i].im*z.re;
      }
   }
}


static void bck_house(int n,complex_dble *b)
{
   int i,j,k;
   complex_dble *bi,*dj,z;

   b[n*n-1].re=dsv[n-1].re;
   b[n*n-1].im=dsv[n-1].im;

   for (k=(n-2);k>=0;k--)
   {
      z.re=dsv[k].re;
      z.im=dsv[k].im;
      dsv[k].re=b[n*k+k].re;
      dsv[k].im=b[n*k+k].im;
      b[n*k+k].re=z.re;
      b[n*k+k].im=z.im;

      for (j=(k+1);j<n;j++)
      {
         dsv[j].re=b[n*j+k].re;
         dsv[j].im=b[n*j+k].im;
         b[n*j+k].re=0.0;
         b[n*j+k].im=0.0;
      }

      for (i=0;i<n;i++)
      {
         z.re=0.0;
         z.im=0.0;
         bi=b+n*i+k;
         dj=dsv+k;

         for (j=k;j<n;j++)
         {
            z.re+=((*bi).re*(*dj).re-(*bi).im*(*dj).im);
            z.im+=((*bi).re*(*dj).im+(*bi).im*(*dj).re);
            bi+=1;
            dj+=1;
         }

         z.re*=rsv[k];
         z.im*=rsv[k];
         bi=b+n*i+k;
         dj=dsv+k;
         
         for (j=k;j<n;j++)
         {
            (*bi).re-=(z.re*(*dj).re+z.im*(*dj).im);
            (*bi).im+=(z.re*(*dj).im-z.im*(*dj).re);
            bi+=1;
            dj+=1;            
         }
      }
   }
}


int cmat_inv_dble(int n,complex_dble *a,complex_dble *b,double *k)
{
   int ie;
   double fnsq,fnsqi;
   complex_dble *bb,*bm;

   *k=1.0/DBL_EPSILON;
   
   if (n>nmax)
   {
      if (alloc_arrays(n)!=0)
         return 1;
   }

   ie=fwd_house(n,a,b,&fnsq);

   if (ie!=0)
      return ie;

   solv_sys(n,b);
   bck_house(n,b);   

   bb=b;
   bm=bb+n*n;
   fnsqi=0.0;

   for (;bb<bm;bb++)
      fnsqi+=((*bb).re*(*bb).re+(*bb).im*(*bb).im);

   *k=sqrt(fnsq*fnsqi);
   
   return 0;
}

