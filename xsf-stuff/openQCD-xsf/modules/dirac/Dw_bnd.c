
/*******************************************************************************
*
* File Dw_bnd.c
*
* Copyright (C) 2005, 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Block boundary part of the Wilson-Dirac operator
*
* The externally accessible function is
*
*   void Dw_bnd(blk_grid_t grid,int n,int k,int l)
*     Applies the boundary part of the Wilson-Dirac operator to the field
*     b.s[k] on the n'th block b of the specified block grid and assigns
*     the result to the field bb.w[l] on the boundary bb of the block. On
*     exit b.s[k] is unchanged except on the points at global time 0 and
*     NPROC0*L0-1, where it is set to zero. The fields bb.w[l] at time -1
*     and NPROC0*L0 are set to zero too.
*
* Notes:
*
* The boundary part of the Wilson-Dirac operator on a block is the sum of 
* the hopping terms that lead from the block to its exterior boundary. If
* the faces of the block in the -0,+0,...,-3,+3 directions are labeled by
* an integer ifc=0,..,7, the Dirac spinors psi computed by Dw_bnd() along
* the boundary number ifc satisfy
*
*   theta[ifc]*psi=0
*
* (see sflds/Pbnd.c for the definition of the projectors theta[ifc]). The
* program Dw_bnd() assigns the upper two components of psi to the Weyl
* fields on the boundaries of the block.
*
* The program in this module does not perform any communications and can be
* called locally.
*
*******************************************************************************/

#define DW_BND_C

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "su3.h"
#include "utils.h"
#include "block.h"
#include "dirac.h"
#include "global.h"

#if (defined x64)
#include "sse2.h"

#define _load_cst(c) \
__asm__ __volatile__ ("movss %0, %%xmm15 \n\t" \
                      "shufps $0x0, %%xmm15, %%xmm15" \
                      : \
                      : \
                      "m" (c) \
                      : \
                      "xmm15")

#define _mul_cst() \
__asm__ __volatile__ ("mulps %%xmm15, %%xmm0 \n\t" \
                      "mulps %%xmm15, %%xmm1 \n\t" \
                      "mulps %%xmm15, %%xmm2" \
                      : \
                      : \
                      : \
                      "xmm0", "xmm1", "xmm2")

#define _load_zero() \
__asm__ __volatile__ ("xorps %%xmm0, %%xmm0 \n\t" \
                      "xorps %%xmm1, %%xmm1 \n\t" \
                      "xorps %%xmm2, %%xmm2 \n\t" \
                      "xorps %%xmm3, %%xmm3 \n\t" \
                      "xorps %%xmm4, %%xmm4 \n\t" \
                      "xorps %%xmm5, %%xmm5" \
                      : \
                      : \
                      : \
                      "xmm0", "xmm1", "xmm2", "xmm3", \
                      "xmm4", "xmm5")

#define _set_s2zero(s) \
__asm__ __volatile__ ("movaps %%xmm0, %0 \n\t" \
                      "movaps %%xmm1, %2 \n\t" \
                      "movaps %%xmm2, %4" \
                      : \
                      "=m" ((s).c1.c1), \
                      "=m" ((s).c1.c2), \
                      "=m" ((s).c1.c3), \
                      "=m" ((s).c2.c1), \
                      "=m" ((s).c2.c2), \
                      "=m" ((s).c2.c3)); \
__asm__ __volatile__ ("movaps %%xmm3, %0 \n\t" \
                      "movaps %%xmm4, %2 \n\t" \
                      "movaps %%xmm5, %4" \
                      : \
                      "=m" ((s).c3.c1), \
                      "=m" ((s).c3.c2), \
                      "=m" ((s).c3.c3), \
                      "=m" ((s).c4.c1), \
                      "=m" ((s).c4.c2), \
                      "=m" ((s).c4.c3))

#define _set_w2zero(w) \
__asm__ __volatile__ ("movaps %%xmm0, %0 \n\t" \
                      "movaps %%xmm1, %2 \n\t" \
                      "movaps %%xmm2, %4" \
                      : \
                      "=m" ((w).c1.c1), \
                      "=m" ((w).c1.c2), \
                      "=m" ((w).c1.c3), \
                      "=m" ((w).c2.c1), \
                      "=m" ((w).c2.c2), \
                      "=m" ((w).c2.c3))


static void mul_umat(su3 *u)
{
   _sse_su3_multiply(*u);
}


static void mul_uinv(su3 *u)
{
   _sse_su3_inverse_multiply(*u);
}


void Dw_bnd(blk_grid_t grid,int n,int k,int l)
{
   int nb,isw,*ipp;
   float moh;
   su3 *u;
   weyl *w,*wm;
   spinor *s,*sn;
   block_t *b;
   bndry_t *bb;   

   b=blk_list(grid,&nb,&isw);

   if ((n<0)||(n>=nb))
   {
      error_loc(1,1,"Dw_bnd [Dw_bnd.c]",
                "Block grid is not allocated or block number out of range");
      return;
   }   

   b+=n;
   bb=(*b).bb;
   
   if ((k<0)||(k>=(*b).ns)||((*b).u==NULL)||(bb==NULL)||(l>=(*bb).nw))
   {
      error_loc(1,1,"Dw_bnd [Dw_bnd.c]",
                "Attempt to access unallocated memory space");
      return;
   }       

   moh=-0.5f;
   _load_cst(moh);   
   s=(*b).s[k];
   
/********************************** face -0 ***********************************/

   ipp=(*bb).ipp;   
   w=(*bb).w[l];
   wm=w+(*bb).vol;   
   
   if ((cpr[0]==0)&&((*b).bo[0]==0))
   {
      _load_zero();      

      for (;w<wm;w++)
      {
         sn=s+(*(ipp++));
         _set_s2zero(*sn);
         _set_w2zero(*w);
      }
   }
   else
   {
      u=(*bb).u;
      sn=s+(*(ipp++));
      
      for (;w<wm;w++)
      {
         _sse_pair_load((*sn).c1,(*sn).c2);
         _sse_pair_load_up((*sn).c3,(*sn).c4);

         sn=s+(*(ipp++));
         _prefetch_spinor(sn);

         _sse_vector_add();
         _mul_cst();

         u+=3;
         _prefetch_su3(u);
         u-=3;
         mul_umat(u);

         _sse_pair_store_up((*w).c1,(*w).c2);
         u+=1;
      }
   }

/********************************** face +0 ***********************************/

   bb+=1;
   ipp=(*bb).ipp;
   w=(*bb).w[l];
   wm=w+(*bb).vol;   

   if ((cpr[0]==(NPROC0-1))&&(((*b).bo[0]+(*b).bs[0])==L0))
   {
      _load_zero();      

      for (;w<wm;w++)
      {
         sn=s+(*(ipp++));
         _set_s2zero(*sn);
         _set_w2zero(*w);
      }      
   }
   else
   {
      u=(*bb).u;
      sn=s+(*(ipp++));

      for (;w<wm;w++)
      {
         _sse_pair_load((*sn).c1,(*sn).c2);
         _sse_pair_load_up((*sn).c3,(*sn).c4);

         sn=s+(*(ipp++));
         _prefetch_spinor(sn);

         _sse_vector_sub();
         _mul_cst();

         u+=3;
         _prefetch_su3(u);
         u-=3;
         mul_uinv(u);

         _sse_pair_store_up((*w).c1,(*w).c2);
         u+=1;
      }
   }

/********************************** face -1 ***********************************/

   bb+=1;
   ipp=(*bb).ipp;
   w=(*bb).w[l];
   wm=w+(*bb).vol;
   u=(*bb).u;
   sn=s+(*(ipp++));

   for (;w<wm;w++)
   {
      _sse_pair_load((*sn).c1,(*sn).c2);
      _sse_pair_load_up((*sn).c4,(*sn).c3);

      sn=s+(*(ipp++));
      _prefetch_spinor(sn);

      _sse_vector_i_add();
      _mul_cst();

      u+=3;
      _prefetch_su3(u);
      u-=3;
      mul_umat(u);

      _sse_pair_store_up((*w).c1,(*w).c2);
      u+=1;
   }

/********************************** face +1 ***********************************/

   bb+=1;
   ipp=(*bb).ipp;
   w=(*bb).w[l];
   wm=w+(*bb).vol;
   u=(*bb).u;
   sn=s+(*(ipp++));

   for (;w<wm;w++)
   {
      _sse_pair_load((*sn).c1,(*sn).c2);
      _sse_pair_load_up((*sn).c4,(*sn).c3);

      sn=s+(*(ipp++));
      _prefetch_spinor(sn);

      _sse_vector_i_sub();
      _mul_cst();

      u+=3;
      _prefetch_su3(u);
      u-=3;
      mul_uinv(u);

      _sse_pair_store_up((*w).c1,(*w).c2);
      u+=1;
   }

/********************************** face -2 ***********************************/

   bb+=1;
   ipp=(*bb).ipp;
   w=(*bb).w[l];
   wm=w+(*bb).vol;
   u=(*bb).u;
   sn=s+(*(ipp++));

   for (;w<wm;w++)
   {
      _sse_pair_load((*sn).c1,(*sn).c2);
      _sse_pair_load_up((*sn).c4,(*sn).c3);

      sn=s+(*(ipp++));
      _prefetch_spinor(sn);

      _sse_vector_addsub();
      _mul_cst();

      u+=3;
      _prefetch_su3(u);
      u-=3;
      mul_umat(u);

      _sse_pair_store_up((*w).c1,(*w).c2);
      u+=1;
   }

/********************************** face +2 ***********************************/

   bb+=1;
   ipp=(*bb).ipp;
   w=(*bb).w[l];
   wm=w+(*bb).vol;
   u=(*bb).u;
   sn=s+(*(ipp++));

   for (;w<wm;w++)
   {
      _sse_pair_load((*sn).c1,(*sn).c2);
      _sse_pair_load_up((*sn).c4,(*sn).c3);

      sn=s+(*(ipp++));
      _prefetch_spinor(sn);

      _sse_vector_subadd();
      _mul_cst();

      u+=3;
      _prefetch_su3(u);
      u-=3;
      mul_uinv(u);

      _sse_pair_store_up((*w).c1,(*w).c2);
      u+=1;
   }

/********************************** face -3 ***********************************/

   bb+=1;
   ipp=(*bb).ipp;
   w=(*bb).w[l];
   wm=w+(*bb).vol;   
   u=(*bb).u;
   sn=s+(*(ipp++));

   for (;w<wm;w++)
   {
      _sse_pair_load((*sn).c1,(*sn).c2);
      _sse_pair_load_up((*sn).c3,(*sn).c4);

      sn=s+(*(ipp++));
      _prefetch_spinor(sn);

      _sse_vector_i_addsub();
      _mul_cst();

      u+=3;
      _prefetch_su3(u);
      u-=3;
      mul_umat(u);

      _sse_pair_store_up((*w).c1,(*w).c2);
      u+=1;
   }

/********************************** face +3 ***********************************/

   bb+=1;
   ipp=(*bb).ipp;
   w=(*bb).w[l];
   wm=w+(*bb).vol;
   u=(*bb).u;
   sn=s+(*(ipp++));

   for (;w<wm;w++)
   {
      _sse_pair_load((*sn).c1,(*sn).c2);
      _sse_pair_load_up((*sn).c3,(*sn).c4);

      sn=s+(*(ipp++));
      _prefetch_spinor(sn);

      _sse_vector_i_subadd();
      _mul_cst();

      u+=3;
      _prefetch_su3(u);
      u-=3;
      mul_uinv(u);

      _sse_pair_store_up((*w).c1,(*w).c2);
      u+=1;
   }
}

#else

static weyl chi;
static const weyl w0={{{0.0f}}};
static const spinor s0={{{0.0f}}};


static void mul_umat(weyl *s,su3 *u,weyl *r)
{
   _su3_multiply((*r).c1,*u,(*s).c1);
   _su3_multiply((*r).c2,*u,(*s).c2);
}

static void mul_uinv(weyl *s,su3 *u,weyl *r)
{
   _su3_inverse_multiply((*r).c1,*u,(*s).c1);
   _su3_inverse_multiply((*r).c2,*u,(*s).c2);
}


void Dw_bnd(blk_grid_t grid,int n,int k,int l)
{
   int nb,isw,*ipp;
   float moh;
   su3 *u;
   weyl *w,*wm;
   spinor *s,*sn;
   block_t *b;
   bndry_t *bb;   

   b=blk_list(grid,&nb,&isw);

   if ((n<0)||(n>=nb))
   {
      error_loc(1,1,"Dw_bnd [Dw_bnd.c]",
                "Block grid is not allocated or block number out of range");
      return;
   }   

   b+=n;
   bb=(*b).bb;
   
   if ((k<0)||(k>=(*b).ns)||((*b).u==NULL)||(bb==NULL)||(l>=(*bb).nw))
   {
      error_loc(1,1,"Dw_bnd [Dw_bnd.c]",
                "Attempt to access unallocated memory space");
      return;
   }       

   moh=-0.5f;
   s=(*b).s[k];

/********************************** face -0 ***********************************/

   ipp=(*bb).ipp;
   w=(*bb).w[l];
   wm=w+(*bb).vol;
   
   if ((cpr[0]==0)&&((*b).bo[0]==0))
   {
      for (;w<wm;w++)
      {
         sn=s+(*ipp);
         ipp+=1;
         (*sn)=s0;
         (*w)=w0;
      }
   }
   else
   {
      u=(*bb).u;

      for (;w<wm;w++)
      {
         sn=s+(*ipp);
         ipp+=1;
         _vector_add(chi.c1,(*sn).c1,(*sn).c3);
         _vector_add(chi.c2,(*sn).c2,(*sn).c4);
         _vector_mul(chi.c1,moh,chi.c1);
         _vector_mul(chi.c2,moh,chi.c2);
         mul_umat(&chi,u,w);
         u+=1;
      }
   }

/********************************** face +0 ***********************************/

   bb+=1;
   ipp=(*bb).ipp;
   w=(*bb).w[l];
   wm=w+(*bb).vol;
   
   if ((cpr[0]==(NPROC0-1))&&(((*b).bo[0]+(*b).bs[0])==L0))
   {
      for (;w<wm;w++)
      {
         sn=s+(*ipp);
         ipp+=1;
         (*sn)=s0;
         (*w)=w0;
      }
   }
   else
   {
      u=(*bb).u;

      for (;w<wm;w++)
      {
         sn=s+(*ipp);
         ipp+=1;
         _vector_sub(chi.c1,(*sn).c1,(*sn).c3);
         _vector_sub(chi.c2,(*sn).c2,(*sn).c4);
         _vector_mul(chi.c1,moh,chi.c1);
         _vector_mul(chi.c2,moh,chi.c2);
         mul_uinv(&chi,u,w);
         u+=1;
      }
   }

/********************************** face -1 ***********************************/

   bb+=1;
   ipp=(*bb).ipp;
   w=(*bb).w[l];
   wm=w+(*bb).vol;
   u=(*bb).u;
   
   for (;w<wm;w++)
   {
      sn=s+(*ipp);
      ipp+=1;
      _vector_i_add(chi.c1,(*sn).c1,(*sn).c4);
      _vector_i_add(chi.c2,(*sn).c2,(*sn).c3);
      _vector_mul(chi.c1,moh,chi.c1);
      _vector_mul(chi.c2,moh,chi.c2);
      mul_umat(&chi,u,w);
      u+=1;
   }

/********************************** face +1 ***********************************/

   bb+=1;
   ipp=(*bb).ipp;
   w=(*bb).w[l];
   wm=w+(*bb).vol;
   u=(*bb).u;
   
   for (;w<wm;w++)
   {
      sn=s+(*ipp);
      ipp+=1;
      _vector_i_sub(chi.c1,(*sn).c1,(*sn).c4);
      _vector_i_sub(chi.c2,(*sn).c2,(*sn).c3);
      _vector_mul(chi.c1,moh,chi.c1);
      _vector_mul(chi.c2,moh,chi.c2);
      mul_uinv(&chi,u,w);
      u+=1;
   }

/********************************** face -2 ***********************************/

   bb+=1;
   ipp=(*bb).ipp;
   w=(*bb).w[l];
   wm=w+(*bb).vol;
   u=(*bb).u;
   
   for (;w<wm;w++)
   {
      sn=s+(*ipp);
      ipp+=1;
      _vector_add(chi.c1,(*sn).c1,(*sn).c4);
      _vector_sub(chi.c2,(*sn).c2,(*sn).c3);
      _vector_mul(chi.c1,moh,chi.c1);
      _vector_mul(chi.c2,moh,chi.c2);
      mul_umat(&chi,u,w);
      u+=1;
   }

/********************************** face +2 ***********************************/

   bb+=1;
   ipp=(*bb).ipp;
   w=(*bb).w[l];
   wm=w+(*bb).vol;
   u=(*bb).u;
   
   for (;w<wm;w++)
   {
      sn=s+(*ipp);
      ipp+=1;
      _vector_sub(chi.c1,(*sn).c1,(*sn).c4);
      _vector_add(chi.c2,(*sn).c2,(*sn).c3);
      _vector_mul(chi.c1,moh,chi.c1);
      _vector_mul(chi.c2,moh,chi.c2);
      mul_uinv(&chi,u,w);
      u+=1;
   }

/********************************** face -3 ***********************************/

   bb+=1;
   ipp=(*bb).ipp;
   w=(*bb).w[l];
   wm=w+(*bb).vol;
   u=(*bb).u;
   
   for (;w<wm;w++)
   {
      sn=s+(*ipp);
      ipp+=1;
      _vector_i_add(chi.c1,(*sn).c1,(*sn).c3);
      _vector_i_sub(chi.c2,(*sn).c2,(*sn).c4);
      _vector_mul(chi.c1,moh,chi.c1);
      _vector_mul(chi.c2,moh,chi.c2);
      mul_umat(&chi,u,w);
      u+=1;
   }

/********************************** face +3 ***********************************/

   bb+=1;
   ipp=(*bb).ipp;
   w=(*bb).w[l];
   wm=w+(*bb).vol;
   u=(*bb).u;
   
   for (;w<wm;w++)
   {
      sn=s+(*ipp);
      ipp+=1;
      _vector_i_sub(chi.c1,(*sn).c1,(*sn).c3);
      _vector_i_add(chi.c2,(*sn).c2,(*sn).c4);
      _vector_mul(chi.c1,moh,chi.c1);
      _vector_mul(chi.c2,moh,chi.c2);
      mul_uinv(&chi,u,w);
      u+=1;
   }
}

#endif

