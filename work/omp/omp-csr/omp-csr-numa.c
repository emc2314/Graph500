/* -*- mode: C; mode: folding; fill-column: 70; -*- */
/* Copyright 2010-2011,  Georgia Institute of Technology, USA. */
/* Copyright 2013,  Regents of the University of California, USA. */
/* See COPYING for license. */
#include "../compat.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <assert.h>

#include <alloca.h>

static int64_t int64_fetch_add (int64_t* p, int64_t incr);
static int64_t int64_casval(int64_t* p, int64_t oldval, int64_t newval);
static int int64_cas(int64_t* p, int64_t oldval, int64_t newval);

#include "../graph500.h"
#include "../xalloc.h"
#include "../generator/graph_generator.h"
#include "../timer.h"

#include "bitmap-numa.h"

#define MINVECT_SIZE 0 //2
#define THREAD_BUF_LEN 65536 //16384
#define ALPHA 14
#define BETA  24

#define OMP_NT 80
#define CORE_NT 40

static int64_t mid, maxvtx, nv, sz;
static int64_t *restrict xoff0; /* Length 2*nv+2 */
static int64_t *restrict xoff1;
static int64_t *restrict xadj0; /* Length xoff[nv] == n_edge) */
static int64_t *restrict xadj1;

static int64_t * restrict xoffo0; /* Length 2*mid+2 */
static int64_t * restrict xoffo1; /* Length 2*mid+2 */
static int64_t * restrict xadjo0; /* Length xoffo[nv] == n_edge */
static int64_t * restrict xadjo1;

#define XOFF0(k) (xoff0[2*(k)])
#define XOFF1(k) (xoff1[2*(k)])
#define XENDOFF0(k) (xoff0[1+2*(k)])
#define XENDOFF1(k) (xoff1[1+2*(k)])

#define XOFFO0(k) (xoffo0[2*(k)])
#define XENDOFFO0(k) (xoffo0[1+2*(k)])
#define XOFFO1(k) (xoffo1[2*(k-mid)])
#define XENDOFFO1(k) (xoffo1[1+2*(k-mid)])

/*
static void
find_nvo (const struct packed_edge * restrict IJ, const int64_t nedge)
{
  maxvtx = -1;
  OMP("omp parallel") {
    int64_t k, gmaxvtx, tmaxvtx = -1;

    OMP("omp for")
      for (k = 0; k < nedge; ++k) {
  if (get_v0_from_edge(&IJ[k]) > tmaxvtx)
    tmaxvtx = get_v0_from_edge(&IJ[k]);
  if (get_v1_from_edge(&IJ[k]) > tmaxvtx)
    tmaxvtx = get_v1_from_edge(&IJ[k]);
      }
    gmaxvtx = maxvtx;
    while (tmaxvtx > gmaxvtx)
      gmaxvtx = int64_casval (&maxvtx, gmaxvtx, tmaxvtx);
  }
  nv = 1+maxvtx;
}
*/
static int
alloc_grapho (int64_t nedge)
{
  xoffo0 = nmalloc_large((2*mid+2)*sizeof(*xoffo0), 0);
  xoffo1 = nmalloc_large((2*mid+2)*sizeof(*xoffo1), 1);
  if (!(xoffo0&&xoffo1))
    return -1;
  return 0;
}

static void
free_grapho (void)
{
  nfree_large(xoffo0);
  nfree_large(xadjo0);
  nfree_large(xoffo1);
  nfree_large(xadjo1);
}


static int64_t
prefix_sumo0 (int64_t *buf)
{
  int nt, tid;
  int64_t slice_begin, slice_end, t1, t2, k;

  nt = omp_get_num_threads ();
  tid = omp_get_thread_num ();

  t1 = mid / nt;
  t2 = mid % nt;
  slice_begin = t1 * tid + (tid < t2? tid : t2);
  slice_end = t1 * (tid+1) + ((tid+1) < t2? (tid+1) : t2);

  buf[tid] = 0;
  for (k = slice_begin; k < slice_end; ++k){
    buf[tid] += XOFFO0(k);
  }
  OMP("omp barrier");
  OMP("omp single")
    for (k = 1; k < nt; ++k)
      buf[k] += buf[k-1];
  if (tid)
    t1 = buf[tid-1];
  else
    t1 = 0;
  for (k = slice_begin; k < slice_end; ++k) {
    int64_t tmp = XOFFO0(k);
    XOFFO0(k) = t1;
    t1 += tmp;
  }
  OMP("omp flush (xoffo0)");
  OMP("omp barrier");
  return buf[nt-1];
}

static int64_t
prefix_sumo1 (int64_t *buf)
{
  int nt, tid;
  int64_t slice_begin, slice_end, t1, t2, k;

  nt = omp_get_num_threads ();
  tid = omp_get_thread_num ();

  t1 = mid / nt;
  t2 = mid % nt;
  slice_begin = mid + t1 * tid + (tid < t2? tid : t2);
  slice_end = mid + t1 * (tid+1) + ((tid+1) < t2? (tid+1) : t2);

  buf[tid] = 0;
  for (k = slice_begin; k < slice_end; ++k){
    buf[tid] += XOFFO1(k);
  }
  OMP("omp barrier");
  OMP("omp single")
    for (k = 1; k < nt; ++k)
      buf[k] += buf[k-1];
  if (tid)
    t1 = buf[tid-1];
  else
    t1 = 0;
  for (k = slice_begin; k < slice_end; ++k) {
    int64_t tmp = XOFFO1(k);
    XOFFO1(k) = t1;
    t1 += tmp;
  }
  OMP("omp flush (xoffo1)");
  OMP("omp barrier");
  return buf[nt-1];
}

static int
setup_deg_offo (const struct packed_edge * restrict IJ, int64_t nedge)
{
  int err = 0;
  int64_t *buf = NULL;
  xadjo0 = NULL;
  xadjo1 = NULL;
  OMP("omp parallel") {
    int64_t k, accum0, accum1;
    OMP("omp for")
    for (k = 0; k < 2*mid+2; ++k){
      xoffo0[k] = 0;
      xoffo1[k] = 0;
    }
    OMP("omp for")
    for (k = 0; k < nedge; ++k) {
      int64_t i = get_v0_from_edge(&IJ[k]);
      int64_t j = get_v1_from_edge(&IJ[k]);
      if (i != j) { /* Skip self-edges. */
        if (i >= mid){
          OMP("omp atomic")
          ++XOFFO1(i);
        }else{
          OMP("omp atomic")
          ++XOFFO0(i);
        }
        if (j >= mid){
          OMP("omp atomic")
          ++XOFFO1(j);
        }else{
          OMP("omp atomic")
          ++XOFFO0(j);
        }
      }
    }
    OMP("omp single") {
      buf = alloca (omp_get_num_threads () * sizeof (*buf));
      if (!buf) {
        perror ("alloca for prefix-sum hosed");
        abort ();
      }
    }
    /*
    OMP("omp for")
    for (k = 0; k < nv; ++k){
      if (XOFFO0(k) < MINVECT_SIZE) XOFFO0(k) = MINVECT_SIZE;
      if (XOFFO1(k) < MINVECT_SIZE) XOFFO1(k) = MINVECT_SIZE;
    }
    */
    OMP("barrier");
    accum0 = prefix_sumo0(buf);
    OMP("barrier");
    accum1 = prefix_sumo1(buf);

    OMP("omp for")
    for (k = 0; k < nv; ++k){
      if(k<mid)
        XENDOFFO0(k) = XOFFO0(k);
      else
        XENDOFFO1(k) = XOFFO1(k);
    }
    OMP("omp single") {
      XOFFO0(mid) = accum0;
      XOFFO1(nv) = accum1;
      if (!((xadjo1 = nmalloc_large(XOFFO1(nv) * sizeof (*xadjo1), 1))&&(xadjo0 = nmalloc_large(XOFFO0(mid)*sizeof(*xadjo0), 0))))
        err = -1;
      if (!err) {
        for (k = 0; k < XOFFO0(mid); ++k)
          xadjo0[k] = -1;
        for (k = mid; k < XOFFO1(nv); ++k)
          xadjo1[k] = -1;
      }
    }
  }
  return !(xadjo0&&xadjo1);
}

static void
scatter_edgeo (const int64_t i, const int64_t j)
{
  int64_t where;
  if(i<mid){
    where = int64_fetch_add (&XENDOFFO0(i), 1);
    xadjo0[where] = j;
  }else{
    where = int64_fetch_add (&XENDOFFO1(i), 1);
    xadjo1[where] = j;
  }
}

static int
i64cmpo (const void *a, const void *b)
{
  const int64_t ia = *(const int64_t*)a;
  const int64_t ib = *(const int64_t*)b;
  if (ia < ib) return -1;
  if (ia > ib) return 1;
  return 0;
}

static int
vdcmpo (const void *a, const void *b)
{
  int64_t ia, ib;
  if(*(int64_t *) a < mid)
    ia = XENDOFFO0(*(int64_t *) a)-XOFFO0(*(int64_t *) a);
  else
    ia = XENDOFFO1(*(int64_t *) a)-XOFFO1(*(int64_t *) a);
  if(*(int64_t *) b < mid)
    ib = XENDOFFO0(*(int64_t *) b)-XOFFO0(*(int64_t *) b);
  else
    ib = XENDOFFO1(*(int64_t *) b)-XOFFO1(*(int64_t *) b);
  if (ia < ib) return 1;
  if (ia > ib) return -1;
  return 0;
}


static void
pack_vtx_edgeso (const int64_t i)
{
  int64_t kcur, k;
  if(i<mid){
    if (XOFFO0(i)+1 >= XENDOFFO0(i)) return;
    qsort (&xadjo0[XOFFO0(i)], XENDOFFO0(i)-XOFFO0(i), sizeof(*xadjo0), i64cmpo);
    kcur = XOFFO0(i);
    for (k = XOFFO0(i)+1; k < XENDOFFO0(i); ++k)
      if (xadjo0[k] != xadjo0[kcur])
        xadjo0[++kcur] = xadjo0[k];
    ++kcur;
    for (k = kcur; k < XENDOFFO0(i); ++k)
      xadjo0[k] = -1;
    XENDOFFO0(i) = kcur;
  }else{
    if (XOFFO1(i)+1 >= XENDOFFO1(i)) return;
    qsort (&xadjo1[XOFFO1(i)], XENDOFFO1(i)-XOFFO1(i), sizeof(*xadjo1), i64cmpo);
    kcur = XOFFO1(i);
    for (k = XOFFO1(i)+1; k < XENDOFFO1(i); ++k)
      if (xadjo1[k] != xadjo1[kcur])
        xadjo1[++kcur] = xadjo1[k];
    ++kcur;
    for (k = kcur; k < XENDOFFO1(i); ++k)
      xadjo1[k] = -1;
    XENDOFFO1(i) = kcur;
  }
}

static void
pack_edgeso (void)
{
  int64_t v;

  OMP("omp for")
  for (v = 0; v < nv; ++v)
    pack_vtx_edgeso (v);
  ///*
  OMP("omp for")
  for (v = 0; v < nv; ++v){
    if(v<mid){
      //qsort (&xadj[XOFF(v)], XENDOFF(v)-XOFF(v), sizeof(*xadj), vdcmp);
      int64_t max=xadjo0[XOFFO0(v)];
      for(int64_t i=XOFFO0(v); i<XENDOFFO0(v);i++){
        if(vdcmpo(&xadjo0[i], &max) < 0){
            max=xadjo0[i];
        }
      }
      int64_t temp,temp2;
      //temp=xadj[XOFF(v)];
      //xadj[XOFF(v)]=xadj[max];
      //xadj[max]=temp;
      for(int64_t i=XOFFO0(v), temp=xadjo0[i], temp2=xadjo0[i];i<XENDOFFO0(v);i++){
        if(temp2!=max){
            temp2=xadjo0[i+1];
            xadjo0[i+1]=temp;
            temp=temp2;
        }else{
          break;
        }
      }
      xadjo0[XOFFO0(v)]=max;
    }else{
      //qsort (&xadj[XOFF(v)], XENDOFF(v)-XOFF(v), sizeof(*xadj), vdcmp);
      int64_t max=xadjo1[XOFFO1(v)];
      for(int64_t i=XOFFO1(v); i<XENDOFFO1(v);i++){
        if(vdcmpo(&xadjo1[i], &max) < 0){
            max=xadjo1[i];
        }
      }
      int64_t temp,temp2;
      //temp=xadj[XOFF(v)];
      //xadj[XOFF(v)]=xadj[max];
      //xadj[max]=temp;
      for(int64_t i=XOFFO1(v), temp=xadjo1[i], temp2=xadjo1[i];i<XENDOFFO1(v);i++){
        if(temp2!=max){
            temp2=xadjo1[i+1];
            xadjo1[i+1]=temp;
            temp=temp2;
        }else{
          break;
        }
      }
      xadjo1[XOFFO1(v)]=max;
    }
  }
  //*/
}

static void
gather_edgeso (const struct packed_edge * restrict IJ, int64_t nedge)
{
  OMP("omp parallel") {
    int64_t k;

    OMP("omp for")
    for (k = 0; k < nedge; ++k) {
      int64_t i = get_v0_from_edge(&IJ[k]);
      int64_t j = get_v1_from_edge(&IJ[k]);
      if (i >= 0 && j >= 0 && i != j) {
        scatter_edgeo (i, j);
        scatter_edgeo (j, i);
      }
    }

    pack_edgeso ();
  }
}

int
create_graph_from_edgelisto (struct packed_edge *IJ, int64_t nedge)
{
  if (alloc_grapho (nedge)) return -1;
  if (setup_deg_offo (IJ, nedge)) {
    abort();
  }
  gather_edgeso (IJ, nedge);
  return 0;
}







static void
find_nv (const struct packed_edge * restrict IJ, const int64_t nedge)
{
  maxvtx = -1;
  OMP("omp parallel") {
    int64_t k, gmaxvtx, tmaxvtx = -1;

    OMP("omp for")
    for (k = 0; k < nedge; ++k) {
      if (get_v0_from_edge(&IJ[k]) > tmaxvtx)
        tmaxvtx = get_v0_from_edge(&IJ[k]);
      if (get_v1_from_edge(&IJ[k]) > tmaxvtx)
        tmaxvtx = get_v1_from_edge(&IJ[k]);
    }
    gmaxvtx = maxvtx;
    while (tmaxvtx > gmaxvtx)
      gmaxvtx = int64_casval (&maxvtx, gmaxvtx, tmaxvtx);
  }
  nv = 1 + maxvtx;

  if (nv % 2) {
      nv++;
      maxvtx++;
  }

  mid = nv / 2;
}

static int
alloc_graph (int64_t nedge)
{
  sz = (2 * nv + 2) * sizeof(*xoff0);
  xoff0 = nmalloc_large(sz, 0);
  xoff1 = nmalloc_large(sz, 1);
  if (!(xoff0 && xoff1))
    return -1;
  return 0;
}

static void
free_graph (void)
{
  nfree_large(xoff0);
  nfree_large(xadj0);
  nfree_large(xoff1);
  nfree_large(xadj1);
}


static int64_t
prefix_sum0 (int64_t *buf)
{
  int nt, tid;
  int64_t slice_begin, slice_end, t1, t2, k;

  nt = OMP_NT;
  tid = omp_get_thread_num ();

  t1 = nv / nt;
  t2 = nv % nt;
  slice_begin = t1 * tid + (tid < t2? tid : t2);
  slice_end = t1 * (tid+1) + ((tid+1) < t2? (tid+1) : t2);

  buf[tid] = 0;
  for (k = slice_begin; k < slice_end; ++k)
    buf[tid] += XOFF0(k);
  OMP("omp barrier");
  OMP("omp single")
  for (k = 1; k < nt; ++k)
    buf[k] += buf[k-1];
  if (tid)
    t1 = buf[tid-1];
  else
    t1 = 0;
  for (k = slice_begin; k < slice_end; ++k) {
    int64_t tmp = XOFF0(k);
    XOFF0(k) = t1;
    t1 += tmp;
  }
  OMP("omp flush (xoff0)");
  OMP("omp barrier");
  return buf[nt-1];
}

static int64_t
prefix_sum1 (int64_t *buf)
{
  int nt, tid;
  int64_t slice_begin, slice_end, t1, t2, k;

  nt = OMP_NT;
  tid = omp_get_thread_num ();

  t1 = nv / nt;
  t2 = nv % nt;
  slice_begin = t1 * tid + (tid < t2? tid : t2);
  slice_end = t1 * (tid+1) + ((tid+1) < t2? (tid+1) : t2);

  buf[tid] = 0;
  for (k = slice_begin; k < slice_end; ++k)
    buf[tid] += XOFF1(k);
  OMP("omp barrier");
  OMP("omp single")
  for (k = 1; k < nt; ++k)
    buf[k] += buf[k-1];
  if (tid)
    t1 = buf[tid-1];
  else
    t1 = 0;
  for (k = slice_begin; k < slice_end; ++k) {
    int64_t tmp = XOFF1(k);
    XOFF1(k) = t1;
    t1 += tmp;
  }
  OMP("omp flush (xoff1)");
  OMP("omp barrier");
  return buf[nt-1];
}

static int
setup_deg_off (const struct packed_edge * restrict IJ, int64_t nedge)
{
  int64_t *buf = NULL;
  xadj0 = NULL;
  xadj1 = NULL;
  OMP("omp parallel") {
    int64_t k, accum0, accum1;
    OMP("omp for")
    for (k = 0; k < 2*nv+2; ++k){
      xoff0[k] = 0;
      xoff1[k] = 0;
    }
    OMP("omp for")
    for (k = 0; k < nedge; ++k) {
      int64_t i = get_v0_from_edge(&IJ[k]);
      int64_t j = get_v1_from_edge(&IJ[k]);
      if (i != j) { /* Skip self-edges. */
        if (i >= 0) {
          if (j >= mid){
            OMP("omp atomic")
            ++XOFF1(i);
          }
          else{
            OMP("omp atomic")
            ++XOFF0(i);
          }
        }
        if (j >= 0){
          if (i >= mid){
            OMP("omp atomic")
            ++XOFF1(j);
          }
          else{
            OMP("omp atomic")
            ++XOFF0(j);
          }
        }
      }
    }
    OMP("omp single") {
      buf = alloca (OMP_NT * sizeof (*buf));
      if (!buf) {
        perror ("alloca for prefix-sum hosed");
        abort ();
      }
    }
    OMP("omp for")
    for (k = 0; k < nv; ++k){
      if (XOFF0(k) < MINVECT_SIZE)
        XOFF0(k) = MINVECT_SIZE;
      if (XOFF1(k) < MINVECT_SIZE)
        XOFF1(k) = MINVECT_SIZE;
    }
    OMP("omp barrier");
    accum0 = prefix_sum0 (buf);
    OMP("omp barrier");
    accum1 = prefix_sum1 (buf);

    OMP("omp for")
    for (k = 0; k < nv; ++k){
      XENDOFF0(k) = XOFF0(k);
      XENDOFF1(k) = XOFF1(k);
    }
    OMP("omp single") {
      XOFF0(nv) = accum0;
      XOFF1(nv) = accum1;
      xadj0 = nmalloc_large(XOFF0(nv) * sizeof (*xadj0), 0);
      xadj1 = nmalloc_large(XOFF1(nv) * sizeof (*xadj1), 1);
    }
  }
  return !(xadj0 && xadj1);
}

static void
scatter_edge (const int64_t i, const int64_t j)
{
  int64_t where;
  if(j >= mid){
    where = int64_fetch_add (&XENDOFF1(i), 1);
    xadj1[where] = j;
  }
  else{
    where = int64_fetch_add (&XENDOFF0(i), 1);
    xadj0[where] = j;
  }
}

static int
i64cmp (const void *a, const void *b)
{
  const int64_t ia = *(const int64_t*)a;
  const int64_t ib = *(const int64_t*)b;
  if (ia < ib) return -1;
  if (ia > ib) return 1;
  return 0;
}

static int
vdcmp0 (const void *a, const void *b)
{
  const int64_t ia = XENDOFF0(*(int64_t *) a)-XOFF0(*(int64_t *) a);
  const int64_t ib = XENDOFF0(*(int64_t *) b)-XOFF0(*(int64_t *) b);
  if (ia < ib) return 1;
  if (ia > ib) return -1;
  return 0;
}

static int
vdcmp1 (const void *a, const void *b)
{
  const int64_t ia = XENDOFF1(*(int64_t *) a)-XOFF1(*(int64_t *) a);
  const int64_t ib = XENDOFF1(*(int64_t *) b)-XOFF1(*(int64_t *) b);
  if (ia < ib) return 1;
  if (ia > ib) return -1;
  return 0;
}

static void
pack_vtx_edges0 (const int64_t i)
{
  int64_t kcur, k;
  if (XOFF0(i)+1 >= XENDOFF0(i)) return;
  qsort (&xadj0[XOFF0(i)], XENDOFF0(i)-XOFF0(i), sizeof(*xadj0), i64cmp);
  kcur = XOFF0(i);
  for (k = XOFF0(i)+1; k < XENDOFF0(i); ++k)
    if (xadj0[k] != xadj0[kcur])
      xadj0[++kcur] = xadj0[k];
  ++kcur;
  for (k = kcur; k < XENDOFF0(i); ++k)
    xadj0[k] = -1;
  XENDOFF0(i) = kcur;

  int64_t max=xadj0[XOFF0(i)];
  for(int64_t v=XOFF0(i); v<XENDOFF0(i);v++){
    if(vdcmp0(&xadj0[v], &max) < 0){
      max=xadj0[v];
    }
  }
  int64_t temp,temp2;
  //temp=xadj0[XOFF0(v)];
  //xadj0[XOFF0(v)]=xadj0[max];
  //xadj0[max]=temp;
  for(int64_t v=XOFF0(i), temp=xadj0[v], temp2=xadj0[v];v<XENDOFF0(i);v++){
    if(temp2!=max){
      temp2=xadj0[v+1];
      xadj0[v+1]=temp;
      temp=temp2;
    }else{
      break;
    }
  }
  xadj0[XOFF0(i)]=max;
}

static void
pack_vtx_edges1 (const int64_t i)
{
  int64_t kcur, k;
  if (XOFF1(i)+1 >= XENDOFF1(i)) return;
  qsort (&xadj1[XOFF1(i)], XENDOFF1(i)-XOFF1(i), sizeof(*xadj1), i64cmp);
  kcur = XOFF1(i);
  for (k = XOFF1(i)+1; k < XENDOFF1(i); ++k)
    if (xadj1[k] != xadj1[kcur])
      xadj1[++kcur] = xadj1[k];
  ++kcur;
  for (k = kcur; k < XENDOFF1(i); ++k)
    xadj1[k] = -1;
  XENDOFF1(i) = kcur;

  int64_t max=xadj1[XOFF1(i)];
  for(int64_t v=XOFF1(i); v<XENDOFF1(i);v++){
    if(vdcmp1(&xadj1[v], &max) < 0){
      max=xadj1[v];
    }
  }
  int64_t temp,temp2;
  //temp=xadj1[XOFF1(v)];
  //xadj1[XOFF1(v)]=xadj1[max];
  //xadj1[max]=temp;
  for(int64_t v=XOFF1(i), temp=xadj1[v], temp2=xadj1[v];v<XENDOFF1(i);v++){
    if(temp2!=max){
      temp2=xadj1[v+1];
      xadj1[v+1]=temp;
      temp=temp2;
    }else{
      break;
    }
  }
  xadj1[XOFF1(i)]=max;
}

static void
pack_edges (void)
{
  int64_t v;

  OMP("omp for")
    for (v = 0; v < nv; ++v){
      pack_vtx_edges0(v);
      pack_vtx_edges1(v);
    }
}

static void
gather_edges (const struct packed_edge * restrict IJ, int64_t nedge)
{
  OMP("omp parallel") {
    int64_t k;

    OMP("omp for")
    for (k = 0; k < nedge; ++k) {
      int64_t i = get_v0_from_edge(&IJ[k]);
      int64_t j = get_v1_from_edge(&IJ[k]);
      if (i >= 0 && j >= 0 && i != j) {
        scatter_edge (i, j);
        scatter_edge (j, i);
      }
    }

    pack_edges ();
  }
}


void verify(void){
  fprintf(stderr,"verifying graph\n");
  OMP("omp parallel for")
  for(int64_t v=0;v<nv;v++){
    int flag;
    if(v<mid){
      if(XENDOFFO0(v)-XOFFO0(v)!=XENDOFF0(v)+XENDOFF1(v)-XOFF0(v)-XOFF1(v))
        printf("vertex %ld degree: %ld != %ld\n", v, XENDOFFO0(v)-XOFFO0(v), XENDOFF0(v)+XENDOFF1(v)-XOFF0(v)-XOFF1(v));
      for(int64_t i=XOFF0(v);i<XENDOFF0(v);i++){
        flag=1;
        for(int64_t j=XOFFO0(v);j<XENDOFFO0(v);j++)
          if(xadj0[i]==xadjo0[j])
            flag=0;
        if(flag){
          fprintf(stderr, "failed\n");
        }
      }
      for(int64_t i=XOFF1(v);i<XENDOFF1(v);i++){
        flag=1;
        for(int64_t j=XOFFO0(v);j<XENDOFFO0(v);j++)
          if(xadj1[i]==xadjo0[j])
            flag=0;
        if(flag){
          fprintf(stderr,"failed\n");
          abort();
        }
      }
    }else{
      if(XENDOFFO1(v)-XOFFO1(v)!=XENDOFF0(v)+XENDOFF1(v)-XOFF0(v)-XOFF1(v))
        printf("vertex %ld degree: %ld != %ld\n", v, XENDOFFO1(v)-XOFFO1(v), XENDOFF0(v)+XENDOFF1(v)-XOFF0(v)-XOFF1(v));
      for(int64_t i=XOFF0(v);i<XENDOFF0(v);i++){
        flag=1;
        for(int64_t j=XOFFO1(v);j<XENDOFFO1(v);j++)
          if(xadj0[i]==xadjo1[j])
            flag=0;
        if(flag){
          fprintf(stderr,"failed\n");
        }
      }
      for(int64_t i=XOFF1(v);i<XENDOFF1(v);i++){
        flag=1;
        for(int64_t j=XOFFO1(v);j<XENDOFFO1(v);j++)
          if(xadj1[i]==xadjo1[j])
            flag=0;
        if(flag){
          fprintf(stderr,"failed\n");
          abort();
        }
      }
    }
  }
    fprintf(stderr,"done\n");
}


int
create_graph_from_edgelist (struct packed_edge *IJ, int64_t nedge)
{
  find_nv (IJ, nedge);
  if (alloc_graph (nedge)) return -1;
  if (setup_deg_off (IJ, nedge)) {
    nfree_large(xoff0);
    nfree_large(xoff1);
    return -1;
  }
  gather_edges (IJ, nedge);
  create_graph_from_edgelisto(IJ, nedge);
  //verify();
  return 0;
}






static void
fill_bitmap_from_queue(bitmap_T *bm, int64_t *vlist, int64_t k)
{
  OMP("omp for")
    for (int64_t q_index=0; q_index<k; q_index++)
      bm_set_bit_atomic(bm, vlist[q_index]);
}

static void
fill_queue_from_bitmap(bitmap_T *bm, int64_t *vlist, int64_t *in,
               int64_t *local)
{
  OMP("omp single") {
    *in = 0;
  }
  OMP("omp barrier");
  int64_t nodes_per_thread = (nv + OMP_NT - 1) /
                 OMP_NT;
  int64_t i = nodes_per_thread * omp_get_thread_num();
  int local_index = 0;
  if (i < nv) {
    int64_t i_end = i + nodes_per_thread;
    if (i_end >= nv)
      i_end = nv;
    if (bm_get_bit(bm, i))
      local[local_index++] = i;
    i = bm_get_next_bit(bm,i);
    while ((i != -1) && (i < i_end)) {
      local[local_index++] = i;
      i = bm_get_next_bit(bm,i);
      /*if (local_index == THREAD_BUF_LEN) {
        int my_in = int64_fetch_add(in, THREAD_BUF_LEN);
        for (local_index=0; local_index<THREAD_BUF_LEN; local_index++) {
          vlist[my_in + local_index] = local[local_index];
        }
        local_index = 0;
      }*/
    }
  }
  int my_in = int64_fetch_add(in, local_index);
  for (int i=0; i<local_index; i++) {
    vlist[my_in + i] = local[i];
  }
}

static int64_t
bfs_bottom_up_step(int64_t *bfs_tree, bitmap_T *past, bitmap_T *next, int64_t node, int64_t tid)
{
    /*
  static int64_t *buf = NULL;
  OMP("omp single") {
    buf = alloca (OMP_NT * sizeof (*buf));
    if (!buf) {
      perror ("alloca for Bottom-up hosed");
      abort ();
    }
    memset(buf,0,OMP_NT * sizeof (*buf));
  }
  OMP("omp barrier");
    */
  //fprintf(stderr, "?\n");
  OMP("omp single") {
    bm_swap(past, next);
  }
  bm_reset(next);
  OMP("omp barrier");
  int64_t count = 0;
  static int64_t awake_count;
  //static int64_t trvt;
  OMP("omp single"){
    awake_count = 0;
    //trvt=0;
  }
  //fprintf(stderr, "!\n");
  OMP("omp barrier");
  //OMP("omp for reduction(+ : awake_count)")//, trvt)")
  if(node == 0){
    const int64_t t1 = mid / CORE_NT;
    const int64_t t2 = mid % CORE_NT;
    const int64_t slice_begin = t1 * tid + (tid < t2? tid : t2);
    const int64_t slice_end = t1 * (tid+1) + ((tid+1) < t2? (tid+1) : t2);
    //fprintf(stderr, "node:%d\t,%ld,%ld\n",(*ni).node,slice_begin,slice_end);
    for (int64_t k = slice_begin; k < slice_end; ++k) {
      if (bfs_tree[k] == -1) {
        for (int64_t vo = XOFFO0(k); vo < XENDOFFO0(k); vo++) {
          const int64_t j = xadjo0[vo];
          if (bm_get_bit(past, j)) {
            // printf("%lu\n",i);
            bfs_tree[k] = j;
            bm_set_bit_atomic(next, k);
            count++;
            //trvt+=vo-XOFF(i)+1;
            break;
          }
        }
      }
    }
  }else{
    const int64_t t1 = mid / CORE_NT;
    const int64_t t2 = mid % CORE_NT;
    const int64_t slice_begin = mid + t1 * tid + (tid < t2? tid : t2);
    const int64_t slice_end = mid + t1 * (tid+1) + ((tid+1) < t2? (tid+1) : t2);
    //fprintf(stderr, "node:%d\t,%ld,%ld\n",(*ni).node,slice_begin,slice_end);
    for (int64_t k = slice_begin; k < slice_end; ++k) {
      if (bfs_tree[k] == -1) {
        for (int64_t vo = XOFFO1(k); vo < XENDOFFO1(k); vo++) {
          const int64_t j = xadjo1[vo];
          if (bm_get_bit(past, j)) {
            // printf("%lu\n",i);
            bfs_tree[k] = j;
            bm_set_bit_atomic(next, k);
            count++;
            //trvt+=vo-XOFF(i)+1;
            break;
          }
        }
      }
    }
  }
  OMP("omp atomic")
      awake_count += count;
  OMP("omp barrier");
  //OMP("omp single") {
  //fprintf(stderr, "%lu\n", trvt);
  //}
  return awake_count;
}




static void
bfs_top_down_step(int64_t *bfs_tree, int64_t *vlist0, int64_t *local, int64_t *k0_p, int64_t node, int64_t tid)
{
  int64_t oldk0 = *k0_p;
  int64_t kbuf = 0, voff;
  OMP("omp barrier");
  if(node == 0){
    const int64_t t1 = oldk0 / CORE_NT;
    const int64_t t2 = oldk0 % CORE_NT;
    const int64_t slice_begin = t1 * tid + (tid < t2? tid : t2);
    const int64_t slice_end = t1 * (tid+1) + ((tid+1) < t2? (tid+1) : t2);
    //fprintf(stderr, "node:%d\t,%ld,%ld\n",(*ni).node,slice_begin,slice_end);
    for (int64_t k = slice_begin; k < slice_end; ++k) {
      const int64_t v = vlist0[k];
      const int64_t veo = XENDOFF0(v);
      int64_t vo;
      for (vo = XOFF0(v); vo < veo; ++vo) {
        //fprintf(stderr, "%ld->%ld\n", vo, veo);
        const int64_t j = xadj0[vo];
        //fprintf(stderr, "%ld\n", j);
        if (bfs_tree[j] == -1) {
          //fprintf(stderr,"%ld\t", j);
          if (int64_cas (&bfs_tree[j], -1, v)) {
        //if (kbuf < THREAD_BUF_LEN) {
            local[kbuf++] = j;
        //} else {
        //  int64_t voff = int64_fetch_add (k2_p, THREAD_BUF_LEN), vk;
        //  assert (voff + THREAD_BUF_LEN <= nv);
        //  for (vk = 0; vk < THREAD_BUF_LEN; ++vk)
        //    vlist[voff + vk] = local[vk];
        //  local[0] = j;
        //  kbuf = 1;
        //}
          }
        }
      }
    }
  }else{
    const int64_t t1 = oldk0 / CORE_NT;
    const int64_t t2 = oldk0 % CORE_NT;
    const int64_t slice_begin = t1 * tid + (tid < t2? tid : t2);
    const int64_t slice_end = t1 * (tid+1) + ((tid+1) < t2? (tid+1) : t2);
    //fprintf(stderr, "node:%d\t,%ld,%ld\n",(*ni).node,slice_begin,slice_end);
    for (int64_t k = slice_begin; k < slice_end; ++k) {
      const int64_t v = vlist0[k];
      const int64_t veo = XENDOFF1(v);
      int64_t vo;
      for (vo = XOFF1(v); vo < veo; ++vo) {
        const int64_t j = xadj1[vo];
        if (bfs_tree[j] == -1) {
          //fprintf(stderr,"%ld\t", j);
          if (int64_cas (&bfs_tree[j], -1, v)) {
        //if (kbuf < THREAD_BUF_LEN) {
            local[kbuf++] = j;
        //} else {
        //  int64_t voff = int64_fetch_add (k2_p, THREAD_BUF_LEN), vk;
        //  assert (voff + THREAD_BUF_LEN <= nv);
        //  for (vk = 0; vk < THREAD_BUF_LEN; ++vk)
        //    vlist[voff + vk] = local[vk];
        //  local[0] = j;
        //  kbuf = 1;
        //}
          }
        }
      }
    }
  }
  OMP("omp single")
    *k0_p = 0;
  OMP("omp barrier"); //?
  if (kbuf) {
    voff = int64_fetch_add (k0_p, kbuf);
    //fprintf(stderr, "\n%ld\t%ld\n", voff, *ktmp);
    for (int64_t vk = 0; vk < kbuf; ++vk){
      vlist0[voff + vk] = local[vk];
    }
  }
  OMP("omp barrier");
  /*
  OMP("omp single"){
    *k1_p=*k0_p;
  }
  OMP("omp barrier");
  */
  //*k1_p = *k0_p;

  return;
}

int
make_bfs_tree (int64_t *bfs_tree_out, int64_t *max_vtx_out,
           int64_t srcvtx)
{
  int64_t * restrict bfs_tree = bfs_tree_out;

  int64_t * restrict vlist0 = NULL;
  int64_t * restrict vlist1 = NULL;
  int64_t k0;

  *max_vtx_out = maxvtx;

  vlist0 = nmalloc (nv * sizeof (*vlist0), 0);

  if (!vlist0) abort();

  vlist0[0] = srcvtx;

  k0 = 1;
  bfs_tree[srcvtx] = srcvtx;

  bitmap_T past;
  bitmap_T next;
  bm_init(&past, nv);
  bm_init(&next, nv);

  if (!(past.start && next.start)) abort();

  const int64_t down_cutoff = nv / BETA;
  int64_t scout_count = XENDOFF0(srcvtx) - XOFF0(srcvtx) + XENDOFF1(srcvtx) - XOFF1(srcvtx);

  OMP("omp parallel shared(k0, scout_count)") {
    int64_t k;
    int64_t *nbuf = (int64_t *)malloc(THREAD_BUF_LEN * sizeof(int64_t));
    int64_t awake_count = 1;
    int64_t edges_to_check = XOFF0(nv) + XOFF1(nv);

    const int tid = omp_get_thread_num() % (omp_get_num_threads() / 2);
    const int node = omp_get_thread_num() < (omp_get_num_threads() / 2);

    OMP("omp for")
    for (k = 0; k < srcvtx; ++k)
      bfs_tree[k] = -1;
    OMP("omp for")
    for (k = srcvtx+1; k < nv; ++k)
      bfs_tree[k] = -1;

    while (awake_count != 0) {
      // Top-down
      if (scout_count < ((edges_to_check - scout_count)/ALPHA)) {
//        OMP("omp barrier");
        bfs_top_down_step(bfs_tree, vlist0, nbuf, &k0, node, tid);

        edges_to_check -= scout_count;
        awake_count = k0;
      // Bottom-up
      } else {
        fill_bitmap_from_queue(&next, vlist0, k0);
        //fprintf(stderr, "next0:%lu\n", bm_get_num(&next0));
        do {
          awake_count = bfs_bottom_up_step(bfs_tree, &past, &next, node, tid);
        } while ((awake_count > down_cutoff));
        fill_queue_from_bitmap(&next, vlist0, &k0, nbuf);
//        OMP("omp barrier");
      }
      // Count the number of edges in the frontier
      OMP("omp single")
      scout_count = 0;
      OMP("omp for reduction(+ : scout_count)")
      for (int64_t i=0; i<k0; i++) {
        int64_t v = vlist0[i];
        scout_count += XENDOFF0(v) - XOFF0(v) + XENDOFF1(v) - XOFF1(v);
      }
    }
    free(nbuf);
  }

  //fprintf(stderr, "\n");

  bm_free(&past);
  bm_free(&next);
  nfree_large(vlist0);

  return 0;
}

void
destroy_graph (void)
{
  free_graph ();
  free_grapho ();
  //fprintf(stderr, "Too lazy to free mem.\n");
}

#if defined(_OPENMP)
#ifndef SYS//defined(__GNUC__)||defined(__INTEL_COMPILER)
#warning "SYS"
int64_t
int64_fetch_add (int64_t* p, int64_t incr)
{
  return __sync_fetch_and_add (p, incr);
}
int64_t
int64_casval(int64_t* p, int64_t oldval, int64_t newval)
{
  return __sync_val_compare_and_swap (p, oldval, newval);
}
int
int64_cas(int64_t* p, int64_t oldval, int64_t newval)
{
  return __sync_bool_compare_and_swap (p, oldval, newval);
}
#else
#warning "OMP"
/* XXX: These are not correct, but suffice for the above uses. */
int64_t
int64_fetch_add (int64_t* p, int64_t incr)
{
  int64_t t;
  OMP("omp critical") {
    t = *p;
    *p += incr;
  }
  OMP("omp flush (p)");
  return t;
}
int64_t
int64_casval(int64_t* p, int64_t oldval, int64_t newval)
{
  int64_t v;
  OMP("omp critical (CAS)") {
    v = *p;
    if (v == oldval)
      *p = newval;
  }
  OMP("omp flush (p)");
  return v;
}
int
int64_cas(int64_t* p, int64_t oldval, int64_t newval)
{
  int out = 0;
  OMP("omp critical (CAS)") {
    int64_t v = *p;
    if (v == oldval) {
      *p = newval;
      out = 1;
    }
  }
  OMP("omp flush (p)");
  return out;
}
#endif
#else
int64_t
int64_fetch_add (int64_t* p, int64_t incr)
{
  int64_t t = *p;
  *p += incr;
  return t;
}
int64_t
int64_casval(int64_t* p, int64_t oldval, int64_t newval)
{
  int64_t v = *p;
  if (v == oldval)
    *p = newval;
  return v;
}
int
int64_cas(int64_t* p, int64_t oldval, int64_t newval)
{
  int64_t v = *p;
  int out = 0;
  if (v == oldval) {
    *p = newval;
    out = 1;
  }
  return out;
}
#endif
