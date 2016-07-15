/* -*- mode: C; mode: folding; fill-column: 70; -*- */
/* Copyright 2010-2011,  Georgia Institute of Technology, USA. */
/* Copyright 2013,  Regents of the University of California, USA. */
/* See COPYING for license. */
#include "../compat.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include <assert.h>

#include <alloca.h>

static int64_t int64_fetch_add (int64_t* p, int64_t incr);
static int64_t int64_casval(int64_t* p, int64_t oldval, int64_t newval);
static int int64_cas(int64_t* p, int64_t oldval, int64_t newval);

#include "../graph500.h"
#include "../xalloc.h"
#include "../generator/graph_generator.h"
#include "../timer.h"

#include "bitmap.h"

#define THREAD_BUF_LEN 262144//65536
#define ALPHA 14
#define BETA  24
#define MAX_DEG_NUM 8

static int64_t ipart, npart, mpart; //npart is number of vertices per process, ipart=rank*npart, mpart=npart/2
static int64_t maxvtx, nv, pnv, awake_count, scout_count, edges_to_check; //pnv is previous nv
static int nt_cpu; //threads per CPU
static int rank, nodes; //Process information

typedef struct {
  int64_t index;
  int64_t degree;
} Record;

Record * restrict record; //record the degree and index of a vertex
static int64_t * restrict new2old; //
static int64_t * restrict old2new; // transformation

/* Devide the graph into 4*nodes parts:
 *
 * td0\
 * td1 |
 *     | for every process
 * bu0 |
 * bu1/
 *
 * td and bu stands for top-down and bottom-up
 * 0 and 1 stands for numa cpu in a single node
*/
static int64_t * restrict xofftd0; /* Length 2*nv+2 */
static int64_t * restrict xadjtd0; /* Length xofftd[nv] */
static int64_t * restrict xoffbu0; /* Length 2*mpart+2 */
static int64_t * restrict xadjbu0; /* Length xoffbu[mpart] */

static int64_t * restrict xofftd1; /* Length 2*nv+2 */
static int64_t * restrict xadjtd1; /* Length xofftd[nv] */
static int64_t * restrict xoffbu1; /* Length 2*mpart+2 */
static int64_t * restrict xadjbu1; /* Length xoffbu[mpart] */

#define XOFFTD(k) (xofftd[2*(k)])
#define XENDOFFTD(k) (xofftd[1+2*(k)])
#define XOFFBU(k) (xoffbu[2*(k-ipart-part)])
#define XENDOFFBU(k) (xoffbu[1+2*(k-ipart-part)])

#define XOFFTD0(k) (xofftd0[2*(k)])
#define XOFFBU0(k) (xoffbu0[2*(k-ipart)])
#define XENDOFFTD0(k) (xofftd0[1+2*(k)])
#define XENDOFFBU0(k) (xoffbu0[1+2*(k-ipart)])

#define XOFFTD1(k) (xofftd1[2*(k)])
#define XOFFBU1(k) (xoffbu1[2*(k-ipart-mpart)])
#define XENDOFFTD1(k) (xofftd1[1+2*(k)])
#define XENDOFFBU1(k) (xoffbu1[1+2*(k-ipart-mpart)])

static void
find_nv (const struct packed_edge * restrict IJ, const int64_t nedge) //find the number of vertices
{
  maxvtx = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nodes);
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
    nt_cpu = omp_get_num_threads() / 2;
  }

  pnv = 1 + maxvtx;
  //nv = pnv + 128 * nodes - (pnv % (128 * nodes));
}

static int
alloc_graph (int64_t nedge) //alloc the offset array
{
  int64_t sz = (2*nv+2) * sizeof (*xofftd0);
  xofftd0 = nmalloc_large (sz, 0); //a wrapper to alloc memory on a certain numa cpu
  xofftd1 = nmalloc_large (sz, 1);

  sz = (2*mpart+2) * sizeof (*xoffbu0);
  xoffbu0 = nmalloc_large (sz, 0);
  xoffbu1 = nmalloc_large (sz, 1);
  if (!(xofftd0 && xofftd1 && xoffbu0 && xoffbu1)){
    perror("Failure in alloc_graph\n");
    return -1;
  }
  return 0;
}

static void
free_graph (void) //free related data structure
{
  if(!xadjbu0)
    nfree_large (xadjbu0);
  if(!xadjtd0)
    nfree_large (xadjtd0);
  if(!xadjbu1)
    nfree_large (xadjbu1);
  if(!xadjtd1)
    nfree_large (xadjtd1);
  if(!xoffbu0)
    nfree_large (xoffbu0);
  if(!xofftd0)
    nfree_large (xofftd0);
  if(!xoffbu1)
    nfree_large (xoffbu1);
  if(!xofftd1)
    nfree_large (xofftd1);
  if(!old2new)
    free(old2new);
  if(!new2old)
    free(new2old);
}

static int64_t
prefix_sum_td (int64_t *xofftd, int64_t *buf) // do prefix sum to setup degree
{
  int nt, tid;
  int64_t slice_begin, slice_end, t1, t2, k;

  nt = omp_get_num_threads ();
  tid = omp_get_thread_num ();

  t1 = nv / nt;
  t2 = nv % nt;
  slice_begin = t1 * tid + (tid < t2? tid : t2);
  slice_end = t1 * (tid+1) + ((tid+1) < t2? (tid+1) : t2);

  buf[tid] = 0;
  for (k = slice_begin; k < slice_end; ++k)
    buf[tid] += XOFFTD(k);
  OMP("omp barrier");
  OMP("omp single")
    for (k = 1; k < nt; ++k)
      buf[k] += buf[k-1];
  if (tid)
    t1 = buf[tid-1];
  else
    t1 = 0;
  for (k = slice_begin; k < slice_end; ++k) {
    int64_t tmp = XOFFTD(k);
    XOFFTD(k) = t1;
    t1 += tmp;
  }
  OMP("omp flush (xofftd)");
  OMP("omp barrier");
  return buf[nt-1];
}

static int64_t
prefix_sum_bu (int64_t *xoffbu, int64_t part, int64_t *buf) // same as above
{
  int nt, tid;
  int64_t slice_begin, slice_end, t1, t2, k;

  nt = omp_get_num_threads ();
  tid = omp_get_thread_num ();

  t1 = mpart / nt;
  t2 = mpart % nt;
  slice_begin = ipart + part + t1 * tid + (tid < t2? tid : t2);
  slice_end = ipart + part + t1 * (tid+1) + ((tid+1) < t2? (tid+1) : t2);

  buf[tid] = 0;
  for (k = slice_begin; k < slice_end; ++k)
    buf[tid] += XOFFBU(k);
  OMP("omp barrier");
  OMP("omp single")
    for (k = 1; k < nt; ++k)
      buf[k] += buf[k-1];
  if (tid)
    t1 = buf[tid-1];
  else
    t1 = 0;
  for (k = slice_begin; k < slice_end; ++k) {
    int64_t tmp = XOFFBU(k);
    XOFFBU(k) = t1;
    t1 += tmp;
  }
  OMP("omp flush (xoffbu)");
  OMP("omp barrier");
  return buf[nt-1];
}

static int
setup_td_deg_off (const struct packed_edge * restrict IJ, int64_t nedge) // calculate its partial degree and alloc adjacent array
{
  int64_t *buf = NULL;
  xadjtd0= NULL;
  xadjtd1 = NULL;
  OMP("omp parallel") {
    int64_t k, accum0, accum1;
    OMP("omp for")
      for (k = 0; k < 2*nv+2; ++k){
        xofftd0[k] = 0;
        xofftd1[k] = 0;
      }
    OMP("omp for")
      for (k = 0; k < nedge; ++k) {
        int64_t i = old2new[get_v0_from_edge(&IJ[k])];
        int64_t j = old2new[get_v1_from_edge(&IJ[k])];
        if (i != j) { /* Skip self-edges. */
          if (i >= ipart)
            if(i < ipart + mpart)
              OMP("omp atomic")
                ++XOFFTD0(j);
            else if(i < ipart + npart)
              OMP("omp atomic")
                ++XOFFTD1(j);
          if(j >= ipart)
            if(j < ipart + mpart)
              OMP("omp atomic")
                ++XOFFTD0(i);
            else if(j < ipart + npart)
              OMP("omp atomic")
                ++XOFFTD1(i);
        }
      }
    OMP("omp single") {
      buf = alloca (omp_get_num_threads () * sizeof (*buf));
      if (!buf) {
        perror ("alloca for prefix-sum hosed");
        abort ();
      }
    }
    OMP("omp barrier");
    accum0 = prefix_sum_td (xofftd0, buf);
    OMP("omp barrier");
    accum1 = prefix_sum_td (xofftd1, buf);
    OMP("omp barrier");

    OMP("omp for")
      for (k = 0; k < nv; ++k){
        XENDOFFTD0(k) = XOFFTD0(k);
        XENDOFFTD1(k) = XOFFTD1(k);
      }
    OMP("omp single") {
      XOFFTD0(nv) = accum0;
      XOFFTD1(nv) = accum1;
      if (!((xadjtd0 = nmalloc_large (XOFFTD0(nv) * sizeof (*xadjtd0), 0)) && (xadjtd1 = nmalloc_large (XOFFTD1(nv) * sizeof (*xadjtd1), 1)))){
        perror("Fail to malloc xadjtd\n");
      }
    }
  }
  return !(xadjtd0 && xadjtd1);
}

static int
setup_bu_deg_off (const struct packed_edge * restrict IJ, int64_t nedge) //same as above
{
  int64_t *buf = NULL;
  xadjbu0 = NULL;
  xadjbu1 = NULL;
  OMP("omp parallel") {
    int64_t k, accum0, accum1;
    OMP("omp for")
      for (k = 0; k < 2*mpart+2; ++k){
        xoffbu0[k] = 0;
        xoffbu1[k] = 0;
      }
    OMP("omp for")
      for (k = 0; k < nedge; ++k) {
        int64_t i = old2new[get_v0_from_edge(&IJ[k])];
        int64_t j = old2new[get_v1_from_edge(&IJ[k])];
        if (i != j) { /* Skip self-edges. */
          if (i >= ipart)
            if(i < ipart + mpart)
              OMP("omp atomic")
                ++XOFFBU0(i);
            else if(i < ipart + npart)
              OMP("omp atomic")
                ++XOFFBU1(i);
          if(j >= ipart)
            if(j < ipart + mpart)
              OMP("omp atomic")
                ++XOFFBU0(j);
            else if(j < ipart + npart)
              OMP("omp atomic")
                ++XOFFBU1(j);
        }
      }
    OMP("omp single") {
      buf = alloca (omp_get_num_threads () * sizeof (*buf));
      if (!buf) {
        perror ("alloca for prefix-sum hosed");
        abort ();
      }
    }

    OMP("omp barrier");
    accum0 = prefix_sum_bu (xoffbu0, 0, buf);
    OMP("omp barrier");
    accum1 = prefix_sum_bu (xoffbu1, mpart, buf);
    OMP("omp barrier");

    OMP("omp for")
      for (k = ipart; k < ipart + npart; ++k)
        if(k < ipart + mpart)
          XENDOFFBU0(k) = XOFFBU0(k);
        else
          XENDOFFBU1(k) = XOFFBU1(k);
    OMP("omp single") {
      XOFFBU0(ipart+mpart) = accum0;
      XOFFBU1(ipart+npart) = accum1;
      if (!((xadjbu0 = nmalloc_large (accum0 * sizeof (*xadjbu0), 0)) && (xadjbu1 = nmalloc_large (accum1 * sizeof (*xadjbu1), 1)))){
        perror("malloc xadjbu failed\n");
      }
    }
  }
  return !(xadjbu0 && xadjbu1);
}

static int
setup_deg_off (const struct packed_edge * restrict IJ, int64_t nedge){
  int td=setup_td_deg_off(IJ, nedge);
  int bu=setup_bu_deg_off(IJ, nedge);
  return (td && bu);
}

static void
scatter_edge_td (const int64_t i, const int64_t j) //put edges into arrays
{
  int64_t where;
  if(j >= ipart){
    if(j < ipart + mpart){
      where = int64_fetch_add (&XENDOFFTD0(i), 1);
      xadjtd0[where] = j;
    }else if(j < ipart + npart){
      where = int64_fetch_add (&XENDOFFTD1(i), 1);
      xadjtd1[where] = j;
    }
  }
}

static void
scatter_edge_bu (const int64_t i, const int64_t j)
{
  int64_t where;
  if(i >= ipart){
    if(i < ipart + mpart){
      where = int64_fetch_add (&XENDOFFBU0(i), 1);
      xadjbu0[where] = j;
    } else if(i < ipart + npart){
      where = int64_fetch_add (&XENDOFFBU1(i), 1);
      xadjbu1[where] = j;
    }
  }
}

static void scatter_edge (const int64_t i, const int64_t j){ //scatter edgelist to make graph
  scatter_edge_td(i,j);
  scatter_edge_bu(i,j);
}

static int
i64cmp (const void *a, const void *b) //used to sort vertex by index
{
  const int64_t ia = *(const int64_t*)a;
  const int64_t ib = *(const int64_t*)b;
  if (ia < ib) return -1;
  if (ia > ib) return 1;
  return 0;
}

static void
pack_vtx_edges_td (int64_t* xofftd, int64_t* xadjtd, const int64_t i) //sort vertices by index and remove verbose edges
{
  int64_t kcur, k;
  if(XOFFTD(i)+1 >= XENDOFFTD(i)) return;
  qsort (&xadjtd[XOFFTD(i)], XENDOFFTD(i)-XOFFTD(i), sizeof(*xadjtd), i64cmp);
  kcur = XOFFTD(i);
  for (k = XOFFTD(i)+1; k < XENDOFFTD(i); ++k)
    if (xadjtd[k] != xadjtd[kcur])
      xadjtd[++kcur] = xadjtd[k];
  ++kcur;
  for (k = kcur; k < XENDOFFTD(i); ++k)
    xadjtd[k] = -1;
  XENDOFFTD(i) = kcur;
}

static void
pack_vtx_edges_bu (int64_t* xoffbu, int64_t* xadjbu, int64_t part, const int64_t i)
{
  int64_t kcur, k;
  if ((i<ipart+part)||(i>ipart+part+mpart)) return;
  if (XOFFBU(i)+1 >= XENDOFFBU(i)) return;
  qsort (&xadjbu[XOFFBU(i)], XENDOFFBU(i)-XOFFBU(i), sizeof(*xadjbu), i64cmp);
  kcur = XOFFBU(i);
  for (k = XOFFBU(i)+1; k < XENDOFFBU(i); ++k)
    if (xadjbu[k] != xadjbu[kcur])
      xadjbu[++kcur] = xadjbu[k];
  ++kcur;
  for (k = kcur; k < XENDOFFBU(i); ++k)
    xadjbu[k] = -1;
  XENDOFFBU(i) = kcur;
}

static void
pack_vtx_edges(const int64_t i){ //pack edges in adjacent array
  pack_vtx_edges_td(xofftd0, xadjtd0, i);
  pack_vtx_edges_td(xofftd1, xadjtd1, i);
  pack_vtx_edges_bu(xoffbu0, xadjbu0, 0, i);
  pack_vtx_edges_bu(xoffbu1, xadjbu1, mpart, i);
}

static void
pack_edges (void)
{
  int64_t v;

  OMP("omp for")
    for (v = 0; v < nv; ++v)
      pack_vtx_edges (v);
}

static void
gather_edges (const struct packed_edge * restrict IJ, int64_t nedge) // fill graph data structure
{
  OMP("omp parallel") {
    int64_t k;

    OMP("omp for")
      for (k = 0; k < nedge; ++k) {
        int64_t i = old2new[get_v0_from_edge(&IJ[k])];
        int64_t j = old2new[get_v1_from_edge(&IJ[k])];
        if (i >= 0 && j >= 0 && i != j) {
          scatter_edge (i, j);
          scatter_edge (j, i);
        }
      }

    pack_edges ();
  }
}

int
comp(const void *a, const void *b){ // compare degree of vertices
  if (((Record *)b)->degree > ((Record *)a)->degree)
    return 1;
  else if (((Record *)b)->degree < ((Record *)a)->degree)
    return -1;
  else return (((Record *)b)->index - ((Record *)a)->index);
}

void
transform(struct packed_edge *IJ, int64_t nedge){ //subtle optimization
  int64_t k, l;
  record = (Record *) malloc (pnv * sizeof(Record));
  OMP("omp parallel for")
    for(k = 0; k < pnv; ++k){
      record[k].degree = 0;
    }

  OMP("omp parallel for")
    for (k = 0; k < nedge; ++k) {
      int64_t i = get_v0_from_edge(&IJ[k]);
      int64_t j = get_v1_from_edge(&IJ[k]);
      if (i != j) { /* Skip self-edges. */
        OMP("omp atomic")
        ++ record[i].degree;
        OMP("omp atomic")
        ++ record[j].degree;
      }
    }
  OMP("omp parallel for")
    for (k = 0; k < pnv; ++k) {
      record[k].index = k;
    }
  qsort(record, pnv, sizeof(Record), comp);
  int64_t nnv;
  OMP("omp parallel for shared(nnv)")
    for(k = 0; k < pnv - 1; ++k){
      if(record[k].degree > 0 && record[k+1].degree == 0){
        nnv = k;
      }
    }

  nv = nnv + 1 + 128 * nodes - ((nnv + 1) % (128 * nodes));
  npart = nv / nodes;
  mpart = npart / 2;
  ipart = rank * npart;

  Record *temp = (Record *) malloc (pnv * sizeof(Record));
  OMP("omp parallel for")
    for(k = 0; k < nv; ++k){
      int64_t i = k / nodes;
      int64_t j = k % nodes;
      temp[j * npart + i] = record[k];
    }

  /*
  Record t;
  for(k = 1, l = npart; k < nodes; ++k, l += npart){
      t = temp[k];
      temp[k] = record[l];
      record[l] = t;
  }
  */

  old2new = (int64_t *) malloc (pnv * sizeof(int64_t));
  memset(old2new, -1, pnv * sizeof(int64_t));
  new2old = (int64_t *) malloc (nv * sizeof(int64_t));

  OMP("omp parallel for")
    for(k = 0; k < nv; ++k){
      int64_t t = temp[k].index;
      new2old[k] = t;
      old2new[t] = k;
    }
  free(record);
  free(temp);
}

int
create_graph_from_edgelist (struct packed_edge *IJ, int64_t nedge)
{
    find_nv (IJ, nedge);
    transform(IJ,nedge);

    if (alloc_graph (nedge)) return -1;

    if (setup_deg_off (IJ, nedge)) {
        perror("setup_deg failed\n");
        free_graph();
        return -1;
    }  gather_edges (IJ, nedge);

  return 0;
}

  static void
fill_bitmap_from_queue(bitmap_T *bm, int64_t *vlist, int64_t out, int64_t in)
{
  OMP("omp for")
    for (long q_index=out; q_index<in; q_index++)
      bm_set_bit_atomic(bm, vlist[q_index]);
}

  static void
fill_queue_from_bitmap(bitmap_T *bm, int64_t *vlist, int64_t *out, int64_t *in,
    int64_t *local)
{
  OMP("omp single") {
    *out = 0;
    *in = 0;
  }
  OMP("omp barrier");
  int64_t nodes_per_thread = (nv + omp_get_num_threads() - 1) /
    omp_get_num_threads();
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
      if (local_index == THREAD_BUF_LEN) {
        int my_in = int64_fetch_add(in, THREAD_BUF_LEN);
        for (local_index=0; local_index<THREAD_BUF_LEN; local_index++) {
          vlist[my_in + local_index] = local[local_index];
        }
        local_index = 0;
      }
    }
  }
  int my_in = int64_fetch_add(in, local_index);
  for (int i=0; i<local_index; i++) {
    vlist[my_in + i] = local[i];
  }
}

  static int64_t
bfs_bottom_up_step(int64_t *bfs_tree, bitmap_T *past, bitmap_T *next, int64_t* xoffbu, int64_t *xadjbu, int64_t part, int tid)
{
  bm_reset(next);
  int64_t count = 0;
  static int64_t awake_sum;
  OMP("omp single")
    awake_sum = 0;
  OMP("omp barrier");
  const int64_t t1 = mpart / nt_cpu;
  const int64_t t2 = mpart % nt_cpu;
  const int64_t slice_begin = part + ipart + t1 * tid + (tid < t2 ? tid : t2);
  const int64_t slice_end = part + ipart + t1 * (tid + 1) + ((tid + 1) < t2 ? (tid + 1) : t2);
  for (int64_t i = slice_begin; i < slice_end; ++i) {
    if (bfs_tree[i-ipart] == -1) {
      for (int64_t vo = XOFFBU(i); vo < XENDOFFBU(i); vo++) {
        const int64_t j = xadjbu[vo];
        if (bm_get_bit(past, j)) {
          // printf("%lu\n",i);
          bfs_tree[i-ipart] = j;
          bm_set_bit_atomic(next, i);
          count++;
          break;
        }
      }
    }
  }
  OMP("omp atomic")
    awake_sum += count;
  OMP("omp barrier");
  return awake_sum;
}

  static void
bfs_top_down_step(int64_t *bfs_tree, int64_t *vlist, int64_t *vtemp, int64_t *local, int64_t oldk2, int64_t *k2_p, int64_t* xofftd, int64_t *xadjtd, int tid)
{
  int64_t kbuf = 0;
  OMP("omp single")
    *k2_p = 0;
  OMP("omp barrier");
  const int64_t t1 = oldk2 / nt_cpu;
  const int64_t t2 = oldk2 % nt_cpu;
  const int64_t slice_begin = t1 * tid + (tid < t2 ? tid : t2);
  const int64_t slice_end = t1 * (tid + 1) + ((tid + 1) < t2 ? (tid + 1) : t2);
  for (int64_t k = slice_begin; k < slice_end; ++k) {
    const int64_t v = vlist[k];
    const int64_t veo = XENDOFFTD(v);
    int64_t vo;
    for (vo = XOFFTD(v); vo < veo; ++vo) {
      const int64_t j = xadjtd[vo];
      if (bfs_tree[j-ipart] == -1) {
        if (int64_cas (&bfs_tree[j-ipart], -1, v)) {
          if (kbuf < THREAD_BUF_LEN) {
            local[kbuf++] = j;
          } else {
            fprintf(stderr, "not enough buff length\n");
            int64_t voff = int64_fetch_add (k2_p, THREAD_BUF_LEN), vk;
            assert (voff + THREAD_BUF_LEN <= nv);
            for (vk = 0; vk < THREAD_BUF_LEN; ++vk)
              vlist[voff + vk] = local[vk];
            local[0] = j;
            kbuf = 1;
          }
        }
      }
    }
  }
    //fprintf(stderr,"%ld\n", kbuf);
  if (kbuf) {
    int64_t voff = int64_fetch_add (k2_p, kbuf), vk;
    //fprintf(stderr,"before:%ld, after:%ld\n", voff, *k2_p);
    assert (voff + kbuf <= nv);
    for (vk = 0; vk < kbuf; ++vk)
      vtemp[voff + vk] = local[vk];
  }
  return;
}

  int
make_bfs_tree (int64_t *bfs_tree_out, int64_t *max_vtx_out,
    int64_t srcvtx)
{
  srcvtx = old2new[srcvtx];
  int64_t * restrict bfs_tree = xmalloc_large (npart * sizeof(*bfs_tree));
  int64_t * restrict bfs_temp = xmalloc_large(nv*sizeof(*bfs_temp));
  int64_t err = 0;

  int64_t * restrict vlist = NULL;
  int64_t * restrict vtemp = NULL;
  int64_t k2;

  int * restrict displs = malloc(nodes * sizeof (*displs));
  int * restrict recvcounts = malloc(nodes * sizeof (*recvcounts));

  *max_vtx_out = pnv - 1;

  vlist = xmalloc_large (nv * sizeof (*vlist));
  if (!vlist) return -1;

  vtemp = xmalloc_large (nv * sizeof (*vtemp));
  if (!vtemp) return -1;

  vlist[0] = srcvtx;
  k2 = 1;
  awake_count = 1;
  if((srcvtx>=ipart)&&(srcvtx<npart+ipart))
    bfs_tree[srcvtx-ipart] = srcvtx;

  bitmap_T past, next;
  bm_init(&past, nv);
  bm_init(&next, nv);

  int64_t down_cutoff = nv / BETA;
  scout_count = XENDOFFTD0(srcvtx) - XOFFTD0(srcvtx) + XENDOFFTD1(srcvtx) - XOFFTD1(srcvtx);
  edges_to_check = XOFFTD0(nv) + XOFFTD1(nv);
  MPI_Allreduce(MPI_IN_PLACE, &scout_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &edges_to_check, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

  OMP("omp parallel shared(k2)") {
    const int tid = omp_get_thread_num() % (omp_get_num_threads() / 2);
    const int node = omp_get_thread_num() < (omp_get_num_threads() / 2);

    int64_t k;
    int64_t *nbuf = (int64_t *)malloc(THREAD_BUF_LEN * sizeof(int64_t));//[THREAD_BUF_LEN];

    if((srcvtx >= ipart)&&(srcvtx < ipart + npart)){
      OMP("omp for")
        for (k = 0; k < srcvtx-ipart; ++k)
          bfs_tree[k] = -1;
      OMP("omp for")
        for (k = srcvtx+1-ipart; k < npart; ++k)
          bfs_tree[k] = -1;
    }else{
      OMP("omp for")
        for (k = 0; k < npart; k++)
          bfs_tree[k] = -1;
    }
    while (1) {
      // Top-down
      if (scout_count < ((edges_to_check - scout_count)/ALPHA)) {
        if(node){
          bfs_top_down_step(bfs_tree, vlist, vtemp, nbuf, awake_count, &k2, xofftd0, xadjtd0, tid);
        }
        else{
          bfs_top_down_step(bfs_tree, vlist, vtemp, nbuf, awake_count, &k2, xofftd1, xadjtd1, tid);
        }
        OMP("omp barrier");
        OMP("omp single"){
          //fprintf(stderr,"k2:%ld, process:%d\n",k2,rank);
          int k2temp = (int) k2;
          MPI_Allgather(&k2temp, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
          displs[0]=0;
          for(int i=1;i<nodes;i++)
            displs[i]=displs[i-1]+recvcounts[i-1];
        }
        OMP("omp barrier");
        OMP("omp single"){
          MPI_Allgatherv(vtemp, k2, MPI_INT64_T, vlist, recvcounts, displs, MPI_INT64_T, MPI_COMM_WORLD);
          awake_count = 0;
        }
        OMP("omp barrier");
        OMP("omp for reduction(+ : awake_count)")
          for(int i=0;i<nodes;i++){
            awake_count += recvcounts[i];
          }

/*
        OMP("omp single"){
          if(rank==0){
          for(int i=0;i<awake_count;i++){
            printf("%ld\t",new2old[vlist[i]]);
          }
          printf("\n.%ld.\n",awake_count);
        }
        }
          */
        //fprintf(stderr,"awake_count:%ld, process:%d\n",awake_count,rank);
        if(awake_count == 0)
          break;
        edges_to_check -= scout_count;
        // Bottom-up
      } else {
        fill_bitmap_from_queue(&past, vlist, 0, awake_count);
        do {
          if(node)
            awake_count = bfs_bottom_up_step(bfs_tree, &past, &next, xoffbu0, xadjbu0, 0, tid);
          else
            awake_count = bfs_bottom_up_step(bfs_tree, &past, &next, xoffbu1, xadjbu1, mpart, tid);
          OMP("omp barrier");
          OMP("omp single"){
            MPI_Allreduce(MPI_IN_PLACE, &awake_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allgather(&(next.start[ipart / 64]), next.size / nodes, MPI_UINT64_T, past.start, past.size / nodes, MPI_UINT64_T, MPI_COMM_WORLD);
            /*
            for(int i=0;i<nv;i++){
              if(bm_get_bit(&next,i)==1)
                printf("%ld\t",new2old[i]);
            }
            printf("\n");
            */
          }
          OMP("omp barrier");
        } while ((awake_count > down_cutoff));
        //if(awake_count == 0)
        //  break;
        fill_queue_from_bitmap(&past, vlist, &err, &k2, nbuf);
        OMP("omp barrier");
      }
      // Count the number of edges in the frontier
      OMP("omp single")
        scout_count = 0;
      OMP("omp for reduction(+ : scout_count)")
        for (int64_t i=0; i<awake_count; i++) {
          int64_t v = vlist[i];
          scout_count += XENDOFFTD0(v) - XOFFTD0(v) + XENDOFFTD1(v) - XOFFTD1(v);
        }
      OMP("omp single")
        MPI_Allreduce(MPI_IN_PLACE, &scout_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
      OMP("omp barrier");
    }
    free(nbuf);
    OMP("omp single")
        MPI_Allgather(bfs_tree, npart, MPI_INT64_T, bfs_temp, npart, MPI_INT64_T, MPI_COMM_WORLD);
    OMP("omp barrier");
    OMP("omp for")
    for(int64_t i=0;i<nv;i++){
      int t = bfs_temp[i];
      bfs_tree_out[new2old[i]]=t >= 0 ? new2old[t]:-1;
    }
  }

  free(bfs_temp);
  bm_free(&past);
  bm_free(&next);
  xfree_large (vlist);
  xfree_large (vtemp);
  xfree_large (bfs_tree);
  free(recvcounts);
  free(displs);

  return (int)err;
}

  void
destroy_graph (void)
{
  free_graph ();
}

#if defined(_OPENMP)
#if defined(__GNUC__)||defined(__INTEL_COMPILER)
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
