/* -*- mode: C; mode: folding; fill-column: 70; -*- */
/* Copyright 2010-2011,  Georgia Institute of Technology, USA. */
/* See COPYING for license. */
#include "compat.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <errno.h>
#include <limits.h>
#include <unistd.h>
#include <time.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/mman.h>

#if defined(HAVE_LIBNUMA)
#include <numa.h>
#include <numaif.h>
#else
#if defined(__linux__) && !defined(__ANDROID__)
#include <syscall.h>
#include <sys/mman.h>
extern long int syscall(long int __sysno, ...);

static inline int mbind(void *addr, unsigned long len, int mode,
      unsigned long *nodemask, unsigned long maxnode, unsigned flags) {
  return syscall(SYS_mbind, addr, len, mode, nodemask, maxnode, flags);
}
#endif
#endif

#if !defined(MAP_POPULATE)
#define MAP_POPULATE 0
#endif
#if !defined(MAP_NOSYNC)
#define MAP_NOSYNC 0
#endif
/* Used... for now. */
#if !defined(MAP_HUGETLB)
#define MAP_HUGETLB 0
#endif
#if !defined(MAP_HUGE_1GB)
#define MAP_HUGE_1GB 0
#endif

extern void *xmalloc (size_t);

#if defined(HAVE_LIBNUMA)
#define MAX_NUMA 64
static int n_numa_alloc = 0;
static struct{
  void *p;
  size_t sz;
  int node;
} numa_allocs[MAX_NUMA];
#endif


#if defined(__MTA__)||defined(USE_MMAP_LARGE)||defined(USE_MMAP_LARGE_EXT)
#define MAX_LARGE 32
static int n_large_alloc = 0;
static struct {
  void * p;
  size_t sz;
  int fd;
} large_alloc[MAX_LARGE];

static int installed_handler = 0;
static void (*old_abort_handler)(int);

static void
exit_handler (void)
{
  int k;
  for (k = 0; k < n_large_alloc; ++k) {
    if (large_alloc[k].p)
      munmap (large_alloc[k].p, large_alloc[k].sz);
    if (large_alloc[k].fd >= 0)
      close (large_alloc[k].fd);
    large_alloc[k].p = NULL;
    large_alloc[k].fd = -1;
  }
}

static void
abort_handler (int passthrough)
{
  exit_handler ();
  if (old_abort_handler) old_abort_handler (passthrough);
}
#endif

#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#define MAP_ANONYMOUS MAP_ANON
#endif

void *
xmalloc_large (size_t sz)
{
#if defined(__MTA__)||defined(USE_MMAP_LARGE)
  void *out;
  int which = n_large_alloc++;
  if (n_large_alloc > MAX_LARGE) {
    fprintf (stderr, "Too many large allocations. %d %d\n", n_large_alloc, MAX_LARGE);
    --n_large_alloc;
    abort ();
  }
  large_alloc[which].p = NULL;
  large_alloc[which].fd = -1;
  out = mmap (NULL, sz, PROT_READ|PROT_WRITE,
	      MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE|MAP_HUGETLB|MAP_HUGE_1GB, 0, 0);
  if (out == MAP_FAILED || !out) {
    perror ("mmap failed");
    abort ();
  }
  large_alloc[which].p = out;
  large_alloc[which].sz = sz;
  return out;
#else
  return xmalloc (sz);
#endif
}

void
xfree_large (void *p)
{
#if defined(__MTA__)||defined(USE_MMAP_LARGE)||defined(USE_MMAP_LARGE_EXT)
  int k, found = 0;
  for (k = 0; k < n_large_alloc; ++k) {
    if (p == large_alloc[k].p) {
      munmap (p, large_alloc[k].sz);
      large_alloc[k].p = NULL;
      if (large_alloc[k].fd >= 0) {
	close (large_alloc[k].fd);
	large_alloc[k].fd = -1;
      }
      found = 1;
      break;
    }
  }
  if (found) {
    --n_large_alloc;
    for (; k < n_large_alloc; ++k)
      large_alloc[k] = large_alloc[k+1];
  } else
    free (p);
#else
  xfree (p);
#endif
}

void *
xmalloc_large_ext (size_t sz)
{
#if !defined(__MTA__)&&defined(USE_MMAP_LARGE_EXT)
  char extname[PATH_MAX+1];
  char *tmppath;
  void *out;
  int fd, which;

  if (getenv ("TMPDIR"))
    tmppath = getenv ("TMPDIR");
  else if (getenv ("TEMPDIR"))
    tmppath = getenv ("TEMPDIR");
  else
    tmppath = "/tmp";

  sprintf (extname, "%s/graph500-ext-XXXXXX", tmppath);

  which = n_large_alloc++;
  if (n_large_alloc > MAX_LARGE) {
    fprintf (stderr, "Out of large allocations.\n");
    abort ();
  }
  large_alloc[which].p = 0;
  large_alloc[which].fd = -1;

  fd = mkstemp (extname);
  if (fd < 0) {
    perror ("xmalloc_large_ext failed to make a file");
    abort ();
  }
  if (unlink (extname)) {
    perror ("UNLINK FAILED!");
    goto errout;
  }

#if _XOPEN_SOURCE >= 500
  if (pwrite (fd, &fd, sizeof (fd), sz - sizeof(fd)) != sizeof (fd)) {
    perror ("resizing pwrite failed");
    goto errout;
  }
#else
  if (lseek (fd, sz - sizeof(fd), SEEK_SET) < 0) {
    perror ("lseek failed");
    goto errout;
  }
  if (write (fd, &fd, sizeof(fd)) != sizeof (fd)) {
    perror ("resizing write failed");
    goto errout;
  }
#endif
  fcntl (fd, F_SETFD, O_ASYNC);

  out = mmap (NULL, sz, PROT_READ|PROT_WRITE,
	      MAP_SHARED|MAP_POPULATE|MAP_NOSYNC, fd, 0);
  if (MAP_FAILED == out || !out) {
    perror ("mmap ext failed");
    goto errout;
  }

  if (!installed_handler) {
    installed_handler = 1;
    if (atexit (exit_handler)) {
      perror ("failed to install exit handler");
      goto errout;
    }

    old_abort_handler = signal (SIGABRT, abort_handler);
    if (SIG_ERR == old_abort_handler) {
      perror ("failed to install cleanup handler");
      goto errout;
    }
  }

  large_alloc[which].p = out;
  large_alloc[which].sz = sz;
  large_alloc[which].fd = fd;

  return out;

 errout:
  if (fd >= 0) close (fd);
  abort ();
#else
  return xmalloc_large (sz);
#endif
}

void *
nmalloc(size_t size, const int onnode){
  void *p = NULL;
  if(size == 0)
    return p;
#if !defined(HAVE_LIBNUMA)
  int which = n_numa_alloc++;
  if (n_numa_alloc > MAX_NUMA) {
    fprintf (stderr, "Too many NUMA allocations. %d %d\n", n_numa_alloc, MAX_NUMA);
    --n_numa_alloc;
    abort ();
  }
  numa_allocs[which].p = NULL;
  numa_allocs[which].sz = 0;
  numa_allocs[which].node = -1;
  p = numa_alloc_onnode(size, onnode);
  if (p == NULL) {
    perror ("numa alloc failed");
    abort ();
  }
  numa_allocs[which].p = p;
  numa_allocs[which].sz = size;
  numa_allocs[which].node = onnode
  return p;
#else
  p = malloc(size);
#endif
  return p;
}

void *
nmalloc_large(size_t size, const int onnode){
  void* p = NULL;
  if(size == 0)
    return p;
#if defined(__MTA__)||defined(USE_MMAP_LARGE)
#if !defined(HAVE_LIBNUMA)
  enum MBIND_FLAGS {
    MPOL_MF_STRICT   = (1<<0),  /* Verify existing pages in the mapping */
    MPOL_MF_MOVE     = (1<<1),  /* Move pages owned by this process to conform to mapping */
    MPOL_MF_MOVE_ALL = (1<<2)   /* Move every page to conform to mapping */
  };
  enum MBIND_MODE {
    MPOL_DEFAULT     = (0),
    MPOL_PREFERRED   = (1),
    MPOL_BIND        = (2),
    MPOL_INTERLEAVE  = (3)
  };
#endif
  p = xmalloc_large(size);
  const unsigned long mask = 1UL << onnode;
  if(mbind(p, size, MPOL_BIND, &mask, 2, MPOL_MF_MOVE|MPOL_MF_STRICT) == -1){
    int err = errno;
    fprintf(stderr, "failed to bind mem, errno is %d\n", err);
  }
#else
  p = nmalloc(size, onnode);
#endif
  return p;
}

void nfree(void *p){
#if !defined(HAVE_LIBNUMA)
  int k, found = 0;
  for (k = 0; k < n_numa_alloc; ++k) {
    if (p == numa_allocs[k].p) {
      numa_free(p, numa_allocs[k].sz);
      numa_allocs[k].p = NULL;
      numa_allocs[k].sz = 0
      numa_allocs[k].node = -1;
      }
      found = 1;
      break;
    }
  }
  if (found) {
    --n_numa_alloc;
    for (; k < n_numa_alloc; ++k)
      numa_allocs[k] = numa_allocs[k+1];
  } else
    free (p);
#else
  free(p);
#endif
}

void nfree_large(void *p){
#if defined(__MTA__)||defined(USE_MMAP_LARGE)
  xfree_large(p);
#else
  nfree(p);
#endif
}
