/* ---------------------------------------------------------------------- *
 *
 * Copyright (C) 2014 Yuichiro Yasui < yuichiro.yasui@gmail.com >
 *
 * This file is part of ULIBC.
 *
 * ULIBC is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ULIBC is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with ULIBC.  If not, see <http://www.gnu.org/licenses/>.
 * ---------------------------------------------------------------------- */
#ifndef ULIBC_COMMON_H
#define ULIBC_COMMON_H

#define U32_SHIFT 32
#define U64_SHIFT 64

/* MAX value */
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
#ifndef LINE_MAX
#define LINE_MAX 4096
#endif
#ifndef MAX_NODES
#define MAX_NODES 256
#endif
#ifndef MAX_CPUS
#define MAX_CPUS 4096
#endif
#ifndef SQRT_MAX_NODES
#define SQRT_MAX_NODES 16
#endif

#ifndef ROUNDUP
#  define ROUNDUP(x,a) (((x)+(a)-1) & ~((a)-1))
#endif
#ifndef ALIGN_UP
#  define ALIGN_UP(x,a)   (((x) + (a)) & ~((a) - 1))
#endif
#ifndef ALIGN_DOWN
#  define ALIGN_DOWN(x,a) ((x) & ~((a) - 1))
#endif
#ifndef MIN
#  define MIN(x1,x2) ( (x1) < (x2) ? (x1) : (x2) )
#endif
#ifndef MAX
#  define MAX(x1,x2) ( (x1) > (x2) ? (x1) : (x2) )
#endif

#ifndef HANDLE_ERROR
#  define HANDLE_ERROR(...)					\
  do {								\
    char err[256];						\
    sprintf(err, __VA_ARGS__);					\
    perror(err);						\
    fprintf(stderr, "[ULIBC] ERROR %s:%d:%s (errno:%d)\n",	\
	    __FILE__, __LINE__, __func__, errno);		\
    exit(EXIT_FAILURE);						\
  } while (0)
#endif

#ifndef PROFILED
#  define PROFILED(t, X) do { \
    double tt=get_msecs(); \
    X; \
    t=get_msecs()-tt; \
    if (ULIBC_verbose()) printf("ULIBC: %s (%f ms)\n", #X, t); \
  } while (0)
#endif

#ifndef TIMED
#  define TIMED(X) do { \
    double tt=get_msecs(); \
    X; \
    tt=get_msecs()-tt; \
    if (ULIBC_verbose()) printf("ULIBC: %s (%f ms)\n", #X, tt); \
  } while (0)
#endif

#ifndef TOPLEVEL_PROFILED
#  define TOPLEVEL_PROFILED(X) do { \
    double tt=get_msecs(); \
    X; \
    tt=get_msecs()-tt; \
    if (ULIBC_verbose()) printf("ULIBC: @ %s (%f ms)\n", #X, tt); \
  } while (0)
#endif

#endif /* ULIBC_COMMON_H */
