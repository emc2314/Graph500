# -*- Makefile -*-
# Copyright 2010, Georgia Institute of Technology, USA.
# See COPYING for license.
CC = mpicc
CFLAGS = -g3 -std=c99 -O0
#CFLAGS = -std=c99 -O3 -march=native -ipo #-fgcse-sm -fgcse-las -fgcse-after-reload -floop-strip-mine -ftree-loop-im -fivopts -funswitch-loops
LDLIBS = -lm -lrt
//CPPFLAGS = -DUSE_MMAP_LARGE

CPPFLAGS += -DHAVE_LIBNUMA#-DUSE_MMAP_LARGE_EXT
LDLIBS += -lnuma

#LDLIBS += -lulibc
#CFLAGS += -I./ulibc/ -L./ulibc/

BUILD_OPENMP = Yes
CFLAGS_OPENMP = -fopenmp
