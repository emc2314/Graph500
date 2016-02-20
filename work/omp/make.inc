# -*- Makefile -*-
# Copyright 2010, Georgia Institute of Technology, USA.
# See COPYING for license.
CC = icc
#CFLAGS = -g -std=c99
CFLAGS = -std=c99 -O2 -fomit-frame-pointer -march=native -ipo
LDLIBS = -lrt
CPPFLAGS = -DUSE_MMAP_LARGE -DUSE_MMAP_LARGE_EXT

BUILD_OPENMP = Yes
CFLAGS_OPENMP = -fopenmp
