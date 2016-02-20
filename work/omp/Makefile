# -*- Makefile -*-
# Copyright 2010,  Georgia Institute of Technology, USA.
# See COPYING for license.
include make.inc

GRAPH500_SOURCES=graph500.c options.c rmat.c kronecker.c verify.c prng.c \
	xalloc.c timer.c 

MAKE_EDGELIST_SOURCES=make-edgelist.c options.c rmat.c kronecker.c prng.c \
	xalloc.c timer.c 



GENERATOR_SRCS=splittable_mrg.c	\
	graph_generator.c \
	make_graph.c utils.c

CPPFLAGS += -I./generator

all: omp-csr/omp-csr omp-csr/omp-csrtd omp-csr/omp-csrbu

omp-csr/omp-csr: CFLAGS:=$(CFLAGS) $(CFLAGS_OPENMP)
omp-csr/omp-csr: omp-csr/omp-csr.c $(GRAPH500_SOURCES) \
	$(addprefix generator/,$(GENERATOR_SRCS))

omp-csr/omp-csrtd: CFLAGS:=$(CFLAGS) $(CFLAGS_OPENMP)
omp-csr/omp-csrtd: omp-csr/omp-csrtd.c $(GRAPH500_SOURCES) \
	$(addprefix generator/,$(GENERATOR_SRCS))

omp-csr/omp-csrbu: CFLAGS:=$(CFLAGS) $(CFLAGS_OPENMP)
omp-csr/omp-csrbu: omp-csr/omp-csrbu.c $(GRAPH500_SOURCES) \
	$(addprefix generator/,$(GENERATOR_SRCS))
run:
	omp-csr/omp-csr

.PHONY:	clean
clean: 
	rm -f omp-csr/omp-csr

