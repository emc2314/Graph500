# -*- Makefile -*-
# Copyright 2010,  Georgia Institute of Technology, USA.
# See COPYING for license.
include make.inc

GRAPH500_SOURCES=graph500.c options.c rmat.c kronecker.c verify.c prng.c \
	xalloc.c timer.c 

MAKE_EDGELIST_SOURCES=make-edgelist.c options.c rmat.c kronecker.c prng.c \
	xalloc.c timer.c 

BIN=make-edgelist

ifeq ($(BUILD_OPENMP), Yes)
BIN += omp-csr/omp-csr omp-csr/omp-csr-numa
endif

GENERATOR_SRCS=splittable_mrg.c	\
	graph_generator.c \
	make_graph.c utils.c

.PHONY: all
all: $(BIN)

CPPFLAGS += -I./generator

make-edgelist: CFLAGS:=$(CFLAGS) $(CFLAGS_OPENMP)
make-edgelist:	make-edgelist.c options.c rmat.c kronecker.c prng.c \
	xalloc.c timer.c $(addprefix generator/,$(GENERATOR_SRCS))

omp-csr/omp-csr-numa: CFLAGS:=$(CFLAGS) $(CFLAGS_OPENMP)
omp-csr/omp-csr-numa: omp-csr/omp-csr-numa.c $(GRAPH500_SOURCES) \
	$(addprefix generator/,$(GENERATOR_SRCS))

omp-csr/omp-csr: CFLAGS:=$(CFLAGS) $(CFLAGS_OPENMP)
omp-csr/omp-csr: omp-csr/omp-csr.c $(GRAPH500_SOURCES) \
	$(addprefix generator/,$(GENERATOR_SRCS))

generator/generator_test_seq: generator/generator_test_seq.c $(GENERATOR_SRCS)

generator/generator_test_omp: generator/generator_test_omp.c $(GENERATOR_SRCS)

.PHONY:	clean
clean:
	rm -f generator/generator_test_seq generator/generator_test_omp \
		$(BIN)
