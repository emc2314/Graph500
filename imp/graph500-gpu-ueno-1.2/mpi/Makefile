SW = GNU3

# default values
MPICC = mpicc
MPICPP = mpicxx

NVFLAG_BASE = -Drestrict=__restrict__ -Xcompiler -fopenmp,-ffast-math,-Wall -arch=compute_35 -code=sm_35
ifeq ($(SW), GNU3)
CFLAGS = -fopenmp -g -Wall -Drestrict=__restrict__ -O3 -DNDEBUG -ffast-math # -g -pg
NVCCFLAGS = $(NVFLAG_BASE) -g -O3 -DNDEBUG -keep
LDFLAGS = -fopenmp -g # -g -pg
endif
ifeq ($(SW), GNU0)
CFLAGS = -fopenmp -g3 -Wall -Drestrict=__restrict__ -O0 -ffast-math
#CFLAGS = -g3 -Wall -Drestrict=__restrict__ -O0 -ffast-math
#NVCCFLAGS = $(NVFLAG_BASE) -g -G -O0
NVCCFLAGS = $(NVFLAG_BASE) -g -O0
LDFLAGS = -fopenmp -g # -g -pg
endif
ifeq ($(SW), INTEL3)
CFLAGS = -ipo -no-prec-div -openmp -g3 -Drestrict=__restrict__ -O3 -DNDEBUG
#CFLAGS = -ipo -openmp -g3 -Drestrict=__restrict__ -O2 -DNDEBUG
NVCCFLAGS = $(NVFLAG_BASE) -g -O3 -DNDEBUG -keep
LDFLAGS = -openmp -g 
endif
ifeq ($(SW), INTEL0)
CFLAGS = -ipo -no-prec-div -openmp -g3 -Drestrict=__restrict__ -O0
NVCCFLAGS = $(NVFLAG_BASE) -g -G -O0
LDFLAGS = -openmp -g 
endif
ifeq ($(SW), FUJI3)
MPICC = mpifccpx
MPICPP = mpiFCCpx
CFLAGS = -Kopenmp -Koptmsg=2 -Xg -Kfast -g -Wall -Drestrict=__restrict__ -O3 -DNDEBUG -DNNUMA -ffast-math # -g -pg
LDFLAGS = -fopenmp -g # -g -pg
endif
ifeq ($(SW), FUJI0)
MPICC = mpifccpx
MPICPP = mpiFCCpx
CFLAGS = -Kopenmp -Koptmsg=2 -Xg -g -Wall -Drestrict=__restrict__ -O0 -ffast-math
#CFLAGS = -g3 -Wall -Drestrict=__restrict__ -O0 -ffast-math
LDFLAGS = -fopenmp -g # -g -pg
endif

CPPFLAGS = $(CFLAGS)

LIBS = -lm -lnuma
CUDALIB = -L$(CUDA_PATH)/lib64 -lcudart -lcuda
#NVCC = $(HOME)/apps/cuda4.1/bin/nvcc
NVCC = nvcc

INC = -I$(CUDA_PATH)/include
BINS = runnable
OBJS = main.o ../generator/splittable_mrg.o
CUOBJ = cuda_kernel.o

all: gpu

cpu: $(OBJS)
	$(MPICPP) $(LDFLAGS) -o runnable $(OBJS) $(LIBS)
	
gpu: $(OBJS) $(CUOBJ)
	$(MPICPP) $(LDFLAGS) -o runnable $(OBJS) $(CUOBJ) $(LIBS) $(CUDALIB)

.SUFFIXES: .o .c .cpp .cc .cu

.c.o:
	$(MPICC) $(INC) -c $(CFLAGS) $< -o $*.o
	
.cpp.o:
	$(MPICPP) $(INC) -c $(CPPFLAGS) $< -o $*.o
	
.cc.o:
	$(MPICPP) $(INC) -c $(CPPFLAGS) $< -o $*.o
	
.cu.o:
	$(NVCC) -c $(NVCCFLAGS) $< -o $*.o
	
.PHONY: clean
clean:
	-rm -f $(BINS) $(OBJS) $(CUOBJ)

