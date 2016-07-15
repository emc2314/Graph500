ULIBCDIR := .
TESTDIR  := test
-include make.rule
# -include ../make.rule

OBJECTS := \
init.o \
topology.o \
online_topology.o \
numa_mapping.o \
numa_threads.o \
numa_malloc.o \
numa_loops.o \
tools.o

ifeq ($(USE_PTHREAD_BARRIER), yes)
OBJECTS += numa_barrier.o
else
OBJECTS += numa_barrier_opt.o
endif


.PHONY: all lib clean update_make_headers
all: $(HWLOC_HOME) lib

lib: $(LIBNAME)

$(LIBNAME): $(OBJECTS)
	$(RM) $@
	$(AR) $@ $^
ifdef RANLIB
	$(RANLIB) $@
endif

test: lib $(TESTDIR)/$(wildcard *.c)

$(TESTDIR)/$(wildcard *.c):
	$(MAKE) -C $(TESTDIR)

update_make_headers:
	$(CC) -MM -I$(ULIBCDIR) *.c > make.headers

-include make.headers

clean:
	$(RM) *~ \#* *.o gmon.out $(LIBNAME) $(SOLIBNAME)
	$(MAKE) -C $(TESTDIR) clean

distclean:
	$(RM) $(HWLOC_HOME) $(HWLOC_DIR) $(HWLOC_TAR)

### Local variables:
### mode: makefile-bsdmake
### coding: utf-8-unix
### indent-tabs-mode: nil
### End:
