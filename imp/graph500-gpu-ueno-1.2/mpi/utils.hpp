/*
 * Copyright (C) Koji Ueno 2012-2013.
 *
 * This file is part of Graph500 Ueno implementation.
 *
 *  Graph500 Ueno implementation is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Graph500 Ueno implementation is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Graph500 Ueno implementation.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef UTILS_IMPL_HPP_
#define UTILS_IMPL_HPP_

// for affinity setting //
#include <unistd.h>
#include <sched.h>
#include <numa.h>
#include <omp.h>

#include <sys/types.h>
#include <sys/time.h>

#include <algorithm>
#include <vector>
#include <deque>

#include "mpi_workarounds.h"
#include "utils_core.h"
#include "primitives.hpp"
#if CUDA_ENABLED
#include "gpu_host.hpp"
#endif

struct MPI_GLOBALS {
	int rank;
	int size_;

	// 2D
	int rank_2d;
	int rank_2dr;
	int rank_2dc;
	int size_2d;
	int size_2dr;
	int size_2dc;
	MPI_Comm comm_2d;
	MPI_Comm comm_2dr;
	MPI_Comm comm_2dc;
	bool isPadding2D;

	// utility method
	bool isMaster() const { return rank == 0; }
	bool isRmaster() const { return rank == size_-1; }
};

MPI_GLOBALS mpi;

//-------------------------------------------------------------//
// For generic typing
//-------------------------------------------------------------//

template <> struct MpiTypeOf<char> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<char>::type = MPI_CHAR;
template <> struct MpiTypeOf<short> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<short>::type = MPI_SHORT;
template <> struct MpiTypeOf<int> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<int>::type = MPI_INT;
template <> struct MpiTypeOf<long> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<long>::type = MPI_LONG;
template <> struct MpiTypeOf<long long> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<long long>::type = MPI_LONG_LONG;
template <> struct MpiTypeOf<float> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<float>::type = MPI_FLOAT;
template <> struct MpiTypeOf<double> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<double>::type = MPI_DOUBLE;
template <> struct MpiTypeOf<unsigned char> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<unsigned char>::type = MPI_UNSIGNED_CHAR;
template <> struct MpiTypeOf<unsigned short> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<unsigned short>::type = MPI_UNSIGNED_SHORT;
template <> struct MpiTypeOf<unsigned int> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<unsigned int>::type = MPI_UNSIGNED;
template <> struct MpiTypeOf<unsigned long> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<unsigned long>::type = MPI_UNSIGNED_LONG;
template <> struct MpiTypeOf<unsigned long long> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<unsigned long long>::type = MPI_UNSIGNED_LONG_LONG;


template <typename T> struct template_meta_helper { typedef void type; };

template <typename T> MPI_Datatype get_mpi_type(T& instance) {
	return MpiTypeOf<T>::type;
}

//-------------------------------------------------------------//
// Memory Allocation
//-------------------------------------------------------------//

void* xMPI_Alloc_mem(size_t nbytes) {
  void* p;
  MPI_Alloc_mem(nbytes, MPI_INFO_NULL, &p);
  if (nbytes != 0 && !p) {
    fprintf(stderr, "MPI_Alloc_mem failed for size %zu\n", nbytes);
    throw "OutOfMemoryExpception";
  }
  return p;
}

void* cache_aligned_xcalloc(const size_t size)
{
	void* p;
	if(posix_memalign(&p, CACHE_LINE, size)){
		fprintf(stderr, "Out of memory trying to allocate %zu byte(s)\n", size);
		throw "OutOfMemoryExpception";
	}
	memset(p, 0, size);
	return p;
}
void* cache_aligned_xmalloc(const size_t size)
{
	void* p;
	if(posix_memalign(&p, CACHE_LINE, size)){
		fprintf(stderr, "Out of memory trying to allocate %zu byte(s)\n", size);
		throw "OutOfMemoryExpception";
	}
	return p;
}

void* page_aligned_xcalloc(const size_t size)
{
	void* p;
	if(posix_memalign(&p, PAGE_SIZE, size)){
		fprintf(stderr, "Out of memory trying to allocate %zu byte(s)\n", size);
		throw "OutOfMemoryExpception";
	}
	memset(p, 0, size);
	return p;
}
void* page_aligned_xmalloc(const size_t size)
{
	void* p;
	if(posix_memalign(&p, PAGE_SIZE, size)){
		fprintf(stderr, "Out of memory trying to allocate %zu byte(s)\n", size);
		throw "OutOfMemoryExpception";
	}
	return p;
}

//-------------------------------------------------------------//
// CPU Affinity Setting
//-------------------------------------------------------------//

int g_GpuIndex = -1;

#ifndef NNUMA
typedef struct cpuid_register_t {
    unsigned long eax;
    unsigned long ebx;
    unsigned long ecx;
    unsigned long edx;
} cpuid_register_t;

void cpuid(unsigned int eax, cpuid_register_t *r)
{
    __asm__ volatile (
        "cpuid"
        :"=a"(r->eax), "=b"(r->ebx), "=c"(r->ecx), "=d"(r->edx)
        :"a"(eax)
    );
    return;
}

#define CPU_TO_PROC_MAP 2

const int cpu_to_proc_map2[] = { 0,0,0,0, 0,0,0,0, 0,0,0,0, 1,1,1,1, 1,1,1,1, 1,1,1,1 };

#if CPU_TO_PROC_MAP == 1
const int cpu_to_proc_map[] = { 9,9,9,9, 9,9,9,9, 9,9,9,9, 9,9,9,9, 9,9,9,9, 9,9,9,9 };
#elif CPU_TO_PROC_MAP == 2
const int * const cpu_to_proc_map = cpu_to_proc_map2;
#elif CPU_TO_PROC_MAP == 3
const int cpu_to_proc_map[] = { 0,0,0,0, 0,0,0,0, 2,2,2,2, 2,2,2,2, 1,1,1,1, 1,1,1,1 };
#elif CPU_TO_PROC_MAP == 4
const int cpu_to_proc_map[] = { 0,0,0,0, 0,2,0,2, 0,2,0,2, 2,1,2,1, 2,1,2,1, 1,1,1,1 };
#endif

void setNumaAffinity(bool round_robin)
{
	if(numa_available() < 0) {
		fprintf(stderr, "No NUMA support available on this system.\n");
		return ;
	}
	int NUM_SOCKET = numa_max_node() + 1;
	if(round_robin) {
		int part = (mpi.size_ + NUM_SOCKET -1) / NUM_SOCKET;
		numa_run_on_node(mpi.rank / part);
		numa_set_preferred(mpi.rank / part);
	}
	else {
		numa_run_on_node(mpi.rank % NUM_SOCKET);
		numa_set_preferred(mpi.rank % NUM_SOCKET);
	}
}

void setAffinity()
{
	int NUM_PROCS = sysconf(_SC_NPROCESSORS_CONF);
	cpu_set_t set;
	int i;

	const char* num_node_str = getenv("MPI_NUM_NODE");
	if(num_node_str == NULL) {
		if(mpi.rank == mpi.size_ - 1) {
			fprintf(stderr, "Error: failed to get # of node. Please set MPI_NUM_NODE=<# of node>\n");
		}
		return ;
	}
	const char* dist_round_robin = getenv("MPI_ROUND_ROBIN");
	int num_node = atoi(num_node_str);

	int32_t core_list[NUM_PROCS];
	for(i = 0; i < NUM_PROCS; i++) {
		CPU_ZERO(&set);
		CPU_SET(i, &set);
		sched_setaffinity(0, sizeof(set), &set);
		sleep(0);
		cpuid_register_t reg;
		cpuid(1, &reg);
		int apicid = (reg.ebx >> 24) & 0xFF;
	//	printf("%d-th -> apicid=%d\n", i, apicid);
		core_list[i] = (apicid << 16) | i;
	}

	std::sort(core_list ,core_list + NUM_PROCS);
#if 0
	printf("sort\n");
	for(int i = 0; i < NUM_PROCS; i++) {
		int core_id = core_list[i] & 0xFFFF;
		int apicid = core_list[i] >> 16;
		printf("%d-th -> apicid=%d\n", core_id, apicid);
	}
#endif
	int max_procs_per_node = (mpi.size_ + num_node - 1) / num_node;
	int proc_num = (dist_round_robin ? (mpi.rank / num_node) : (mpi.rank % max_procs_per_node));
	g_GpuIndex = proc_num;
#if CPU_TO_PROC_MAP != 2
	int node_num = mpi.rank % num_node;
	int split = ((mpi.size_ - 1) % num_node) + 1;
#endif

	if(mpi.isRmaster()) {
		fprintf(stderr, "process distribution : %s\n", dist_round_robin ? "round robin" : "partition");
	}
//#if SET_AFFINITY
	if(max_procs_per_node == 3) {
		if(numa_available() < 0) {
			printf("No NUMA support available on this system.\n");
		}
		else {
			int NUM_SOCKET = numa_max_node() + 1;
#if GPU_COMM_OPT
			numa_set_preferred(std::min<int>(proc_num, NUM_SOCKET - 1));
#else
			if(proc_num < NUM_SOCKET) {
				//numa_run_on_node(proc_num);
				numa_set_preferred(proc_num);
			}
#endif
		}

		CPU_ZERO(&set);
		int enabled = 0;
#if CPU_TO_PROC_MAP != 2
		const int *cpu_map = (dist_round_robin && node_num >= split) ? cpu_to_proc_map2 : cpu_to_proc_map;
#else
		const int *cpu_map = cpu_to_proc_map;
#endif
		for(i = 0; i < NUM_PROCS; i++) {
			if(cpu_map[i] == proc_num) {
				int core_id = core_list[i] & 0xFFFF;
				CPU_SET(core_id, &set);
				enabled = 1;
			}
		}
		if(enabled == 0) {
			for(i = 0; i < NUM_PROCS; i++) {
				CPU_SET(i, &set);
			}
		}
		sched_setaffinity(0, sizeof(set), &set);

		if(mpi.isRmaster()) { /* print from max rank node for easy debugging */
		  fprintf(stderr, "affinity for executing 3 processed per node is enabled.\n");
		}
	}
	else if(max_procs_per_node > 1) {
		setNumaAffinity(dist_round_robin ? 1 : 0);
		if(mpi.rank == mpi.size_-1) { /* print from max rank node for easy debugging */
		  fprintf(stderr, "NUMA node affinity is enabled.\n");
		}
	}
	else
//#endif
	{
		//
		if(mpi.isRmaster()) { /* print from max rank node for easy debugging */
		  fprintf(stderr, "affinity is disabled.\n");
		}
		CPU_ZERO(&set);
		for(i = 0; i < NUM_PROCS; i++) {
			CPU_SET(i, &set);
		}
		sched_setaffinity(0, sizeof(set), &set);
	}
}
#endif

//-------------------------------------------------------------//
// ?
//-------------------------------------------------------------//

static void setup_2dcomm(bool row_major)
{
	const int log_size = get_msb_index(mpi.size_);

	const char* twod_r_str = getenv("TWOD_R");
	int log_size_r = log_size / 2;
	if(twod_r_str){
		int twod_r = atoi((char*)twod_r_str);
		if(twod_r == 0 || /* Check for power of 2 */ (twod_r & (twod_r - 1)) != 0) {
			fprintf(stderr, "Number of Rows %d is not a power of two.\n", twod_r);
		}
		else {
			log_size_r = get_msb_index(twod_r);
		}
	}

	int log_size_c = log_size - log_size_r;

	mpi.size_2dr = (1 << log_size_r);
	mpi.size_2dc = (1 << log_size_c);

	if(mpi.isMaster()) fprintf(stderr, "Process Dimension: (%dx%d)\n", mpi.size_2dr, mpi.size_2dc);

	if(row_major) {
		// row major
		mpi.rank_2dr = mpi.rank / mpi.size_2dc;
		mpi.rank_2dc = mpi.rank % mpi.size_2dc;
	}
	else {
		// column major
		mpi.rank_2dr = mpi.rank / mpi.size_2dr;
		mpi.rank_2dc = mpi.rank % mpi.size_2dr;
	}

	mpi.rank_2d = mpi.rank_2dr + mpi.rank_2dc * mpi.size_2dr;
	mpi.size_2d = mpi.size_2dr * mpi.size_2dc;
	MPI_Comm_split(MPI_COMM_WORLD, 0, mpi.rank_2d, &mpi.comm_2d);
	MPI_Comm_split(MPI_COMM_WORLD, mpi.rank_2dc, mpi.rank_2dr, &mpi.comm_2dc);
	MPI_Comm_split(MPI_COMM_WORLD, mpi.rank_2dr, mpi.rank_2dc, &mpi.comm_2dr);
}

// assume rank = XYZ
static void setup_2dcomm_on_3d()
{
	const int log_size = get_msb_index(mpi.size_);

	const char* threed_map_str = getenv("THREED_MAP");
	if(threed_map_str) {
		int X, Y, Z1, Z2, A, B;
		sscanf(threed_map_str, "%dx%dx%dx%d", &X, &Y, &Z1, &Z2);
		A = X * Z1;
		B = Y * Z2;
		mpi.size_2dr = 1 << get_msb_index(A);
		mpi.size_2dc = 1 << get_msb_index(B);

		if(mpi.isMaster()) fprintf(stderr, "Dimension: (%dx%dx%dx%d) -> (%dx%d) -> (%dx%d)\n", X, Y, Z1, Z2, A, B, mpi.size_2dr, mpi.size_2dc);
		if(mpi.size_ < A*B) {
			if(mpi.isMaster()) fprintf(stderr, "Error: There are not enough processes.\n");
		}

		int x, y, z1, z2;
		x = mpi.rank % X;
		y = (mpi.rank / X) % Y;
		z1 = (mpi.rank / (X*Y)) % Z1;
		z2 = mpi.rank / (X*Y*Z1);
		mpi.rank_2dr = z1 * X + x;
		mpi.rank_2dc = z2 * Y + y;

		mpi.rank_2d = mpi.rank_2dr + mpi.rank_2dc * mpi.size_2dr;
		mpi.size_2d = mpi.size_2dr * mpi.size_2dc;
		mpi.isPadding2D = (mpi.rank_2dr >= mpi.size_2dr || mpi.rank_2dc >= mpi.size_2dc) ? true : false;
		MPI_Comm_split(MPI_COMM_WORLD, mpi.isPadding2D ? 1 : 0, mpi.rank_2d, &mpi.comm_2d);
		if(mpi.isPadding2D == false) {
			MPI_Comm_split(mpi.comm_2d, mpi.rank_2dc, mpi.rank_2dr, &mpi.comm_2dc);
			MPI_Comm_split(mpi.comm_2d, mpi.rank_2dr, mpi.rank_2dc, &mpi.comm_2dr);
		}
	}
	else if((1 << log_size) != mpi.size_) {
		if(mpi.isMaster()) fprintf(stderr, "The program needs dimension information when mpi processes is not a power of two.\n");
	}

}

void cleanup_2dcomm()
{
	MPI_Comm_free(&mpi.comm_2d);
	if(mpi.isPadding2D == false) {
		MPI_Comm_free(&mpi.comm_2dr);
		MPI_Comm_free(&mpi.comm_2dc);
	}
}

void setup_globals(int argc, char** argv, int SCALE, int edgefactor)
{
	{
		int prov;
		MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &prov);
		MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
		MPI_Comm_size(MPI_COMM_WORLD, &mpi.size_);
	}

	if(mpi.isMaster()) {
		fprintf(stderr, "Graph500 Benchmark: SCALE: %d, edgefactor: %d %s\n", SCALE, edgefactor,
#ifdef NDEBUG
				""
#else
				"(Debug Mode)"
#endif
		);
		fprintf(stderr, "Running Binary: %s\n", argv[0]);
	}

	if(getenv("THREED_MAP")) {
		setup_2dcomm_on_3d();
	}
	else {
		setup_2dcomm(true);
	}

	// enables nested
	omp_set_nested(1);

	// change default error handler
	MPI_File_set_errhandler(MPI_FILE_NULL, MPI_ERRORS_ARE_FATAL);

#ifdef _OPENMP
	if(mpi.isRmaster()){
#if _OPENMP >= 200805
	  omp_sched_t kind;
	  int modifier;
	  omp_get_schedule(&kind, &modifier);
	  const char* kind_str = "unknown";
	  switch(kind) {
		case omp_sched_static:
		  kind_str = "omp_sched_static";
		  break;
		case omp_sched_dynamic:
		  kind_str = "omp_sched_dynamic";
		  break;
		case omp_sched_guided:
		  kind_str = "omp_sched_guided";
		  break;
		case omp_sched_auto:
		  kind_str = "omp_sched_auto";
		  break;
	  }
	  fprintf(stderr, "OpenMP default scheduling : %s, %d\n", kind_str, modifier);
#else
	  fprintf(stderr, "OpenMP version : %d\n", _OPENMP);
#endif
	}
#endif

	UnweightedEdge::initialize();
	UnweightedPackedEdge::initialize();
	WeightedEdge::initialize();

	// check page size
	if(mpi.isMaster()) {
		long page_size = sysconf(_SC_PAGESIZE);
		if(page_size != PAGE_SIZE) {
			fprintf(stderr, "System Page Size: %ld\n", page_size);
			fprintf(stderr, "Error: PAGE_SIZE(%d) is not correct.\n", PAGE_SIZE);
		}
	}
#ifndef NNUMA
	// set affinity
	if(getenv("NO_AFFINITY") == NULL) {
		setAffinity();
	}
#endif

#if CUDA_ENABLED
	CudaStreamManager::initialize_cuda(g_GpuIndex);

	MPI_INFO_ON_GPU mpig;
	mpig.rank = mpi.rank;
	mpig.size = mpi.size_2d;
	mpig.rank_2d = mpi.rank_2d;
	mpig.rank_2dr = mpi.rank_2dr;
	mpig.rank_2dc = mpi.rank_2dc;
	CudaStreamManager::begin_cuda();
	setup_gpu_global(&mpig);
	CudaStreamManager::end_cuda();
#endif
}

void cleanup_globals()
{
	cleanup_2dcomm();

	UnweightedEdge::uninitialize();
	UnweightedPackedEdge::uninitialize();
	WeightedEdge::uninitialize();

#if CUDA_ENABLED
	CudaStreamManager::finalize_cuda();
#endif

	MPI_Finalize();
}

//-------------------------------------------------------------//
// Multithread Partitioning and Scatter
//-------------------------------------------------------------//

// Usage: get_counts -> sum -> get_offsets
template <typename T>
class ParallelPartitioning
{
public:
	explicit ParallelPartitioning(int num_partitions)
		: num_partitions_(num_partitions)
		, max_threads_(omp_get_max_threads())
		, thread_counts_(NULL)
		, thread_offsets_(NULL)
	{
		buffer_width_ = std::max<int>(CACHE_LINE/sizeof(T), num_partitions_);
		thread_counts_ = static_cast<T*>(cache_aligned_xmalloc(buffer_width_ * (max_threads_*2 + 1) * sizeof(T)));
		thread_offsets_ = thread_counts_ + buffer_width_*max_threads_;

		partition_size_ = static_cast<T*>(cache_aligned_xmalloc((num_partitions_*2 + 1) * sizeof(T)));
		partition_offsets_ = partition_size_ + num_partitions_;
	}
	~ParallelPartitioning()
	{
		::free(thread_counts_);
		::free(partition_size_);
	}
	T sum() {
		const int width = buffer_width_;
		// compute sum of thread local count values
		for(int r = 0; r < num_partitions_; ++r) {
			int sum = 0;
			for(int t = 0; t < max_threads_; ++t) {
				sum += thread_counts_[t*width + r];
			}
			partition_size_[r] = sum;
		}
		// compute offsets
		partition_offsets_[0] = 0;
		for(int r = 0; r < num_partitions_; ++r) {
			partition_offsets_[r + 1] = partition_offsets_[r] + partition_size_[r];
		}
		// assert (send_counts[size] == bufsize*2);
		// compute offset of each threads
		for(int r = 0; r < num_partitions_; ++r) {
			thread_offsets_[0*width + r] = partition_offsets_[r];
			for(int t = 0; t < max_threads_; ++t) {
				thread_offsets_[(t+1)*width + r] = thread_offsets_[t*width + r] + thread_counts_[t*width + r];
			}
			assert (thread_offsets_[max_threads_*width + r] == partition_offsets_[r + 1]);
		}
		return partition_offsets_[num_partitions_];
	}
	T* get_counts() {
		T* counts = &thread_counts_[buffer_width_*omp_get_thread_num()];
		memset(counts, 0x00, buffer_width_*sizeof(T));
		return counts;
	}
	T* get_offsets() { return &thread_offsets_[buffer_width_*omp_get_thread_num()]; }

	void check() {
#ifndef	NDEBUG
		const int width = buffer_width_;
		// check offset of each threads
		for(int r = 0; r < num_partitions_; ++r) {
			assert (thread_offsets_[0*width + r] == partition_offsets_[r] + thread_counts_[0*width + r]);
			for(int t = 1; t < max_threads_; ++t) {
				assert (thread_offsets_[t*width + r] == thread_offsets_[(t-1)*width + r] + thread_counts_[t*width + r]);
			}
		}
#endif
	}
private:
	int num_partitions_;
	int buffer_width_;
	int max_threads_;
	T* thread_counts_;
	T* thread_offsets_;
	T* partition_size_;
	T* partition_offsets_;
};

// Usage: get_counts -> sum -> get_offsets -> scatter -> gather
class ScatterContext
{
public:
	explicit ScatterContext(MPI_Comm comm)
		: comm_(comm)
		, max_threads_(omp_get_max_threads())
		, thread_counts_(NULL)
		, thread_offsets_(NULL)
		, send_counts_(NULL)
		, send_offsets_(NULL)
		, recv_counts_(NULL)
		, recv_offsets_(NULL)
	{
		MPI_Comm_size(comm_, &comm_size_);

		buffer_width_ = std::max<int>(CACHE_LINE/sizeof(int), comm_size_);
		thread_counts_ = static_cast<int*>(cache_aligned_xmalloc(buffer_width_ * (max_threads_*2 + 1) * sizeof(int)));
		thread_offsets_ = thread_counts_ + buffer_width_*max_threads_;

		send_counts_ = static_cast<int*>(cache_aligned_xmalloc((comm_size_*2 + 1) * 2 * sizeof(int)));
		send_offsets_ = send_counts_ + comm_size_;
		recv_counts_ = send_offsets_ + comm_size_ + 1;
		recv_offsets_ = recv_counts_ + comm_size_;
	}

	~ScatterContext()
	{
		::free(thread_counts_);
		::free(send_counts_);
	}

	int* get_counts() {
		int* counts = &thread_counts_[buffer_width_*omp_get_thread_num()];
		memset(counts, 0x00, buffer_width_*sizeof(int));
		return counts;
	}
	int* get_offsets() { return &thread_offsets_[buffer_width_*omp_get_thread_num()]; }

	void sum() {
		const int width = buffer_width_;
		// compute sum of thread local count values
		for(int r = 0; r < comm_size_; ++r) {
			int sum = 0;
			for(int t = 0; t < max_threads_; ++t) {
				sum += thread_counts_[t*width + r];
			}
			send_counts_[r] = sum;
		}
		// compute offsets
		send_offsets_[0] = 0;
		for(int r = 0; r < comm_size_; ++r) {
			send_offsets_[r + 1] = send_offsets_[r] + send_counts_[r];
		}
		// assert (send_counts[size] == bufsize*2);
		// compute offset of each threads
		for(int r = 0; r < comm_size_; ++r) {
			thread_offsets_[0*width + r] = send_offsets_[r];
			for(int t = 0; t < max_threads_; ++t) {
				thread_offsets_[(t+1)*width + r] = thread_offsets_[t*width + r] + thread_counts_[t*width + r];
			}
			assert (thread_offsets_[max_threads_*width + r] == send_offsets_[r + 1]);
		}
	}

	int get_send_count() { return send_offsets_[comm_size_]; }
	int get_recv_count() { return recv_offsets_[comm_size_]; }

	template <typename T>
	T* scatter(T* send_data) {
#ifndef	NDEBUG
		const int width = buffer_width_;
		// check offset of each threads
		for(int r = 0; r < comm_size_; ++r) {
			assert (thread_offsets_[0*width + r] == send_offsets_[r] + thread_counts_[0*width + r]);
			for(int t = 1; t < max_threads_; ++t) {
				assert (thread_offsets_[t*width + r] == thread_offsets_[(t-1)*width + r] + thread_counts_[t*width + r]);
			}
		}
#endif
#if NETWORK_PROBLEM_AYALISYS
		if(mpi.isMaster()) fprintf(stderr, "MPI_Alltoall(MPI_INT, comm_size=%d)...\n", comm_size_);
#endif
		MPI_Alltoall(send_counts_, 1, MPI_INT, recv_counts_, 1, MPI_INT, comm_);
#if NETWORK_PROBLEM_AYALISYS
		if(mpi.isMaster()) fprintf(stderr, "OK\n");
#endif
		// calculate offsets
		recv_offsets_[0] = 0;
		for(int r = 0; r < comm_size_; ++r) {
			recv_offsets_[r + 1] = recv_offsets_[r] + recv_counts_[r];
		}
		T* recv_data = static_cast<T*>(xMPI_Alloc_mem(recv_offsets_[comm_size_] * sizeof(T)));
#if NETWORK_PROBLEM_AYALISYS
		if(mpi.isMaster()) fprintf(stderr, "MPI_Alltoallv(send_offsets_[%d]=%d\n", comm_size_, send_offsets_[comm_size_]);
#endif
		MPI_Alltoallv(send_data, send_counts_, send_offsets_, MpiTypeOf<T>::type,
				recv_data, recv_counts_, recv_offsets_, MpiTypeOf<T>::type, comm_);
#if NETWORK_PROBLEM_AYALISYS
		if(mpi.isMaster()) fprintf(stderr, "OK\n");
#endif
		return recv_data;
	}

	template <typename T>
	T* gather(T* send_data) {
		T* recv_data = static_cast<T*>(xMPI_Alloc_mem(send_offsets_[comm_size_] * sizeof(T)));
		MPI_Alltoallv(send_data, recv_counts_, recv_offsets_, MpiTypeOf<T>::type,
				recv_data, send_counts_, send_offsets_, MpiTypeOf<T>::type, comm_);
		return recv_data;
	}

	template <typename T>
	void free(T* buffer) {
		MPI_Free_mem(buffer);
	}

private:
	MPI_Comm comm_;
	int comm_size_;
	int buffer_width_;
	int max_threads_;
	int* thread_counts_;
	int* thread_offsets_;
	int* restrict send_counts_;
	int* restrict send_offsets_;
	int* restrict recv_counts_;
	int* restrict recv_offsets_;

};

//-------------------------------------------------------------//
// MPI helper
//-------------------------------------------------------------//

namespace MpiCollective {

template <typename Mapping>
void scatter(const Mapping mapping, int data_count, MPI_Comm comm)
{
	ScatterContext scatter(comm);
	typename Mapping::send_type* restrict partitioned_data = static_cast<typename Mapping::send_type*>(
						cache_aligned_xmalloc(data_count*sizeof(typename Mapping::send_type)));
#pragma omp parallel
	{
		int* restrict counts = scatter.get_counts();

#pragma omp for schedule(static)
		for (int i = 0; i < data_count; ++i) {
			(counts[mapping.target(i)])++;
		} // #pragma omp for schedule(static)

#pragma omp master
		{ scatter.sum(); } // #pragma omp master
#pragma omp barrier
		;
		int* restrict offsets = scatter.get_offsets();

#pragma omp for schedule(static)
		for (int i = 0; i < data_count; ++i) {
			partitioned_data[(offsets[mapping.target(i)])++] = mapping.get(i);
		} // #pragma omp for schedule(static)
	} // #pragma omp parallel

	typename Mapping::send_type* recv_data = scatter.scatter(partitioned_data);
	int recv_count = scatter.get_recv_count();
	free(partitioned_data); partitioned_data = NULL;

	int i;
#pragma omp parallel for lastprivate(i), schedule(static)
	for(i = 0; i < (recv_count&(~3)); i += 4) {
		mapping.set(i+0, recv_data[i+0]);
		mapping.set(i+1, recv_data[i+1]);
		mapping.set(i+2, recv_data[i+2]);
		mapping.set(i+3, recv_data[i+3]);
	} // #pragma omp parallel for
	for( ; i < recv_count; ++i) {
		mapping.set(i, recv_data[i]);
	}

	scatter.free(recv_data);
}

template <typename Mapping>
void gather(const Mapping mapping, int data_count, MPI_Comm comm)
{
	ScatterContext scatter(comm);

	int* restrict local_indices = static_cast<int*>(
			cache_aligned_xmalloc(data_count*sizeof(int)));
	typename Mapping::send_type* restrict partitioned_data = static_cast<typename Mapping::send_type*>(
			cache_aligned_xmalloc(data_count*sizeof(typename Mapping::send_type)));

#pragma omp parallel
	{
		int* restrict counts = scatter.get_counts();

#pragma omp for schedule(static)
		for (int i = 0; i < data_count; ++i) {
			(counts[mapping.target(i)])++;
		} // #pragma omp for schedule(static)

#pragma omp master
		{ scatter.sum(); } // #pragma omp master
#pragma omp barrier
		;
		int* restrict offsets = scatter.get_offsets();

#pragma omp for schedule(static)
		for (int i = 0; i < data_count; ++i) {
			int pos = (offsets[mapping.target(i)])++;
			assert (pos < data_count);
			local_indices[i] = pos;
			partitioned_data[pos] = mapping.get(i);
			//// user defined ////
		} // #pragma omp for schedule(static)
	} // #pragma omp parallel

	// send and receive requests
	typename Mapping::send_type* restrict reply_verts = scatter.scatter(partitioned_data);
	int recv_count = scatter.get_recv_count();
	free(partitioned_data);

	// make reply data
	typename Mapping::recv_type* restrict reply_data = static_cast<typename Mapping::recv_type*>(
			cache_aligned_xmalloc(recv_count*sizeof(typename Mapping::recv_type)));
#pragma omp parallel for
	for (int i = 0; i < recv_count; ++i) {
		reply_data[i] = mapping.map(reply_verts[i]);
	}
	scatter.free(reply_verts);

	// send and receive reply
	typename Mapping::recv_type* restrict recv_data = scatter.gather(reply_data);
	free(reply_data);

	// apply received data to edges
#pragma omp parallel for
	for (int i = 0; i < data_count; ++i) {
		mapping.set(i, recv_data[local_indices[i]]);
	}

	scatter.free(recv_data);
	free(local_indices);
}

} // namespace MpiCollective { //

//-------------------------------------------------------------//
// Other functions
//-------------------------------------------------------------//

int64_t get_time_in_microsecond()
{
	struct timeval l;
	gettimeofday(&l, NULL);
	return ((int64_t)l.tv_sec*1000000 + l.tv_usec);
}

int64_t get_time_in_nanosecond()
{
	struct timespec l;
	clock_gettime(CLOCK_MONOTONIC, &l);
	return ((int64_t)l.tv_sec*1000000000 + l.tv_nsec);
}

template <int width> size_t roundup(size_t size)
{
	return (size + width - 1) / width * width;
}

template <int width> size_t get_blocks(size_t size)
{
	return (size + width - 1) / width;
}

inline size_t roundup_2n(size_t size, size_t width)
{
	return (size + width - 1) & -width;
}

inline size_t get_blocks(size_t size, size_t width)
{
	return (size + width - 1) / width;
}

//-------------------------------------------------------------//
// VarInt Encoding
//-------------------------------------------------------------//

enum VARINT_CODING_ENUM {
	VARINT_MAX_CODE_LENGTH_32 = 5,
	VARINT_MAX_CODE_LENGTH_64 = 9,
};

#define VARINT_ENCODE_MACRO_32(p, v, l) \
if(v < 128) { \
	p[0] = (uint8_t)v; \
	l = 1; \
} \
else if(v < 128*128) { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7); \
	l = 2; \
} \
else if(v < 128*128*128) { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14); \
	l = 3; \
} \
else if(v < 128*128*128*128) { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21); \
	l = 4; \
} \
else { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21) | 0x80; \
	p[4]= (uint8_t)(v >> 28); \
	l = 5; \
}

#define VARINT_ENCODE_MACRO_64(p, v, l) \
if(v < 128) { \
	p[0] = (uint8_t)v; \
	l = 1; \
} \
else if(v < 128*128) { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7); \
	l = 2; \
} \
else if(v < 128*128*128) { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14); \
	l = 3; \
} \
else if(v < 128*128*128*128) { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21); \
	l = 4; \
} \
else if(v < 128LL*128*128*128*128){ \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21) | 0x80; \
	p[4]= (uint8_t)(v >> 28); \
	l = 5; \
} \
else if(v < 128LL*128*128*128*128*128){ \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21) | 0x80; \
	p[4]= (uint8_t)(v >> 28) | 0x80; \
	p[5]= (uint8_t)(v >> 35); \
	l = 6; \
} \
else if(v < 128LL*128*128*128*128*128*128){ \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21) | 0x80; \
	p[4]= (uint8_t)(v >> 28) | 0x80; \
	p[5]= (uint8_t)(v >> 35) | 0x80; \
	p[6]= (uint8_t)(v >> 42); \
	l = 7; \
} \
else if(v < 128LL*128*128*128*128*128*128*128){ \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21) | 0x80; \
	p[4]= (uint8_t)(v >> 28) | 0x80; \
	p[5]= (uint8_t)(v >> 35) | 0x80; \
	p[6]= (uint8_t)(v >> 42) | 0x80; \
	p[7]= (uint8_t)(v >> 49); \
	l = 8; \
} \
else { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21) | 0x80; \
	p[4]= (uint8_t)(v >> 28) | 0x80; \
	p[5]= (uint8_t)(v >> 35) | 0x80; \
	p[6]= (uint8_t)(v >> 42) | 0x80; \
	p[6]= (uint8_t)(v >> 49) | 0x80; \
	p[8]= (uint8_t)(v >> 56); \
	l = 9; \
}

#define VARINT_DECODE_MACRO_32(p, v, l) \
if(p[0] < 128) { \
	v = p[0]; \
	l = 1; \
} \
else if(p[1] < 128) { \
	v = (p[0] & 0x7F) | ((uint32_t)p[1] << 7); \
	l = 2; \
} \
else if(p[2] < 128) { \
	v = (p[0] & 0x7F) | ((uint32_t)(p[1] & 0x7F) << 7) | \
			((uint32_t)(p[2]) << 14); \
	l = 3; \
} \
else if(p[3] < 128) { \
	v = (p[0] & 0x7F) | ((uint32_t)(p[1] & 0x7F) << 7) | \
			((uint32_t)(p[2] & 0x7F) << 14) | ((uint32_t)(p[3]) << 21); \
	l = 4; \
} \
else { \
	v = (p[0] & 0x7F) | ((uint32_t)(p[1] & 0x7F) << 7) | \
			((uint32_t)(p[2] & 0x7F) << 14) | ((uint32_t)(p[3] & 0x7F) << 21) | \
			((uint32_t)(p[4]) << 28); \
	l = 5; \
}

#define VARINT_DECODE_MACRO_64(p, v, l) \
if(p[0] < 128) { \
	v = p[0]; \
	l = 1; \
} \
else if(p[1] < 128) { \
	v = (p[0] & 0x7F) | ((uint64_t)p[1] << 7); \
	l = 2; \
} \
else if(p[2] < 128) { \
	v = (p[0] & 0x7F) | ((uint64_t)(p[1] & 0x7F) << 7) | \
			((uint64_t)(p[2]) << 14); \
	l = 3; \
} \
else if(p[3] < 128) { \
	v = (p[0] & 0x7F) | ((uint64_t)(p[1] & 0x7F) << 7) | \
			((uint64_t)(p[2] & 0x7F) << 14) | ((uint64_t)(p[3]) << 21); \
	l = 4; \
} \
else if(p[4] < 128) { \
	v = (p[0] & 0x7F) | ((uint64_t)(p[1] & 0x7F) << 7) | \
			((uint64_t)(p[2] & 0x7F) << 14) | ((uint64_t)(p[3] & 0x7F) << 21) | \
			((uint64_t)(p[4]) << 28); \
	l = 5; \
} \
else if(p[5] < 128) { \
	v= (p[0] & 0x7F) | ((uint64_t)(p[1] & 0x7F) << 7) | \
			((uint64_t)(p[2] & 0x7F) << 14) | ((uint64_t)(p[3] & 0x7F) << 21) | \
			((uint64_t)(p[4] & 0x7F) << 28) | ((uint64_t)(p[5]) << 35); \
	l = 6; \
} \
else if(p[6] < 128) { \
	v = (p[0] & 0x7F) | ((uint64_t)(p[1] & 0x7F) << 7) | \
			((uint64_t)(p[2] & 0x7F) << 14) | ((uint64_t)(p[3] & 0x7F) << 21) | \
			((uint64_t)(p[4] & 0x7F) << 28) | ((uint64_t)(p[5] & 0x7F) << 35) | \
			((uint64_t)(p[6]) << 42); \
	l = 7; \
} \
else if(p[7] < 128) { \
	v = (p[0] & 0x7F) | ((uint64_t)(p[1] & 0x7F) << 7) | \
			((uint64_t)(p[2] & 0x7F) << 14) | ((uint64_t)(p[3] & 0x7F) << 21) | \
			((uint64_t)(p[4] & 0x7F) << 28) | ((uint64_t)(p[5] & 0x7F) << 35) | \
			((uint64_t)(p[6] & 0x7F) << 42) | ((uint64_t)(p[7]) << 49); \
	l = 8; \
} \
else { \
	v = (p[0] & 0x7F) | ((uint64_t)(p[1] & 0x7F) << 7) | \
			((uint64_t)(p[2] & 0x7F) << 14) | ((uint64_t)(p[3] & 0x7F) << 21) | \
			((uint64_t)(p[4] & 0x7F) << 28) | ((uint64_t)(p[5] & 0x7F) << 35) | \
			((uint64_t)(p[6] & 0x7F) << 42) | ((uint64_t)(p[7] & 0x7F) << 49) | \
			((uint64_t)(p[8]) << 56); \
	l = 9; \
}

int varint_encode_stream(const uint32_t* input, int length, uint8_t* output)
{
	uint8_t* p = output;
	for(int k = 0; k < length; ++k) {
		uint32_t v = input[k];
		int len;
		VARINT_ENCODE_MACRO_32(p, v, len);
		p += len;
	}
	return p - output;
}

int varint_encode_stream(const uint64_t* input, int length, uint8_t* output)
{
	uint8_t* p = output;
	for(int k = 0; k < length; ++k) {
		uint64_t v = input[k];
		int len;
		VARINT_ENCODE_MACRO_64(p, v, len);
		p += len;
	}
	return p - output;
}

int varint_decode_stream(const uint8_t* input, int length, uint32_t* output)
{
	const uint8_t* p = input;
	const uint8_t* p_end = input + length;
	int n = 0;
	for(; p < p_end; ++n) {
		uint32_t v;
		int len;
		VARINT_DECODE_MACRO_32(p, v, len);
		output[n] = v;
		p += len;
	}
	return n;
}

int varint_decode_stream(const uint8_t* input, int length, uint64_t* output)
{
	const uint8_t* p = input;
	const uint8_t* p_end = input + length;
	int n = 0;
	for(; p < p_end; ++n) {
		uint64_t v;
		int len;
		VARINT_DECODE_MACRO_64(p, v, len);
		output[n] = v;
		p += len;
	}
	return n;
}

int varint_encode_stream_gpu_compat(const uint32_t* input, int length, uint8_t* output)
{
	enum { MAX_CODE_LENGTH = VARINT_MAX_CODE_LENGTH_32, SIMD_WIDTH = 32 };
	uint8_t tmp_buffer[SIMD_WIDTH][MAX_CODE_LENGTH];
	uint8_t code_length[SIMD_WIDTH];
	int count[MAX_CODE_LENGTH + 1];

	uint8_t* out_ptr = output;
	for(int i = 0; i < length; i += SIMD_WIDTH) {
		int width = std::min(length - i, (int)SIMD_WIDTH);

		for(int k = 0; k < MAX_CODE_LENGTH; ++k) {
			count[k] = 0;
		}
		count[MAX_CODE_LENGTH] = 0;

		for(int k = 0; k < width; ++k) {
			uint32_t v = input[i + k];
			uint8_t* dst = tmp_buffer[k];
			int len;
			VARINT_ENCODE_MACRO_32(dst, v, len);
			code_length[k] = len;
			for(int r = 0; r < len; ++r) {
				++count[r + 1];
			}
		}

		for(int k = 1; k < MAX_CODE_LENGTH; ++k) count[k + 1] += count[k];

		for(int k = 0; k < width; ++k) {
			for(int r = 0; r < code_length[k]; ++r) {
				out_ptr[count[r]++] = tmp_buffer[k][r];
			}
		}

		out_ptr += count[MAX_CODE_LENGTH];
	}

	return out_ptr - output;
}

int varint_encode_stream_gpu_compat(const uint64_t* input, int length, uint8_t* output)
{
	enum { MAX_CODE_LENGTH = VARINT_MAX_CODE_LENGTH_64, SIMD_WIDTH = 32 };
	uint8_t tmp_buffer[SIMD_WIDTH][MAX_CODE_LENGTH];
	uint8_t code_length[SIMD_WIDTH];
	int count[MAX_CODE_LENGTH + 1];

	uint8_t* out_ptr = output;
	for(int i = 0; i < length; i += SIMD_WIDTH) {
		int width = std::min(length - i, (int)SIMD_WIDTH);

		for(int k = 0; k < MAX_CODE_LENGTH; ++k) {
			count[k] = 0;
		}
		count[MAX_CODE_LENGTH] = 0;

		for(int k = 0; k < width; ++k) {
			uint64_t v = input[i + k];
			uint8_t* dst = tmp_buffer[k];
			int len;
			VARINT_ENCODE_MACRO_64(dst, v, len);
			code_length[k] = len;
			for(int r = 0; r < len; ++r) {
				++count[r + 1];
			}
		}

		for(int k = 1; k < MAX_CODE_LENGTH; ++k) count[k + 1] += count[k];

		for(int k = 0; k < width; ++k) {
			for(int r = 0; r < code_length[k]; ++r) {
				out_ptr[count[r]++] = tmp_buffer[k][r];
			}
		}

		out_ptr += count[MAX_CODE_LENGTH];
	}

	return out_ptr - output;
}

int varint_encode_stream_gpu_compat_signed(const int64_t* input, int length, uint8_t* output)
{
	enum { MAX_CODE_LENGTH = VARINT_MAX_CODE_LENGTH_64, SIMD_WIDTH = 32 };
	uint8_t tmp_buffer[SIMD_WIDTH][MAX_CODE_LENGTH];
	uint8_t code_length[SIMD_WIDTH];
	int count[MAX_CODE_LENGTH + 1];

	uint8_t* out_ptr = output;
	for(int i = 0; i < length; i += SIMD_WIDTH) {
		int width = std::min(length - i, (int)SIMD_WIDTH);

		for(int k = 0; k < MAX_CODE_LENGTH; ++k) {
			count[k] = 0;
		}
		count[MAX_CODE_LENGTH] = 0;

		for(int k = 0; k < width; ++k) {
			int64_t v_raw = input[i + k];
			uint64_t v = (v_raw < 0) ? ((uint64_t)(~v_raw) << 1) | 1 : ((uint64_t)v_raw << 1);
			assert ((int64_t)v >= 0);
			uint8_t* dst = tmp_buffer[k];
			int len;
			VARINT_ENCODE_MACRO_64(dst, v, len);
			code_length[k] = len;
			for(int r = 0; r < len; ++r) {
				++count[r + 1];
			}
		}

		for(int k = 1; k < MAX_CODE_LENGTH; ++k) count[k + 1] += count[k];

		for(int k = 0; k < width; ++k) {
			for(int r = 0; r < code_length[k]; ++r) {
				out_ptr[count[r]++] = tmp_buffer[k][r];
			}
		}

		out_ptr += count[MAX_CODE_LENGTH];
	}

	return out_ptr - output;
}

int varint_get_sparsity_factor(int64_t range, int64_t num_values)
{
	if(num_values == 0) return 0;
	const double sparsity = (double)range / (double)num_values;
	int scale;
	if(sparsity < 1.0)
		scale = 1;
	else if(sparsity < 128)
		scale = 2;
	else if(sparsity < 128LL*128)
		scale = 3;
	else if(sparsity < 128LL*128*128)
		scale = 4;
	else if(sparsity < 128LL*128*128*128)
		scale = 5;
	else if(sparsity < 128LL*128*128*128*128)
		scale = 6;
	else if(sparsity < 128LL*128*128*128*128*128)
		scale = 7;
	else if(sparsity < 128LL*128*128*128*128*128*128)
		scale = 8;
	else if(sparsity < 128LL*128*128*128*128*128*128*128)
		scale = 9;
	else
		scale = 10;
	return scale;
}

namespace memory {

typedef void* (*allocator_t)(const size_t size);

template <typename T>
class Pool {
public:
	Pool(allocator_t allocator__)
		: allocator_(allocator__)
	{
	}
	virtual ~Pool() {
		clear();
	}

	virtual T* get() {
		if(free_list_.empty()) {
			return new (allocator_(sizeof(T))) T();
		}
		T* buffer = free_list_.back();
		free_list_.pop_back();
		return buffer;
	}

	virtual void free(T* buffer) {
		free_list_.push_back(buffer);
	}

	bool empty() const {
		return free_list_.size() == 0;
	}

	size_t size() const {
		return free_list_.size();
	}

	void clear() {
		for(int i = 0; i < (int)free_list_.size(); ++i) {
			free_list_[i]->~T();
			::free(free_list_[i]);
		}
		free_list_.clear();
	}

protected:
	std::vector<T*> free_list_;
	allocator_t allocator_;
};

template <typename T>
class ConcurrentPool : public Pool<T> {
	typedef Pool<T> super_;
public:
	ConcurrentPool(allocator_t allocator__)
		: Pool<T>(allocator__)
	{
		pthread_mutex_init(&thread_sync_, NULL);
	}
	virtual ~ConcurrentPool()
	{
		pthread_mutex_lock(&thread_sync_);
	}

	virtual T* get() {
		pthread_mutex_lock(&thread_sync_);
		if(this->free_list_.empty()) {
			pthread_mutex_unlock(&thread_sync_);
			T* new_buffer = new (this->allocator_(sizeof(T))) T();
			return new_buffer;
		}
		T* buffer = this->free_list_.back();
		this->free_list_.pop_back();
		pthread_mutex_unlock(&thread_sync_);
		return buffer;
	}

	virtual void free(T* buffer) {
		pthread_mutex_lock(&thread_sync_);
		this->free_list_.push_back(buffer);
		pthread_mutex_unlock(&thread_sync_);
	}

	/*
	bool empty() const { return super_::empty(); }
	size_t size() const { return super_::size(); }
	void clear() { super_::clear(); }
	*/
protected:
	pthread_mutex_t thread_sync_;
};

template <typename T>
class vector_w : public std::vector<T*>
{
	typedef std::vector<T*> super_;
public:
	~vector_w() {
		for(typename super_::iterator it = this->begin(); it != this->end(); ++it) {
			(*it)->~T();
			::free(*it);
		}
		super_::clear();
	}
};

template <typename T>
class deque_w : public std::deque<T*>
{
	typedef std::deque<T*> super_;
public:
	~deque_w() {
		for(typename super_::iterator it = this->begin(); it != this->end(); ++it) {
			(*it)->~T();
			::free(*it);
		}
		super_::clear();
	}
};

template <typename T>
class Store {
public:
	Store() {
	}
	void init(Pool<T>* pool) {
		pool_ = pool;
		filled_length_ = 0;
		buffer_length_ = 0;
		resize_buffer(16);
	}
	~Store() {
		for(int i = 0; i < filled_length_; ++i){
			pool_->free(buffer_[i]);
		}
		filled_length_ = 0;
		buffer_length_ = 0;
		::free(buffer_); buffer_ = NULL;
	}

	void submit(T* value) {
		const int offset = filled_length_++;

		if(buffer_length_ == filled_length_)
			expand();

		buffer_[offset] = value;
	}

	void clear() {
		for(int i = 0; i < filled_length_; ++i){
			buffer_[i]->clear();
			assert (buffer_[i]->size() == 0);
			pool_->free(buffer_[i]);
		}
		filled_length_ = 0;
	}

	T* front() {
		if(filled_length_ == 0) {
			push();
		}
		return buffer_[filled_length_ - 1];
	}

	void push() {
		submit(pool_->get());
	}

	int64_t size() const { return filled_length_; }
	T* get(int index) const { return buffer_[index]; }
private:

	void resize_buffer(int allocation_size)
	{
		T** new_buffer = (T**)malloc(allocation_size*sizeof(buffer_[0]));
		if(buffer_length_ != 0) {
			memcpy(new_buffer, buffer_, filled_length_*sizeof(buffer_[0]));
			::free(buffer_);
		}
		buffer_ = new_buffer;
		buffer_length_ = allocation_size;
	}

	void expand()
	{
		if(filled_length_ == buffer_length_)
			resize_buffer(std::max<int64_t>(buffer_length_*2, 16));
	}

	int64_t filled_length_;
	int64_t buffer_length_;
	T** buffer_;
	Pool<T>* pool_;
};

} // namespace memory

namespace profiling {

class ProfilingInformationStore {
public:
	void submit(double span, const char* content, int n1, int n2 = 0) {
#pragma omp critical (pis_submit_time)
		times_.push_back(TimeElement(span, content, n1, n2));
	}
	void submit(int64_t span_micro, const char* content, int n1, int n2 = 0) {
#pragma omp critical (pis_submit_time)
		times_.push_back(TimeElement((double)span_micro / 1000000.0, content, n1, n2));
	}
	void submitCounter(int64_t counter, const char* content, int n1, int n2 = 0) {
#pragma omp critical (pis_submit_counter)
		counters_.push_back(CountElement(counter, content, n1, n2));
	}
	void reset() {
		times_.clear();
		counters_.clear();
	}
	void printResult() {
		printTimeResult();
		printCountResult();
	}
private:
	struct TimeElement {
		double span;
		const char* content;
		int n1, n2;

		TimeElement(double span__, const char* content__, int n1__, int n2__)
			: span(span__), content(content__), n1(n1__), n2(n2__) { }
	};
	struct CountElement {
		int64_t count;
		const char* content;
		int n1, n2;

		CountElement(int64_t count__, const char* content__, int n1__, int n2__)
			: count(count__), content(content__), n1(n1__), n2(n2__) { }
	};

	template <typename T> void calcDiff(T* src, T* sum, T* dst, int length, int count) {
		for(int i = 0; i < length; ++i) {
			T diff = src[i] - sum[i] / count;
			dst[i] = diff * diff;
		}
	}

	void printTimeResult() {
		int num_times = times_.size();
		double *dbl_times = new double[num_times];
		double *sum_times = new double[num_times];
		double *max_times = new double[num_times];

		for(int i = 0; i < num_times; ++i) {
			dbl_times[i] = times_[i].span;
		}

		MPI_Allreduce(dbl_times, sum_times, num_times, MPI_DOUBLE, MPI_SUM, mpi.comm_2d);
		MPI_Reduce(dbl_times, max_times, num_times, MPI_DOUBLE, MPI_MAX, 0, mpi.comm_2d);
		calcDiff(dbl_times, sum_times, dbl_times, num_times, mpi.size_2d);
		MPI_Reduce(mpi.rank_2d == 0 ? MPI_IN_PLACE : dbl_times, dbl_times, num_times, MPI_DOUBLE, MPI_SUM, 0, mpi.comm_2d);

		if(mpi.isMaster()) {
			fprintf(stderr, "Name, Outer Step, Inner Step, Average, Maximum, Standard Deviation, (ms)\n");
			for(int i = 0; i < num_times; ++i) {
				double avg = sum_times[i] / mpi.size_2d * 1000.0, maximum = max_times[i] * 1000.0;
				double stddev = sqrt(dbl_times[i] / mpi.size_2d) * 1000.0;
				fprintf(stderr, "Time of %s, %d, %d, %f, %f, %f\n",
						times_[i].content, times_[i].n1, times_[i].n2,
						avg, maximum, stddev);
			}
		}

		delete [] dbl_times;
		delete [] sum_times;
		delete [] max_times;
	}

	double displayValue(int64_t value) {
		if(value < int64_t(1000))
			return (double)value;
		else if(value < int64_t(1000)*1000)
			return value / 1000.0;
		else if(value < int64_t(1000)*1000*1000)
			return value / (1000.0*1000);
		else if(value < int64_t(1000)*1000*1000*1000)
			return value / (1000.0*1000*1000);
		else
			return value / (1000.0*1000*1000*1000);
	}

	const char* displaySuffix(int64_t value) {
		if(value < int64_t(1000))
			return "";
		else if(value < int64_t(1000)*1000)
			return "K";
		else if(value < int64_t(1000)*1000*1000)
			return "M";
		else if(value < int64_t(1000)*1000*1000*1000)
			return "G";
		else
			return "T";
	}

	void printCountResult() {
		int num_times = counters_.size();
		int64_t *dbl_times = new int64_t[num_times];
		int64_t *sum_times = new int64_t[num_times];
		int64_t *max_times = new int64_t[num_times];

		for(int i = 0; i < num_times; ++i) {
			dbl_times[i] = counters_[i].count;
		}

		MPI_Allreduce(dbl_times, sum_times, num_times, MPI_INT64_T, MPI_SUM, mpi.comm_2d);
		MPI_Reduce(dbl_times, max_times, num_times, MPI_INT64_T, MPI_MAX, 0, mpi.comm_2d);
		calcDiff(dbl_times, sum_times, dbl_times, num_times, mpi.size_2d);
		MPI_Reduce(mpi.rank_2d == 0 ? MPI_IN_PLACE : dbl_times, dbl_times, num_times, MPI_INT64_T, MPI_SUM, 0, mpi.comm_2d);

		if(mpi.isMaster()) {
			fprintf(stderr, "Name, Outer Step, Inner Step, Summation, Average, Maximum, Standard Deviation\n");
			for(int i = 0; i < num_times; ++i) {
				int64_t sum = sum_times[i], avg = sum_times[i] / mpi.size_2d, maximum = max_times[i];
				int64_t stddev = sqrt(dbl_times[i] / mpi.size_2d);
				fprintf(stderr, "Count %s, %d, %d, %ld, %ld, %ld, %ld\n",
						counters_[i].content, counters_[i].n1, counters_[i].n2,
						sum, avg, maximum, stddev);
			}
		}

		delete [] dbl_times;
		delete [] sum_times;
		delete [] max_times;
	}

	std::vector<TimeElement> times_;
	std::vector<CountElement> counters_;
};

ProfilingInformationStore g_pis;

class TimeKeeper {
public:
	TimeKeeper() : start_(get_time_in_microsecond()){ }
	void submit(const char* content, int n1, int n2 = 0) {
		int64_t end = get_time_in_microsecond();
		g_pis.submit(end - start_, content, n1, n2);
		start_ = end;
	}
	int64_t getSpanAndReset() {
		int64_t end = get_time_in_microsecond();
		int64_t span = end - start_;
		start_ = end;
		return span;
	}
private:
	int64_t start_;
};

class TimeSpan {
	TimeSpan(int64_t init) : span_(init) { }
public:
	TimeSpan() : span_(0) { }
	TimeSpan(TimeKeeper& keeper) : span_(keeper.getSpanAndReset()) { }

	void reset() { span_ = 0; }
	TimeSpan& operator += (TimeKeeper& keeper) {
		__sync_fetch_and_add(&span_, keeper.getSpanAndReset());
		return *this;
	}
	TimeSpan& operator -= (TimeKeeper& keeper) {
		__sync_fetch_and_add(&span_, - keeper.getSpanAndReset());
		return *this;
	}
	TimeSpan& operator += (TimeSpan span) {
		__sync_fetch_and_add(&span_, span.span_);
		return *this;
	}
	TimeSpan& operator -= (TimeSpan span) {
		__sync_fetch_and_add(&span_, - span.span_);
		return *this;
	}
	TimeSpan& operator += (int64_t span) {
		__sync_fetch_and_add(&span_, span);
		return *this;
	}
	TimeSpan& operator -= (int64_t span) {
		__sync_fetch_and_add(&span_, - span);
		return *this;
	}

	TimeSpan operator + (TimeSpan span) {
		return TimeSpan(span_ + span.span_);
	}
	TimeSpan operator - (TimeSpan span) {
		return TimeSpan(span_ - span.span_);
	}

	void submit(const char* content, int n1, int n2 = 0) {
		g_pis.submit(span_, content, n1, n2);
		span_ = 0;
	}
	double getSpan() {
		return (double)span_ / 1000000.0;
	}
private:
	int64_t span_;
};

} // namespace profiling

#if VERVOSE_MODE
volatile int64_t g_fold_send;
volatile int64_t g_fold_recv;
volatile int64_t g_bitmap_send;
volatile int64_t g_bitmap_recv;
volatile int64_t g_exs_send;
volatile int64_t g_exs_recv;
volatile double g_gpu_busy_time;
#endif

/* edgefactor = 16, seed1 = 2, seed2 = 3 */
int64_t pf_nedge[] = {
	-1,
	32, // 1
	64,
	128,
	256,
	512,
	1024,
	2048,
	4096 , // 8
	8192 ,
	16383 ,
	32767 ,
	65535 ,
	131070 ,
	262144 ,
	524285 ,
	1048570 ,
	2097137 ,
	4194250 ,
	8388513 ,
	16776976 ,
	33553998 ,
	67108130 ,
	134216177 ,
	268432547 ,
	536865258 ,
	1073731075 ,
	2147462776 ,
	4294927670 ,
	8589858508 ,
	17179724952 ,
	34359466407 ,
	68718955183 , // = 2^36 - 521553
	137437972330, // 33
	274876029861, // 34
	549752273512, // 35
	1099505021204, // 36
	0, // 37
	0 // 38
};

#endif /* UTILS_IMPL_HPP_ */
