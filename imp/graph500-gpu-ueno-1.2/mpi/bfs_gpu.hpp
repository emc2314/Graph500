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
#ifndef BFS_GPU_HPP_
#define BFS_GPU_HPP_

#include "bfs.hpp"
#include "gpu_host.hpp"
#include "gpu_core.h"

#include <vector>

#include <unistd.h>

struct BfsOnGPU_Params {
	typedef uint32_t BitmapType;
	enum {
		LOG_PACKING_EDGE_LISTS = GPU_PARAMS::LOG_PACKING_EDGE_LISTS,
		LOG_CQ_SUMMARIZING = GPU_PARAMS::LOG_CQ_SUMMARIZING,
	};
};

void cuda_memcheck(bool req_ctx) {
	if(req_ctx) CudaStreamManager::begin_cuda();
	void *ptr;
	CUDA_CHECK(cudaMallocHost(&ptr, 1024*40));
	CUDA_CHECK(cudaFreeHost(ptr));
	if(req_ctx) CudaStreamManager::end_cuda();
}

template <typename LocalVertsIndex, bool graph_on_gpu_ = true>
class BfsOnGPU
	: public BfsBase<Pack48bit, LocalVertsIndex, BfsOnGPU_Params>
{
	typedef BfsOnGPU<LocalVertsIndex, graph_on_gpu_> ThisType;
	typedef BfsBase<Pack48bit, LocalVertsIndex, BfsOnGPU_Params> super_;
	typedef typename BfsOnGPU_Params::BitmapType BitmapType;
public:
	enum {
		MAX_NUM_STREAM_JOBS = 128,
#if DISABLE_CUDA_CONCCURENT
		NUM_IO_BUFFERS = 1,
		NUM_CUDA_STREAMS = 1,
#else
		NUM_IO_BUFFERS = graph_on_gpu_ ? 3 : 4,
		NUM_CUDA_STREAMS = 4,
#endif
	};
	BfsOnGPU()
		: super_(true)
		, cuda_man_(CudaStreamManager::get_instance())
		, gpu_cq_comm_(this, true)
		, gpu_visited_comm_(this, false)
		, cu_recv_task_(65536)
		, create_cq_list_(this)
	{
	}

	virtual ~BfsOnGPU()
	{
		CudaStreamManager::begin_cuda();
		// delete device graph
		if(graph_on_gpu_) {
			CUDA_CHECK(cudaFree(this->dev_row_starts_)); this->dev_row_starts_ = NULL;
			CUDA_CHECK(cudaFree(this->dev_index_array_high_)); this->dev_index_array_high_ = NULL;
			CUDA_CHECK(cudaFree(this->dev_index_array_low_)); this->dev_index_array_low_ = NULL;
		}
		CUDA_CHECK(cudaFree(this->dev_invert_vertex_mapping_)); this->dev_invert_vertex_mapping_ = NULL;
		CudaStreamManager::end_cuda();
	}

	template <typename EdgeList>
	void construct(EdgeList* edge_list)
	{
		super_::construct(edge_list);

		// transfer data to GPU
		int64_t num_columns = this->get_number_of_edge_columns();
		int64_t index_size = this->graph_.row_starts_[num_columns];
		int64_t num_local_vertices = this->get_number_of_local_vertices();

		CudaStreamManager::begin_cuda();
		if(graph_on_gpu_) {
			CUDA_CHECK(cudaMalloc((void**)&this->dev_row_starts_,
					sizeof(this->dev_row_starts_[0])*(num_columns+2)));
			CUDA_CHECK(cudaMalloc((void**)&this->dev_index_array_high_,
					sizeof(this->dev_index_array_high_[0])*index_size));
			CUDA_CHECK(cudaMalloc((void**)&this->dev_index_array_low_,
					sizeof(this->dev_index_array_low_[0])*index_size));
		}
		else {
			this->dev_row_starts_ = NULL;
			this->dev_index_array_high_ = NULL;
			this->dev_index_array_low_ = NULL;
		}
		CUDA_CHECK(cudaMalloc((void**)&this->dev_invert_vertex_mapping_,
				sizeof(this->dev_invert_vertex_mapping_[0])*num_local_vertices));

		if(graph_on_gpu_) {
			CUDA_CHECK(cudaMemcpy(this->dev_row_starts_, this->graph_.row_starts_,
					sizeof(this->dev_row_starts_[0])*(num_columns+1), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(this->dev_index_array_high_, this->graph_.index_array_.get_ptr_high(),
					sizeof(this->dev_index_array_high_[0])*index_size, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(this->dev_index_array_low_, this->graph_.index_array_.get_ptr_low(),
					sizeof(this->dev_index_array_low_[0])*index_size, cudaMemcpyHostToDevice));
			// add an empty column
			CUDA_CHECK(cudaMemcpy(this->dev_row_starts_ + num_columns + 1, &index_size,
					sizeof(this->dev_row_starts_[0]), cudaMemcpyHostToDevice));
		}
		CUDA_CHECK(cudaMemcpy(this->dev_invert_vertex_mapping_, this->graph_.invert_vertex_mapping_,
				sizeof(this->dev_invert_vertex_mapping_[0])*num_local_vertices, cudaMemcpyHostToDevice));
		CudaStreamManager::end_cuda();

#if 0
		printf("L:%d:map[93982]=%d\n", __LINE__, this->graph_.vertex_mapping_[93982]);
#endif

		//cuda_memcheck(true);
	}

	void prepare_bfs() {
		// print information of base class
		super_::printInformation();

		allocate_memory_gpu();

		using namespace GPU_PARAMS;
		if(mpi.isMaster()) {
			fprintf(stderr, "graph_on_gpu_=%d.\n", graph_on_gpu_);
			fprintf(stderr, "WARPS_PER_BLOCK=%d.\n", WARPS_PER_BLOCK);
			fprintf(stderr, "THREADS_PER_BLOCK=%d.\n", THREADS_PER_BLOCK);
			fprintf(stderr, "ACTIVE_THREAD_BLOCKS=%d.\n", ACTIVE_THREAD_BLOCKS);
			fprintf(stderr, "MAX_ACTIVE_WARPS=%d.\n", MAX_ACTIVE_WARPS);
			fprintf(stderr, "READ_GRAPH_OUTBUF_SIZE=%d.\n", READ_GRAPH_OUTBUF_SIZE);
			fprintf(stderr, "EXPAND_DECODE_BLOCK_LENGTH=%d.\n", EXPAND_DECODE_BLOCK_LENGTH);
			fprintf(stderr, "NUM_IO_BUFFERS=%d.\n", NUM_IO_BUFFERS);
			fprintf(stderr, "sizeof(ReadGraphBuffer)=%f MB.\n", sizeof(cuda::ReadGraphBuffer) / 1000000.0);
			fprintf(stderr, "sizeof(RecvProcBuffer)=%f MB.\n", sizeof(cuda::RecvProcBuffer) / 1000000.0);
			fprintf(stderr, "sizeof(FoldIOBuffer)=%f MB.\n", sizeof(cuda::FoldIOBuffer) / 1000000.0);
			fprintf(stderr, "sizeof(FoldGpuBuffer)=%f MB.\n", sizeof(cuda::FoldGpuBuffer) / 1000000.0);
			fprintf(stderr, "sizeof(UpdateProcBuffer)=%f MB.\n", sizeof(cuda::UpdateProcBuffer) / 1000000.0);
			fprintf(stderr, "blocks_per_launch=%d\n", this->blocks_per_launch_);

			int64_t num_columns = this->get_number_of_edge_columns();
			int64_t index_size = this->graph_.row_starts_[num_columns];
			int64_t num_local_vertices = this->get_number_of_local_vertices();

			double device_graph_data_size =
					sizeof(this->dev_row_starts_[0])*(num_columns+2) +
					sizeof(this->dev_index_array_high_[0])*index_size +
					sizeof(this->dev_index_array_low_[0])*index_size +
					sizeof(this->dev_invert_vertex_mapping_[0])*num_local_vertices;
			double bfs_fixed_data_size =
					sizeof(int64_t) * this->get_actual_number_of_local_vertices() +
					sizeof(this->cq_bitmap_[0])*this->get_bitmap_size_v0() +
					sizeof(this->cq_summary_[0])*this->get_summary_size_v0() +
					sizeof(this->shared_visited_[0])*this->get_bitmap_size_v1() +
					sizeof(this->nq_bitmap_[0])*this->get_bitmap_size_visited() +
					sizeof(this->nq_bitmap_[0])*this->get_bitmap_size_visited() +
					sizeof(this->visited_[0])*this->get_bitmap_size_visited();
			double bfs_temp_buffer_size =
					sizeof(int)*this->get_summary_size_v0() +
					sizeof(this->dev_[0]) +
					std::max<int64_t>(
						sizeof(this->dev_column_buffer_[0]) * this->get_bitmap_size_v0() +
						sizeof(cuda::FoldIOBuffer) * NUM_IO_BUFFERS + sizeof(cuda::FoldGpuBuffer),
						sizeof(cuda::UpdateProcBuffer) * NUM_IO_BUFFERS);

			fprintf(stderr, "device_graph_data_size=%f MB.\n", device_graph_data_size / 1000000.0);
			fprintf(stderr, "device_bfs_fixed_data_size=%f MB.\n", bfs_fixed_data_size / 1000000.0);
			fprintf(stderr, "device_bfs_temp_buffer_size=%f MB.\n", bfs_temp_buffer_size / 1000000.0);
		}
	}

	void run_bfs(int64_t root, int64_t* pred)
	{
		using namespace BFS_PARAMS;
#if VERVOSE_MODE
		double tmp = MPI_Wtime();
		double start_time = tmp;
		double prev_time = tmp;
		double expand_time = 0.0, fold_time = 0.0, stall_time = 0.0;
		g_fold_send = g_fold_recv = g_bitmap_send = g_bitmap_recv = g_exs_send = g_exs_recv = 0;
		kernel_launch_create_cq_list_ = 0;
		kernel_launch_read_graph_ = 0;
		retry_count_read_graph_ = 0;
		kernel_launch_receive_process_ = 0;
		kernel_launch_update_bitmap_ = 0;
		cuda_io_bytes_ = 0;

		int kernel_launch_read_graph_total = 0;
		int retry_count_read_graph_total = 0;
		int kernel_launch_receive_process_total = 0;
		int64_t cuda_io_bytes_total = 0;
#endif

		const int log_size = get_msb_index(mpi.size_2d);
		const int size_mask = mpi.size_2d - 1;
#define VERTEX_OWNER(v) ((v) & size_mask)
#define VERTEX_LOCAL(v) ((v) >> log_size)

		// threshold of scheduling for extracting CQ. (for read graph from CPU)
		enum {
			LONG_JOB_WIDTH = 8,
			SHORT_JOB_WIDTH = 8*16,
		};
		const int64_t sched_threshold = this->get_number_of_local_vertices() * mpi.size_2dr / 16;
		const int long_job_length = get_blocks(this->get_summary_size_v0(), LONG_JOB_WIDTH);
		const int short_job_length = get_blocks(this->get_summary_size_v0(), SHORT_JOB_WIDTH);

		//cuda_memcheck(true);
		expand_job_id_ = 0;
		fold_io_buf_count_ = 0;

		initialize_memory_gpu();

		//cuda_memcheck(true);

		super_::current_level_ = 0;
		int root_owner = (int)VERTEX_OWNER(root);
		if(root_owner == mpi.rank_2d) {
			int64_t root_local = VERTEX_LOCAL(root);

#if VERTEX_SORTING
			int64_t sorted_root_local = this->graph_.vertex_mapping_[root_local];
			int64_t word_idx = sorted_root_local / super_::NUMBER_PACKING_EDGE_LISTS;
			int bit_idx = sorted_root_local % super_::NUMBER_PACKING_EDGE_LISTS;
#else
			int64_t word_idx = root / super_::NUMBER_PACKING_EDGE_LISTS;
			int bit_idx = root % super_::NUMBER_PACKING_EDGE_LISTS;
#endif
			host_->root = root;
			host_->mask = BitmapType(1) << bit_idx;

			//cuda_memcheck(true);

			CudaStreamManager::begin_cuda();
			CUDA_CHECK(cudaMemcpy(this->pred_ + root_local, &host_->root,
					sizeof(int64_t), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(dev_visited_ + word_idx, &host_->mask,
					sizeof(BitmapType), cudaMemcpyHostToDevice));
#if VERVOSE_MODE
			cuda_io_bytes_ += sizeof(int64_t) + sizeof(BitmapType);
#endif
			CudaStreamManager::end_cuda();

			this->expand_root(root_local, &gpu_cq_comm_, &gpu_visited_comm_);
		}
		else {
			this->expand_root(-1, &gpu_cq_comm_, &gpu_visited_comm_);
		}
#if VERVOSE_MODE
	tmp = MPI_Wtime();
	if(mpi.isMaster()) fprintf(stderr, "Time of first expansion: %f ms\n", (tmp - prev_time) * 1000.0);
	expand_time += tmp - prev_time; prev_time = tmp;
#endif
#undef VERTEX_OWNER
#undef VERTEX_LOCAL

		//cuda_memcheck(true);

		int64_t num_columns_per_launch = blocks_per_launch_ * GPU_PARAMS::THREADS_PER_BLOCK;

		while(true) {
			++super_::current_level_;
#if VERVOSE_MODE
		double level_start_time = MPI_Wtime();
#endif

			prepare_fold();

			this->fiber_man_.begin_processing();

			// compute number of jobs
			int num_submit_jobs;
			if(graph_on_gpu_) {
				this->d_->num_remaining_extract_jobs_ =
						(this->host_->num_non_zero_columns + num_columns_per_launch - 1)
									/ num_columns_per_launch;
				num_submit_jobs = std::min<int>(this->d_->num_remaining_extract_jobs_, num_stream_jobs_);
			}
			else {
				int number_of_cpu_jobs;
				if(this->d_->num_vertices_in_cq_ >= sched_threshold) {
					columns_per_task_ = LONG_JOB_WIDTH;
					number_of_cpu_jobs = long_job_length;
				}
				else {
					columns_per_task_ = SHORT_JOB_WIDTH;
					number_of_cpu_jobs = short_job_length;
				}
				this->d_->num_remaining_extract_jobs_ =	num_submit_jobs
						= std::min<int>(std::min<int>(number_of_cpu_jobs, omp_get_max_threads()+5), num_stream_jobs_);
			}
#if VERVOSE_MODE
			if(mpi.isMaster()) fprintf(stderr, "num_submit_jobs=%d\n", num_submit_jobs);
#endif

			for(int i = 0; i < num_submit_jobs; ++i) {
				if(graph_on_gpu_)
					cuda_man_->submit(extract_edge_array_[i], 0);
				else
					this->fiber_man_.submit(extract_edge_array_[i], 0);
			}
			if(num_submit_jobs == 0) {
				this->fiber_man_.submit_array(this->sched_.fold_end_job, mpi.size_2dc, 0);
			}

//#pragma omp parallel
			this->fiber_man_.enter_processing();
#if VERVOSE_MODE
		tmp = MPI_Wtime(); fold_time += tmp - prev_time; prev_time = tmp;
#endif
			int64_t num_nq_vertices = host_->nq_count;
			int64_t global_nq_vertices;
			MPI_Allreduce(&num_nq_vertices, &global_nq_vertices, 1,
					get_mpi_type(num_nq_vertices), MPI_SUM, mpi.comm_2d);
#if VERVOSE_MODE
		tmp = MPI_Wtime(); stall_time += tmp - prev_time; prev_time = tmp;
		extract_cuman_cpu_time_.submit("extract cuda man cpu", super_::current_level_);
		fold_compress_time_.submit("cpu fold compress", super_::current_level_);
		proc_fold_compress_time_.submit("cpu fold compress thread", super_::current_level_);
		this->comm_.submit_wait_time("fold send wait", super_::current_level_);
		recv_cuman_cpu_time_.submit("recv cuda man cpu", super_::current_level_);
		create_cq_kernel_time_.submit("kernel create cq", super_::current_level_);
		read_graph_kernel_time_.submit("kernel read graph", super_::current_level_);
		filter_edge_kernel_time_.submit("kernel filte edge", super_::current_level_);
		recv_proc_kernel_time_.submit("kernel recv proc", super_::current_level_);

		profiling::g_pis.submitCounter(kernel_launch_read_graph_, "kernel_launch_read_graph_", super_::current_level_);
		profiling::g_pis.submitCounter(retry_count_read_graph_, "retry_count_read_graph_", super_::current_level_);
		profiling::g_pis.submitCounter(kernel_launch_receive_process_, "kernel_launch_receive_process_", super_::current_level_);
		profiling::g_pis.submitCounter(cuda_io_bytes_, "cuda_io_bytes_", super_::current_level_);

		kernel_launch_read_graph_total += kernel_launch_read_graph_; kernel_launch_read_graph_ = 0;
		retry_count_read_graph_total += retry_count_read_graph_; retry_count_read_graph_ = 0;
		kernel_launch_receive_process_total += kernel_launch_receive_process_; kernel_launch_receive_process_ = 0;
		cuda_io_bytes_total += cuda_io_bytes_; cuda_io_bytes_ = 0;

		if(mpi.isMaster()) fprintf(stderr, "global_nq_vertices=%"PRId64"\n", global_nq_vertices);
#endif
			if(global_nq_vertices == 0)
				break;

			this->expand(global_nq_vertices, &gpu_cq_comm_, &gpu_visited_comm_);
#if VERVOSE_MODE
		tmp = MPI_Wtime();
		if(mpi.isMaster()) fprintf(stderr, "Time of levle %d: %f ms\n", this->current_level_, (MPI_Wtime() - level_start_time) * 1000.0);
		expand_time += tmp - prev_time; prev_time = tmp;
#endif
		}
#if VERVOSE_MODE
	if(mpi.isMaster()) fprintf(stderr, "Time of BFS: %f ms\n", (MPI_Wtime() - start_time) * 1000.0);
	double time3[3] = { fold_time, expand_time, stall_time };
	double timesum3[3];
	int64_t commd[7] = { g_fold_send, g_fold_recv, g_bitmap_send, g_bitmap_recv, g_exs_send, g_exs_recv, cuda_io_bytes_total };
	int64_t commdsum[7];
	int kernel_count[5] = { kernel_launch_create_cq_list_, kernel_launch_read_graph_total, kernel_launch_receive_process_total, kernel_launch_update_bitmap_, retry_count_read_graph_total };
	int kernel_count_sum[5];
	MPI_Reduce(time3, timesum3, 3, MPI_DOUBLE, MPI_SUM, 0, mpi.comm_2d);
	MPI_Reduce(commd, commdsum, 7, get_mpi_type(commd[0]), MPI_SUM, 0, mpi.comm_2d);
	MPI_Reduce(kernel_count, kernel_count_sum, 5, get_mpi_type(kernel_count[0]), MPI_SUM, 0, mpi.comm_2d);
	if(mpi.isMaster()) {
		fprintf(stderr, "Avg time of fold: %f ms\n", timesum3[0] / mpi.size_2d * 1000.0);
		fprintf(stderr, "Avg time of expand: %f ms\n", timesum3[1] / mpi.size_2d * 1000.0);
		fprintf(stderr, "Avg time of stall: %f ms\n", timesum3[2] / mpi.size_2d * 1000.0);
		fprintf(stderr, "Avg fold_send: %"PRId64"\n", commdsum[0] / mpi.size_2d);
		fprintf(stderr, "Avg fold_recv: %"PRId64"\n", commdsum[1] / mpi.size_2d);
		fprintf(stderr, "Avg bitmap_send: %"PRId64"\n", commdsum[2] / mpi.size_2d);
		fprintf(stderr, "Avg bitmap_recv: %"PRId64"\n", commdsum[3] / mpi.size_2d);
		fprintf(stderr, "Avg exs_send: %"PRId64"\n", commdsum[4] / mpi.size_2d);
		fprintf(stderr, "Avg exs_recv: %"PRId64"\n", commdsum[5] / mpi.size_2d);
		fprintf(stderr, "Avg cuda io bytes: %"PRId64"\n", commdsum[6] / mpi.size_2d);
		fprintf(stderr, "Avg kernel_launch_create_cq_list_=%f\n", (double)kernel_count_sum[0] / mpi.size_2d);
		fprintf(stderr, "Avg kernel_launch_read_graph_=%f\n", (double)kernel_count_sum[1] / mpi.size_2d);
		fprintf(stderr, "Avg kernel_launch_receive_process_=%f\n", (double)kernel_count_sum[2] / mpi.size_2d);
		fprintf(stderr, "Avg kernel_launch_update_bitmap_=%f\n", (double)kernel_count_sum[3] / mpi.size_2d);
		fprintf(stderr, "Avg retry_count_read_graph_=%f\n", (double)kernel_count_sum[4] / mpi.size_2d);
	}
	MPI_Reduce(time3, timesum3, 3, MPI_DOUBLE, MPI_MAX, 0, mpi.comm_2d);
	MPI_Reduce(commd, commdsum, 7, get_mpi_type(commd[0]), MPI_MAX, 0, mpi.comm_2d);
	MPI_Reduce(kernel_count, kernel_count_sum, 5, get_mpi_type(kernel_count[0]), MPI_MAX, 0, mpi.comm_2d);
	if(mpi.isMaster()) {
		fprintf(stderr, "Max time of fold: %f ms\n", timesum3[0] * 1000.0);
		fprintf(stderr, "Max time of expand: %f ms\n", timesum3[1] * 1000.0);
		fprintf(stderr, "Max time of stall: %f ms\n", timesum3[2] * 1000.0);
		fprintf(stderr, "Max fold_send: %"PRId64"\n", commdsum[0]);
		fprintf(stderr, "Max fold_recv: %"PRId64"\n", commdsum[1]);
		fprintf(stderr, "Max bitmap_send: %"PRId64"\n", commdsum[2]);
		fprintf(stderr, "Max bitmap_recv: %"PRId64"\n", commdsum[3]);
		fprintf(stderr, "Max exs_send: %"PRId64"\n", commdsum[4]);
		fprintf(stderr, "Max exs_recv: %"PRId64"\n", commdsum[5]);
		fprintf(stderr, "Max cuda io bytes: %"PRId64"\n", commdsum[6]);
		fprintf(stderr, "Max kernel_launch_create_cq_list_=%f\n", (double)kernel_count_sum[0]);
		fprintf(stderr, "Max kernel_launch_read_graph_=%f\n", (double)kernel_count_sum[1]);
		fprintf(stderr, "Max kernel_launch_receive_process_=%f\n", (double)kernel_count_sum[2]);
		fprintf(stderr, "Max kernel_launch_update_bitmap_=%f\n", (double)kernel_count_sum[3]);
		fprintf(stderr, "Max retry_count_read_graph_=%f\n", (double)kernel_count_sum[4]);
	}
#endif
	}

	void get_pred(int64_t* pred) {
		CudaStreamManager::begin_cuda();
		CUDA_CHECK(cudaMemcpy(pred, this->pred_,
				sizeof(int64_t) * this->get_actual_number_of_local_vertices(),
				cudaMemcpyDeviceToHost));
		CudaStreamManager::end_cuda();
	}

	void end_bfs() {
		deallocate_memory_gpu();
	}

// protected:

	struct CuCreateCQList : public CudaCommand
	{
		CuCreateCQList(ThisType* this__) : this_(this__) { }
		virtual void send(cudaStream_t stream) { };
		virtual void launch_kernel(cudaStream_t stream) {
#if VERVOSE_MODE
			profiling::TimeKeeper tk;
#endif
			cudaStream_t master_stream = this_->cuda_stream_array_[0];
			// per level initialization is here.
			this_->host_->nq_count = 0;
			CUDA_CHECK(cudaMemcpyAsync(&this_->dev_->nq_count,
					&this_->host_->nq_count, sizeof(this_->host_->nq_count),
					cudaMemcpyHostToDevice, master_stream));
#if VERVOSE_MODE
			this_->cuda_io_bytes_ +=  sizeof(this_->host_->nq_count);
#endif

			cuda::create_cq_list(this_->dev_cq_bitmap_, this_->dev_cq_summary_,
					this_->get_summary_size_v0(), this_->dev_cq_count_,
					this_->dev_column_buffer_, this_->get_number_of_edge_columns(),
					master_stream);

#if VERVOSE_MODE
			++this_->kernel_launch_create_cq_list_;
			tk.submit("create cq kernel", this_->current_level_);
#endif
		};
		virtual void receive(cudaStream_t stream) {
			int* dev_last_offset = &this_->dev_cq_count_->get_buffer()[this_->get_summary_size_v0()];
			CUDA_CHECK(cudaMemcpyAsync(&this_->host_->num_non_zero_columns, dev_last_offset,
					sizeof(dev_last_offset[0]), cudaMemcpyDeviceToHost, stream));
#if VERVOSE_MODE
			this_->cuda_io_bytes_ += sizeof(dev_last_offset[0]);
#endif

		};
		virtual void complete() {
			this_->fiber_man_.end_processing();
		};
		ThisType* this_;
	};

	struct CuExtractEdge : public CudaCommand, public Runnable
	{
		CuExtractEdge(ThisType* this__, cuda::FoldIOBuffer* host_buf)
			: this_(this__)
			, host_buf_(host_buf)
			, dev_buf_(NULL)
			, column_start_(0)
			, column_end_(0)
			, need_retry_(false)
		{ }
		virtual ~CuExtractEdge() { }
		virtual bool init() {
#if VERVOSE_MODE
			profiling::TimeKeeper tk;
#endif
			// get job and assign device buffer
			int64_t num_columns_per_launch = this_->blocks_per_launch_ * GPU_PARAMS::THREADS_PER_BLOCK;

			if(need_retry_ == false) {
				column_start_ = this_->current_column_index_;
				if(column_start_ >= this_->host_->num_non_zero_columns) {
					return false;
				}
				this_->current_column_index_ += num_columns_per_launch;
			}

			column_end_ = std::min<int64_t>(this_->host_->num_non_zero_columns,
					column_start_ + num_columns_per_launch);
			num_blocks_ = get_blocks<GPU_PARAMS::THREADS_PER_BLOCK>(column_end_ - column_start_);
/*
			cuda::ReadGraphInput* input = &host_buf_->input;
			input->intermid_offset = 0;
			input->output_offset = 0;
			input->skipped = false;
*/
#if VERVOSE_MODE
			this_->extract_cuman_cpu_time_ += tk;
#endif
			return true;
		}
		virtual void send(cudaStream_t stream) { }
		virtual void launch_kernel(cudaStream_t stream) {
#if VERVOSE_MODE
			profiling::TimeKeeper tk;
#endif
#if 0
			printf("L:%d:R(%d):launch read_graph kernel\n", __LINE__, mpi.rank_2d);
#endif
			dev_buf_ = &this_->dev_fold_io_buffer_[(this_->fold_io_buf_count_++) % NUM_IO_BUFFERS].read_graph;
			// launch kernel
			cuda::read_graph_1(&dev_buf_->input,
					this_->dev_row_starts_,
					this_->dev_index_array_high_,
					this_->dev_index_array_low_,
					this_->dev_column_buffer_,
					column_start_, column_end_,
					this_->get_number_of_edge_columns(),
					this_->dev_shared_visited_,
					this_->dev_fold_proc_buffer_->read_graph.out_columns,
					this_->dev_fold_proc_buffer_->read_graph.out_indices,
					dev_buf_->edges,
					stream);
#if 1
#if VERVOSE_MODE
			CUDA_CHECK(cudaStreamSynchronize(stream));
			this_->read_graph_kernel_time_ += tk;
#endif
#endif
			cuda::read_graph_2(&dev_buf_->input,
					this_->dev_row_starts_,
					this_->dev_index_array_high_,
					this_->dev_index_array_low_,
					this_->dev_column_buffer_,
					column_start_, column_end_,
					this_->get_number_of_edge_columns(),
					this_->dev_shared_visited_,
					this_->dev_fold_proc_buffer_->read_graph.out_columns,
					this_->dev_fold_proc_buffer_->read_graph.out_indices,
					dev_buf_->edges,
					stream);

#if VERVOSE_MODE
			CUDA_CHECK(cudaThreadSynchronize());
			this_->filter_edge_kernel_time_ += tk;
			++this_->kernel_launch_read_graph_;
#endif
#if 0
			printf("L:%d:R(%d):end read_graph kernel\n", __LINE__, mpi.rank_2d);
#endif
		};
		virtual void receive(cudaStream_t stream) {
#if VERVOSE_MODE
			profiling::TimeKeeper tk;
#endif
			CUDA_CHECK(cudaMemcpyAsync(&host_buf_->read_graph.input, &dev_buf_->input,
					sizeof(host_buf_->read_graph.input), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK(cudaStreamSynchronize(stream));
#if VERVOSE_MODE
			this_->cuda_io_bytes_ +=  sizeof(host_buf_->read_graph.input);
#endif

			int64_t output_length = host_buf_->read_graph.input.output_offset;
			need_retry_ = host_buf_->read_graph.input.skipped;

			if(output_length > GPU_PARAMS::FILTER_EDGE_OUTBUF_SIZE) {
				fprintf(stderr, "Fatal error: The GPU output data size exceeds the buffer size!!!\n");
			}

			CUDA_CHECK(cudaMemcpyAsync(&host_buf_->read_graph.edges, &dev_buf_->edges,
					sizeof(host_buf_->read_graph.edges[0])*output_length, cudaMemcpyDeviceToHost, stream));
#if VERVOSE_MODE
			this_->cuda_io_bytes_ +=  sizeof(host_buf_->read_graph.edges[0])*output_length;
#endif

			// diagnostics
			if(need_retry_) {
				if(output_length == 0) {
					printf("Fatal Error: The length of output buffer"
							" for read graph kernel is too small for this graph.\n");
					sleep(1);
				}
				else {
#if VERVOSE_MODE
					//printf("Warning: Retrying read graph happens. This degrade performance a lot !\n");
					++this_->retry_count_read_graph_;
#endif
				}
			}
#if VERVOSE_MODE
			this_->extract_cuman_cpu_time_ += tk;
#endif
		};
		virtual void complete() {
			dev_buf_ = NULL;
			this_->fiber_man_.submit(this, 0);
		}
		virtual void run() {

#if VERVOSE_MODE
			profiling::TimeKeeper tk;
#endif

#pragma omp parallel
			process_send_edges();

#if VERVOSE_MODE
			this_->fold_compress_time_ += tk;
#endif

			if(need_retry_ == false) {
				volatile int* jobs_ptr = &this_->d_->num_remaining_extract_jobs_;
				if(__sync_add_and_fetch(jobs_ptr, -1) == 0) {
#if 0
		if(mpi.isMaster()) printf("L:%d:R(%d):end fold processing\n", __LINE__, mpi.rank_2d);
#endif
#pragma omp parallel for
					for(int i = 0; i < mpi.size_2dc; ++i) {
#if VERVOSE_MODE
						profiling::TimeKeeper tk;
#endif
						this_->sched_.fold_end_job[i].run();
#if VERVOSE_MODE
						this_->proc_fold_compress_time_ += tk;
#endif
					} // #pragma omp parallel for
					return ;
				}
			}

			// submit next job
			this_->cuda_man_->submit(this, 0);
		}
		void process_send_edges()
		{
#if VERVOSE_MODE
			profiling::TimeKeeper tk;
#endif
			// process sending
			typename super_::ThreadLocalBuffer* tl = this_->thread_local_buffer_[omp_get_thread_num()];
			bfs_detail::FoldPacket* packet_array = tl->fold_packet;
			const int log_local_verts = this_->graph_.log_local_verts();
			const int64_t v0_high_mask = int64_t(mpi.rank_2dc) << this_->graph_.log_local_v0();
			const int64_t local_verts_mask = this_->get_number_of_local_vertices() - 1;

			int64_t output_length = host_buf_->read_graph.input.output_offset;
			const long2* buffer = host_buf_->read_graph.edges;

#if DEBUG_PRINT
			printf("L:%d:R(%d):received from gpu. output_length=%"PRId64"\n", __LINE__, mpi.rank_2d, output_length);
#endif
#pragma omp for
			for(int i = 0; i < output_length; ++i) {
				int64_t v0_intermid = buffer[i].x;
				int64_t c1 = buffer[i].y;
				assert(v0_intermid >= 0);
#if DEBUG_PRINT
				if(v0_intermid >= (int64_t(1) << this_->graph_.log_local_v0())) {
					printf("L:%d:v0_intermid(%"PRId64") is out of range.i=%d\n", __LINE__, v0_intermid, i);
				}
#endif
				assert(v0_intermid < (int64_t(1) << this_->graph_.log_local_v0()));
				assert(c1 >= 0);
				assert(c1 < (int64_t(1) << this_->graph_.log_local_v1()));
#if 0
		if(c1 == 96366 && mpi.rank_2dr == 1) {
			printf("Debug Info: r:%d Found Sending c1==96366(1683)\n", mpi.rank_2d);
		}
#endif
				int64_t v1_local = c1 & local_verts_mask;
				int64_t dest_c = c1 >> log_local_verts;
				int64_t v0_swizzled = v0_high_mask | v0_intermid;

				bfs_detail::FoldPacket* packet = &packet_array[dest_c];
				packet->v0_list[packet->num_edges] = v0_swizzled;
				packet->v1_list[packet->num_edges] = v1_local;
				if(++packet->num_edges == BFS_PARAMS::PACKET_LENGTH) {
					this_->fold_send_packet(packet, dest_c);
					packet->num_edges = 0;
				}
			} // #pragma omp for
#if VERVOSE_MODE
			this_->proc_fold_compress_time_ += tk;
#endif
#if 0
			printf("L:%d:R(%d):process_send_edges complete.\n", __LINE__, mpi.rank_2d);
#endif
		}
		ThisType* this_;
		cuda::FoldIOBuffer* host_buf_;

		cuda::ReadGraphBuffer* dev_buf_;
		int64_t column_start_;
		int64_t column_end_;
		bool need_retry_;
		int num_blocks_;
	};

	struct CuExtractEdge2 : public CuExtractEdge
	{
		CuExtractEdge2(ThisType* this__, cuda::FoldIOBuffer* host_buf)
			: CuExtractEdge(this__, host_buf)
			, this_(this__)
			, read_phase_(true)
			, end_task_(false)
			, num_read_edges_(0)
			, i_progress(0)
			, ii_progress(0)
			, k_progress(0)
		{ }
		virtual ~CuExtractEdge2() { }
		virtual bool init() { return true; }
		virtual void send(cudaStream_t stream) {
			in_dev_buf_ = &this_->dev_fold_io_buffer_[(this_->fold_io_buf_count_++) % NUM_IO_BUFFERS].edge_io;
			CUDA_CHECK(cudaMemcpyAsync(in_dev_buf_->columns, this->host_buf_->edge_io.columns,
					sizeof(this->host_buf_->edge_io.columns[0])*num_read_edges_, cudaMemcpyHostToDevice, stream));
			CUDA_CHECK(cudaMemcpyAsync(in_dev_buf_->indices, this->host_buf_->edge_io.indices,
					sizeof(this->host_buf_->edge_io.indices[0])*num_read_edges_, cudaMemcpyHostToDevice, stream));
#if VERVOSE_MODE
			this_->cuda_io_bytes_ +=  sizeof(this->host_buf_->edge_io.columns[0])*num_read_edges_;
			this_->cuda_io_bytes_ +=  sizeof(this->host_buf_->edge_io.indices[0])*num_read_edges_;
#endif
		}
		virtual void launch_kernel(cudaStream_t stream) {
#if DEBUG_PRINT
			printf("L:%d:R(%d):launch filter edges kernel. number of edges=%"PRId64"\n", __LINE__, mpi.rank_2d, num_read_edges_);
#endif
			this->dev_buf_ = &this_->dev_fold_io_buffer_[(this_->fold_io_buf_count_++) % NUM_IO_BUFFERS].read_graph;
			cuda::filter_edges(&this->dev_buf_->input,
					num_read_edges_,
					in_dev_buf_->columns,
					in_dev_buf_->indices,
					this_->dev_shared_visited_,
					this->dev_buf_->edges,
					stream);

#if VERVOSE_MODE
#if 1
			CUDA_CHECK(cudaStreamSynchronize(stream));
#else
			CUDA_CHECK(cudaThreadSynchronize());
#endif
			++this_->kernel_launch_read_graph_;
#endif
			in_dev_buf_ = NULL;
		}
		virtual void run() {
			if(read_phase_) {
				read_phase_ = false;
				while(extract_edge_cpu()) {
#if DEBUG_PRINT
				printf("L:%d:R(%d):next extrace task\n", __LINE__, mpi.rank_2d);
#endif
					if(get_new_job() == false) {
#if DEBUG_PRINT
				printf("L:%d:R(%d):end extrace task\n", __LINE__, mpi.rank_2d);
#endif
						this->column_start_ = 0;
						this->column_end_ = 0;
						i_progress = 0;
						ii_progress = 0;
						k_progress = 0;
						end_task_ = true;
						// launch kernel
						this_->cuda_man_->submit(this, 0);
						return ;
					}
				}
			}
			else {
				read_phase_ = true;
				this->process_send_edges();
				num_read_edges_ = 0;
				if(end_task_) {
					end_task_ = false;
#if DEBUG_PRINT
				printf("L:%d:R(%d):shin! end extrace task\n", __LINE__, mpi.rank_2d);
#endif
					volatile int* jobs_ptr = &this_->d_->num_remaining_extract_jobs_;
					if(__sync_add_and_fetch(jobs_ptr, -1) == 0) {
						this_->fiber_man_.submit_array(this_->sched_.fold_end_job, mpi.size_2dc, 0);
					}
				}
				else {
					this_->fiber_man_.submit(this, 0);
				}
			}
		}
		// return false if yield occurs
		bool extract_edge_cpu()
		{
			using namespace BFS_PARAMS;
			BitmapType* cq_bitmap = this_->cq_bitmap_;
			BitmapType* cq_summary = this_->cq_summary_;
			const int64_t* row_starts = this_->graph_.row_starts_;
			const Pack48bit& index_array = this_->graph_.index_array_;
			int64_t i_end = this->column_end_;
			int num_read_edges = num_read_edges_;
			cuda::EdgeIOBuffer* io_buf = &this->host_buf_->edge_io;

#if DEBUG_PRINT
			if(i_end != 0) {
				printf("L:%d:R(%d):start extract_edge_cpu. i_start=%"PRId64", i_end=%"PRId64", ii_progress=%"PRId64", k_progress=%"PRId64"\n", __LINE__, mpi.rank_2d,
						i_progress, i_end, ii_progress, k_progress);
			}
#endif

			// TODO: optimize access to member valiables
			for(int64_t i = i_progress; i < i_end; ++i) {
				BitmapType summary_i = cq_summary[i];
				if(summary_i == 0) continue;
				for(int ii = ii_progress; ii < (int)sizeof(cq_summary[0])*8; ++ii) {
					if(summary_i & (BitmapType(1) << ii)) {
						int64_t cq_base_offset = ((int64_t)i*sizeof(cq_summary[0])*8 + ii)*super_::NUMBER_CQ_SUMMARIZING;
						for(int k = k_progress; k < super_::NUMBER_CQ_SUMMARIZING; ++k) {
							int64_t e0 = cq_base_offset + k;
							BitmapType bitmap_k = cq_bitmap[e0];
							if(bitmap_k == 0) continue;
							///////////////
							int64_t edge_list_length = row_starts[e0+1] - row_starts[e0];
							if(edge_list_length + num_read_edges > GPU_PARAMS::READ_GRAPH_OUTBUF_SIZE) {
								if(num_read_edges == 0) {
									printf("Fatal Error: The length of output buffer"
											" for input to kernel is too small for this graph.\n");
									sleep(1);
								}
								// launch kernel and yield loop
								i_progress = i;
								ii_progress = ii;
								k_progress = k;
								num_read_edges_ = num_read_edges;
#if DEBUG_PRINT
				printf("L:%d:R(%d):launch kernel. i_progress=%"PRId64", ii_progress=%"PRId64", k_progress=%"PRId64"\n", __LINE__, mpi.rank_2d,
						i_progress, ii_progress, k_progress);
#endif
								this_->cuda_man_->submit(this, 0);
								return false;
							}
							///////////////
							for(int64_t ri = row_starts[e0]; ri < row_starts[e0+1]; ++ri) {
								int8_t row_lowbits = (int8_t)(index_array.low_bits(ri) % super_::NUMBER_PACKING_EDGE_LISTS);
								if ((bitmap_k & (int64_t(1) << row_lowbits)) != 0){
									io_buf->columns[num_read_edges] = (int)e0;
									io_buf->indices[num_read_edges] = index_array(ri);
									++num_read_edges;
								}
							}
							// write zero after read
							cq_bitmap[e0] = 0;
						}
						k_progress = 0;
					}
				}
				// write zero after read
				cq_summary[i] = 0;
				ii_progress = 0;
			}
#if DEBUG_PRINT
				printf("L:%d:R(%d):end extract_edge_cpu. i_end=%"PRId64"\n", __LINE__, mpi.rank_2d, i_end);
#endif
			num_read_edges_ = num_read_edges;
			return true;
		}
		bool get_new_job()
		{
			int64_t summary_start =
					__sync_fetch_and_add(&this_->current_column_index_, this_->columns_per_task_);
			int64_t summary_size = this_->get_summary_size_v0();
			if(summary_start >= summary_size) {
				return false;
			}
			this->column_start_ = i_progress = summary_start;
			this->column_end_ = std::min<int64_t>(summary_size, this->column_start_ + this_->columns_per_task_);
			ii_progress = 0;
			k_progress = 0;
			return true;
		}

		ThisType* this_;
		cuda::EdgeIOBuffer* in_dev_buf_;
		bool read_phase_;
		bool end_task_;
		int64_t num_read_edges_;
		int64_t i_progress;
		int64_t ii_progress;
		int64_t k_progress;
	};

	struct CuReceiveProcessing : public CudaCommand
	{
		CuReceiveProcessing(ThisType* this__)
			: this_(this__)
		{
			CUDA_CHECK(cudaMallocHost((void**)&host_packet_list_, sizeof(int4)*GPU_PARAMS::GPU_BLOCK_MAX_PACKTES));
		}
		~CuReceiveProcessing()
		{
			CUDA_CHECK(cudaFreeHost(host_packet_list_));
		}
		void set(std::vector<bfs_detail::FoldCommBuffer*>& data, bool last_task)
		{
#if VERVOSE_MODE
			profiling::TimeKeeper tk;
#endif
			last_task_ = last_task;
			int packet_offset = 0;
			int v0_offset = 0;
			int v1_offset = 0;
			for(int i = 0; i < (int)data.size(); ++i) {
				RecvData rd;
				rd.data = data[i];
				rd.packet_offset = packet_offset;
				rd.v0_offset = v0_offset;
				rd.v1_offset = v1_offset;
				recv_data_.push_back(rd);

				packet_offset += data[i]->num_packets;
				v0_offset += data[i]->v0_stream_length;
				v1_offset += data[i]->num_edges;
			}
			RecvData rd;
			rd.data = NULL;
			rd.packet_offset = packet_offset;
			rd.v0_offset = v0_offset;
			rd.v1_offset = v1_offset;
			recv_data_.push_back(rd);
#if DEBUG_PRINT
			printf("L:%d:R(%d):ReceiveProcessing. v1_length=%d\n", __LINE__, mpi.rank_2d, v1_offset);
#endif

			num_packets_ = packet_offset;
			assert(num_packets_ < GPU_PARAMS::GPU_BLOCK_MAX_PACKTES);

			// create packet list
			int packet_i = 0;
			for(int i = 0; i < (int)recv_data_.size() - 1; ++i) {
				bfs_detail::FoldCommBuffer* data = recv_data_[i].data;
				int v0_offset = recv_data_[i].v0_offset;
				int v1_offset = recv_data_[i].v1_offset;
				const bfs_detail::PacketIndex* packet_index =
						&data->v0_stream->index[data->v0_stream->packet_index_start];

				for(int k = 0; k < data->num_packets; ++k) {
					assert (packet_index[k].length > 0);
					assert (packet_index[k].num_int > 0);
					int stream_length = packet_index[k].length;
					int num_edges = packet_index[k].num_int;
					host_packet_list_[packet_i++] =
							make_int4(v0_offset, num_edges, v1_offset, stream_length);
					v0_offset += stream_length;
					v1_offset += num_edges;
				}
				assert (recv_data_[i + 1].v0_offset - (int)sizeof(bfs_detail::PacketIndex) < v0_offset &&
						v0_offset <= recv_data_[i + 1].v0_offset);
				assert (recv_data_[i + 1].v1_offset == v1_offset);
			}
#if VERVOSE_MODE
			this_->recv_cuman_cpu_time_ += tk;
#endif
		}
		virtual bool init() { return true; }
		virtual void send(cudaStream_t stream) {
#if VERVOSE_MODE
			profiling::TimeKeeper tk;
#endif
			dev_buf_ = &this_->dev_fold_io_buffer_[(this_->fold_io_buf_count_++) % NUM_IO_BUFFERS].recv_proc;
			CUDA_CHECK(cudaMemcpyAsync(dev_buf_->packet_list, host_packet_list_,
					sizeof(int4)*num_packets_, cudaMemcpyHostToDevice, stream));
#if VERVOSE_MODE
			this_->cuda_io_bytes_ +=  sizeof(int4)*num_packets_;
#endif
			for(int i = 0; i < (int)recv_data_.size() - 1; ++i) {
				bfs_detail::FoldCommBuffer* data = recv_data_[i].data;
				CUDA_CHECK(cudaMemcpyAsync(dev_buf_->v0_stream + recv_data_[i].v0_offset,
						data->v0_stream->stream, data->v0_stream_length,
						cudaMemcpyHostToDevice, stream));
				CUDA_CHECK(cudaMemcpyAsync(dev_buf_->v1_list + recv_data_[i].v1_offset,
						data->v1_list, data->num_edges * sizeof(data->v1_list[0]),
						cudaMemcpyHostToDevice, stream));
#if VERVOSE_MODE
				this_->cuda_io_bytes_ +=  data->v0_stream_length;
				this_->cuda_io_bytes_ +=  data->num_edges * sizeof(data->v1_list[0]);
#endif
			}
#if VERVOSE_MODE
			this_->recv_cuman_cpu_time_ += tk;
#endif
		};
		virtual void launch_kernel(cudaStream_t stream) {
#if VERVOSE_MODE
			profiling::TimeKeeper tk;
#endif
#if 0
			printf("L:%d:R(%d):ReceiveProcessing. launch decode kernel.\n", __LINE__, mpi.rank_2d);
#endif
			int64_t* dev_v0_list = this_->dev_fold_proc_buffer_->recv_proc.v0_list;
			cuda::decode_varint_stream_signed(
					dev_buf_->packet_list,
					dev_buf_->v0_stream,
					num_packets_,
					dev_v0_list,
					stream);
#if DEBUG_PRINT
			// TODO: short cut if 0
			int64_t* host_v0_list;
			int num_edges = recv_data_[(int)recv_data_.size() - 1].v1_offset;
			if(num_edges > 0) {
				CUDA_CHECK(cudaMallocHost((void**)&host_v0_list, num_edges*sizeof(int64_t)));
				CUDA_CHECK(cudaThreadSynchronize());
				CUDA_CHECK(cudaMemcpy(host_v0_list, dev_v0_list, num_edges*sizeof(int64_t), cudaMemcpyDeviceToHost));

				for(int i = 0; i < num_edges; ++i) {
					int64_t v = host_v0_list[i];
					assert(v < (int64_t(1) << this_->graph_.log_global_verts()));
				}
				for(int i = 0; i < num_packets_; ++i) {
					int4& packet = host_packet_list_[i];
					int64_t* v_start = host_v0_list + packet.z;
					int64_t v = 0;
					for(int k = 0; k < packet.y; ++k) {
						v += v_start[k];
						assert(v >= 0);
						assert(v < (int64_t(1) << this_->graph_.log_global_verts()));
					}
				}

				CUDA_CHECK(cudaFreeHost(host_v0_list));
			}
			printf("L:%d:R(%d):ReceiveProcessing. decode complete.\n", __LINE__, mpi.rank_2d);
#endif
			cuda::receiver_processing(
					dev_buf_->packet_list,			// input_packets
					num_packets_,					// num_packets
					dev_v0_list,					// v0_list
					dev_buf_->v1_list,				// v1_list
					this_->dev_nq_bitmap_,			// nq_bitmap
					this_->dev_nq_sorted_bitmap_,	// nq_sorted_bitmap
					this_->dev_visited_,			// visited
					this_->pred_,					// pred
					this_->dev_invert_vertex_mapping_,	// v1_map
					this_->graph_.log_local_verts(),// log_local_verts
					get_msb_index(mpi.size_2d),		// log_size
					this_->get_number_of_local_vertices() - 1, // local_verts_mask
					this_->current_level_,			// current_level
					&this_->dev_->nq_count,			// nq_count_ptr
					stream);						// stream

#if 0
			printf("L:%d:R(%d):ReceiveProcessing. processing kernel complete.\n", __LINE__, mpi.rank_2d);
#endif
#if VERVOSE_MODE
			CUDA_CHECK(cudaThreadSynchronize());
			++this_->kernel_launch_receive_process_;
			this_->recv_proc_kernel_time_ += tk;
#endif
			dev_buf_ = NULL;
		}
		virtual void receive(cudaStream_t stream) {
			if(last_task_) {
#if VERVOSE_MODE
				profiling::TimeKeeper tk;
#endif
				// Receive NQ, NQ sorted, visited
				CUDA_CHECK(cudaMemcpyAsync(&this_->host_->nq_count, &this_->dev_->nq_count, sizeof(int), cudaMemcpyDeviceToHost, stream));
				int64_t transfer_size = sizeof(this_->nq_bitmap_[0])*this_->get_bitmap_size_visited();
				CUDA_CHECK(cudaMemcpyAsync(this_->nq_bitmap_, this_->dev_nq_bitmap_, transfer_size, cudaMemcpyDeviceToHost, stream));
				CUDA_CHECK(cudaMemcpyAsync(this_->nq_sorted_bitmap_, this_->dev_nq_sorted_bitmap_, transfer_size, cudaMemcpyDeviceToHost, stream));
				CUDA_CHECK(cudaMemcpyAsync(this_->visited_, this_->dev_visited_, transfer_size, cudaMemcpyDeviceToHost, stream));
#if VERVOSE_MODE
				this_->cuda_io_bytes_ +=  sizeof(int);
				this_->cuda_io_bytes_ +=  transfer_size * 3;
				this_->recv_cuman_cpu_time_ += tk;
#endif
			}
		};
		virtual void complete() {
#if VERVOSE_MODE
			profiling::TimeKeeper tk;
#endif
#if 0
			printf("L:%d:R(%d):ReceiveProcessing release fold buffer. last_task(%d)\n", __LINE__, mpi.rank_2d, last_task_);
#endif
			for(int i = 0; i < (int)recv_data_.size() - 1; ++i) {
				this_->comm_.relase_fold_buffer(recv_data_[i].data);
			}
			recv_data_.clear();
#if 0
			printf("L:%d:R(%d):ReceiveProcessing complete. last_task(%d)\n", __LINE__, mpi.rank_2d, last_task_);
#endif
			if(last_task_) {
				// signal !!!
				this_->fiber_man_.end_processing();

				cuda::cu_clear_nq(
					this_->get_bitmap_size_visited(),
					this_->dev_nq_bitmap_,
					this_->dev_nq_sorted_bitmap_,
					this_->cuda_stream_array_,
					NUM_CUDA_STREAMS);
			}
			this_->cu_recv_task_.push(this);
#if VERVOSE_MODE
			this_->recv_cuman_cpu_time_ += tk;
#endif
		};

		struct RecvData {
			bfs_detail::FoldCommBuffer* data;
			int packet_offset;
			int v0_offset;
			int v1_offset;
		};

		ThisType* this_;
		std::vector<RecvData> recv_data_;
		bool last_task_;
		int num_packets_;
		int4* host_packet_list_;
		cuda::RecvProcBuffer* dev_buf_;
	};

	int64_t get_number_of_edge_columns()
	{
		return int64_t(1) << (super_::graph_.log_local_v0() - super_::LOG_PACKING_EDGE_LISTS);
	}

	void allocate_memory_gpu()
	{
		using namespace BFS_PARAMS;
		using namespace GPU_PARAMS;

		const int max_threads = omp_get_max_threads();

		if(graph_on_gpu_) {
			num_stream_jobs_ = 8;
		}
		else {
			num_stream_jobs_ = max_threads + 4;
		}

		CudaStreamManager::begin_cuda();
		this->nq_bitmap_ = (BitmapType*)
				page_aligned_xcalloc(sizeof(this->nq_bitmap_[0])*this->get_bitmap_size_visited());
		CUDA_CHECK(cudaHostRegister(this->nq_bitmap_,
				sizeof(this->nq_bitmap_[0])*this->get_bitmap_size_visited(), 0));

		this->nq_sorted_bitmap_ = (BitmapType*)
				page_aligned_xcalloc(sizeof(this->nq_bitmap_[0])*this->get_bitmap_size_visited());
		CUDA_CHECK(cudaHostRegister(this->nq_sorted_bitmap_,
				sizeof(this->nq_bitmap_[0])*this->get_bitmap_size_visited(), 0));

		this->visited_ = (BitmapType*)
				page_aligned_xcalloc(sizeof(this->nq_bitmap_[0])*this->get_bitmap_size_visited());
		CUDA_CHECK(cudaHostRegister(this->visited_,
				sizeof(this->nq_bitmap_[0])*this->get_bitmap_size_visited(), 0));

		// Device Memory
		CUDA_CHECK(cudaMalloc((void**)&this->pred_,
				sizeof(int64_t) * this->get_actual_number_of_local_vertices()));
		if(graph_on_gpu_) {
			CUDA_CHECK(cudaMalloc((void**)&this->dev_cq_bitmap_,
					sizeof(this->cq_bitmap_[0])*this->get_bitmap_size_v0()));
			CUDA_CHECK(cudaMalloc((void**)&this->dev_cq_summary_,
					sizeof(this->cq_summary_[0])*this->get_summary_size_v0()));
		}
		else {
			this->cq_bitmap_ = (BitmapType*)
					page_aligned_xcalloc(sizeof(this->cq_bitmap_[0])*this->get_bitmap_size_v0());
			this->cq_summary_ = (BitmapType*)
					malloc(sizeof(this->cq_summary_[0])*this->get_summary_size_v0());
			this->dev_cq_bitmap_ = NULL;
			this->dev_cq_summary_ = NULL;
		}
		CUDA_CHECK(cudaMalloc((void**)&this->dev_shared_visited_ ,
				sizeof(this->shared_visited_[0])*this->get_bitmap_size_v1()));
		CUDA_CHECK(cudaMalloc((void**)&this->dev_nq_bitmap_,
				sizeof(this->nq_bitmap_[0])*this->get_bitmap_size_visited()));
		CUDA_CHECK(cudaMalloc((void**)&this->dev_nq_sorted_bitmap_,
				sizeof(this->nq_bitmap_[0])*this->get_bitmap_size_visited()));
		CUDA_CHECK(cudaMalloc((void**)&this->dev_visited_,
				sizeof(this->visited_[0])*this->get_bitmap_size_visited()));

		this->dev_cq_count_ = new PrefixSumGPU<int>(this->get_summary_size_v0());

		this->tmp_packet_max_length_ = sizeof(BitmapType) *
				this->get_bitmap_size_visited() / PACKET_LENGTH + omp_get_max_threads()*2;
		this->tmp_packet_index_ = (bfs_detail::PacketIndex*)
				malloc(sizeof(bfs_detail::PacketIndex)*this->tmp_packet_max_length_);

		this->gpu_cq_comm_.local_buffer_ = (bfs_detail::CompressedStream*)
				page_aligned_xcalloc(sizeof(BitmapType)*this->get_bitmap_size_visited());
		CUDA_CHECK(cudaHostRegister(this->gpu_cq_comm_.local_buffer_,
				sizeof(BitmapType)*this->get_bitmap_size_visited(), 0));
#if VERTEX_SORTING
		this->gpu_visited_comm_.local_buffer_ = (bfs_detail::CompressedStream*)
				page_aligned_xcalloc(sizeof(BitmapType)*this->get_bitmap_size_visited());
		CUDA_CHECK(cudaHostRegister(this->gpu_visited_comm_.local_buffer_,
				sizeof(BitmapType)*this->get_bitmap_size_visited(), 0));
#else
		this->gpu_visited_comm_.local_buffer_ = this->gpu_cq_comm_.local_buffer_;
#endif
		this->gpu_cq_comm_.recv_buffer_ = (typename super_::ExpandCommBuffer*)
				page_aligned_xcalloc(sizeof(BitmapType)*this->get_bitmap_size_v0());
		CUDA_CHECK(cudaHostRegister(this->gpu_cq_comm_.recv_buffer_,
				sizeof(BitmapType)*this->get_bitmap_size_v0(), 0));

		this->gpu_visited_comm_.recv_buffer_ = (typename super_::ExpandCommBuffer*)
				page_aligned_xcalloc(sizeof(BitmapType)*this->get_bitmap_size_v1());
		CUDA_CHECK(cudaHostRegister(this->gpu_visited_comm_.recv_buffer_,
				sizeof(BitmapType)*this->get_bitmap_size_v1(), 0));

		this->thread_local_buffer_ = (typename super_::ThreadLocalBuffer**)
				malloc(sizeof(this->thread_local_buffer_[0])*max_threads);
		this->d_ = (typename super_::DynamicDataSet*)malloc(sizeof(this->d_[0]));

		const int buffer_width = roundup<CACHE_LINE>(
				sizeof(typename super_::ThreadLocalBuffer) + sizeof(bfs_detail::FoldPacket) * mpi.size_2dc);
		this->buffer_.thread_local_ = cache_aligned_xcalloc(buffer_width*max_threads);
		for(int i = 0; i < max_threads; ++i) {
			this->thread_local_buffer_[i] = (typename super_::ThreadLocalBuffer*)
					((uint8_t*)this->buffer_.thread_local_ + buffer_width*i);
		}

		CUDA_CHECK(cudaMallocHost((void**)&this->host_, sizeof(this->host_[0])));
		CUDA_CHECK(cudaMalloc((void**)&this->dev_, sizeof(this->dev_[0])));

		dev_buffer_size_ = std::max<int64_t>(
				sizeof(this->dev_column_buffer_[0]) * this->get_bitmap_size_v0() +
				sizeof(cuda::FoldIOBuffer) * NUM_IO_BUFFERS + sizeof(cuda::FoldGpuBuffer),
				sizeof(cuda::UpdateProcBuffer) * NUM_IO_BUFFERS);
		CUDA_CHECK(cudaMalloc((void**)&this->dev_buffer_, dev_buffer_size_));

		this->dev_fold_io_buffer_ = (cuda::FoldIOBuffer*)this->dev_buffer_;
		this->dev_fold_proc_buffer_ = (cuda::FoldGpuBuffer*)&this->dev_fold_io_buffer_[NUM_IO_BUFFERS];
		this->dev_column_buffer_ = (uint2*)&this->dev_fold_proc_buffer_[1];
		this->dev_expand_buffer_ = (cuda::UpdateProcBuffer*)this->dev_buffer_;

		// for receive expand data
		if(graph_on_gpu_) {
			this->cq_bitmap_ = this->gpu_cq_comm_.recv_buffer_->bitmap;
		}
		this->shared_visited_ = this->gpu_visited_comm_.recv_buffer_->bitmap;

		CUDA_CHECK(cudaMallocHost((void**)&this->host_read_graph_buffer_,
				sizeof(this->host_read_graph_buffer_[0])*num_stream_jobs_));

		//int num_extract_jobs = get_blocks<EXTRACT_BLOCKS_PER_LAUNCH>(get_bitmap_size_v0());
		int64_t num_columns = this->get_number_of_edge_columns();
		int64_t index_size = this->graph_.row_starts_[num_columns];
		double elements_per_column = (double)index_size / num_columns;
		this->blocks_per_launch_ = (int)std::max((GPU_PARAMS::BLOCKS_PER_LAUNCH_RATE / elements_per_column), 1.0);

		for(int i = 0; i < num_stream_jobs_; ++i) {
			CuExtractEdge* job;
			if(graph_on_gpu_)
				job = new CuExtractEdge(this, &this->host_read_graph_buffer_[i]);
			else
				job = new CuExtractEdge2(this, &this->host_read_graph_buffer_[i]);
			this->extract_edge_array_[i] = job;
		}

		for(int i = 0; i < NUM_CUDA_STREAMS; ++i) {
			CUDA_CHECK(cudaStreamCreate(&this->cuda_stream_array_[i]));
		}

		this->sched_.fold_end_job = new typename super_::ExtractEnd[mpi.size_2dc];
		for(int i = 0; i < mpi.size_2dc; ++i) {
			this->sched_.fold_end_job[i].this_ = this;
			this->sched_.fold_end_job[i].dest_c_ = i;
		}

		this->num_recv_tasks_ = std::max(8, mpi.size_2d);
	//	this->num_recv_tasks_ = std::max(1, mpi.size_2d);
		for(int i = 0; i < this->num_recv_tasks_; ++i) {
			this->cu_recv_task_.push(new CuReceiveProcessing(this));
		}
		CudaStreamManager::end_cuda();
	}

	void deallocate_memory_gpu()
	{
		CudaStreamManager::begin_cuda();
		CUDA_CHECK(cudaHostUnregister(this->nq_bitmap_));
		free(this->nq_bitmap_); this->nq_bitmap_ = NULL;

		CUDA_CHECK(cudaHostUnregister(this->nq_sorted_bitmap_));
		free(this->nq_sorted_bitmap_); this->nq_sorted_bitmap_ = NULL;

		CUDA_CHECK(cudaHostUnregister(this->visited_));
		free(this->visited_); this->visited_ = NULL;

		// Device Memory
		CUDA_CHECK(cudaFree(this->pred_)); this->pred_ = NULL;
		if(graph_on_gpu_) {
			CUDA_CHECK(cudaFree(this->dev_cq_bitmap_)); this->dev_cq_bitmap_ = NULL;
			CUDA_CHECK(cudaFree(this->dev_cq_summary_)); this->dev_cq_summary_ = NULL;
		}
		else {
			free(this->cq_bitmap_); this->cq_bitmap_ = NULL;
			free(this->cq_summary_); this->cq_summary_ = NULL;
		}
		CUDA_CHECK(cudaFree(this->dev_shared_visited_)); this->dev_shared_visited_ = NULL;
		CUDA_CHECK(cudaFree(this->dev_nq_bitmap_)); this->dev_nq_bitmap_ = NULL;
		CUDA_CHECK(cudaFree(this->dev_nq_sorted_bitmap_)); this->dev_nq_sorted_bitmap_ = NULL;
		CUDA_CHECK(cudaFree(this->dev_visited_)); this->dev_visited_ = NULL;

		delete this->dev_cq_count_; this->dev_cq_count_ = NULL;

		free(this->tmp_packet_index_); this->tmp_packet_index_ = NULL;

		CUDA_CHECK(cudaHostUnregister(this->gpu_cq_comm_.local_buffer_));
		free(this->gpu_cq_comm_.local_buffer_); this->gpu_cq_comm_.local_buffer_ = NULL;
#if VERTEX_SORTING
		CUDA_CHECK(cudaHostUnregister(this->gpu_visited_comm_.local_buffer_));
		free(this->gpu_visited_comm_.local_buffer_); this->gpu_visited_comm_.local_buffer_ = NULL;
#endif
		CUDA_CHECK(cudaHostUnregister(this->gpu_cq_comm_.recv_buffer_));
		free(this->gpu_cq_comm_.recv_buffer_); this->gpu_cq_comm_.recv_buffer_ = NULL;

		CUDA_CHECK(cudaHostUnregister(this->gpu_visited_comm_.recv_buffer_));
		free(this->gpu_visited_comm_.recv_buffer_); this->gpu_visited_comm_.recv_buffer_ = NULL;

		free(this->thread_local_buffer_); this->thread_local_buffer_ = NULL;
		free(this->d_); this->d_ = NULL;
		free(this->buffer_.thread_local_); this->buffer_.thread_local_ = NULL;

		CUDA_CHECK(cudaFreeHost(this->host_)); this->host_ = NULL;
		CUDA_CHECK(cudaFree(this->dev_)); this->dev_ = NULL;

		CUDA_CHECK(cudaFree(this->dev_buffer_)); this->dev_buffer_ = NULL;
		this->dev_fold_io_buffer_ = NULL;
		this->dev_fold_proc_buffer_ = NULL;
		this->dev_column_buffer_ = NULL;
		this->dev_expand_buffer_ = NULL;

		CUDA_CHECK(cudaFreeHost(this->host_read_graph_buffer_)); this->host_read_graph_buffer_ = NULL;

		for(int i = 0; i < NUM_CUDA_STREAMS; ++i) {
			CUDA_CHECK(cudaStreamDestroy(this->cuda_stream_array_[i]));
		}

		delete [] this->sched_.fold_end_job; this->sched_.fold_end_job = NULL;

		for(int i = 0; i < this->num_recv_tasks_; ++i) {
			delete this->cu_recv_task_.pop();
		}
		CudaStreamManager::end_cuda();
	}

	void initialize_memory_gpu()
	{
		using namespace BFS_PARAMS;

		// initialzie memory
		CudaStreamManager::begin_cuda();
		cuda::cu_initialize_memory(
			this->get_actual_number_of_local_vertices(),
			this->get_bitmap_size_visited(),
			this->get_bitmap_size_v0(),
			this->get_bitmap_size_v1(),
			this->get_summary_size_v0(),
			this->pred_,
			this->dev_nq_bitmap_,
			this->dev_nq_sorted_bitmap_,
			this->dev_visited_,
			this->dev_cq_summary_,
			this->dev_cq_bitmap_,
			this->dev_shared_visited_,
			this->cuda_stream_array_,
			NUM_CUDA_STREAMS);
		CudaStreamManager::end_cuda();

#pragma omp parallel
		{
			const int64_t summary_size = this->get_summary_size_v0();
			BitmapType* cq_bitmap = this->cq_bitmap_;
			BitmapType* cq_summary = this->cq_summary_;

			if(graph_on_gpu_ == false) {
				// clear CQ and CQ summary
#pragma omp for nowait
				for(int64_t i = 0; i < summary_size; ++i) {
					cq_summary[i] = 0;
					for(int k = 0; k < super_::MINIMUN_SIZE_OF_CQ_BITMAP; ++k) {
						cq_bitmap[i*super_::MINIMUN_SIZE_OF_CQ_BITMAP + k] = 0;
					}
				}
			}

			// clear fold packet buffer
			bfs_detail::FoldPacket* packet_array = this->thread_local_buffer_[omp_get_thread_num()]->fold_packet;
			for(int i = 0; i < mpi.size_2dc; ++i) {
				packet_array[i].num_edges = 0;
			}
		}
	}

	void prepare_fold()
	{
		current_column_index_ = 0;
		recv_st_.num_packets = 0;
		recv_st_.v0_length = 0;
		recv_st_.v1_length = 0;

		CudaStreamManager::begin_cuda();

#if VERVOSE_MODE
			profiling::TimeKeeper tk;
#endif

		cudaStream_t master_stream = cuda_stream_array_[0];
		// per level initialization.
		host_->nq_count = 0;
		CUDA_CHECK(cudaMemcpyAsync(&dev_->nq_count, &host_->nq_count, sizeof(host_->nq_count),
				cudaMemcpyHostToDevice, master_stream));

		if(graph_on_gpu_) {
			cuda::create_cq_list(dev_cq_bitmap_, dev_cq_summary_,
								this->get_summary_size_v0(), dev_cq_count_,
								dev_column_buffer_, this->get_number_of_edge_columns(),
								master_stream);
			int* dev_last_offset = &dev_cq_count_->get_buffer()[this->get_summary_size_v0()];
			CUDA_CHECK(cudaMemcpyAsync(&host_->num_non_zero_columns, dev_last_offset,
					sizeof(dev_last_offset[0]), cudaMemcpyDeviceToHost, master_stream));
#if VERVOSE_MODE
				cuda_io_bytes_ +=  sizeof(dev_last_offset[0]);
#endif
		}

		CUDA_CHECK(cudaStreamSynchronize(master_stream));

		CudaStreamManager::end_cuda();

		// begin receiver processing before starting fold main processing. It is OK, I think.
		this->comm_.begin_fold_comm();

		if(graph_on_gpu_) {
#if DEBUG_PRINT
				if(mpi.isMaster()) printf("num_non_zero_columns=%d\n", this->host_->num_non_zero_columns);
#endif
		}

#if VERVOSE_MODE
		++kernel_launch_create_cq_list_;
		create_cq_kernel_time_ += tk;
#endif
	}

	void submit_recv_proc(bool last_task)
	{
		CuReceiveProcessing* proc = cu_recv_task_.pop();
		proc->set(recv_data_, last_task);
		recv_data_.clear();
		recv_st_.num_packets = 0;
		recv_st_.v0_length = 0;
		recv_st_.v1_length = 0;
		cuda_man_->submit(proc, 1);
	}

	// impelementation of virtual functions
	virtual int varint_encode(const int64_t* input, int length, uint8_t* output, bfs_detail::VARINT_BFS_KIND kind)
	{
		if(graph_on_gpu_ == false && kind == bfs_detail::VARINT_EXPAND_CQ) {
			return varint_encode_stream((const uint64_t*)input, length, output);
		}
		return varint_encode_stream_gpu_compat_signed(input, length, output);
	}

	virtual void fold_received(bfs_detail::FoldCommBuffer* data)
	{
		recv_st_.v0_length += data->v0_stream_length;
		recv_st_.v1_length += data->num_edges;
		recv_st_.num_packets += data->num_packets;
		recv_data_.push_back(data);
		if(recv_st_.v0_length >= GPU_PARAMS::GPU_BLOCK_V0_THRESHOLD ||
			recv_st_.v1_length >= GPU_PARAMS::GPU_BLOCK_V1_THRESHOLD ||
			recv_st_.num_packets >= GPU_PARAMS::GPU_BLOCK_MAX_PACKTES)
		{
			submit_recv_proc(false);
		}
	}

	virtual void fold_finish()
	{
#if 0
		if(mpi.isMaster()) printf("L:%d:R(%d):fold_finish\n", __LINE__, mpi.rank_2d);
#endif
		submit_recv_proc(true);
	}

	struct GpuExpandCommCommand : public super_::ExpandCommCommand
	{
		GpuExpandCommCommand(ThisType* this__, bool cq_or_visited)
			: super_::ExpandCommCommand(this__, cq_or_visited)
			, this_(this__)
		{
			//
		}
		virtual void comm() {

			this->gather_comm();

			this_->fiber_man_.submit(this, 0);
		}
		void launch_update_kernel(const int4* index, LocalVertsIndex num_packet,
				const uint8_t* v_stream, LocalVertsIndex stream_length, uint32_t* target, uint32_t* summary)
		{
			// launch kernel
			int stream_number = (this_->expand_job_id_++) % std::min<int>(2, NUM_CUDA_STREAMS);
			int4* dev_packet_list = this_->dev_expand_buffer_[stream_number].packet_list;
			uint8_t* dev_v_stream = this_->dev_expand_buffer_[stream_number].v_stream;
			uint32_t* dev_v_list = this_->dev_expand_buffer_[stream_number].v_list;
			cudaStream_t cuda_stream = this_->cuda_stream_array_[stream_number];
			int log_src_factor = this_->log_local_bitmap_ + super_::LOG_PACKING_EDGE_LISTS;
#if 0
		if(mpi.isMaster()) printf("L:%d:update bitmap kernel launch: stream_number=%d,num_packet=%d\n", __LINE__, stream_number, num_packet);
#endif
			CUDA_CHECK(cudaMemcpyAsync(dev_packet_list, index,
					sizeof(index[0])*num_packet, cudaMemcpyHostToDevice, cuda_stream));
#if 0
		if(mpi.isMaster()) printf("L:%d:cudaMemcpyAsync(%p,%p,%d,HtoD,%p)\n", __LINE__, dev_v_stream, v_stream,
				(int)stream_length, cuda_stream);
#endif
			CUDA_CHECK(cudaMemcpyAsync(dev_v_stream, v_stream,
					stream_length, cudaMemcpyHostToDevice, cuda_stream));
#if VERVOSE_MODE
				this_->cuda_io_bytes_ +=  sizeof(index[0])*num_packet;
				this_->cuda_io_bytes_ +=  stream_length;
#endif
			// TODO: We don't need signed coding.
			// 		 To process with unsigned coding, modify the processing of encoding.
			cuda::decode_varint_stream_signed(dev_packet_list,
					dev_v_stream, num_packet, (int32_t*)dev_v_list, cuda_stream);
#if DEBUG_PRINT
			int32_t* host_v0_list;
			int num_edges = index[num_packet - 1].y + index[num_packet - 1].z;
			if(num_edges > 0) {
				CUDA_CHECK(cudaMallocHost((void**)&host_v0_list, num_edges*sizeof(int64_t)));
				CUDA_CHECK(cudaThreadSynchronize());
				CUDA_CHECK(cudaMemcpy(host_v0_list, dev_v_list, num_edges*sizeof(int64_t), cudaMemcpyDeviceToHost));

				for(int i = 0; i < num_edges; ++i) {
					int32_t v = host_v0_list[i];
					if(!(v < (1 << this_->graph_.log_local_verts()))) {
						printf("Error:");
						for(int i = 0; i < num_edges; ++i) {
							printf("list[%d]=(%d)", i, host_v0_list[i]);
						}
						printf("\n");
					}
					assert(v < (1 << this_->graph_.log_local_verts()));
				}
				for(int i = 0; i < (int)num_packet; ++i) {
					const int4& packet = index[i];
					int32_t* v_start = host_v0_list + packet.z;
					int32_t v = 0;
					for(int k = 0; k < packet.y; ++k) {
						v += v_start[k];
						assert(v >= 0);
						assert(v < (1 << this_->graph_.log_local_verts()));
					}
				}

				CUDA_CHECK(cudaFreeHost(host_v0_list));
			}
#endif
			cuda::update_bitmap(dev_packet_list, num_packet, dev_v_list,
					target, summary, log_src_factor, cuda_stream);
#if VERVOSE_MODE
			++this_->kernel_launch_update_bitmap_;
#endif
		}
		void gpu_update_from_stream(const uint8_t* byte_stream, int* offset, int* count,
				int num_src, uint32_t* target, uint32_t* summary)
		{
#if VERVOSE_MODE
			profiling::TimeKeeper tk_all;
			profiling::TimeKeeper tk_kernel;
			profiling::TimeSpan sp_kernel;
#endif
			LocalVertsIndex packet_count[num_src], packet_offset[num_src+1];
			packet_offset[0] = 0;

			// compute number of packets
			for(int i = 0; i < num_src; ++i) {
				int index_start = ((bfs_detail::CompressedStream*)(byte_stream + offset[i]))->packet_index_start;
				packet_count[i] = (count[i] - index_start * sizeof(bfs_detail::PacketIndex) -
						offsetof(bfs_detail::CompressedStream, index)) / sizeof(bfs_detail::PacketIndex);
				packet_offset[i+1] = packet_offset[i] + packet_count[i];
			}

			// make a list of all packets
			int64_t num_total_packets = packet_offset[num_src];
#if DEBUG_PRINT
		if(mpi.isMaster()) printf("L:%d:num_total_packets=%"PRId64"\n", __LINE__, num_total_packets);
#endif
			if(num_total_packets == 0) {
				// nothing to do
#if VERVOSE_MODE
				tk_all.submit("expand recv cpu", this_->current_level_, this->cq_or_visited_ ? 0 : 1);
#endif
				return ;
			}

			//cuda_memcheck(false);
			// (stream_offset, num_vertices, src_num, stream_length)
			int4* index;
#if VERVOSE_MODE
			tk_kernel.getSpanAndReset();
#endif
			CUDA_CHECK(cudaMallocHost((void**)&index, sizeof(index[0])*num_total_packets));
#if VERVOSE_MODE
			sp_kernel += tk_kernel;
#endif
			LocalVertsIndex total_stream_length = offset[num_src];

			//cuda_memcheck(false);
			int64_t base_i = 0;
			int64_t base_stream_offset = 0;
			int64_t v_offset = 0;
			for(int i = 0; i < num_src; ++i) {
				bfs_detail::CompressedStream* stream = (bfs_detail::CompressedStream*)(byte_stream + offset[i]);
				int64_t base = packet_offset[i];
				bfs_detail::PacketIndex* packet_index = &stream->index[stream->packet_index_start];
				int64_t stream_offset = offset[i] + offsetof(bfs_detail::CompressedStream, stream) - base_stream_offset;

				for(int64_t k = 0; k < packet_count[i]; ++k) {
					const int num_int = packet_index[k].num_int;
					const int length = packet_index[k].length;
					const int num_packet = (int)(base + k - base_i);

					if(num_packet == GPU_PARAMS::EXPAND_PACKET_LIST_LENGTH ||
						v_offset + num_int > GPU_PARAMS::EXPAND_DECODE_BLOCK_LENGTH ||
						stream_offset + length > GPU_PARAMS::EXPAND_STREAM_BLOCK_LENGTH)
					{
#if VERVOSE_MODE
						tk_kernel.getSpanAndReset();
#endif
						launch_update_kernel(index + base_i, num_packet,
								byte_stream + base_stream_offset, stream_offset, target, summary);
#if VERVOSE_MODE
						sp_kernel += tk_kernel;
#endif

						base_i = base + k;
						base_stream_offset += stream_offset;
						stream_offset = 0;
						v_offset = 0;
					}

					index[base + k] = make_int4(stream_offset, packet_index[k].num_int, v_offset, i);

					v_offset += num_int;
					stream_offset += length;
				}
				assert((roundup<sizeof(bfs_detail::PacketIndex)>(
						(stream_offset + base_stream_offset - offset[i] - offsetof(bfs_detail::CompressedStream, stream)))
						/ sizeof(bfs_detail::PacketIndex)) == stream->packet_index_start);
			}
			//cuda_memcheck(false);

			LocalVertsIndex stream_offset = total_stream_length - base_stream_offset;
			LocalVertsIndex num_packet = num_total_packets - base_i;
			if(num_packet > 0) {
#if VERVOSE_MODE
				tk_kernel.getSpanAndReset();
#endif
				launch_update_kernel(index + base_i, num_packet,
						byte_stream + base_stream_offset, stream_offset, target, summary);
#if VERVOSE_MODE
				sp_kernel += tk_kernel;
#endif
			}

#if VERVOSE_MODE
				tk_kernel.getSpanAndReset();
#endif
			//cuda_memcheck(false);

			CUDA_CHECK(cudaThreadSynchronize());
#if VERVOSE_MODE
			sp_kernel += tk_kernel;
			profiling::TimeSpan sp_cpu;
			sp_cpu += tk_all;
			sp_cpu -= sp_kernel;
			this_->update_bitmap_kernel_time_ += sp_kernel;
			sp_cpu.submit("expand recv cpu", this_->current_level_, this->cq_or_visited_ ? 0 : 1);
#endif
			//cuda_memcheck(false);
		}
		virtual void run() {
#if VERVOSE_MODE
			profiling::TimeKeeper tk_kernel;
			profiling::TimeSpan sp_kernel;
#endif
			CudaStreamManager::begin_cuda();
			if(this->cq_or_visited_) {
				// current queue
				if(graph_on_gpu_) {
					if(this->stream_or_bitmap_) {
#if VERVOSE_MODE
						sp_kernel += tk_kernel;
#endif
						// stream update
						gpu_update_from_stream(
								(uint8_t*)this->recv_buffer_, this->offset_, this->count_,
								this->comm_size_, this_->dev_cq_bitmap_, this_->dev_cq_summary_);
#if VERVOSE_MODE
						tk_kernel.getSpanAndReset();
#endif
					}
					else {
						int64_t bitmap_size = this_->get_bitmap_size_v0();
						CUDA_CHECK(cudaMemcpy(this_->dev_cq_bitmap_, this->recv_buffer_->bitmap,
								sizeof(BitmapType)*bitmap_size, cudaMemcpyHostToDevice));
						// fill 1 to summary
						cuda::fill_1_summary(this_->dev_cq_summary_, this_->get_summary_size_v0());
#if VERVOSE_MODE
						this_->cuda_io_bytes_ +=  sizeof(BitmapType)*bitmap_size;
#endif
					}
				}
				else {
					// update CQ (on CPU)
					if(this->stream_or_bitmap_) {
						// stream
						const LocalVertsIndex bitmap_size = this_->get_bitmap_size_v0();
						BitmapType* cq_bitmap = this_->cq_bitmap_;
						// update
						this_->d_->num_vertices_in_cq_ = this_->update_from_stream(
								&this->recv_buffer_->stream, this->offset_, this->count_,
								this->comm_size_, cq_bitmap, bitmap_size, this_->cq_summary_);
					}
					else {
						// bitmap
						this_->d_->num_vertices_in_cq_ = this_->get_bitmap_size_v0() * super_::NUMBER_PACKING_EDGE_LISTS;
						// fill 1 to summary
						memset(this_->cq_summary_, -1,
								sizeof(BitmapType) * this_->get_summary_size_v0());
					}
				}
			}
			else {
				//
				if(this->stream_or_bitmap_) {
#if VERVOSE_MODE
						sp_kernel += tk_kernel;
#endif
					// stream update
					gpu_update_from_stream(
							(uint8_t*)this->recv_buffer_, this->offset_, this->count_,
							this->comm_size_, this_->dev_shared_visited_, NULL);
#if VERVOSE_MODE
						tk_kernel.getSpanAndReset();
#endif
				}
				else {
					int64_t bitmap_size = this_->get_bitmap_size_v1();
					CUDA_CHECK(cudaMemcpy(this_->dev_shared_visited_, this->recv_buffer_->bitmap,
							sizeof(BitmapType)*bitmap_size, cudaMemcpyHostToDevice));
#if VERVOSE_MODE
					this_->cuda_io_bytes_ += sizeof(BitmapType)*bitmap_size;
#endif
				}
			}
			CudaStreamManager::end_cuda();

			if(this->exit_fiber_proc_) {
				this_->fiber_man_.end_processing();
			}
#if VERVOSE_MODE
			sp_kernel += tk_kernel;
			this_->update_bitmap_kernel_time_ += sp_kernel;
			this_->update_bitmap_kernel_time_.submit("expand kernel", this_->current_level_, this->cq_or_visited_ ? 0 : 1);
#endif
		}
		ThisType* this_;
	};

	// members g_GpuIndex
	CudaStreamManager* cuda_man_;

	GpuExpandCommCommand gpu_cq_comm_;
	GpuExpandCommCommand gpu_visited_comm_;

	ConcurrentStack<CuReceiveProcessing*> cu_recv_task_;

	// device graph
	int64_t* dev_row_starts_;
	int32_t* dev_index_array_high_;
	uint16_t* dev_index_array_low_;
	uint32_t* dev_invert_vertex_mapping_;

	// device bitmap
	BitmapType* dev_cq_bitmap_;
	BitmapType* dev_cq_summary_; // 128bytes -> 1bit
	BitmapType* dev_shared_visited_;
	BitmapType* dev_visited_;
	BitmapType* dev_nq_bitmap_;
	BitmapType* dev_nq_sorted_bitmap_;

	cuda::BfsGPUContext* host_;
	cuda::BfsGPUContext* dev_;

	// device buffer
	PrefixSumGPU<int>* dev_cq_count_; // length: summary_size
	int64_t dev_buffer_size_;
	void* dev_buffer_;

	// pointer to device buffer
	cuda::FoldIOBuffer* dev_fold_io_buffer_; // device transfer buffer
	cuda::FoldGpuBuffer* dev_fold_proc_buffer_; // device temporal buffer
	cuda::FoldIOBuffer* host_read_graph_buffer_; // host processing buffer
	cuda::UpdateProcBuffer* dev_expand_buffer_;
	uint2* dev_column_buffer_;

	int num_stream_jobs_;
	int num_recv_tasks_;
	int fold_io_buf_count_;
	int expand_job_id_;
	int64_t current_column_index_;
	int blocks_per_launch_; // for read from GPU
	int columns_per_task_; // for read on CPU
	CuCreateCQList create_cq_list_;
	CuExtractEdge *extract_edge_array_[MAX_NUM_STREAM_JOBS];

	cudaStream_t cuda_stream_array_[NUM_CUDA_STREAMS];

	std::vector<bfs_detail::FoldCommBuffer*> recv_data_;
	struct {
		int v0_length;
		int v1_length;
		int num_packets;
	} recv_st_;

#if VERVOSE_MODE
	int kernel_launch_create_cq_list_;
	int kernel_launch_read_graph_; int retry_count_read_graph_;
	int kernel_launch_receive_process_;
	int kernel_launch_update_bitmap_;
	int64_t cuda_io_bytes_;
	profiling::TimeSpan extract_cuman_cpu_time_;
	profiling::TimeSpan fold_compress_time_;
	profiling::TimeSpan proc_fold_compress_time_;
	profiling::TimeSpan recv_cuman_cpu_time_;
	profiling::TimeSpan create_cq_kernel_time_;
	profiling::TimeSpan read_graph_kernel_time_;
	profiling::TimeSpan filter_edge_kernel_time_;
	profiling::TimeSpan recv_proc_kernel_time_;
	profiling::TimeSpan update_bitmap_kernel_time_;
#endif
};

#endif /* BFS_GPU_HPP_ */
