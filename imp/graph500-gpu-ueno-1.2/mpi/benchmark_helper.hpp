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
/*
 * Portions are:
 *
 * Copyright (C) 2009-2010 The Trustees of Indiana University.
 *
 * Use, modification and distribution is subject to the Boost Software
 * License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 *  Authors: Jeremiah Willcock
 *           Andrew Lumsdaine
 */
#ifndef BENCHMARK_HELPER_HPP_
#define BENCHMARK_HELPER_HPP_

class ProgressReport
{
public:
	ProgressReport(int max_progress)
		: max_progress_(max_progress)
		, my_progress_(0)
		, send_req_(new MPI_Request[max_progress]())
		, recv_req_(NULL)
		, send_buf_(new int[max_progress]())
		, recv_buf_(NULL)
		, g_progress_(NULL)
	{
		for(int i = 0; i < max_progress; ++i) {
			send_req_[i] = MPI_REQUEST_NULL;
			send_buf_[i] = i + 1;
		}
		pthread_mutex_init(&thread_sync_, NULL);
		pthread_cond_init(&thread_state_,  NULL);
		if(mpi.isMaster()) {
			recv_req_ = new MPI_Request[mpi.size_2d]();
			recv_buf_  = new int[mpi.size_2d]();
			g_progress_ = new int[mpi.size_2d]();
			for(int i = 0; i < mpi.size_2d; ++i) {
				recv_req_[i] = MPI_REQUEST_NULL;
			}
		}
	}
	~ProgressReport() {
		pthread_mutex_destroy(&thread_sync_);
		pthread_cond_destroy(&thread_state_);
		delete [] send_req_; send_req_ = NULL;
		delete [] recv_req_; recv_req_ = NULL;
		delete [] send_buf_; send_buf_ = NULL;
		delete [] recv_buf_; recv_buf_ = NULL;
		delete [] g_progress_; g_progress_ = NULL;
	}
	void begin_progress() {
		my_progress_ = 0;
		if(mpi.isMaster()) {
			pthread_create(&thread_, NULL, update_status_thread, this);
			printf("Begin Reporting Progress. Info: Rank is 1D MPI rank, not 2D.\n");
		}
	}
	void advace() {
		pthread_mutex_lock(&thread_sync_);
		MPI_Isend(&send_buf_[my_progress_], 1, MPI_INT, 0, 0, mpi.comm_2d, &send_req_[my_progress_]);
		int index, flag;
		MPI_Testany(max_progress_, send_req_, &index, &flag, MPI_STATUS_IGNORE);
		pthread_mutex_unlock(&thread_sync_);
		++my_progress_;
	}
	void end_progress() {
		if(mpi.isMaster()) {
			pthread_join(thread_, NULL);
		}
		MPI_Waitall(max_progress_, send_req_, MPI_STATUSES_IGNORE);
	}

private:
	static void* update_status_thread(void* this_) {
		static_cast<ProgressReport*>(this_)->update_status();
		return NULL;
	}
	// return : complete or not
	void update_status() {
		for(int i = 0; i < mpi.size_2d; ++i) {
			g_progress_[i] = 0;
			recv_buf_[i] = 0; // ?????
			MPI_Irecv(&recv_buf_[i], 1, MPI_INT, i, 0, mpi.comm_2d, &recv_req_[i]);
		}
		int* tmp_progress = new int[mpi.size_2d];
		int* node_list = new int[mpi.size_2d];
		bool complete = false;
		double print_time = MPI_Wtime();
		while(complete == false) {
			usleep(50*1000); // sleep 50 ms
			if(MPI_Wtime() - print_time >= 2.0) {
				print_time = MPI_Wtime();
				for(int i = 0; i < mpi.size_2d; ++i) {
					tmp_progress[i] = g_progress_[i];
					node_list[i] = i;
				}
				sort2(tmp_progress, node_list, mpi.size_2d);
				printf("(Rank,Iter)=");
				for(int i = 0; i < std::min(mpi.size_2d, 8); ++i) {
					printf("(%d,%d)", node_list[i], tmp_progress[i]);
				}
				printf("\n");
			}
			pthread_mutex_lock(&thread_sync_);
			while(true) {
				int index, flag;
				MPI_Testany(mpi.size_2d, recv_req_, &index, &flag, MPI_STATUS_IGNORE);
				if(flag == 0) break;
				if(index == MPI_UNDEFINED) {
					complete = true;
					break;
				}
				g_progress_[index] = recv_buf_[index];
				if(g_progress_[index] < max_progress_) {
					MPI_Irecv(&recv_buf_[index], 1, MPI_INT, index, 0, mpi.comm_2d, &recv_req_[index]);
				}
			}
			pthread_mutex_unlock(&thread_sync_);
		}
		delete [] tmp_progress;
		delete [] node_list;
	}

	pthread_t thread_;
	pthread_mutex_t thread_sync_;
	pthread_cond_t thread_state_;
	int max_progress_;
	int my_progress_;
	MPI_Request *send_req_; // length=max_progress
	MPI_Request *recv_req_; // length=mpi.size_2d
	int* send_buf_; // length=max_progress
	int* recv_buf_; // length=mpi.size_2d
	int* g_progress_; // length=mpi.size_2d
};

template <typename EdgeList>
void generate_graph(EdgeList* edge_list, const GraphGenerator<typename EdgeList::edge_type>* generator)
{
	typedef typename EdgeList::edge_type EdgeType;
	EdgeType* edge_buffer = static_cast<EdgeType*>
						(cache_aligned_xmalloc(EdgeList::CHUNK_SIZE*sizeof(EdgeType)));
	edge_list->beginWrite();
	const int64_t num_global_edges = generator->num_global_edges();
	const int64_t num_global_chunks = (num_global_edges + EdgeList::CHUNK_SIZE - 1) / EdgeList::CHUNK_SIZE;
	const int64_t num_iterations = (num_global_chunks + mpi.size_2d - 1) / mpi.size_2d;
	double logging_time = MPI_Wtime();
	ProgressReport* report = new ProgressReport(num_iterations);
	if(mpi.isMaster()) {
		double global_data_size = (double)num_global_edges * 16.0 / 1000000000.0;
		double local_data_size = global_data_size / mpi.size_2d;
		printf("Graph data size: %f GB ( %f GB per process )\n", global_data_size, local_data_size);
		printf("Using storage: %s\n", edge_list->data_is_in_file() ? "yes" : "no");
		if(edge_list->data_is_in_file()) {
			printf("Filepath: %s 1 2 ...\n", edge_list->get_filepath());
		}
		printf("Communication chunk size: %d\n", EdgeList::CHUNK_SIZE);
		printf("Generating graph: Total number of iterations: %"PRId64"\n", num_iterations);
	}
	report->begin_progress();
#pragma omp parallel
	for(int64_t i = 0; i < num_iterations; ++i) {
		const int64_t start_edge = std::min((mpi.size_2d*i + mpi.rank_2d) * EdgeList::CHUNK_SIZE, num_global_edges);
		const int64_t end_edge = std::min(start_edge + EdgeList::CHUNK_SIZE, num_global_edges);
		generator->generateRange(edge_buffer, start_edge, end_edge);
#if defined(__INTEL_COMPILER)
#pragma omp barrier
#endif
		// we need to synchronize before this code.
		// There is the implicit barrier on the end of for loops.
#pragma omp master
		{
#if 0
			for(int64_t i = start_edge; i < end_edge; ++i) {
				if( edge_buffer[i-start_edge].weight_ != 0xBEEF ) {
		//			fprintf(stderr, "Weight > 32: idx: %"PRId64"\n", i);
				}
			}
#endif
			edge_list->write(edge_buffer, end_edge - start_edge);

			if(mpi.isMaster()) {
				printf("Time for iteration %"PRId64" is %f \n", i, MPI_Wtime() - logging_time);
				logging_time = MPI_Wtime();
			}

			report->advace();
		}
#pragma omp barrier

	}
	report->end_progress();
	edge_list->endWrite();
	delete report; report = NULL;
	free(edge_buffer);
	if(mpi.isMaster()) printf("Finished generating.\n");
}

template <typename EdgeList>
void generate_graph_spec2010(EdgeList* edge_list, int scale, int edge_factor, int max_weight = 0)
{
	RmatGraphGenerator<typename EdgeList::edge_type, 5700, 1900> generator(scale, edge_factor, 255,
			BFS_PARAMS::USERSEED1, BFS_PARAMS::USERSEED2, InitialEdgeType::NONE);
	generate_graph(edge_list, &generator);
}

template <typename EdgeList>
void generate_graph_spec2012(EdgeList* edge_list, int scale, int edge_factor, int max_weight)
{
	RmatGraphGenerator<typename EdgeList::edge_type, 5500, 100> generator(scale, edge_factor, max_weight,
			BFS_PARAMS::USERSEED1, BFS_PARAMS::USERSEED2, InitialEdgeType::BINARY_TREE);
	generate_graph(edge_list, &generator);
}


// using SFINAE
// function #1
template <typename EdgeList>
void redistribute_edge_2d(EdgeList* edge_list, typename EdgeList::edge_type::has_weight dummy = 0)
{
	typedef typename EdgeList::edge_type EdgeType;
	ScatterContext scatter(mpi.comm_2d);
	EdgeType* edges_to_send = static_cast<EdgeType*>(
			xMPI_Alloc_mem(EdgeList::CHUNK_SIZE * sizeof(EdgeType)));
	int num_loops = edge_list->beginRead();
	edge_list->beginWrite();

	if(mpi.isMaster()) printf("%d iterations.\n", num_loops);

	const int rmask = ((1 << get_msb_index(mpi.size_2dr)) - 1);
	const int cmask = ((1 << get_msb_index(mpi.size_2d)) - 1 - rmask);
#define EDGE_OWNER(v0, v1) (((v0) & cmask) | ((v1) & rmask))

	for(int loop_count = 0; loop_count < num_loops; ++loop_count) {
		EdgeType* edge_data;
		const int edge_data_length = edge_list->read(&edge_data);

#pragma omp parallel
		{
			int* restrict counts = scatter.get_counts();

#pragma omp for schedule(static)
			for(int i = 0; i < edge_data_length; ++i) {
				const int64_t v0 = edge_data[i].v0();
				const int64_t v1 = edge_data[i].v1();
				(counts[EDGE_OWNER(v0,v1)])++;
			} // #pragma omp for schedule(static)

#pragma omp master
			{ scatter.sum(); } // #pragma omp master
#pragma omp barrier

			int* restrict offsets = scatter.get_offsets();

#pragma omp for schedule(static)
			for(int i = 0; i < edge_data_length; ++i) {
				const int64_t v0 = edge_data[i].v0();
				const int64_t v1 = edge_data[i].v1();
				const int weight = edge_data[i].weight_;
				//assert (offsets[edge_owner(v0,v1)] < 2 * FILE_CHUNKSIZE);
				edges_to_send[(offsets[EDGE_OWNER(v0,v1)])++].set(v0,v1,weight);
			} // #pragma omp for schedule(static)
		} // #pragma omp parallel

		if(mpi.isMaster()) printf("Scatter edges.\n");

		EdgeType* recv_edges = scatter.scatter(edges_to_send);
		const int64_t num_recv_edges = scatter.get_recv_count();
		edge_list->write(recv_edges, num_recv_edges);
		scatter.free(recv_edges);

		if(mpi.isMaster()) printf("Iteration %d finished.\n", loop_count);
	}
	if(mpi.isMaster()) printf("Finished.\n");
	edge_list->endWrite();
	edge_list->endRead();
	MPI_Free_mem(edges_to_send);
}

// function #2
template <typename EdgeList>
void redistribute_edge_2d(EdgeList* edge_list, typename EdgeList::edge_type::no_weight dummy = 0)
{
	typedef typename EdgeList::edge_type EdgeType;
	ScatterContext scatter(mpi.comm_2d);
	EdgeType* edges_to_send = static_cast<EdgeType*>(
			xMPI_Alloc_mem(EdgeList::CHUNK_SIZE * sizeof(EdgeType)));
	int num_loops = edge_list->beginRead();
	edge_list->beginWrite();

	const int rmask = ((1 << get_msb_index(mpi.size_2dr)) - 1);
	const int cmask = ((1 << get_msb_index(mpi.size_2d)) - 1 - rmask);
#define EDGE_OWNER(v0, v1) (((v0) & cmask) | ((v1) & rmask))
#ifndef NDEBUG
	const int log_size_r = get_msb_index(mpi.size_2dr);
#define VERTEX_OWNER_R(v) ((v) & rmask)
#define VERTEX_OWNER_C(v) (((v) & cmask) >> log_size_r)
#endif

	for(int loop_count = 0; loop_count < num_loops; ++loop_count) {
		EdgeType* edge_data;
		const int edge_data_length = edge_list->read(&edge_data);

#pragma omp parallel
		{
			int* restrict counts = scatter.get_counts();

#pragma omp for schedule(static)
			for(int i = 0; i < edge_data_length; ++i) {
				const int64_t v0 = edge_data[i].v0();
				const int64_t v1 = edge_data[i].v1();
				(counts[EDGE_OWNER(v0,v1)])++;
			} // #pragma omp for schedule(static)

#pragma omp master
			{ scatter.sum(); } // #pragma omp master
#pragma omp barrier

			int* restrict offsets = scatter.get_offsets();

#pragma omp for schedule(static)
			for(int i = 0; i < edge_data_length; ++i) {
				const int64_t v0 = edge_data[i].v0();
				const int64_t v1 = edge_data[i].v1();
				//assert (offsets[edge_owner(v0,v1)] < 2 * FILE_CHUNKSIZE);
				edges_to_send[(offsets[EDGE_OWNER(v0,v1)])++].set(v0,v1);
			} // #pragma omp for schedule(static)
		} // #pragma omp parallel

		EdgeType* recv_edges = scatter.scatter(edges_to_send);
		const int64_t num_recv_edges = scatter.get_recv_count();
#ifndef NDEBUG
		for(int64_t i = 0; i < num_recv_edges; ++i) {
			const int64_t v0 = recv_edges[i].v0();
			const int64_t v1 = recv_edges[i].v1();
			assert (VERTEX_OWNER_C(v0) == mpi.rank_2dc);
			assert (VERTEX_OWNER_R(v1) == mpi.rank_2dr);
		}
#undef VERTEX_OWNER_R
#undef VERTEX_OWNER_C
#endif
		edge_list->write(recv_edges, num_recv_edges);
		scatter.free(recv_edges);
	}
	edge_list->endWrite();
	edge_list->endRead();
	MPI_Free_mem(edges_to_send);
}

template <typename GraphType>
void decode_edge(GraphType& g, int64_t e0, int64_t e1, int64_t& v0, int64_t& v1, int& weight)
{
	const int log_size_r = get_msb_index(mpi.size_2dr);
	const int log_size = get_msb_index(mpi.size_2d);
	const int mask_packing_edge_lists = ((1 << g.log_packing_edge_lists()) - 1);
	const int log_weight_bits = g.log_packing_edge_lists_;

	const int packing_edge_lists = g.log_packing_edge_lists();
	const int log_local_verts = g.log_local_verts();
	const int64_t v0_high_mask = ((INT64_C(1) << (log_local_verts - packing_edge_lists)) - 1);

	const int rank_c = mpi.rank_2dc;
	const int rank_r = mpi.rank_2dr;

	int v0_r = e0 >> (log_local_verts - packing_edge_lists);
	int64_t v0_high = e0 & v0_high_mask;
	int64_t v0_middle = e1 & mask_packing_edge_lists;
	v0 = (((v0_high << packing_edge_lists) | v0_middle) << log_size) | ((rank_c << log_size_r) | v0_r);

	int64_t v1_and_weight = e1 >> packing_edge_lists;
	weight = v1_and_weight & ((1 << log_weight_bits) - 1);
	int64_t v1_high = v1_and_weight >> log_weight_bits;
	v1 = (v1_high << log_size_r) | rank_r;
}

template <typename GraphType>
void find_roots(GraphType& g, int64_t* bfs_roots, int& num_bfs_roots)
{
	using namespace BFS_PARAMS;
	/* Find roots and max used vertex */
	int64_t counter = 0;
	const int64_t nglobalverts = INT64_C(1) << g.log_actual_global_verts();
	int bfs_root_idx;
	for (bfs_root_idx = 0; bfs_root_idx < num_bfs_roots; ++bfs_root_idx) {
		int64_t root;
		while (1) {
			double d[2];
			make_random_numbers(2, USERSEED1, USERSEED2, counter, d);
			root = (int64_t)((d[0] + d[1]) * nglobalverts) % nglobalverts;
			counter += 2;
			if (counter > 2 * nglobalverts) break;
			int is_duplicate = 0;
			int i;
			for (i = 0; i < bfs_root_idx; ++i) {
				if (root == bfs_roots[i]) {
					is_duplicate = 1;
					break;
				}
			}
			if (is_duplicate) continue; /* Everyone takes the same path here */
			int root_ok = 0;
			if (g.get_vertex_rank_c(root) == mpi.rank_2dc) {
				root_ok = (int)g.has_edge(root);
			}
			MPI_Allreduce(MPI_IN_PLACE, &root_ok, 1, MPI_INT, MPI_LOR, mpi.comm_2d);
			if (root_ok) break;
		}
		bfs_roots[bfs_root_idx] = root;
	}
	num_bfs_roots = bfs_root_idx;
}

template <typename GraphType>
int64_t find_max_used_vertex(GraphType& g)
{
	int64_t max_used_vertex = 0;
	const int64_t nlocal = INT64_C(1) << g.log_actual_local_verts();
	for (int64_t i = nlocal; (i > 0) && (max_used_vertex == 0); --i) {
		int64_t local = i - 1;
		for(int64_t j = mpi.size_2dr; (j > 0) && (max_used_vertex == 0); --j) {
			int64_t r = j - 1;
			int64_t v0 = local * mpi.size_2d + mpi.rank_2dc * mpi.size_2dr + r;
			if (g.has_edge(v0)) {
				max_used_vertex = v0;
			}
		}
	}
	MPI_Allreduce(MPI_IN_PLACE, &max_used_vertex, 1, MPI_INT64_T, MPI_MAX, mpi.comm_2d);
	return max_used_vertex;
}

static int compare_doubles(const void* a, const void* b) {
  double aa = *(const double*)a;
  double bb = *(const double*)b;
  return (aa < bb) ? -1 : (aa == bb) ? 0 : 1;
}

enum {s_minimum, s_firstquartile, s_median, s_thirdquartile, s_maximum, s_mean, s_std, s_LAST};
static void get_statistics(const double x[], int n, double r[s_LAST]) {
  double temp;
  int i;
  /* Compute mean. */
  temp = 0;
  for (i = 0; i < n; ++i) temp += x[i];
  temp /= n;
  r[s_mean] = temp;
  /* Compute std. dev. */
  temp = 0;
  for (i = 0; i < n; ++i) temp += (x[i] - r[s_mean]) * (x[i] - r[s_mean]);
  temp /= n - 1;
  r[s_std] = sqrt(temp);
  /* Sort x. */
  double* xx = (double*)cache_aligned_xmalloc(n * sizeof(double));
  memcpy(xx, x, n * sizeof(double));
  qsort(xx, n, sizeof(double), compare_doubles);
  /* Get order statistics. */
  r[s_minimum] = xx[0];
  r[s_firstquartile] = (xx[(n - 1) / 4] + xx[n / 4]) * .5;
  r[s_median] = (xx[(n - 1) / 2] + xx[n / 2]) * .5;
  r[s_thirdquartile] = (xx[n - 1 - (n - 1) / 4] + xx[n - 1 - n / 4]) * .5;
  r[s_maximum] = xx[n - 1];
  /* Clean up. */
  free(xx);
}

void print_bfs_result(
	int num_bfs_roots,
	double* bfs_times,
	double* validate_times,
	double* edge_counts,
	bool result_ok)
{
	if (!result_ok) {
	  fprintf(stdout, "No results printed for invalid run.\n");
	} else {
	  int i;
	  double stats[s_LAST];
	  get_statistics(bfs_times, num_bfs_roots, stats);
	  fprintf(stdout, "min_time:                       %g\n", stats[s_minimum]);
	  fprintf(stdout, "firstquartile_time:             %g\n", stats[s_firstquartile]);
	  fprintf(stdout, "median_time:                    %g\n", stats[s_median]);
	  fprintf(stdout, "thirdquartile_time:             %g\n", stats[s_thirdquartile]);
	  fprintf(stdout, "max_time:                       %g\n", stats[s_maximum]);
	  fprintf(stdout, "mean_time:                      %g\n", stats[s_mean]);
	  fprintf(stdout, "stddev_time:                    %g\n", stats[s_std]);
	  get_statistics(edge_counts, num_bfs_roots, stats);
	  fprintf(stdout, "min_nedge:                      %.11g\n", stats[s_minimum]);
	  fprintf(stdout, "firstquartile_nedge:            %.11g\n", stats[s_firstquartile]);
	  fprintf(stdout, "median_nedge:                   %.11g\n", stats[s_median]);
	  fprintf(stdout, "thirdquartile_nedge:            %.11g\n", stats[s_thirdquartile]);
	  fprintf(stdout, "max_nedge:                      %.11g\n", stats[s_maximum]);
	  fprintf(stdout, "mean_nedge:                     %.11g\n", stats[s_mean]);
	  fprintf(stdout, "stddev_nedge:                   %.11g\n", stats[s_std]);
	  double* secs_per_edge = (double*)cache_aligned_xmalloc(num_bfs_roots * sizeof(double));
	  for (i = 0; i < num_bfs_roots; ++i) secs_per_edge[i] = bfs_times[i] / edge_counts[i];
	  get_statistics(secs_per_edge, num_bfs_roots, stats);
	  fprintf(stdout, "min_TEPS:                       %g\n", 1. / stats[s_maximum]);
	  fprintf(stdout, "firstquartile_TEPS:             %g\n", 1. / stats[s_thirdquartile]);
	  fprintf(stdout, "median_TEPS:                    %g\n", 1. / stats[s_median]);
	  fprintf(stdout, "thirdquartile_TEPS:             %g\n", 1. / stats[s_firstquartile]);
	  fprintf(stdout, "max_TEPS:                       %g\n", 1. / stats[s_minimum]);
	  fprintf(stdout, "harmonic_mean_TEPS:             %g\n", 1. / stats[s_mean]);
	  /* Formula from:
	   * Title: The Standard Errors of the Geometric and Harmonic Means and
	   *        Their Application to Index Numbers
	   * Author(s): Nilan Norris
	   * Source: The Annals of Mathematical Statistics, Vol. 11, No. 4 (Dec., 1940), pp. 445-448
	   * Publisher(s): Institute of Mathematical Statistics
	   * Stable URL: http://www.jstor.org/stable/2235723
	   * (same source as in specification). */
	  fprintf(stdout, "harmonic_stddev_TEPS:           %g\n", stats[s_std] / (stats[s_mean] * stats[s_mean] * sqrt(num_bfs_roots - 1)));
	  free(secs_per_edge); secs_per_edge = NULL;
	  get_statistics(validate_times, num_bfs_roots, stats);
	  fprintf(stdout, "min_validate:                   %g\n", stats[s_minimum]);
	  fprintf(stdout, "firstquartile_validate:         %g\n", stats[s_firstquartile]);
	  fprintf(stdout, "median_validate:                %g\n", stats[s_median]);
	  fprintf(stdout, "thirdquartile_validate:         %g\n", stats[s_thirdquartile]);
	  fprintf(stdout, "max_validate:                   %g\n", stats[s_maximum]);
	  fprintf(stdout, "mean_validate:                  %g\n", stats[s_mean]);
	  fprintf(stdout, "stddev_validate:                %g\n", stats[s_std]);
#if 0
	  for (i = 0; i < num_bfs_roots; ++i) {
		fprintf(stdout, "Run %3d:                        %g s, validation %g s\n", i + 1, bfs_times[i], validate_times[i]);
	  }
#endif
	}
}

struct LogFileTime {
	double bfs_time;
	double validate_time;
	int64_t edge_counts;
};

struct LogFileFormat {
	int scale;
	int edge_factor;
	int mpi_size;
	int num_runs;
	double generation_time;
	double construction_time;
	double redistribution_time;
	LogFileTime times[64];
};

int read_log_file(LogFileFormat* log, int SCALE, int edgefactor, double* bfs_times, double* validate_times, double* edge_counts)
{
	int resume_root_idx = 0;
	const char* logfilename = getenv("LOGFILE");
	if(logfilename) {
		if(mpi.isMaster()) {
			FILE* fp = fopen(logfilename, "rb");
			if(fp != NULL) {
				fread(log, sizeof(log[0]), 1, fp);
				if(log->scale != SCALE || log->edge_factor != edgefactor || log->mpi_size != mpi.size_2d) {
					fprintf(stderr, "Log file is not match the current run: params:(current),(log): SCALE:%d,%d, edgefactor:%d,%d, size:%d,%d\n",
					SCALE, log->scale, edgefactor, log->edge_factor, mpi.size_2d, log->mpi_size);
					resume_root_idx = -2;
				}
				else {
					resume_root_idx = log->num_runs;
					fprintf(stderr, "===== LOG START =====\n");
					fprintf(stderr, "graph_generation:               %f s\n", log->generation_time);
					fprintf(stderr, "construction_time:              %f s\n", log->construction_time);
					int i;
					for (i = 0; i < resume_root_idx; ++i) {
						fprintf(stderr, "Running BFS %d\n", i);
						fprintf(stderr, "Time for BFS %d is %f\n", i, log->times[i].bfs_time);
						fprintf(stderr, "Validating BFS %d\n", i);
						fprintf(stderr, "Validate time for BFS %d is %f\n", i, log->times[i].validate_time);
						fprintf(stderr, "TEPS for BFS %d is %g\n", i, log->times[i].edge_counts / log->times[i].bfs_time);

						bfs_times[i] = log->times[i].bfs_time;
						validate_times[i] = log->times[i].validate_time;
						edge_counts[i] = log->times[i].edge_counts;
					}
					fprintf(stderr, "=====  LOG END  =====\n");

				}
				fclose(fp);
			}
		}
		MPI_Bcast(&resume_root_idx, 1, MPI_INT, 0, mpi.comm_2d);
		if(resume_root_idx == -2) {
			MPI_Abort(mpi.comm_2d, 1);
		}
	}
	return resume_root_idx;
}

void update_log_file(LogFileFormat* log, double bfs_time, double validate_time, int64_t edge_counts)
{
	const char* logfilename = getenv("LOGFILE");
	if(logfilename && mpi.isMaster()) {
		int run_num = log->num_runs++;
		log->times[run_num].bfs_time = bfs_time;
		log->times[run_num].validate_time = validate_time;
		log->times[run_num].edge_counts = edge_counts;
		// save log;
		FILE* fp = fopen(logfilename, "wb");
		if(fp == NULL) {
			fprintf(stderr, "Cannot create log file ... skipping\n");
		}
		else {
			fwrite(log, sizeof(log[0]), 1, fp);
			fclose(fp);
		}
	}
}

void init_log(int SCALE, int edgefactor, double gen_time, double cons_time, double redis_time, LogFileFormat* log)
{
	log->scale = SCALE;
	log->edge_factor = edgefactor;
	log->mpi_size = mpi.size_2d;
	log->generation_time = gen_time;
	log->construction_time = cons_time;
	log->redistribution_time = redis_time;
	log->num_runs = 0;
}

#endif /* BENCHMARK_HELPER_HPP_ */
