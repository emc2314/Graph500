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
#ifndef GRAPH_CONSTRUCTOR_HPP_
#define GRAPH_CONSTRUCTOR_HPP_

#include "parameters.h"

//-------------------------------------------------------------//
// 2D partitioning
//-------------------------------------------------------------//

template <typename IndexArray, typename LocalVertsIndex>
class Graph2DCSR
{
public:
	Graph2DCSR()
	: row_starts_(NULL)
	, vertex_mapping_(NULL)
	, invert_vertex_mapping_(NULL)
	, log_actual_global_verts_(0)
	, log_packing_edge_lists_(-1)
	, extra_cols_(NULL)
	, extra_col_map_(NULL)
	, extra_col_invert_map_(NULL)
	{ }
	~Graph2DCSR()
	{
		clean();
	}

	void clean()
	{
		free(row_starts_); row_starts_ = NULL;
		index_array_.free();
		free(vertex_mapping_); vertex_mapping_ = NULL;
		free(invert_vertex_mapping_); invert_vertex_mapping_ = NULL;
		free(extra_cols_); extra_cols_ = NULL;
		free(extra_col_map_); extra_col_map_ = NULL;
		free(extra_col_invert_map_); extra_col_invert_map_ = NULL;
	}

	int log_actual_global_verts() const { return log_actual_global_verts_; }
	int log_actual_local_verts() const { return log_actual_global_verts_ - get_msb_index(mpi.size_2d); }
	int log_global_verts() const { return log_global_verts_; }
	int log_local_verts() const { return log_global_verts_ - get_msb_index(mpi.size_2d); }
	int log_packing_edge_lists() const { return log_packing_edge_lists_; }
	int log_edge_lists() const { return log_local_v0() - log_packing_edge_lists_; }
	int log_local_v0() const { return log_global_verts_ - get_msb_index(mpi.size_2dc); }
	int log_local_v1() const { return log_global_verts_ - get_msb_index(mpi.size_2dr); }

	// Reference Functions
	int get_vertex_rank(int64_t v)
	{
		const int64_t mask = mpi.size_2d - 1;
		return v & mask;
	}

	int get_vertex_rank_r(int64_t v)
	{
		const int64_t rmask = mpi.size_2dr - 1;
		return v & rmask;
	}

	int get_vertex_rank_c(int64_t v)
	{
		const int64_t cmask = mpi.size_2dc - 1;
		const int log_size_r = get_msb_index(mpi.size_2dr);
		return (v >> log_size_r) & cmask;
	}

	LocalVertsIndex get_edge_list_index(int64_t v0)
	{
		const int64_t rmask = mpi.size_2dr - 1;
		const int log_local_verts_minus_packing_edge_lists =
				log_local_verts() - log_packing_edge_lists();
		const int log_size_plus_packing_edge_lists =
				get_msb_index(mpi.size_2d) + log_packing_edge_lists();
		return ((v0 & rmask) << log_local_verts_minus_packing_edge_lists) |
				(v0 >> log_size_plus_packing_edge_lists);
	}

	int64_t get_v0_from_edge(int64_t e0, int64_t e1)
	{
		const int log_size_r = get_msb_index(mpi.size_2dr);
		const int log_size = get_msb_index(mpi.size_2d);
		const int mask_packing_edge_lists = ((1 << log_packing_edge_lists()) - 1);

		const int packing_edge_lists = log_packing_edge_lists();
		const int log_local_verts_ = log_local_verts();
		const int64_t v0_high_mask = ((INT64_C(1) << (log_local_verts_ - packing_edge_lists)) - 1);

		const int rank_c = mpi.rank_2dc;

		int v0_r = e0 >> (log_local_verts_ - packing_edge_lists);
		int64_t v0_high = e0 & v0_high_mask;
		int64_t v0_middle = e1 & mask_packing_edge_lists;
		return (((v0_high << packing_edge_lists) | v0_middle) << log_size) | ((rank_c << log_size_r) | v0_r);
	}

	int64_t get_v1_from_edge(int64_t e1, bool has_weight = false)
	{
		const int log_size_r = get_msb_index(mpi.size_2dr);

		const int packing_edge_lists = log_packing_edge_lists();
		const int rank_r = mpi.rank_2dr;

		if(has_weight) {
			int64_t v1_and_weight = e1 >> packing_edge_lists;
			int64_t v1_high = v1_and_weight >> log_max_weight_;
			return (v1_high << log_size_r) | rank_r;
		}
		else {
			int64_t v1_high = e1 >> packing_edge_lists;
			return (v1_high << log_size_r) | rank_r;
		}
	}

	int get_weight_from_edge(int64_t e1)
	{
		const int packing_edge_lists = log_packing_edge_lists();

		int64_t v1_and_weight = e1 >> packing_edge_lists;
		return v1_and_weight & ((1 << log_max_weight_) - 1);
	}

	bool has_edge(int64_t v0, bool has_weight = false)
	{
		int64_t column = get_edge_list_index(v0);
		for(int i = row_starts_[column]; i < row_starts_[column+1]; ++i) {
			if (v0 == get_v0_from_edge(column, index_array_(i))) {
				return true;
			}
		}
		if(extra_cols_ && extra_col_map_[column] != LocalVertsIndex(-1)) {
			const ExtraColumn& excol = extra_cols_[extra_col_map_[column]];
			int end = row_starts_[excol.column_start + excol.number_of_columns];
			for(int i = row_starts_[column]; i < end; ++i) {
				if (v0 == get_v0_from_edge(column, index_array_(i))) {
					return true;
				}
			}
		}
		return false;
	}

//private:
	int64_t* row_starts_;
	IndexArray index_array_;
	LocalVertsIndex* vertex_mapping_;
	LocalVertsIndex* invert_vertex_mapping_;

	int log_actual_global_verts_;
	int log_global_verts_;
	int log_packing_edge_lists_;
	int log_max_weight_;

	int max_weight_;

	// For folding
	struct ExtraColumn {
		LocalVertsIndex column_start;
		LocalVertsIndex number_of_columns;
	};

	LocalVertsIndex num_extra_cols_;
	int max_width_;
	ExtraColumn* extra_cols_;
	LocalVertsIndex* extra_col_map_;
	LocalVertsIndex* extra_col_invert_map_;
};

namespace detail {

template <typename IndexArray, typename LocalVertsIndex, typename EdgeList>
class GraphConstructor2DCSR
{
public:
	typedef Graph2DCSR<IndexArray, LocalVertsIndex> GraphType;
	typedef typename EdgeList::edge_type EdgeType;

	GraphConstructor2DCSR()
		: log_size_(get_msb_index(mpi.size_2d))
		, rmask_((1 << get_msb_index(mpi.size_2dr)) - 1)
		, cmask_((1 << get_msb_index(mpi.size_2d)) - 1 - rmask_)
		, edge_counts_(NULL)
		, degree_counts_(NULL)
	{ }
	~GraphConstructor2DCSR()
	{
		free(edge_counts_); edge_counts_ = NULL;
		free(degree_counts_); degree_counts_ = NULL;
	}

	void construct(EdgeList* edge_list, int log_minimum_global_verts, bool sort_by_degree, bool enable_folding, GraphType& g)
	{
		log_minimum_global_verts_ = log_minimum_global_verts;
		g.log_actual_global_verts_ = 0;
		do { // loop for max vertex estimation failure
			scatterAndCountEdge(edge_list, sort_by_degree, enable_folding, g);
		} while(g.log_actual_global_verts_ == 0);

		if(sort_by_degree) {
			if(mpi.isMaster()) printf("Making vertex mapping.\n");
			makeVertexMapping(g);
		}

		if(mpi.isMaster()) printf("Prepare for storing edges.\n");

		const int64_t num_edge_lists = (INT64_C(1) << g.log_edge_lists());
		g.row_starts_ = static_cast<int64_t*>(
				cache_aligned_xmalloc((num_edge_lists + 1)*sizeof(g.row_starts_[0])));
		g.row_starts_[0] = 0;
		for(int64_t i = 0; i < num_edge_lists; ++i){
			g.row_starts_[i + 1] = g.row_starts_[i] + edge_counts_[i];
		}
		const int64_t num_local_edges = g.row_starts_[num_edge_lists];
		g.index_array_.alloc(num_local_edges);
		memset(edge_counts_, 0x00, num_edge_lists*sizeof(edge_counts_[0]));

#if VERVOSE_MODE
		{
			// diagnostics
			int64_t graph_bytes = num_edge_lists*sizeof(size_t) + num_local_edges*IndexArray::bytes_per_edge;
			int64_t max_graph_bytes;
			int64_t sum_graph_bytes;
			MPI_Reduce(&graph_bytes, &max_graph_bytes, 1, MPI_INT64_T, MPI_MAX, 0, mpi.comm_2d);
			MPI_Reduce(&graph_bytes, &sum_graph_bytes, 1, MPI_INT64_T, MPI_SUM, 0, mpi.comm_2d);
			if(mpi.isMaster()) {
				double average = (double)(num_edge_lists*sizeof(size_t) + sum_graph_bytes/mpi.size_2d);
				fprintf(stderr, "max constructed graph size : %f GB (%f %%)\n",
						max_graph_bytes / (1024.0*1024.0*1024.0), max_graph_bytes / average * 100.0);
			}
		}
#endif

		scatterAndConstruct(edge_list, sort_by_degree, g);
		sortEdges(g);

		if(enable_folding) {
			foldGraph(g);
		}

		free(edge_counts_); edge_counts_ = NULL;
	}

private:
	int edge_owner(int64_t v0, int64_t v1) const { return (v0 & cmask_) | (v1 & rmask_); }
	int vertex_owner(int64_t v) const { return v & (mpi.size_2d - 1); }
	int64_t vertex_local(int64_t v) { return v >> log_size_; }

	void initializeParameters(
		int log_max_vertex,
		int64_t num_global_edges,
		bool sort_by_degree,
		bool enable_folding,
		GraphType& g)
	{
		g.log_actual_global_verts_ = log_max_vertex;
		g.log_global_verts_ = std::max(log_minimum_global_verts_, log_max_vertex);

		if(g.log_packing_edge_lists_ == -1){
			int average_degree = static_cast<int>(num_global_edges * 2 / (INT64_C(1) << log_max_vertex));
			if(mpi.isMaster()) {
				fprintf(stderr, "num_global_edges = %"PRId64", SCALE = %d, average_degree = %d\n",
						num_global_edges, log_max_vertex, average_degree);
			}
			int optimized_num_packing_edge_lists =
					(mpi.size_2dc * CACHE_LINE) / (average_degree * IndexArray::bytes_per_edge);
			if(optimized_num_packing_edge_lists < 2) {
				g.log_packing_edge_lists_ = 1;
			}
			else {
				g.log_packing_edge_lists_ = get_msb_index(optimized_num_packing_edge_lists);
			}
		}

		log_local_verts_ = g.log_local_verts();

		if(enable_folding) {
			int average_degree = static_cast<int>(num_global_edges * 2 / (INT64_C(1) << log_max_vertex));
			g.max_width_ = 2 * (1 << g.log_packing_edge_lists_) * 2 * average_degree;
		}

		const int64_t num_edge_lists = (INT64_C(1) << g.log_edge_lists());
		const int64_t num_local_verts = (INT64_C(1) << g.log_local_verts());

		edge_counts_ = static_cast<int64_t*>
			(cache_aligned_xmalloc(num_edge_lists*sizeof(edge_counts_[0])));
		memset(edge_counts_, 0x00, num_edge_lists*sizeof(edge_counts_[0]));

		if(sort_by_degree) {
			degree_counts_ = static_cast<int64_t*>
				(cache_aligned_xmalloc(num_local_verts*sizeof(degree_counts_[0])));
			memset(degree_counts_, 0x00, num_local_verts*sizeof(degree_counts_[0]));
		}
	}

	void countEdges(UnweightedPackedEdge* edges, int num_edges, GraphType& g) {
		const int64_t local_v0_mask = (int64_t(1) << g.log_local_v0()) - 1;
		const int log_packing_edge_lists = g.log_packing_edge_lists_;
#define COMPRESS_V0_INDEX(v) (((v) & local_v0_mask) >> log_packing_edge_lists)
		int i;
#pragma omp parallel for lastprivate(i), schedule(static)
		for(i = 0; i < (num_edges&(~3)); i += 4) {
			__sync_fetch_and_add(&edge_counts_[COMPRESS_V0_INDEX(edges[i+0].v0())], 1);
			__sync_fetch_and_add(&edge_counts_[COMPRESS_V0_INDEX(edges[i+1].v0())], 1);
			__sync_fetch_and_add(&edge_counts_[COMPRESS_V0_INDEX(edges[i+2].v0())], 1);
			__sync_fetch_and_add(&edge_counts_[COMPRESS_V0_INDEX(edges[i+3].v0())], 1);
		}
		for( ; i < num_edges; ++i) {
			edge_counts_[COMPRESS_V0_INDEX(edges[i].v0())]++;
		}
#undef COMPRESS_V0_INDEX
	}

	class CountDegree
	{
	public:
		typedef LocalVertsIndex send_type;

		CountDegree(GraphConstructor2DCSR* this__,
				UnweightedPackedEdge* edges, int64_t* degree_counts, int log_local_verts)
			: this_(this__)
			, edges_(edges)
			, degree_counts_(degree_counts)
			, log_local_verts_plus_size_r_(log_local_verts + get_msb_index(mpi.size_2dr))
			, local_verts_mask_((int64_t(1) << log_local_verts) - 1)
		{ }
		int target(int i) const {
			const int64_t v1_swizzled = edges_[i].v1();
			assert ((v1_swizzled >> log_local_verts_plus_size_r_) < mpi.size_2dc);
			return v1_swizzled >> log_local_verts_plus_size_r_;
		}
		LocalVertsIndex get(int i) const {
			return edges_[i].v1() & local_verts_mask_;
		}
		void set(int i, LocalVertsIndex v1) const {
			__sync_fetch_and_add(&degree_counts_[v1], 1);
		}
	private:
		GraphConstructor2DCSR* const this_;
		const UnweightedPackedEdge* const edges_;
		int64_t* degree_counts_;
		const int log_local_verts_plus_size_r_;
		const int64_t local_verts_mask_;
	};

	// using SFINAE
	// function #1
	template<typename EdgeType>
	void scanEdges(const EdgeType* edge_data, const int edge_data_length,
			int* restrict counts, uint64_t& max_vertex, int& max_weight, typename EdgeType::has_weight dummy = 0)
	{
#pragma omp for schedule(static)
		for(int i = 0; i < edge_data_length; ++i) {
			const int64_t v0 = edge_data[i].v0();
			const int64_t v1 = edge_data[i].v1();
			const int weight = edge_data[i].weight_;
			if (v0 == v1) continue;
			max_vertex |= (uint64_t)(v0 | v1);
			if(max_weight < weight) max_weight = weight;
			(counts[edge_owner(v0,v1)])++;
			(counts[edge_owner(v1,v0)])++;
		} // #pragma omp for schedule(static), reduction(|:max_vertex)
	}

	// function #2
	template<typename EdgeType>
	void scanEdges(const EdgeType* edge_data, const int edge_data_length,
			int* restrict counts, uint64_t& max_vertex, int& max_weight, typename EdgeType::no_weight dummy = 0)
	{
#pragma omp for schedule(static)
		for(int i = 0; i < edge_data_length; ++i) {
			const int64_t v0 = edge_data[i].v0();
			const int64_t v1 = edge_data[i].v1();
			if (v0 == v1) continue;
			max_vertex |= (uint64_t)(v0 | v1);
			(counts[edge_owner(v0,v1)])++;
			(counts[edge_owner(v1,v0)])++;
		} // #pragma omp for schedule(static), reduction(|:max_vertex)
	}

	// using SFINAE
	// function #1
	template<typename EdgeType>
	void reduceMaxWeight(int max_weight, GraphType& g, typename EdgeType::has_weight dummy = 0)
	{
		int global_max_weight;
		MPI_Allreduce(&max_weight, &global_max_weight, 1, MPI_INT, MPI_MAX, mpi.comm_2d);
		g.max_weight_ = global_max_weight;
		g.log_max_weight_ = get_msb_index(global_max_weight);
	}

	// function #2
	template<typename EdgeType>
	void reduceMaxWeight(int max_weight, GraphType& g, typename EdgeType::no_weight dummy = 0)
	{
	}

	void scatterAndCountEdge(EdgeList* edge_list, bool sort_by_degree, bool enable_folding, GraphType& g) {
		ScatterContext scatter(mpi.comm_2d);
		UnweightedPackedEdge* edges_to_send = static_cast<UnweightedPackedEdge*>(
				xMPI_Alloc_mem(2 * EdgeList::CHUNK_SIZE * sizeof(UnweightedPackedEdge)));
		int num_loops = edge_list->beginRead();
		uint64_t max_vertex = 0;
		int max_weight = 0;

		if(mpi.isMaster()) printf("Begin counting edges. Number of iterations is %d.\n", num_loops);

		for(int loop_count = 0; loop_count < num_loops; ++loop_count) {
			EdgeType* edge_data;
			const int edge_data_length = edge_list->read(&edge_data);

#pragma omp parallel reduction(|:max_vertex)
			{
				int* restrict counts = scatter.get_counts();
				scanEdges(edge_data, edge_data_length, counts, max_vertex, max_weight);
			} // #pragma omp parallel

			scatter.sum();

#if NETWORK_PROBLEM_AYALISYS
			if(mpi.isMaster()) printf("MPI_Allreduce...\n");
#endif

			MPI_Allreduce(MPI_IN_PLACE, &max_vertex, 1, MpiTypeOf<uint64_t>::type, MPI_BOR, mpi.comm_2d);

#if NETWORK_PROBLEM_AYALISYS
			if(mpi.isMaster()) printf("OK! \n");
#endif

			const int log_max_vertex = get_msb_index(max_vertex) + 1;
			if(g.log_actual_global_verts_ == 0) {
				initializeParameters(log_max_vertex,
						edge_list->num_local_edges()*mpi.size_2d, sort_by_degree, enable_folding, g);
			}
			else if(log_max_vertex != g.log_actual_global_verts_) {
				// max vertex estimation failure
				if (mpi.isMaster() == 0) {
					fprintf(stderr, "Restarting because of change of log_max_vertex from %d"
							"to %d\n", g.log_actual_global_verts_, log_max_vertex);
				}

				free(edge_counts_); edge_counts_ = NULL;
				free(degree_counts_); degree_counts_ = NULL;

				break;
			}

			const int log_local_verts = log_local_verts_;
			const int64_t log_size = get_msb_index(mpi.size_2d);
			const int64_t size_mask = mpi.size_2d - 1;
#define SWIZZLE_VERTEX(c) (((c) >> log_size) | (((c) & size_mask) << log_local_verts))

#pragma omp parallel
			{
				int* restrict offsets = scatter.get_offsets();

#pragma omp for schedule(static)
				for(int i = 0; i < edge_data_length; ++i) {
					const int64_t v0 = edge_data[i].v0();
					const int64_t v1 = edge_data[i].v1();
					if (v0 == v1) continue;
					const int64_t v0_swizzled = SWIZZLE_VERTEX(v0);
					const int64_t v1_swizzled = SWIZZLE_VERTEX(v1);
					//assert (offsets[edge_owner(v0,v1)] < 2 * FILE_CHUNKSIZE);
					edges_to_send[(offsets[edge_owner(v0,v1)])++].set(v0_swizzled,v1_swizzled);
					//assert (offsets[edge_owner(v1,v0)] < 2 * FILE_CHUNKSIZE);
					edges_to_send[(offsets[edge_owner(v1,v0)])++].set(v1_swizzled,v0_swizzled);
				} // #pragma omp for schedule(static)
			} // #pragma omp parallel

#if NETWORK_PROBLEM_AYALISYS
			if(mpi.isMaster()) printf("MPI_Alltoall...\n");
#endif

#undef SWIZZLE_VERTEX
			UnweightedPackedEdge* recv_edges = scatter.scatter(edges_to_send);

#if NETWORK_PROBLEM_AYALISYS
			if(mpi.isMaster()) printf("OK! \n");
#endif

			const int64_t num_recv_edges = scatter.get_recv_count();
			countEdges(recv_edges, num_recv_edges, g);

#ifndef NDEBUG
			const int rmask = ((1 << get_msb_index(mpi.size_2dr)) - 1);
			const int log_size_r = get_msb_index(mpi.size_2dr);
#define VERTEX_OWNER_R(v) (((v) >> log_local_verts) & rmask)
#define VERTEX_OWNER_C(v) ((((v) >> log_local_verts) & size_mask) >> log_size_r)
			for(int i = 0; i < num_recv_edges; ++i) {
				const int64_t v0 = recv_edges[i].v0();
				const int64_t v1 = recv_edges[i].v1();
				assert (VERTEX_OWNER_C(v0) == mpi.rank_2dc);
				assert (VERTEX_OWNER_R(v1) == mpi.rank_2dr);
			}
#endif

			if(sort_by_degree) {
				// check
				if(g.log_local_verts() >= static_cast<int>(sizeof(LocalVertsIndex)*8)) {
					fprintf(stderr, "Error: Bit length of LocalVertsIndex in not enough to process this graph.");
					throw "InvalidOperationException";
				}

#if NETWORK_PROBLEM_AYALISYS
				if(mpi.isMaster()) printf("MPI_Alltoall...\n");
#endif

				MpiCollective::scatter(CountDegree(this, recv_edges, degree_counts_, log_local_verts),
						num_recv_edges, mpi.comm_2dr);

#if NETWORK_PROBLEM_AYALISYS
				if(mpi.isMaster()) printf("OK! \n");
#endif
#ifndef NDEBUG
				for(int i = 0; i < num_recv_edges; ++i) {
					const int64_t v1 = recv_edges[i].v1();
					assert (VERTEX_OWNER_R(v1) == mpi.rank_2dr);
				}
#undef VERTEX_OWNER_R
#undef VERTEX_OWNER_C
#endif
			}

			scatter.free(recv_edges);

			if(mpi.isMaster()) printf("Iteration %d finished.\n", loop_count);
		}
		edge_list->endRead();
		MPI_Free_mem(edges_to_send);

		reduceMaxWeight<EdgeType>(max_weight, g);

		if(mpi.isMaster()) printf("Finished counting edges.\n");
	}

	void makeVertexMapping(GraphType& g) {
		const int64_t num_local_verts = (INT64_C(1) << g.log_local_verts());
		g.vertex_mapping_ = static_cast<LocalVertsIndex*>(
				cache_aligned_xmalloc(num_local_verts*sizeof(LocalVertsIndex)));
#ifndef NDEBUG
		// for "assert(g->vert_map[g->vert_invmap[i]] == 0);"
		memset(g.vertex_mapping_, 0x00, num_local_verts*sizeof(LocalVertsIndex));
#endif
		g.invert_vertex_mapping_ = static_cast<LocalVertsIndex*>(
				cache_aligned_xmalloc(num_local_verts*sizeof(LocalVertsIndex)));

#pragma omp parallel for
		for(int64_t i = 0; i < num_local_verts; ++i) {
			g.invert_vertex_mapping_[i] = i;
		}
		// sort
		sort2(degree_counts_, g.invert_vertex_mapping_, num_local_verts, std::greater<uint16_t>());
		free(degree_counts_); degree_counts_ = NULL;

#pragma omp parallel for
		for(int64_t i = 0; i < num_local_verts; ++i) {
			assert(g.vertex_mapping_[g.invert_vertex_mapping_[i]] == 0);
			g.vertex_mapping_[g.invert_vertex_mapping_[i]] = i;
		}
#ifndef NDEBUG
		unsigned long *check_bitmap = static_cast<unsigned long*>(
				cache_aligned_xmalloc(num_local_verts/8));
		for(int64_t i = 0; i < num_local_verts; ++i) {
			int word_idx = g.invert_vertex_mapping_[i] / (sizeof(unsigned long)*8);
			int bit_idx = g.invert_vertex_mapping_[i] % (sizeof(unsigned long)*8);
			check_bitmap[word_idx] |= (1UL << bit_idx);
		}
		for(int64_t i = 0; i <
			num_local_verts/static_cast<int64_t>(sizeof(unsigned long)*8); ++i)
		{
			if(check_bitmap[i] != 0xFFFFFFFFFFFFFFFFUL) {
				fprintf(stderr, "error : check_bitmap[i]=0x%lx\n", check_bitmap[i]);
				assert (0);
			}
		}
		free(check_bitmap); check_bitmap = NULL;
#endif
	}

	// using SFINAE
	// function #1
	template<typename EdgeType>
	void writeSendEdges(const EdgeType* edge_data, const int edge_data_length,
			int* restrict offsets, EdgeType* edges_to_send, typename EdgeType::has_weight dummy = 0)
	{
		const int log_local_verts = log_local_verts_;
		const int64_t log_size = get_msb_index(mpi.size_2d);
		const int64_t size_mask = mpi.size_2d - 1;
#define SWIZZLE_VERTEX(c) (((c) >> log_size) | (((c) & size_mask) << log_local_verts))
#pragma omp for schedule(static)
		for(int i = 0; i < edge_data_length; ++i) {
			const int64_t v0 = edge_data[i].v0();
			const int64_t v1 = edge_data[i].v1();
			if (v0 == v1) continue;
			const int64_t v0_swizzled = SWIZZLE_VERTEX(v0);
			const int64_t v1_swizzled = SWIZZLE_VERTEX(v1);
			//assert (offsets[edge_owner(v0,v1)] < 2 * FILE_CHUNKSIZE);
			edges_to_send[(offsets[edge_owner(v0,v1)])++].set(v0_swizzled, v1_swizzled, edge_data[i].weight_);
			//assert (offsets[edge_owner(v1,v0)] < 2 * FILE_CHUNKSIZE);
			edges_to_send[(offsets[edge_owner(v1,v0)])++].set(v1_swizzled, v0_swizzled, edge_data[i].weight_);
		} // #pragma omp for schedule(static)
#undef SWIZZLE_VERTEX
	}

	// function #2
	template<typename EdgeType>
	void writeSendEdges(const EdgeType* edge_data, const int edge_data_length,
			int* restrict offsets, EdgeType* edges_to_send, typename EdgeType::no_weight dummy = 0)
	{
		const int log_local_verts = log_local_verts_;
		const int64_t log_size = get_msb_index(mpi.size_2d);
		const int64_t size_mask = mpi.size_2d - 1;
#define SWIZZLE_VERTEX(c) (((c) >> log_size) | (((c) & size_mask) << log_local_verts))
#pragma omp for schedule(static)
		for(int i = 0; i < edge_data_length; ++i) {
			const int64_t v0 = edge_data[i].v0();
			const int64_t v1 = edge_data[i].v1();
			if (v0 == v1) continue;
			const int64_t v0_swizzled = SWIZZLE_VERTEX(v0);
			const int64_t v1_swizzled = SWIZZLE_VERTEX(v1);
			//assert (offsets[edge_owner(v0,v1)] < 2 * FILE_CHUNKSIZE);
			edges_to_send[(offsets[edge_owner(v0,v1)])++].set(v0_swizzled, v1_swizzled);
			//assert (offsets[edge_owner(v1,v0)] < 2 * FILE_CHUNKSIZE);
			edges_to_send[(offsets[edge_owner(v1,v0)])++].set(v1_swizzled, v0_swizzled);
		} // #pragma omp for schedule(static)
#undef SWIZZLE_VERTEX
	}

	// using SFINAE
	// function #1
	template<typename EdgeType>
	void addEdges(EdgeType* edges, int num_edges, GraphType& g, typename EdgeType::has_weight dummy = 0)
	{
		const int mask_packing_edge_lists = ((1 << g.log_packing_edge_lists()) - 1);
		const int log_weight_bits = g.log_packing_edge_lists_;
		const int log_local_verts = log_local_verts_;
		const int log_local_verts_plus_size_r = log_local_verts + get_msb_index(mpi.size_2dr);
		const int64_t local_v0_mask = (int64_t(1) << g.log_local_v0()) - 1;
		const int64_t local_verts_mask = (int64_t(1) << g.log_local_verts()) - 1;
		const int log_packing_edge_lists = g.log_packing_edge_lists();
#define COMPRESS_V0_INDEX(v) (((v) & local_v0_mask) >> log_packing_edge_lists)

#pragma omp parallel for schedule(static)
		for(int i = 0; i < num_edges; ++i) {
			const int64_t v0 = edges[i].v0();
			const int64_t v1 = edges[i].v1();
			const int weight = edges[i].weight_;
			const int64_t r_index = COMPRESS_V0_INDEX(v0);
			const int64_t c_index = g.row_starts_[r_index] + __sync_fetch_and_add(&edge_counts_[r_index], 1);

			// high bits of v1 | low bits of v0
			const int64_t b0 = v1 >> log_local_verts_plus_size_r;
			const int64_t b1 = v1 & local_verts_mask;
			const int64_t b3 = v0 & mask_packing_edge_lists;
			const int64_t value = (((((b0 << log_local_verts) | b1) << log_weight_bits) | weight) << log_packing_edge_lists) | b3;

			// random access (write)
			assert( g.index_array_(c_index) == 0 );
			g.index_array_.set(c_index, value);
		}
#undef COMPRESS_V0_INDEX
	}

	// function #2
	template<typename EdgeType>
	void addEdges(EdgeType* edges, int num_edges, GraphType& g, typename EdgeType::no_weight dummy = 0)
	{
		const int mask_packing_edge_lists = ((1 << g.log_packing_edge_lists()) - 1);
		const int log_local_verts = log_local_verts_;
		const int log_local_verts_plus_size_r = log_local_verts + get_msb_index(mpi.size_2dr);
		const int64_t local_v0_mask = (int64_t(1) << g.log_local_v0()) - 1;
		const int64_t local_verts_mask = (int64_t(1) << g.log_local_verts()) - 1;
		const int log_packing_edge_lists = g.log_packing_edge_lists();
#define COMPRESS_V0_INDEX(v) (((v) & local_v0_mask) >> log_packing_edge_lists)

#pragma omp parallel for schedule(static)
		for(int i = 0; i < num_edges; ++i) {
			const int64_t v0 = edges[i].v0();
			const int64_t v1 = edges[i].v1();
			const int64_t r_index = COMPRESS_V0_INDEX(v0);
			const int64_t c_index = g.row_starts_[r_index] + __sync_fetch_and_add(&edge_counts_[r_index], 1);

			// high bits of v0 | low bits of v1
			const int64_t b0 = v1 >> log_local_verts_plus_size_r;
			const int64_t b1 = v1 & local_verts_mask;
			const int64_t b3 = v0 & mask_packing_edge_lists;
			const int64_t value = (((b0 << log_local_verts) | b1) << log_packing_edge_lists) | b3;

			// random access (write)
#ifndef NDEBUG
			assert( g.index_array_(c_index) == 0 );
#endif
			g.index_array_.set(c_index, value);
		}
#undef COMPRESS_V0_INDEX
	}

	template <typename EdgeType>
	class VertexSortingConversion
	{
	public:
		typedef LocalVertsIndex send_type;
		typedef int64_t recv_type;

		VertexSortingConversion(GraphConstructor2DCSR* this__,
				EdgeType* edges, LocalVertsIndex* vertex_mapping, int log_local_verts)
			: this_(this__)
			, edges_(edges)
			, vertex_mapping_(vertex_mapping)
			, log_local_verts_plus_size_r_(log_local_verts + get_msb_index(mpi.size_2dr))
			, local_verts_mask_((int64_t(1) << log_local_verts) - 1)
			, size_mask_(int64_t(mpi.rank_2d) << log_local_verts)
		{ }
		int target(int i) const {
			const int64_t v1_swizzled = edges_[i].v1();
			assert ((v1_swizzled >> log_local_verts_plus_size_r_) < mpi.size_2dc);
			return v1_swizzled >> log_local_verts_plus_size_r_;
		}
		LocalVertsIndex get(int i) const {
			return edges_[i].v1() & local_verts_mask_;
		}
		int64_t map(LocalVertsIndex value) const {
			return vertex_mapping_[value] | size_mask_;
		}
		void set(int i, int64_t v1) const {
			const int64_t v0 = edges_[i].v0();
			edges_[i].set(v0, v1);
		}
	private:
		GraphConstructor2DCSR* const this_;
		EdgeType* const edges_;
		const LocalVertsIndex* const vertex_mapping_;
		const int log_local_verts_plus_size_r_;
		const int64_t local_verts_mask_;
		const int64_t size_mask_;
	};

	void scatterAndConstruct(EdgeList* edge_list, bool sort_by_degree, GraphType& g) {
		ScatterContext scatter(mpi.comm_2d);
		EdgeType* edges_to_send = static_cast<EdgeType*>(
				xMPI_Alloc_mem(2 * EdgeList::CHUNK_SIZE * sizeof(EdgeType)));
		int num_loops = edge_list->beginRead();

		if(mpi.isMaster()) printf("Begin construction. Number of iterations is %d.\n", num_loops);

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
					if (v0 == v1) continue;
					(counts[edge_owner(v0,v1)])++;
					(counts[edge_owner(v1,v0)])++;
				} // #pragma omp for schedule(static)
			} // #pragma omp parallel

			scatter.sum();

#pragma omp parallel
			{
				int* offsets = scatter.get_offsets();
				writeSendEdges(edge_data, edge_data_length, offsets, edges_to_send);
			}

			if(mpi.isMaster()) printf("Scatter edges.\n");

			EdgeType* recv_edges = scatter.scatter(edges_to_send);
			const int num_recv_edges = scatter.get_recv_count();

			if(sort_by_degree) {
				MpiCollective::gather(VertexSortingConversion<EdgeType>(this, recv_edges,
						g.vertex_mapping_, g.log_local_verts()), num_recv_edges, mpi.comm_2dr);
			}

			if(mpi.isMaster()) printf("Add edges.\n");

			addEdges(recv_edges, num_recv_edges, g);

			scatter.free(recv_edges);

			if(mpi.isMaster()) printf("Iteration %d finished.\n", loop_count);
		}

		if(mpi.isMaster()) printf("Finished construction.\n");

		edge_list->endRead();
		MPI_Free_mem(edges_to_send);
	}

	// using SFINAE
	// function #1
	template<typename EdgeType>
	void sortEdgesInner(GraphType& g, typename EdgeType::has_weight dummy = 0)
	{
		int64_t sort_buffer_length = 2*1024;
		int64_t* restrict sort_buffer = (int64_t*)cache_aligned_xmalloc(sizeof(int64_t)*sort_buffer_length);
		const int64_t num_edge_lists = (int64_t(1) << g.log_edge_lists());
		const int log_weight_bits = g.log_packing_edge_lists_;

		const int log_packing_edge_lists = g.log_packing_edge_lists();
		const int index_bits = g.log_global_verts() - get_msb_index(mpi.size_2dr);
		const int64_t mask_packing_edge_lists = (int64_t(1) << log_packing_edge_lists) - 1;
		const int64_t mask_weight = (int64_t(1) << log_weight_bits) - 1;
		const int64_t mask_index = (int64_t(1) << index_bits) - 1;
		const int64_t mask_index_compare =
				(mask_index << (log_packing_edge_lists + log_weight_bits)) |
				mask_packing_edge_lists;

#define ENCODE(v) \
		(((((v & mask_packing_edge_lists) << log_weight_bits) | \
		((v >> log_packing_edge_lists) & mask_weight)) << index_bits) | \
		(v >> (log_packing_edge_lists + log_weight_bits)))
#define DECODE(v) \
		(((((v & mask_index) << log_weight_bits) | \
		((v >> index_bits) & mask_weight)) << log_packing_edge_lists) | \
		(v >> (index_bits + log_weight_bits)))

#pragma omp for
		for(int64_t i = 0; i < num_edge_lists; ++i) {
			const int64_t edge_count = edge_counts_[i];
			const int64_t rowstart_i = g.row_starts_[i];
			assert (g.row_starts_[i+1] - g.row_starts_[i] == edge_counts_[i]);

			if(edge_count > sort_buffer_length) {
				free(sort_buffer);
				while(edge_count > sort_buffer_length) sort_buffer_length *= 2;
				sort_buffer = (int64_t*)cache_aligned_xmalloc(sizeof(int64_t)*sort_buffer_length);
			}

			for(int64_t c = 0; c < edge_count; ++c) {
				const int64_t v = g.index_array_(rowstart_i + c);
				sort_buffer[c] = ENCODE(v);
				assert(v == DECODE(ENCODE(v)));
			}
			// sort sort_buffer
			std::sort(sort_buffer, sort_buffer + edge_count);

			int64_t idx = rowstart_i;
			int64_t prev_v = -1;
			for(int64_t c = 0; c < edge_count; ++c) {
				const int64_t sort_v = sort_buffer[c];
				// TODO: now duplicated edges are not merged because sort order is
				// v0 row bits > weight > index
				// To reduce parallel edges, sort by the order of
				// v0 row bits > index > weight
				// and if you want to optimize SSSP, sort again by the order of
				// v0 row bits > weight > index
			//	if((prev_v & mask_index_compare) != (sort_v & mask_index_compare)) {
					assert (prev_v < sort_v);
					const int64_t v = DECODE(sort_v);
					g.index_array_.set(idx, v);
			//		prev_v = sort_v;
					idx++;
			//	}
			}
		//	if(edge_counts_[i] > idx - rowstart_i) {
				edge_counts_[i] = idx - rowstart_i;
		//	}
		} // #pragma omp for

#undef ENCODE
#undef DECODE

		free(sort_buffer);
	}

	// function #2
	template<typename EdgeType>
	void sortEdgesInner(GraphType& g, typename EdgeType::no_weight dummy = 0)
	{
		int64_t sort_buffer_length = 2*1024;
		int64_t* restrict sort_buffer = (int64_t*)cache_aligned_xmalloc(sizeof(int64_t)*sort_buffer_length);
		const int64_t num_edge_lists = (INT64_C(1) << g.log_edge_lists());

		const int log_packing_edge_lists = g.log_packing_edge_lists();
		const int index_bits = g.log_global_verts() - get_msb_index(mpi.size_2dr);
		const int64_t mask_packing_edge_lists = (INT64_C(1) << log_packing_edge_lists) - 1;
		const int64_t mask_index = (INT64_C(1) << index_bits) - 1;

#define ENCODE(v) \
		((((v) & mask_packing_edge_lists) << index_bits) | ((v) >> log_packing_edge_lists))
#define DECODE(v) \
		((((v) & mask_index) << log_packing_edge_lists) | ((v) >> index_bits))

#pragma omp for
		for(int64_t i = 0; i < num_edge_lists; ++i) {
			const int64_t edge_count = edge_counts_[i];
			const int64_t rowstart_i = g.row_starts_[i];
			assert (g.row_starts_[i+1] - g.row_starts_[i] == edge_counts_[i]);

			if(edge_count > sort_buffer_length) {
				free(sort_buffer);
				while(edge_count > sort_buffer_length) sort_buffer_length *= 2;
				sort_buffer = (int64_t*)cache_aligned_xmalloc(sizeof(int64_t)*sort_buffer_length);
			}

			for(int64_t c = 0; c < edge_count; ++c) {
				const int64_t v = g.index_array_(rowstart_i + c);
				sort_buffer[c] = ENCODE(v);
				assert(v == DECODE(ENCODE(v)));
			}
			// sort sort_buffer
			std::sort(sort_buffer, sort_buffer + edge_count);

			int64_t idx = rowstart_i;
			int64_t prev_v = -1;
			for(int64_t c = 0; c < edge_count; ++c) {
				const int64_t sort_v = sort_buffer[c];
				if(prev_v != sort_v) {
					assert (prev_v < sort_v);
					const int64_t v = DECODE(sort_v);
					g.index_array_.set(idx, v);
					prev_v = sort_v;
					idx++;
				}
			}
		//	if(edge_counts_[i] > idx - rowstart_i) {
				edge_counts_[i] = idx - rowstart_i;
		//	}
		} // #pragma omp for

#undef ENCODE
#undef DECODE

		free(sort_buffer);
	}

	void sortEdges(GraphType& g) {

		if(mpi.isMaster()) printf("Sorting edges.\n");

#pragma omp parallel
		sortEdgesInner<EdgeType>(g);

		const int64_t num_edge_lists = (INT64_C(1) << g.log_edge_lists());
		// this loop can't be parallel
		int64_t rowstart_new = 0;
		for(int64_t i = 0; i < num_edge_lists; ++i) {
			const int64_t edge_count_new = edge_counts_[i];
			const int64_t rowstart_old = g.row_starts_[i]; // read before write
			g.row_starts_[i] = rowstart_new;
			if(rowstart_new != rowstart_old) {
				g.index_array_.move(rowstart_new, rowstart_old, edge_count_new);
			}
			rowstart_new += edge_count_new;
		}
//#if PRINT_CONSTRUCTION_TIME_DETAIL
		 int64_t num_edge_sum[2] = {0};
		 int64_t num_edge[2] = {g.row_starts_[num_edge_lists], rowstart_new};
		MPI_Reduce(num_edge, num_edge_sum, 2, MPI_INT64_T, MPI_SUM, 0, mpi.comm_2d);
		if(mpi.isMaster()) fprintf(stderr, "# of edges is reduced from %zd to %zd (%f%%)\n",
				num_edge_sum[0], num_edge_sum[1], (double)(num_edge_sum[0] - num_edge_sum[1])/(double)num_edge_sum[0]*100.0);
//#endif // #if PRINT_CONSTRUCTION_TIME_DETAIL
		g.row_starts_[num_edge_lists] = rowstart_new;
	}

	void foldGraph(GraphType& g) {
		const int64_t num_edge_lists = (INT64_C(1) << g.log_edge_lists());

		g.extra_col_map_ = static_cast<LocalVertsIndex*>(
				cache_aligned_xmalloc((num_edge_lists)*sizeof(g.extra_col_map_[0])));
		int64_t *original_row_starts = g.row_starts_;
		int max_width = g.max_width_;
		LocalVertsIndex num_long_edge_lists = 0, num_extra_cols = 0;

		// count a number of long edge lists
#pragma omp parallel for reduction(+: num_long_edge_lists)
		for(int64_t i = 0; i < num_edge_lists; ++i) {
			g.extra_col_map_[i] = LocalVertsIndex(-1);
			if(edge_counts_[i] > max_width) {
				++num_long_edge_lists;
			}
		}

		g.row_starts_ = static_cast<int64_t*>(
				cache_aligned_xmalloc((num_edge_lists + num_extra_cols + 1)*sizeof(g.row_starts_[0])));
		g.extra_cols_ = static_cast<typename GraphType::ExtraColumn*>(
				cache_aligned_xmalloc((num_long_edge_lists)*sizeof(g.extra_cols_[0])));

		// create extra column index mapping
		for(int64_t i = 0, k = 0; i < num_edge_lists; ++i) {
			g.row_starts_[i + 1] = g.row_starts_[i] + edge_counts_[i];

			if(edge_counts_[i] > max_width) {
				int num = ((edge_counts_[i] - 1) / max_width);
				g.extra_cols_[k].column_start = num_extra_cols;
				g.extra_cols_[k].number_of_columns = num;
				g.extra_col_map_[i] = k++;
				num_extra_cols += num;
				g.row_starts_[i + 1] -= max_width * num;
			}
		}

		// create extra column part of row starts and inversion mapping
		g.num_extra_cols_ = num_extra_cols;
		for(int64_t i = 0; i < num_edge_lists; ++i) {
			if(g.extra_col_map_[i] != LocalVertsIndex(-1)) {
				assert (edge_counts_[i] > max_width) ;
				int k = g.extra_col_map_[i];
				const int index = g.extra_cols_[k].column_start;
				const int num = g.extra_cols_[k].number_of_columns;
				const int base_index = g.row_starts_[num_edge_lists + index];
				for(int j = 0; j < num; ++j) {
					g.extra_col_invert_map_[index + j] = i;
					g.row_starts_[num_edge_lists + index + j + 1] = base_index + max_width * (j + 1);
				}
			}
		}
		assert (g.row_starts_[num_edge_lists + num_extra_cols] == original_row_starts[num_edge_lists]);

		int64_t num_local_edges = g.row_starts_[num_edge_lists];
		IndexArray original_index_array;
		original_index_array.alloc(num_local_edges);
		original_index_array.copy_from(0, g.index_array_, 0, num_local_edges);

		// create folding CSR
		for(int64_t i = 0; i < num_edge_lists; ++i) {
			const int64_t edge_list_length = g.row_starts_[i + 1] - g.row_starts_[i];
			g.index_array_.copy_from(g.row_starts_[i], original_index_array, original_row_starts[i], edge_list_length);

			if(g.extra_col_map_[i] != LocalVertsIndex(-1)) {
				int k = g.extra_col_map_[i];
				const int index = g.extra_cols_[k].column_start;
				const int num = g.extra_cols_[k].number_of_columns;
				assert (max_width * num ==
						(g.row_starts_[num_edge_lists + index + num] - g.row_starts_[num_edge_lists + index]));
				g.index_array_.copy_from(g.row_starts_[num_edge_lists + index],
					original_index_array, original_row_starts[i] + edge_list_length,
					max_width * num);
			}
		}
	}

	const int log_size_;
	const int rmask_;
	const int cmask_;
	int log_minimum_global_verts_;
	int log_local_verts_;

	int64_t* edge_counts_;
	int64_t* degree_counts_;
};

} // namespace detail {

template <typename IndexArray, typename LocalVertsIndex, typename EdgeList>
void construct_graph(EdgeList* edge_list, bool sort_by_degree, bool enable_folding,
		Graph2DCSR<IndexArray, LocalVertsIndex>& g)
{
	detail::GraphConstructor2DCSR<IndexArray, LocalVertsIndex, EdgeList> constructor;
	constructor.construct(edge_list, sort_by_degree, enable_folding, g);
}


#endif /* GRAPH_CONSTRUCTOR_HPP_ */
