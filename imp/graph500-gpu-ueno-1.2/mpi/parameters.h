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
#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "utils_core.h"

// Project includes
#define VERVOSE_MODE 1
#define SHARED_VISITED_OPT 1

// set 0 with CUDA
#define SHARED_VISITED_STRIPE 0

// Validation Level: 0: No validation, 1: validate at first time only, 2: validate all results
// Note: To conform to the specification, you must set 2
#define VALIDATION_LEVEL 1

#define CUDA_ENABLED 1
#define CUDA_COMPUTE_EXCLUSIVE_THREAD_MODE 1
#define CUDA_CHECK_PRINT_RANK

// atomic level of scanning Sahred Visited
// 0: no atomic operation
// 1: non atomic read and atomic write
// 2: atomic read-write
#define SV_ATOMIC_LEVEL 1

#define SIMPLE_FOLD_COMM 1
#define DEBUG_PRINT 0
#define KD_PRINT 0

#define DISABLE_CUDA_CONCCURENT 0

#define GPU_COMM_OPT 1

#define NETWORK_PROBLEM_AYALISYS 0

namespace BFS_PARAMS {

#define SIZE_OF_SUMMARY_IS_EQUAL_TO_WARP_SIZE

enum {
	USERSEED1 = 2,
	USERSEED2 = 3,
	NUM_BFS_ROOTS = 16, // spec: 64
#if CUDA_ENABLED
	PACKET_LENGTH = 256,
	LOG_PACKET_LENGTH = 8,
#else
	PACKET_LENGTH = 256,
#endif
	BULK_TRANS_SIZE = 16*1024,

	COMM_V0_TAG = 0,
	COMM_V1_TAG = 1,

	DENOM_SHARED_VISITED_PART = 16,

	// non-parameters
	MAX_PACKETS_PER_BLOCK = BULK_TRANS_SIZE / PACKET_LENGTH,
	BLOCK_V0_LEGNTH = ((BULK_TRANS_SIZE*2 + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)) - sizeof(uint32_t),
	BLOCK_V1_LENGTH = BULK_TRANS_SIZE,
};
} // namespace BFS_PARAMS {

namespace GPU_PARAMS {
enum {
	LOG_WARP_SIZE = 5, // 32
	LOG_WARPS_PER_BLOCK = 3, // 8
	LOOPS_PER_THREAD = 8,
	ACTIVE_THREAD_BLOCKS = 64,
	NUMBER_CUDA_STREAMS = 4,

	// Dynamic Scheduling
	DS_TILE_SCALE = 16,
	DS_BLOCK_SCALE = 8,
	BLOCKS_PER_LAUNCH_RATE = 64*1024,

	GPU_BLOCK_V0_THRESHOLD = 256*1024 * 2 * 2,
	GPU_BLOCK_V1_THRESHOLD = 256*1024 * 2, // about 10MB
	GPU_BLOCK_V0_LEGNTH = GPU_BLOCK_V0_THRESHOLD + BFS_PARAMS::BLOCK_V0_LEGNTH,
	GPU_BLOCK_V1_LENGTH = GPU_BLOCK_V1_THRESHOLD + BFS_PARAMS::BLOCK_V1_LENGTH, // about 15MB
	GPU_BLOCK_MAX_PACKTES = GPU_BLOCK_V1_LENGTH / BFS_PARAMS::PACKET_LENGTH * 2,

	EXPAND_DECODE_BLOCK_LENGTH = 16*1024*1024, // 64MB
	EXPAND_STREAM_BLOCK_LENGTH = EXPAND_DECODE_BLOCK_LENGTH * 2, // 32MB
	EXPAND_PACKET_LIST_LENGTH = EXPAND_DECODE_BLOCK_LENGTH / BFS_PARAMS::PACKET_LENGTH * 2,

	LOG_PACKING_EDGE_LISTS = 5, // 2^5 = 32 // for debug
	LOG_CQ_SUMMARIZING = LOG_WARP_SIZE, // 32 ( number of threads in a warp )

	// non-parameters
	WARP_SIZE = 1 << LOG_WARP_SIZE,
	WARPS_PER_BLOCK = 1 << LOG_WARPS_PER_BLOCK,
	LOG_THREADS_PER_BLOCK = LOG_WARP_SIZE + LOG_WARPS_PER_BLOCK,
	THREADS_PER_BLOCK = WARP_SIZE*WARPS_PER_BLOCK,
	// Number of max threads launched by 1 GPU kernel.
	MAX_ACTIVE_WARPS = WARPS_PER_BLOCK*ACTIVE_THREAD_BLOCKS,
	TEMP_BUFFER_LINES = MAX_ACTIVE_WARPS*LOOPS_PER_THREAD,

	READ_GRAPH_OUTBUF_SIZE = THREADS_PER_BLOCK * BLOCKS_PER_LAUNCH_RATE * 2,
//	READ_GRAPH_OUTBUF_SIZE = THREADS_PER_BLOCK * BLOCKS_PER_LAUNCH_RATE / 2,

	// It is highly recommended to set the same value as READ_GRAPH_OUTBUF_SIZE.
	// However it is possible to set smaller value and reduce the memory consumption on GPU.
	// This might cause a memory bus error on GPU.
	FILTER_EDGE_OUTBUF_SIZE = READ_GRAPH_OUTBUF_SIZE / 4,

	NUMBER_PACKING_EDGE_LISTS = (1 << LOG_PACKING_EDGE_LISTS),
	NUMBER_CQ_SUMMARIZING = (1 << LOG_CQ_SUMMARIZING),
};
} // namespace GPU_PARAMS {


#endif /* PARAMETERS_H_ */
