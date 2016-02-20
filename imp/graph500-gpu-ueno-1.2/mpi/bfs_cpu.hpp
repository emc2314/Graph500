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
#ifndef BFS_CPU_HPP_
#define BFS_CPU_HPP_

#include "bfs.hpp"

struct BfsOnCPU_Params {
	typedef uint64_t BitmapType;
	enum {
		LOG_PACKING_EDGE_LISTS = 6, // 2^6 = 64
		LOG_CQ_SUMMARIZING = 4, // 2^4 = 16 -> sizeof(int64_t)*32 = 128bytes
	};
};

template <typename IndexArray, typename LocalVertsIndex>
class BfsOnCPU
	: public BfsBase<IndexArray, LocalVertsIndex, BfsOnCPU_Params>
{
public:
	BfsOnCPU()
	: BfsBase<IndexArray, LocalVertsIndex, BfsOnCPU_Params>(false)
	  { }
};

#endif /* BFS_CPU_HPP_ */
