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
#ifndef PRIMITIVES_HPP_
#define PRIMITIVES_HPP_

#include "mpi_workarounds.h"

//-------------------------------------------------------------//
// Edge Types
//-------------------------------------------------------------//

struct UnweightedEdge;
template <> struct MpiTypeOf<UnweightedEdge> { static MPI_Datatype type; };
MPI_Datatype MpiTypeOf<UnweightedEdge>::type = MPI_DATATYPE_NULL;

struct UnweightedEdge {
	int64_t v0_;
	int64_t v1_;

	typedef int no_weight;

	int64_t v0() const { return v0_; }
	int64_t v1() const { return v1_; }
	void set(int64_t v0, int64_t v1) { v0_ = v0; v1_ = v1; }

	static void initialize()
	{
		int block_length[] = {1, 1};
		MPI_Aint displs[] = {
				reinterpret_cast<MPI_Aint>(&(static_cast<UnweightedEdge*>(NULL)->v0_)),
				reinterpret_cast<MPI_Aint>(&(static_cast<UnweightedEdge*>(NULL)->v1_)) };
		MPI_Type_create_hindexed(2, block_length, displs, MPI_INT64_T, &MpiTypeOf<UnweightedEdge>::type);
		MPI_Type_commit(&MpiTypeOf<UnweightedEdge>::type);
	}

	static void uninitialize()
	{
		MPI_Type_free(&MpiTypeOf<UnweightedEdge>::type);
	}
};

struct UnweightedPackedEdge;
template <> struct MpiTypeOf<UnweightedPackedEdge> { static MPI_Datatype type; };
MPI_Datatype MpiTypeOf<UnweightedPackedEdge>::type = MPI_DATATYPE_NULL;

struct UnweightedPackedEdge {
	uint32_t v0_low_;
	uint32_t v1_low_;
	uint32_t high_;

	typedef int no_weight;

	int64_t v0() const { return (v0_low_ | (static_cast<int64_t>(high_ & 0xFFFF) << 32)); }
	int64_t v1() const { return (v1_low_ | (static_cast<int64_t>(high_ >> 16) << 32)); }
	void set(int64_t v0, int64_t v1) {
		v0_low_ = static_cast<uint32_t>(v0);
		v1_low_ = static_cast<uint32_t>(v1);
		high_ = ((v0 >> 32) & 0xFFFF) | ((v1 >> 16) & 0xFFFF0000U);
	}

	static void initialize()
	{
		int block_length[] = {1, 1, 1};
		MPI_Aint displs[] = {
				reinterpret_cast<MPI_Aint>(&(static_cast<UnweightedPackedEdge*>(NULL)->v0_low_)),
				reinterpret_cast<MPI_Aint>(&(static_cast<UnweightedPackedEdge*>(NULL)->v1_low_)),
				reinterpret_cast<MPI_Aint>(&(static_cast<UnweightedPackedEdge*>(NULL)->high_)) };
		MPI_Type_create_hindexed(3, block_length, displs, MPI_UINT32_T, &MpiTypeOf<UnweightedPackedEdge>::type);
		MPI_Type_commit(&MpiTypeOf<UnweightedPackedEdge>::type);
	}

	static void uninitialize()
	{
		MPI_Type_free(&MpiTypeOf<UnweightedPackedEdge>::type);
	}
};

struct WeightedEdge;
template <> struct MpiTypeOf<WeightedEdge> { static MPI_Datatype type; };
MPI_Datatype MpiTypeOf<WeightedEdge>::type = MPI_DATATYPE_NULL;

struct WeightedEdge {
	uint32_t v0_low_;
	uint32_t v1_low_;
	uint32_t high_;
	int weight_;

	typedef int has_weight;

	int64_t v0() const { return (v0_low_ | (static_cast<int64_t>(high_ & 0xFFFF) << 32)); }
	int64_t v1() const { return (v1_low_ | (static_cast<int64_t>(high_ >> 16) << 32)); }
	int weight() const { return weight_; }
	void set(int64_t v0, int64_t v1) {
		v0_low_ = static_cast<uint32_t>(v0);
		v1_low_ = static_cast<uint32_t>(v1);
		high_ = ((v0 >> 32) & 0xFFFF) | ((v1 >> 16) & 0xFFFF0000U);
	}
	void set(int64_t v0, int64_t v1, int weight) {
		set(v0, v1);
		weight_ = weight;
	}

	static void initialize()
	{
		int block_length[] = {1, 1, 1, 1};
		MPI_Aint displs[] = {
				reinterpret_cast<MPI_Aint>(&(static_cast<WeightedEdge*>(NULL)->v0_low_)),
				reinterpret_cast<MPI_Aint>(&(static_cast<WeightedEdge*>(NULL)->v1_low_)),
				reinterpret_cast<MPI_Aint>(&(static_cast<WeightedEdge*>(NULL)->high_)),
				reinterpret_cast<MPI_Aint>(&(static_cast<WeightedEdge*>(NULL)->weight_)) };
		MPI_Datatype types[] = {MPI_UINT32_T, MPI_UINT32_T, MPI_UINT32_T, MPI_INT};
		MPI_Type_create_struct(4, block_length, displs, types, &MpiTypeOf<WeightedEdge>::type);
		MPI_Type_commit(&MpiTypeOf<WeightedEdge>::type);
	}

	static void uninitialize()
	{
		MPI_Type_free(&MpiTypeOf<WeightedEdge>::type);
	}
};

//-------------------------------------------------------------//
// Index Array Types
//-------------------------------------------------------------//

class Pack40bit {
public:
	static const int bytes_per_edge = 5;

	Pack40bit() : i32_(NULL), i8_(NULL) { }
	~Pack40bit() { this->free(); }
	void alloc(int64_t length) {
		i32_ = static_cast<int32_t*>(cache_aligned_xmalloc(length*sizeof(int32_t)));
		i8_ = static_cast<uint8_t*>(cache_aligned_xmalloc(length*sizeof(uint8_t)));
#ifndef NDEBUG
		memset(i32_, 0x00, length*sizeof(i32_[0]));
		memset(i8_, 0x00, length*sizeof(i8_[0]));
#endif
	}
	void free() { ::free(i32_); i32_ = NULL; ::free(i8_); i8_ = NULL; }

	int64_t operator()(int64_t index) const {
		return static_cast<int64_t>(i8_[index]) |
				(static_cast<int64_t>(i32_[index]) << 8);
	}
	uint8_t low_bits(int64_t index) const { return i8_[index]; }
	void set(int64_t index, int64_t value) {
		i8_[index] = static_cast<uint8_t>(value);
		i32_[index] = static_cast<int32_t>(value >> 8);
	}
	void move(int64_t to, int64_t from, int64_t size) {
		memmove(i32_ + to, i32_ + from, sizeof(i32_[0])*size);
		memmove(i8_ + to, i8_ + from, sizeof(i8_[0])*size);
	}
	void copy_from(int64_t to, Pack40bit& array, int64_t from, int64_t size) {
		memcpy(i32_ + to, array.i32_ + from, sizeof(i32_[0])*size);
		memcpy(i8_ + to, array.i8_ + from, sizeof(i8_[0])*size);
	}
private:
	int32_t *i32_;
	uint8_t *i8_;
};

class Pack48bit {
public:
	static const int bytes_per_edge = 6;

	Pack48bit() : i32_(NULL), i16_(NULL) { }
	~Pack48bit() { this->free(); }
	void alloc(int64_t length) {
		i32_ = static_cast<int32_t*>(cache_aligned_xmalloc(length*sizeof(int32_t)));
		i16_ = static_cast<uint16_t*>(cache_aligned_xmalloc(length*sizeof(uint16_t)));
#ifndef NDEBUG
		memset(i32_, 0x00, length*sizeof(i32_[0]));
		memset(i16_, 0x00, length*sizeof(i16_[0]));
#endif
	}
	void free() { ::free(i32_); i32_ = NULL; ::free(i16_); i16_ = NULL; }

	int64_t operator()(int64_t index) const {
		return static_cast<int64_t>(i16_[index]) |
				(static_cast<int64_t>(i32_[index]) << 16);
	}
	uint16_t low_bits(int64_t index) const { return i16_[index]; }
	void set(int64_t index, int64_t value) {
		i16_[index] = static_cast<uint16_t>(value);
		i32_[index] = static_cast<int32_t>(value >> 16);
	}
	void move(int64_t to, int64_t from, int64_t size) {
		memmove(i32_ + to, i32_ + from, sizeof(i32_[0])*size);
		memmove(i16_ + to, i16_ + from, sizeof(i16_[0])*size);
	}
	void copy_from(int64_t to, Pack48bit& array, int64_t from, int64_t size) {
		memcpy(i32_ + to, array.i32_ + from, sizeof(i32_[0])*size);
		memcpy(i16_ + to, array.i16_ + from, sizeof(i16_[0])*size);
	}

	int32_t* get_ptr_high() { return i32_; }
	uint16_t* get_ptr_low() { return i16_; }
private:
	int32_t *i32_;
	uint16_t *i16_;
};

class Pack64bit {
public:
	static const int bytes_per_edge = 8;

	Pack64bit() : i64_(NULL) { }
	~Pack64bit() { this->free(); }
	void alloc(int64_t length) {
		i64_ = static_cast<int64_t*>(cache_aligned_xmalloc(length*sizeof(int64_t)));
#ifndef NDEBUG
		memset(i64_, 0x00, length*sizeof(int64_t));
#endif
	}
	void free() { ::free(i64_); i64_ = NULL; }

	int64_t operator()(int64_t index) const { return i64_[index]; }
	int64_t low_bits(int64_t index) const { return i64_[index]; }
	void set(int64_t index, int64_t value) { i64_[index] = value; }
	void move(int64_t to, int64_t from, int64_t size) {
		memmove(i64_ + to, i64_ + from, sizeof(i64_[0])*size);
	}
	void copy_from(int64_t to, Pack64bit& array, int64_t from, int64_t size) {
		memcpy(i64_ + to, array.i64_ + from, sizeof(i64_[0])*size);
	}

	int64_t* get_ptr() { return i64_; }
private:
	int64_t *i64_;
};


#endif /* PRIMITIVES_HPP_ */
