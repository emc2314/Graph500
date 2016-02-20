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
#ifndef BFS_HPP_
#define BFS_HPP_

#include <pthread.h>

#include <deque>

#if CUDA_ENABLED
#include "gpu_host.hpp"
#endif

#include "utils.hpp"
#include "double_linked_list.h"
#include "fiber.hpp"

namespace bfs_detail {
using namespace BFS_PARAMS;

enum VARINT_BFS_KIND {
	VARINT_FOLD,
	VARINT_EXPAND_CQ, // current queue
	VARINT_EXPAND_SV, // shared visited
};

struct PacketIndex {
	uint16_t length;
	uint16_t num_int;
};

struct CompressedStream {
	uint32_t packet_index_start;
	union {
		uint8_t stream[BLOCK_V0_LEGNTH];
		PacketIndex index[BLOCK_V0_LEGNTH/sizeof(PacketIndex)];
	};
};

struct FoldCommBuffer {
	CompressedStream* v0_stream;
	uint32_t* v1_list; // length: BLOCK_V1_LENGTH
	// info
	uint8_t complete_flag; // flag is set when send or receive complete. 0: plist, 1: vlist
	int target; // rank to which send or from which receive
	int num_packets;
	int num_edges;
	int v0_stream_length;
	ListEntry free_link;
	ListEntry extra_buf_link;
};

struct FoldPacket {
	int num_edges;
	int64_t v0_list[PACKET_LENGTH];
	uint32_t v1_list[PACKET_LENGTH]; // SCALE - lgsize <= 32
};

inline int get_num_packet(CompressedStream* stream, int stream_length, bool pair_packet)
{
	return (stream_length - offsetof(CompressedStream, index)) / sizeof(PacketIndex)
			- stream->packet_index_start;
}

class Bfs2DComm
{
public:
	class EventHandler {
	public:
		virtual ~EventHandler() { }
		virtual void fold_received(FoldCommBuffer* data) = 0;
		virtual void fold_finish() = 0;
	};

	class Communicatable {
	public:
		virtual ~Communicatable() { }
		virtual void comm() = 0;
	};

	Bfs2DComm(EventHandler* event_handler, bool cuda_enabled)
		: event_handler_(event_handler)
		, cuda_enabled_(cuda_enabled)
	{
		d_ = new DynamicDataSet();
		pthread_mutex_init(&d_->thread_sync_, NULL);
		pthread_cond_init(&d_->thread_state_,  NULL);
		d_->cleanup_ = false;
		d_->command_active_ = false;
		d_->suspended_ = false;
		d_->terminated_ = false;
		initializeListHead(&d_->recv_free_buffer_);
		initializeListHead(&d_->extra_recv_buffer_);

		// check whether the size of CompressedStream is page-aligned
		// which v1_list of FoldCommBuffer needs to page-align. allocate_fold_comm_buffer
		const int foldcomm_width = roundup<CACHE_LINE>(sizeof(FoldCommBuffer));
		const int foldbuf_width = sizeof(uint32_t) * BLOCK_V1_LENGTH +
				roundup<PAGE_SIZE>(sizeof(CompressedStream));

		buffer_.fold_comm_ = cache_aligned_xcalloc(foldcomm_width*4*mpi.size_2dc);
		buffer_.fold_ = page_aligned_xcalloc(foldbuf_width*4*mpi.size_2dc);
		fold_node_ = (FoldNode*)malloc(sizeof(fold_node_[0])*mpi.size_2dc);
		fold_node_comm_ = (FoldNodeComm*)malloc(sizeof(fold_node_comm_[0])*mpi.size_2dc);
		mpi_reqs_ = (MPI_Request*)malloc(sizeof(mpi_reqs_[0])*mpi.size_2dc*REQ_TOTAL);

#if CUDA_ENABLED
		if(cuda_enabled_) {
			CudaStreamManager::begin_cuda();
			CUDA_CHECK(cudaHostRegister(buffer_.fold_, foldbuf_width*4*mpi.size_2dc, cudaHostRegisterPortable));
			CudaStreamManager::end_cuda();
		}
#endif

		for(int i = 0; i < mpi.size_2dc; ++i) {
			pthread_mutex_init(&fold_node_[i].send_mutex, NULL);
			pthread_cond_init(&fold_node_[i].send_state,  NULL);

			initializeListHead(&fold_node_[i].free_buffer);
			initializeListHead(&fold_node_[i].sending_buffer);

			FoldCommBuffer *buf[4];
			for(int k = 0; k < 4; ++k) {
				buf[k] = (FoldCommBuffer*)((uint8_t*)buffer_.fold_comm_ + foldcomm_width*(i*4 + k));
				buf[k]->v1_list = (uint32_t*)((uint8_t*)buffer_.fold_ + foldbuf_width*(i*4 + k));
				buf[k]->v0_stream = (CompressedStream*)&buf[k]->v1_list[BLOCK_V1_LENGTH];
				buf[k]->num_edges = 0;
				buf[k]->num_packets = 0;
				buf[k]->v0_stream_length = 0;
				initializeListEntry(&buf[k]->free_link);
				initializeListEntry(&buf[k]->extra_buf_link);
			}

			listInsertBack(&fold_node_[i].free_buffer, &buf[0]->free_link);
			listInsertBack(&fold_node_[i].free_buffer, &buf[1]->free_link);
			listInsertBack(&d_->recv_free_buffer_, &buf[2]->free_link);
			listInsertBack(&d_->recv_free_buffer_, &buf[3]->free_link);

			for(int k = 0; k < REQ_TOTAL; ++k) {
				mpi_reqs_[REQ_TOTAL*i + k] = MPI_REQUEST_NULL;
			}
			fold_node_comm_[i].recv_buffer = NULL;
			fold_node_comm_[i].send_buffer = NULL;
		}

		pthread_create(&d_->thread_, NULL, comm_thread_routine_, this);
	}

	virtual ~Bfs2DComm()
	{
		if(!d_->cleanup_) {
			d_->cleanup_ = true;
			pthread_mutex_lock(&d_->thread_sync_);
			d_->terminated_ = true;
			d_->suspended_ = true;
			d_->command_active_ = true;
			pthread_mutex_unlock(&d_->thread_sync_);
			pthread_cond_broadcast(&d_->thread_state_);
			pthread_join(d_->thread_, NULL);
			pthread_mutex_destroy(&d_->thread_sync_);
			pthread_cond_destroy(&d_->thread_state_);

			for(int i = 0; i < mpi.size_2dc; ++i) {
				pthread_mutex_destroy(&fold_node_[i].send_mutex);
				pthread_cond_destroy(&fold_node_[i].send_state);
			}

			while(listIsEmpty(&d_->extra_recv_buffer_) == false) {
				FoldCommBuffer* sb = CONTAINING_RECORD(d_->extra_recv_buffer_.fLink,
						FoldCommBuffer, extra_buf_link);
				listRemove(&sb->extra_buf_link);
#if CUDA_ENABLED
				if(cuda_enabled_) {
					CudaStreamManager::begin_cuda();
					CUDA_CHECK(cudaHostUnregister(sb));
					CudaStreamManager::end_cuda();
				}
#endif
				free(sb);
			}

#if CUDA_ENABLED
			if(cuda_enabled_) {
				CudaStreamManager::begin_cuda();
				CUDA_CHECK(cudaHostUnregister(buffer_.fold_));
				CudaStreamManager::end_cuda();
			}
#endif

			free(buffer_.fold_comm_); buffer_.fold_comm_ = NULL;
			free(buffer_.fold_); buffer_.fold_ = NULL;
			free(fold_node_); fold_node_ = NULL;
			free(fold_node_comm_); fold_node_comm_ = NULL;
			free(mpi_reqs_); mpi_reqs_ = NULL;

			delete d_; d_ = NULL;
		}
	}

	void begin_fold_comm()
	{
		CommCommand cmd;
		cmd.kind = FOLD_SEND_START;
		d_->fold_sending_count_ = mpi.size_2dc;
		put_command(cmd);
	}

	void fold_send(
			const uint8_t* stream_buffer,
			const uint32_t* v1_list,
			int stream_length,
			int num_edges,
			int dest_c)
	{
		const int max_v0_stream_length = sizeof(PacketIndex) *
				(BLOCK_V0_LEGNTH/sizeof(PacketIndex) - MAX_PACKETS_PER_BLOCK);

		FoldNode& dest_node = fold_node_[dest_c];
		assert(num_edges > 0);
		assert(stream_length >= num_edges);
#if VERVOSE_MODE
		profiling::TimeKeeper wait_;
#endif
		pthread_mutex_lock(&dest_node.send_mutex);

		while(listIsEmpty(&dest_node.free_buffer)) {
			pthread_cond_wait(&dest_node.send_state, &dest_node.send_mutex);
		}
		FoldCommBuffer* sb = CONTAINING_RECORD(dest_node.free_buffer.fLink, FoldCommBuffer, free_link);
		// if buffer is full
		while(sb->num_packets + 1 > MAX_PACKETS_PER_BLOCK
			|| sb->v0_stream_length + stream_length > max_v0_stream_length
			|| sb->num_edges + num_edges > BLOCK_V1_LENGTH)
		{
			assert (sb->num_edges <= BLOCK_V1_LENGTH);
			fold_send_submit(dest_c);
			sb = CONTAINING_RECORD(dest_node.free_buffer.fLink, FoldCommBuffer, free_link);
			//	assert (sb->num_edges == 0); // This assertion is NOT guaranteed.
		}
		// add to send buffer
		memcpy(sb->v0_stream->stream + sb->v0_stream_length, stream_buffer, sizeof(stream_buffer[0])*stream_length);
		memcpy(sb->v1_list + sb->num_edges, v1_list, sizeof(v1_list[0])*num_edges);
		dest_node.sb_index[sb->num_packets].length = stream_length;
		dest_node.sb_index[sb->num_packets].num_int = num_edges;
		sb->v0_stream_length += stream_length;
		sb->num_edges += num_edges;
		++sb->num_packets;
		pthread_mutex_unlock(&dest_node.send_mutex);
#if VERVOSE_MODE
		send_wait_time_ += wait_;
#endif
	}

	void fold_send_end(int dest_c)
	{
		FoldNode& dest_node = fold_node_[dest_c];
#if VERVOSE_MODE
		profiling::TimeKeeper wait_;
#endif
		pthread_mutex_lock(&dest_node.send_mutex);

		while(listIsEmpty(&dest_node.free_buffer)) {
			pthread_cond_wait(&dest_node.send_state, &dest_node.send_mutex);
		}
		FoldCommBuffer* sb = CONTAINING_RECORD(dest_node.free_buffer.fLink, FoldCommBuffer, free_link);

		if(sb->num_edges > 0) {
			fold_send_submit(dest_c);
			sb = CONTAINING_RECORD(dest_node.free_buffer.fLink, FoldCommBuffer, free_link);
		}

		assert(sb->num_edges == 0);
		fold_send_submit(dest_c);

		pthread_mutex_unlock(&dest_node.send_mutex);
#if VERVOSE_MODE
		send_wait_time_ += wait_;
#endif

		if(__sync_add_and_fetch(&d_->fold_sending_count_, -1) == 0) {
			CommCommand cmd;
			cmd.kind = FOLD_SEND_END;
			put_command(cmd);
		}
	}

	void input_command(Communicatable* comm)
	{
		CommCommand cmd;
		cmd.kind = MANUAL_COMM;
		cmd.cmd = comm;
		put_command(cmd);
	}

	void relase_fold_buffer(FoldCommBuffer* buf)
	{
		pthread_mutex_lock(&d_->thread_sync_);
		listInsertBack(&d_->recv_free_buffer_, &buf->free_link);
		pthread_mutex_unlock(&d_->thread_sync_);
	}

	int64_t expand_max_cq_size(int64_t cq_size) const
	{
		int64_t max_cq_size;
	    MPI_Allreduce(&cq_size, &max_cq_size, 1, get_mpi_type(cq_size), MPI_MAX, mpi.comm_2dc);
	    return max_cq_size;
	}
#if VERVOSE_MODE
	void submit_wait_time(const char* content, int n1, int n2 = 0) {
		send_wait_time_.submit(content, n1, n2);
	}
	void reset_wait_time() {
		send_wait_time_.reset();
	}
#endif
private:
	EventHandler* event_handler_;
	bool cuda_enabled_;

	enum COMM_COMMAND {
		FOLD_SEND_START,
		FOLD_SEND,
		FOLD_SEND_END,
		MANUAL_COMM,
	};

	struct CommCommand {
		COMM_COMMAND kind;
		union {
			// FOLD_SEND
			int dest_c;
			// COMM_COMMAND
			Communicatable* cmd;
		};
	};

	struct DynamicDataSet {
		// lock topology
		// FoldNode::send_mutex -> thread_sync_
		pthread_t thread_;
		pthread_mutex_t thread_sync_;
		pthread_cond_t thread_state_;

		bool cleanup_;

		// monitor : thread_sync_
		volatile bool command_active_;
		volatile bool suspended_;
		volatile bool terminated_;
		std::deque<CommCommand> command_queue_;
		ListEntry recv_free_buffer_;

		// accessed by comm thread only
		ListEntry extra_recv_buffer_;

		int fold_sending_count_;
	} *d_;

	struct FoldNode {
		pthread_mutex_t send_mutex;
		pthread_cond_t send_state;

		// monitor : send_mutex
		PacketIndex sb_index[MAX_PACKETS_PER_BLOCK];
		ListEntry free_buffer;

		// monitor : thread_sync_
		ListEntry sending_buffer;
	};

	struct FoldNodeComm {
		FoldCommBuffer* recv_buffer;
		FoldCommBuffer* send_buffer;
	};

	enum MPI_REQ_INDEX {
		REQ_SEND_V0 = 0,
		REQ_SEND_V1 = 1,
		REQ_RECV_V0 = 2,
		REQ_RECV_V1 = 3,
		REQ_TOTAL = 4,
	};

	FoldNode* fold_node_;
	FoldNodeComm* fold_node_comm_;

	// accessed by communication thread only
	MPI_Request* mpi_reqs_;

#if VERVOSE_MODE
	profiling::TimeSpan send_wait_time_;
#endif

	struct {
		void* fold_comm_;
		void* fold_;
	} buffer_;

	static void* comm_thread_routine_(void* pThis) {
		static_cast<Bfs2DComm*>(pThis)->comm_thread_routine();
		pthread_exit(NULL);
	}
	void comm_thread_routine()
	{
		int num_recv_active = 0;
		bool send_active = false;

		// command loop
		while(true) {
			if(d_->command_active_) {
				pthread_mutex_lock(&d_->thread_sync_);
				CommCommand cmd;
				while(pop_command(&cmd)) {
					pthread_mutex_unlock(&d_->thread_sync_);
					switch(cmd.kind) {
					case FOLD_SEND_START:
						for(int i = 0; i < mpi.size_2dc; ++i) {
							fold_set_receive_buffer(i);
							++num_recv_active;
						}
						send_active = true;
						break;
					case FOLD_SEND:
						fold_set_send_buffer(cmd.dest_c);
						break;
					case FOLD_SEND_END:
						send_active = false;
						if(num_recv_active == 0) {
							event_handler_->fold_finish();
						}
						break;
					case MANUAL_COMM:
						cmd.cmd->comm();
						break;
					}
					pthread_mutex_lock(&d_->thread_sync_);
				}
				pthread_mutex_unlock(&d_->thread_sync_);
			}
			if(num_recv_active == 0 && send_active == false) {
				pthread_mutex_lock(&d_->thread_sync_);
				if(d_->command_active_ == false) {
					d_->suspended_ = true;
					if( d_->terminated_ ) { pthread_mutex_unlock(&d_->thread_sync_); break; }
					pthread_cond_wait(&d_->thread_state_, &d_->thread_sync_);
				}
				pthread_mutex_unlock(&d_->thread_sync_);
			}

			int index;
			int flag;
			MPI_Status status;
			MPI_Testany(mpi.size_2dc * (int)REQ_TOTAL, mpi_reqs_, &index, &flag, &status);

			if(flag == 0 || index == MPI_UNDEFINED) {
				continue;
			}

			const int src_c = index/REQ_TOTAL;
			const MPI_REQ_INDEX req_kind = (MPI_REQ_INDEX)(index%REQ_TOTAL);
			const bool b_send = ((int)req_kind / 2) == 0;
			const int complete_flag = 1 << ((int)req_kind % 2);

			FoldNodeComm& comm_node = fold_node_comm_[src_c];
			FoldCommBuffer* buf = b_send ? comm_node.send_buffer : comm_node.recv_buffer;

			assert (mpi_reqs_[index] == MPI_REQUEST_NULL);
			mpi_reqs_[index] = MPI_REQUEST_NULL;
			buf->complete_flag |= complete_flag;

			switch(req_kind) {
			case REQ_RECV_V0:
				{
					int count;
					MPI_Get_count(&status, MPI_BYTE, &count);
					buf->v0_stream_length = buf->v0_stream->packet_index_start * sizeof(PacketIndex);
					buf->num_packets = get_num_packet(buf->v0_stream, count, true);
#if VERVOSE_MODE
					g_fold_recv += count;
#endif
				}
				break;
			case REQ_RECV_V1:
				{
					MPI_Get_count(&status, MPI_UNSIGNED, &buf->num_edges);
#if VERVOSE_MODE
					g_fold_recv += buf->num_edges * sizeof(uint32_t);
#endif
				}
				break;
			default:
				break;
			}

			if(buf->complete_flag == 3) {
				// complete
				if(b_send) {
					// send buffer
					FoldNode& node = fold_node_[src_c];
					buf->num_packets = 0;
					buf->num_edges = 0;
					buf->v0_stream_length = 0;
					comm_node.send_buffer = NULL;
					pthread_mutex_lock(&node.send_mutex);
					listInsertBack(&node.free_buffer, &buf->free_link);
					pthread_mutex_unlock(&node.send_mutex);
					fold_set_send_buffer(src_c);
					pthread_cond_broadcast(&node.send_state);
				}
				else {
					// recv buffer
					if(buf->num_edges == 0) {
						// received fold completion
						--num_recv_active;
						fold_node_comm_[src_c].recv_buffer = NULL;
						pthread_mutex_lock(&d_->thread_sync_);
						listInsertBack(&d_->recv_free_buffer_, &buf->free_link);
						pthread_mutex_unlock(&d_->thread_sync_);
						if(num_recv_active == 0 && send_active == false) {
							event_handler_->fold_finish();
						}
					}
					else {
						// received both plist and vlist
						// set new buffer for next receiving
						fold_set_receive_buffer(src_c);

						event_handler_->fold_received(buf);
					}
				}
			}

		}
	}

	bool pop_command(CommCommand* cmd) {
		if(d_->command_queue_.size()) {
			*cmd = d_->command_queue_[0];
			d_->command_queue_.pop_front();
			return true;
		}
		d_->command_active_ = false;
		return false;
	}

	void put_command(CommCommand& cmd)
	{
		bool command_active;

		pthread_mutex_lock(&d_->thread_sync_);
		d_->command_queue_.push_back(cmd);
		command_active = d_->command_active_;
		if(command_active == false) d_->command_active_ = true;
		pthread_mutex_unlock(&d_->thread_sync_);

		if(command_active == false) pthread_cond_broadcast(&d_->thread_state_);
	}

	FoldCommBuffer* allocate_fold_comm_buffer()
	{
		const int v0_offset = roundup<PAGE_SIZE>(sizeof(FoldCommBuffer));
		const int v1_offset = v0_offset + roundup<PAGE_SIZE>(sizeof(CompressedStream));
		const int mem_length = v1_offset + BLOCK_V1_LENGTH * sizeof(uint32_t);
		uint8_t* new_buffer = (uint8_t*)page_aligned_xcalloc(mem_length);
#if CUDA_ENABLED
		if(cuda_enabled_) {
			CudaStreamManager::begin_cuda();
			CUDA_CHECK(cudaHostRegister(new_buffer, mem_length, cudaHostRegisterPortable));
			CudaStreamManager::end_cuda();
		}
#endif
		FoldCommBuffer* r = (FoldCommBuffer*)new_buffer;
		r->v0_stream = (CompressedStream*)(new_buffer + v0_offset);
		r->v1_list = (uint32_t*)(new_buffer + v1_offset);
		initializeListEntry(&r->free_link);
		initializeListEntry(&r->extra_buf_link);
#if 0
		memset(r->v0_stream, 0, sizeof(CompressedStream));
		memset(r->v1_list, 0, BLOCK_V1_LENGTH*sizeof(r->v1_list[0]));
		memset(r, 0, sizeof(FoldCommBuffer));
		r->v0_stream = (CompressedStream*)(new_buffer + v0_offset);
		r->v1_list = (uint32_t*)(new_buffer + v1_offset);
#endif
		return r;
	}

	FoldCommBuffer* get_recv_buffer()
	{
		pthread_mutex_lock(&d_->thread_sync_);
		if(listIsEmpty(&d_->recv_free_buffer_)) {
			pthread_mutex_unlock(&d_->thread_sync_);
			FoldCommBuffer* new_buffer = allocate_fold_comm_buffer();
			pthread_mutex_lock(&d_->thread_sync_);
			listInsertBack(&d_->extra_recv_buffer_, &new_buffer->extra_buf_link);
			pthread_mutex_unlock(&d_->thread_sync_);
			return new_buffer;
		}
		FoldCommBuffer* rb = CONTAINING_RECORD(d_->recv_free_buffer_.fLink,
				FoldCommBuffer, free_link);
		listRemove(&rb->free_link);
		pthread_mutex_unlock(&d_->thread_sync_);
		return rb;
	}

	void fold_set_receive_buffer(int src_c)
	{
		FoldNodeComm& comm_node = fold_node_comm_[src_c];
		MPI_Request* recv_reqs_v0 = &mpi_reqs_[REQ_TOTAL*src_c + REQ_RECV_V0];
		MPI_Request* recv_reqs_v1 = &mpi_reqs_[REQ_TOTAL*src_c + REQ_RECV_V1];

		FoldCommBuffer *rb = get_recv_buffer();

		comm_node.recv_buffer = rb;
		MPI_Irecv(rb->v0_stream, BLOCK_V0_LEGNTH + sizeof(uint32_t),
				MPI_BYTE, src_c, COMM_V0_TAG, mpi.comm_2dr, recv_reqs_v0);
		MPI_Irecv(rb->v1_list, BLOCK_V1_LENGTH,
				MPI_UNSIGNED, src_c, COMM_V1_TAG, mpi.comm_2dr, recv_reqs_v1);

		rb->complete_flag = 0;
	}

	void fold_set_send_buffer(int dest_c)
	{
		FoldNode& node = fold_node_[dest_c];
		FoldNodeComm& comm_node = fold_node_comm_[dest_c];
		FoldCommBuffer* sb = NULL;

		if(comm_node.send_buffer) {
			return ;
		}

		pthread_mutex_lock(&d_->thread_sync_);
		if(listIsEmpty(&node.sending_buffer) == false) {
			sb = CONTAINING_RECORD(node.sending_buffer.fLink, FoldCommBuffer, free_link);
			listRemove(&sb->free_link);
		}
		pthread_mutex_unlock(&d_->thread_sync_);

		if(sb) {
			comm_node.send_buffer = sb;
			int stream_length = offsetof(CompressedStream, index) +
					sizeof(PacketIndex) * (sb->v0_stream->packet_index_start + sb->num_packets);
			MPI_Isend(sb->v0_stream, stream_length,
					MPI_BYTE, dest_c, COMM_V0_TAG, mpi.comm_2dr, &mpi_reqs_[4*dest_c + REQ_SEND_V0]);
			MPI_Isend(sb->v1_list, sb->num_edges,
					MPI_UNSIGNED, dest_c, COMM_V1_TAG, mpi.comm_2dr, &mpi_reqs_[4*dest_c + REQ_SEND_V1]);
			sb->complete_flag = 0;
		}

	}

	void fold_send_submit(int dest_c)
	{
		FoldNode& dest_node = fold_node_[dest_c];
		FoldCommBuffer* sb = CONTAINING_RECORD(dest_node.free_buffer.fLink, FoldCommBuffer, free_link);

		int packet_index_start = get_blocks<sizeof(PacketIndex)>(sb->v0_stream_length);
		sb->v0_stream->packet_index_start = packet_index_start;
		// copy sb_index to tail plist
		memcpy(&sb->v0_stream->index[packet_index_start],
				dest_node.sb_index, sizeof(dest_node.sb_index[0])*sb->num_packets);

		sb->target = dest_c;
	#if 0
		ffprintf(stderr, stderr, "send[from=%d,to=%d,npacket=%d,ptr_packet_index=%d,vlist_length=%d]\n",
				rank, sb->target, sb->npacket, sb->ptr_packet_index, sb->vlist_length);
	#endif

		listRemove(&sb->free_link);

		CommCommand cmd;
		cmd.kind = FOLD_SEND;
		cmd.dest_c = dest_c;
		bool command_active;

		pthread_mutex_lock(&d_->thread_sync_);
		listInsertBack(&dest_node.sending_buffer, &sb->free_link);
		d_->command_queue_.push_back(cmd);
		command_active = d_->command_active_;
		if(command_active == false) d_->command_active_ = true;
		pthread_mutex_unlock(&d_->thread_sync_);

		if(command_active == false) pthread_cond_broadcast(&d_->thread_state_);

#if PRINT_BFS_TIME_DETAIL
	double prev_time = MPI_Wtime();
#endif
		while(listIsEmpty(&dest_node.free_buffer)) {
			pthread_cond_wait(&dest_node.send_state, &dest_node.send_mutex);
		}
#if PRINT_BFS_TIME_DETAIL
	gtl_send_wait += MPI_Wtime() - prev_time;
#endif
	}
};

} // namespace detail {

#define VERTEX_SORTING 1
//#define SHARED_VISITED_DIRECT 1

template <typename IndexArray, typename LocalVertsIndex, typename PARAMS>
class BfsBase
	: private bfs_detail::Bfs2DComm::EventHandler
{
public:
	typedef typename PARAMS::BitmapType BitmapType;
	typedef BfsBase<IndexArray, LocalVertsIndex, PARAMS> ThisType;
	enum {
		// parameters
		// Number of edge lists packed as 1 edge list.
		// Since manipulating 64 bit integer value in NVIDIA GPU is not efficient,
		// max size of this parameter is limited to 32.
		LOG_PACKING_EDGE_LISTS = PARAMS::LOG_PACKING_EDGE_LISTS, // 2^6 = 64
		// Number of CQ bitmap entries represent as 1 bit in summary.
		// Since the type of bitmap entry is int32_t and 1 cache line is composed of 32 bitmap entries,
		// 32 is effective value.
		LOG_CQ_SUMMARIZING = PARAMS::LOG_CQ_SUMMARIZING, // 2^4 = 16 -> sizeof(int64_t)*32 = 128bytes
		ENABLE_WRITING_DEPTH = 1,

		BATCH_PACKETS = 16,

		EXPAND_COMM_THRESOLD_AVG = 50, // 50%
		EXPAND_COMM_THRESOLD_MAX = 75, // 75%
		EXPAND_COMM_DENOMINATOR = 100,

		// non-parameters
		NUMBER_PACKING_EDGE_LISTS = (1 << LOG_PACKING_EDGE_LISTS),
		NUMBER_CQ_SUMMARIZING = (1 << LOG_CQ_SUMMARIZING),

		// "sizeof(BitmapType) * 8" is the number of bits in 1 entry of summary.
		MINIMUN_SIZE_OF_CQ_BITMAP = NUMBER_CQ_SUMMARIZING * sizeof(BitmapType) * 8,
	};

	BfsBase(bool cuda_enabled)
		: comm_(this, cuda_enabled)
#if 0
		, recv_task_(65536)
#endif
		, cq_comm_(this, true)
		, visited_comm_(this, false)
	{
		//
	}

	virtual ~BfsBase()
	{
		//
	}

	template <typename EdgeList>
	void construct(EdgeList* edge_list)
	{
		// minimun requirement of CQ
		// CPU: MINIMUN_SIZE_OF_CQ_BITMAP words -> MINIMUN_SIZE_OF_CQ_BITMAP * NUMBER_PACKING_EDGE_LISTS * mpi.size_2dc
		// GPU: THREADS_PER_BLOCK words -> THREADS_PER_BLOCK * NUMBER_PACKING_EDGE_LISTS * mpi.size_2dc

		if(mpi.isMaster()) {
			fprintf(stderr, "Index Type: %d bit per edge\n", IndexArray::bytes_per_edge * 8);
		}

		int log_min_vertices = get_msb_index(MINIMUN_SIZE_OF_CQ_BITMAP * NUMBER_PACKING_EDGE_LISTS * mpi.size_2dc);
		graph_.log_packing_edge_lists_ = get_msb_index(NUMBER_PACKING_EDGE_LISTS);

		detail::GraphConstructor2DCSR<IndexArray, LocalVertsIndex, EdgeList> constructor;
		constructor.construct(edge_list, log_min_vertices, true /* sorting vertex */, false, graph_);

		log_local_bitmap_ = std::max(0,
				std::max(graph_.log_local_verts() - LOG_PACKING_EDGE_LISTS,
					get_msb_index(MINIMUN_SIZE_OF_CQ_BITMAP) - get_msb_index(mpi.size_2dr)));
	}

	void prepare_bfs() {
		printInformation();
		allocate_memory(graph_);
	}

	void run_bfs(int64_t root, int64_t* pred);

	void get_pred(int64_t* pred) { }

	void end_bfs() {
		deallocate_memory();
	}

	Graph2DCSR<IndexArray, LocalVertsIndex> graph_;

// protected:
	struct ThreadLocalBuffer {
		uint8_t stream_buffer[BFS_PARAMS::PACKET_LENGTH*7*BATCH_PACKETS]; // 48bit per vertex
		int64_t decode_buffer[BFS_PARAMS::PACKET_LENGTH*(BATCH_PACKETS+1)];
		int64_t num_nq_vertices;
		bfs_detail::PacketIndex packet_index[BATCH_PACKETS];
		bfs_detail::FoldPacket fold_packet[1];
	};

	union ExpandCommBuffer {
		BitmapType bitmap[1];
		bfs_detail::CompressedStream stream;
	};

	int64_t get_bitmap_size_v0() const {
		return (int64_t(1) << log_local_bitmap_) * mpi.size_2dr;
	}
	int64_t get_summary_size_v0() const {
		return get_bitmap_size_v0() / MINIMUN_SIZE_OF_CQ_BITMAP;
	}
	int64_t get_bitmap_size_v1() const {
		return (int64_t(1) << log_local_bitmap_) * mpi.size_2dc;
	}
	int64_t get_bitmap_size_visited() const {
		return (int64_t(1) << log_local_bitmap_);
	}
	int64_t get_number_of_local_vertices() const {
		return (int64_t(1) << graph_.log_local_verts());
	}
	int64_t get_actual_number_of_local_vertices() const {
		return (int64_t(1) << graph_.log_actual_local_verts());
	}

	// virtual functions
	virtual int varint_encode(const int64_t* input, int length, uint8_t* output, bfs_detail::VARINT_BFS_KIND kind)
	{
		return varint_encode_stream((const uint64_t*)input, length, output);
	}

	int64_t get_nq_threshold()
	{
		return sizeof(BitmapType) * graph_.get_bitmap_size_visited() -
				sizeof(bfs_detail::PacketIndex) * (omp_get_max_threads()*2 + 16);
	}

	void allocate_memory(Graph2DCSR<IndexArray, LocalVertsIndex>& g)
	{
		const int max_threads = omp_get_max_threads();
		cq_bitmap_ = (BitmapType*)
				page_aligned_xcalloc(sizeof(cq_bitmap_[0])*get_bitmap_size_v0());
		cq_summary_ = (BitmapType*)
				malloc(sizeof(cq_summary_[0])*get_summary_size_v0());
		shared_visited_ = (BitmapType*)
				page_aligned_xcalloc(sizeof(shared_visited_[0])*get_bitmap_size_v1());
		nq_bitmap_ = (BitmapType*)
				page_aligned_xcalloc(sizeof(nq_bitmap_[0])*get_bitmap_size_visited());
		nq_sorted_bitmap_ = (BitmapType*)
				page_aligned_xcalloc(sizeof(nq_bitmap_[0])*get_bitmap_size_visited());
		visited_ = (BitmapType*)
				page_aligned_xcalloc(sizeof(visited_[0])*get_bitmap_size_visited());

		tmp_packet_max_length_ = sizeof(BitmapType) *
				get_bitmap_size_visited() / BFS_PARAMS::PACKET_LENGTH + omp_get_max_threads()*2;
		tmp_packet_index_ = (bfs_detail::PacketIndex*)
				malloc(sizeof(bfs_detail::PacketIndex)*tmp_packet_max_length_);
		cq_comm_.local_buffer_ = (bfs_detail::CompressedStream*)
				page_aligned_xcalloc(sizeof(BitmapType)*get_bitmap_size_visited());
#if VERTEX_SORTING
		visited_comm_.local_buffer_ = (bfs_detail::CompressedStream*)
				page_aligned_xcalloc(sizeof(BitmapType)*get_bitmap_size_visited());
#else
		visited_comm_.local_buffer_ = cq_comm_.local_buffer_;
#endif
		cq_comm_.recv_buffer_ = (ExpandCommBuffer*)
				page_aligned_xcalloc(sizeof(BitmapType)*get_bitmap_size_v0());
		visited_comm_.recv_buffer_ = (ExpandCommBuffer*)
				page_aligned_xcalloc(sizeof(BitmapType)*get_bitmap_size_v1());

		thread_local_buffer_ = (ThreadLocalBuffer**)
				malloc(sizeof(thread_local_buffer_[0])*max_threads);
		d_ = (DynamicDataSet*)malloc(sizeof(d_[0]));

		const int buffer_width = roundup<CACHE_LINE>(
				sizeof(ThreadLocalBuffer) + sizeof(bfs_detail::FoldPacket) * mpi.size_2dc);
		buffer_.thread_local_ = cache_aligned_xcalloc(buffer_width*max_threads);
		for(int i = 0; i < max_threads; ++i) {
			thread_local_buffer_[i] = (ThreadLocalBuffer*)
					((uint8_t*)buffer_.thread_local_ + buffer_width*i);
		}

		sched_.long_job_length = std::max(1, std::min<int>(get_summary_size_v0() / 8, 2048)); // 32KB chunk of CQ
		sched_.short_job_length = std::max(1, std::min<int>(get_summary_size_v0() / (8*16), 128)); // 512KB chunk of CQ
		sched_.job_array = new ExtractEdge[sched_.long_job_length + sched_.short_job_length];
		sched_.long_job = sched_.job_array;
		sched_.short_job = sched_.job_array + sched_.long_job_length;
		int64_t long_job_chunk = (get_summary_size_v0() + sched_.long_job_length - 1) / sched_.long_job_length;
		int64_t shor_job_chunk = (get_summary_size_v0() + sched_.short_job_length - 1) / sched_.short_job_length;
		int64_t offset = 0;
		for(int i = 0; i < sched_.long_job_length; ++i) {
			sched_.long_job[i].this_ = this;
			sched_.long_job[i].i_start_ = offset;
			assert (offset < get_summary_size_v0());
			offset += long_job_chunk;
			sched_.long_job[i].i_end_ = std::min(offset, get_summary_size_v0());
		}
		assert (offset >= get_summary_size_v0());
		offset = 0;
		for(int i = 0; i < sched_.short_job_length; ++i) {
			sched_.short_job[i].this_ = this;
			sched_.short_job[i].i_start_ = offset;
			assert (offset < get_summary_size_v0());
			offset += shor_job_chunk;
			sched_.short_job[i].i_end_ = offset;
		}
		assert (offset >= get_summary_size_v0());

		sched_.fold_end_job = new ExtractEnd[mpi.size_2dc];
		for(int i = 0; i < mpi.size_2dc; ++i) {
			sched_.fold_end_job[i].this_ = this;
			sched_.fold_end_job[i].dest_c_ = i;
		}

#if 0
		num_recv_tasks_ = max_threads * 3;
		for(int i = 0; i < num_recv_tasks_; ++i) {
			recv_task_.push(new ReceiverProcessing(this));
		}
#endif
	}

	void deallocate_memory()
	{
		free(cq_bitmap_); cq_bitmap_ = NULL;
		free(cq_summary_); cq_summary_ = NULL;
		free(shared_visited_); shared_visited_ = NULL;
		free(nq_bitmap_); nq_bitmap_ = NULL;
		free(nq_sorted_bitmap_); nq_sorted_bitmap_ = NULL;
		free(visited_); visited_ = NULL;
		free(tmp_packet_index_); tmp_packet_index_ = NULL;
		free(cq_comm_.local_buffer_); cq_comm_.local_buffer_ = NULL;
#if VERTEX_SORTING
		free(visited_comm_.local_buffer_); visited_comm_.local_buffer_ = NULL;
#endif
		free(cq_comm_.recv_buffer_); cq_comm_.recv_buffer_ = NULL;
		free(visited_comm_.recv_buffer_); visited_comm_.recv_buffer_ = NULL;
		free(thread_local_buffer_); thread_local_buffer_ = NULL;
		free(d_); d_ = NULL;
		delete [] sched_.job_array; sched_.job_array = NULL;
		delete [] sched_.fold_end_job; sched_.fold_end_job = NULL;
		free(buffer_.thread_local_); buffer_.thread_local_ = NULL;
#if 0
		for(int i = 0; i < num_recv_tasks_; ++i) {
			delete recv_task_.pop();
		}
#endif
	}

	void initialize_memory(int64_t* pred)
	{
		using namespace BFS_PARAMS;
		const int64_t num_local_vertices = get_actual_number_of_local_vertices();
		const int64_t bitmap_size_visited = get_bitmap_size_visited();
		const int64_t bitmap_size_v1 = get_bitmap_size_v1();
		const int64_t summary_size = get_summary_size_v0();

		BitmapType* cq_bitmap = cq_bitmap_;
		BitmapType* cq_summary = cq_summary_;
		BitmapType* nq_bitmap = nq_bitmap_;
#if VERTEX_SORTING
		BitmapType* nq_sorted_bitmap = nq_sorted_bitmap_;
#endif
		BitmapType* visited = visited_;
		BitmapType* shared_visited = shared_visited_;

#pragma omp parallel
		{
#if 1	// Only Spec2010 needs this initialization
#pragma omp for nowait
			for(int64_t i = 0; i < num_local_vertices; ++i) {
				pred[i] = -1;
			}
#endif
			// clear NQ and visited
#pragma omp for nowait
			for(int64_t i = 0; i < bitmap_size_visited; ++i) {
				nq_bitmap[i] = 0;
#if VERTEX_SORTING
				nq_sorted_bitmap[i] = 0;
#endif
				visited[i] = 0;
			}
			// clear CQ and CQ summary
#pragma omp for nowait
			for(int64_t i = 0; i < summary_size; ++i) {
				cq_summary[i] = 0;
				for(int k = 0; k < MINIMUN_SIZE_OF_CQ_BITMAP; ++k) {
					cq_bitmap[i*MINIMUN_SIZE_OF_CQ_BITMAP + k] = 0;
				}
			}
			// clear shared visited
#pragma omp for nowait
			for(int64_t i = 0; i < bitmap_size_v1; ++i) {
				shared_visited[i] = 0;
			}

			// clear fold packet buffer
			bfs_detail::FoldPacket* packet_array = thread_local_buffer_[omp_get_thread_num()]->fold_packet;
			for(int i = 0; i < mpi.size_2dc; ++i) {
				packet_array[i].num_edges = 0;
			}
		}
	}

	int64_t get_compressed_stream_length(int64_t num_packets, int64_t content_length)
	{
		return num_packets*sizeof(bfs_detail::PacketIndex) +
				roundup<sizeof(bfs_detail::PacketIndex)>(content_length) +
				offsetof(bfs_detail::CompressedStream, stream);
	}

	bool expand_batch_wrtie_packet(int64_t* vertex_buffer, int num_vertices,
			int64_t threshold, bfs_detail::CompressedStream* output, bool cq_or_visited)
	{
		using namespace BFS_PARAMS;
		using namespace bfs_detail;

		PacketIndex* local_packet_index = thread_local_buffer_[omp_get_thread_num()]->packet_index;
		uint8_t* local_stream_buffer = thread_local_buffer_[omp_get_thread_num()]->stream_buffer;
		// encode using varint
		int stream_buffer_offset = 0;
		int num_packets = get_blocks<PACKET_LENGTH>(num_vertices);
		for(int k = 0; k < num_packets; ++k) {
			int num_vertices_in_this_packet = std::min<int>(PACKET_LENGTH, num_vertices - k*PACKET_LENGTH);
			int64_t* vertex_buffer_k = vertex_buffer + k*PACKET_LENGTH;
			// compute differentials
#ifndef NDEBUG
			int64_t num_local_vertices = get_actual_number_of_local_vertices();
			for(int64_t r = 0; r < num_vertices_in_this_packet; ++r) {
				assert (vertex_buffer_k[r] >= 0);
				assert (vertex_buffer_k[r] < num_local_vertices);
			}
#endif
			for(int r = num_vertices_in_this_packet - 1; r > 0; --r) {
				vertex_buffer_k[r] -= vertex_buffer_k[r-1];
				assert (vertex_buffer_k[r] >= 0);
			}
			int length = varint_encode(vertex_buffer_k,
					num_vertices_in_this_packet, local_stream_buffer + stream_buffer_offset,
					cq_or_visited ? bfs_detail::VARINT_EXPAND_CQ : bfs_detail::VARINT_EXPAND_SV);
			local_packet_index[k].length = length;
			local_packet_index[k].num_int = num_vertices_in_this_packet;
			stream_buffer_offset += length;
		}
		// store to global buffer
		int64_t copy_packet_offset;
		int64_t copy_offset;
		int64_t total_num_packet;
		int64_t total_stream_length;
#pragma omp critical
		{
			copy_packet_offset = d_->num_tmp_packets_;
			copy_offset = d_->tmp_packet_offset_;
			total_num_packet = copy_packet_offset + num_packets;
			total_stream_length = copy_offset + stream_buffer_offset;
			d_->num_tmp_packets_ = total_num_packet;
			d_->tmp_packet_offset_ = total_stream_length;
		}
		if(get_compressed_stream_length
				(total_num_packet, total_stream_length) >= threshold)
		{
			return false;
		}
		memcpy(tmp_packet_index_ + copy_packet_offset,
				local_packet_index, sizeof(PacketIndex)*num_packets);
		memcpy(output->stream + copy_offset,
				local_stream_buffer, sizeof(local_stream_buffer[0]) * stream_buffer_offset);
		return true;
	}

	// return : true if creating succeeded,
	// 			false if stream length exceeded the length of bitmap
	bool expand_create_stream(const BitmapType* bitmap, bfs_detail::CompressedStream* output, int64_t* data_size, bool cq_or_visited)
	{
		using namespace BFS_PARAMS;
		using namespace bfs_detail;

		d_->num_tmp_packets_ = 0;
		d_->tmp_packet_offset_ = 0;

		const int64_t bitmap_size = get_bitmap_size_visited();
		const int64_t threshold = bitmap_size * sizeof(BitmapType);
		bool b_break = false;

#pragma omp parallel reduction(|:b_break)
		{
			int64_t* local_buffer = thread_local_buffer_[omp_get_thread_num()]->decode_buffer;
			int local_buffer_offset = 0;

			int64_t chunk_size = (bitmap_size + omp_get_num_threads() - 1) / omp_get_num_threads();
			int64_t i_start = chunk_size * omp_get_thread_num();
			int64_t i_end = std::min(i_start + chunk_size, bitmap_size);

			for(int64_t i = i_start; i < i_end; ++i) {
				const BitmapType bitmap_i = bitmap[i];
				if(bitmap_i == 0) {
					continue;
				}
				// read bitmap and store to temporary buffer
				for(int bit_idx = 0; bit_idx < NUMBER_PACKING_EDGE_LISTS; ++bit_idx) {
					if(bitmap_i & (int64_t(1) << bit_idx)) {
						local_buffer[local_buffer_offset++] = i*NUMBER_PACKING_EDGE_LISTS + bit_idx;
					}
				}

				if(local_buffer_offset >= PACKET_LENGTH*BATCH_PACKETS)
				{
					if(expand_batch_wrtie_packet(local_buffer,
							PACKET_LENGTH*BATCH_PACKETS, threshold, output, cq_or_visited) == false)
					{
						b_break = true;
						break;
					}
					local_buffer_offset -= PACKET_LENGTH*BATCH_PACKETS;
					memcpy(local_buffer, local_buffer + PACKET_LENGTH*BATCH_PACKETS,
							sizeof(local_buffer[0]) * local_buffer_offset);
				}
			}
			// flush local buffer
			if(b_break == false) {
				if(expand_batch_wrtie_packet(local_buffer,
						local_buffer_offset, threshold, output, cq_or_visited) == false)
				{
					b_break = true;
				}
			}
		} // #pragma omp parallel reduction(|:b_break)

		if(b_break) {
			*data_size = threshold;
			return false;
		}

		// write packet index
		int packet_index_start = get_blocks<sizeof(PacketIndex)>(d_->tmp_packet_offset_);
		output->packet_index_start = packet_index_start;
		// copy sb_index to tail plist
		memcpy(&output->index[packet_index_start],
				tmp_packet_index_, sizeof(tmp_packet_index_[0])*d_->num_tmp_packets_);

		*data_size = get_compressed_stream_length(d_->num_tmp_packets_, d_->tmp_packet_offset_);
		return true;
	}

	struct ExpandCommCommand
		: public bfs_detail::Bfs2DComm::Communicatable
		, public Runnable
	{
		ExpandCommCommand(ThisType* this__, bool cq_or_visited)
			: this_(this__)
			, cq_or_visited_(cq_or_visited)
		{
			if(cq_or_visited) {
				// current queue
				comm_ = mpi.comm_2dc;
				comm_size_ = mpi.size_2dr;
			}
			else {
				// visited
				comm_ = mpi.comm_2dr;
				comm_size_ = mpi.size_2dc;
			}
			count_ = (int*)malloc(sizeof(int)*2*(comm_size_+1));
			offset_ = count_ + comm_size_+1;
		}
		~ExpandCommCommand()
		{
			free(count_); count_ = NULL;
		}
		void start_comm(LocalVertsIndex local_data_size, bool force_bitmap, bool exit_fiber_proc)
		{
			exit_fiber_proc_ = exit_fiber_proc;
			force_bitmap_ = force_bitmap;
			local_size_ = local_data_size;
			this_->comm_.input_command(this);
		}

		bfs_detail::CompressedStream* local_buffer_; // allocated by parent
		ExpandCommBuffer* recv_buffer_; // allocated by parent

	protected:
		void gather_comm()
		{
#if VERVOSE_MODE
			profiling::TimeKeeper tk;
#endif
			stream_or_bitmap_ = false;
			if(force_bitmap_ == false) {
				LocalVertsIndex max_size;
				int64_t bitmap_size_in_bytes = this_->get_bitmap_size_visited() * sizeof(BitmapType);
				int64_t threshold = int64_t((double)bitmap_size_in_bytes *
						(double)EXPAND_COMM_THRESOLD_MAX / (double)EXPAND_COMM_DENOMINATOR);
#if VERVOSE_MODE
				tk.getSpanAndReset();
#endif
				MPI_Allreduce(&local_size_, &max_size, 1,
						get_mpi_type(local_size_), MPI_MAX, comm_);
#if VERVOSE_MODE
				tk.submit("expand comm allreduce", this_->current_level_, cq_or_visited_ ? 0 : 1);
#endif
				if((int64_t)max_size <= threshold) {
					stream_or_bitmap_ = true;
				}
			}
			if(stream_or_bitmap_) {
				// transfer compressed stream
#if VERVOSE_MODE
				tk.getSpanAndReset();
#endif
				MPI_Allgather(&local_size_, 1, get_mpi_type(local_size_),
						count_, 1, get_mpi_type(count_[0]), comm_);
#if VERVOSE_MODE
				tk.submit("expand comm stream allgather", this_->current_level_, cq_or_visited_ ? 0 : 1);
#endif
				offset_[0] = 0;
				for(int i = 0; i < comm_size_; ++i) {
					// take care of alignment !!!
					offset_[i+1] = roundup<sizeof(bfs_detail::PacketIndex)>
									(offset_[i] + count_[i]);
				}
#if VERVOSE_MODE
				tk.getSpanAndReset();
#endif
				MPI_Allgatherv(local_buffer_, local_size_, MPI_BYTE,
						recv_buffer_, count_, offset_, MPI_BYTE, comm_);
#if VERVOSE_MODE
				tk.submit("expand comm stream allgatherv", this_->current_level_, cq_or_visited_ ? 0 : 1);
#endif
#if VERVOSE_MODE
				g_exs_send += local_size_;
				g_exs_recv += offset_[comm_size_];

				int64_t s_size = offset_[comm_size_], r_size;
				MPI_Allreduce(&s_size, &r_size, 1, MpiTypeOf<int64_t>::type, MPI_SUM, mpi.comm_2d);
				if(mpi.isMaster()) {
					fprintf(stderr, "Complessed List Received: %ld\n", r_size);
				}
#endif
			}
			else {
				// transfer bitmap
				BitmapType* bitmap;
				void* recv_buffer;
				if(cq_or_visited_) {
					// current queue
					bitmap = this_->nq_bitmap_;
					recv_buffer = this_->cq_bitmap_;
				}
				else {
					// visited
					bitmap = this_->visited_;
#if SHARED_VISITED_STRIPE
					recv_buffer = recv_buffer_;
#else
					recv_buffer = this_->shared_visited_;
#endif
				}
				int bitmap_size = this_->get_bitmap_size_visited();
#if VERVOSE_MODE
				tk.getSpanAndReset();
#endif
				// transfer bitmap
				MPI_Allgather(bitmap, bitmap_size, get_mpi_type(bitmap[0]),
						recv_buffer, bitmap_size, get_mpi_type(bitmap[0]), comm_);
#if VERVOSE_MODE
				tk.submit("expand comm bitmap allgather", this_->current_level_, cq_or_visited_ ? 0 : 1);
				g_bitmap_send += bitmap_size * sizeof(BitmapType);
				g_bitmap_recv += bitmap_size * comm_size_* sizeof(BitmapType);
#endif
			}
		}

		virtual void comm() {

			gather_comm();

			this_->fiber_man_.submit(this, 0);
			if(exit_fiber_proc_) {
				this_->fiber_man_.end_processing();
			}
		}

		virtual void run() {
			const LocalVertsIndex local_bitmap_size = this_->get_bitmap_size_visited();

			if(cq_or_visited_) {
				// current queue
#if VERTEX_SORTING
				// clear NQ
				BitmapType* nq_bitmap = this_->nq_bitmap_;
#pragma omp parallel for
				for(int64_t i = 0; i < local_bitmap_size; ++i) {
					nq_bitmap[i] = 0;
				}
#endif
				if(stream_or_bitmap_) {
					// stream
					const LocalVertsIndex bitmap_size = this_->get_bitmap_size_v0();
					BitmapType* cq_bitmap = this_->cq_bitmap_;
#if 0
					// initialize CQ bitmap
#pragma omp parallel for
					for(LocalVertsIndex i = 0; i < bitmap_size; ++i) {
						cq_bitmap[i] = 0;
					}
#endif
					// update
					this_->d_->num_vertices_in_cq_ = this_->update_from_stream(
							&recv_buffer_->stream, offset_, count_,
							comm_size_, cq_bitmap, bitmap_size, this_->cq_summary_);
				}
				else {
					// bitmap
					this_->d_->num_vertices_in_cq_ = this_->get_bitmap_size_v0() * NUMBER_PACKING_EDGE_LISTS;
					// fill 1 to summary
					memset(this_->cq_summary_, -1,
							sizeof(BitmapType) * this_->get_summary_size_v0());
				}
			}
			else {
				// shared visited
				// clear NQ
#if VERTEX_SORTING
				BitmapType* nq_bitmap = this_->nq_sorted_bitmap_;
#else
				BitmapType* nq_bitmap = this_->nq_bitmap_;
#endif
#pragma omp parallel for
				for(int64_t i = 0; i < local_bitmap_size; ++i) {
					nq_bitmap[i] = 0;
				}

				if(stream_or_bitmap_) {
					// stream
					this_->update_from_stream(&recv_buffer_->stream, offset_, count_,
							comm_size_, this_->shared_visited_, this_->get_bitmap_size_v1(), NULL);
				}
				else {
#if SHARED_VISITED_STRIPE
					// bitmap
					this_->update_shared_visited_from_bitmap(recv_buffer_->bitmap);
#endif
				}
				this_->check_shared_visited();
			}
		}

		ThisType* const this_;
		const bool cq_or_visited_;
		MPI_Comm comm_;
		int comm_size_;
		bool exit_fiber_proc_;
		bool force_bitmap_;
		bool stream_or_bitmap_;
		int* count_;
		int* offset_;
		LocalVertsIndex local_size_;
	};

	struct UpdatePacketIndex {
		LocalVertsIndex offset;
		int16_t length;
		int16_t src_num;
#ifndef NDEBUG
		int16_t num_vertices;
#endif
	};

	void check_shared_visited()
	{
#ifndef NDEBUG
		int64_t num_local_vertices = get_actual_number_of_local_vertices();
#if SHARED_VISITED_STRIPE
		const LocalVertsIndex src_num_factor = LocalVertsIndex(mpi.rank_2dc)*NUMBER_CQ_SUMMARIZING*NUMBER_PACKING_EDGE_LISTS;
		const int log_size_c = get_msb_index(mpi.size_2dc);
		const LocalVertsIndex mask2 = LocalVertsIndex(NUMBER_CQ_SUMMARIZING*NUMBER_PACKING_EDGE_LISTS) - 1;
		const LocalVertsIndex mask1 = get_number_of_local_vertices() - 1 - mask2;
#else
		int64_t word_idx_base = get_bitmap_size_visited() * mpi.rank_2dc;
#endif
		for(int64_t i = 0; i < num_local_vertices; ++i) {
#if VERTEX_SORTING
			int64_t v = graph_.vertex_mapping_[i];
#else
			int64_t v = i;
#endif
#if SHARED_VISITED_STRIPE
			LocalVertsIndex sv_idx = src_num_factor | ((v & mask1) << log_size_c) | (v & mask2);
			int64_t word_idx = sv_idx / NUMBER_PACKING_EDGE_LISTS;
#else
			int64_t word_idx = word_idx_base + v / NUMBER_PACKING_EDGE_LISTS;
#endif
			int bit_idx = v % NUMBER_PACKING_EDGE_LISTS;
			bool visited = (shared_visited_[word_idx] & (BitmapType(1) << bit_idx)) != 0;
			assert((visited && (pred_[i] != -1)) || (!visited && (pred_[i] == -1)));
		}
#endif
	}

	// return number of vertices received
	int64_t update_from_stream(bfs_detail::CompressedStream* stream, int* offset, int* count,
			int num_src, BitmapType* target, LocalVertsIndex length, BitmapType* summary)
	{
		LocalVertsIndex packet_count[num_src], packet_offset[num_src+1];
		packet_offset[0] = 0;
		const uint8_t* byte_stream = (uint8_t*)stream;

		// compute number of packets
		for(int i = 0; i < num_src; ++i) {
			int index_start = ((bfs_detail::CompressedStream*)(byte_stream + offset[i]))->packet_index_start;
			packet_count[i] = (count[i] - index_start * sizeof(bfs_detail::PacketIndex) -
					offsetof(bfs_detail::CompressedStream, index)) / sizeof(bfs_detail::PacketIndex);
			packet_offset[i+1] = packet_offset[i] + packet_count[i];
		}

		// make a list of all packets
		LocalVertsIndex num_total_packets = packet_offset[num_src];
		UpdatePacketIndex* index = new UpdatePacketIndex[num_total_packets];
		int64_t num_vertices_received = 0;

#pragma omp parallel if(offset[num_src] > 4096), reduction(+:num_vertices_received)
		{
#pragma omp for
			for(int i = 0; i < num_src; ++i) {
				bfs_detail::CompressedStream* stream = (bfs_detail::CompressedStream*)(byte_stream + offset[i]);
				LocalVertsIndex base = packet_offset[i];
				bfs_detail::PacketIndex* packet_index = &stream->index[stream->packet_index_start];
				LocalVertsIndex stream_offset = offset[i] + offsetof(bfs_detail::CompressedStream, stream);

				for(int64_t k = 0; k < packet_count[i]; ++k) {
					index[base + k].offset = stream_offset;
					index[base + k].length = packet_index[k].length;
					index[base + k].src_num = i;
#ifndef NDEBUG
					index[base + k].num_vertices = packet_index[k].num_int;
#endif
					num_vertices_received += packet_index[k].num_int;
					stream_offset += packet_index[k].length;
				}
				assert((roundup<sizeof(bfs_detail::PacketIndex)>(
						(stream_offset - offset[i] - offsetof(bfs_detail::CompressedStream, stream)))
						/ sizeof(bfs_detail::PacketIndex)) == stream->packet_index_start);
			}

			int64_t* decode_buffer = thread_local_buffer_[omp_get_thread_num()]->decode_buffer;

			// update bitmap
#pragma omp for
			for(int64_t i = 0; i < (int64_t)num_total_packets; ++i) {
				int num_vertices = varint_decode_stream(byte_stream + index[i].offset,
						index[i].length, (uint64_t*)decode_buffer);
#ifndef NDEBUG
				assert (num_vertices == index[i].num_vertices);
				int64_t num_local_vertices = get_actual_number_of_local_vertices();
				int64_t dbg_vertices_id = 0;
				for(int64_t r = 0; r < num_vertices; ++r) {
					assert (decode_buffer[r] >= 0);
					assert (decode_buffer[r] < num_local_vertices);
					dbg_vertices_id += decode_buffer[r];
					assert (dbg_vertices_id < num_local_vertices);
				}
#endif
				if(summary) {
					// CQ
					LocalVertsIndex v_swizzled = (LocalVertsIndex(1) << log_local_bitmap_) * NUMBER_PACKING_EDGE_LISTS * index[i].src_num;
					for(int k = 0; k < num_vertices; ++k) {
						v_swizzled += decode_buffer[k];
						LocalVertsIndex word_idx = v_swizzled / NUMBER_PACKING_EDGE_LISTS;
						int bit_idx = v_swizzled % NUMBER_PACKING_EDGE_LISTS;
						BitmapType mask = BitmapType(1) << bit_idx;
						const BitmapType fetch_result = __sync_fetch_and_or(&target[word_idx], mask);
						if(fetch_result == 0) {
							const int64_t bit_offset = word_idx / NUMBER_CQ_SUMMARIZING;
							const int64_t summary_word_idx = bit_offset / (sizeof(summary[0]) * 8);
							const int64_t summary_bit_idx = bit_offset % (sizeof(summary[0]) * 8);
							BitmapType summary_mask = BitmapType(1) << summary_bit_idx;

							if((summary[summary_word_idx] & summary_mask) == 0) {
								__sync_fetch_and_or(&summary[summary_word_idx], summary_mask);
							}
						}
					}
				}
				else {
					// shared visited
#if SHARED_VISITED_STRIPE
					//const int log_summarizing_verts = LOG_CQ_SUMMARIZING + LOG_PACKING_EDGE_LISTS;
					const LocalVertsIndex src_num_factor = LocalVertsIndex(index[i].src_num)*NUMBER_CQ_SUMMARIZING*NUMBER_PACKING_EDGE_LISTS;
					const int log_size_c = get_msb_index(mpi.size_2dc);
					const LocalVertsIndex mask2 = LocalVertsIndex(NUMBER_CQ_SUMMARIZING*NUMBER_PACKING_EDGE_LISTS) - 1;
					const LocalVertsIndex mask1 = get_number_of_local_vertices() - 1 - mask2;
					LocalVertsIndex v_swizzled = 0;
#else
					LocalVertsIndex v_swizzled = (LocalVertsIndex(1) << log_local_bitmap_) * NUMBER_PACKING_EDGE_LISTS * index[i].src_num;
#endif

					for(int k = 0; k < num_vertices; ++k) {
						v_swizzled += decode_buffer[k];
#if SHARED_VISITED_STRIPE
						LocalVertsIndex sv_idx = src_num_factor | ((v_swizzled & mask1) << log_size_c) | (v_swizzled & mask2);
#else
						LocalVertsIndex sv_idx = v_swizzled;
#endif
						LocalVertsIndex word_idx = sv_idx / NUMBER_PACKING_EDGE_LISTS;
						int bit_idx = sv_idx % NUMBER_PACKING_EDGE_LISTS;
						BitmapType mask = BitmapType(1) << bit_idx;
				//		if((target[word_idx] & mask) == 0) {
							__sync_fetch_and_or(&target[word_idx], mask);
#ifndef NDEBUG
							LocalVertsIndex base = LocalVertsIndex(1) << log_local_bitmap_;
							if(index[i].src_num == mpi.rank_2dc) {
								LocalVertsIndex local_word_idx = word_idx - base * index[i].src_num;
								assert((visited_[local_word_idx] & mask) != 0);
							}
							else {
								assert ((word_idx < (base * mpi.rank_2dc)) || (word_idx >= (base * (mpi.rank_2dc + 1))));
							}
#endif
				//		}
					}
				}
			}
		} // #pragma omp parallel

		delete [] index;
		return num_vertices_received;
	}

	void update_shared_visited_from_bitmap(BitmapType* source)
	{
		// shared visited
		BitmapType* shared_visited = this->shared_visited_;
	//	const int64_t bitmap_size = this->get_bitmap_size_v1();
		const int64_t local_bitmap_size_in_lines = get_bitmap_size_visited() >> LOG_CQ_SUMMARIZING;

#pragma omp parallel for
		for(int i = 0; i < local_bitmap_size_in_lines; ++i) {
			for(int k = 0; k < mpi.size_2dc; ++k) {
				// copy cache line
				BitmapType* src = source + NUMBER_CQ_SUMMARIZING * (local_bitmap_size_in_lines * k + i);
				BitmapType* dst = shared_visited + NUMBER_CQ_SUMMARIZING * (mpi.size_2dc * i + k);
				for(int r = 0; r < NUMBER_CQ_SUMMARIZING; ++r) {
					dst[r] = src[r];
				}
			}
		}
	}

	void expand(int64_t global_nq_vertices, ExpandCommCommand* ex_cq, ExpandCommCommand* ex_vi)
	{
		int64_t global_bitmap_size_in_bytes =
				(int64_t(1) << (graph_.log_global_verts() - LOG_PACKING_EDGE_LISTS)) * sizeof(BitmapType);
		int64_t threshold = int64_t((double)global_bitmap_size_in_bytes *
				(double)EXPAND_COMM_THRESOLD_AVG / (double)EXPAND_COMM_DENOMINATOR);
		int64_t local_data_size;

#if VERVOSE_MODE
		profiling::TimeKeeper tk;
#endif
		fiber_man_.begin_processing();
		if(global_nq_vertices > threshold) {
#if VERVOSE_MODE
			if(mpi.isMaster()) fprintf(stderr, "Expand using Bitmap.\n");
#endif
			ex_cq->start_comm(0, true, false);
			ex_vi->start_comm(0, true, true);
		}
		else {
			expand_create_stream(nq_bitmap_, ex_cq->local_buffer_, &local_data_size, true);
			ex_cq->start_comm(local_data_size, false, false);
#if VERTEX_SORTING
			expand_create_stream(nq_sorted_bitmap_, ex_vi->local_buffer_, &local_data_size, false);
#endif
			ex_vi->start_comm(local_data_size, false, true);
		}
#if VERVOSE_MODE
		tk.submit("expand send cpu", current_level_);
#endif
		fiber_man_.enter_processing();
	}

	void expand_root(int64_t root_local, ExpandCommCommand* ex_cq, ExpandCommCommand* ex_vi)
	{
		using namespace BFS_PARAMS;
		fiber_man_.begin_processing();
		if(root_local != -1) {
#if VERTEX_SORTING
			int64_t sortd_root_local = graph_.vertex_mapping_[root_local];
#else
			int64_tsortd_root_local = root_local;
#endif

			int stream_length = varint_encode(&root_local, 1, ex_cq->local_buffer_->stream, bfs_detail::VARINT_EXPAND_CQ);
			int packet_index_start = get_blocks<sizeof(bfs_detail::PacketIndex)>(stream_length);
			ex_cq->local_buffer_->packet_index_start = packet_index_start;
			ex_cq->local_buffer_->index[packet_index_start].length = stream_length;
			ex_cq->local_buffer_->index[packet_index_start].num_int = 1;
			LocalVertsIndex cq_stream_size = get_compressed_stream_length(1, stream_length);
			ex_cq->start_comm(cq_stream_size, false, false);

#if VERTEX_SORTING
			stream_length = varint_encode(&sortd_root_local, 1, ex_vi->local_buffer_->stream, bfs_detail::VARINT_EXPAND_SV);
			packet_index_start = get_blocks<sizeof(bfs_detail::PacketIndex)>(stream_length);
			ex_vi->local_buffer_->packet_index_start = packet_index_start;
			ex_vi->local_buffer_->index[packet_index_start].length = stream_length;
			ex_vi->local_buffer_->index[packet_index_start].num_int = 1;
			ex_vi->start_comm(get_compressed_stream_length(1, stream_length), false, true);
#else
			ex_vi->start_comm(cq_stream_size, false, true);
#endif
		}
		else {
			ex_cq->local_buffer_->packet_index_start = 0;
			LocalVertsIndex stream_size = get_compressed_stream_length(0, 0);
			ex_cq->start_comm(stream_size, false, false);
#if VERTEX_SORTING
			ex_vi->local_buffer_->packet_index_start = 0;
#endif
			ex_vi->start_comm(stream_size, false, true);
		}
		fiber_man_.enter_processing();
	}

	void fold_send_packet(bfs_detail::FoldPacket* packet, int dest_c)
	{
		for(int r = packet->num_edges - 1; r > 0; --r) {
			packet->v0_list[r] -= packet->v0_list[r-1];
//			assert(packet->v0_list[r] >= 0);
		}
		uint8_t* stream_buffer = thread_local_buffer_[omp_get_thread_num()]->stream_buffer;
		int stream_length = varint_encode(packet->v0_list, packet->num_edges, stream_buffer, bfs_detail::VARINT_FOLD);
		comm_.fold_send(stream_buffer, packet->v1_list, stream_length, packet->num_edges, dest_c);
	}

	struct ExtractEdge : public Runnable {
		virtual void run() {
			using namespace BFS_PARAMS;
			BitmapType* cq_bitmap = this_->cq_bitmap_;
			BitmapType* cq_summary = this_->cq_summary_;
			BitmapType* shared_visited = this_->shared_visited_;
			const int64_t* row_starts = this_->graph_.row_starts_;
			const IndexArray& index_array = this_->graph_.index_array_;
			const int log_local_verts = this_->graph_.log_local_verts();
			const int64_t local_verts_mask = this_->get_number_of_local_vertices() - 1;
			const int64_t threshold = this_->get_number_of_local_vertices() / BFS_PARAMS::DENOM_SHARED_VISITED_PART;
			int64_t i_start = i_start_, i_end = i_end_;
			bfs_detail::FoldPacket* packet_array =
					this_->thread_local_buffer_[omp_get_thread_num()]->fold_packet;

#if SHARED_VISITED_STRIPE
			const int log_size_c = get_msb_index(mpi.size_2dc);
			const int64_t mask2 = LocalVertsIndex(NUMBER_CQ_SUMMARIZING*NUMBER_PACKING_EDGE_LISTS) - 1;
			const int64_t mask1 = this_->get_number_of_local_vertices() - 1 - mask2;
#endif

			for(int64_t i = i_start; i < i_end; ++i) {
				BitmapType summary_i = cq_summary[i];
				if(summary_i == 0) continue;
				for(int ii = 0; ii < (int)sizeof(cq_summary[0])*8; ++ii) {
					if(summary_i & (BitmapType(1) << ii)) {
						int64_t cq_base_offset = ((int64_t)i*sizeof(cq_summary[0])*8 + ii)*NUMBER_CQ_SUMMARIZING;
						for(int k = 0; k < NUMBER_CQ_SUMMARIZING; ++k) {
							int64_t e0 = cq_base_offset + k;
							BitmapType bitmap_k = cq_bitmap[e0];
							if(bitmap_k == 0) continue;
							// TODO: this_->graph_.log_local_v0()
							int64_t v0_high = (e0 << LOG_PACKING_EDGE_LISTS) |
									(int64_t(mpi.rank_2dc) << this_->graph_.log_local_v0());
							for(int64_t ri = row_starts[e0]; ri < row_starts[e0+1]; ++ri) {
								int8_t row_lowbits = (int8_t)(index_array.low_bits(ri) % NUMBER_PACKING_EDGE_LISTS);
								if ((bitmap_k & (int64_t(1) << row_lowbits)) != 0){
									int64_t c1 = index_array(ri) >> LOG_PACKING_EDGE_LISTS;
									int64_t v1_local = c1 & local_verts_mask;
									int64_t dest_c = c1 >> log_local_verts;
#if SHARED_VISITED_OPT
									if(v1_local < threshold) {
										// --- new algorithm begin ---
#if SHARED_VISITED_STRIPE
										int64_t sv_idx = (dest_c*NUMBER_CQ_SUMMARIZING*NUMBER_PACKING_EDGE_LISTS) |
												((c1 & mask1) << log_size_c) | (c1 & mask2);
#else
										int64_t sv_idx = c1;
#endif
										int64_t word_idx = sv_idx / NUMBER_PACKING_EDGE_LISTS;
										int bit_idx = sv_idx % NUMBER_PACKING_EDGE_LISTS;
										BitmapType mask = BitmapType(1) << bit_idx;

										if((shared_visited[word_idx] & mask) ||
											(__sync_fetch_and_or(&shared_visited[word_idx], mask) & mask))
										{
											continue;
										}
										// --- new algorithm end ---
									}
#endif
									// TODO:
									int64_t v0_swizzled = v0_high | row_lowbits;

									bfs_detail::FoldPacket* packet = &packet_array[dest_c];
									packet->v0_list[packet->num_edges] = v0_swizzled;
									packet->v1_list[packet->num_edges] = v1_local;
									if(++packet->num_edges == PACKET_LENGTH) {
										this_->fold_send_packet(packet, dest_c);
										packet->num_edges = 0;
									}
								}
							}
							// write zero after read
							cq_bitmap[e0] = 0;
						}
					}
				}
				// write zero after read
				cq_summary[i] = 0;
			}
			//
			volatile int* jobs_ptr = &this_->d_->num_remaining_extract_jobs_;
			if(__sync_add_and_fetch(jobs_ptr, -1) == 0) {
				this_->fiber_man_.submit_array(this_->sched_.fold_end_job, mpi.size_2dc, 0);
			}
		}
		ThisType* this_;
		int64_t i_start_, i_end_;
	};

	struct ReceiverProcessing : public Runnable {
		ReceiverProcessing(ThisType* this__)
			: this_(this__) 	{ }
		virtual void run() {
			using namespace BFS_PARAMS;
			ThreadLocalBuffer* tl = this_->thread_local_buffer_[omp_get_thread_num()];
			int64_t* decode_buffer = tl->decode_buffer;
			const uint8_t* v0_stream = data_->v0_stream->stream;
			const bfs_detail::PacketIndex* packet_index =
					&data_->v0_stream->index[data_->v0_stream->packet_index_start];
			BitmapType* restrict const visited = this_->visited_;
			BitmapType* restrict const nq_bitmap = this_->nq_bitmap_;
			BitmapType* restrict const nq_sorted_bitmap = this_->nq_sorted_bitmap_;
			int64_t* restrict const pred = this_->pred_;
			const int cur_level = this_->current_level_;
			int v0_offset = 0, v1_offset = 0;
			int64_t num_nq_vertices = 0;

			const int log_local_verts = this_->graph_.log_local_verts();
			const int64_t log_size = get_msb_index(mpi.size_2d);
			const int64_t local_verts_mask = this_->get_number_of_local_vertices() - 1;
#define UNSWIZZLE_VERTEX(c) (((c) >> log_local_verts) | (((c) & local_verts_mask) << log_size))

			for(int i = 0; i < data_->num_packets; ++i) {
				int stream_length = packet_index[i].length;
				int num_edges = packet_index[i].num_int;
#ifndef NDEBUG
				int decoded_elements =
#endif
				varint_decode_stream(&v0_stream[v0_offset],
						stream_length, (uint64_t*)decode_buffer);
				assert (decoded_elements == num_edges);
				const uint32_t* const v1_list = &data_->v1_list[v1_offset];
				int64_t v0_swizzled = 0;

				for(int c = 0; c < num_edges; ++c) {
					v0_swizzled += decode_buffer[c];

					const LocalVertsIndex v1_local = v1_list[c];
					const LocalVertsIndex word_idx = v1_local / NUMBER_PACKING_EDGE_LISTS;
					const int bit_idx = v1_local % NUMBER_PACKING_EDGE_LISTS;
					const BitmapType mask = BitmapType(1) << bit_idx;

					if((visited[word_idx] & mask) == 0) { // if this vertex has not visited
						if((__sync_fetch_and_or(&visited[word_idx], mask) & mask) == 0) {
							const int64_t v0 = UNSWIZZLE_VERTEX(v0_swizzled);
		//					const int64_t pred_v = (v0 & int64_t(0xFFFFFFFFFFFF)) | ((int64_t)cur_level << 48);
							const int64_t pred_v = v0 | ((int64_t)cur_level << 48);
#if VERTEX_SORTING
							__sync_fetch_and_or(&nq_sorted_bitmap[word_idx], mask);
							const LocalVertsIndex orig_v1_local = this_->graph_.invert_vertex_mapping_[v1_local];
							const LocalVertsIndex orig_word_idx = orig_v1_local / NUMBER_PACKING_EDGE_LISTS;
							const int orig_bit_idx = orig_v1_local % NUMBER_PACKING_EDGE_LISTS;
							const BitmapType orig_mask = BitmapType(1) << orig_bit_idx;
							assert (pred[orig_v1_local] == -1);
							__sync_fetch_and_or(&nq_bitmap[orig_word_idx], orig_mask);
							pred[orig_v1_local] = pred_v;
#else
							assert (pred[v1_local] == -1);
							__sync_fetch_and_or(&nq_bitmap[word_idx], mask);
							pred[v1_local] = pred_v;
#endif
							++num_nq_vertices;
						}
					}
				}
#undef UNSWIZZLE_VERTEX
				v0_offset += stream_length;
				v1_offset += num_edges;
			}
			this_->comm_.relase_fold_buffer(data_);
			tl->num_nq_vertices += num_nq_vertices;
#if 0
			this_->recv_task_.push(this);
#else
			delete this;
#endif
		}
		ThisType* const this_;
		bfs_detail::FoldCommBuffer* data_;
	};

	struct ExtractEnd : public Runnable {
		virtual void run() {
			// flush buffer
			for(int i = 0; i < omp_get_num_threads(); ++i) {
				bfs_detail::FoldPacket* packet_array =
						this_->thread_local_buffer_[i]->fold_packet;
				if(packet_array[dest_c_].num_edges > 0) {
					this_->fold_send_packet(&packet_array[dest_c_], dest_c_);
					packet_array[dest_c_].num_edges = 0;
				}
			}
			this_->comm_.fold_send_end(dest_c_);
		}
		ThisType* this_;
		int dest_c_;
	};

	virtual void fold_received(bfs_detail::FoldCommBuffer* data)
	{
#if 0
		ReceiverProcessing* proc = recv_task_.pop();
#else
		ReceiverProcessing* proc = new ReceiverProcessing(this);
#endif
		proc->data_ = data;
		fiber_man_.submit(proc, 1);
	}

	virtual void fold_finish()
	{
		fiber_man_.end_processing();
	}

	void printInformation()
	{
		if(mpi.isMaster() == false) return ;
		using namespace BFS_PARAMS;
		//fprintf(stderr, "Welcome to Graph500 Benchmark World.\n");
		//fprintf(stderr, "Check it out! You are running highly optimized BFS implementation.\n");

		fprintf(stderr, "===== Settings and Parameters. ====\n");
		fprintf(stderr, "NUM_BFS_ROOTS=%d.\n", NUM_BFS_ROOTS);
		fprintf(stderr, "OMP_NUM_THREADS=%d.\n", omp_get_max_threads());
		fprintf(stderr, "sizeof(BitmapType)=%zd.\n", sizeof(BitmapType));
		fprintf(stderr, "Index Type of Graph: %d bytes per edge.\n", IndexArray::bytes_per_edge);
		fprintf(stderr, "sizeof(LocalVertsIndex)=%zd.\n", sizeof(LocalVertsIndex));
		fprintf(stderr, "PACKET_LENGTH=%d.\n", PACKET_LENGTH);
		fprintf(stderr, "NUM_BFS_ROOTS=%d.\n", NUM_BFS_ROOTS);
		fprintf(stderr, "NUMBER_PACKING_EDGE_LISTS=%d.\n", NUMBER_PACKING_EDGE_LISTS);
		fprintf(stderr, "NUMBER_CQ_SUMMARIZING=%d.\n", NUMBER_CQ_SUMMARIZING);
		fprintf(stderr, "MINIMUN_SIZE_OF_CQ_BITMAP=%d.\n", MINIMUN_SIZE_OF_CQ_BITMAP);
		fprintf(stderr, "BLOCK_V0_LEGNTH=%d.\n", BLOCK_V0_LEGNTH);
		fprintf(stderr, "VERVOSE_MODE=%d.\n", VERVOSE_MODE);
		fprintf(stderr, "SHARED_VISITED_OPT=%d.\n", SHARED_VISITED_OPT);
		fprintf(stderr, "VALIDATION_LEVEL=%d.\n", VALIDATION_LEVEL);
		fprintf(stderr, "DENOM_SHARED_VISITED_PART=%d.\n", DENOM_SHARED_VISITED_PART);
	}

	void prepare_sssp() { }
	void run_sssp(int64_t root, int64_t* pred) { }
	void end_sssp() { }

	// members
	bfs_detail::Bfs2DComm comm_;
	FiberManager fiber_man_;
#if 0
	ConcurrentStack<ReceiverProcessing*> recv_task_;
#endif
	ThreadLocalBuffer** thread_local_buffer_;
	BitmapType* cq_bitmap_;
	BitmapType* cq_summary_; // 128bytes -> 1bit
	BitmapType* shared_visited_;
	BitmapType* visited_;
	BitmapType* nq_bitmap_;
	BitmapType* nq_sorted_bitmap_;
	int64_t* pred_;

	struct DynamicDataSet {
		int64_t num_tmp_packets_;
		int64_t tmp_packet_offset_;
		// We count only if CQ is transfered by stream. Otherwise, 0.
		int64_t num_vertices_in_cq_;
		int num_remaining_extract_jobs_;
	} *d_;

	int log_local_bitmap_;
#if 0
	int num_recv_tasks_;
#endif
	int64_t tmp_packet_max_length_;
	bfs_detail::PacketIndex* tmp_packet_index_;
	ExpandCommCommand cq_comm_;
	ExpandCommCommand visited_comm_;

	struct {
		ExtractEdge* job_array;
		ExtractEdge* long_job;
		ExtractEdge* short_job;
		ExtractEnd* fold_end_job;
		int long_job_length;
		int short_job_length;
	} sched_;

	int current_level_;

	struct {
		void* thread_local_;
	} buffer_;
#if VERVOSE_MODE
	profiling::TimeSpan extract_edge_time_;
	profiling::TimeSpan recv_proc_time_;
#endif
};

template <typename IndexArray, typename LocalVertsIndex, typename PARAMS>
void BfsBase<IndexArray, LocalVertsIndex, PARAMS>::
	run_bfs(int64_t root, int64_t* pred)
{
	using namespace BFS_PARAMS;
	pred_ = pred;
#if VERVOSE_MODE
	double tmp = MPI_Wtime();
	double start_time = tmp;
	double prev_time = tmp;
	double expand_time = 0.0, fold_time = 0.0, stall_time = 0.0;
	g_fold_send = g_fold_recv = g_bitmap_send = g_bitmap_recv = g_exs_send = g_exs_recv = 0;
#endif
	// threshold of scheduling for extracting CQ.
	const int64_t sched_threshold = get_number_of_local_vertices() * mpi.size_2dr / 16;

	const int log_size = get_msb_index(mpi.size_2d);
	const int size_mask = mpi.size_2d - 1;
#define VERTEX_OWNER(v) ((v) & size_mask)
#define VERTEX_LOCAL(v) ((v) >> log_size)

	initialize_memory(pred);
#if VERVOSE_MODE
	if(mpi.isMaster()) fprintf(stderr, "Time of initialize memory: %f ms\n", (MPI_Wtime() - prev_time) * 1000.0);
	prev_time = MPI_Wtime();
#endif
	current_level_ = 0;
	int root_owner = (int)VERTEX_OWNER(root);
	if(root_owner == mpi.rank_2d) {
		int64_t root_local = VERTEX_LOCAL(root);
		pred_[root_local] = root;
#if VERTEX_SORTING
		int64_t sortd_root_local = graph_.vertex_mapping_[root_local];
		int64_t word_idx = sortd_root_local / NUMBER_PACKING_EDGE_LISTS;
		int bit_idx = sortd_root_local % NUMBER_PACKING_EDGE_LISTS;
#else
		int64_t word_idx = root / NUMBER_PACKING_EDGE_LISTS;
		int bit_idx = root % NUMBER_PACKING_EDGE_LISTS;
#endif
		visited_[word_idx] |= BitmapType(1) << bit_idx;
		expand_root(root_local, &cq_comm_, &visited_comm_);
	}
	else {
		expand_root(-1, &cq_comm_, &visited_comm_);
	}
#if VERVOSE_MODE
	tmp = MPI_Wtime();
	if(mpi.isMaster()) fprintf(stderr, "Time of first expansion: %f ms\n", (tmp - prev_time) * 1000.0);
	expand_time += tmp - prev_time; prev_time = tmp;
#endif
#undef VERTEX_OWNER
#undef VERTEX_LOCAL

	while(true) {
		++current_level_;
#if VERVOSE_MODE
		double level_start_time = MPI_Wtime();
#endif
		for(int i = 0; i < omp_get_max_threads(); ++i)
			thread_local_buffer_[i]->num_nq_vertices = 0;

		fiber_man_.begin_processing();
		comm_.begin_fold_comm();

		// submit graph extraction job
		if(d_->num_vertices_in_cq_ >= sched_threshold) {
			fiber_man_.submit_array(sched_.long_job, sched_.long_job_length, 0);
			d_->num_remaining_extract_jobs_ = sched_.long_job_length;
		}
		else {
			fiber_man_.submit_array(sched_.short_job, sched_.short_job_length, 0);
			d_->num_remaining_extract_jobs_ = sched_.short_job_length;
		}

#pragma omp parallel
		fiber_man_.enter_processing();

#if VERVOSE_MODE
		tmp = MPI_Wtime(); fold_time += tmp - prev_time; prev_time = tmp;
#endif

		int64_t num_nq_vertices = 0;
		for(int i = 0; i < omp_get_max_threads(); ++i)
			num_nq_vertices += thread_local_buffer_[i]->num_nq_vertices;

		int64_t global_nq_vertices;
		MPI_Allreduce(&num_nq_vertices, &global_nq_vertices, 1,
				get_mpi_type(num_nq_vertices), MPI_SUM, mpi.comm_2d);
#if VERVOSE_MODE
		tmp = MPI_Wtime(); stall_time += tmp - prev_time; prev_time = tmp;
#endif
#if DEBUG_PRINT
	if(mpi.isMaster()) printf("global_nq_vertices=%"PRId64"\n", global_nq_vertices);
#endif
		if(global_nq_vertices == 0)
			break;

		expand(global_nq_vertices, &cq_comm_, &visited_comm_);
#if VERVOSE_MODE
		tmp = MPI_Wtime();
		if(mpi.isMaster()) fprintf(stderr, "Time of levle %d: %f ms\n", current_level_, (MPI_Wtime() - level_start_time) * 1000.0);
		expand_time += tmp - prev_time; prev_time = tmp;
#endif
	}
#if VERVOSE_MODE
	if(mpi.isMaster()) fprintf(stderr, "Time of BFS: %f ms\n", (MPI_Wtime() - start_time) * 1000.0);
	double time3[3] = { fold_time, expand_time, stall_time };
	double timesum3[3];
	int64_t commd[6] = { g_fold_send, g_fold_recv, g_bitmap_send, g_bitmap_recv, g_exs_send, g_exs_recv };
	int64_t commdsum[6];
	MPI_Reduce(time3, timesum3, 3, MPI_DOUBLE, MPI_SUM, 0, mpi.comm_2d);
	MPI_Reduce(commd, commdsum, 6, get_mpi_type(commd[0]), MPI_SUM, 0, mpi.comm_2d);
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
	}
#endif
}

#endif /* BFS_HPP_ */
