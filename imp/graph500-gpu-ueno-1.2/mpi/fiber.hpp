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
#ifndef FIBER_HPP_
#define FIBER_HPP_

#include <pthread.h>

#include <deque>
#include <vector>

class Runnable
{
public:
	virtual ~Runnable() { }
	virtual void run() = 0;
};

class FiberManager
{
public:
	static const int MAX_PRIORITY = 4;

	FiberManager()
		: command_active_(false)
		, terminated_(false)
		, suspended_(0)
		, max_priority_(0)
	{
		pthread_mutex_init(&thread_sync_, NULL);
		pthread_cond_init(&thread_state_,  NULL);
		cleanup_ = false;
	}

	~FiberManager()
	{
		if(!cleanup_) {
			cleanup_ = true;
			pthread_mutex_destroy(&thread_sync_);
			pthread_cond_destroy(&thread_state_);
		}
	}

	void begin_processing()
	{
		terminated_ = false;
	}

	void enter_processing()
	{
		// command loop
		while(true) {
			if(command_active_) {
				pthread_mutex_lock(&thread_sync_);
				Runnable* cmd;
				while(pop_command(&cmd)) {
					pthread_mutex_unlock(&thread_sync_);
					cmd->run();
					pthread_mutex_lock(&thread_sync_);
				}
				pthread_mutex_unlock(&thread_sync_);
			}
			if(command_active_ == false) {
				pthread_mutex_lock(&thread_sync_);
				if(command_active_ == false) {
					if( terminated_ ) { pthread_mutex_unlock(&thread_sync_); break; }
					++suspended_;
					pthread_cond_wait(&thread_state_, &thread_sync_);
					--suspended_;
				}
				pthread_mutex_unlock(&thread_sync_);
			}
		}
	}

	void end_processing()
	{
		pthread_mutex_lock(&thread_sync_);
		terminated_ = true;
		pthread_mutex_unlock(&thread_sync_);
		pthread_cond_broadcast(&thread_state_);
	}

	void submit(Runnable* r, int priority)
	{
		pthread_mutex_lock(&thread_sync_);
		command_active_ = true;
		command_queue_[priority].push_back(r);
		max_priority_ = std::max(priority, max_priority_);
		int num_suspended = suspended_;
		pthread_mutex_unlock(&thread_sync_);
		if(num_suspended > 0) pthread_cond_broadcast(&thread_state_);
	}

	template <typename T>
	void submit_array(T* runnable_array, size_t length, int priority)
	{
		pthread_mutex_lock(&thread_sync_);
		command_active_ = true;
		std::deque<Runnable*>& queue = command_queue_[priority];
		size_t pos = queue.size();
		queue.insert(queue.end(), length, NULL);
		for(size_t i = 0; i < length; ++i) {
			queue[pos + i] = &runnable_array[i];
		}
		max_priority_ = std::max(priority, max_priority_);
		int num_suspended = suspended_;
		pthread_mutex_unlock(&thread_sync_);
		if(num_suspended > 0) pthread_cond_broadcast(&thread_state_);
	}

private:
	//
	bool cleanup_;
	pthread_mutex_t thread_sync_;
	pthread_cond_t thread_state_;

	volatile bool command_active_;

	bool terminated_;
	int suspended_;
	int max_priority_;

	std::deque<Runnable*> command_queue_[MAX_PRIORITY];

	bool pop_command(Runnable** cmd) {
		int i = max_priority_ + 1;
		while(i-- > 0) {
			assert (i < MAX_PRIORITY);
			if(command_queue_[i].size()) {
				*cmd = command_queue_[i][0];
				command_queue_[i].pop_front();
				max_priority_ = i;
				return true;
			}
		}
		max_priority_ = 0;
		command_active_ = false;
		return false;
	}
};

template <typename T>
class ConcurrentStack
{
public:
	ConcurrentStack(int limit)
		: limit_(limit)
	{
		pthread_mutex_init(&thread_sync_, NULL);
		pthread_cond_init(&thread_state_,  NULL);
		cleanup_ = false;
	}

	~ConcurrentStack()
	{
		if(!cleanup_) {
			cleanup_ = true;
			pthread_mutex_destroy(&thread_sync_);
			pthread_cond_destroy(&thread_state_);
		}
	}

	void push(const T& d)
	{
		pthread_mutex_lock(&thread_sync_);
		while((int)stack_.size() >= limit_) {
			pthread_cond_wait(&thread_state_, &thread_sync_);
		}
		int old_size = (int)stack_.size();
		stack_.push_back(d);
		pthread_mutex_unlock(&thread_sync_);
		if(old_size == 0) pthread_cond_broadcast(&thread_state_);
	}

	T pop()
	{
		pthread_mutex_lock(&thread_sync_);
		while(stack_.size() == 0) {
			pthread_cond_wait(&thread_state_, &thread_sync_);
		}
		int old_size = (int)stack_.size();
		T r = stack_.back(); stack_.pop_back();
		pthread_mutex_unlock(&thread_sync_);
		if(old_size == limit_) pthread_cond_broadcast(&thread_state_);
		return r;
	}

private:
	int limit_;
	bool cleanup_;
	std::vector<T> stack_;
	pthread_mutex_t thread_sync_;
	pthread_cond_t thread_state_;
};


#endif /* FIBER_HPP_ */
