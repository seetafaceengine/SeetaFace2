#ifndef __WRITE_FIRST_RW_LOCK_H
#define __WRITE_FIRST_RW_LOCK_H

#include <mutex>
#include <condition_variable>

namespace seeta
{
	class rwmutex
	{
	public:
		rwmutex() = default;
		~rwmutex() = default;

		rwmutex(const rwmutex &) = delete;
		rwmutex &operator=(const rwmutex &) = delete;

		rwmutex(rwmutex &&) = delete;
		rwmutex &operator=(rwmutex &&) = delete;

		void lock_read()
		{
			std::unique_lock<std::mutex> _locker(m_mutex_counter);
			m_cond_read.wait(_locker, [this]()-> bool { return m_write_count == 0; });
			++m_read_count;
		}

		void lock_write()
		{
			std::unique_lock<std::mutex> _locker(m_mutex_counter);
			++m_write_count;
			m_cond_write.wait(_locker, [this]()-> bool { return m_read_count == 0 && !m_writing; });
			m_writing = true;
		}

		void release_read()
		{
			std::unique_lock<std::mutex> _locker(m_mutex_counter);

			if (--m_read_count == 0 && m_write_count > 0)
			{
				m_cond_write.notify_one();
			}
		}

		void release_write()
		{
			std::unique_lock<std::mutex> _locker(m_mutex_counter);

			if (--m_write_count == 0)
			{
				m_cond_read.notify_all();
			}
			else
			{
				m_cond_write.notify_one();
			}

			m_writing = false;
		}

	private:
		volatile size_t m_read_count = 0;
		volatile size_t m_write_count = 0;
		volatile bool m_writing = false;

		std::mutex m_mutex_counter;
		std::condition_variable m_cond_write;
		std::condition_variable m_cond_read;
	};

	template <typename _RWLockable>
	class unique_write_lock
	{
	public:
		explicit unique_write_lock(_RWLockable &rw_lockable) : m_ptr_rw_lockable(&rw_lockable)
		{
			m_ptr_rw_lockable->lock_write();
		}

		unique_write_lock(unique_write_lock &&other)
		{
			*this = std::move(other);
		}

		unique_write_lock &operator=(unique_write_lock &&other)
		{
			std::swap(m_ptr_rw_lockable, other.m_ptr_rw_lockable);
			return *this;
		}

		~unique_write_lock()
		{
			if (m_ptr_rw_lockable)
				m_ptr_rw_lockable->release_write();
		}

		unique_write_lock() = delete;
		unique_write_lock(const unique_write_lock &) = delete;
		unique_write_lock &operator=(const unique_write_lock &) = delete;

	private:
		_RWLockable *m_ptr_rw_lockable = nullptr;
	};

	template <typename _RWLockable>
	class unique_read_lock
	{
	public:
		explicit unique_read_lock(_RWLockable &rw_lockable) : m_ptr_rw_lockable(&rw_lockable)
		{
			m_ptr_rw_lockable->lock_read();
		}

		unique_read_lock(unique_read_lock &&other)
		{
			*this = std::move(other);
		}

		unique_read_lock &operator=(unique_read_lock &&other)
		{
			std::swap(m_ptr_rw_lockable, other.m_ptr_rw_lockable);
			return *this;
		}

		~unique_read_lock()
		{
			if (m_ptr_rw_lockable)
				m_ptr_rw_lockable->release_read();
		}

		unique_read_lock() = delete;
		unique_read_lock(const unique_read_lock &) = delete;
		unique_read_lock &operator=(const unique_read_lock &) = delete;

	private:
		_RWLockable *m_ptr_rw_lockable = nullptr;
	};
}

#endif
