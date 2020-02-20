//
// Created by Lby on 2017/8/23.
//

#include "vat.h"
#include <algorithm>

#include "../tools/ctxmgr_lite_support.h"

namespace seeta
{
	namespace orz
	{
		Vat::Vat()
		{
		}

		void *Vat::malloc(size_t _size)
		{
			if (_size == 0)
				return nullptr;

			// find first small piece
			Pot pot;

			if (!m_heap.empty())
			{
				size_t i = 0;

				for (; i < m_heap.size() - 1; ++i)
				{
					if (m_heap[i].capacity() >= _size) break;
				}

				pot = m_heap[i];
				m_heap.erase(m_heap.begin() + i);
			}

			void *ptr = pot.malloc(_size);
			m_dict.insert(std::pair<void *, orz::Pot>(ptr, pot));

			return ptr;
		}

		void Vat::free(const void *ptr)
		{
			if (ptr == nullptr)
				return;

			auto key = const_cast<void *>(ptr);
			auto it = m_dict.find(key);

			if (it == m_dict.end())
			{
				throw std::logic_error("Can not free this ptr");
			}

			auto &pot = it->second;
			auto ind = m_heap.begin();

			while (ind != m_heap.end() && ind->capacity() < pot.capacity())
				++ind;

			m_heap.insert(ind, pot);
			m_dict.erase(key);
		}

		void Vat::reset()
		{
			for (auto &pair : m_dict)
			{
				m_heap.push_back(pair.second);
			}

			m_dict.clear();

			std::sort(m_heap.begin(), m_heap.end(), [](const orz::Pot &p1, const orz::Pot &p2)
			{
				return p1.capacity() < p2.capacity();
			});
		}

		void Vat::dispose()
		{
			m_dict.clear();
			m_heap.clear();
		}

		void Vat::swap(Vat &that)
		{
			this->m_heap.swap(that.m_heap);
			this->m_dict.swap(that.m_dict);
		}

		Vat::Vat(Vat &&that)
		{
			this->swap(that);
		}

		Vat &Vat::operator=(Vat &&that)
		{
			this->swap(that);
			return *this;
		}
	}

	ORZ_LITE_CONTEXT(orz::Vat)
}
