//
// Created by Lby on 2017/8/12.
//

#include "pot.h"
#include <cstring>
#include <stdlib.h>

namespace seeta
{

namespace orz
{

static std::shared_ptr<void> nolambda_template_allocator(size_t _size)
{
    return std::shared_ptr<void>(malloc(_size), free);
}

Pot::Pot()
    : Pot(nolambda_template_allocator)
{
}

Pot::Pot(const allocator &ator)
    :  m_allocator(ator), m_capacity(0), m_data()
{

}

void *Pot::malloc(size_t _size)
{
    if (_size > m_capacity)
    {
        m_data = m_allocator(_size);
        m_capacity = _size;
    }
    return m_data.get();
}

void *Pot::relloc(size_t _size)
{
    if (_size > m_capacity)
    {
        auto new_data = m_allocator(_size);
#if _MSC_VER >= 1600
        memcpy_s(new_data.get(), _size, m_data.get(), m_capacity);
#else
        memcpy(new_data.get(), m_data.get(), m_capacity);
#endif
        m_data = new_data;
        m_capacity = _size;
    }
    return m_data.get();
}

void *Pot::data() const
{
    return m_data.get();
}

size_t Pot::capacity() const
{
    return m_capacity;
}

void Pot::dispose()
{
    m_capacity = 0;
    m_data.reset();
}
}

}
