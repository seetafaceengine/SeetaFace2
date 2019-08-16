//
// Created by Lby on 2017/8/12.
//

#ifndef ORZ_MEM_POT_H
#define ORZ_MEM_POT_H

#include <mutex>
#include <memory>

namespace seeta
{

    namespace orz
    {

        class Pot {
        public:
            using allocator = std::function<std::shared_ptr<void>( size_t )>;

            Pot();
            Pot( const allocator &ator );

            void *malloc( size_t _size );

            void *relloc( size_t _size );

            template<typename T>
            T *calloc( size_t _count, bool copy = false ) {
                if( copy )
                    return reinterpret_cast<T *>( this->relloc( sizeof( T ) * _count ) );
                else
                    return reinterpret_cast<T *>( this->malloc( sizeof( T ) * _count ) );
            }

            void *data() const;

            size_t capacity() const;

            void dispose();

        private:
            allocator m_allocator;

        private:
            size_t m_capacity;
            std::shared_ptr<void> m_data;
        };
    }

}
using namespace seeta;

#endif //ORZ_MEM_POT_H
