//
// Created by Lby on 2017/8/11.
//

#ifndef ORZ_SYNC_CANYON_H
#define ORZ_SYNC_CANYON_H

#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <queue>
#include <future>

#include "../tools/void_bind.h"

namespace seeta
{

    namespace orz
    {

        class Canyon {
        public:
            enum Action
            {
                DISCARD,
                WAITING
            };

            explicit Canyon( int size = -1, Action act = WAITING );

            ~Canyon();

            template<typename FUNC>
            void operator()( FUNC func ) const {
                auto op = [ = ]() -> void { func(); };
                this->push( void_bind( func ) );
            }

            template<typename FUNC, typename... Args>
            void operator()( FUNC func, Args &&... args ) const {
                this->push( void_bind( func, std::forward<Args>( args )... ) );
            }

            void join() const;

        private:
            Canyon( const Canyon &that ) = delete;

            const Canyon &operator=( const Canyon &that ) = delete;

            void push( const VoidOperator &op ) const;

            void operating() const;

            mutable std::queue<VoidOperator> _task;
            mutable std::mutex _mutex;
            mutable std::condition_variable _cond;
            std::atomic<bool> _work;
            int _size;
            Action _act;

            std::thread _core;
        };

    }

}
using namespace seeta;

#endif //ORZ_SYNC_CANYON_H
