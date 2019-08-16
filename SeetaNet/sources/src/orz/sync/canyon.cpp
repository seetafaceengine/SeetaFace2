//
// Created by Lby on 2017/8/11.
//

#include "canyon.h"

namespace seeta
{

    namespace orz
    {

        Canyon::Canyon( int size, Action act )
            : _work( true ), _size( size ), _act( act )
        {
            this->_core = std::thread( &Canyon::operating, this );
        }

        Canyon::~Canyon()
        {
            this->join();
            _work = false;
            _cond.notify_all();
            _core.join();
        }

        void Canyon::join() const
        {
            std::unique_lock<std::mutex> _locker( _mutex );
            while( _task.size() ) _cond.wait( _locker );
        }

        void Canyon::push( const VoidOperator &op ) const
        {
            std::unique_lock<std::mutex> _locker( _mutex );
            while( _size > 0 && _task.size() >= static_cast<size_t>( _size ) )
            {
                switch( _act )
                {
                    case WAITING:
                        _cond.wait( _locker );
                        break;
                    case DISCARD:
                        return;
                }
            }
            _task.push( op );
            _cond.notify_all();
        }

        void Canyon::operating() const
        {
            std::unique_lock<std::mutex> _locker( _mutex );
            while( _work )
            {
                while( _work && _task.size() == 0 ) _cond.wait( _locker );
                if( !_work ) break;
                auto func = _task.front();
                _task.pop();
                func();
                _cond.notify_all();
            }
        }
    }

}
