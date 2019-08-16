#include "shotgun.h"

#include "../tools/ctxmgr_lite_support.h"

namespace seeta
{

    namespace orz
    {
        Shotgun::Shotgun( size_t clip_size )
            : clip( clip_size )
        {
            for( int i = 0; i < static_cast<int>( clip_size ); ++i )
            {
                clip[i] = new Cartridge();
                chest.push_back( i ); // push all cartridge into chest
            }
        }

        Shotgun::~Shotgun()
        {
            for( int i = 0; i < static_cast<int>( clip.size() ); ++i )
            {
                delete clip[i];
            }
        }

        Cartridge *Shotgun::fire( const Cartridge::bullet_type &bullet )
        {
            if( clip.size() == 0 )
            {
                bullet( 0 );
                return nullptr;
            }
            else
            {
                int signet = load();
                Cartridge *cart = this->clip[signet];
                cart->fire( signet, bullet,
                            Cartridge::shell_type( std::bind( &Shotgun::recycling_cartridge, this, std::placeholders::_1 ) ) );
                return cart;
            }

        }

        Cartridge *Shotgun::fire( const Cartridge::bullet_type &bullet, const Cartridge::shell_type &shell )
        {
            if( clip.size() == 0 )
            {
                bullet( 0 );
                return nullptr;
            }
            else
            {
                int signet = load();
                Cartridge *cart = this->clip[signet];
                cart->fire( signet, bullet, [this, shell]( int id ) -> void
                {
                    shell( id );
                    this->recycling_cartridge( id );
                } );
                return cart;
            }

        }

        int Shotgun::load()
        {
            std::unique_lock <std::mutex> locker( chest_mutex );
            while( this->chest.empty() ) chest_cond.wait( locker );
            int signet = this->chest.front();
            this->chest.pop_front();
            return signet;
        }

        void Shotgun::join()
        {
            std::unique_lock <std::mutex> locker( chest_mutex );
            while( this->chest.size() != this->clip.size() ) chest_cond.wait( locker );
        }

        bool Shotgun::busy()
        {
            if( !chest_mutex.try_lock() ) return false;
            bool is_busy = this->chest.size() != this->clip.size();
            chest_mutex.unlock();
            return is_busy;
        }

        size_t Shotgun::size() const
        {
            return clip.size();
        }

        void Shotgun::recycling_cartridge( int signet )
        {
            std::unique_lock <std::mutex> locker( chest_mutex );
            this->chest.push_back( signet );
            chest_cond.notify_all();
        }

    }

    ORZ_LITE_CONTEXT( orz::Shotgun )

}
