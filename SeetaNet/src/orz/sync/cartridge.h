#ifndef ORZ_SYNC_BULLET_H
#define ORZ_SYNC_BULLET_H

#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>

namespace seeta
{
    namespace orz
    {
        class Cartridge 
		{
        public:
            Cartridge();
            ~Cartridge();

            using bullet_type = std::function<void( int )>;
            using shell_type = std::function<void( int )>;

            /**
             * @brief fire Asynchronous build and fire bullet, first calls the bullet, then calls the shell.
             * @param signet the index to call `bullet(signet)` and `shell(signet)`
             * @param bullet the function call in thread
             * @param shell call it after bullet called
             */
            void fire( int signet, const bullet_type &bullet, const shell_type &shell = nullptr );

            bool busy();
            void join();

        private:
            Cartridge( const Cartridge &that ) = delete;
            const Cartridge &operator=( const Cartridge &that ) = delete;

            void operating();

            std::mutex fire_mutex;              ///< mutex control each fire
            std::condition_variable fire_cond;  ///< condition to tell if fire finished
            std::atomic<bool> dry;              ///< object only work when dry is true

            int signet;                         ///< the argument to call `bullet(signet)` and `shell(signet)`
            bullet_type bullet = nullptr;      ///< main function call in thread
            shell_type shell = nullptr;        ///< side function call after `bullet` called

            std::thread powder;                 ///< working thread
        };
    }
}

using namespace seeta;

#endif // ORZ_SYNC_BULLET_H
