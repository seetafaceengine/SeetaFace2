#ifndef ORZ_SYNC_SHOTGUN_H
#define ORZ_SYNC_SHOTGUN_H

#include "cartridge.h"

#include <vector>
#include <deque>

namespace seeta
{
	namespace orz
	{

		/**
		 * @brief The Shotgun class the thread pool
		 */
		class Shotgun
		{
		public:
			/**
			 * @brief Shotgun
			 * @param clip_size The cartridge number in clip. Number of threads
			 */
			Shotgun(size_t clip_size);

			~Shotgun();

			/**
			 * @brief fire Find ready cartridge, build bullet and fire.
			 * @param bullet the work ready to run
			 * @return The cartridge running bullet
			 */
			Cartridge *fire(const Cartridge::bullet_type &bullet);

			/**
			 * @brief fire Find ready cartridge, build bullet and fire.
			 * @param bullet the work ready to run
			 * @param shell the work after bullet finished
			 * @return The cartridge running bullet
			 */
			Cartridge *fire(const Cartridge::bullet_type &bullet, const Cartridge::shell_type &shell);

			/**
			 * @brief join Wait all cartridge working finish.
			 */
			void join();

			/**
			 * @brief busy Return if there are work running in thread
			 * @return True if busy
			 */
			bool busy();

			/**
			 * @brief size Get number of threads
			 * @return Number of threads
			 */
			size_t size() const;

		private:
			Shotgun(const Shotgun &that) = delete;
			const Shotgun &operator=(const Shotgun &that) = delete;

			/**
			 * @brief load Get cartridge ready to fire
			 * @return Get ready cartridge
			 */
			int load();

			/**
			 * @brief recycling_cartridge Recycle cartridge
			 * @param signet cartridge index
			 */
			void recycling_cartridge(int signet);

			std::vector<Cartridge *> clip;          ///< all cartridges

			std::mutex chest_mutex;                 ///< mutex to get cartridges
			std::condition_variable chest_cond;     ///< active when cartridge pushed in chest
			std::deque<int> chest;                 ///< save all cartridge ready to fire
		};
	}
}

using namespace seeta;

#endif // ORZ_SYNC_SHOTGUN_H
