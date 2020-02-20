#include "cartridge.h"

namespace seeta
{
	namespace orz
	{
		Cartridge::Cartridge() : dry(true), bullet(nullptr), shell(nullptr)
		{
			this->powder = std::thread(&Cartridge::operating, this);
		}

		Cartridge::~Cartridge()
		{
			dry = false;
			fire_cond.notify_all();
			powder.join();
		}

		void Cartridge::fire(int signet, const Cartridge::bullet_type &bullet, const Cartridge::shell_type &shell)
		{
			std::unique_lock<std::mutex> locker(fire_mutex);
			this->signet = signet;
			this->bullet = bullet;
			this->shell = shell;
			fire_cond.notify_all();
		}

		bool Cartridge::busy()
		{
			if (!fire_mutex.try_lock())
				return false;

			bool is_busy = bullet != nullptr;
			fire_mutex.unlock();
			return is_busy;
		}

		void Cartridge::join()
		{
			std::unique_lock<std::mutex> locker(fire_mutex);

			while (bullet)
				fire_cond.wait(locker);
		}

		void Cartridge::operating()
		{
			std::unique_lock<std::mutex> locker(fire_mutex);

			while (dry)
			{
				while (dry && !bullet)
					fire_cond.wait(locker);

				if (!dry)
					break;

				bullet(signet);

				if (shell)
					shell(signet);

				bullet = nullptr;
				shell = nullptr;
				fire_cond.notify_all();
			}
		}
	}
}
