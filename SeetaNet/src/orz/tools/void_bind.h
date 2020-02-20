//
// Created by lby on 2018/1/23.
//

#ifndef ORZ_TOOLS_VOID_BIND_H
#define ORZ_TOOLS_VOID_BIND_H

#include <functional>

namespace seeta
{
	namespace orz
	{
		using VoidOperator = std::function<void()>;

		// for error C3848 in MSVC

		template<typename Ret, typename FUNC>
		class _Operator
		{
		public:
			static VoidOperator bind(FUNC func)
			{
				return [func]() -> void
				{
					// for error C3848 in MSVC
					FUNC non_const_func = func;
					non_const_func();
				};
			}
		};

		template<typename FUNC>
		class _Operator<void, FUNC>
		{
		public:
			static VoidOperator bind(FUNC func)
			{
				return func;
			}
		};

		template<typename FUNC, typename... Args>
		inline VoidOperator void_bind(FUNC func, Args &&... args)
		{
			auto inner_func = std::bind(func, std::forward<Args>(args)...);
			using Ret = decltype(inner_func());
			using RetOperator = _Operator<Ret, decltype(inner_func)>;

			return RetOperator::bind(inner_func);
		}

		template<typename FUNC, typename... Args>
		inline void void_call(FUNC func, Args &&... args)
		{
			std::bind(func, std::forward<Args>(args)...)();
		};
	}
}

using namespace seeta;

#endif //ORZ_TOOLS_VOID_BIND_H
