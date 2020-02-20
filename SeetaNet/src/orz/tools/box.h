//
// Created by kier on 2018/10/31.
//

#ifndef ORZ_TOOLS_BOX_H
#define ORZ_TOOLS_BOX_H

#include <string>
#include <vector>
#include <utility>
#include <chrono>

namespace seeta
{
	namespace orz
	{
		using time_point = decltype(std::chrono::system_clock::now());

		/**
		 * get edit distance of edit lhs to rhs
		 * @param lhs original string
		 * @param rhs wanted string
		 * @return edit distance, 0 for `lhs == rhs`
		 */
		int edit_distance(const std::string &lhs, const std::string &rhs);

		/**
		 * get `bins` bins split set [first, second)
		 * @param first min number
		 * @param second max number
		 * @param bins number of bins
		 * @return A list contains splited bins
		 * @note Example input(0, 10, 3) returns [(0, 4), (4, 8), (8, 10)]
		 */
		std::vector<std::pair<int, int>> split_bins(int first, int second, int bins);

		/**
		 * get `bins` bins split set [first, second)
		 * @param first min number
		 * @param second max number
		 * @param bins number of bins
		 * @return A list contains splited bins
		 * @note Example input(0, 10, 3) returns [(0, 4), (4, 8), (8, 10)]
		 */
		std::vector<std::pair<size_t, size_t>> lsplit_bins(size_t first, size_t second, size_t bins);

		std::string to_string(time_point tp, const std::string &format = "%Y-%m-%d %H:%M:%S");

		/**
		 * return format now time string
		 * @param format same as strftime
		 * @return string contains now time
		 * @see strftime
		 */
		std::string now_time(const std::string &format = "%Y-%m-%d %H:%M:%S");
	}
}

using namespace seeta;

#endif //ORZ_TOOLS_BOX_H
