#include "common_alignment.h"

#include <memory>
#include <algorithm>
#include <vector>
#include <numeric>
#include <functional>
#include <cfloat>
#include <cstring>
#include <cmath>

template <typename T>
void CopyData(T *dst, const T *src, size_t _count)
{
#if _MSC_VER >= 1600
	memcpy_s(dst, sizeof(T) * _count, src, sizeof(T) * _count);
#else
	memcpy(dst, src, sizeof(T) * _count);
#endif
}

static const int TFORM_SIZE = 6;

/**
 * \brief
 * \param crop_width Crop
 * \param crop_height Crop
 * \param points format {(x1, y1), (x2, y2), ...}
 * \param points_num
 * \param mean_shape
 * \param mean_shape_width
 * \param mean_shape_height
 * \param transformation size: N * TFORM_SIZE
 * \param N always 1
 * \return
 */
static bool transformation_maker(
	int crop_width, int crop_height,
	const float *points, int points_num,
	const float *mean_shape, int mean_shape_width, int mean_shape_height,
	double *transformation, int N = 1)
{
	std::unique_ptr<float[]> raw_std_points(new float[points_num * 2]);
	float *std_points = raw_std_points.get();

	for (int i = 0; i < points_num; ++i)
	{
		std_points[i * 2] = mean_shape[i * 2] * crop_width / mean_shape_width;
		std_points[i * 2 + 1] = mean_shape[i * 2 + 1] * crop_height / mean_shape_height;
	}

	const float *feat_points = points;

	for (int n = 0; n < N; ++n)
	{
		double sum_x = 0, sum_y = 0;
		double sum_u = 0, sum_v = 0;
		double sum_xx_yy = 0;
		double sum_ux_vy = 0;
		double sum_vx__uy = 0;

		for (int c = 0; c < points_num; ++c)
		{
			int x_off = n * points_num * 2 + c * 2;
			int y_off = x_off + 1;

			sum_x += std_points[c * 2];
			sum_y += std_points[c * 2 + 1];
			sum_u += feat_points[x_off];
			sum_v += feat_points[y_off];

			sum_xx_yy += std_points[c * 2] * std_points[c * 2] + std_points[c * 2 + 1] * std_points[c * 2 + 1];
			sum_ux_vy += std_points[c * 2] * feat_points[x_off] + std_points[c * 2 + 1] * feat_points[y_off];
			sum_vx__uy += feat_points[y_off] * std_points[c * 2] - feat_points[x_off] * std_points[c * 2 + 1];
		}

		if (sum_xx_yy <= FLT_EPSILON)
			return false;

		double q = sum_u - sum_x * sum_ux_vy / sum_xx_yy + sum_y * sum_vx__uy / sum_xx_yy;
		double p = sum_v - sum_y * sum_ux_vy / sum_xx_yy - sum_x * sum_vx__uy / sum_xx_yy;
		double r = points_num - (sum_x * sum_x + sum_y * sum_y) / sum_xx_yy;

		if (!(r > FLT_EPSILON || r < -FLT_EPSILON))
			return false;

		double a = (sum_ux_vy - sum_x * q / r - sum_y * p / r) / sum_xx_yy;
		double b = (sum_vx__uy + sum_y * q / r - sum_x * p / r) / sum_xx_yy;
		double c = q / r;
		double d = p / r;

		double *tform = transformation + n * TFORM_SIZE;
		tform[0] = tform[4] = a;
		tform[1] = -b;
		tform[3] = b;
		tform[2] = c;
		tform[5] = d;
	}

	return true;
}

static inline double Cubic(double x)
{
	double ax = std::fabs(x), ax2, ax3;
	ax2 = ax * ax;
	ax3 = ax2 * ax;

	if (ax <= 1)
		return 1.5 * ax3 - 2.5 * ax2 + 1;

	if (ax <= 2)
		return -0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2;

	return 0;
}

static inline void Norm(std::vector<double> &weights)
{
	double sum = 0;

	for (double w : weights)
		sum += w;;

	for (double &w : weights)
		w /= sum;
}

/**
* \brief
* \param image_data
* \param image_width
* \param image_height
* \param image_channels
* \param x
* \param y
* \param pixel size[image_channels]
*/
static void near_sampling(const uint8_t *image_data, int image_width, int image_height, int image_channels, int x, int y, uint8_t *pixel)
{
	if (x < 0)
		x = 0;

	if (x >= image_height)
		x = image_height - 1;

	if (y < 0)
		y = 0;

	if (y >= image_width)
		y = image_width - 1;

	int offset = (x * image_width + y) * image_channels;

	for (int c = 0; c < image_channels; ++c)
	{
		pixel[c] = image_data[offset + c];
	}
}

#define BICUBIC_KERNEL 4

/**
* \brief
* \param image_data
* \param image_width
* \param image_height
* \param image_channels
* \param scale
* \param x
* \param y
* \param pixel size[image_channels]
* \param weights_x
* \param weights_y
* \param indices_x
* \param indices_y
* \param type
*/
static void sampling(
	const uint8_t *image_data, int image_width, int image_height, int image_channels,
	double scale,
	double x, double y, uint8_t *pixel,
	std::vector<double> &weights_x, std::vector<double> &weights_y,
	std::vector<int> &indices_x, std::vector<int> &indices_y,
	SAMPLING_TYPE type = LINEAR,
	PADDING_TYPE ptype = ZERO_PADDING)
{
	if (type == LINEAR)
	{
		// bilinear subsampling
		int ux = static_cast<int>(std::floor(x)), uy = static_cast<int>(std::floor(y));

		if (ux >= 0 && ux < image_height - 1 && uy >= 0 && uy < image_width - 1)
		{
			double cof_x = x - ux;
			double cof_y = y - uy;

			for (int c = 0; c < image_channels; ++c)
			{
				double ans = 0;
				int offset = (ux * image_width + uy) * image_channels + c;
				ans = (1 - cof_y) * image_data[offset] + cof_y * image_data[offset + image_channels];
				ans = (1 - cof_x) * ans + cof_x * ((1 - cof_y) * image_data[offset + image_width * image_channels]
					+ cof_y * image_data[offset + image_width * image_channels + image_channels]);
				pixel[c] = static_cast<uint8_t>(std::max<double>(0.0f, std::min<double>(255.0f, ans)));
			}
		}
		else
		{
			switch (ptype)
			{
			default:
				memset(pixel, 0, sizeof(uint8_t) * image_channels);
				break;
			case NEAREST_PADDING:
				near_sampling(image_data, image_width, image_height, image_channels, ux, uy, pixel);
				break;
			}
		}
	}
	else
		if (type == BICUBIC)
		{
			// bicubic subsampling
			if (x >= 0 && x < image_height && y >= 0 && y < image_width)
			{
				scale = std::min<double>(scale, double(1.0));
				double kernel_width = std::max<double>(BICUBIC_KERNEL, BICUBIC_KERNEL / scale); // bicubic kernel width

				//std::vector<double> weights_x, weights_y;
				//std::vector<int>  indices_x, indices_y;
				//weights_x.reserve(kernel_width), indices_x.reserve(kernel_width);
				//weights_y.reserve(kernel_width), indices_y.reserve(kernel_width);

				weights_x.clear();
				indices_x.clear();
				weights_y.clear();
				indices_y.clear();

				// get indices and weight along x axis
				int ux_left = std::max<int>(0, static_cast<int>(std::ceil(x - kernel_width / 2)));
				int ux_right = std::min<int>(image_height - 1, static_cast<int>(std::floor(x + kernel_width / 2)));

				for (int ux = ux_left; ux <= ux_right; ++ux)
				{
					double weight = Cubic((x - ux) * scale);

					// if (weight == 0) continue;
					indices_x.push_back(ux);
					weights_x.push_back(weight);
				}

				// get indices and weight along y axis
				int uy_left = std::max<int>(0, static_cast<int>(std::ceil(y - kernel_width / 2)));
				int uy_right = std::min<int>(image_width - 1, static_cast<int>(std::floor(y + kernel_width / 2)));

				for (int uy = uy_left; uy <= uy_right; ++uy)
				{
					double weight = Cubic((y - uy) * scale);

					// if (weight == 0) continue;
					indices_y.push_back(uy);
					weights_y.push_back(weight);
				}

				// normalize the weights
				Norm(weights_x);
				Norm(weights_y);
				size_t lx = weights_x.size(), ly = weights_y.size();

				for (int c = 0; c < image_channels; ++c)
				{
					double ans = 0;
					double val = 0;

					for (size_t i = 0; i < lx; ++i)
					{
						val = 0;
						int offset = indices_x[i] * image_width * image_channels;

						for (size_t j = 0; j < ly; ++j)
						{
							val += image_data[offset + indices_y[j] * image_channels + c] * weights_y[j];
						}

						ans += val * weights_x[i];
					}

					pixel[c] = static_cast<uint8_t>(std::max<double>(0.0f, std::min<double>(255.0f, ans)));
				}
			}
			else
			{
				switch (ptype)
				{
				default:
					memset(pixel, 0, sizeof(uint8_t) * image_channels);
					break;
				case NEAREST_PADDING:
					near_sampling(image_data, image_width, image_height, image_channels, int(x), int(y), pixel);
					break;
				}
			}
		}
		else
		{
			int ux = static_cast<int>(x + 0.5), uy = static_cast<int>(y + 0.5);

			if (ux >= 0 && ux < image_height && uy >= 0 && uy < image_width)
			{
				int offset = (ux * image_width + uy) * image_channels;
				CopyData(pixel, &image_data[offset], image_channels);
			}
			else
			{
				switch (ptype)
				{
				default:
					memset(pixel, 0, sizeof(uint8_t) * image_channels);
					break;
				case NEAREST_PADDING:
					near_sampling(image_data, image_width, image_height, image_channels, ux, uy, pixel);
					break;
				}
			}
		}
}

/**
 * \brief
 * \param image_data
 * \param image_width
 * \param image_height
 * \param image_channels
 * \param crop_data
 * \param crop_width
 * \param crop_height
 * \param transformation
 * \param pad_top
 * \param pad_bottom
 * \param pad_left
 * \param pad_right
 * \param type
 * \param N always 1
 * \return
 */
static bool spatial_transform(
	const uint8_t *image_data, int image_width, int image_height, int image_channels,
	uint8_t *crop_data, int crop_width, int crop_height,
	const double *transformation,
	int pad_top = 0, int pad_bottom = 0, int pad_left = 0, int pad_right = 0,
	SAMPLING_TYPE type = LINEAR,
	PADDING_TYPE dtype = ZERO_PADDING,
	int N = 1)
{
	// const double *theta_data = transformation;
	// int src_w = image_width;
	// int src_h = image_height;
	int channels = image_channels;
	int dst_h = crop_height + pad_top + pad_bottom;
	int dst_w = crop_width + pad_left + pad_right;
	uint8_t *output_data = crop_data;

	//bool normalized_tform_ = false;   // @todo it does not work now

	std::vector<double> weights_x, weights_y;
	std::vector<int>  indices_x, indices_y;

	for (int n = 0; n < N; ++n)
	{
		const double *theta_data = transformation + n * TFORM_SIZE;
		double scale = std::sqrt(theta_data[0] * theta_data[0] + theta_data[3] * theta_data[3]);

		for (int x = 0; x < dst_h; ++x)
		{
			for (int y = 0; y < dst_w; ++y)
			{
				// Convet the point into crop axis
				int bx = x - pad_top;
				int by = y - pad_left;

				// Get the source position of each point on the destination feature map.
				double src_y = theta_data[0] * by + theta_data[1] * bx + theta_data[2];
				double src_x = theta_data[3] * by + theta_data[4] * bx + theta_data[5];

				uint8_t *current_channel_data = &output_data[n * dst_h * dst_w * channels + x * dst_w * channels + y * channels];
				sampling(image_data, image_width, image_height, image_channels, 1.0 / scale,
					src_x, src_y, current_channel_data,
					weights_x, weights_y, indices_x, indices_y,
					type,
					dtype);
			}
		}
	}

	return true;
}

/**
 * \brief
 * \param points
 * \param points_num
 * \param transformation
 * \param pad_top
 * \param pad_left
 * \param final_points
 * \return
 */
bool caculate_final_points(
	const float *points, int points_num,
	const double *transformation,
	int pad_top, int pad_left,
	float *final_points)
{
	const double *t = transformation;
	double t3t1_t0t4 = t[3] * t[1] - t[0] * t[4];

	if (t3t1_t0t4 < FLT_EPSILON && t3t1_t0t4 > -FLT_EPSILON)
		t3t1_t0t4 = FLT_EPSILON * 2;

	for (int i = 0; i < points_num; ++i)
	{
		float x = points[2 * i];
		float y = points[2 * i + 1];

		double fy = ((t[3] * x - t[0] * y) - (t[3] * t[2] - t[0] * t[5])) / t3t1_t0t4 + pad_top;
		double fx = ((t[1] * y - t[4] * x) - (t[1] * t[5] - t[4] * t[2])) / t3t1_t0t4 + pad_left;

		final_points[2 * i] = static_cast<float>(fx);
		final_points[2 * i + 1] = static_cast<float>(fy);
	}

	return true;
}

bool face_crop_core_ex(
	const uint8_t *image_data, int image_width, int image_height, int image_channels,
	uint8_t *crop_data, int crop_width, int crop_height,
	const float *points, int points_num,
	const float *mean_shape, int mean_shape_width, int mean_shape_height,
	int pad_top, int pad_bottom, int pad_left, int pad_right,
	float *final_points,
	SAMPLING_TYPE type,
	PADDING_TYPE ptype)
{
	//std::unique_ptr<double[]> transformation(new double[TFORM_SIZE]);
	double transformation[TFORM_SIZE];
	bool check1 = transformation_maker(
		crop_width, crop_height,
		points, points_num, mean_shape, mean_shape_width, mean_shape_height,
		transformation);

	if (!check1)
		return false;

	bool check2 = spatial_transform(image_data, image_width, image_height, image_channels,
		crop_data, crop_width, crop_height,
		transformation,
		pad_top, pad_bottom, pad_left, pad_right,
		type,
		ptype);

	if (!check2)
		return false;

	bool check3 = true;

	if (final_points)
	{
		check3 = caculate_final_points(points, points_num, transformation, pad_top, pad_left, final_points);
	}

	if (!check3)
		return false;

	return true;
}

bool face_crop_core(
	const uint8_t *image_data, int image_width, int image_height, int image_channels,
	uint8_t *crop_data, int crop_width, int crop_height,
	const float *points, int points_num,
	const float *mean_shape, int mean_shape_width, int mean_shape_height,
	int pad_top, int pad_bottom, int pad_left, int pad_right,
	float *final_points,
	SAMPLING_TYPE type)
{
	return face_crop_core_ex(
		image_data, image_width, image_height, image_channels,
		crop_data, crop_width, crop_height,
		points, points_num,
		mean_shape, mean_shape_width, mean_shape_height,
		pad_top, pad_bottom, pad_left, pad_right,
		final_points,
		type,
		ZERO_PADDING);
}
