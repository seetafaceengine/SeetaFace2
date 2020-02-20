//
// Created by kier on 2019-09-16.
//

#include <iostream>
#include "seeta/QualityAssessor.h"

#include "seeta/DataHelper.h"
#include "seeta/ImageProcess.h"

#include "PoseQuality.h"
#include "ClarityQuality.h"

namespace seeta
{
	namespace v2
	{
		QualityAssessor::QualityAssessor()
		{
			setFaceSize(80);
		}

		QualityAssessor::~QualityAssessor()
		{
		}

		int QualityAssessor::setFaceSize(int size)
		{
			m_FaceSize = size;
			return 0;
		}

		int QualityAssessor::getFaceSize() const
		{
			return m_FaceSize;
		}

		/**
		 * evaluate lightness of image
		 * @param image base image
		 * @param face face position
		 * @return lightning score [0, 255]
		 */
		float evaluate_lightness(const SeetaImageData &image, const SeetaRect &face)
		{
			auto patch = seeta::crop(image, face);
			patch = seeta::gray(patch);

			auto count = patch.width() * patch.height();
			double sum = 0;

			for (int i = 0; i < count; ++i)
			{
				sum += patch.data(i);
			}

			return float(sum / count);
		}

		/**
		 * check if lightness if ok
		 * @param image original image
		 * @param face face position
		 * @param low low lightness bound
		 * @param high height lightness bound
		 * @return return true if lightness in (low, high)
		 */
		bool check_lightness(const SeetaImageData &image, const SeetaRect &face, float low = 40, float high = 180)
		{
			auto score = evaluate_lightness(image, face);

			// std::cout << "lightness: " << score << std::endl;
			return score > low && score < high;
		}

		/**
		 * check if face width ok
		 * @param face face position
		 * @param size min face size
		 * @return return true if face.width > size
		 */
		bool check_face_size(const SeetaRect &face, int size = 80)
		{
			// std::cout << "face.width: " << face.width << std::endl;
			return face.width > size;
		}

		/**
		 *
		 * @param image original image
		 * @param face face position
		 * @param points 5 landmarks
		 * @return if pose ok
		 */
		bool check_pose(const SeetaImageData &image, const SeetaRect &face, const SeetaPointF *points)
		{
			static const float roll0 = 1 / 3.0f;
			static const float yaw0 = 0.5f;
			static const float pitch0 = 0.5f;
			float roll, yaw, pitch;
			evaluate_pose(image, face, points, roll, yaw, pitch);

			//  std::cout << "roll: " << roll << ", " << "yaw: " << yaw << ", " << "pitch: " << pitch << std::endl;
			return roll < roll0 && yaw < yaw0 && pitch < pitch0;
		}

		/**
		 *
		 * @param image original image
		 * @param face face position
		 * @param score output clarity score
		 * @return if clarity > thresh
		 */
		bool check_clarity(const SeetaImageData &image, const SeetaRect &face, float &score)
		{
			static const float thresh = 0.3f;
			score = evaluate_clarity(image, face);
			// std::cout << "clarity: " << score << std::endl;
			return score > thresh;
		}

		float QualityAssessor::evaluate(const SeetaImageData &image, const SeetaRect &face, const SeetaPointF *points) const
		{
			// std::cout << "=============================" << std::endl;
			float clarity;

			if (check_lightness(image, face)
				&& check_face_size(face, getFaceSize())
				&& check_pose(image, face, points)
				&& check_clarity(image, face, clarity))
			{
				return clarity;
			}
			else
			{
				return 0;
			}
		}

		int QualityAssessor::evaluate(const SeetaImageData &image,
			const SeetaRect &face,
			const SeetaPointF *points,
			float &score) const
		{
			int ret = ERROR_OK;

			if (!check_lightness(image, face))
				ret |= ERROR_LIGHTNESS;

			if (!check_face_size(face, getFaceSize()))
				ret |= ERROR_FACE_SIZE;

			if (!check_pose(image, face, points))
				ret |= ERROR_FACE_POSE;

			if (!check_clarity(image, face, score))
				ret |= ERROR_CLARITY;

			return ret;
		}
	}
}
