//
// Created by seetadev on 2019/10/31.
//

#ifndef SEETAFACE_POSEQUALITY_H
#define SEETAFACE_POSEQUALITY_H

#include <seeta/Struct.h>

/**
 *
 * @param image original image
 * @param face face position
 * @param points 5 landmarks
 * @param roll
 * @param yaw
 * @param pitch
 */
void evaluate_pose(const SeetaImageData &image, const SeetaRect &face, const SeetaPointF *points,
	float &roll, float &yaw, float &pitch);

#endif //SEETAFACE_POSEQUALITY_H
