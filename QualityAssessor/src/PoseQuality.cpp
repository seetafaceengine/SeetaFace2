//
// Created by seetadev on 2019/10/31.
//

#include "PoseQuality.h"

#include <cmath>
#include <cfloat>
#include <climits>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


static SeetaPointF operator+(const SeetaPointF &lhs, const SeetaPointF &rhs) {
    SeetaPointF result;
    result.x = lhs.x + rhs.x;
    result.y = lhs.y + rhs.y;

    return result;
}

static SeetaPointF operator-(const SeetaPointF &lhs, const SeetaPointF &rhs) {
    SeetaPointF result;
    result.x = lhs.x - rhs.x;
    result.y = lhs.y - rhs.y;

    return result;
}

static SeetaPointF operator/(const SeetaPointF &lhs, double rhs) {
    SeetaPointF result;
    result.x = lhs.x / rhs;
    result.y = lhs.y / rhs;

    return result;
}

static SeetaPointF operator*(const SeetaPointF &lhs, double rhs) {
    SeetaPointF result;
    result.x = lhs.x * rhs;
    result.y = lhs.y * rhs;

    return result;
}

static double operator^(const SeetaPointF &lhs, const SeetaPointF &rhs) {
    auto dx = lhs.x - rhs.x;
    auto dy = lhs.y - rhs.y;
    return std::sqrt(dx * dx + dy * dy);
}

/**
 * line for ax + by + c = 0
 */
class Line {
public:
    Line() = default;
    Line(double a, double b, double c)
            : a(a), b(b), c(c) {}

    Line(const SeetaPointF &a, const SeetaPointF &b) {
        auto x1 = a.x;
        auto y1 = a.y;
        auto x2 = b.x;
        auto y2 = b.y;
        // for (y2-y1)x-(x2-x1)y-x1(y2-y1)+y1(x2-x1)=0
        this->a = y2 - y1;
        this->b = x1 - x2;
        this->c = y1 * (x2 - x1) - x1 * (y2 - y1);
    }

    double distance(const SeetaPointF &p) const {
        return std::fabs(a * p.x + b * p.y + c) / std::sqrt(a * a + b * b);
    }

    static bool near_zero(double f) {
        return f <= DBL_EPSILON && -f <= DBL_EPSILON;
    }

    SeetaPointF projection(const SeetaPointF &p) const {
        if (near_zero(a)) {
            SeetaPointF result;
            result.x = p.x;
            result.y = -c / b;
            return  result;
        }
        if (near_zero(b)) {
            SeetaPointF result;
            result.x = -c / a;
            result.y = p.y;
            return result;
        }
        // y = kx + b  <==>  ax + by + c = 0
        auto k = -a / b;
        SeetaPointF o = {0, -c / b};
        SeetaPointF project = {0};
        project.x = (float) ((p.x / k + p.y - o.y) / (1 / k + k));
        project.y = (float) (-1 / k * (project.x - p.x) + p.y);
        return project;
    }

    double a = 0;
    double b = 0;
    double c = 0;
};


void
evaluate_pose(const SeetaImageData &image, const SeetaRect &face, const SeetaPointF *points, float &roll, float &yaw,
              float &pitch) {
    static const float nose_center = 0.5f;
    // static const float roll0 = 1 / 6.0f;
    // static const float yaw0 = 0.2f;
    // static const float pitch0 = 0.2f;

    auto point_center_eye = (points[0] + points[1]) / 2;
    auto point_center_mouth = (points[3] + points[4]) / 2;

    Line line_eye_mouth(point_center_eye, point_center_mouth);

    auto vector_left2right = points[1] - points[0];

    auto rad = atan2(vector_left2right.y, vector_left2right.x);
    auto angle = rad * 180 * M_PI;

    auto roll_dist = fabs(angle) / 180;

    auto raw_yaw_dist = line_eye_mouth.distance(points[2]);
    auto yaw_dist = raw_yaw_dist / (points[0] ^ points[1]);

    auto point_suppose_projection = point_center_eye * nose_center + point_center_mouth * (1 - nose_center);
    auto point_projection = line_eye_mouth.projection(points[2]);
    auto raw_pitch_dist = point_projection ^ point_suppose_projection;
    auto pitch_dist = raw_pitch_dist / (point_center_eye ^ point_center_mouth);

    roll = roll_dist;
    yaw = yaw_dist;
    pitch = pitch_dist;
}
