#pragma once

#include "Struct.h"
#include "CTrackingFaceInfo.h"
#include <string>
#include <vector>

namespace seeta {
    namespace v2 {
        class FaceTracker {
        public:
            /**
             * \brief initialize FaceTracker with face detector model
             * \param setting model used by FaceDetector5.1.0
             * \param video_width input video frame width
             * \param video_height input video frame height
             */
            SEETA_API explicit FaceTracker(const SeetaModelSetting &setting);

            SEETA_API explicit FaceTracker(const SeetaModelSetting &setting, int core_width, int core_height);

            SEETA_API ~FaceTracker();

            SEETA_API SeetaTrackingFaceInfoArray track(const SeetaImageData &image, int frame_no = -1) const;

            enum Property {
                PROPERTY_MIN_FACE_SIZE,
                PROPERTY_THRESHOLD1,
                PROPERTY_THRESHOLD2,
                PROPERTY_THRESHOLD3,
                PROPERTY_VIDEO_STABLE,
            };

            /**
             * \brief set property
             * \param [in] property face detector property to set
             * \param [in] value value of corresponding property to set
             */
            SEETA_API void set(Property property, double value);

            /**
             * \brief get property
             * \param [in] property to get
             * \return property value to get
             */
            SEETA_API double get(Property property) const;

        private:
            FaceTracker(const FaceTracker &other) = delete;

            const FaceTracker &operator=(const FaceTracker &other) = delete;

        private:
            class Implement;

            Implement *m_impl;
        };
    }
    using namespace v2;
}

