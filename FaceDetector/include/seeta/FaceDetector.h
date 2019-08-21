//
// Created by kier on 19-4-24.
//

#ifndef INC_SEETA_FACEDETECTOR_H
#define INC_SEETA_FACEDETECTOR_H

#include "Struct.h"
#include "CFaceInfo.h"
#include "SeetaDetectorExport.h"

namespace seeta
{
    namespace v2
    {

        class FaceDetector {
        public:
            enum Property
            {
                PROPERTY_MIN_FACE_SIZE,
                PROPERTY_THRESHOLD1,
                PROPERTY_THRESHOLD2,
                PROPERTY_THRESHOLD3,
                PROPERTY_VIDEO_STABLE,
            };
            /**
             * \brief load model
             * \param [in] setting model file
             */
            SEETA_DETECTOR_API explicit FaceDetector( const SeetaModelSetting &setting );

            /**
             * \brief load model
             * \param [in] setting model file
             * \param [in] core_width width of calculation core
             * \param [in] core_height height of calculation core
             */
            SEETA_DETECTOR_API explicit FaceDetector( const SeetaModelSetting &setting, int core_width, int core_height );
            SEETA_DETECTOR_API ~FaceDetector();

            /**
             * \brief detect faces
             * \param [in] image image data to input
             * \return detected faces info array
             */
            SEETA_DETECTOR_API SeetaFaceInfoArray detect( const SeetaImageData &image ) const;

            /**
             * \brief set property
             * \param [in] property face detector property to set
             * \param [in] value value of corresponding property to set
             */
            SEETA_DETECTOR_API void set( Property property, double value );

            /**
             * \brief get property
             * \param [in] property to get
             * \return property value to get
             */
            SEETA_DETECTOR_API double get( Property property ) const;

        private:
            FaceDetector( const FaceDetector & ) = delete;
            const FaceDetector &operator=( const FaceDetector & ) = delete;

        private:
            void *m_impl;
        };
    }
    using namespace v2;
}

#endif //INC_SEETA_FACEDETECTOR_H
