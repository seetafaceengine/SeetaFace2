#ifndef SEETA_FACELANDMARKER_FACELANDMARKER_H
#define SEETA_FACELANDMARKER_FACELANDMARKER_H

#include "Struct.h"
#include "SeetaLandmarkerExport.h"

namespace seeta
{
    namespace v2
    {
        class FaceLandmarker {
        public:
            /**
             * \brief load model
             * \param [in] setting model file
             * \return
             */
            SEETA_LANDMARKER_API explicit FaceLandmarker( const SeetaModelSetting &setting );
            SEETA_LANDMARKER_API ~FaceLandmarker();

            /**
             * \brief get face landmarks number
             * \return face landmarks number
             */
            SEETA_LANDMARKER_API int number() const;

            /**
             * \brief detect face landmars of corresponding face
             * \param [in] image image data to input
             * \param [in] face face location
             * \param [out] points face landmarks
             */
            SEETA_LANDMARKER_API void mark( const SeetaImageData &image, const SeetaRect &face, SeetaPointF *points ) const;

            /**
             * \brief detect face landmars of corresponding face
             * \param [in] image image data to input
             * \param [in] face face location
             * \return face landmarks
             */
            std::vector<SeetaPointF> mark( const SeetaImageData &image, const SeetaRect &face ) const {
                std::vector<SeetaPointF> points( this->number() );
                mark( image, face, points.data() );
                return points;
            }

        private:
            FaceLandmarker( const FaceLandmarker & ) = delete;
            const FaceLandmarker &operator=( const FaceLandmarker & ) = delete;

        private:

            void *m_impl;
        };
    }
    using namespace v2;
}

#endif //SEETA_FACELANDMARKER_FACELANDMARKER_H
