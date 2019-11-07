#ifndef _INC_SEETA_QUALITY_ASSESSOR_H
#define _INC_SEETA_QUALITY_ASSESSOR_H

#include "Struct.h"

namespace seeta
{
    namespace v2
    {
        class QualityAssessor {
        public:
            /**
             * \brief load model
             * \return
             */
            SEETA_API explicit QualityAssessor();
            SEETA_API ~QualityAssessor();

            /**
             * \brief evaluate score
             * \param image original image
             * \param face face position
             * \param points 5 landmarks
             * \return 0 if not satisfied, of return score
             */
            SEETA_API float evaluate(const SeetaImageData &image,
                    const SeetaRect &face,
                    const SeetaPointF *points) const;
            enum ERROR_CODE
            {
                ERROR_OK        =  0,
                ERROR_LIGHTNESS =  0x01,
                ERROR_FACE_SIZE =  0x02,
                ERROR_FACE_POSE =  0x04,
                ERROR_CLARITY   =  0x08
            };
            /**
             * @brief evaluate score
             * @param image original image
             * @param face face position
             * @param points 5 landmarks
             * @param clarity 
             * @return ERROR_OK if is ok. other reture ERROR_CODE combination.
             */
            SEETA_API int evaluate(const SeetaImageData &image,
                                   const SeetaRect &face,
                                   const SeetaPointF *points, float &score) const;
            SEETA_API int setFaceSize(int size);
            SEETA_API int getFaceSize() const;

        private:
            QualityAssessor( const QualityAssessor & ) = delete;
            const QualityAssessor &operator=( const QualityAssessor & ) = delete;

        private:
            class Implement;
            Implement *m_impl;
            int m_FaceSize;
        };
    }
    using namespace v2;
}

#endif //_INC_SEETA_QUALITY_ASSESSOR_H
