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
