#include "seeta/FaceLandmarker.h"
#include "common_alignment.h"

#include <fstream>
#include <cfloat>
#include <cmath>

#include <CStruct.h>

#include "FaceLandmarkerPrivate.h"

namespace seeta
{
    namespace v2
    {

        FaceLandmarker::FaceLandmarker( const SeetaModelSetting &setting )
            : m_impl( new FaceLandmarkerPrivate( setting.model[0], setting.device, setting.id ) )
        {

        }

        FaceLandmarker::~FaceLandmarker()
        {
            FaceLandmarkerPrivate *ptr = ( FaceLandmarkerPrivate * )m_impl;
            delete ptr;
            ptr = nullptr;
        }

        int FaceLandmarker::number() const
        {

            FaceLandmarkerPrivate *ptr = ( FaceLandmarkerPrivate * )m_impl;
            return ptr->LandmarkNum();
        }

        void FaceLandmarker::mark( const SeetaImageData &image, const SeetaRect &face, SeetaPointF *points ) const
        {
            FaceLandmarkerPrivate *ptr = ( FaceLandmarkerPrivate * )m_impl;
            bool result = ptr->PointDetectLandmarks( image, face, points, nullptr );
        }
    }
}
