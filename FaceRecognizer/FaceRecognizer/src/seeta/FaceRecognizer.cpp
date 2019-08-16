//
// Created by kier on 19-4-24.
//

#include "seeta/FaceRecognizer.h"
#include "seeta/FaceRecognizerPrivate.h"


namespace seeta
{
    namespace v2
    {

        FaceRecognizer::FaceRecognizer( const SeetaModelSetting &setting )
            : m_impl( new FaceRecognizerPrivate( setting.model[0], setting.device, setting.id ) )
        {

        }

        FaceRecognizer::~FaceRecognizer()
        {
            FaceRecognizerPrivate *ptr = ( FaceRecognizerPrivate * )m_impl;
            delete ptr;
            m_impl = nullptr;
        }

        int FaceRecognizer::GetCropFaceWidth()
        {

            FaceRecognizerPrivate *ptr = ( FaceRecognizerPrivate * )m_impl;
            return ptr->GetCropWidth();
        }

        int FaceRecognizer::GetCropFaceHeight()
        {

            FaceRecognizerPrivate *ptr = ( FaceRecognizerPrivate * )m_impl;
            return ptr->GetCropHeight();
        }

        int FaceRecognizer::GetCropFaceChannels()
        {

            FaceRecognizerPrivate *ptr = ( FaceRecognizerPrivate * )m_impl;
            return ptr->GetCropChannels();
        }

        int FaceRecognizer::GetExtractFeatureSize() const
        {

            FaceRecognizerPrivate *ptr = ( FaceRecognizerPrivate * )m_impl;
            return ptr->GetFeatureSize();
        }

        bool FaceRecognizer::ExtractCroppedFace( const SeetaImageData &image, float *features ) const
        {

            FaceRecognizerPrivate *ptr = ( FaceRecognizerPrivate * )m_impl;
            return ptr->ExtractFeatureNormalized( image, features );
        }

        float FaceRecognizer::CalculateSimilarity( const float *features1, const float *features2 ) const
        {

            FaceRecognizerPrivate *ptr = ( FaceRecognizerPrivate * )m_impl;
            return ptr->CalcSimilarityNormalized( features1, features2 );
        }

        bool FaceRecognizer::Extract( const SeetaImageData &image, const SeetaPointF *points, float *features ) const
        {
            if( features == nullptr ) return false;

            FaceRecognizerPrivate *ptr = ( FaceRecognizerPrivate * )m_impl;
            return ptr->ExtractFeatureWithCropNormalized( image, points, features );
        }

        bool FaceRecognizer::CropFace( const SeetaImageData &image, const SeetaPointF *points, SeetaImageData &face )
        {

            if( points == nullptr ) return false;

            FaceRecognizerPrivate *ptr = ( FaceRecognizerPrivate * )m_impl;
            return ptr->CropFace( image, points, face );
        }
    }

}
