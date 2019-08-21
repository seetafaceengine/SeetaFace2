//
// Created by kier on 19-4-24.
//

#ifndef SEETA_FACERECOGNIZER_FACERECOGNIZER_H
#define SEETA_FACERECOGNIZER_FACERECOGNIZER_H

#include "CStruct.h"
#include "SeetaRecognizerExport.h"

namespace seeta
{
    namespace v2
    {
        class FaceRecognizer {
        public:
            /**
             * \brief load model
             * \param [in] setting model file
             */
            SEETA_RECOGNIZER_API explicit FaceRecognizer( const SeetaModelSetting &setting );
            SEETA_RECOGNIZER_API ~FaceRecognizer();

            /**
             * \brief get cropped face width
             * \return cropped face width
             */
            SEETA_RECOGNIZER_API int GetCropFaceWidth();

            /**
             * \brief get cropped face height
             * \return cropped face height
             */
            SEETA_RECOGNIZER_API int GetCropFaceHeight();

            /**
             * \brief get cropped face channels
             * \return cropped face channels
             */
            SEETA_RECOGNIZER_API int GetCropFaceChannels();

            /**
             * \brief crop face
             * \param [in] image image data to input
             * \param [in] points face landmars
             * \param [out] face cropped face data
             */
            SEETA_RECOGNIZER_API bool CropFace( const SeetaImageData &image, const SeetaPointF *points, SeetaImageData &face );


            /**
             * \brief get extracted feature size
             * \return extracted feature size
             */
            SEETA_RECOGNIZER_API int GetExtractFeatureSize() const;

            /**
             * \brief extract cropped face image's feature
             * \param [in] image face image data to input
             * \param [out] features extracted features of face
             * \return true if extract feature success
             */
            SEETA_RECOGNIZER_API bool ExtractCroppedFace( const SeetaImageData &image, float *features ) const;

            /**
             * \brief extract face feature in origin image
             * \param [in] image origin image data to input
             * \param [in] points face landmarks
             * \param [out] feature face feature extracted
             * \return true if extract feature success
             */
            SEETA_RECOGNIZER_API bool Extract( const SeetaImageData &image, const SeetaPointF *points, float *features ) const;

            /**
             * \brief calculate two faces's similarity
             * \param [in] features1 face1 features
             * \param [in] features2 face2 features
             * \return two faces similarity
             */
            SEETA_RECOGNIZER_API float CalculateSimilarity( const float *features1, const float *features2 ) const;


        private:
            FaceRecognizer( const FaceRecognizer & ) = delete;
            const FaceRecognizer &operator=( const FaceRecognizer & ) = delete;

        private:
            void *m_impl;
        };
    }
    using namespace v2;
}

#endif //SEETA_FACERECOGNIZER_FACERECOGNIZER_H
