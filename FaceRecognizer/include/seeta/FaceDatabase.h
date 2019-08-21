#ifndef _SEETANET_FACEDATABASE_H_
#define _SEETANET_FACEDATABASE_H_
#include "Struct.h"
#include "Stream.h"
#include "SeetaRecognizerExport.h"

#include <string>
#include <vector>

namespace seeta
{
    namespace v2
    {
        class FaceRecognizer;
        /**
         * \brief Only support single thread running
         */
        class FaceDatabase {
        public:
            SEETA_RECOGNIZER_API explicit FaceDatabase( const SeetaModelSetting &setting );
            SEETA_RECOGNIZER_API ~FaceDatabase();



            SEETA_RECOGNIZER_API int GetCropFaceWidth();
            SEETA_RECOGNIZER_API int GetCropFaceHeight();
            SEETA_RECOGNIZER_API int GetCropFaceChannels();

            SEETA_RECOGNIZER_API bool CropFace( const SeetaImageData &image, const SeetaPointF *points, SeetaImageData &face );

            SEETA_RECOGNIZER_API float Compare(
                const SeetaImageData &image1, const SeetaPointF *points1,
                const SeetaImageData &image2, const SeetaPointF *points2 ) const;

            SEETA_RECOGNIZER_API float CompareByCroppedFace(
                const SeetaImageData &cropped_face_image1,
                const SeetaImageData &cropped_face_image2 ) const;

            SEETA_RECOGNIZER_API int64_t Register( const SeetaImageData &image, const SeetaPointF *points );
            SEETA_RECOGNIZER_API int64_t RegisterByCroppedFace( const SeetaImageData &cropped_face_image );
            SEETA_RECOGNIZER_API int Delete( int64_t index );  // return effected lines, 1 for succeed, 0 for nothing
            SEETA_RECOGNIZER_API void Clear(); // clear all faces

            SEETA_RECOGNIZER_API size_t Count() const;
            SEETA_RECOGNIZER_API int64_t Query( const SeetaImageData &image, const SeetaPointF *points, float *similarity = nullptr ) const;  // return max index
            SEETA_RECOGNIZER_API int64_t QueryByCroppedFace( const SeetaImageData &cropped_face_image, float *similarity = nullptr ) const;  // return max index
            /**
             * \biref query top N face
             * \param image The original image
             * \param points Pointer to 5 SeetaPoints
             * \param N Querying top N
             * \param [out] index Pointer to array saving result, have at least N space.
             * \param [out] similarity Pointer to array saving result, have at least N space.
             * \return Queried face number
             */
            SEETA_RECOGNIZER_API size_t QueryTop( const SeetaImageData &image, const SeetaPointF *points, size_t N, int64_t *index, float *similarity ) const;  // return top N faces
            SEETA_RECOGNIZER_API size_t QueryTopByCroppedFace( const SeetaImageData &cropped_face_image, size_t N, int64_t *index, float *similarity ) const;  // return top N faces

            /**
            * \biref query all faces similarity greater than threhold
            * \param image The original image
            * \param points Pointer to 5 SeetaPoints
            * \param threshold Threshold
            * \param N size of `index` and `similarity`
            * \param [out] index Pointer to array saving result, have at least N space.
            * \param [out] similarity Pointer to array saving result, have at least N space.
            * \return Queried face number
            */
            SEETA_RECOGNIZER_API size_t QueryAbove( const SeetaImageData &image, const SeetaPointF *points, float threshold, size_t N, int64_t *index, float *similarity ) const;
            SEETA_RECOGNIZER_API size_t QueryAboveByCroppedFace( const SeetaImageData &cropped_face_image, float threshold, size_t N, int64_t *index, float *similarity ) const;


            SEETA_RECOGNIZER_API bool Save( const char *path ) const;
            SEETA_RECOGNIZER_API bool Load( const char *path );

            SEETA_RECOGNIZER_API bool Save( StreamWriter &writer ) const;
            SEETA_RECOGNIZER_API bool Load( StreamReader &reader );

            SEETA_RECOGNIZER_API FaceRecognizer *ExtractionCore( int i = 0 );

        private:
            FaceDatabase( const FaceDatabase &other ) = delete;
            const FaceDatabase &operator=( const FaceDatabase &other ) = delete;

        private:
            class Implement;
            Implement *m_impl;
        };
    }
    using namespace v2;
}

#endif
