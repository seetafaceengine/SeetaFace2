#pragma once

#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>
#include <seeta/FaceDatabase.h>

namespace seeta
{
    class FaceEngine {
    public:
        FaceEngine( const SeetaModelSetting &FD_model, const SeetaModelSetting &PD_model, const SeetaModelSetting &FR_model )
            : FD( FD_model ), PD( PD_model ), FDB( FR_model ) {
        }
        FaceEngine( const SeetaModelSetting &FD_model, const SeetaModelSetting &PD_model, const SeetaModelSetting &FR_model,
                    int extraction_core_number, int comparation_core_number )
            : FD( FD_model ), PD( PD_model ), FDB( FR_model ) {
        }
        ~FaceEngine() = default;

        static void SetLogLevel( int level ) {
            //seeta::FaceDetector::SetLogLevel(level);
            //seeta::FaceDatabase::SetLogLevel(level);
        }

        static void SetSingleCalculationThreads( int num ) {
            //seeta::FaceDetector::SetSingleCalculationThreads(num);
            //seeta::FaceDatabase::SetSingleCalculationThreads(num);
        }

        std::vector<SeetaFaceInfo> DetectFaces( const SeetaImageData &image ) const {
            auto faces = FD.detect( image );
            return std::vector<SeetaFaceInfo>( faces.data, faces.data + faces.size );
        }

        std::vector<SeetaPointF> DetectPoints( const SeetaImageData &image, const SeetaRect &face ) const {
            std::vector<SeetaPointF> points( PD.number() );
            PD.mark( image, face, points.data() );
            return std::move( points );
        }

        std::vector<SeetaPointF> DetectPoints( const SeetaImageData &image, const SeetaFaceInfo &face ) const {
            return this->DetectPoints( image, face.pos );
        }

        std::vector<SeetaPointF> DetectPoints( const SeetaImageData &image ) const {
            auto faces = this->DetectFaces( image );
            if( faces.empty() ) return std::vector<SeetaPointF>();
            return this->DetectPoints( image, faces[0] );
        }

        float Compare(
            const SeetaImageData &image1, const SeetaRect &face1,
            const SeetaImageData &image2, const SeetaRect &face2 ) const {
            auto points1 = this->DetectPoints( image1, face1 );
            auto points2 = this->DetectPoints( image2, face2 );
            return FDB.Compare( image1, points1.data(), image2, points2.data() );
        }

        float Compare(
            const SeetaImageData &image1, const SeetaFaceInfo &face1,
            const SeetaImageData &image2, const SeetaFaceInfo &face2 ) const {
            return this->Compare( image1, face1.pos, image2, face2.pos );
        }

        float Compare( const SeetaImageData &image1, const SeetaImageData &image2 ) const {
            auto faces1 = this->DetectFaces( image1 );
            if( faces1.empty() ) return 0;
            auto faces2 = this->DetectFaces( image2 );
            if( faces2.empty() ) return 0;
            return this->Compare( image1, faces1[0], image2, faces2[0] );
        }

        int64_t Register( const SeetaImageData &image ) {
            auto faces = this->DetectFaces( image );
            if( faces.empty() ) return -1;
            return this->Register( image, faces[0] );
        }

        int64_t Register( const SeetaImageData &image, const SeetaFaceInfo &face ) {
            return this->Register( image, face.pos );
        }

        int64_t Register( const SeetaImageData &image, const SeetaRect &face ) {
            auto points = this->DetectPoints( image, face );
            return FDB.Register( image, points.data() );
        }

        int64_t Query( const SeetaImageData &image, float *similarity = nullptr ) const {
            auto faces = this->DetectFaces( image );
            if( faces.empty() ) return -1;
            return this->Query( image, faces[0], similarity );
        }

        int64_t Query( const SeetaImageData &image, const SeetaFaceInfo &face, float *similarity = nullptr ) const {
            return this->Query( image, face.pos, similarity );
        }

        int64_t Query( const SeetaImageData &image, const SeetaRect &face, float *similarity = nullptr ) const {
            auto points = DetectPoints( image, face );
            return FDB.Query( image, points.data(), similarity );
        }

        size_t QueryTop( const SeetaImageData &image, size_t N, int64_t *index, float *similarity ) const {
            auto faces = this->DetectFaces( image );
            if( faces.empty() ) return -1;
            return this->QueryTop( image, faces[0], N, index, similarity );
        }

        size_t QueryTop( const SeetaImageData &image, const SeetaFaceInfo &face, size_t N, int64_t *index, float *similarity ) const {
            return this->QueryTop( image, face.pos, N, index, similarity );
        }

        size_t QueryTop( const SeetaImageData &image, const SeetaRect &face, size_t N, int64_t *index, float *similarity ) const {
            auto points = this->DetectPoints( image, face );
            return FDB.QueryTop( image, points.data(), N, index, similarity );
        }

        /*
        void RegisterParallel(const SeetaImageData &image, int64_t *index)
        {
            auto faces = this->DetectFaces(image);
            if (faces.empty()) return;
            this->RegisterParallel(image, faces[0], index);
        }

        void RegisterParallel(const SeetaImageData &image, const SeetaFaceInfo &face, int64_t *index)
        {
            this->RegisterParallel(image, face.pos, index);
        }

        void RegisterParallel(const SeetaImageData &image, const SeetaRect &face, int64_t *index)
        {
            auto points = this->DetectPoints(image, face);
            FDB.RegisterParallel(image, points.data(), index);
        }
        */
        float Compare(
            const SeetaImageData &image1, const SeetaPointF *points1,
            const SeetaImageData &image2, const SeetaPointF *points2 ) const {
            return FDB.Compare( image1, points1, image2, points2 );
        }

        int64_t Register( const SeetaImageData &image, const SeetaPointF *points ) {
            return FDB.Register( image, points );
        }

        int Delete( int64_t index ) {
            return FDB.Delete( index );
        }

        void Clear() {
            FDB.Clear();
        }

        size_t Count() const {
            return FDB.Count();
        }

        int64_t Query( const SeetaImageData &image, const SeetaPointF *points, float *similarity = nullptr ) const {
            return FDB.Query( image, points, similarity );
        }

        size_t QueryTop( const SeetaImageData &image, const SeetaPointF *points, size_t N, int64_t *index, float *similarity ) const {
            return FDB.QueryTop( image, points, N, index, similarity );
        }

        /*
        void RegisterParallel(const SeetaImageData &image, const SeetaPointF *points, int64_t *index)
        {
            FDB.RegisterParallel(image, points, index);
        }

        void Join() const
        {
            FDB.Join();
        }
        */
        bool Save( const char *path ) const {
            return FDB.Save( path );
        }

        bool Load( const char *path ) {
            return FDB.Load( path );
        }

        bool Save( StreamWriter &writer ) const {
            return FDB.Save( writer );
        }
        bool Load( StreamReader &reader ) {
            return FDB.Load( reader );
        }

        FaceEngine( const FaceEngine & ) = delete;
        const FaceEngine &operator=( const FaceEngine & ) = delete;

        FaceDetector FD;
        FaceLandmarker PD;
        FaceDatabase FDB;
    };
}
