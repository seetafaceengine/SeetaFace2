#include "FaceDatabase.h"
#include "FaceRecognizer.h"

#include <array>
#include <cmath>

#include <iostream>
#include <fstream>
#include <map>
#include "Mutex.h"
#include <stack>
#include "seeta/common_alignment.h"



namespace seeta
{
    namespace v2
    {
        class FaceDatabase::Implement {
        public:
            using self = Implement;

            Implement( const SeetaModelSetting &setting ) {
                seeta::ModelSetting exciting = setting;
                auto models = exciting.get_model();
                if( models.size() != 1 ) {
                    std::cout << "FaceDatabase Must input 1 model." << std::endl;
                    exit( -1 );
                }
                // std::cout << "FaceDatabase Loading models..." << std::endl;

                std::string model_filename = models[0];
                m_cores.resize( 1 );
                for( auto &core : m_cores ) {
                    core = std::make_shared<seeta::FaceRecognizer>( setting );
                }
                m_main_core = m_cores[0];

            }

            FaceRecognizer &core() {
                return *m_main_core;
            }
            const FaceRecognizer &core() const {
                return *m_main_core;
            }

            bool ExtractCroppedFace( const SeetaImageData &image, float *features ) const {
                m_cores[0]->ExtractCroppedFace( image, features );
                return true;
            }


            bool Extract( const SeetaImageData &image, const SeetaPointF *points, float *features ) const {
                m_cores[0]->Extract( image, points, features );
                return true;
            }

            bool Compare( const float *features1, const float *features2, float *similarity ) const {
                if( !features1 || !features2 || !similarity ) return false;
                *similarity = m_main_core->CalculateSimilarity( features1, features2 );
                return true;
            }

            int64_t Insert( const std::shared_ptr<float> &features ) const {
                unique_write_lock<rwmutex> _locker( m_db_mutex );
                auto new_index = m_max_index++;
                m_db.insert( std::make_pair( new_index, features ) );
                return new_index;
            }

            int Delete( int64_t index ) {
                unique_write_lock<rwmutex> _locker( m_db_mutex );
                return int( m_db.erase( index ) );
            }

            size_t Count() const {
                unique_read_lock<rwmutex> _locker( m_db_mutex );
                return m_db.size();
            }

            void Clear() {
                unique_write_lock<rwmutex> _locker( m_db_mutex );
                m_db.clear();
                m_max_index = 0;
            }

            size_t QueryTop( const float *features, size_t N, int64_t *index, float *similarity ) const {
                unique_read_lock<rwmutex> _read_locker( m_db_mutex );

                std::vector<std::pair<int64_t, float>> result( m_db.size() );
                {
                    std::unique_lock<std::mutex> _locker( m_comparation_mutex );
                    size_t i = 0;
                    for( auto &line : m_db ) {
                        result[i].first = line.first;
                        Compare( features, line.second.get(), &result[i].second );
                        i++;
                    }
                }

                std::partial_sort( result.begin(), result.begin() + N, result.end(), [](
                                       const std::pair<int64_t, float> &a, const std::pair<int64_t, float> &b ) -> bool
                {
                    return a.second > b.second;
                } );
                const size_t top_n = std::min( N, result.size() );
                for( size_t i = 0; i < top_n; ++i ) {
                    index[i] = result[i].first;
                    similarity[i] = result[i].second;
                }
                return top_n;
            }

            class IndexWithSimilarity {
            public:
                IndexWithSimilarity() = default;
                IndexWithSimilarity( int64_t index, float similarity )
                    : index( index ), similarity( similarity ) {}

                int64_t index = -1;
                float similarity = 0;
            };

            static size_t SortAbove( IndexWithSimilarity *data, size_t N, float threshold ) {
                if( N == 0 ) return 0;
                std::stack<std::pair<int64_t, int64_t>> sort_stack;
                sort_stack.push( std::make_pair( 0, N - 1 ) );
                int64_t left_bound = -1;
                int64_t right_bound = int64_t( N );
                while( !sort_stack.empty() ) {
                    const auto const_left = sort_stack.top().first;
                    const auto const_right = sort_stack.top().second;
                    sort_stack.pop();
                    // end case
                    if( const_right < const_left ) {
                        continue;
                    }
                    else
                        if( const_right == const_left ) {
                            const auto bound = const_left;
                            if( data[bound].similarity >= threshold ) {
                                left_bound = bound;
                            }
                            else {
                                right_bound = bound;
                            }
                            continue;
                        }
                    // sort part
                    auto left = const_left;
                    auto right = const_right;
                    const auto flag = data[left];
                    while( left < right ) {
                        while( left < right && data[right].similarity <= flag.similarity ) --right;
                        data[left] = data[right];
                        while( left < right && data[left].similarity >= flag.similarity ) ++left;
                        data[right] = data[left];
                    }
                    const auto bound = left;
                    data[bound] = flag;
                    if( flag.similarity >= threshold ) {
                        left_bound = bound;
                        sort_stack.push( std::make_pair( const_left, bound ) );
                        sort_stack.push( std::make_pair( bound + 1, const_right ) );
                    }
                    else {
                        right_bound = bound;
                        sort_stack.push( std::make_pair( const_left, bound ) );
                    }
                }

                int64_t sorted_size = left_bound + 1;
                for( ; sorted_size < right_bound; ++sorted_size ) {
                    if( data[sorted_size].similarity < threshold ) break;
                }

                return size_t( sorted_size );
            }

            size_t QueryAbove( const float *features, float threshold, size_t N, int64_t *index, float *similarity ) const {
                unique_read_lock<rwmutex> _read_locker( m_db_mutex );

                std::vector<IndexWithSimilarity> result( m_db.size() );
                {
                    std::unique_lock<std::mutex> _locker( m_comparation_mutex );
                    size_t i = 0;
                    for( auto &line : m_db ) {
                        result[i].index = line.first;
                        Compare( features, line.second.get(), &result[i].similarity );
                        i++;
                    }
                }
                // sort all above threshold
                size_t sorted = SortAbove( result.data(), m_db.size(), threshold );
                const size_t top_n = std::min( N, sorted );
                for( size_t i = 0; i < top_n; ++i ) {
                    index[i] = result[i].index;
                    similarity[i] = result[i].similarity;
                }
                return top_n;
            }

            template <typename T>
            static size_t Write( StreamWriter &writer, const T &value ) {
                return writer.write( reinterpret_cast<const char *>( &value ), sizeof( T ) );
            }

            template <typename T>
            static size_t Read( StreamReader &reader, T &value ) {
                return reader.read( reinterpret_cast<char *>( &value ), sizeof( T ) );
            }

            template <typename T>
            static size_t Write( StreamWriter &writer, const T *arr, size_t size ) {
                return writer.write( reinterpret_cast<const char *>( arr ), sizeof( T ) * size );
            }

            template <typename T>
            static size_t Read( StreamReader &reader, T *arr, size_t size ) {
                return reader.read( reinterpret_cast<char *>( arr ), sizeof( T ) * size );
            }

#define MAGIC_SERIAL 0x7726

            bool Save( StreamWriter &writer ) const {
                unique_read_lock<rwmutex> _locker( m_db_mutex );
                const int flag = MAGIC_SERIAL;
                Write( writer, flag );

                const uint64_t num = m_db.size();
                const uint64_t dim = m_main_core->GetExtractFeatureSize();

                Write( writer, num );
                Write( writer, dim );

                for( auto &line : m_db ) {
                    auto &index = line.first;
                    auto &features = line.second;
                    // do save
                    Write( writer, index );
                    Write( writer, features.get(), size_t( dim ) );
                }

                std::cout << "FaceDatabase Loaded " << num << " faces" << std::endl;
                return true;
            }

            bool Load( StreamReader &reader ) {
                unique_write_lock<rwmutex> _locker( m_db_mutex );

                int flag;
                Read( reader, flag );
                if( flag != MAGIC_SERIAL ) {
                    std::cout << "FaceDatabase Load terminated, unsupported file format" << std::endl;
                    return false;
                }

                uint64_t num;
                uint64_t dim;
                Read( reader, num );
                Read( reader, dim );

                if( m_main_core != nullptr ) {
                    if( dim != uint64_t( m_main_core->GetExtractFeatureSize() ) ) {
                        std::cout << "FaceDatabase Load terminated, mismatch feature size" << std::endl;
                        return false;
                    }
                }

                m_db.clear();
                m_max_index = -1;

                for( size_t i = 0; i < num; ++i ) {
                    int64_t index;
                    std::shared_ptr<float> features( new float[size_t( dim )], std::default_delete<float[]>() );

                    Read( reader, index );
                    Read( reader, features.get(), size_t( dim ) );

                    m_db.insert( std::make_pair( index, features ) );
                    m_max_index = std::max( m_max_index, index );
                }
                m_max_index++;

                std::cout << "FaceDatabase Loaded " << num << " faces" << std::endl;

                return true;
            }

            FaceRecognizer *ExtractionCore( int id = 0 ) {
                if( id < 0 || size_t( id ) >= m_cores.size() ) {
                    return nullptr;
                }
                return m_cores[id].get();
            }

        private:
            std::shared_ptr<FaceRecognizer> m_main_core;
            std::vector<std::shared_ptr<FaceRecognizer>> m_cores;

            mutable std::map<int64_t, std::shared_ptr<float>> m_db; // saving face db
            mutable int64_t m_max_index = 0;    ///< next saving id
            mutable rwmutex m_db_mutex;
            mutable std::mutex m_comparation_mutex;
        };



        FaceDatabase::FaceDatabase( const SeetaModelSetting &setting )
            : m_impl( new Implement( setting ) )
        {
        }

        FaceDatabase::~FaceDatabase()
        {
            delete m_impl;
        }

        int FaceDatabase::GetCropFaceWidth()
        {
            return m_impl->core().GetCropFaceWidth();
        }

        int FaceDatabase::GetCropFaceHeight()
        {
            return m_impl->core().GetCropFaceHeight();
        }

        int FaceDatabase::GetCropFaceChannels()
        {
            return m_impl->core().GetCropFaceChannels();
        }

        bool FaceDatabase::CropFace( const SeetaImageData &image, const SeetaPointF *points, SeetaImageData &face )
        {

            float mean_shape[10] =
            {
                89.3095f, 72.9025f,
                169.3095f, 72.9025f,
                127.8949f, 127.0441f,
                96.8796f, 184.8907f,
                159.1065f, 184.7601f,
            };
            float local_points[10];
            for( int i = 0; i < 5; ++i )
            {
                local_points[2 * i] = float( points[i].x );
                local_points[2 * i + 1] = float( points[i].y );
            }

            face_crop_core( image.data, image.width, image.height, image.channels, face.data, GetCropFaceWidth(), GetCropFaceHeight(), local_points, 5, mean_shape, 256, 256 );

            return true;
        }


        float FaceDatabase::Compare( const SeetaImageData &image1, const SeetaPointF *points1,
                                     const SeetaImageData &image2, const SeetaPointF *points2 ) const
        {
            auto feature_size = m_impl->core().GetExtractFeatureSize();
            std::unique_ptr<float[]> features( new float[2 * feature_size] );

            m_impl->Extract( image1, points1, features.get() );
            m_impl->Extract( image2, points2, features.get() + feature_size );

            return m_impl->core().CalculateSimilarity( features.get(), features.get() + feature_size );
        }

        float FaceDatabase::CompareByCroppedFace( const SeetaImageData &cropped_face_image1,
                const SeetaImageData &cropped_face_image2 ) const
        {
            auto feature_size = m_impl->core().GetExtractFeatureSize();
            std::unique_ptr<float[]> features( new float[2 * feature_size] );

            m_impl->ExtractCroppedFace( cropped_face_image1, features.get() );
            m_impl->ExtractCroppedFace( cropped_face_image2, features.get() + feature_size );

            return m_impl->core().CalculateSimilarity( features.get(), features.get() + feature_size );
        }

        int64_t FaceDatabase::Register( const SeetaImageData &image, const SeetaPointF *points )
        {
            auto feature_size = m_impl->core().GetExtractFeatureSize();
            std::shared_ptr<float> features( new float[feature_size], std::default_delete<float[]>() );

            m_impl->Extract( image, points, features.get() );
            int64_t index = m_impl->Insert( features );
            return index;
        }

        int64_t FaceDatabase::RegisterByCroppedFace( const SeetaImageData &cropped_face_image )
        {
            auto feature_size = m_impl->core().GetExtractFeatureSize();
            std::shared_ptr<float> features( new float[feature_size], std::default_delete<float[]>() );

            m_impl->ExtractCroppedFace( cropped_face_image, features.get() );
            int64_t index = m_impl->Insert( features );
            return index;
        }

        int FaceDatabase::Delete( int64_t index )
        {
            return m_impl->Delete( index );
        }

        void FaceDatabase::Clear()
        {
            m_impl->Clear();
        }

        size_t FaceDatabase::Count() const
        {
            return m_impl->Count();
        }

        int64_t FaceDatabase::Query( const SeetaImageData &image, const SeetaPointF *points, float *similarity ) const
        {
            int64_t index = -1;
            float local_similarity = 0;
            size_t top_n = QueryTop( image, points, 1, &index, &local_similarity );
            if( top_n < 1 ) return index;
            if( similarity != nullptr ) *similarity = local_similarity;
            return index;
        }

        int64_t FaceDatabase::QueryByCroppedFace( const SeetaImageData &cropped_face_image, float *similarity ) const
        {
            int64_t index = -1;
            float local_similarity = 0;
            size_t top_n = QueryTopByCroppedFace( cropped_face_image, 1, &index, &local_similarity );
            if( top_n < 1 ) return index;
            if( similarity != nullptr ) *similarity = local_similarity;
            return index;
        }

        size_t FaceDatabase::QueryTop( const SeetaImageData &image, const SeetaPointF *points, size_t N, int64_t *index,
                                       float *similarity ) const
        {
            if( !index || !similarity ) return 0;
            const auto count = this->Count();
            if( count == 0 ) return 0;
            const auto feature_size = m_impl->core().GetExtractFeatureSize();
            std::unique_ptr<float[]> features( new float[feature_size] );

            if( m_impl->Extract( image, points, features.get() ) )
            {
                return m_impl->QueryTop( features.get(), N, index, similarity );
            }
            return 0;
        }

        size_t FaceDatabase::QueryTopByCroppedFace( const SeetaImageData &cropped_face_image, size_t N, int64_t *index,
                float *similarity ) const
        {
            if( !index || !similarity ) return 0;
            const auto count = this->Count();
            if( count == 0 ) return 0;
            const auto feature_size = m_impl->core().GetExtractFeatureSize();
            std::unique_ptr<float[]> features( new float[feature_size] );

            if( m_impl->ExtractCroppedFace( cropped_face_image, features.get() ) )
            {
                return m_impl->QueryTop( features.get(), N, index, similarity );
            }
            return 0;
        }

        size_t FaceDatabase::QueryAbove( const SeetaImageData &image, const SeetaPointF *points, float threshold, size_t N,
                                         int64_t *index, float *similarity ) const
        {
            if( !index || !similarity ) return 0;
            const auto count = this->Count();
            if( count == 0 ) return 0;
            const auto feature_size = m_impl->core().GetExtractFeatureSize();
            std::unique_ptr<float[]> features( new float[feature_size] );

            if( m_impl->Extract( image, points, features.get() ) )
            {
                return m_impl->QueryAbove( features.get(), threshold, N, index, similarity );
            }
            return 0;
        }

        size_t FaceDatabase::QueryAboveByCroppedFace( const SeetaImageData &cropped_face_image, float threshold, size_t N,
                int64_t *index, float *similarity ) const
        {
            if( !index || !similarity ) return 0;
            const auto count = this->Count();
            if( count == 0 ) return 0;
            const auto feature_size = m_impl->core().GetExtractFeatureSize();
            std::unique_ptr<float[]> features( new float[feature_size] );

            if( m_impl->ExtractCroppedFace( cropped_face_image, features.get() ) )
            {
                return m_impl->QueryAbove( features.get(), threshold, N, index, similarity );
            }
            return 0;
        }


        bool FaceDatabase::Save( const char *path ) const
        {
            FileWriter ofile( path, FileWriter::Binary );
            if( !ofile.is_opened() ) return false;
            return Save( ofile );
        }

        bool FaceDatabase::Load( const char *path )
        {
            FileReader ifile( path, FileWriter::Binary );
            if( !ifile.is_opened() ) return false;
            return Load( ifile );
        }

        bool FaceDatabase::Save( StreamWriter &writer ) const
        {
            return m_impl->Save( writer );
        }

        bool FaceDatabase::Load( StreamReader &reader )
        {
            return m_impl->Load( reader );
        }


        FaceRecognizer *FaceDatabase::ExtractionCore( int i )
        {
            return m_impl->ExtractionCore( i );
        }


    }
}
