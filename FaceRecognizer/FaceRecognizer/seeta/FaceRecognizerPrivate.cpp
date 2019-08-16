#include "FaceRecognizerPrivate.h"
#include <cmath>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>
#include <iostream>

#include <SeetaNetForward.h>
#include "SeetaModelHeader.h"
#include "seeta/common_alignment.h"
#include "seeta/ImageProcess.h"




class FaceRecognizerPrivate::Recognizer {
public:
    SeetaNet_Model *model = nullptr;
    SeetaNet_Net *net = nullptr;
    seeta::FRModelHeader header;

    SeetaDevice device = SEETA_DEVICE_AUTO;

    // for memory share
    SeetaNet_SharedParam *param = nullptr;

    std::string version;
    std::string date;
    std::string name;
    std::function<float( float )> trans_func;

    static int max_batch_global;
    int max_batch_local;

    // for YuLe setting
    int sqrt_times = -1;
    std::string default_method = "crop";
    std::string method = "";

    static int core_number_global;
    int recognizer_number_threads = 2;
    std::vector<SeetaNet_Net *> cores;

    Recognizer() {
        header.width = 256;
        header.height = 256;
        header.channels = 3;
        max_batch_local = max_batch_global;
        recognizer_number_threads = core_number_global;
    }

    void free() {
        if( model ) SeetaReleaseModel( model );
        model = nullptr;
        if( net ) SeetaReleaseNet( net );
        net = nullptr;
        for( size_t i = 1; i < cores.size(); ++i ) {
            SeetaReleaseNet( cores[i] );
        }
        cores.clear();
    }

    float trans( float similar ) const {
        if( trans_func ) {
            return trans_func( similar );
        }
        return similar;
    }

    int GetMaxBatch() {
        return max_batch_local;
    }

    int GetCoreNumber() {
        return 1;
    }

    ~Recognizer() {
        Recognizer::free();
    }

    void fix() {
        if( this->sqrt_times < 0 ) {
            this->sqrt_times = this->header.feature_size >= 1024 ? 1 : 0;
        }

        if( this->method.empty() ) {
            this->method = this->header.feature_size >= 1024 ? this->default_method : "resize";
        }
    }
};

int FaceRecognizerPrivate::Recognizer::max_batch_global = 1;
int FaceRecognizerPrivate::Recognizer::core_number_global = 1;


float sigmoid( float x, float a = 0, float b = 1 )
{
    return 1 / ( 1 + exp( a - b * x ) );
}

float poly( float x, const std::vector<float> &params )
{
    if( params.empty() ) return x;
    float y = 0;

    for( size_t i = 0; i < params.size(); ++i )
    {
        int p = static_cast<int>( params.size() - 1 - i );
        y += params[i] * std::pow( x, p );
    }
    return std::max<float>( 0, std::min<float>( 1, y ) );
}


FaceRecognizerModel::FaceRecognizerModel( const char *model_path, int device )
    : m_impl( new FaceRecognizerPrivate::Recognizer )
{

    auto recognizer = reinterpret_cast<FaceRecognizerPrivate::Recognizer *>( m_impl );

    if( !model_path )
    {
        std::cout << "Can not load empty model" << std::endl;
        exit( -1 );
    }

    int gpu_id = 0;
    SeetaNet_DEVICE_TYPE type = SEETANET_CPU_DEVICE;
    recognizer->device = SeetaDevice( device );

    std::shared_ptr<char> sta_buffer;
    char *buffer = nullptr;
    int64_t buffer_len = 0;

    std::ifstream inf( model_path, std::ios::binary );

    if( !inf.is_open() )
    {
        std::cout << "Can not access \"" << model_path << "\"" << std::endl;
        exit( -1 );
    }

    inf.seekg( 0, std::ios::end );
    auto sta_length = inf.tellg();
    sta_buffer.reset( new char[size_t( sta_length )], std::default_delete<char[]>() );
    inf.seekg( 0, std::ios::beg );
    inf.read( sta_buffer.get(), sta_length );

    buffer = sta_buffer.get();
    buffer_len = sta_length;

    inf.close();
    // read header
    size_t header_size = recognizer->header.read_ex( buffer, size_t( buffer_len ) );

    // convert the model
    if( SeetaReadModelFromBuffer( buffer + header_size, size_t( buffer_len - header_size ), &recognizer->model ) )
    {
        std::cout << "Got an broken model file" << std::endl;
        exit( -1 );
    }
    // create the net
    int err_code;
    err_code = SeetaCreateNetSharedParam( recognizer->model, 1, type, &recognizer->net, &recognizer->param );

    if( err_code )
    {
        SeetaReleaseModel( recognizer->model );
        recognizer->model = nullptr;
        std::cout << "Can not init net from broken model" << std::endl;
        exit( -1 );
    }

    recognizer->fix();

    // here, we got model, net, and param
}

FaceRecognizerModel::~FaceRecognizerModel()
{
    auto recognizer = reinterpret_cast<FaceRecognizerPrivate::Recognizer *>( m_impl );
    delete recognizer;
}

const FaceRecognizerPrivate::Param *FaceRecognizerPrivate::GetParam() const
{
    return reinterpret_cast<const Param *>( SeetaGetSharedParam( recognizer->net ) );
}

FaceRecognizerPrivate::FaceRecognizerPrivate( const Param *param )
    : recognizer( new Recognizer )
{
    #ifdef SEETA_CHECK_INIT
    SEETA_CHECK_INIT;
    #endif
    recognizer->param =
        const_cast<SeetaNet_SharedParam *>(
            reinterpret_cast<const SeetaNet_SharedParam *>(
                param ) );
}


FaceRecognizerPrivate::FaceRecognizerPrivate( const FaceRecognizerModel &model )
    : recognizer( new Recognizer )
{
    #ifdef SEETA_CHECK_INIT
    SEETA_CHECK_INIT;
    #endif
    auto other = reinterpret_cast<FaceRecognizerPrivate::Recognizer *>( model.m_impl );

    auto device = other->device;

    using self = FaceRecognizerPrivate;
    SeetaNet_DEVICE_TYPE type = SEETANET_CPU_DEVICE;

    *recognizer = *other;
    recognizer->model = nullptr;
    recognizer->net = nullptr;

    int err_code;
    err_code = SeetaCreateNetSharedParam( other->model, GetMaxBatch(), type, &recognizer->net, &other->param );

    if( err_code )
    {
        std::cout << "Can not init net from unload model" << std::endl;
        exit( -1 );
    }

    SeetaKeepBlob( recognizer->net, recognizer->header.blob_name.c_str() );

}


FaceRecognizerPrivate::FaceRecognizerPrivate( const char *modelPath )
    : FaceRecognizerPrivate( modelPath, SEETA_DEVICE_AUTO, 0 )
{
}

FaceRecognizerPrivate::FaceRecognizerPrivate( const char *modelPath, SeetaDevice device, int gpuid )
    : recognizer( new Recognizer )
{
    #ifdef SEETA_CHECK_INIT
    SEETA_CHECK_INIT;
    #endif
    if( modelPath && !LoadModel( modelPath, device, gpuid ) )
    {
        std::cerr << "Error: Can not access \"" << modelPath << "\"!" << std::endl;
        throw std::logic_error( "Missing model" );
    }
}

FaceRecognizerPrivate::FaceRecognizerPrivate( const char *modelBuffer, size_t bufferLength, SeetaDevice device, int gpuid )
    : recognizer( new Recognizer )
{
    #ifdef SEETA_CHECK_INIT
    SEETA_CHECK_INIT;
    #endif
    if( modelBuffer && !LoadModel( modelBuffer, bufferLength, device, gpuid ) )
    {
        std::cerr << "Error: Can not initialize from memory!" << std::endl;
        throw std::logic_error( "Missing model" );
    }
}

FaceRecognizerPrivate::~FaceRecognizerPrivate()
{
    delete recognizer;
}


bool FaceRecognizerPrivate::LoadModel( const char *modelPath )
{
    return LoadModel( modelPath, SEETA_DEVICE_AUTO, 0 );
}


bool FaceRecognizerPrivate::LoadModel( const char *modelPath, SeetaDevice device, int gpuid )
{
    if( modelPath == NULL ) return false;

    recognizer->trans_func = nullptr;

    char *buffer = nullptr;
    int64_t buffer_len = 0;
    if( SeetaReadAllContentFromFile( modelPath, &buffer, &buffer_len ) )
    {
        return false;
    }

    bool loaded = LoadModel( buffer, size_t( buffer_len ), device, gpuid );
    SeetaFreeBuffer( buffer );

    recognizer->fix();

    return loaded;

}

bool FaceRecognizerPrivate::LoadModel( const char *modelBuffer, size_t bufferLength, SeetaDevice device, int gpuid )
{
    // Code
    #ifdef NEED_CHECK
    checkit();
    #endif

    if( modelBuffer == NULL )
    {
        return false;
    }

    recognizer->free();

    using self = FaceRecognizerPrivate;
    SeetaNet_DEVICE_TYPE type = SEETANET_CPU_DEVICE;

    recognizer->device = device;

    // read header
    size_t header_size = recognizer->header.read_ex( modelBuffer, bufferLength );

    // convert the model
    if( SeetaReadModelFromBuffer( modelBuffer + header_size, bufferLength - header_size, &recognizer->model ) )
    {
        return false;
    }
    // create the net
    int err_code;
    err_code = SeetaCreateNetSharedParam( recognizer->model, GetMaxBatch(), type, &recognizer->net, &recognizer->param );

    if( err_code )
    {
        SeetaReleaseModel( recognizer->model );
        recognizer->model = nullptr;
        return false;
    }

    SeetaKeepBlob( recognizer->net, recognizer->header.blob_name.c_str() );

    SeetaReleaseModel( recognizer->model );
    recognizer->model = nullptr;

    return true;
}


uint32_t FaceRecognizerPrivate::GetFeatureSize()
{
    return recognizer->header.feature_size;
}

uint32_t FaceRecognizerPrivate::GetCropWidth()
{
    return recognizer->header.width;
}

uint32_t FaceRecognizerPrivate::GetCropHeight()
{
    return recognizer->header.height;
}

uint32_t FaceRecognizerPrivate::GetCropChannels()
{
    return recognizer->header.channels;
}

bool FaceRecognizerPrivate::CropFace( const SeetaImageData &srcImg, const SeetaPointF *llpoint, SeetaImageData &dstImg, uint8_t posNum )
{
    float mean_shape[10] =
    {
        89.3095f, 72.9025f,
        169.3095f, 72.9025f,
        127.8949f, 127.0441f,
        96.8796f, 184.8907f,
        159.1065f, 184.7601f,
    };
    float points[10];
    for( int i = 0; i < 5; ++i )
    {
        points[2 * i] = float( llpoint[i].x );
        points[2 * i + 1] = float( llpoint[i].y );
    }

    if( GetCropHeight() == 256 && GetCropWidth() == 256 )
    {
        face_crop_core( srcImg.data, srcImg.width, srcImg.height, srcImg.channels, dstImg.data, GetCropWidth(), GetCropHeight(), points, 5, mean_shape, 256, 256 );
    }
    else
    {
        if( recognizer->method == "resize" )
        {
            seeta::Image face256x256( 256, 256, 3 );
            face_crop_core( srcImg.data, srcImg.width, srcImg.height, srcImg.channels, face256x256.data(), 256, 256, points, 5, mean_shape, 256, 256 );
            seeta::Image fixed = seeta::resize( face256x256, seeta::Size( GetCropWidth(), GetCropHeight() ) );
            fixed.copy_to( dstImg.data );
        }
        else
        {
            face_crop_core( srcImg.data, srcImg.width, srcImg.height, srcImg.channels, dstImg.data, GetCropWidth(), GetCropHeight(), points, 5, mean_shape, 256, 256 );
        }
    }

    return true;
}

bool FaceRecognizerPrivate::ExtractFeature( const SeetaImageData &cropImg, float *feats )
{
    std::vector<SeetaImageData> faces = { cropImg };
    return ExtractFeature( faces, feats, false );
}

static void normalize( float *features, int num )
{
    double norm = 0;
    float *dim = features;
    for( int i = 0; i < num; ++i )
    {
        norm += *dim * *dim;
        ++dim;
    }
    norm = std::sqrt( norm ) + 1e-5;
    dim = features;
    for( int i = 0; i < num; ++i )
    {
        *dim /= float( norm );
        ++dim;
    }
}

bool FaceRecognizerPrivate::ExtractFeatureNormalized( const SeetaImageData &cropImg, float *feats )
{
    std::vector<SeetaImageData> faces = { cropImg };
    return ExtractFeature( faces, feats, true );
}

bool FaceRecognizerPrivate::ExtractFeatureWithCrop( const SeetaImageData &srcImg, const SeetaPointF *llpoint, float *feats, uint8_t posNum )
{
    SeetaImageData dstImg;
    dstImg.width = GetCropWidth();
    dstImg.height = GetCropHeight();
    dstImg.channels = srcImg.channels;
    std::unique_ptr<uint8_t[]> dstImgData( new uint8_t[dstImg.width * dstImg.height * dstImg.channels] );
    dstImg.data = dstImgData.get();
    CropFace( srcImg, llpoint, dstImg, posNum );
    ExtractFeature( dstImg, feats );
    return true;
}

bool FaceRecognizerPrivate::ExtractFeatureWithCropNormalized( const SeetaImageData &srcImg, const SeetaPointF *llpoint, float *feats, uint8_t posNum )
{
    if( ExtractFeatureWithCrop( srcImg, llpoint, feats, posNum ) )
    {
        normalize( feats, GetFeatureSize() );
        return true;
    }
    return false;
}

float FaceRecognizerPrivate::CalcSimilarity( const float *fc1, const float *fc2, long dim )
{
    if( dim <= 0 ) dim = GetFeatureSize();
    double dot = 0;
    double norm1 = 0;
    double norm2 = 0;
    for( size_t i = 0; i < dim; ++i )
    {
        dot += fc1[i] * fc2[i];
        norm1 += fc1[i] * fc1[i];
        norm2 += fc2[i] * fc2[i];
    }
    double similar = dot / ( sqrt( norm1 * norm2 ) + 1e-5 );

    return recognizer->trans( float( similar ) );
}

float FaceRecognizerPrivate::CalcSimilarityNormalized( const float *fc1,  const float *fc2, long dim )
{
    if( dim <= 0 ) dim = GetFeatureSize();
    double dot = 0;

    const float *fc1_dim = fc1;
    const float *fc2_dim = fc2;
    for( int i = 0; i < dim; ++i )
    {
        dot += *fc1_dim * *fc2_dim;
        ++fc1_dim;
        ++fc2_dim;
    }

    double similar = dot;
    return recognizer->trans( float( similar ) );
}

int FaceRecognizerPrivate::SetMaxBatchGlobal( int max_batch )
{
    std::swap( max_batch, Recognizer::max_batch_global );
    return max_batch;
}

int FaceRecognizerPrivate::GetMaxBatch()
{
    return recognizer->GetMaxBatch();
}

int FaceRecognizerPrivate::SetCoreNumberGlobal( int core_number )
{
    std::swap( core_number, Recognizer::core_number_global );
    return core_number;

}

int FaceRecognizerPrivate::GetCoreNumber()
{
    return recognizer->GetCoreNumber();
}

template <typename T>
static void CopyData( T *dst, const T *src, size_t count )
{
    #if _MSC_VER >= 1600
    memcpy_s( dst, count * sizeof( T ), src, count * sizeof( T ) );
    #else
    memcpy( dst, src, count * sizeof( T ) );
    #endif
}

static bool LocalExtractFeature(
    int number, int width, int height, int channels, unsigned char *data,
    SeetaNet_Net *net, int max_batch, const char *blob_name, int feature_size,
    float *feats,
    bool normalization, int sqrt_times = 0 )
{
    if( !net ) return false;
    if( data == nullptr || number <= 0 ) return true;

    auto single_image_size = channels * height * width;

    if( number > max_batch )
    {
        // Divide and Conquer
        int end = number;
        int step = max_batch;
        int left = 0;
        while( left < end )
        {
            int right = std::min( left + step, end );

            int local_number = right - left;
            unsigned char *local_data = data + left * single_image_size;
            float *local_feats = feats + left * feature_size;

            if( !LocalExtractFeature(
                        local_number, width, height, channels, local_data,
                        net, max_batch, blob_name, feature_size,
                        local_feats,
                        normalization,
                        sqrt_times
                    ) ) return false;
            left = right;
        }
        return true;
    }

    SeetaNet_InputOutputData himg;
    himg.number = number;
    himg.channel = channels;
    himg.height = height;
    himg.width = width;
    himg.buffer_type = SEETANET_BGR_IMGE_CHAR;
    himg.data_point_char = data;

    // do forward
    if( SeetaRunNetChar( net, 1, &himg ) )
    {
        std::cout << "SeetaRunNetChar failed." << std::endl;
        return false;
    }

    // get the output
    SeetaNet_InputOutputData output;
    if( SeetaGetFeatureMap( net, blob_name, &output ) )
    {

        std::cout << "SeetaGetFeatureMap failed." << std::endl;
        return false;
    }

    // check the output size
    if( output.channel * output.height * output.width != feature_size || output.number != himg.number )
    {

        std::cout << "output shape missmatch. " << feature_size << " expected. but " << output.channel *output.height *output.width << " given" << std::endl;
        return false;
    }

    // copy data for output
    CopyData( feats, output.data_point_float, output.number * feature_size );

    int32_t all_feats_size = output.number * feature_size;
    float *all_feats = feats;

    #if defined(DOUBLE_SQRT) || defined(SINGLE_SQRT)
    for( int i = 0; i != all_feats_size; i++ )
    {
        #if defined(DOUBLE_SQRT)
        feat[i] = sqrt( sqrt( feat[i] ) );
        #elif defined(SINGLE_SQRT)
        all_feats[i] = sqrt( all_feats[i] );
        #endif // DOUBLE_SQRT
    }
    #endif // DOUBLE_SQRT || SINGLE_SQRT

    if( sqrt_times > 0 )
    {
        while( sqrt_times-- )
        {
            for( int i = 0; i != all_feats_size; i++ ) all_feats[i] = std::sqrt( all_feats[i] );
        }
    }

    if( normalization )
    {
        for( int i = 0; i < number; ++i )
        {
            float *local_feats = feats + i * feature_size;
            normalize( local_feats, feature_size );
        }
    }

    return true;
}


bool FaceRecognizerPrivate::ExtractFeature( const std::vector<SeetaImageData> &faces, float *feats, bool normalization )
{

    if( !recognizer->net ) return false;
    if( faces.empty() ) return true;

    int number = int( faces.size() );
    int channels = GetCropChannels();
    int height = GetCropHeight();
    int width = GetCropWidth();


    auto single_image_size = channels * height * width;
    std::unique_ptr<unsigned char[]> data_point_char( new unsigned char[number * single_image_size] );
    for( int i = 0; i < number; ++i )
    {

        if( faces[i].channels == channels &&
                faces[i].height == height &&
                faces[i].width == width )
        {
            CopyData( &data_point_char[i * single_image_size], faces[i].data, single_image_size );
            continue;
        }

        if( recognizer->method == "resize" )
        {
            seeta::Image face( faces[i].data, faces[i].width, faces[i].height, faces[i].channels );
            seeta::Image fixed = seeta::resize( face, seeta::Size( GetCropWidth(), GetCropHeight() ) );
            CopyData( &data_point_char[i * single_image_size], fixed.data(), single_image_size );
        }
        else
        {
            seeta::Image face( faces[i].data, faces[i].width, faces[i].height, faces[i].channels );
            seeta::Rect rect( ( GetCropWidth() - faces[i].width ) / 2, ( GetCropHeight() - faces[i].height ) / 2, GetCropWidth(), GetCropHeight() );
            seeta::Image fixed = seeta::crop_resize( face, rect, seeta::Size( GetCropWidth(), GetCropHeight() ) );
            CopyData( &data_point_char[i * single_image_size], fixed.data(), single_image_size );
        }

    }
    return LocalExtractFeature(
               number, width, height, channels, data_point_char.get(),
               recognizer->net, GetMaxBatch(), recognizer->header.blob_name.c_str(), GetFeatureSize(),
               feats,
               normalization,
               recognizer->sqrt_times );
}

bool FaceRecognizerPrivate::ExtractFeatureNormalized( const std::vector<SeetaImageData> &faces, float *feats )
{
    return ExtractFeature( faces, feats, true );
}


// on checking param, sure right
static bool CropFaceBatch( FaceRecognizerPrivate &FR, const std::vector<SeetaImageData> &images,
                           const std::vector<SeetaPointF> &points, unsigned char *faces_data )
{
    const int PN = 5;
    const auto single_image_size = FR.GetCropChannels() * FR.GetCropHeight() * FR.GetCropWidth();
    unsigned char *single_face_data = faces_data;
    const SeetaPointF *single_points = points.data();
    for( size_t i = 0; i < images.size(); ++i )
    {
        SeetaImageData face;
        face.width = FR.GetCropWidth();
        face.height = FR.GetCropHeight();
        face.channels = FR.GetCropChannels();
        face.data = single_face_data;

        if( !FR.CropFace( images[i], single_points, face ) ) return false;

        single_points += PN;
        single_face_data += single_image_size;
    }
    return true;
}

bool FaceRecognizerPrivate::ExtractFeatureWithCrop( const std::vector<SeetaImageData> &images,
        const std::vector<SeetaPointF> &points, float *feats, bool normalization )
{
    if( !recognizer->net ) return false;
    if( images.empty() ) return true;

    const int PN = 5;

    if( images.size() * PN != points.size() )
    {
        return false;
    }

    // crop face
    std::unique_ptr<unsigned char[]> faces_data( new unsigned char[images.size() * GetCropChannels() * GetCropHeight() * GetCropWidth()] );
    ::CropFaceBatch( *this, images, points, faces_data.get() );

    int number = int( images.size() );
    int channels = GetCropChannels();
    int height = GetCropHeight();
    int width = GetCropWidth();
    return LocalExtractFeature(
               number, width, height, channels, faces_data.get(),
               recognizer->net, GetMaxBatch(), recognizer->header.blob_name.c_str(), GetFeatureSize(),
               feats,
               normalization,
               recognizer->sqrt_times
           );
}

bool FaceRecognizerPrivate::ExtractFeatureWithCropNormalized( const std::vector<SeetaImageData> &images,
        const std::vector<SeetaPointF> &points, float *feats )
{
    return ExtractFeatureWithCrop( images, points, feats, true );
}
