#include "FaceDetectorPrivate.h"
#include "SeetaNetForward.h"

#include <ctime>
#include <algorithm>
#include <memory>
#include <fstream>
#include <iostream>
#include <cstring>

#include <thread>
#include <sstream>

#include "CFaceInfo.h"

#include "seeta/ImageProcess.h"

#define  CLAMP(x, l, u)   ((x) < (l) ? (l) : ((x) > (u) ? (u) : (x)))




static std::vector<std::string> Split( const std::string &str, char ch )
{
    std::vector<std::string> result;
    std::string::size_type left = 0, right;

    while( true )
    {
        right = str.find( ch, left );
        result.push_back( str.substr( left, right == std::string::npos ? std::string::npos : right - left ) );
        if( right == std::string::npos ) break;
        left = right + 1;
    }
    return std::move( result );
}


typedef struct Rect
{
    int32_t x;
    int32_t y;
    int32_t width;
    int32_t height;
    int32_t scale;
    double conf;
    Rect( int32_t x_ = 0, int32_t y_ = 0, int32_t w_ = 0, int32_t h_ = 0, int32_t s_ = 0, double c_ = 0 )
        : x( x_ ), y( y_ ), width( w_ ), height( h_ ), scale( s_ ), conf( c_ ) {
    }
} Rect;

class Impl {
public:
    Impl() {

    }
    // removed, not used
    // void LoadModel(const char* model_path12, const char* model_path24, const char* model_path48, FaceDetectorPrivate::Device device);
    void LoadModelBuffer( const char *model_buffer,
                          int64_t buffer_lenght12,
                          int64_t buffer_lenght24,
                          int64_t buffer_lenght48, SeetaDevice device, int gpuid );

    inline bool IsLegalImage( const SeetaImageData &image ) {
        return ( ( image.channels == 3 || image.channels == 1 )
                 && image.width > 0 && image.height > 0 &&
                 image.data != nullptr );
    }

    bool ResizeImage( const unsigned char *src_im, int src_width, int src_height, int src_channels,
                      unsigned char *dst_im, int dst_width, int dst_height, int dst_channels,
                      int crop_x = -1, int crop_y = -1, int crop_w = -1, int crop_h = -1 );

    bool PadImage( const unsigned char *src_im, int src_width, int src_height, int src_channels,
                   unsigned char *dst_im, int pad_w, int pad_h );

    bool Legal( int x, int y, const SeetaImageData &img ) {
        if( x >= 0 && x < img.width && y >= 0 && y < img.height )
            return true;
        else
            return false;
    }

    float IoU( const Rect &w1, const Rect &w2 ) {
        int xOverlap = std::max( 0, std::min( w1.x + w1.width - 1, w2.x + w2.width - 1 ) - std::max( w1.x, w2.x ) + 1 );
        int yOverlap = std::max( 0, std::min( w1.y + w1.height - 1, w2.y + w2.height - 1 ) - std::max( w1.y, w2.y ) + 1 );
        int intersection = xOverlap * yOverlap;
        int unio = w1.width * w1.height + w2.width * w2.height - intersection;
        return float( intersection ) / unio;
    }

    static bool CompareWin( const Rect &w1, const Rect &w2 ) {
        return w1.conf > w2.conf;
    }

    std::vector<Rect> NMS( std::vector<Rect> &winList, bool local, float threshold );

    std::vector<SeetaFaceInfo> TransWindow( const SeetaImageData &img, const SeetaImageData &img_pad, std::vector<Rect> &winList );

    std::vector<Rect> SlidingWindow( const SeetaImageData &img, const SeetaImageData &img_pad, SeetaNet_Net *&net, float thres, int local_min_face = -1, int local_max_face = -1 );

    std::vector<Rect> RunNet( const SeetaImageData &img, SeetaNet_Net *&net, float thres, int dim, std::vector<Rect> &winList );

    void SetInput( const SeetaImageData &img, int dim, std::vector<Rect> &winList, uint8_t *tmp_data );

public:
    int min_face_;
    float scale_;
    float class_threshold_[3];
    SeetaNet_Model *model_[3];
    SeetaNet_Net *net_[3];
    int stride_ = 4;
    float nms_threshold_[3];
    int max_pad_w;
    int max_pad_h;
    float max_pad_ratio;
    int max_batch_size[2];

    bool stable_ = false;
    std::vector<Rect> preList_;

    bool end2end_ = true;
    struct
    {
        std::string version;
        std::string date;
        std::string name;
    } sta;
    float thresh1_ = 0.7f;
    float thresh2_ = 0.7f;
    float thresh3_ = 0.85f;
    int width_limit_ = 0;
    int height_limit_ = 0;

    int max_face_ = -1;

};

std::vector<Rect> Impl::SlidingWindow( const SeetaImageData &img, const SeetaImageData &img_pad, SeetaNet_Net *&net, float thres, int local_min_face, int local_max_face )
{
    int height = ( img_pad.height - img.height ) / 2;
    int width = ( img_pad.width - img.width ) / 2;
    std::vector<Rect> winList;
    int net_size = 12;
    if( local_min_face <= 0 ) local_min_face = min_face_;
    float cur_scale = local_min_face / float( net_size );
    if( local_max_face <= 0 ) local_max_face = max_face_;
    float max_scale = local_max_face / float( net_size );
    SeetaImageData img_resized;
    img_resized.width = int( img.width / cur_scale );
    img_resized.height =  int( img.height / cur_scale );
    img_resized.channels =  img.channels;
    img_resized.data = new uint8_t[img_resized.height * img_resized.width * img_resized.channels]; // malloc image buffer
    int count_scale = 0;
    while( std::min( img_resized.width, img_resized.height ) >= net_size )
    {
        if( local_max_face > 0 )
        {
            if( cur_scale > max_scale ) break;
        }

        ResizeImage( img.data, img.width, img.height, img.channels,
                     img_resized.data, img_resized.width, img_resized.height, img_resized.channels );
        SeetaNet_InputOutputData tmp_input;
        tmp_input.number = 1;
        tmp_input.channel = img_resized.channels;
        tmp_input.height = img_resized.height;
        tmp_input.width = img_resized.width;
        tmp_input.buffer_type = SEETANET_BGR_IMGE_CHAR;
        tmp_input.data_point_char = img_resized.data;

        SeetaRunNetChar( net, 1, &tmp_input );

        SeetaNet_InputOutputData reg_res, cls_res;
        SeetaGetFeatureMap( net, "bbox_reg", &reg_res );
        SeetaGetFeatureMap( net, "cls_prob", &cls_res );

        int n_reg = reg_res.number;
        int c_reg = reg_res.channel;
        int h_reg = reg_res.height;
        int w_reg = reg_res.width;
        int n_cls = cls_res.number;
        int c_cls = cls_res.channel;
        int h_cls = cls_res.height;
        int w_cls = cls_res.width;

        ( void )( n_reg );
        ( void )( c_reg );
        ( void )( n_cls );
        ( void )( c_cls );
        ( void )( h_cls );
        ( void )( w_cls );

        float *reg_data = reg_res.data_point_float;
        float *cls_data = cls_res.data_point_float;

        float w = net_size * cur_scale;
        for( int i = 0; i < h_reg; i++ )
        {
            for( int j = 0; j < w_reg; j++ )
            {
                if( cls_data[( 1 * h_reg + i ) * w_reg + j] > thres )
                {
                    float sn = reg_data[( 0 * h_reg + i ) * w_reg + j];
                    float xn = reg_data[( 1 * h_reg + i ) * w_reg + j];
                    float yn = reg_data[( 2 * h_reg + i ) * w_reg + j];

                    int rx, ry, rw;

                    if( end2end_ )
                    {
                        int crop_x = int( j * cur_scale * stride_ );
                        int crop_y = int( i * cur_scale * stride_ );
                        int crop_w = int( w );
                        rx = int( crop_x - 0.5 * sn * crop_w + crop_w * sn * xn + 0.5 * crop_w ) + width;
                        ry = int( crop_y - 0.5 * sn * crop_w + crop_w * sn * yn + 0.5 * crop_w ) + height;
                        rw = int( sn * crop_w );
                    }
                    else
                    {
                        rx = int( j * cur_scale * stride_ + xn * w ) + width;
                        ry = int( i * cur_scale * stride_ + yn * w ) + height;
                        rw = int( w * sn );
                    }

                    if( Legal( rx, ry, img_pad ) && Legal( rx + rw - 1, ry + rw - 1, img_pad ) )
                    {
                        winList.push_back( Rect( rx, ry, rw, rw, count_scale, cls_data[( 1 * h_reg + i ) * w_reg + j] ) );
                    }
                }
            }
        }

        img_resized.height = int( img_resized.height / scale_ );
        img_resized.width = int( img_resized.width / scale_ );
        cur_scale = float( img.height ) / img_resized.height;
        count_scale++;
    }
    delete[] img_resized.data;
    return winList;
}

void Impl::SetInput( const SeetaImageData &img, int dim, std::vector<Rect> &winList, uint8_t *tmp_data )
{
    SeetaImageData tmp;
    tmp.width = dim;
    tmp.height = dim;
    tmp.channels = img.channels;
    for( size_t i = 0; i < winList.size(); i++ )
    {
        ResizeImage(
            img.data, img.width, img.height, img.channels,
            tmp_data + i * tmp.channels * tmp.height * tmp.width, tmp.width, tmp.height, tmp.channels,
            winList[i].x, winList[i].y, winList[i].width, winList[i].height );
    }
}

std::vector<Rect> Impl::RunNet( const SeetaImageData &img, SeetaNet_Net *&net, float thres, int dim, std::vector<Rect> &winList )
{
    if( winList.size() == 0 )
        return winList;
    std::vector<Rect> ret;
    int local_max_batch_size = ( dim == 24 ) ? max_batch_size[0] : max_batch_size[1];
    if( int( winList.size() ) < local_max_batch_size ) local_max_batch_size = int( winList.size() );
    uint8_t *tmp_data = new uint8_t[local_max_batch_size * img.channels * dim * dim];
    while( winList.size() )
    {
        std::vector<Rect> tmp_winList;
        int count = local_max_batch_size;
        while( winList.size() && count )
        {
            tmp_winList.push_back( winList.back() );
            winList.pop_back();
            count--;
        }
        SetInput( img, dim, tmp_winList, tmp_data );
        SeetaNet_InputOutputData tmp_input;
        tmp_input.number = int( tmp_winList.size() );
        tmp_input.channel = img.channels;
        tmp_input.height = dim;
        tmp_input.width = dim;
        tmp_input.buffer_type = SEETANET_BGR_IMGE_CHAR;
        tmp_input.data_point_char = tmp_data;
        SeetaRunNetChar( net, 1, &tmp_input );

        SeetaNet_InputOutputData reg_res, cls_res;
        SeetaGetFeatureMap( net, "bbox_reg", &reg_res );
        SeetaGetFeatureMap( net, "cls_prob", &cls_res );

        int n_reg = reg_res.number;
        int c_reg = reg_res.channel;
        int h_reg = reg_res.height;
        int w_reg = reg_res.width;
        int n_cls = cls_res.number;
        int c_cls = cls_res.channel;
        int h_cls = cls_res.height;
        int w_cls = cls_res.width;

        ( void )( n_reg );

        float *reg_data = reg_res.data_point_float;
        float *cls_data = cls_res.data_point_float;

        for( int i = 0; i < n_cls; i++ )
        {
            if( cls_data[( i * c_cls + 1 ) * h_cls * w_cls] > thres )
            {
                float sn = reg_data[( i * c_reg + 0 ) * h_reg * w_reg];
                float xn = reg_data[( i * c_reg + 1 ) * h_reg * w_reg];
                float yn = reg_data[( i * c_reg + 2 ) * h_reg * w_reg];

                int rx, ry, rw;

                if( end2end_ )
                {
                    int crop_x = tmp_winList[i].x;
                    int crop_y = tmp_winList[i].y;
                    int crop_w = tmp_winList[i].width;
                    rw = int( sn * crop_w );
                    rx = int( crop_x - 0.5 * sn * crop_w + crop_w * sn * xn + 0.5 * crop_w );
                    ry = int( crop_y - 0.5 * sn * crop_w + crop_w * sn * yn + 0.5 * crop_w );
                }
                else
                {
                    rx = int( tmp_winList[i].x + xn * tmp_winList[i].width );
                    ry = int( tmp_winList[i].y + yn * tmp_winList[i].width );
                    rw = int( tmp_winList[i].width * sn );
                }

                if( Legal( rx, ry, img ) && Legal( rx + rw - 1, ry + rw - 1, img ) )
                {
                    ret.push_back( Rect( rx, ry, rw, rw, tmp_winList[i].scale, cls_data[( i * c_cls + 1 ) * h_cls * w_cls] ) );
                }
            }
        }

    }
    delete[] tmp_data;
    return ret;
}

std::vector<SeetaFaceInfo> Impl::TransWindow( const SeetaImageData &img, const SeetaImageData &img_pad, std::vector<Rect> &winList )
{
    int row = ( img_pad.height - img.height ) / 2;
    int col = ( img_pad.width - img.width ) / 2;

    std::vector<SeetaFaceInfo> ret;
    for( size_t i = 0; i < winList.size(); i++ )
    {
        winList[i].x -= col;
        winList[i].y -= row;
        winList[i].y -= int( 0.1 * winList[i].height );
        winList[i].height = int( 1.2 * winList[i].height );
        int x1 = CLAMP( winList[i].x, 0, img.width - 1 );
        int y1 = CLAMP( winList[i].y, 0, img.height - 1 );
        int x2 = CLAMP( winList[i].x + winList[i].width - 1, 0, img.width - 1 );
        int y2 = CLAMP( winList[i].y + winList[i].height - 1, 0, img.height - 1 );
        int w = x2 - x1 + 1;
        int h = y2 - y1 + 1;
        if( w > 0 && h > 0 )
        {
            SeetaFaceInfo f;
            f.pos.x = x1;
            f.pos.y = y1;
            f.pos.width = w;
            f.pos.height = h;
            f.score = float(winList[i].conf);
            ret.push_back( f );
        }
    }
    return ret;
}

std::vector<Rect> Impl::NMS( std::vector<Rect> &winList, bool local, float threshold )
{
    if( winList.size() == 0 )
        return winList;
    std::sort( winList.begin(), winList.end(), CompareWin );
    std::vector<bool> flag( winList.size(), false );
    for( size_t i = 0; i < winList.size(); i++ )
    {
        if( flag[i] )
            continue;
        for( size_t j = i + 1; j < winList.size(); j++ )
        {
            if( local && winList[i].scale != winList[j].scale )
                continue;
            if( IoU( winList[i], winList[j] ) > threshold )
                flag[j] = true;
        }
    }
    std::vector<Rect> ret;
    for( size_t i = 0; i < winList.size(); i++ )
    {
        if( !flag[i] ) ret.push_back( winList[i] );
    }
    return ret;
}

void Impl::LoadModelBuffer( const char *model_buffer, int64_t buffer_lenght12, int64_t buffer_lenght24, int64_t buffer_lenght48, SeetaDevice device, int gpuid )
{
    using self = FaceDetectorPrivate;
    SeetaNet_DEVICE_TYPE type = SEETANET_CPU_DEVICE;

    max_pad_ratio = 0.2f;
    nms_threshold_[0] = 0.8f;
    nms_threshold_[1] = 0.8f;
    nms_threshold_[2] = 0.3f;
    max_pad_w = 100;
    max_pad_h = 100;
    max_batch_size[0] = 1000;
    max_batch_size[1] = 500;

    int64_t model_index12 = 0;
    int64_t model_index24 = model_index12 + buffer_lenght12;
    int64_t model_index48 = model_index24 + buffer_lenght24;

    const char *model_buffer12 = model_buffer + model_index12;
    const char *model_buffer24 = model_buffer + model_index24;
    const char *model_buffer48 = model_buffer + model_index48;

    SeetaReadModelFromBuffer( model_buffer12, size_t( buffer_lenght12 ), &model_[0] );
    SeetaModelResetInput( model_[0], width_limit_, height_limit_ );
    SeetaCreateNet( model_[0], 1, type, &net_[0] );

    SeetaReadModelFromBuffer( model_buffer24, size_t( buffer_lenght24 ), &model_[1] );
    SeetaCreateNet( model_[1], max_batch_size[0], type, &net_[1] );

    SeetaReadModelFromBuffer( model_buffer48, size_t( buffer_lenght48 ), &model_[2] );
    SeetaCreateNet( model_[2], max_batch_size[1], type, &net_[2] );
}

bool Impl::ResizeImage( const unsigned char *src_im, int src_width, int src_height, int src_channels,
                        unsigned char *dst_im, int dst_width, int dst_height, int dst_channels,
                        int crop_x, int crop_y, int crop_w, int crop_h )
{
    if( src_channels != dst_channels || !( src_channels == 3 || src_channels == 1 ) )
    {
        std::cout << "<Illegal image channels!>" << std::endl;
        std::cout << "src_img: " << src_channels << std::endl;
        std::cout << "dst_img: " << dst_channels << std::endl;
        return false;
    }

    if( crop_x == -1 )
    {
        crop_x = 0;
        crop_y = 0;
        crop_w = src_width;
        crop_h = src_height;
    }

    float lfx_scl = float( crop_w ) / dst_width;
    float lfy_scl = float( crop_h ) / dst_height;

    float *wx = new float[dst_width];
    float *wy = new float[dst_height];
    int *nx = new int[dst_width];
    int *ny = new int[dst_height];

    for( int n_y_d = 0; n_y_d < dst_height; n_y_d++ )
    {
        float lf_y_s = lfy_scl * n_y_d + crop_y;
        ny[n_y_d] = int( lf_y_s );
        if( ny[n_y_d] == src_height - 1 )
            ny[n_y_d] -= 1;
        wy[n_y_d] = lf_y_s - ny[n_y_d];
    }

    for( int n_x_d = 0; n_x_d < dst_width; n_x_d++ )
    {
        float lf_x_s = lfx_scl * n_x_d + crop_x;
        nx[n_x_d] = int( lf_x_s );
        if( nx[n_x_d] == src_width - 1 )
            nx[n_x_d] -= 1;
        wx[n_x_d] = lf_x_s - nx[n_x_d];
    }

    for( int i = 0; i < dst_height * dst_width; i++ )
    {
        int n_y_d = i / dst_width;
        int n_x_d = i - n_y_d * dst_width;
        int n_x_s = nx[n_x_d];
        int n_y_s = ny[n_y_d];

        float lf_weight_x = wx[n_x_d];
        float lf_weight_y = wy[n_y_d];

        float s1 = ( 1 - lf_weight_y ) * ( 1 - lf_weight_x );
        float s2 = ( 1 - lf_weight_y ) * lf_weight_x;
        float s3 = lf_weight_y * ( 1 - lf_weight_x );
        float s4 = lf_weight_y * lf_weight_x;
        int d_index = ( n_y_d * dst_width + n_x_d ) * dst_channels;
        int s_index1 = ( n_y_s * src_width + n_x_s ) * src_channels;
        int s_index2 = ( n_y_s * src_width + n_x_s + 1 ) * src_channels;
        int s_index3 = ( ( n_y_s + 1 ) * src_width + n_x_s ) * src_channels;
        int s_index4 = ( ( n_y_s + 1 ) * src_width + n_x_s + 1 ) * src_channels;
        for( int c = 0; c < src_channels; c++ )
        {
            dst_im[d_index + c] = static_cast<uint8_t>(
                                      s1 * src_im[s_index1 + c] +
                                      s2 * src_im[s_index2 + c] +
                                      s3 * src_im[s_index3 + c] +
                                      s4 * src_im[s_index4 + c] );
        }
    }

    delete[] wx;
    delete[] wy;
    delete[] nx;
    delete[] ny;
    return true;
}

bool Impl::PadImage( const unsigned char *src_im, int src_width, int src_height, int src_channels,
                     unsigned char *dst_im, int pad_w, int pad_h )
{
    int dst_width = src_width + 2 * pad_w;
    int dst_height = src_height + 2 * pad_h;
    memset( dst_im, 0, sizeof( unsigned char ) * src_channels * dst_height * dst_width );

    for( int i = 0; i < src_height; i++ )
    {
        memcpy( dst_im + ( ( i + pad_h ) * dst_width + pad_w ) * src_channels, src_im + i * src_width * src_channels, sizeof( unsigned char ) * src_width * src_channels );
    }
    return true;
}

int foo( char *c )
{
    int n = 0;
    for( int i = 0; i < 4; i++ )
    {
        n <<= 8;
        n += ( unsigned char )c[i];
    }
    return n;
}


FaceDetectorPrivate::FaceDetectorPrivate( const char *model_path )
    : FaceDetectorPrivate( model_path, SEETA_DEVICE_AUTO, 0 )
{
}

FaceDetectorPrivate::FaceDetectorPrivate( const char *model_path, SeetaDevice device, int gpuid )
    : FaceDetectorPrivate( model_path, CoreSize( -1, -1 ), device, gpuid )
{
}

FaceDetectorPrivate::FaceDetectorPrivate( const char *model_path, const CoreSize &core_size )
    : FaceDetectorPrivate( model_path, core_size, SEETA_DEVICE_AUTO, 0 )
{
}

static std::vector<int> version( const std::string &v )
{
    std::vector<int> vi;
    auto vs = Split( v, '.' );
    for( auto &s : vs ) vi.push_back( std::atoi( s.c_str() ) );
    while( vi.size() < 3 ) vi.push_back( 0 );
    return std::move( vi );
}

static int toint( const std::string &str )
{
    if( str.empty() ) return 0;
    if( str.back() == 'k' || str.back() == 'K' )
    {
        return std::atoi( str.substr( 0, str.length() - 1 ).c_str() ) * 1000;
    }
    return std::atoi( str.c_str() );
}

static bool match_end( const char *str, int i )
{
    const char &ch = str[i];
    if( ch == '\0' ) return true;
    return false;
}

static bool match_num2( const char *str, int i )
{
    const char &ch = str[i];
    if( ch == '\0' ) return true;
    if( '0' <= ch && ch <= '9' ) return match_num2( str, i + 1 );
    if( ch == 'k' || ch == 'K' ) return match_end( str, i + 1 );
    return false;
}

static bool match_pre_num2( const char *str, int i )
{
    const char &ch = str[i];
    if( ch == '\0' ) return false;
    if( '1' <= ch && ch <= '9' ) return match_num2( str, i + 1 );
    return false;
}

static bool match_x( const char *str, int i )
{
    const char &ch = str[i];
    if( ch != 'x' ) return false;
    return match_num2( str, i + 1 );
}

static bool match_num1( const char *str, int i )
{
    const char &ch = str[i];
    if( !ch ) return false;
    if( '0' <= ch && ch <= '9' ) return match_num1( str, i + 1 );
    if( ch == 'k' || ch == 'K' ) return match_x( str, i + 1 );
    if( ch == 'x' ) return match_pre_num2( str, i + 1 );
    return false;
}

static bool match_pre_num1( const char *str, int i )
{
    const char &ch = str[i];
    if( !ch ) return false;
    if( '1' <= ch && ch <= '9' ) return match_num1( str, i + 1 );
    return false;
}

static bool match_widthxheight( const std::string &str )
{
    return match_pre_num1( str.c_str(), 0 );
}

static void SplitImageLimit( const std::string &model_path, int *width, int *height )
{
    auto model_marks = Split( model_path, '.' );
    std::string widthxheight;
    for( auto &mark : model_marks )
    {
        if( match_widthxheight( mark ) )
        {
            widthxheight = mark;
            break;
        }
    }
    if( widthxheight.empty() ) return;
    auto pos = widthxheight.find( 'x' );
    *width = toint( widthxheight.substr( 0, pos ) );
    *height = toint( widthxheight.substr( pos + 1 ) );
}


FaceDetectorPrivate::FaceDetectorPrivate( const char *model_path, const CoreSize &core_size, SeetaDevice deviece, int gpuid )
    : impl_( new Impl() )
{
    std::ifstream inf( model_path, std::ios::binary );

    if( !inf.is_open() )
    {
        std::cerr << "Error: Can not access \"" << model_path << "\"" << std::endl;
        throw std::logic_error( "Model missing" );
    }


    Impl *p = ( Impl * )impl_;

    int width_limit = 640;
    int height_limit = 480;

    //SplitImageLimit(model_path, &width_limit, &height_limit);
    p->width_limit_ = width_limit;
    p->height_limit_ = height_limit;

    std::shared_ptr<char> sta_buffer;
    std::string sta_data;
    char *buffer;

    inf.seekg( 0, std::ios::end );

    auto sta_length = inf.tellg();
    sta_buffer.reset( new char[size_t( sta_length )], std::default_delete<char[]>() );


    inf.seekg( 0, std::ios::beg );
    inf.read( sta_buffer.get(), sta_length );
    buffer = sta_buffer.get();

    inf.close();
    if( core_size.width > 0 ) p->width_limit_ = std::max<int>( 100, core_size.width );
    if( core_size.height > 0 ) p->height_limit_ = std::max<int>( 100, core_size.height );

    int length1 = foo( buffer );
    int length2 = foo( buffer + 4 );
    int length3 = foo( buffer + 8 );

    int offset = 12;
    p->LoadModelBuffer( buffer + offset, length1, length2, length3, deviece, gpuid );

    SetMinFaceSize( 40 );
    SetScoreThresh( p->thresh1_, p->thresh2_, p->thresh3_ );
    SetImagePyramidScaleFactor( 1.414f );

    std::cout << "[INFO] FaceDetector: " << "Core size: " << p->width_limit_ << "x" << p->height_limit_ << std::endl;
}


FaceDetectorPrivate::~FaceDetectorPrivate()
{
    Impl *p = ( Impl * )impl_;
    if( !p ) return;

    SeetaReleaseModel( p->model_[0] );
    SeetaReleaseModel( p->model_[1] );
    SeetaReleaseModel( p->model_[2] );
    SeetaReleaseNet( p->net_[0] );
    SeetaReleaseNet( p->net_[1] );
    SeetaReleaseNet( p->net_[2] );

    delete p;
}

void FaceDetectorPrivate::SetVideoStable( bool stable )
{
    Impl *p = ( Impl * )impl_;
    if( stable != p->stable_ ) p->preList_.clear();
    p->stable_ = stable;
}

bool FaceDetectorPrivate::GetVideoStable() const
{
    Impl *p = ( Impl * )impl_;
    return p->stable_;
}

FaceDetectorPrivate::CoreSize FaceDetectorPrivate::GetCoreSize() const
{
    Impl *p = ( Impl * )impl_;
    return CoreSize( p->width_limit_, p->height_limit_ );
}

static seeta::Image ScaleImage( const seeta::Image &image, int width, int height, float *scale = nullptr )
{
    if( scale ) *scale = 1;
    float scale_w = 1.0f * width / image.width();
    float scale_h = 1.0f * height / image.height();
    float scale_a = std::min( scale_w, scale_h );
    if( scale_a >= 1 ) return image;
    seeta::Image resized_image = seeta::resize( image, seeta::Size( int( image.width() * scale_a ), int( image.height() * scale_a ) ) );
    if( scale ) *scale = scale_a;
    return resized_image;
}

SeetaFaceInfoArray FaceDetectorPrivate::Detect( const SeetaImageData &image )
{
    SeetaFaceInfoArray  ret;
    ret.size = 0;
    ret.data = nullptr;
    Impl *p = ( Impl * )impl_;
    if( !p->IsLegalImage( image ) )
    {
        return ret;
    }

    // sclae image
    seeta::Image img = image;

    float scale = 1;
    seeta::Image scaled_img = ScaleImage( img, p->width_limit_, p->height_limit_, &scale );
    img = scaled_img;

    img = seeta::color( img );

    int pad_h = std::min( int( p->max_pad_ratio * img.height() ), p->max_pad_h );
    int pad_w = std::min( int( p->max_pad_ratio * img.width() ), p->max_pad_w );
    SeetaImageData img_pad;
    img_pad.width = img.width() + 2 * pad_w;
    img_pad.height = img.height() + 2 * pad_h;
    img_pad.channels = img.channels();
    img_pad.data = new uint8_t[img_pad.channels * img_pad.height * img_pad.width];
    p->PadImage( img.data(), img.width(), img.height(), img.channels(), img_pad.data, pad_w, pad_h );

    auto local_min_face_size = std::max( 12, int( p->min_face_ * scale ) );
    auto local_max_face_size = p->max_face_;
    if( local_max_face_size > 0 ) local_max_face_size = std::max( 12, int( p->max_face_ * scale ) );

    std::vector<Rect> winList;
    winList = p->SlidingWindow( img, img_pad, p->net_[0], p->class_threshold_[0], local_min_face_size, local_max_face_size );
    winList = p->NMS( winList, true, p->nms_threshold_[0] );

    // std::cout << "Stage1 result: " << winList.size() << std::endl;

    winList = p->RunNet( img_pad, p->net_[1], p->class_threshold_[1], 24, winList );
    winList = p->NMS( winList, true, p->nms_threshold_[1] );

    // std::cout << "Stage2 result: " << winList.size() << std::endl;

    winList = p->RunNet( img_pad, p->net_[2], p->class_threshold_[2], 48, winList );
    winList = p->NMS( winList, false, p->nms_threshold_[2] );

    // std::cout << "Stage3 result: " << winList.size() << std::endl;

    // scale result
    for( auto &info : winList )
    {
        info.x -= pad_w;
        info.y -= pad_h;

        info.x = int( info.x / scale );
        info.y = int( info.y / scale );
        info.width = int( info.width / scale );
        info.height = int( info.height / scale );
    }

    std::vector<Rect> &preList = p->preList_;
    if( p->stable_ )
    {
        for( size_t i = 0; i < winList.size(); i++ )
        {
            for( size_t j = 0; j < preList.size(); j++ )
            {
                if( p->IoU( winList[i], preList[j] ) > 0.85 )
                    winList[i] = preList[j];
                else
                    if( p->IoU( winList[i], preList[j] ) > 0.6 )
                    {
                        winList[i].x = ( winList[i].x + preList[j].x ) / 2;
                        winList[i].y = ( winList[i].y + preList[j].y ) / 2;
                        winList[i].width = ( winList[i].width + preList[j].width ) / 2;
                        winList[i].height = ( winList[i].height + preList[j].height ) / 2;
                    }
            }
        }
        preList = winList;
    }

    delete[] img_pad.data;
    m_pre_faces.clear();
    m_pre_faces = p->TransWindow( image, image, winList );
    ret.size = int(m_pre_faces.size());
    ret.data = m_pre_faces.data();
    return ret;
}

void FaceDetectorPrivate::SetMinFaceSize( int32_t size )
{
    Impl *p = ( Impl * )impl_;
    p->min_face_ = size > 20 ? size : 20;
    p->min_face_ = int( p->min_face_ * 1.4f );
}

void FaceDetectorPrivate::SetImagePyramidScaleFactor( float factor )
{
    Impl *p = ( Impl * )impl_;
    p->scale_ = factor > 1.414f ? factor : 1.414f;
}

float FaceDetectorPrivate::GetScoreThresh1() const
{
    Impl *p = ( Impl * )impl_;
    return p->class_threshold_[0];
}

float FaceDetectorPrivate::GetScoreThresh2() const
{
    Impl *p = ( Impl * )impl_;
    return p->class_threshold_[1];
}
float FaceDetectorPrivate::GetScoreThresh3() const
{
    Impl *p = ( Impl * )impl_;
    return p->class_threshold_[2];
}
void FaceDetectorPrivate::SetScoreThresh1( float thresh1 )
{
    Impl *p = ( Impl * )impl_;
    p->class_threshold_[0] = thresh1;
}
void FaceDetectorPrivate::SetScoreThresh2( float thresh2 )
{
    Impl *p = ( Impl * )impl_;
    p->class_threshold_[1] = thresh2;
}
void FaceDetectorPrivate::SetScoreThresh3( float thresh3 )
{
    Impl *p = ( Impl * )impl_;
    p->class_threshold_[2] = thresh3;
}

void FaceDetectorPrivate::SetScoreThresh( float thresh1, float thresh2, float thresh3 )
{
    Impl *p = ( Impl * )impl_;
    p->class_threshold_[0] = thresh1;
    p->class_threshold_[1] = thresh2;
    p->class_threshold_[2] = thresh3;
}

int32_t FaceDetectorPrivate::GetMinFaceSize() const
{
    Impl *p = ( Impl * )impl_;
    return p->min_face_;
}

float FaceDetectorPrivate::GetImagePyramidScaleFactor() const
{
    Impl *p = ( Impl * )impl_;
    return p->scale_;
}

void FaceDetectorPrivate::GetScoreThresh( float *thresh1, float *thresh2, float *thresh3 ) const
{
    Impl *p = ( Impl * )impl_;
    if( thresh1 != nullptr ) *thresh1 = p->class_threshold_[0];
    if( thresh2 != nullptr ) *thresh2 = p->class_threshold_[1];
    if( thresh3 != nullptr ) *thresh3 = p->class_threshold_[2];
}
