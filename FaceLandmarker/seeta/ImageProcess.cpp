#include "ImageProcess.h"

#include "common_alignment.h"
#include <array>
#include <climits>
#include <cfloat>



const seeta::Image seeta::color( const Image &img )
{
    if( img.channels() == 3 ) return img;
    if( img.channels() != 1 ) throw std::logic_error( str( "Can not convert image with channels: ", img.channels() ) );
    Image color_img( img.width(), img.height(), 3 );
    int _count = img.width() * img.height();
    for( int i = 0; i < _count; ++i )
    {
        const seeta::Image::Datum *gray = &img[i];
        seeta::Image::Datum *color = &color_img[i * 3];
        color[0] = color[1] = color[2] = gray[0];
    }
    return std::move( color_img );
}

const seeta::Image seeta::gray( const Image &img )
{
    if( img.channels() == 1 ) return img;
    if( img.channels() != 3 ) throw std::logic_error( str( "Can not convert image with channels: ", img.channels() ) );
    Image gray_img( img.width(), img.height(), 1 );
    int _count = img.width() * img.height();
    for( int i = 0; i < _count; ++i )
    {
        const seeta::Image::Datum *color = &img[i * 3];
        seeta::Image::Datum *gray = &gray_img[i];
        gray[0] = static_cast<seeta::Image::Datum>( 0.1140 * color[0] + 0.5870 * color[1] + 0.2989 * color[2] );
    }
    return std::move( gray_img );
}

const seeta::Image seeta::crop( const Image &img, const Rect &rect )
{
    using namespace std;
    // Adjust rect
    Rect fixed_rect = rect;
    fixed_rect.width += fixed_rect.x;
    fixed_rect.height += fixed_rect.y;
    fixed_rect.x = max( 0, min( img.width() - 1, fixed_rect.x ) );
    fixed_rect.y = max( 0, min( img.height() - 1, fixed_rect.y ) );
    fixed_rect.width = max( 0, min( img.width() - 1, fixed_rect.width ) );
    fixed_rect.height = max( 0, min( img.height() - 1, fixed_rect.height ) );
    fixed_rect.width -= fixed_rect.x;
    fixed_rect.height -= fixed_rect.y;

    // crop image
    Image result( rect.width, rect.height, img.channels() );
    memset( result.data(), 0, sizeof( Image::Datum ) * result.width() * result.height() * result.channels() );

    const Image::Datum *iter_in_ptr = &img.data()[fixed_rect.y * img.width() * img.channels() + fixed_rect.x * img.channels()];
    int iter_in_step = img.width() * img.channels();
    int copy_size = fixed_rect.width * img.channels();
    int iter_size = fixed_rect.height;
    Image::Datum *iter_out_ptr = &result.data()[max( 0, fixed_rect.y - rect.y ) * result.width() * result.channels() + max( 0, fixed_rect.x - rect.x ) * result.channels()];
    int iter_out_step = result.width() * result.channels();

    for( int i = 0; i < iter_size; ++i, iter_in_ptr += iter_in_step, iter_out_ptr += iter_out_step )
    {
        CopyData( iter_out_ptr, iter_in_ptr, copy_size );
    }

    return std::move( result );
}


const seeta::Image seeta::pad( const Image &img, const Padding &padding )
{
    int w = padding.width;
    int h = padding.height;
    if( w * h < 0 )
    {
        throw std::logic_error( str( "Illegal padding arguments (", w, ", ", h, ")" ) );
    }
    if( w == 0 && h == 0 )
    {
        return img;
    }
    if( w < 0 || h < 0 )
    {
        return crop( img, Rect( 0 - w, 0 - h, img.width() + 2 * w, img.height() + 2 * h ) );
    }

    // pad image
    Image result( img.width() + 2 * w, img.height() + 2 * h, img.channels() );
    memset( result.data(), 0, result.count() * sizeof( Image::Datum ) );

    const Image::Datum *iter_in_ptr = &img.data()[0];
    int iter_in_step = img.width() * img.channels();
    int copy_size = img.width() * img.channels();
    int iter_size = img.height();
    Image::Datum *iter_out_ptr = &result.data()[h * result.width() * result.channels() + w * result.channels()];
    int iter_out_step = result.width() * result.channels();

    for( int i = 0; i < iter_size; ++i, iter_in_ptr += iter_in_step, iter_out_ptr += iter_out_step )
    {
        CopyData( iter_out_ptr, iter_in_ptr, copy_size );
    }

    return std::move( result );
}

/**
* Copy from "VIPLNetGenderPredictor.cpp". Original author: ZhangJie, HeZhenliang
*/
const seeta::Image seeta::resize( const Image &img, const Size &size )
{
    using namespace std;

    if( img.width() == size.width && img.height() == size.height )
    {
        return img;
    }

    Image resized_img( size.width, size.height, img.channels() );

    int src_width = img.width();
    int src_height = img.height();
    int src_channels = img.channels();
    int dst_width = resized_img.width();
    int dst_height = resized_img.height();
    int dst_channels = resized_img.channels();
    const seeta::Image::Datum *src_im = img.data();
    seeta::Image::Datum *dst_im = resized_img.data();

    double lfx_scl = static_cast<double>( src_width ) / dst_width;
    double lfy_scl = static_cast<double>( src_height ) / dst_height;
    double bias_x = lfx_scl / 2 - 0.5;
    double bias_y = lfy_scl / 2 - 0.5;

    for( int n_y_d = 0; n_y_d < dst_height; n_y_d++ )
    {
        std::unique_ptr<double[]> raw_channel_buff( new double[src_channels] );
        double *channel_buff = raw_channel_buff.get();
        for( int n_x_d = 0; n_x_d < dst_width; n_x_d++ )
        {
            double lf_x_s = lfx_scl * n_x_d + bias_x;
            double lf_y_s = lfy_scl * n_y_d + bias_y;

            lf_x_s = lf_x_s >= 0 ? lf_x_s : 0;
            lf_x_s = lf_x_s < src_width - 1 ? lf_x_s : src_width - 1 - 1e-5;
            lf_y_s = lf_y_s >= 0 ? lf_y_s : 0;
            lf_y_s = lf_y_s < src_height - 1 ? lf_y_s : src_height - 1 - 1e-5;

            int n_x_s = static_cast<int>( lf_x_s );
            int n_y_s = static_cast<int>( lf_y_s );

            double lf_weight_x = lf_x_s - n_x_s;
            double lf_weight_y = lf_y_s - n_y_s;

            for( int c = 0; c < src_channels; c++ )
            {
                channel_buff[c] =
                    ( 1 - lf_weight_y ) * ( 1 - lf_weight_x ) * src_im[( n_y_s * src_width + n_x_s ) * src_channels + c] +
                    ( 1 - lf_weight_y ) * lf_weight_x * src_im[( n_y_s * src_width + n_x_s + 1 ) * src_channels + c] +
                    lf_weight_y * ( 1 - lf_weight_x ) * src_im[( ( n_y_s + 1 ) * src_width + n_x_s ) * src_channels + c] +
                    lf_weight_y * lf_weight_x * src_im[( ( n_y_s + 1 ) * src_width + n_x_s + 1 ) * src_channels + c];
            }

            // must have same channels
            for( int c = 0; c < dst_channels; c++ )
            {
                dst_im[( n_y_d * dst_width + n_x_d ) * dst_channels + c] =
                    static_cast<seeta::Image::Datum>( max( 0.0f, min( 255.0f, static_cast<float>( channel_buff[c] ) ) ) );
            }
        }
    }

    return std::move( resized_img );
}

const seeta::Image seeta::crop_resize( const Image &img, const Rect &_rect, const Size &size )
{
    using namespace std;

    Image resized_img( size.width, size.height, img.channels() );

    // Ajuest rect
    Rect rect = _rect;
    rect.x = max( 0, min( img.width() - 1, rect.x ) );
    rect.y = max( 0, min( img.height() - 1, rect.y ) );
    rect.width = max( 0, min( img.width() - rect.x, rect.width ) );
    rect.height = max( 0, min( img.height() - rect.y, rect.height ) );

    int src_width = img.width();
    int src_height = img.height();
    int src_channels = img.channels();
    int dst_width = size.width;
    int dst_height = size.height;
    int dst_channels = src_channels;
    int crop_x = rect.x;
    int crop_y = rect.y;
    int crop_w = rect.width;
    int crop_h = rect.height;
    const seeta::Image::Datum *src_im = img.data();
    seeta::Image::Datum *dst_im = resized_img.data();

    float lfx_scl = float( crop_w ) / dst_width;
    float lfy_scl = float( crop_h ) / dst_height;

    std::unique_ptr<float[]> raw_bufferf( new float[dst_width + dst_height] );
    std::unique_ptr<int[]> raw_bufferi( new int[dst_width + dst_height] );

    float *wx = raw_bufferf.get();
    float *wy = raw_bufferf.get() + dst_width;
    int *nx = raw_bufferi.get();
    int *ny = raw_bufferi.get() + dst_width;

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
            ny[n_x_d] -= 1;
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
            float ans = s1 * src_im[s_index1 + c] +
                        s2 * src_im[s_index2 + c] +
                        s3 * src_im[s_index3 + c] +
                        s4 * src_im[s_index4 + c];
            dst_im[d_index + c] =
                static_cast<seeta::Image::Datum>( max( 0.0f, min( 255.0f, static_cast<float>( ans ) ) ) );
        }
    }
    return std::move( resized_img );
}

inline uint8_t round_uint8_t( double v )
{
    int iv = static_cast<int>( std::round( v ) );
    return static_cast<uint8_t>( static_cast<unsigned>( iv ) <= UCHAR_MAX ? iv : iv > 0 ? UCHAR_MAX : 0 );
}

const seeta::Image seeta::equalize_hist( const Image &img )
{
    if( img.channels() == 0 || img.height() == 0 || img.width() == 0 ) return img;
    // get hist
    // int hist[3][256] = { 0 };
    const int HIST_SIZE = 256;
    std::vector<std::array<int, HIST_SIZE>> hist( img.channels() );
    std::vector<int> count( img.channels() );
    for( int c = 0; c < img.channels(); ++c )
    {
        count[c] = img.height() * img.width();
        for( int i = 0; i < HIST_SIZE; ++i ) hist[c][i] = 0;
        for( int h = 0; h < img.height(); ++h )
        {
            for( int w = 0; w < img.width(); ++w )
            {
                ++hist[c][img.data( h, w, c )];
            }
        }
    }
    // get p
    std::vector<std::array<Image::Datum, HIST_SIZE>> LUT( img.channels() );
    for( int c = 0; c < img.channels(); ++c )
    {
        int i = 0;
        // 
        while( !hist[c][i] ) ++i;
        float scale = ( HIST_SIZE - 1.f ) / ( count[c] - hist[c][i] );

        if( hist[c][i] == count[c] )
        {
            LUT[c][i] = i;
            continue;
        }

        // LUT[c][0] = static_cast<Image::Datum>(hist[c][0] * scale);
        int sum = 0;
        for( LUT[c][i++] = 0; i < HIST_SIZE; ++i )
        {
            sum += hist[c][i];
            LUT[c][i] = round_uint8_t( sum * scale );
        }
    }

    // equalize
    Image result( img.width(), img.height(), img.channels() );
    for( int c = 0; c < img.channels(); ++c )
    {
        for( int h = 0; h < img.height(); ++h )
        {
            for( int w = 0; w < img.width(); ++w )
            {
                result.data( h, w, c ) = LUT[c][img.data( h, w, c )];
            }
        }
    }

    return std::move( result );
}

void seeta::fill( Image &img, const Point &point, const Image &patch )
{
    if( img.channels() != patch.channels() )
    {
        throw std::logic_error( str( "Can not file image with mismatch channels ", img.channels(), " vs ", patch.channels() ) );
    }

    int dst_y_start = std::max<int>( 0, point.y );
    int dst_y_end = std::min<int>( img.height(), point.y + patch.height() );
    int src_y_start = dst_y_start - point.y;
    // int src_y_end = src_y_start + (dst_y_end - dst_y_start);
    int copy_times = dst_y_end - dst_y_start;
    if( copy_times <= 0 ) return;
    int dst_x_start = std::max<int>( 0, point.x );
    int dst_x_end = std::min<int>( img.width(), point.x + patch.width() );
    int src_x_start = dst_x_start - point.x;
    // int src_x_end = src_x_start + (dst_x_end - dst_x_start);
    int copy_size = ( dst_x_end - dst_x_start ) * patch.channels();
    if( copy_size <= 0 ) return;
    int dst_step = img.width() * img.channels();
    int src_step = patch.width() * patch.channels();
    auto *dst_ptr = &img.data( dst_y_start, dst_x_start, 0 );
    const auto *src_ptr = &patch.data( src_y_start, src_x_start, 0 );
    for( int i = 0; i < copy_times; ++i )
    {
        CopyData( dst_ptr, src_ptr, copy_size );
        dst_ptr += dst_step;
        src_ptr += src_step;
    }
}

void seeta::fill( Image &img, const Rect &rect, const Image &patch )
{
    Image fixed_patch = patch;
    if( patch.width() != rect.width || patch.height() != rect.height )
    {
        fixed_patch = resize( patch, rect );
    }
    fill( img, Point( rect ), fixed_patch );
}

const seeta::Meanshape seeta::face_meanshape( int num, int id )
{
    Meanshape shape;
    if( num != 5 || ( id != 0 && id != 1 ) )
    {
        return shape;
    }

    if( id == 0 )
    {
        shape.points =
        {
            { 89.3095, 72.9025 },
            { 169.3095, 72.9025 },
            { 127.8949, 127.0441 },
            { 96.8796, 184.8907 },
            { 159.1065, 184.7601 },
        };
        shape.size = { 256, 256 };
    }
    else
    {
        shape.points =
        {
            { 89.3095, 102.9025 },
            { 169.3095, 102.9025 },
            { 127.8949, 157.0441 },
            { 96.8796, 214.8907 },
            { 159.1065, 214.7601 },
        };
        shape.size = { 256, 256 };
    }

    return shape;
}

const seeta::Meanshape seeta::resize( const Meanshape &shape, double scaler )
{
    Meanshape resized_shape = shape;
    for( size_t i = 0; i < shape.points.size(); ++i )
    {
        resized_shape.points[i].x *= scaler;
        resized_shape.points[i].y *= scaler;
    }
    resized_shape.size.width = int( resized_shape.size.width * scaler );
    resized_shape.size.height = int( resized_shape.size.height * scaler );
    return resized_shape;
}

const seeta::Meanshape seeta::resize( const Meanshape &shape, const Size &size )
{
    if( size.width == shape.size.width && size.height == shape.size.height ) return shape;
    return resize( shape, std::min( static_cast<double>( size.width ) / shape.size.width, static_cast<double>( size.height ) / shape.size.height ) );
}

static const seeta::Image seeta_crop_face(
    const seeta::Image &img,
    const seeta::Meanshape &shape,
    const seeta::Landmarks &marks,
    seeta::SAMPLING_METHOD method,
    const seeta::Size &final_size,
    seeta::Landmarks *final_points = nullptr )
{
    using namespace seeta;
    if( shape.points.empty() || shape.points.size() != marks.points.size() )
    {
        throw std::logic_error( str( "Illegal meanshape and landmarks number (", shape.points.size(), " VS ", marks.points.size(), ")" ) );
    }
    int num = static_cast<int>( shape.points.size() );

    std::unique_ptr<float[]> crop_points( new float[num * 2] );
    std::unique_ptr<float[]> crop_mean_shape( new float[num * 2] );
    std::unique_ptr<float[]> crop_final_points;
    ::SAMPLING_TYPE type;
    for( int i = 0; i < num; ++i )
    {
        crop_points[i * 2] = static_cast<float>( marks.points[i].x );
        crop_points[i * 2 + 1] = static_cast<float>( marks.points[i].y );
        crop_mean_shape[i * 2] = static_cast<float>( shape.points[i].x );
        crop_mean_shape[i * 2 + 1] = static_cast<float>( shape.points[i].y );
    }
    if( final_points )
    {
        crop_final_points.reset( new float[num * 2] );
    }
    switch( method )
    {
        default:
            type = LINEAR;
            break;
        case BY_LINEAR:
            type = LINEAR;
            break;
        case BY_BICUBIC:
            type = BICUBIC;
            break;
    }
    seeta::Image dst( final_size.width, final_size.height, img.channels() );
    bool success = face_crop_core(
                       img.data(), img.width(), img.height(), img.channels(),
                       dst.data(), shape.size.width, shape.size.height,
                       crop_points.get(), num,
                       crop_mean_shape.get(), shape.size.width, shape.size.height,
                       ( final_size.height - shape.size.height ) / 2,
                       ( final_size.height - shape.size.height ) - ( final_size.height - shape.size.height ) / 2,
                       ( final_size.width - shape.size.width ) / 2,
                       ( final_size.width - shape.size.width ) - ( final_size.width - shape.size.width ) / 2,
                       final_points ? crop_final_points.get() : nullptr,
                       type );
    if( final_points )
    {
        final_points->points.resize( num );
        for( int i = 0; i < num; ++i )
        {
            final_points->points[i].x = crop_final_points.get()[2 * i];
            final_points->points[i].y = crop_final_points.get()[2 * i + 1];
        }
    }
    return success ? dst : seeta::Image();
}

const seeta::Image seeta::crop_face( const Image &img, const Meanshape &shape, const Landmarks &marks, SAMPLING_METHOD method )
{
    return seeta_crop_face( img, shape, marks, method, shape.size );
}

const seeta::Image seeta::crop_face( const Image &img, const Meanshape &shape, const Landmarks &marks, SAMPLING_METHOD method, const Size &final_size )
{
    return seeta_crop_face( img, shape, marks, method, final_size );
}

const seeta::Image seeta::crop_face( const Image &img, const Meanshape &shape, const Landmarks &marks, SAMPLING_METHOD method, const Size &final_size, Landmarks &final_points )
{
    return seeta_crop_face( img, shape, marks, method, final_size, &final_points );
}

static void near_sample( const seeta::Image &image, const seeta::PointF &point, uint8_t *pixel )
{
    ( void )( &near_sample );

    const auto &channels = image.channels();
    auto x = int( std::round( point.x ) );
    auto y = int( std::round( point.y ) );
    if( x < 0 || y < 0 || x >= image.width() || y >= image.height() )
    {
        std::memset( pixel, 0, channels );
    }
    else
    {
        std::memcpy( pixel, &image.data( y, x, 0 ), channels );
    }
}

static void linear_sample( const seeta::Image &image, const seeta::PointF &point, uint8_t *pixel )
{
    const auto &channels = image.channels();

    auto x = point.x;
    auto y = point.y;

    int left_x = int( std::floor( x ) );
    int right_x = left_x + 1;
    int top_y = int( std::floor( y ) );
    int bottom_y = top_y + 1;

    if( left_x < 0 || top_y < 0 || right_x >= image.width() || bottom_y >= image.height() )
    {
        std::memset( pixel, 0, channels );
        return;
    }

    double ratio_left = right_x - x;
    double ratio_right = x - left_x;
    double ratio_top = bottom_y - y;
    double ratio_bottom = y - top_y;

    for( int i = 0; i < channels; ++i )
    {
        double value_left_x = ratio_top * image.data( top_y, left_x, i ) + ratio_bottom * image.data( bottom_y, left_x, i );
        double value_right_x = ratio_top * image.data( top_y, right_x, i ) + ratio_bottom * image.data( bottom_y, right_x, i );
        double value = ratio_left * value_left_x + ratio_right * value_right_x;
        pixel[i] = static_cast<uint8_t>( std::max<double>( 0.0f, std::min<double>( 255.0f, value ) ) );
    }
}

const seeta::Image seeta::sample( const Image &image, const Size &size, const Trans2D<double> &transformation )
{
    seeta::Image patch( size.width, size.height, image.channels() );
    for( int y = 0; y < size.height; ++y )
    {
        for( int x = 0; x < size.width; ++x )
        {
            auto pixel = &patch.data( y, x, 0 );
            auto location = seeta::transform( transformation, Vec2D<double>( x, y ) );
            linear_sample( image, seeta::PointF( location.x, location.y ), pixel );
        }
    }
    return std::move( patch );
}

