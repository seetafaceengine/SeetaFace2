#pragma once

#include "Struct.h"
#include "DataHelper.h"
#include <sstream>

namespace seeta
{
    class Meanshape {
    public:
        std::vector<PointF> points;
        Size size;
    };

    class Landmarks {
    public:
        Landmarks() {};
        std::vector<PointF> points;
    };

    class Image : public Blob<uint8_t> {
    public:
        using self = Image;
        using supper = Blob<uint8_t>;
        using Datum = uint8_t;

        Image( const SeetaImageData &vimg ) : Image( vimg.data, vimg.width, vimg.height, vimg.channels ) {}
        operator SeetaImageData() const {
            SeetaImageData vimg = { width(), height(), channels(), const_cast<Datum *>( data() ) };
            return vimg;
        }
        Image( const ImageData &vimg ) : Image( vimg.data, vimg.width, vimg.height, vimg.channels ) {}
        operator ImageData() const {
            ImageData vimg( const_cast<Datum *>( data() ), width(), height(), channels() );
            return std::move( vimg );
        }

        Image( int width, int height, int channels )
            : supper( height, width, channels ) {
        }

        Image( const uint8_t *data, int width, int height, int channels )
            : supper( data, height, width, channels ) {
        }

        Image()
            : Image( 0, 0, 0 ) {
        }

        int height() const {
            return supper::shape( 1 );
        }

        int width() const {
            return supper::shape( 2 );
        }

        int channels() const {
            return supper::shape( 3 );
        }

        template <typename U>
        static Image FromBlob( const Blob<U> &blob ) {
            if( blob.shape( 0 ) != 1 ) throw  std::logic_error( "Can not convert multi images." );
            Image image( blob.shape( 2 ), blob.shape( 1 ), blob.shape( 3 ) );
            for( int i = 0; i < blob.count(); ++i ) {
                image[i] = static_cast<uint8_t>( std::max<U>( 0, std::min<U>( 255, blob[i] ) ) );
            }
            return std::move( image );
        }

    };

    inline void _out_str( std::ostream &out ) { }

    // there is no error anyway, this is a new feature about C++11
    template<typename T, typename... Args>
    inline void _out_str( std::ostream &out, const T &t, Args... args )
    {
        _out_str( out << t, args... );
    }

    template<typename... Args>
    inline const std::string str( Args... args )
    {
        std::ostringstream oss;
        _out_str( oss, args... );
        return std::move( oss.str() );
    }
}
