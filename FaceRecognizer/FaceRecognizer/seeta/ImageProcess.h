#pragma once

#include "DataHelper.h"
#include "CommonStruct.h"
#include "graphics2d.h"

namespace seeta
{
    enum SAMPLING_METHOD
    {
        BY_LINEAR,  ///< 线性采样
        BY_BICUBIC  ///< Cubic 采样
    };

    const Image color( const Image &img );
    const Image gray( const Image &img );
    const Image crop( const Image &img, const Rect &rect );
    using Padding = Size;
    const Image pad( const Image &img, const Padding &padding );
    const Image resize( const Image &img, const Size &size );
    const Image crop_resize( const Image &img, const Rect &rect, const Size &size );
    const Image equalize_hist( const Image &img );

    void fill( Image &img, const Point &point, const Image &patch );
    void fill( Image &img, const Rect &rect, const Image &patch );

    const Meanshape face_meanshape( int num, int id = 0 );
    const Meanshape resize( const Meanshape &shape, double scaler );
    const Meanshape resize( const Meanshape &shape, const Size &size );
    const Image crop_face( const Image &img, const Meanshape &shape, const Landmarks &marks, SAMPLING_METHOD type );
    const Image crop_face( const Image &img, const Meanshape &shape, const Landmarks &marks, SAMPLING_METHOD type, const Size &final_size );
    const Image crop_face( const Image &img, const Meanshape &shape, const Landmarks &marks, SAMPLING_METHOD type, const Size &final_size, Landmarks &final_points );

    // sample image with `size` on `image` by trasfromation
    const Image sample( const Image &image, const Size &size, const Trans2D<double> &transformation );
}
