#pragma once

#include "seeta/DataHelper.h"
#include <cmath>
#include "Struct.h"
#include <array>
#include <cassert>
#include <cfloat>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif


namespace seeta
{
    template <typename T, size_t N>
    class Buf {
    public:
        using self = Buf;
        const static size_t Size = N;

        T &operator[]( size_t i ) {
            return this->m_data[i];
        }

        const T &operator[]( size_t i ) const {
            return this->m_data[i];
        }

        T &data( size_t n ) {
            return this->m_data[n];
        }

        const T &data( size_t n ) const {
            return this->m_data[n];
        }

        T *data() {
            return this->m_data;
        }

        const T *data() const {
            return this->m_data;
        }

    private:
        T m_data[N];
    };

    template <typename T, size_t M, size_t N>
    class Mat : public Buf<T, M *N> {
    public:
        using self = Mat;
        using supper = Buf<T, M *N>;
        const static size_t Rows = M;
        const static size_t Cols = N;

        using supper::data;

        T &data( size_t m, size_t n ) {
            return this->data( m * N + n );
        }

        const T &data( size_t m, size_t n ) const {
            return this->data( m * N + n );
        }
    };

    /**
     * \brief means column vector
     */
    template <typename T, size_t N>
    class Vec : public Buf<T, N> {
    public:
        using self = Vec;
        using supper = Buf<T, N>;
    };

    template <typename T>
    class Trans2D : public Mat<T, 3, 3> {
    public:
        using self = Trans2D;
        using supper = Mat<T, 3, 3>;
        Trans2D() {}
        Trans2D(
            T r1c1, T r1c2, T r1c3,
            T r2c1, T r2c2, T r2c3,
            T r3c1, T r3c2, T r3c3 ) {
            this->data( 0 ) = r1c1;
            this->data( 1 ) = r1c2;
            this->data( 2 ) = r1c3;
            this->data( 3 ) = r2c1;
            this->data( 4 ) = r2c2;
            this->data( 5 ) = r2c3;
            this->data( 6 ) = r3c1;
            this->data( 7 ) = r3c2;
            this->data( 8 ) = r3c3;
        }
    };

    template <typename T>
    class Vec3D : public Vec<T, 3> {
    public:
        using self = Vec3D;
        using supper = Vec<T, 3>;

        T &x;
        T &y;
        T &z;

        Vec3D()
            : x( this->data( 0 ) )
            , y( this->data( 1 ) )
            , z( this->data( 2 ) ) {
        }
        Vec3D( T x, T y, T z )
            : Vec3D() {
            this->data( 0 ) = x;
            this->data( 1 ) = y;
            this->data( 2 ) = z;
        }
        Vec3D( T x, T y ) : self( x, y, T( 1 ) ) {}

    };

    template <typename T>
    class Vec2D : public Vec<T, 2> {
    public:
        using self = Vec2D;
        using supper = Vec<T, 2>;

        T &x;
        T &y;

        Vec2D()
            : x( this->data( 0 ) )
            , y( this->data( 1 ) ) {
        }
        Vec2D( T x, T y )
            : Vec2D() {
            this->data( 0 ) = x;
            this->data( 1 ) = y;
        }
    };


    /**
     * \brief get A dot B
     * \tparam T dtype
     * \param A left matrix
     * \param B right matrix
     * \return A dot B
     */
    template <typename T>
    Trans2D<T> dot( const Trans2D<T> &A, const Trans2D<T> &B )
    {
#define __INNER_PRODUCT_Am_Bn(m, n) (A.data(m, 0) * B.data(0, n) + A.data(m, 1) * B.data(1, n) + A.data(m, 2) * B.data(2, n))
        return Trans2D<T>(
                   __INNER_PRODUCT_Am_Bn( 0, 0 ),
                   __INNER_PRODUCT_Am_Bn( 0, 1 ),
                   __INNER_PRODUCT_Am_Bn( 0, 2 ),
                   __INNER_PRODUCT_Am_Bn( 1, 0 ),
                   __INNER_PRODUCT_Am_Bn( 1, 1 ),
                   __INNER_PRODUCT_Am_Bn( 1, 2 ),
                   __INNER_PRODUCT_Am_Bn( 2, 0 ),
                   __INNER_PRODUCT_Am_Bn( 2, 1 ),
                   __INNER_PRODUCT_Am_Bn( 2, 2 )
               );
#undef __INNER_PRODUCT_Am_Bn
    }

    /**
     * \brief A = B dot A
     * \tparam T dtype
     * \param A stacked matrix
     * \param B add new trasnformation
     */
    template <typename T>
    void stack( Trans2D<T> &A, const Trans2D<T> &B )
    {
        A = dot( B, A );
    }

    /**
     * \brief
     * \tparam T dtype
     * \param M transmation matrix
     * \param p 2D point in 3D-vector
     * \return M dot p
     */
    template <typename T>
    Vec3D<T> transform( const Trans2D<T> &M, const Vec3D<T> &p )
    {
#define __INNER_PRODUCT_Mn_p(n) (M.data(n, 0) * p.data(0) + M.data(n, 1) * p.data(1) + M.data(n, 2) * p.data(2))
        return Vec3D<T>(
                   __INNER_PRODUCT_Mn_p( 0 ),
                   __INNER_PRODUCT_Mn_p( 1 ),
                   __INNER_PRODUCT_Mn_p( 2 )
               );
#undef __INNER_PRODUCT_Mn_p
    }

    /**
     * \brief
     * \tparam T dtype
     * \param M transmation matrix
     * \param p 2D point in 3D-vector
     * \return M dot p
     */
    template <typename T>
    Vec2D<T> transform( const Trans2D<T> &M, const Vec2D<T> &p )
    {
        auto transed = transform<T>( M, Vec3D<T>( p.data( 0 ), p.data( 1 ), T( 1 ) ) );
        return Vec2D<T>( transed.data( 0 ), transed.data( 1 ) );
    }

    namespace affine
    {
        /**
         * \brief get inverse transformation matrix, perform p(x', y') = M dot p(x, y)
         * \tparam T dtype
         * \param M origin matrix
         * \return the inverse matrix
         */
        template <typename T>
        Trans2D<T> inverse( const Trans2D<T> &M )
        {
            assert( M[6] == 0 );
            assert( M[7] == 0 );
            assert( M[8] == 1 );
            T t3t1_t0t4 = M[3] * M[1] - M[0] * M[4];
            if( t3t1_t0t4 < FLT_EPSILON && t3t1_t0t4 > -FLT_EPSILON ) t3t1_t0t4 = FLT_EPSILON * 2;
            T recip_t3t1_t0t4 = 1 / t3t1_t0t4;
            return Trans2D<T>(
                       -M[4] * recip_t3t1_t0t4, M[1] * recip_t3t1_t0t4, -( M[1] * M[5] - M[4] * M[2] ) * recip_t3t1_t0t4,
                       M[3] * recip_t3t1_t0t4, -M[0] * recip_t3t1_t0t4, -( M[3] * M[2] - M[0] * M[5] ) * recip_t3t1_t0t4,
                       T( 0 ), T( 0 ), T( 1 )
                   );
        }

        /**
         * \brief get transformation matrix, perform p(x', y') = M dot p(x, y)
         * \tparam T dtype
         * \param degree degree
         * \return the rotatation matrix
         */
        template <typename T>
        Trans2D<T> rotate( T degree )
        {
            const static double pi_dev_180 = M_PI / 180.0;
            auto cos_degree = std::cos( degree * pi_dev_180 );
            auto sin_degree = std::sin( degree * pi_dev_180 );
            return Trans2D<T>(
                       T( cos_degree ), T( -sin_degree ), T( 0 ),
                       T( sin_degree ), T( cos_degree ), T( 0 ),
                       T( 0 ), T( 0 ), T( 1 )
                   );
        }

        /**
         * \brief get transformation matrix, perform p(x', y') = M dot p(x, y)
         * \tparam T dtype
         * \param tx shift on x-dim
         * \param ty shift on y-dim
         * \return the traslation matrix
         */
        template <typename T>
        Trans2D<T> translate( T tx, T ty )
        {
            return Trans2D<T>(
                       T( 1 ), T( 0 ), tx,
                       T( 0 ), T( 1 ), ty,
                       T( 0 ), T( 0 ), T( 1 )
                   );
        }

        /**
         * \brief get identity matrix, perform p(x, y) = M dot p(x, y)
         * \tparam T dtype
         * \return indentiy matrix
         */
        template <typename T>
        Trans2D<T> identity()
        {
            return Trans2D<T>(
                       T( 1 ), T( 0 ), T( 0 ),
                       T( 0 ), T( 1 ), T( 0 ),
                       T( 0 ), T( 0 ), T( 1 )
                   );
        }

        /**
         * \brief get transformation matrix, perform p(x', y') = M dot p(x, y)
         * \tparam T dtype
         * \param sx scale on x-dim
         * \param sy scale on y-dim
         * \return the scale matrix
         */
        template <typename T>
        Trans2D<T> scale( T sx, T sy )
        {
            return Trans2D<T>(
                       sx, T( 0 ), T( 0 ),
                       T( 0 ), sy, T( 0 ),
                       T( 0 ), T( 0 ), T( 1 )
                   );
        }

        /**
         * \brief get transformation matrix, perform p(x', y') = M dot p(x, y)
         * \tparam T dtype
         * \param s scale on x-dim and y-dim
         * \return the scale matrix
         */
        template <typename T>
        Trans2D<T> scale( T s )
        {
            return scale<T>( s, s );
        }

        /**
         * \brief get transformation matrix, perform p(x', y') = M dot p(x, y)
         * \tparam T dtype
         * \param shx shear on x-dim
         * \param shy shear on y-dim
         * \return the shear matrix
         */
        template <typename T>
        Trans2D<T> shear( T shx, T shy )
        {
            return Trans2D<T>(
                       T( 1 ), shx, T( 0 ),
                       shy, T( 1 ), T( 0 ),
                       T( 0 ), T( 0 ), T( 1 )
                   );
        }

        /**
         * \brief get transformation matrix, perform p(x', y') = M dot p(x, y)
         * \tparam T dtype
         * \param sh shear on x-dim
         * \return the shear matrix
         */
        template <typename T>
        Trans2D<T> shear_x( T sh )
        {
            return shear<T>( sh, 0 );
        }

        /**
         * \brief get transformation matrix, perform p(x', y') = M dot p(x, y)
         * \tparam T dtype
         * \param sh shear on y-dim
         * \return the shear matrix
         */
        template <typename T>
        Trans2D<T> shear_y( T sh )
        {
            return shear<T>( 0, sh );
        }

        /**
         * \brief get transformation matrix, perform p(x', y') = M dot p(x, y)
         * \tparam T dtype
         * \param fx if reflect on x-dim
         * \param fy if reflect on y-dim
         * \return the reflection matrix
         */
        template <typename T>
        Trans2D<T> reflect( bool fx, bool fy )
        {
            return Trans2D<T>(
                       fx ? T( -1 ) : T( 1 ), T( 0 ), T( 0 ),
                       T( 0 ), fy ? T( -1 ) : T( 1 ), T( 0 ),
                       T( 0 ), T( 0 ), T( 1 )
                   );
        }

        /**
         * \brief get transformation matrix, perform p(x', y') = M dot p(x, y)
         * \tparam T dtype
         * \return the reflection matrix aouble origin
         */
        template <typename T>
        Trans2D<T> reflect_about_origin()
        {
            return Trans2D<T>(
                       T( -1 ), T( 0 ), T( 0 ),
                       T( 0 ), T( -1 ), T( 0 ),
                       T( 0 ), T( 0 ), T( 1 )
                   );
        }

        /**
        * \brief get transformation matrix, perform p(x', y') = M dot p(x, y)
        * \tparam T dtype
        * \return the reflection matrix about x-axis
        */
        template <typename T>
        Trans2D<T> reflect_about_x_axis( bool fx, bool fy )
        {
            return Trans2D<T>(
                       T( 1 ), T( 0 ), T( 0 ),
                       T( 0 ), T( -1 ), T( 0 ),
                       T( 0 ), T( 0 ), T( 1 )
                   );
        }

        /**
        * \brief get transformation matrix, perform p(x', y') = M dot p(x, y)
        * \tparam T dtype
        * \return the reflection matrix about y-axis
        */
        template <typename T>
        Trans2D<T> reflect_about_y_axis( bool fx, bool fy )
        {
            return Trans2D<T>(
                       T( -1 ), T( 0 ), T( 0 ),
                       T( 0 ), T( 1 ), T( 0 ),
                       T( 0 ), T( 0 ), T( 1 )
                   );
        }
    }

}

