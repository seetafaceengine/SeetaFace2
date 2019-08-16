#ifndef _SEETANET_MATH_CPU_H
#define _SEETANET_MATH_CPU_H

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <math.h>
#include <cstdint>

namespace seeta
{
    namespace blas
    {
        enum Order
        {
            RowMajor = 101,
            ColMajor = 102
        };
        enum Transpose
        {
            NoTrans = 111,
            Trans = 112
        };
    }


    template <typename T>
    inline bool near( T value1, T value2 )
    {
        return ( value1 - value2 == 0 );
    }

    template<>
    inline bool near<double>( double value1, double value2 )
    {
        return ( value1 > value2 ? value1 - value2 : value2 - value1 ) < DBL_EPSILON;
    }

    template<>
    inline bool near<float>( float value1, float value2 )
    {
        return ( value1 > value2 ? value1 - value2 : value2 - value1 ) < FLT_EPSILON;
    }


    template <typename T>
    inline T abs( T value )
    {
        return T( std::abs( value ) );
    }

    template <>
    inline uint8_t abs( uint8_t value )
    {
        return value;
    }

    template <>
    inline uint16_t abs( uint16_t value )
    {
        return value;
    }

    template <>
    inline uint32_t abs( uint32_t value )
    {
        return value;
    }

    template <>
    inline uint64_t abs( uint64_t value )
    {
        return value;
    }

    template <>
    inline float abs( float value )
    {
        using namespace std;
        return fabsf( value );
    }

    template <>
    inline double abs( double value )
    {
        return std::fabs( value );
    }


    template <typename T>
    class math {
    public:

        static T abs( T val );

        static T dot(
            int N,
            const T *x,
            int incx,
            const T *y,
            int incy
        );
        static T dot( int N, const T *x, const T *y );

        static void gemm(
            blas::Order Order,
            blas::Transpose TransA,
            blas::Transpose TransB,
            int M, int N, int K,
            T alpha,
            const T *A, int lda,
            const T *B, int ldb,
            T beta,
            T *C, int ldc );

        static void gemm(
            blas::Transpose TransA,
            blas::Transpose TransB,
            int M, int N, int K,
            T alpha, const T *A, const T *B,
            T beta, T *C );

        static void gemm_pack(
            blas::Transpose TransA,
            blas::Transpose TransB,
            int M, int N, int K,
            T alpha, const T *A, const T *B,
            T beta, T *C);

        static T asum(
            int N,
            const T *x,
            int incx
        );
    };
}

extern template class seeta::math<float>;
extern template class seeta::math<double>;


#endif

