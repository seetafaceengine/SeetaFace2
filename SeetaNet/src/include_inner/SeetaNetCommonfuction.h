#ifndef _SEETANET_COMMONFUNCTION_H_
#define _SEETANET_COMMONFUNCTION_H_
#include <memory>

#include "SeetaNetMacro.h"
#include <math.h>

#include "SeetaNetMathCPU.h"

extern "C" {

}


#include <cmath>

template <typename Dtype>
void seeta_cpu_gemm( seeta::blas::Transpose TransA,
                     seeta::blas::Transpose TransB, const int M, const int N, const int K,
                     const Dtype alpha, const Dtype *A, const Dtype *B, const Dtype beta,
                     Dtype *C )
{
    return;
}

template<>

//inline void seeta_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
inline void seeta_cpu_gemm<float>( seeta::blas::Transpose TransA,
                                   seeta::blas::Transpose TransB, const int M, const int N, const int K,
                                   const float alpha, const float *A, const float *B, const float beta,
                                   float *C )
{

    int lda = ( TransA == seeta::blas::NoTrans ) ? K : M;
    int ldb = ( TransB == seeta::blas::NoTrans ) ? N : K;


    if (seeta::near(alpha, 1.f) && seeta::near(beta, 0.f)) {
        seeta::math<float>::gemm_pack(TransA, TransB, M, N, K, alpha, A, B, beta, C);
    }  
    else {
        seeta::math<float>::gemm(seeta::blas::RowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
            ldb, beta, C, N);
    }  
}

template<>
inline void seeta_cpu_gemm<double>( seeta::blas::Transpose TransA,
                                    seeta::blas::Transpose TransB, const int M, const int N, const int K,
                                    const double alpha, const double *A, const double *B, const double beta,
                                    double *C )
{
    int lda = ( TransA == seeta::blas::NoTrans ) ? K : M;
    int ldb = ( TransB == seeta::blas::NoTrans ) ? N : K;

    seeta::math<double>::gemm( seeta::blas::RowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
                               ldb, beta, C, N );

}

template <typename Dtype>
void seeta_powx( const int n, const Dtype *a, const Dtype b, Dtype *y )
{
    return;
}

// A simple way to define the vsl unary functions with singular parameter b.
// The operation should be in the form e.g. y[i] = pow(a[i], b)
#define DEFINE_VSL_UNARY_FUNC_WITH_PARAM(name, operation) \
    template<typename Dtype> \
    void v##name(const int n, const Dtype* a, const Dtype b, Dtype* y) { \
        \
        for (int i = 0; i < n; ++i) { operation; } \
    } \
    inline void vs##name( \
                          const int n, const float* a, const float b, float* y) { \
        v##name<float>(n, a, b, y); \
    } \
    inline void vd##name( \
                          const int n, const double* a, const float b, double* y) { \
        v##name<double>(n, a, b, y); \
    }

DEFINE_VSL_UNARY_FUNC_WITH_PARAM( Powx, y[i] = std::pow( a[i], b ) );

#define DEFINE_VSL_UNARY_FUNC(name, operation) \
    template<typename Dtype> \
    void v##name(const int n, const Dtype* a, Dtype* y) { \
        \
        for (int i = 0; i < n; ++i) { operation; } \
    } \
    inline void vs##name( \
                          const int n, const float* a, float* y) { \
        v##name<float>(n, a, y); \
    } \
    inline void vd##name( \
                          const int n, const double* a, double* y) { \
        v##name<double>(n, a, y); \
    }
DEFINE_VSL_UNARY_FUNC( Sqr, y[i] = a[i] * a[i] );
DEFINE_VSL_UNARY_FUNC( Exp, y[i] = exp( a[i] ) );
DEFINE_VSL_UNARY_FUNC( Ln, y[i] = log( a[i] ) );
DEFINE_VSL_UNARY_FUNC( Abs, y[i] = fabs( a[i] ) );
template <>
inline void seeta_powx<float>( const int n, const float *a, const float b,
                               float *y )
{
    vsPowx( n, a, b, y );
}

template <>
inline void seeta_powx<double>( const int n, const double *a, const double b,
                                double *y )
{
    vdPowx( n, a, b, y );
}


#define DEFINE_VSL_BINARY_FUNC(name, operation) \
    template<typename Dtype> \
    void v##name(const int n, const Dtype* a, const Dtype* b, Dtype* y) { \
        \
        for (int i = 0; i < n; ++i) { operation; } \
    } \
    inline void vs##name( \
                          const int n, const float* a, const float* b, float* y) { \
        v##name<float>(n, a, b, y); \
    } \
    inline void vd##name( \
                          const int n, const double* a, const double* b, double* y) { \
        v##name<double>(n, a, b, y); \
    }

DEFINE_VSL_BINARY_FUNC( Mul, y[i] = a[i] * b[i] );
DEFINE_VSL_BINARY_FUNC( Div, y[i] = a[i] / b[i] );


template <typename Dtype>
void seeta_mul( const int N, const Dtype *a, const Dtype *b, Dtype *y )
{
    return;
}

template <>
inline void seeta_mul<float>( const int n, const float *a, const float *b,
                              float *y )
{
    vsMul( n, a, b, y );
}

template <>
inline void seeta_mul<double>( const int n, const double *a, const double *b,
                               double *y )
{
    vdMul( n, a, b, y );
}

template <typename Dtype>
void seeta_div( const int N, const Dtype *a, const Dtype *b, Dtype *y )
{
    return;
}

template <>
inline void seeta_div<float>( const int n, const float *a, const float *b,
                              float *y )
{
    vsDiv( n, a, b, y );
}

template <>
inline void seeta_div<double>( const int n, const double *a, const double *b,
                               double *y )
{
    vdDiv( n, a, b, y );
}

template <typename Dtype>
void seeta_set( const int N, const Dtype alpha, Dtype *Y )
{
    if( alpha == 0 )
    {
        memset( Y, 0, sizeof( Dtype ) * N );
        return;
    }
    for( int i = 0; i < N; ++i )
    {
        Y[i] = alpha;
    }
}

template void seeta_set<int>( const int N, const int alpha, int *Y );
template void seeta_set<float>( const int N, const float alpha, float *Y );
template void seeta_set<double>( const int N, const double alpha, double *Y );


template <typename Dtype>
void seeta_sqr( const int N, const Dtype *a, Dtype *y );

template <>
inline void seeta_sqr<float>( const int n, const float *a, float *y )
{
    vsSqr( n, a, y );
}

template <>
inline void seeta_sqr<double>( const int n, const double *a, double *y )
{
    vdSqr( n, a, y );
}

template <typename Dtype>
void seeta_copy( const int N, const Dtype *X, Dtype *Y );

template <typename Dtype>
void seeta_copy( const int N, const Dtype *X, Dtype *Y )
{
    memcpy( Y, X, sizeof( Dtype ) * N );
}

template <typename Dtype>
void seeta_axpy( const int N, const Dtype alpha, const Dtype *X,
                 Dtype *Y );

template <>
inline void seeta_axpy<float>( const int N, const float alpha, const float *X,
                               float *Y )
{
    for( int i = 0; i < N; i++ )
    {
        Y[i] = alpha * X[i] + Y[i];
    }
}

template <>
inline void seeta_axpy<double>( const int N, const double alpha, const double *X,
                                double *Y )
{
    for( int i = 0; i < N; i++ )
    {
        Y[i] = alpha * X[i] + Y[i];
    }
}

template <typename Dtype>
void seeta_exp( const int n, const Dtype *a, Dtype *y );

template <>
inline void seeta_exp<float>( const int n, const float *a, float *y )
{
    vsExp( n, a, y );
}

template <>
inline void seeta_exp<double>( const int n, const double *a, double *y )
{
    vdExp( n, a, y );
}


#endif
