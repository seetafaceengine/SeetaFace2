#include "SeetaNetMathCPU.h"
#include "orz/sync/shotgun.h"
#include "orz/tools/ctxmgr_lite.h"
#include "orz/tools/box.h"

#include <assert.h>
#include <iostream>
#include <cmath>

#include <SeetaNetSimd.h>
#include "orz/tools/ctxmgr_lite.h"
#include "orz/mem/vat.h"

#ifdef SEETA_USE_SSE
    #include <immintrin.h>
#endif

#ifdef SEETA_USE_NEON
	#include <arm_neon.h>
#endif
namespace seeta
{

    template<typename T>
    inline T inline_dot( int N, const T *x, int incx, const T *y, int incy )
    {
        T sum = 0;
        // block: 4
        int i = 0;
        static const int block_size = 4;
        int blocked_N = N % block_size ? N - block_size : N;
        for( ; i < blocked_N; i += block_size )
        {
            sum += *x * *y;
            x += incx;
            y += incy;
            sum += *x * *y;
            x += incx;
            y += incy;
            sum += *x * *y;
            x += incx;
            y += incy;
            sum += *x * *y;
            x += incx;
            y += incy;
        }
        for( ; i < N; ++i )
        {
            sum += *x * *y;
            x += incx;
            y += incy;
        }
        return sum;
    }
    #ifdef SEETA_USE_SSE

    inline float inline_dot_conitnous_float( int N, const float *x, const float *y )
    {
        float sum = 0;
        // block: 4
        int i = 0;
        static const int block_size = 4;
        int blocked_N = N % block_size ? N - block_size : N;
        __m128 simdX, simdY;
        __m128 simdSUM = _mm_setzero_ps();
        float simdBuffer[4];
        for( ; i < blocked_N; i += block_size )
        {
            simdX = _mm_loadu_ps( x );
            simdY = _mm_loadu_ps( y );
            x += 4;
            y += 4;
            simdSUM = _mm_add_ps( simdSUM, _mm_mul_ps( simdX, simdY ) );
        }
        _mm_storeu_ps( simdBuffer, simdSUM );
        sum = simdBuffer[0] + simdBuffer[1] + simdBuffer[2] + simdBuffer[3];
        for( ; i < N; ++i )
        {
            sum += *x * *y;
            x += 1;
            y += 1;
        }
        return sum;
    }

    inline float inline_dot_non_conitnous_float( int N, const float *x, int incx, const float *y, int incy )
    {
        float sum = 0;
        // block: 4
        int i = 0;
        static const int block_size = 4;
        int blocked_N = N % block_size ? N - block_size : N;
        __m128 simdX, simdY;
        __m128 simdSUM = _mm_setzero_ps();
        float simdBuffer[4];
        for( ; i < blocked_N; i += block_size )
        {
            simdBuffer[0] = *x;
            x += incx;
            simdBuffer[1] = *x;
            x += incx;
            simdBuffer[2] = *x;
            x += incx;
            simdBuffer[3] = *x;
            x += incx;
            simdX = _mm_loadu_ps( simdBuffer );
            simdBuffer[0] = *y;
            y += incy;
            simdBuffer[1] = *y;
            y += incy;
            simdBuffer[2] = *y;
            y += incy;
            simdBuffer[3] = *y;
            y += incy;
            simdY = _mm_loadu_ps( simdBuffer );
            simdSUM = _mm_add_ps( simdSUM, _mm_mul_ps( simdX, simdY ) );
        }
        _mm_storeu_ps( simdBuffer, simdSUM );
        sum = simdBuffer[0] + simdBuffer[1] + simdBuffer[2] + simdBuffer[3];
        for( ; i < N; ++i )
        {
            sum += *x * *y;
            x += incx;
            y += incy;
        }
        return sum;
    }
    template <>
    inline float inline_dot<float>( int N, const float *x, int incx, const float *y, int incy )
    {
        if( incx == 1 && incy == 1 ) return inline_dot_conitnous_float( N, x, y );
        return inline_dot_non_conitnous_float( N, x, incx, y, incy );
    }
    #endif
	
	
    #ifdef SEETA_USE_NEON

    inline float inline_dot_conitnous_float( int N, const float *x, const float *y )
    {
        float sum = 0;
        // block: 4
        int i = 0;
        static const int block_size = 4;
        int blocked_N = N % block_size ? N - block_size : N;
        float32x4_t simdX, simdY;
        float32x4_t simdSUM = vdupq_n_f32(0.0f);
        float simdBuffer[4];
        for( ; i < blocked_N; i += block_size )
        {
            simdX = vld1q_f32( x );
            simdY = vld1q_f32( y );
            x += 4;
            y += 4;
            simdSUM = vaddq_f32( simdSUM, vmulq_f32( simdX, simdY ) );
        }
        vst1q_f32( simdBuffer, simdSUM );
        sum = simdBuffer[0] + simdBuffer[1] + simdBuffer[2] + simdBuffer[3];
        for( ; i < N; ++i )
        {
            sum += *x * *y;
            x += 1;
            y += 1;
        }
        return sum;
    }

    inline float inline_dot_non_conitnous_float( int N, const float *x, int incx, const float *y, int incy )
    {
        float sum = 0;
        // block: 4
        int i = 0;
        static const int block_size = 4;
        int blocked_N = N % block_size ? N - block_size : N;
        float32x4_t simdX, simdY;
        float32x4_t simdSUM = vdupq_n_f32(0.0f);
        float simdBuffer[4];
        for( ; i < blocked_N; i += block_size )
        {
            simdBuffer[0] = *x;
            x += incx;
            simdBuffer[1] = *x;
            x += incx;
            simdBuffer[2] = *x;
            x += incx;
            simdBuffer[3] = *x;
            x += incx;
            simdX = vld1q_f32( simdBuffer );
            simdBuffer[0] = *y;
            y += incy;
            simdBuffer[1] = *y;
            y += incy;
            simdBuffer[2] = *y;
            y += incy;
            simdBuffer[3] = *y;
            y += incy;
            simdY = vld1q_f32( simdBuffer );
            simdSUM = vaddq_f32( simdSUM, vmulq_f32( simdX, simdY ) );
        }
        vst1q_f32( simdBuffer, simdSUM );
        sum = simdBuffer[0] + simdBuffer[1] + simdBuffer[2] + simdBuffer[3];
        for( ; i < N; ++i )
        {
            sum += *x * *y;
            x += incx;
            y += incy;
        }
        return sum;
    }
    template <>
    inline float inline_dot<float>( int N, const float *x, int incx, const float *y, int incy )
    {
        if( incx == 1 && incy == 1 ) return inline_dot_conitnous_float( N, x, y );
        return inline_dot_non_conitnous_float( N, x, incx, y, incy );
    }
    #endif
	
    template<typename T>
    inline void inline_zero( int N, T *x, int incx )
    {
        // use thread
        auto gun = orz::ctx::lite::ptr<orz::Shotgun>();
        if( ( gun != nullptr ) && ( gun->size() > 1 ) )
        {
            auto bins = orz::split_bins( 0, N, int( gun->size() ) );
            for( auto &range : bins )
            {
                gun->fire( [ &, range]( int )
                {
                    T *local_x = x + range.first * incx;
                    auto local_i = range.first;
                    auto local_N = range.second;
                    for( ; local_i < local_N; ++local_i )
                    {
                        *local_x = 0;
                        local_x += incx;
                    }
                } );
            }
            gun->join();
            return;
        }
        // block: 4
        int i = 0;
        static const int block_size = 4;
        int blocked_N = N % block_size ? N - block_size : N;
        for( ; i < blocked_N; i += block_size )
        {
            *x = 0;
            x += incx;
            *x = 0;
            x += incx;
            *x = 0;
            x += incx;
            *x = 0;
            x += incx;
        }
        for( ; i < N; ++i )
        {
            *x = 0;
            x += incx;
        }
    }

    template<typename T>
    inline void inline_scal( int N, T alpha, T *x, int incx )
    {
        if( seeta::near( alpha, T( 1 ) ) ) return; // TODO: update float number equal check method
        if( seeta::near( alpha, T( 0 ) ) )
        {
            inline_zero<T>( N, x, incx );
            return;
        }
        // use thread
        auto gun = orz::ctx::lite::ptr<orz::Shotgun>();
        if( ( gun != nullptr ) && ( gun->size() > 1 ) )
        {
            auto bins = orz::split_bins( 0, N, int( gun->size() ) );
            for( auto &range : bins )
            {
                gun->fire( [ &, range]( int )
                {
                    T *local_x = x + range.first * incx;
                    auto local_i = range.first;
                    auto local_N = range.second;
                    for( ; local_i < local_N; ++local_i )
                    {
                        *local_x *= alpha;
                        local_x += incx;
                    }
                } );
            }
            gun->join();
            return;
        }

        // block: 4
        int i = 0;
        static const int block_size = 4;
        int blocked_N = N % block_size ? N - block_size : N;
        for( ; i < blocked_N; i += block_size )
        {
            *x *= alpha;
            x += incx;
            *x *= alpha;
            x += incx;
            *x *= alpha;
            x += incx;
            *x *= alpha;
            x += incx;
        }
        for( ; i < N; ++i )
        {
            *x *= alpha;
            x += incx;
        }
    }


    template<typename T>
    T math<T>::dot( int N, const T *x, int incx, const T *y, int incy )
    {
        auto gun = orz::ctx::lite::ptr<orz::Shotgun>();
        if( gun == nullptr || gun->size() <= 1 )
        {
            return inline_dot<T>( N, x, incx, y, incy );
        }
        auto bins = orz::split_bins( 0, N, int( gun->size() ) );
        std::vector<T> threads_sum( gun->size(), 0 );
        for( auto &range : bins )
        {
            gun->fire( [ &, range]( int id )
            {
                const T *local_x = x + range.first * incx;
                const T *local_y = y + range.first * incy;
                threads_sum[id] = inline_dot<T>( range.second - range.first, local_x, incx, local_y, incy );
            } );
        }
        gun->join();
        T sum = 0;
        for( auto v : threads_sum ) sum += v;
        return sum;
    }

    template<typename T>
    inline void inline_gemm_row_major(
        blas::Transpose TransA,
        blas::Transpose TransB,
        int M, int N, int K,
        T alpha,
        const T *A, int lda,
        const T *B, int ldb,
        T beta,
        T *C, int ldc )
    {
        // TODO: check if lda, ldb, ldc use correct
        if( lda < ( TransA == blas::NoTrans ? K : M ) )
        {
            std::cout << "lda:" << lda  << " < "  << ( ( TransA == blas::NoTrans ) ? K : M ) << std::endl;
            throw std::logic_error("inline_gemm_row_major failed!");
        }

        if( ldb < ( TransB == blas::NoTrans ? N : K ) )
        {
            std::cout << "ldb:" << ldb << " < " << ( ( TransB == blas::NoTrans ) ? N : K ) << std::endl;
            throw std::logic_error("inline_gemm_row_major failed!");
        }
        if( ldc < N )
        {
            std::cout << "ldc:" << ldc << " < " << N << std::endl;
            throw std::logic_error("inline_gemm_row_major failed!");
        }

        auto gun = orz::ctx::lite::ptr<orz::Shotgun>();
        // calculate beta * C
        // C is RowMajor
        if( ldc == N )
        {
            inline_scal( M * N, beta, C, 1 );
        }
        else
        {
            if( gun != nullptr )
            {
                auto bins = orz::split_bins( 0, M, int( gun->size() ) );
                for( auto &range : bins )
                {
                    gun->fire( [ &, range]( int )
                    {
                        T *local_C = C + range.first * ldc;
                        auto local_i = range.first;
                        auto local_M = range.second;
                        for( ; local_i < local_M; ++local_i, local_C += ldc )
                        {
                            inline_scal( N, beta, local_C, 1 );
                        }
                    } );
                }
                gun->join();
            }
            else
            {
                T *C_anchor = C;
                for( int i = 0; i < M; ++i, C_anchor += ldc ) inline_scal( N, beta, C_anchor, 1 );
            }
        }

        if( seeta::near( alpha, T( 0 ) ) ) return;

        unsigned int condition = ( TransA == blas::NoTrans ? 0U : 1U ) | ( ( TransB == blas::NoTrans ? 0U : 2U ) );
        switch( condition )
        {
            case 0: // A: NoTrans, B: NoTrans
                if( gun != nullptr )
                {
                    auto bins = orz::split_bins( 0, M, int( gun->size() ) );
                    for( auto &range : bins )
                    {
                        gun->fire( [ &, range]( int )
                        {
                            for( int i = range.first; i < range.second; ++i )
                            {
                                T *C_anchor = &C[i * ldc];
                                for( int j = 0; j < N; ++j )
                                {
                                    *C_anchor += alpha * inline_dot( K, &A[i * lda], 1, &B[j], ldb );
                                    C_anchor++;
                                }
                            }
                        } );
                    }
                    gun->join();
                }
                else
                {
                    for( int i = 0; i < M; ++i )
                    {
                        T *C_anchor = &C[i * ldc];
                        for( int j = 0; j < N; ++j )
                        {
                            *C_anchor += alpha * inline_dot( K, &A[i * lda], 1, &B[j], ldb );
                            C_anchor++;
                        }
                    }
                }
                break;
            case 1: // A: Trans, B: NoTrans
                if( gun != nullptr )
                {
                    auto bins = orz::split_bins( 0, M, int( gun->size() ) );
                    for( auto &range : bins )
                    {
                        gun->fire( [ &, range]( int )
                        {
                            for( int i = range.first; i < range.second; ++i )
                            {
                                T *C_anchor = &C[i * ldc];
                                for( int j = 0; j < N; ++j )
                                {
                                    *C_anchor += alpha * inline_dot( K, &A[i], lda, &B[j], ldb );
                                    C_anchor++;
                                }
                            }
                        } );
                    }
                    gun->join();
                }
                else
                {
                    for( int i = 0; i < M; ++i )
                    {
                        T *C_anchor = &C[i * ldc];
                        for( int j = 0; j < N; ++j )
                        {
                            *C_anchor += alpha * inline_dot( K, &A[i], lda, &B[j], ldb );
                            C_anchor++;
                        }
                    }
                }
                break;
            case 2: // A: NoTrans, B: Trans
                if( gun != nullptr )
                {
                    auto bins = orz::split_bins( 0, M, int( gun->size() ) );
                    for( auto &range : bins )
                    {
                        gun->fire( [ &, range]( int )
                        {
                            for( int i = range.first; i < range.second; ++i )
                            {
                                T *C_anchor = &C[i * ldc];
                                for( int j = 0; j < N; ++j )
                                {
                                    *C_anchor += alpha * inline_dot( K, &A[i * lda], 1, &B[j * ldb], 1 );
                                    C_anchor++;
                                }
                            }
                        } );
                    }
                    gun->join();
                }
                else
                {
                    for( int i = 0; i < M; ++i )
                    {
                        T *C_anchor = &C[i * ldc];
                        for( int j = 0; j < N; ++j )
                        {
                            *C_anchor += alpha * inline_dot( K, &A[i * lda], 1, &B[j * ldb], 1 );
                            C_anchor++;
                        }
                    }
                }
                break;
            default: // A: Trans, B: Trans
                if( gun != nullptr )
                {
                    auto bins = orz::split_bins( 0, M, int( gun->size() ) );
                    for( auto &range : bins )
                    {
                        gun->fire( [ &, range]( int )
                        {
                            for( int i = range.first; i < range.second; ++i )
                            {
                                T *C_anchor = &C[i * ldc];
                                for( int j = 0; j < N; ++j )
                                {
                                    *C_anchor += alpha * inline_dot( K, &A[i], lda, &B[j * ldb], 1 );
                                    C_anchor++;
                                }
                            }
                        } );
                    }
                    gun->join();
                }
                else
                {
                    for( int i = 0; i < M; ++i )
                    {
                        T *C_anchor = &C[i * ldc];
                        for( int j = 0; j < N; ++j )
                        {
                            *C_anchor += alpha * inline_dot( K, &A[i], lda, &B[j * ldb], 1 );
                            C_anchor++;
                        }
                    }
                }
                break;
        }
    }

    // TODO: it has deviation in some case, when N, M, K is large
    template<typename T>
    void
    math<T>::gemm(
        blas::Order Order,
        blas::Transpose TransA,
        blas::Transpose TransB,
        int M, int N, int K,
        T alpha,
        const T *A, int lda,
        const T *B, int ldb,
        T beta,
        T *C, int ldc )
    {
        if( Order == blas::ColMajor )
        {
            inline_gemm_row_major<T>( TransB, TransA, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc );
        }
        else
        {
            inline_gemm_row_major<T>( TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc );
        }
    }

    template<typename T>
    T math<T>::dot( int N, const T *x, const T *y )
    {
        return dot( N, x, 1, y, 1 );
    }

    template<typename T>
    void math<T>::gemm( blas::Transpose TransA, blas::Transpose TransB, int M, int N, int K, T alpha, const T *A,
                        const T *B, T beta, T *C )
    {
        int lda = ( TransA == blas::NoTrans ? K : M );
        int ldb = ( TransB == blas::NoTrans ? N : K );
        int ldc = N;
        inline_gemm_row_major<T>( TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc );
    }

    template<typename T>
    static void matrix_transpose(const T* A, T* B, int m, int n) {
        int i, j;
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                B[i*m + j] = A[j*n + i];
            }
        }
    }

    template<typename T>
    static void pack_A(
        int row, int col,
        const T *from,
        int lda, T *to) {
        int out_loop = row >> 3;
        int remain = out_loop << 3;

        //T* to_at = to;
        for (int nn = 0; nn < out_loop; nn++) {
            int n = nn * 8;
            const T* k0 = from + n * lda;
            const T* k1 = k0 + lda;
            const T* k2 = k1 + lda;
            const T* k3 = k2 + lda;
            const T* k4 = k3 + lda;
            const T* k5 = k4 + lda;
            const T* k6 = k5 + lda;
            const T* k7 = k6 + lda;

            T* to_at = to + n * col;

            for (int i = 0; i < col; i++) {
                *to_at++ = *k0++;
                *to_at++ = *k1++;
                *to_at++ = *k2++;
                *to_at++ = *k3++;
                *to_at++ = *k4++;
                *to_at++ = *k5++;
                *to_at++ = *k6++;
                *to_at++ = *k7++;
            }
        }

        //NOTE:Maybe i should pack 4x4 on remain size
        //to_at = to + remain * col;

        for (int n = remain; n < row; n++) {
            const T* k0 = from + n * lda;
            T* to_at = to + n * col;
            for (int i = 0; i < col; i++) {
                *to_at++ = *k0++;
            }
        }
    }

    template<typename T>
    static void pack_B(
        int row, int col,
        const T *from,
        int ldb, T *to) {
        int out_loop = col >> 3;
        int remain = out_loop << 3;

        //T* to_at = to;
        for (int nn = 0; nn < out_loop; nn++) {
            int n = nn * 8;
            const T* from_at = from + n;
            T* to_at = to + n * row;

            for (int i = 0; i < row; i++) {
                *to_at++ = from_at[0];
                *to_at++ = from_at[1];
                *to_at++ = from_at[2];
                *to_at++ = from_at[3];
                *to_at++ = from_at[4];
                *to_at++ = from_at[5];
                *to_at++ = from_at[6];
                *to_at++ = from_at[7];

                from_at += ldb;
            }
        }

        //to_at = to + remain * row;
        for (int n = remain; n < col; n++) {
            const T* from_at = from + n;
            T* to_at = to + n * row;

            for (int i = 0; i < row; i++) {
                *to_at++ = from_at[0];
                from_at += ldb;
            }
        }
    }

    template<>
    void pack_B<float>(
        int row, int col,
        const float *from,
        int ldb, float *to) {
        int out_loop = col >> 3;
        int remain = out_loop << 3;

        //float* to_at = to;
        for (int nn = 0; nn < out_loop; nn++) {
            int n = nn * 8;
            const float* from_at = from + n;
            float* to_at = to + n * row;

            for (int i = 0; i < row; i++) {
                float32x4x2 from_at_x4x2(from_at);
                from_at_x4x2.store(to_at);

                from_at += ldb;
                to_at += 8;
            }
        }

        //to_at = to + remain * row;
        for (int n = remain; n < col; n++) {
            const float* from_at = from + n;
            float* to_at = to + n * row;

            for (int i = 0; i < row; i++) {
                *to_at++ = from_at[0];
                from_at += ldb;
            }
        }
    }

    template<typename T>
    inline static void kernel_8x8(int M, int K, int N, T alpha, const T *A, const T *B, T beta, T *C, int ldc) {

    }

    template<>
    inline void kernel_8x8<float>(int M, int K, int N, float alpha, const float *A, const float *B, float beta, float *C, int ldc) {
        const float* p_A = A;
        const float* p_B = B;
        float* p_C = C;

        int out_loop = M >> 3;
        int remain = out_loop << 3;
        float* output_at = p_C;

        for (int mm = 0; mm < out_loop; mm++) {
            int m = mm * 8;
            float* output_row0 = output_at + m * ldc;
            float* output_row1 = output_row0 + ldc;
            float* output_row2 = output_row1 + ldc;
            float* output_row3 = output_row2 + ldc;
            float* output_row4 = output_row3 + ldc;
            float* output_row5 = output_row4 + ldc;
            float* output_row6 = output_row5 + ldc;
            float* output_row7 = output_row6 + ldc;

            const float* A_store = p_A + m * K;

            int n_loop = N >> 3;
            int n_remain = n_loop << 3;
            for (int nn = 0; nn < n_loop; nn++)
            {
                int n = nn * 8;

                const float* A_at = A_store;
                const float* B_at = p_B + n * K;

                float32x4x2 c0(0.f), c1(0.f), c2(0.f), c3(0.f);
                float32x4x2 c4(0.f), c5(0.f), c6(0.f), c7(0.f);

                int k_loop = K >> 2;
                int k_remain = k_loop << 2;
                for (int kk = 0; kk < k_loop; kk++) {
                    //=====================pack_gemm k==0=====================
                    float32x4x2 k0 = broadcast2float32x4x2(A_at);       //[k00,k00,k00,k00,k00,k00,k00,k00]
                    float32x4x2 k1 = broadcast2float32x4x2(A_at + 1);   //[k10,k10,k10,k10,k10,k10,k10,k10]
                    float32x4x2 k2 = broadcast2float32x4x2(A_at + 2);   //[k20,k20,k20,k20,k20,k20,k20,k20]
                    float32x4x2 k3 = broadcast2float32x4x2(A_at + 3);   //[k30,k30,k30,k30,k30,k30,k30,k30]

                    float32x4x2 a0(B_at);                                  //[a00,a01,a02,a03,a04,a05,a06,a07]

                    c0 = fmadd(a0, k0, c0);
                    c1 = fmadd(a0, k1, c1);
                    c2 = fmadd(a0, k2, c2);
                    c3 = fmadd(a0, k3, c3);
                    //Note:The number of registers is limited
                    k0 = broadcast2float32x4x2(A_at + 4);               //[k40,k40,k40,k40,k40,k40,k40,k40]
                    k1 = broadcast2float32x4x2(A_at + 5);               //[k50,k50,k50,k50,k50,k50,k50,k50]
                    k2 = broadcast2float32x4x2(A_at + 6);               //[k60,k60,k60,k60,k60,k60,k60,k60]
                    k3 = broadcast2float32x4x2(A_at + 7);               //[k70,k70,k70,k70,k70,k70,k70,k70]

                    c4 = fmadd(a0, k0, c4);
                    c5 = fmadd(a0, k1, c5);
                    c6 = fmadd(a0, k2, c6);
                    c7 = fmadd(a0, k3, c7);

                    //=====================pack_gemm k==1=====================
                    k0 = broadcast2float32x4x2(A_at + 8);               //[k01,k01,k01,k01,k01,k01,k01,k01]
                    k1 = broadcast2float32x4x2(A_at + 9);               //[k11,k11,k11,k11,k11,k11,k11,k11]
                    k2 = broadcast2float32x4x2(A_at + 10);              //[k21,k21,k21,k21,k21,k21,k21,k21]
                    k3 = broadcast2float32x4x2(A_at + 11);              //[k31,k31,k31,k31,k31,k31,k31,k31]

                    float32x4x2 a1(B_at + 8);                              //[a10,a11,a12,a13,a14,a15,a16,a17]

                    c0 = fmadd(a1, k0, c0);
                    c1 = fmadd(a1, k1, c1);
                    c2 = fmadd(a1, k2, c2);
                    c3 = fmadd(a1, k3, c3);

                    k0 = broadcast2float32x4x2(A_at + 12);              //[k41,k41,k41,k41,k41,k41,k41,k41]
                    k1 = broadcast2float32x4x2(A_at + 13);              //[k51,k51,k51,k51,k51,k51,k51,k51]
                    k2 = broadcast2float32x4x2(A_at + 14);              //[k61,k61,k61,k61,k61,k61,k61,k61]
                    k3 = broadcast2float32x4x2(A_at + 15);              //[k71,k71,k71,k71,k71,k71,k71,k71]

                    c4 = fmadd(a1, k0, c4);
                    c5 = fmadd(a1, k1, c5);
                    c6 = fmadd(a1, k2, c6);
                    c7 = fmadd(a1, k3, c7);
                    //=====================pack_gemm k==2=====================
                    k0 = broadcast2float32x4x2(A_at + 16);              //[k02,k02,k02,k02,k02,k02,k02,k02]
                    k1 = broadcast2float32x4x2(A_at + 17);              //[k12,k12,k12,k12,k12,k12,k12,k12]
                    k2 = broadcast2float32x4x2(A_at + 18);              //[k22,k21,k21,k21,k21,k21,k21,k21]
                    k3 = broadcast2float32x4x2(A_at + 19);              //[k32,k32,k32,k32,k32,k32,k32,k32]

                    float32x4x2 a2(B_at + 16);                             //[a20,a21,a22,a23,a24,a25,a26,a27]

                    c0 = fmadd(a2, k0, c0);
                    c1 = fmadd(a2, k1, c1);
                    c2 = fmadd(a2, k2, c2);
                    c3 = fmadd(a2, k3, c3);

                    k0 = broadcast2float32x4x2(A_at + 20);              //[k42,k42,k42,k42,k42,k42,k42,k42]
                    k1 = broadcast2float32x4x2(A_at + 21);              //[k52,k52,k52,k52,k52,k52,k52,k52]
                    k2 = broadcast2float32x4x2(A_at + 22);              //[k62,k62,k62,k62,k62,k62,k62,k62]
                    k3 = broadcast2float32x4x2(A_at + 23);              //[k72,k72,k72,k72,k72,k72,k72,k72]

                    c4 = fmadd(a2, k0, c4);
                    c5 = fmadd(a2, k1, c5);
                    c6 = fmadd(a2, k2, c6);
                    c7 = fmadd(a2, k3, c7);
                    //=====================pack_gemm k==3=====================
                    k0 = broadcast2float32x4x2(A_at + 24);              //[k03,k03,k03,k03,k03,k03,k03,k03]
                    k1 = broadcast2float32x4x2(A_at + 25);              //[k13,k13,k13,k13,k13,k13,k13,k13]
                    k2 = broadcast2float32x4x2(A_at + 26);              //[k23,k23,k23,k23,k23,k23,k23,k23]
                    k3 = broadcast2float32x4x2(A_at + 27);              //[k33,k33,k33,k33,k33,k33,k33,k33]

                    float32x4x2 a3(B_at + 24);                             //[a30,a31,a32,a33,a34,a35,a36,a37]

                    c0 = fmadd(a3, k0, c0);
                    c1 = fmadd(a3, k1, c1);
                    c2 = fmadd(a3, k2, c2);
                    c3 = fmadd(a3, k3, c3);

                    k0 = broadcast2float32x4x2(A_at + 28);              //[k43,k43,k43,k43,k43,k43,k43,k43]
                    k1 = broadcast2float32x4x2(A_at + 29);              //[k53,k53,k53,k53,k53,k53,k53,k53]
                    k2 = broadcast2float32x4x2(A_at + 30);              //[k63,k63,k63,k63,k63,k63,k63,k63]
                    k3 = broadcast2float32x4x2(A_at + 31);              //[k73,k73,k73,k73,k73,k73,k73,k73]

                    c4 = fmadd(a3, k0, c4);
                    c5 = fmadd(a3, k1, c5);
                    c6 = fmadd(a3, k2, c6);
                    c7 = fmadd(a3, k3, c7);

                    A_at += 32;
                    B_at += 32;
                }

                for (int k = k_remain; k < K; k++) {
                    float32x4x2 k0 = broadcast2float32x4x2(A_at);       //[k00,k00,k00,k00,k00,k00,k00,k00]
                    float32x4x2 k1 = broadcast2float32x4x2(A_at + 1);   //[k10,k10,k10,k10,k10,k10,k10,k10]
                    float32x4x2 k2 = broadcast2float32x4x2(A_at + 2);   //[k20,k20,k20,k20,k20,k20,k20,k20]
                    float32x4x2 k3 = broadcast2float32x4x2(A_at + 3);   //[k30,k30,k30,k30,k30,k30,k30,k30]

                    float32x4x2 a0(B_at);                                  //[a00,a01,a02,a03,a04,a05,a06,a07]

                    c0 = fmadd(a0, k0, c0);
                    c1 = fmadd(a0, k1, c1);
                    c2 = fmadd(a0, k2, c2);
                    c3 = fmadd(a0, k3, c3);

                    k0 = broadcast2float32x4x2(A_at + 4);               //[k40,k40,k40,k40,k40,k40,k40,k40]
                    k1 = broadcast2float32x4x2(A_at + 5);               //[k50,k50,k50,k50,k50,k50,k50,k50]
                    k2 = broadcast2float32x4x2(A_at + 6);               //[k60,k60,k60,k60,k60,k60,k60,k60]
                    k3 = broadcast2float32x4x2(A_at + 7);               //[k70,k70,k70,k70,k70,k70,k70,k70]

                    c4 = fmadd(a0, k0, c4);
                    c5 = fmadd(a0, k1, c5);
                    c6 = fmadd(a0, k2, c6);
                    c7 = fmadd(a0, k3, c7);

                    A_at += 8;
                    B_at += 8;
                }

                c0.store(output_row0); c1.store(output_row1);
                c2.store(output_row2); c3.store(output_row3);
                c4.store(output_row4); c5.store(output_row5);
                c6.store(output_row6); c7.store(output_row7);

                output_row0 += 8; output_row1 += 8;
                output_row2 += 8; output_row3 += 8;
                output_row4 += 8; output_row5 += 8;
                output_row6 += 8; output_row7 += 8;
            }

            for (int n = n_remain; n < N; n++)
            {
                const float* A_at = A_store;
                const float* B_at = p_B + n * K;
                float32x4x2 sum_col0(0.f), sum_col1(0.f), sum_col2(0.f), sum_col3(0.f);
                float32x4x2 sum_col(0.f);

                int k_loop = K >> 2;
                int k_remain = k_loop << 2;
                for (int kk = 0; kk < k_loop; kk++) {
                    // int k = kk * 4;

                    float32x4x2 a0 = broadcast2float32x4x2(B_at);          //[a00,a00,a00,a00,a00,a00,a00,a00]
                    float32x4x2 a1 = broadcast2float32x4x2(B_at + 1);      //[a10,a10,a10,a10,a10,a10,a10,a10]
                    float32x4x2 a2 = broadcast2float32x4x2(B_at + 2);      //[a20,a20,a20,a20,a20,a20,a20,a20]
                    float32x4x2 a3 = broadcast2float32x4x2(B_at + 3);      //[a30,a30,a30,a30,a30,a30,a30,a30]

                    float32x4x2 k0(A_at);                               //[k00,k10,k20,k30,k40,k50,k60,k70]
                    float32x4x2 k1(A_at + 8);                           //[k01,k11,k21,k31,k41,k51,k61,k71]
                    float32x4x2 k2(A_at + 16);                          //[k02,k12,k22,k32,k42,k52,k62,k72]
                    float32x4x2 k3(A_at + 24);                          //[k03,k13,k23,k33,k43,k53,k63,k73]

                    sum_col0 = fmadd(k0, a0, sum_col0);
                    sum_col1 = fmadd(k1, a1, sum_col1);
                    sum_col2 = fmadd(k2, a2, sum_col2);
                    sum_col3 = fmadd(k3, a3, sum_col3);

                    A_at += 32;
                    B_at += 4;
                }

                sum_col0 += sum_col1;
                sum_col2 += sum_col3;
                sum_col += sum_col0;
                sum_col += sum_col2;

                for (int k = k_remain; k < K; k++) {
                    float32x4x2 a0 = broadcast2float32x4x2(B_at);          //[a00,a00,a00,a00,a00,a00,a00,a00]
                    float32x4x2 k0(A_at);                               //[k00,k10,k20,k30,k40,k50,k60,k70]

                    sum_col = fmadd(k0, a0, sum_col);

                    A_at += 8;
                    B_at += 1;
                }

                *output_row0++ = *((float*)&sum_col.value);
                *output_row1++ = *(((float*)&sum_col.value) + 1);
                *output_row2++ = *(((float*)&sum_col.value) + 2);
                *output_row3++ = *(((float*)&sum_col.value) + 3);
                *output_row4++ = *(((float*)&sum_col.value) + 4);
                *output_row5++ = *(((float*)&sum_col.value) + 5);
                *output_row6++ = *(((float*)&sum_col.value) + 6);
                *output_row7++ = *(((float*)&sum_col.value) + 7);
            }
        }

        for (int m = remain; m < M; m++) {
            float* output_row0 = output_at + m * ldc;
            const float* A_store = p_A + m * K;

            int n_loop = N >> 3;
            int n_remain = n_loop << 3;
            for (int nn = 0; nn < n_loop; nn++) {
                int n = nn * 8;

                const float* A_at = A_store;
                const float* B_at = p_B + n * K;

                float32x4x2 c0(0.f);

                int k_loop = K >> 2;
                int k_remain = k_loop << 2;
                for (int kk = 0; kk < k_loop; kk++) {

                    float32x4x2 k0 = broadcast2float32x4x2(A_at);       //[k00,k00,k00,k00,k00,k00,k00,k00]
                    float32x4x2 k1 = broadcast2float32x4x2(A_at + 1);   //[k01,k01,k01,k01,k01,k01,k01,k01]
                    float32x4x2 k2 = broadcast2float32x4x2(A_at + 2);   //[k02,k02,k02,k02,k02,k02,k02,k02]
                    float32x4x2 k3 = broadcast2float32x4x2(A_at + 3);   //[k03,k03,k03,k03,k03,k03,k03,k03]

                    float32x4x2 a0(B_at);                                  //[a00,a01,a02,a03,a04,a05,a06,a07]
                    float32x4x2 a1(B_at + 8);                              //[a10,a11,a12,a13,a14,a15,a16,a17]
                    float32x4x2 a2(B_at + 16);                             //[a20,a21,a22,a23,a24,a25,a26,a27]
                    float32x4x2 a3(B_at + 24);                             //[a30,a31,a32,a33,a34,a35,a36,a37]

                    c0 = fmadd(k0, a0, c0);
                    c0 = fmadd(k1, a1, c0);
                    c0 = fmadd(k2, a2, c0);
                    c0 = fmadd(k3, a3, c0);

                    A_at += 4;
                    B_at += 32;
                }

                for (int k = k_remain; k < K; k++) {
                    float32x4x2 k0 = broadcast2float32x4x2(A_at);        //[k00,k00,k00,k00,k00,k00,k00,k00]
                    float32x4x2 a0(B_at);                                   //[a00,a01,a02,a03,a04,a05,a06,a07]

                    c0 = fmadd(k0, a0, c0);

                    A_at += 1;
                    B_at += 8;
                }

                c0.store(output_row0);
                output_row0 += 8;
            }

            for (int n = n_remain; n < N; n++) {
                float32x4 c0(0.f);
                float sum0 = 0;

                const float* A_at = A_store;
                const float* B_at = p_B + n * K;

                int k_loop = K >> 2;
                int k_remain = k_loop << 2;
                for (int kk = 0; kk < k_loop; kk++) {
                    // int k = kk * 4;
                    float32x4 k0(A_at);
                    float32x4 a0(B_at);

                    c0 = fmadd(k0, a0, c0);

                    A_at += 4;
                    B_at += 4;
                }

                sum0 = seeta::sum(c0);

                for (int k = k_remain; k < K; k++) {
                    sum0 += (*A_at) * (*B_at);
                    A_at++;
                    B_at++;
                }

                *output_row0 = sum0;
                output_row0++;
            }
        }
    }

    //Note:only support alpha==1,beta=0 now
    template<typename T>
    void math<T>::gemm_pack(
        blas::Transpose TransA,
        blas::Transpose TransB,
        int M, int N, int K,
        T alpha, const T *A, const T *B,
        T beta, T *C) {

        if (!near<T>(alpha, 1.f) || !near<T>(beta, 0.f)) {
            std::cout << "alpha shoule be one and beta should be zero!";
            throw std::logic_error("gemm_pack failed!");
        }

        std::shared_ptr<T> A_trans;
        std::shared_ptr<T> B_trans;
        std::shared_ptr<T> A_packed;
        std::shared_ptr<T> B_packed;

        auto vat = orz::ctx::lite::ptr<orz::Vat>();
        if (TransA == blas::Transpose::Trans) {
            A_trans = vat->calloc_shared<T>(M * K);
            matrix_transpose(A, A_trans.get(), K, M);
        }
        if (TransB == blas::Transpose::Trans) {
            B_trans = vat->calloc_shared<T>(K * N);
            matrix_transpose(B, B_trans.get(), N, K);
        }

        A_packed = vat->calloc_shared<T>(M * K);
        if (TransA == blas::Transpose::Trans)
            pack_A(M, K, A_trans.get(), K, A_packed.get());
        else
            pack_A(M, K, A, K, A_packed.get());

        B_packed = vat->calloc_shared<T>(K * N);
        if (TransB == blas::Transpose::Trans)
            pack_B(K, N, B_trans.get(), N, B_packed.get());
        else
            pack_B(K, N, B, N, B_packed.get());

        kernel_8x8(M, K, N, alpha, A_packed.get(), B_packed.get(), beta, C, N);
    }

    template<typename T>
    inline T inline_asum( int N, const T *x, int incx )
    {
        T sum = 0;
        // block: 4
        int i = 0;
        static const int block_size = 4;
        int blocked_N = N % block_size ? N - block_size : N;
        for( ; i < blocked_N; i += block_size )
        {
            sum += abs( *x );
            x += incx;
            sum += abs( *x );
            x += incx;
            sum += abs( *x );
            x += incx;
            sum += abs( *x );
            x += incx;
        }
        for( ; i < N; ++i )
        {
            sum += abs( *x );
            x += incx;
        }
        return sum;
    }

    template<typename T>
    T math<T>::asum( int N, const T *x, int incx )
    {
        auto gun = orz::ctx::lite::ptr<orz::Shotgun>();
        if( gun == nullptr )
        {
            return inline_asum<T>( N, x, incx );
        }
        auto bins = orz::split_bins( 0, N, int( gun->size() ) );
        std::vector<T> threads_sum( gun->size(), 0 );
        for( auto &range : bins )
        {
            gun->fire( [ &, range]( int id )
            {
                const T *local_x = x + range.first * incx;
                threads_sum[id] = inline_asum<T>( range.second - range.first, local_x, incx );
            } );
        }
        gun->join();
        T sum = 0;
        for( auto v : threads_sum ) sum += v;
        return sum;
    }

    template<typename T>
    T math<T>::abs( T val )
    {
        return std::fabs( val );
    }
}


template class seeta::math<float>;
template class seeta::math<double>;

