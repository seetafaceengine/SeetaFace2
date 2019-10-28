#ifndef _SEETANET_SIMD_H
#define _SEETANET_SIMD_H

#include <stdint.h>

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#include <utility>

using _simd_f32x4 = float32x4_t;
using _simd_f32x4x2 = float32x4x2_t;
using _simd_f32x2 = float32x2_t;
using _simd_f32 = float;
using _simd_int32x4 = int32x4_t;
using _simd_int32 = int32_t;
using _simd_int32x4x2 = int32x4x2_t;

inline _simd_int32x4 _simd_int32x4_load(const _simd_int32* p) {
    return vld1q_s32(p);
}

inline _simd_int32x4 _simd_int32x4_set(_simd_int32 a, _simd_int32 b, _simd_int32 c, _simd_int32 d) {
    _simd_int32 array[4] = { a, b, c, d };
    return vld1q_s32(array);
}

inline void _simd_int32x4_store(_simd_int32 *p, _simd_int32x4 m) {
    vst1q_s32(p, m);
}

inline _simd_int32x4 _simd_int32x4_add(_simd_int32x4 lhs, _simd_int32x4 rhs) {
    return vaddq_s32(lhs, rhs);
}

inline _simd_int32x4 _simd_int32x4_sub(_simd_int32x4 lhs, _simd_int32x4 rhs) {
    return vsubq_s32(lhs, rhs);
}

inline _simd_int32x4x2 _simd_int32x4x2_load(const _simd_int32* p) {
    _simd_int32x4x2 res;
    res.val[0] = vld1q_s32(p);
    res.val[1] = vld1q_s32(p + 4);
    return std::move(res);
    //return vld2q_s32(p);
}

inline _simd_int32x4x2 _simd_int32x4x2_set(_simd_int32 a, _simd_int32 b, _simd_int32 c, _simd_int32 d,
    _simd_int32 e, _simd_int32 f, _simd_int32 g, _simd_int32 h) {
    _simd_int32x4x2 res;
    _simd_int32 array_0[4] = { a, b, c, d };
    _simd_int32 array_1[4] = { e, f, g, h };
    res.val[0] = vld1q_s32(array_0); res.val[1] = vld1q_s32(array_1);
    return std::move(res);
    //_simd_int32 array[8] = { a, b, c, d, e, f, g, h };
    //return vld2q_s32(array);
}

inline void _simd_int32x4x2_store(_simd_int32 *p, _simd_int32x4x2 m) {
    vst1q_s32(p, m.val[0]);
    vst1q_s32(p + 4, m.val[1]);
    //vst2q_s32(p, m);
}

inline _simd_int32x4x2 _simd_int32x4x2_add(_simd_int32x4x2 lhs, _simd_int32x4x2 rhs) {
    _simd_int32x4x2 res;
    res.val[0] = vaddq_s32(lhs.val[0], rhs.val[0]);
    res.val[1] = vaddq_s32(lhs.val[1], rhs.val[1]);
    return std::move(res);
}

inline _simd_int32x4x2 _simd_int32x4x2_sub(_simd_int32x4x2 lhs, _simd_int32x4x2 rhs) {
    _simd_int32x4x2 res;
    res.val[0] = vsubq_s32(lhs.val[0], rhs.val[0]);
    res.val[1] = vsubq_s32(lhs.val[1], rhs.val[1]);
    return std::move(res);
}

inline _simd_f32x4 _simd_f32x4_load(const _simd_f32 *p) {
    return vld1q_f32(p);
}

inline _simd_f32x4 _simd_f32x4_set(_simd_f32 a, _simd_f32 b, _simd_f32 c, _simd_f32 d) {
    _simd_f32 array[4] = { a, b, c, d };
    return vld1q_f32(array);
}

inline void _simd_f32x4_store(_simd_f32 *p, _simd_f32x4 m) {
    vst1q_f32(p, m);
}

inline _simd_f32x4 _simd_f32x4_add(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return vaddq_f32(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_sub(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return vsubq_f32(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_mul(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return vmulq_f32(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_div(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    _simd_f32x4 recip = vrecpeq_f32(rhs);
    return vmulq_f32(lhs, recip);
}

inline _simd_f32x4 _simd_f32x4_max(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return vmaxq_f32(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_min(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return vminq_f32(lhs, rhs);
}

inline void _simd_f32x4_transpose4x4(_simd_f32x4& q0, _simd_f32x4& q1, _simd_f32x4& q2, _simd_f32x4& q3) {

    /*
    * q0 = (s00,s01,s02,s03)
    * q1 = (s10,s11,s12,s13)
    * q2 = (s20,s21,s22,s23)
    * q3 = (s30,s31,s32,s33)
    */
    /*
    * q01 = (s00,s10,s02,s12),(s01,s11,s03,s13)
    * q02 = (s20,s30,s22,s32),(s21,s31,s23,s33)
    */
    _simd_f32x4x2 q01 = vtrnq_f32(q0, q1);
    _simd_f32x4x2 q23 = vtrnq_f32(q2, q3);

    _simd_f32x2 d00 = vget_low_f32(q01.val[0]);
    _simd_f32x2 d01 = vget_high_f32(q01.val[0]);

    _simd_f32x2 d10 = vget_low_f32(q01.val[1]);
    _simd_f32x2 d11 = vget_high_f32(q01.val[1]);

    _simd_f32x2 d20 = vget_low_f32(q23.val[0]);
    _simd_f32x2 d21 = vget_high_f32(q23.val[0]);

    _simd_f32x2 d30 = vget_low_f32(q23.val[1]);
    _simd_f32x2 d31 = vget_high_f32(q23.val[1]);

    q0 = vcombine_f32(d00, d20);
    q1 = vcombine_f32(d10, d30);
    q2 = vcombine_f32(d01, d21);
    q3 = vcombine_f32(d11, d31);
}

inline _simd_f32x4 _simd_f32x4_fmadd(const _simd_f32x4& q0, const _simd_f32x4& q1, const _simd_f32x4& q2) {
    return vmlaq_f32(q2, q0, q1);
    //_simd_f32x4 mul_tmp = vmulq_f32(q0, q1);
    //return vaddq_f32(mul_tmp, q2);
}

inline _simd_f32x4 _simd_broadcast2float32x4(const _simd_f32* src) {
    return vdupq_n_f32(*src);
}


inline _simd_f32x4x2 _simd_f32x4x2_load(const _simd_f32 *p) {
    _simd_f32x4x2 res;
    res.val[0] = vld1q_f32(p);
    res.val[1] = vld1q_f32(p + 4);
    return std::move(res);
    //return vld2q_f32(p);
}

inline _simd_f32x4x2 _simd_f32x4x2_set(_simd_f32 a, _simd_f32 b, _simd_f32 c, _simd_f32 d,
    _simd_f32 e, _simd_f32 f, _simd_f32 g, _simd_f32 h) {
    _simd_f32x4x2 res;
    _simd_f32 array_0[4] = { a, b, c, d };
    _simd_f32 array_1[4] = { e, f, g, h };
    res.val[0] = vld1q_f32(array_0); res.val[1] = vld1q_f32(array_1);
    return std::move(res);
    //_simd_f32 array[8] = { a, b, c, d, e, f, g, h };
    //return vld2q_f32(array);
}

inline void _simd_f32x4x2_store(_simd_f32 *p, _simd_f32x4x2 m) {
    vst1q_f32(p, m.val[0]);
    vst1q_f32(p + 4, m.val[1]);
    //vst2q_f32(p, m);
}

inline _simd_f32x4x2 _simd_f32x4x2_add(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    _simd_f32x4x2 res;
    res.val[0] = vaddq_f32(lhs.val[0], rhs.val[0]);
    res.val[1] = vaddq_f32(lhs.val[1], rhs.val[1]);
    return std::move(res);
}

inline _simd_f32x4x2 _simd_f32x4x2_sub(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    _simd_f32x4x2 res;
    res.val[0] = vsubq_f32(lhs.val[0], rhs.val[0]);
    res.val[1] = vsubq_f32(lhs.val[1], rhs.val[1]);
    return std::move(res);
}

inline _simd_f32x4x2 _simd_f32x4x2_mul(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    _simd_f32x4x2 res;
    res.val[0] = vmulq_f32(lhs.val[0], rhs.val[0]);
    res.val[1] = vmulq_f32(lhs.val[1], rhs.val[1]);
    return std::move(res);
}

inline _simd_f32x4x2 _simd_f32x4x2_div(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    _simd_f32x4x2 res;
    _simd_f32x4 recip_0 = vrecpeq_f32(rhs.val[0]);
    _simd_f32x4 recip_1 = vrecpeq_f32(rhs.val[1]);
    res.val[0] = vmulq_f32(lhs.val[0], recip_0);
    res.val[1] = vmulq_f32(lhs.val[1], recip_1);
    return std::move(res);
}

inline _simd_f32x4x2 _simd_f32x4x2_fmadd(_simd_f32x4x2 q0, _simd_f32x4x2 q1, _simd_f32x4x2 q2) {
    _simd_f32x4x2 res;
    res.val[0] = vmlaq_f32(q2.val[0], q0.val[0], q1.val[0]);
    res.val[1] = vmlaq_f32(q2.val[1], q0.val[1], q1.val[1]);
    //_simd_f32x4 mul_tmp_0 = vmulq_f32(q0.val[0], q1.val[0]);
    //_simd_f32x4 mul_tmp_1 = vmulq_f32(q0.val[1], q1.val[1]);
    //res.val[0] = vaddq_f32(mul_tmp_0, q2.val[0]);
    //res.val[1] = vaddq_f32(mul_tmp_1, q2.val[1]);
    return std::move(res);
}

//cast
inline _simd_int32x4x2 _simd_floatx4x2_to_int32x4x2(_simd_f32x4x2 src) {
    _simd_int32x4x2 res;
    res.val[0] = vcvtq_s32_f32(src.val[0]);
    res.val[1] = vcvtq_s32_f32(src.val[1]);
    return std::move(res);
}

inline _simd_f32x4x2 _simd_intx4x2_to_float32x4x2(_simd_int32x4x2 src) {
    _simd_f32x4x2 res;
    res.val[0] = vcvtq_f32_s32(src.val[0]);
    res.val[1] = vcvtq_f32_s32(src.val[1]);
    return std::move(res);
}

//broad cast
inline _simd_f32x4x2 _simd_broadcast2float32x4x2(const _simd_f32* src) {
    _simd_f32x4x2 res;
    res.val[0] = vdupq_n_f32(*src);
    res.val[1] = vdupq_n_f32(*src);
    return std::move(res);
}

#else

#ifdef SEETA_USE_SSE2
#include <immintrin.h>

typedef struct __m128x2
{
    __m128 val[2];
} __m128x2;

typedef struct __m128ix2
{
    __m128i val[2];
}__m128ix2;

using _simd_f32x4 = __m128;
using _simd_f32x4x2 = __m128x2;
using _simd_f32 = float;
using _simd_int32x4 = __m128i;
using _simd_int32 = int32_t;
using _simd_int32x4x2 = __m128ix2;

inline _simd_int32x4 _simd_int32x4_load(const _simd_int32* p) {
    return _mm_loadu_si128((_simd_int32x4*)p);
}

inline _simd_int32x4 _simd_int32x4_set(const _simd_int32 &a, const _simd_int32 &b, const _simd_int32 &c, const _simd_int32 &d) {
    return _mm_set_epi32(d, c, b, a);
}

inline void _simd_int32x4_store(_simd_int32 *p, const _simd_int32x4 &m) {
    _mm_store_si128((_simd_int32x4*)p, m);
}

inline _simd_int32x4 _simd_int32x4_add(const _simd_int32x4 &lhs, const _simd_int32x4 &rhs) {
    return _mm_add_epi32(lhs, rhs);
}

inline _simd_int32x4 _simd_int32x4_sub(const _simd_int32x4 &lhs, const _simd_int32x4 &rhs) {
    return _mm_sub_epi32(lhs, rhs);
}

inline _simd_int32x4x2 _simd_int32x4x2_load(const _simd_int32* p) {
    _simd_int32x4x2 res;
    res.val[0] = _mm_loadu_si128((_simd_int32x4*)p);
    res.val[1] = _mm_loadu_si128((_simd_int32x4*)(p + 4));
    return res;
}

inline _simd_int32x4x2 _simd_int32x4x2_set(_simd_int32 a, _simd_int32 b, _simd_int32 c, _simd_int32 d,
    _simd_int32 e, _simd_int32 f, _simd_int32 g, _simd_int32 h) {
    _simd_int32x4x2 res;
    res.val[0] = _mm_set_epi32(d, c, b, a);
    res.val[1] = _mm_set_epi32(h, g, f, e);
    return res;
}

inline void _simd_int32x4x2_store(_simd_int32 *p, const _simd_int32x4x2 &m) {
    _mm_storeu_si128((_simd_int32x4*)p, m.val[0]);
    _mm_storeu_si128((_simd_int32x4*)(p + 4), m.val[1]);
}

inline _simd_int32x4x2 _simd_int32x4x2_add(const _simd_int32x4x2 &lhs, const _simd_int32x4x2 &rhs) {
    _simd_int32x4x2 res;
    res.val[0] = _mm_add_epi32(lhs.val[0], rhs.val[0]);
    res.val[1] = _mm_add_epi32(lhs.val[1], rhs.val[1]);
    return res;
}

inline _simd_int32x4x2 _simd_int32x4x2_sub(const _simd_int32x4x2 &lhs, const _simd_int32x4x2 &rhs) {
    _simd_int32x4x2 res;
    res.val[0] = _mm_sub_epi32(lhs.val[0], rhs.val[0]);
    res.val[1] = _mm_sub_epi32(lhs.val[1], rhs.val[1]);
    return res;
}


inline _simd_f32x4 _simd_f32x4_load(const _simd_f32 *p) {
    return _mm_loadu_ps(p);
}

inline _simd_f32x4 _simd_f32x4_set(_simd_f32 a, _simd_f32 b, _simd_f32 c, _simd_f32 d) {
    return _mm_set_ps(d, c, b, a);
}

inline void _simd_f32x4_store(_simd_f32 *p, const _simd_f32x4 &m) {
    _mm_storeu_ps(p, m);
}

inline _simd_f32x4 _simd_f32x4_add(const _simd_f32x4 &lhs, const _simd_f32x4 &rhs) {
    return _mm_add_ps(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_sub(const _simd_f32x4 &lhs, const _simd_f32x4 &rhs) {
    return _mm_sub_ps(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_mul(const _simd_f32x4 &lhs, const _simd_f32x4 &rhs) {
    return _mm_mul_ps(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_div(const _simd_f32x4 &lhs, const _simd_f32x4 &rhs) {
    return _mm_div_ps(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_max(const _simd_f32x4 &lhs, const _simd_f32x4 &rhs) {
    return _mm_max_ps(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_min(const _simd_f32x4 &lhs, const _simd_f32x4 &rhs) {
    return _mm_min_ps(lhs, rhs);
}

inline void _simd_f32x4_transpose4x4(_simd_f32x4& q0, _simd_f32x4& q1, _simd_f32x4& q2, _simd_f32x4& q3) {
    _MM_TRANSPOSE4_PS(q0, q1, q2, q3);
}

inline _simd_f32x4 _simulate_simd_f32x4_fmadd(const _simd_f32x4& q0, const _simd_f32x4& q1, const _simd_f32x4& q2) {
#ifdef SEETA_USE_FMA
	return _mm_fmadd_ps(q0, q1, q2);
#else
	return _mm_add_ps(_mm_mul_ps(q0, q1), q2);
#endif
}

inline _simd_f32x4 _simd_f32x4_fmadd(const _simd_f32x4& q0, const _simd_f32x4& q1, const _simd_f32x4& q2) {
    return _simulate_simd_f32x4_fmadd(q0, q1, q2);
}

inline _simd_f32x4 _simd_f32x4_fmadd(const _simd_f32x4& q0, const _simd_f32x4& q1, const _simd_f32x4& q2, const int index) {
    if (index >= 0 && index <= 3) {
        return _simulate_simd_f32x4_fmadd(q0, _mm_set1_ps(*((float*)&q1 + index)), q2);
    }
	return _simd_f32x4_set(0, 0, 0, 0);
}

inline _simd_f32x4 _simd_broadcast2float32x4(const _simd_f32* src) {
    return _mm_set1_ps(*src);
}

inline _simd_f32x4 _simd_f32x4_concat(const _simd_f32x4& q0, const _simd_f32x4& q1, const int index) {
    if (index == 0)
        return q0;
    float res[4];
    for (int i = index; i < 4; i++) {
        res[i - index] = *(((float*)&q0) + i);
    }
    for (int i = 0; i < index; i++) {
        res[i + 4 - index] = *(((float*)&q1) + i);
    }
    return _mm_loadu_ps(res);
}


inline _simd_f32x4x2 _simd_f32x4x2_load(const _simd_f32 *p) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_loadu_ps(p);
    res.val[1] = _mm_loadu_ps(p + 4);
    return res;
}

inline _simd_f32x4x2 _simd_f32x4x2_set(_simd_f32 a, _simd_f32 b, _simd_f32 c, _simd_f32 d,
    _simd_f32 e, _simd_f32 f, _simd_f32 g, _simd_f32 h) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_set_ps(d, c, b, a);
    res.val[1] = _mm_set_ps(h, g, f, e);
    return res;
}

inline void _simd_f32x4x2_store(_simd_f32 *p, const _simd_f32x4x2 &m) {
    _mm_storeu_ps(p, m.val[0]);
    _mm_storeu_ps(p + 4, m.val[1]);
}

inline _simd_f32x4x2 _simd_f32x4x2_add(const _simd_f32x4x2 &lhs, const _simd_f32x4x2 &rhs) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_add_ps(lhs.val[0], rhs.val[0]);
    res.val[1] = _mm_add_ps(lhs.val[1], rhs.val[1]);
    return res;
}

inline _simd_f32x4x2 _simd_f32x4x2_sub(const _simd_f32x4x2 &lhs, const _simd_f32x4x2 &rhs) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_sub_ps(lhs.val[0], rhs.val[0]);
    res.val[1] = _mm_sub_ps(lhs.val[1], rhs.val[1]);
    return res;
}

inline _simd_f32x4x2 _simd_f32x4x2_mul(const _simd_f32x4x2 &lhs, const _simd_f32x4x2 &rhs) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_mul_ps(lhs.val[0], rhs.val[0]);
    res.val[1] = _mm_mul_ps(lhs.val[1], rhs.val[1]);
    return res;
}

inline _simd_f32x4x2 _simd_f32x4x2_div(const _simd_f32x4x2 &lhs, const _simd_f32x4x2 &rhs) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_div_ps(lhs.val[0], rhs.val[0]);
    res.val[1] = _mm_div_ps(lhs.val[1], rhs.val[1]);
    return res;
}

inline _simd_f32x4x2 _simd_f32x4x2_fmadd(const _simd_f32x4x2 &q0, const _simd_f32x4x2 &q1, const _simd_f32x4x2 &q2) {
    _simd_f32x4x2 res;
    res.val[0] = _simulate_simd_f32x4_fmadd(q0.val[0], q1.val[0], q2.val[0]);
    res.val[1] = _simulate_simd_f32x4_fmadd(q0.val[1], q1.val[1], q2.val[1]);
    return res;
}

//cast
inline _simd_int32x4x2 _simd_floatx4x2_to_int32x4x2(const _simd_f32x4x2 &src) {
    _simd_int32x4x2 res;
    res.val[0] = _mm_cvtps_epi32(src.val[0]);
    res.val[1] = _mm_cvtps_epi32(src.val[1]);
    return res;
}

inline _simd_f32x4x2 _simd_intx4x2_to_float32x4x2(const _simd_int32x4x2 &src) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_cvtepi32_ps(src.val[0]);
    res.val[1] = _mm_cvtepi32_ps(src.val[1]);
    return res;
}

inline _simd_f32x4x2 _simd_broadcast2float32x4x2(const _simd_f32* src) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_set1_ps(*src);
    res.val[1] = _mm_set1_ps(*src);
    return res;
}

#else
#include <array>
#include <math.h>

using _simd_f32 = float;
using _simd_f32x4 = std::array<_simd_f32, 4>;
using _simd_f32x4x2 = std::array<_simd_f32, 8>;
using _simd_int32 = int32_t;
using _simd_int32x4 = std::array<_simd_int32, 4>;
using _simd_int32x4x2 = std::array<_simd_int32, 8>;

inline _simd_int32x4 _simd_int32x4_load(const _simd_int32* p) {
    return{ p[0], p[1], p[2], p[3] };
}

inline _simd_int32x4 _simd_int32x4_set(_simd_int32 a, _simd_int32 b, _simd_int32 c, _simd_int32 d) {
    return{ a, b, c, d };
}

inline void _simd_int32x4_store(_simd_int32 *p, const _simd_int32x4 &m) {
    p[0] = m[0];
    p[1] = m[1];
    p[2] = m[2];
    p[3] = m[3];
}

inline _simd_int32x4 _simd_int32x4_add(const _simd_int32x4 &lhs, const _simd_int32x4 &rhs) {
    return{ lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3] };
}

inline _simd_int32x4 _simd_int32x4_sub(const _simd_int32x4 &lhs, const _simd_int32x4 &rhs) {
    return{ lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2], lhs[3] - rhs[3] };
}

inline _simd_int32x4x2 _simd_int32x4x2_load(const _simd_int32* p) {
    return{ p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7] };
}

inline _simd_int32x4x2 _simd_int32x4x2_set(_simd_int32 a, _simd_int32 b, _simd_int32 c, _simd_int32 d,
    _simd_int32 e, _simd_int32 f, _simd_int32 g, _simd_int32 h) {
    return{ a, b, c, d, e, f, g, h };
}

inline void _simd_int32x4x2_store(_simd_int32 *p, const _simd_int32x4x2 &m) {
    p[0] = m[0]; p[1] = m[1];
    p[2] = m[2]; p[3] = m[3];
    p[4] = m[4]; p[5] = m[5];
    p[6] = m[6]; p[7] = m[7];
}

inline _simd_int32x4x2 _simd_int32x4x2_add(const _simd_int32x4x2 &lhs, const _simd_int32x4x2 &rhs) {
    return{ lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3], lhs[4] + rhs[4], lhs[5] + rhs[5], lhs[6] + rhs[6], lhs[7] + rhs[7] };
}

inline _simd_int32x4x2 _simd_int32x4x2_sub(const _simd_int32x4x2 &lhs, const _simd_int32x4x2 &rhs) {
    return{ lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2], lhs[3] - rhs[3], lhs[4] - rhs[4], lhs[5] - rhs[5], lhs[6] - rhs[6], lhs[7] - rhs[7] };
}


inline _simd_f32x4 _simd_f32x4_load(const _simd_f32 *p) {
    return{ p[0], p[1], p[2], p[3] };
}

inline _simd_f32x4 _simd_f32x4_set(_simd_f32 a, _simd_f32 b, _simd_f32 c, _simd_f32 d) {
    return{ a, b, c, d };
}

inline void _simd_f32x4_store(_simd_f32 *p, const _simd_f32x4 &m) {
    p[0] = m[0];
    p[1] = m[1];
    p[2] = m[2];
    p[3] = m[3];
}

inline _simd_f32x4 _simd_f32x4_add(const _simd_f32x4 &lhs, const _simd_f32x4 &rhs) {
    return{ lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3] };
}

inline _simd_f32x4 _simd_f32x4_sub(const _simd_f32x4 &lhs, const _simd_f32x4 &rhs) {
    return{ lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2], lhs[3] - rhs[3] };
}

inline _simd_f32x4 _simd_f32x4_mul(const _simd_f32x4 &lhs, const _simd_f32x4 &rhs) {
    return{ lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2], lhs[3] * rhs[3] };
}

inline _simd_f32x4 _simd_f32x4_div(const _simd_f32x4 &lhs, const _simd_f32x4 &rhs) {
    return{ lhs[0] / rhs[0], lhs[1] / rhs[1], lhs[2] / rhs[2], lhs[3] / rhs[3] };
}

inline _simd_f32x4 _simd_f32x4_max(const _simd_f32x4 &lhs, const _simd_f32x4 &rhs) {
    return{ std::max(lhs[0],rhs[0]), std::max(lhs[1],rhs[1]), std::max(lhs[2],rhs[2]), std::max(lhs[3],rhs[3]) };
}

inline _simd_f32x4 _simd_f32x4_min(const _simd_f32x4 &lhs, const _simd_f32x4 &rhs) {
    return{ std::min(lhs[0],rhs[0]), std::min(lhs[1],rhs[1]), std::min(lhs[2],rhs[2]), std::min(lhs[3],rhs[3]) };
}

inline void _simd_f32x4_transpose4x4(_simd_f32x4& q0, _simd_f32x4& q1, _simd_f32x4& q2, _simd_f32x4& q3) {
    //TODO:optimize?
    /*
    q0[0] = q0[0]; q1[0] = q0[1]; q2[0] = q0[2]; q3[0] = q0[3];
    q0[1] = q1[0]; q1[1] = q1[1]; q2[1] = q1[2]; q3[1] = q1[3];
    q0[2] = q2[0]; q1[2] = q2[1]; q2[2] = q2[2]; q3[2] = q2[3];
    q0[3] = q3[0]; q1[3] = q3[1]; q2[3] = q3[2]; q3[3] = q3[3];
    */
    _simd_f32 t0[4], t1[4], t2[4], t3[4];
    t0[0] = q0[0]; t1[0] = q0[1]; t2[0] = q0[2]; t3[0] = q0[3];
    t0[1] = q1[0]; t1[1] = q1[1]; t2[1] = q1[2]; t3[1] = q1[3];
    t0[2] = q2[0]; t1[2] = q2[1]; t2[2] = q2[2]; t3[2] = q2[3];
    t0[3] = q3[0]; t1[3] = q3[1]; t2[3] = q3[2]; t3[3] = q3[3];
    for (int i = 0; i < 4; i++)
    {
        q0[i] = t0[i]; q1[i] = t1[i]; q2[i] = t2[i]; q3[i] = t3[i];
    }

}

inline _simd_f32x4 _simd_f32x4_fmadd(const _simd_f32x4& q0, const _simd_f32x4& q1, const _simd_f32x4& q2) {
    return{ q0[0] * q1[0] + q2[0], q0[1] * q1[1] + q2[1], q0[2] * q1[2] + q2[2], q0[3] * q1[3] + q2[3] };
}

inline _simd_f32x4 _simd_broadcast2float32x4(const _simd_f32* src) {
    float val = *src;
    return{ val, val, val, val };
}


inline _simd_f32x4x2 _simd_f32x4x2_load(const _simd_f32 *p) {
    return{ p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7] };
}

inline _simd_f32x4x2 _simd_f32x4x2_set(_simd_f32 a, _simd_f32 b, _simd_f32 c, _simd_f32 d,
    _simd_f32 e, _simd_f32 f, _simd_f32 g, _simd_f32 h) {
    return{ a, b, c, d, e, f, g, h };
}

inline void _simd_f32x4x2_store(_simd_f32 *p, const _simd_f32x4x2 &m) {
    p[0] = m[0]; p[1] = m[1];
    p[2] = m[2]; p[3] = m[3];
    p[4] = m[4]; p[5] = m[5];
    p[6] = m[6]; p[7] = m[7];
}

inline _simd_f32x4x2 _simd_f32x4x2_add(const _simd_f32x4x2 &lhs, const _simd_f32x4x2 &rhs) {
    return{ lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3], lhs[4] + rhs[4], lhs[5] + rhs[5], lhs[6] + rhs[6], lhs[7] + rhs[7] };
}

inline _simd_f32x4x2 _simd_f32x4x2_sub(const _simd_f32x4x2 &lhs, const _simd_f32x4x2 &rhs) {
    return{ lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2], lhs[3] - rhs[3], lhs[4] - rhs[4], lhs[5] - rhs[5], lhs[6] - rhs[6], lhs[7] - rhs[7] };
}

inline _simd_f32x4x2 _simd_f32x4x2_mul(const _simd_f32x4x2 &lhs, const _simd_f32x4x2 &rhs) {
    return{ lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2], lhs[3] * rhs[3], lhs[4] * rhs[4], lhs[5] * rhs[5], lhs[6] * rhs[6], lhs[7] * rhs[7] };
}

inline _simd_f32x4x2 _simd_f32x4x2_div(const _simd_f32x4x2 &lhs, const _simd_f32x4x2 &rhs) {
    return{ lhs[0] / rhs[0], lhs[1] / rhs[1], lhs[2] / rhs[2], lhs[3] / rhs[3], lhs[4] / rhs[4], lhs[5] / rhs[5], lhs[6] / rhs[6], lhs[7] / rhs[7] };
}

inline _simd_f32x4x2 _simd_f32x4x2_fmadd(const _simd_f32x4x2 &q0, const _simd_f32x4x2 &q1, const _simd_f32x4x2 &q2) {
    return{ q0[0] * q1[0] + q2[0], q0[1] * q1[1] + q2[1], q0[2] * q1[2] + q2[2], q0[3] * q1[3] + q2[3], q0[4] * q1[4] + q2[4], q0[5] * q1[5] + q2[5], q0[6] * q1[6] + q2[6], q0[7] * q1[7] + q2[7] };
}

//cast
inline _simd_int32x4x2 _simd_floatx4x2_to_int32x4x2(const _simd_f32x4x2 &src) {
    return{ (int32_t)round(src[0]), (int32_t)round(src[1]), (int32_t)round(src[2]), (int32_t)round(src[3]),(int32_t)round(src[4]), (int32_t)round(src[5]), (int32_t)round(src[6]), (int32_t)round(src[7]) };
}

inline _simd_f32x4x2 _simd_intx4x2_to_float32x4x2(const _simd_int32x4x2 &src) {
    return{ (float)src[0], (float)src[1], (float)src[2], (float)src[3],(float)src[4], (float)src[5], (float)src[6], (float)src[7] };
}

//broad cast
inline _simd_f32x4x2 _simd_broadcast2float32x4x2(const _simd_f32* src) {
    float val = *src;
    return{ val, val, val, val, val, val, val, val };
}

#endif //SEETA_USE_SSE2
#endif //defined(__ARM_NEON__) || defined(__ARM_NEON)

namespace seeta {
    template<typename T, int M>
    class simd_base {
    public:
        using self = simd_base;
        using base = T;
        static const int width = M;
    };

    template<typename T, int M>
    class simd : public simd_base<T, M> {
    public:
        using self = simd;
        using supper = simd_base<T, M>;

        void store(typename supper::base *p) const;
    };

    using float32x4 = simd<float, 4>;
    using float32x4x2 = simd<float, 8>;

    using int32x4 = simd<int32_t, 4>;
    using int32x4x2 = simd<int32_t, 8>;

    template<typename T, int M>
    inline T sum(const simd<T, M> &value) {
        T a[M];
        value.store(a);
        T sum = 0;
        for (int i = 0; i < M; ++i) sum += a[i];
        return sum;
    }

    template<typename T>
    inline T sum(const simd<T, 4> &value) {
        T a[4];
        value.store(a);
        return a[0] + a[1] + a[2] + a[3];
    }

    template<typename T>
    inline T sum(const simd<T, 4> &value, int index) {
        T a[4];
        value.store(a);
        T sum = 0;
        for (int i = 0; i < index && i < 4; i++) {
            sum += a[i];
        }
        return sum;
    }

    template<typename T, int M>
    inline const simd<T, M> &operator+=(simd<T, M> &lhs, const simd<T, M> &rhs) {
        return lhs = lhs + rhs;
    }

    template<>
    class simd<float, 4> : public simd_base<float, 4> {
    public:
        using self = simd;
        using type = _simd_f32x4;

        type value;

        simd() = default;

        simd(type value) : value(value) {}

        simd(base a) : simd(a, a, a, a) {}

        simd(int a) : simd(base(a)) {}

        simd(const base *p) : value(_simd_f32x4_load(p)) {}

        simd(base a, base b, base c, base d) : value(_simd_f32x4_set(a, b, c, d)) {}

        void store(base *p) const { _simd_f32x4_store(p, value); }
    };

    inline simd<float, 4> operator+(const simd<float, 4> &lhs, const simd<float, 4> &rhs) {
        return _simd_f32x4_add(lhs.value, rhs.value);
    }

    inline simd<float, 4> operator-(const simd<float, 4> &lhs, const simd<float, 4> &rhs) {
        return _simd_f32x4_sub(lhs.value, rhs.value);
    }

    inline simd<float, 4> operator*(const simd<float, 4> &lhs, const simd<float, 4> &rhs) {
        return _simd_f32x4_mul(lhs.value, rhs.value);
    }

    inline simd<float, 4> operator/(const simd<float, 4> &lhs, const simd<float, 4> &rhs) {
        return _simd_f32x4_div(lhs.value, rhs.value);
    }

    inline simd<float, 4> max_float32x4(const simd<float, 4> &lhs, const simd<float, 4> &rhs) {
        return _simd_f32x4_max(lhs.value, rhs.value);
    }

    inline simd<float, 4> min_float32x4(const simd<float, 4> &lhs, const simd<float, 4> &rhs) {
        return _simd_f32x4_min(lhs.value, rhs.value);
    }

    inline void transposex4x4(simd<float, 4> &q0, simd<float, 4> &q1, simd<float, 4> &q2, simd<float, 4> &q3) {
        return _simd_f32x4_transpose4x4(q0.value, q1.value, q2.value, q3.value);
    }

    inline simd<float, 4> fmadd(const simd<float, 4> &q0, const simd<float, 4> &q1, const simd<float, 4> &q2) {
        return _simd_f32x4_fmadd(q0.value, q1.value, q2.value);
    }

    template<>
    class simd<float, 8> : public simd_base<float, 8> {
    public:
        using self = simd;
        using type = _simd_f32x4x2;

        type value;

        simd() = default;

        simd(const type &value) : value(value) {}

        simd(base a) : simd(a, a, a, a, a, a, a, a) {}

        simd(int a) : simd(base(a)) {}

        simd(const base *p) : value(_simd_f32x4x2_load(p)) {}

        simd(base a, base b, base c, base d, base e, base f, base g, base h) : value(_simd_f32x4x2_set(a, b, c, d, e, f, g, h)) {}

        void store(base *p) const { _simd_f32x4x2_store(p, value); }
    };

    inline simd<float, 8> operator+(const simd<float, 8> &lhs, const simd<float, 8> &rhs) {
        return _simd_f32x4x2_add(lhs.value, rhs.value);
    }

    inline simd<float, 8> operator-(const simd<float, 8> &lhs, const simd<float, 8> &rhs) {
        return _simd_f32x4x2_sub(lhs.value, rhs.value);
    }

    inline simd<float, 8> operator*(const simd<float, 8> &lhs, const simd<float, 8> &rhs) {
        return _simd_f32x4x2_mul(lhs.value, rhs.value);
    }

    inline simd<float, 8> operator/(const simd<float, 8> &lhs, const simd<float, 8> &rhs) {
        return _simd_f32x4x2_div(lhs.value, rhs.value);
    }

    inline simd<float, 8> fmadd(const simd<float, 8> &q0, const simd<float, 8> &q1, const simd<float, 8> &q2) {
        return _simd_f32x4x2_fmadd(q0.value, q1.value, q2.value);
    }

    template<>
    class simd<int32_t, 4> : public simd_base<int32_t, 4> {
    public:
        using self = simd;
        using type = _simd_int32x4;

        type value;

        simd() = default;

        simd(type value) : value(value) {}

        simd(base a) : simd(a, a, a, a) {}

        simd(const base *p) : value(_simd_int32x4_load(p)) {}

        simd(base a, base b, base c, base d) : value(_simd_int32x4_set(a, b, c, d)) {}

        void store(base *p) const { _simd_int32x4_store(p, value); }
    };

    inline simd<int32_t, 4> operator+(const simd<int32_t, 4> &lhs, const simd<int32_t, 4> &rhs) {
        return _simd_int32x4_add(lhs.value, rhs.value);
    }

    inline simd<int32_t, 4> operator-(const simd<int32_t, 4> &lhs, const simd<int32_t, 4> &rhs) {
        return _simd_int32x4_sub(lhs.value, rhs.value);
    }

    template<>
    class simd<int32_t, 8> : public simd_base<int32_t, 8> {
    public:
        using self = simd;
        using type = _simd_int32x4x2;

        type value;

        simd() = default;

        simd(const type &value) : value(value) {}

        simd(base a) : simd(a, a, a, a, a, a, a, a) {}

        simd(const base *p) : value(_simd_int32x4x2_load(p)) {}

        simd(base a, base b, base c, base d, base e, base f, base g, base h) :
            value(_simd_int32x4x2_set(a, b, c, d, e, f, g, h)) {}

        void store(base *p) const { _simd_int32x4x2_store(p, value); }
    };

    inline simd<int32_t, 8> operator+(const simd<int32_t, 8> &lhs, const simd<int32_t, 8> &rhs) {
        return _simd_int32x4x2_add(lhs.value, rhs.value);
    }

    inline simd<int32_t, 8> operator-(const simd<int32_t, 8> &lhs, const simd<int32_t, 8> &rhs) {
        return _simd_int32x4x2_sub(lhs.value, rhs.value);
    }

    //cast
    inline int32x4x2 floatx4x2_to_int32x4x2(const float32x4x2 &lhs) {
        return _simd_floatx4x2_to_int32x4x2(lhs.value);
    }

    inline float32x4x2 intx4x2_to_float32x4x2(const int32x4x2 &lhs) {
        return _simd_intx4x2_to_float32x4x2(lhs.value);
    }

    inline float32x4 broadcast2float32x4(const float* src) {
        return _simd_broadcast2float32x4(src);
    }

    inline float32x4x2 broadcast2float32x4x2(const float* src) {
        return _simd_broadcast2float32x4x2(src);
    }

}

#endif
