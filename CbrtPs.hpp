// The function in this source file computes a cubic root of 4 FP32 numbers in a vector register.
// The implementation requires SSE up to and including 4.1, can optionally use AVX1 when enabled with a macro.
// Copyright (c) 2022 Konstantin, http://const.me
// This source file is subject to MIT license
#pragma once
#include <xmmintrin.h>	// SSE 1
#include <emmintrin.h>	// SSE 2
#include <smmintrin.h>	// SSE 4.1
#ifdef __AVX__
#define USE_AVX 1
#include <immintrin.h>	// AVX
#else
#define USE_AVX 0
#endif

namespace simd
{
	namespace details
	{
		// Divide int32 lanes by 3; backported from assembly made by clang 13 automatic vectorizer: https://godbolt.org/z/c56Kar5aT
		inline __m128i div3_epi32( __m128i vec )
		{
			const __m128i mul = _mm_set1_epi32( 0x55555556 );
			__m128i tmp = _mm_shuffle_epi32( vec, _MM_SHUFFLE( 3, 3, 1, 1 ) );
			vec = _mm_mul_epi32( vec, mul );
			tmp = _mm_mul_epi32( tmp, mul );
			vec = _mm_shuffle_epi32( vec, _MM_SHUFFLE( 3, 3, 1, 1 ) );
			vec = _mm_blend_epi16( vec, tmp, 0b11001100 );
			vec = _mm_add_epi32( vec, _mm_srli_epi32( vec, 31 ) );
			return vec;
		}

		// Halley's refinement method, FP64 precision for optimal accuracy
		// https://web.archive.org/web/20131227144655/http://metamerist.com/cbrt/cbrt.htm
#if USE_AVX
		inline __m256d cbrtRefine( __m256d a, __m256d r )
		{
			__m256d a3 = _mm256_mul_pd( _mm256_mul_pd( a, a ), a ); // a^3
			__m256d tmp = _mm256_add_pd( a3, r );		// a^3 + r
			__m256d mul = _mm256_add_pd( r, tmp );		// a^3 + r + r
			__m256d div = _mm256_add_pd( a3, tmp );		// a^3 + a^3 + r

			mul = _mm256_mul_pd( mul, a );
			return _mm256_div_pd( mul, div );
		}
#else
		inline __m128d cbrtRefine( __m128d a, __m128d r )
		{
			__m128d a3 = _mm_mul_pd( _mm_mul_pd( a, a ), a ); // a^3
			__m128d tmp = _mm_add_pd( a3, r );		// a^3 + r
			__m128d mul = _mm_add_pd( r, tmp );		// a^3 + r + r
			__m128d div = _mm_add_pd( a3, tmp );	// a^3 + a^3 + r

			mul = _mm_mul_pd( mul, a );
			return _mm_div_pd( mul, div );
		}
#endif
	}

	// Compute cubic root of 4 FP32 numbers in the vector register
	inline __m128 cbrt_ps( __m128 x )
	{
		using namespace details;

		// Denormals handling and initial estimate are ported from there:
		// https://github.com/freebsd/freebsd-src/blob/master/lib/msun/src/s_cbrtf.c
		const __m128 signBit = _mm_set1_ps( -0.0f );
		__m128 sign = _mm_and_ps( x, signBit );
		__m128 abs = _mm_andnot_ps( signBit, x );

		__m128 isZero = _mm_cmpeq_ps( x, _mm_setzero_ps() );	// TRUE if the value is +-0.0f, we gonna return zero for these values

		__m128i i = _mm_castps_si128( abs );
		__m128i isNanOrInf = _mm_cmpgt_epi32( i, _mm_set1_epi32( 0x7F7FFFFF ) );	// TRUE if the input is INF or NAN, we gonna return (x+x) in that case

		constexpr int B1 = 709958130;	// B1 = (127-127.0/3-0.03306235651)*2**23
		constexpr int B2 = 642849266;	// B2 = (127-127.0/3-24/3-0.03306235651)*2**23

		// Optionally handle the denormals
		__m128i isSubnorm = _mm_cmplt_epi32( i, _mm_set1_epi32( 0x00800000 ) );

		// Integer constant to add
		__m128i B = _mm_blendv_epi8( _mm_set1_epi32( B1 ), _mm_set1_epi32( B2 ), isSubnorm );

		const __m128 _2e24 = _mm_castsi128_ps( _mm_set1_epi32( 0x4b800000 ) );	// 2^24
		__m128 t = _mm_mul_ps( x, _2e24 );
		t = _mm_andnot_ps( signBit, t );

		// For regular numbers, integer with absolute value of the float; for subnormals, more complicated expression with different estimate
		i = _mm_blendv_epi8( i, _mm_castps_si128( t ), isSubnorm );
		// Divide the integer by 3, and offset
		i = div3_epi32( i );
		i = _mm_add_epi32( i, B );

		// Apply the sign, this gives the correct estimate
		__m128 r = _mm_or_ps( _mm_castsi128_ps( i ), sign );

		// Refine the value.
		// Using FP64 precision because FP32 converges too slow, needs more than 2 refinement iterations.
		// Despite FP64 is slightly slower, and without AVX needs twice as many instructions, the reduced count of iteration still causes a small performance win overall.
#if USE_AVX
		// For AVX we use 32-byte SIMD vectors
		__m256d rd = _mm256_cvtps_pd( r );
		__m256d xd = _mm256_cvtps_pd( x );

		rd = cbrtRefine( rd, xd );
		rd = cbrtRefine( rd, xd );

		r = _mm256_cvtpd_ps( rd );
#else
		// For SSE we split these vectors into low/high slices, and compute things independently.
		__m128d rd1 = _mm_cvtps_pd( r );
		__m128d rd2 = _mm_cvtps_pd( _mm_movehl_ps( r, r ) );

		__m128d xd1 = _mm_cvtps_pd( x );
		__m128d xd2 = _mm_cvtps_pd( _mm_movehl_ps( x, x ) );

		rd1 = cbrtRefine( rd1, xd1 );
		rd2 = cbrtRefine( rd2, xd2 );
		rd1 = cbrtRefine( rd1, xd1 );
		rd2 = cbrtRefine( rd2, xd2 );

		r = _mm_movelh_ps( _mm_cvtpd_ps( rd1 ), _mm_cvtpd_ps( rd2 ) );
#endif

		// Produce the final result, i.e. handle zeros and NAN/INF
		r = _mm_blendv_ps( r, x, isZero );	// When the value was +-0, return the value itself
		r = _mm_blendv_ps( r, _mm_add_ps( x, x ), _mm_castsi128_ps( isNanOrInf ) );	// When the value was NAN or INF, return x+x
		return r;
	}
}