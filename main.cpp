#include "CbrtPs.hpp"
#include <stdio.h>
#include <array>
#include <cmath>

int main()
{
	// Array with test inputs
	std::array<float, 4> vals{ 0.01234f, 1.12f, 3, -0.0f };

	// Compute reference result with the standard library
	std::array<float, 4> stdlib;
	for( int i = 0; i < 4; i++ )
		stdlib[ i ] = std::cbrtf( vals[ i ] );

	// Compute with simd::cbrt_ps
	__m128 vec = _mm_loadu_ps( vals.data() );
	vec = simd::cbrt_ps( vec );
	std::array<float, 4> my;
	_mm_storeu_ps( my.data(), vec );

	// Print these 12 numbers from 3 arrays
	printf( "x\tmy\tstdlib\n" );
	for( int i = 0; i < 4; i++ )
		printf( "%f\t%f\t%f\n", vals[ i ], my[ i ], stdlib[ i ] );
	return 0;
}