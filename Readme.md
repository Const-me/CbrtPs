This small project implements a single function, `cbrt_ps`.<br/>
The function computes cubic root of 4 FP32 values in a vector register.

The implementation requires SSE 4.1 instruction set, and can optionally use AVX 1 if available.
So far, no NEON version is there.

The implementation is OS agnostic, should compile for all of them as long as the target CPU supports SSE 4.1.<br/>
Tested with Visual Studio 2022 on Windows 10, and GCC 7.4 on Linux.

The implementation can be trivially generalized to 32-bytes AVX vectors.

## Usage

Copy-paste `CbrtPs.hpp` header into your project, include the header into your source or header file\[s\].

As you see from the comment in that header, it comes with a copy paste friendly terms of MIT license.

## References

* The Halley’s refinement method is from “[In Search of a Fast Cube Root](https://web.archive.org/web/20131227144655/http://metamerist.com/cbrt/cbrt.htm)” article by [metamerist](http://metamerist.blogspot.com/)

* Denormals handling and initial estimate ported from FreeBSD’s [version](https://github.com/freebsd/freebsd-src/blob/master/lib/msun/src/s_cbrtf.c) of `cbrtf()` library function

* Integer division by 3 back-ported from assembly [made by clang 13](https://godbolt.org/z/c56Kar5aT)