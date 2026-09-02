// Host emulation of the PTX primitives the field math is written against, so the shipped
// device code runs under a plain C++ compiler. CC.CF is one sticky bit: .cc ops write it,
// addc/madc read it, plain addc reads only. Results stage through a temp because the
// shipped code aliases destination and source (e.g. mad_lo_cc(e[0], a[0], b[0], e[0])).
#pragma once

#include <cstdint>

typedef uint32_t u32;
typedef uint64_t u64;

#define __device__
#define __forceinline__ inline

static unsigned g_cf = 0;

#define add_cc_32(r, a, b)      do { u64 _t = (u64)(u32)(a) + (u32)(b);                 (r) = (u32)_t; g_cf = (unsigned)(_t >> 32); } while (0)
#define addc_cc_32(r, a, b)     do { u64 _t = (u64)(u32)(a) + (u32)(b) + g_cf;          (r) = (u32)_t; g_cf = (unsigned)(_t >> 32); } while (0)
#define addc_32(r, a, b)        do { u64 _t = (u64)(u32)(a) + (u32)(b) + g_cf;          (r) = (u32)_t; } while (0)

#define add_cc_64(r, a, b)      do { u64 _a = (u64)(a), _b = (u64)(b), _s = _a + _b;              (r) = _s; g_cf = (_s < _a); } while (0)
#define addc_cc_64(r, a, b)     do { u64 _a = (u64)(a), _b = (u64)(b), _c = g_cf, _s = _a + _b + _c; \
                                     g_cf = (_s < _a) || (_c && _s == _a); (r) = _s; } while (0)
#define addc_64(r, a, b)        do { u64 _a = (u64)(a), _b = (u64)(b), _c = g_cf;                 (r) = _a + _b + _c; } while (0)

#define mul_wide_32(r, a, b)    do { (r) = (u64)(u32)(a) * (u32)(b); } while (0)
#define mul_lo_32(r, a, b)      do { (r) = (u32)((u32)(a) * (u32)(b)); } while (0)

#define mad_lo_cc_32(r, a, b, c)   do { u64 _p = (u64)(u32)(a) * (u32)(b); \
                                        u64 _t = (u64)(u32)_p + (u32)(c);            (r) = (u32)_t; g_cf = (unsigned)(_t >> 32); } while (0)
#define madc_lo_cc_32(r, a, b, c)  do { u64 _p = (u64)(u32)(a) * (u32)(b); \
                                        u64 _t = (u64)(u32)_p + (u32)(c) + g_cf;     (r) = (u32)_t; g_cf = (unsigned)(_t >> 32); } while (0)
#define madc_hi_cc_32(r, a, b, c)  do { u64 _p = (u64)(u32)(a) * (u32)(b); \
                                        u64 _t = (u64)(u32)(_p >> 32) + (u32)(c) + g_cf; (r) = (u32)_t; g_cf = (unsigned)(_t >> 32); } while (0)

#define P_0     0xFFFFFFFEFFFFFC2Full
#define P_123   0xFFFFFFFFFFFFFFFFull
#define P_INV32 0x000003D1
