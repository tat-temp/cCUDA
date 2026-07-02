// cr_prodD: 256x256 -> 512-bit unsigned multiply, little-endian 32-bit limbs.
// Clean-room product-scanning (Comba) with a 3-word (c2:c1:c0) accumulator in
// inline PTX. Each mul.wide.u32 partial a[i]*b[j] with i+j==k is folded into the
// accumulator via mad.lo.cc.u32 / madc.hi.cc.u32 / addc.u32, then out[k]=c0 and
// the accumulator is shifted down (c0=c1, c1=c2, c2=0). Register-frugal: only the
// 3-word accumulator plus the two operand arrays are live.
__device__ __forceinline__ void cr_prodD(uint32_t out[16],
                                          const uint32_t a[8],
                                          const uint32_t b[8])
{
    uint32_t c0 = 0u, c1 = 0u, c2 = 0u;

    // MAC(i,j): accumulate a[i]*b[j] into (c2:c1:c0).
    //   c0 += lo(a*b)             (add.cc -> carry)
    //   c1 += hi(a*b) + carry     (madc.hi.cc -> carry)
    //   c2 += carry
    // Implemented as a single asm block so the hardware carry flag chains cleanly.
    #define MAC(i,j) \
        asm volatile( \
            "mad.lo.cc.u32   %0, %3, %4, %0;\n\t" \
            "madc.hi.cc.u32  %1, %3, %4, %1;\n\t" \
            "addc.u32        %2, %2, 0;\n\t" \
            : "+r"(c0), "+r"(c1), "+r"(c2) \
            : "r"(a[i]), "r"(b[j]))

    // COL_DONE(k): emit out[k] = c0, shift accumulator down.
    #define COL_DONE(k) do { out[k] = c0; c0 = c1; c1 = c2; c2 = 0u; } while(0)

    // Column 0: i+j = 0
    MAC(0,0);
    COL_DONE(0);

    // Column 1: i+j = 1
    MAC(0,1); MAC(1,0);
    COL_DONE(1);

    // Column 2
    MAC(0,2); MAC(1,1); MAC(2,0);
    COL_DONE(2);

    // Column 3
    MAC(0,3); MAC(1,2); MAC(2,1); MAC(3,0);
    COL_DONE(3);

    // Column 4
    MAC(0,4); MAC(1,3); MAC(2,2); MAC(3,1); MAC(4,0);
    COL_DONE(4);

    // Column 5
    MAC(0,5); MAC(1,4); MAC(2,3); MAC(3,2); MAC(4,1); MAC(5,0);
    COL_DONE(5);

    // Column 6
    MAC(0,6); MAC(1,5); MAC(2,4); MAC(3,3); MAC(4,2); MAC(5,1); MAC(6,0);
    COL_DONE(6);

    // Column 7
    MAC(0,7); MAC(1,6); MAC(2,5); MAC(3,4); MAC(4,3); MAC(5,2); MAC(6,1); MAC(7,0);
    COL_DONE(7);

    // Column 8
    MAC(1,7); MAC(2,6); MAC(3,5); MAC(4,4); MAC(5,3); MAC(6,2); MAC(7,1);
    COL_DONE(8);

    // Column 9
    MAC(2,7); MAC(3,6); MAC(4,5); MAC(5,4); MAC(6,3); MAC(7,2);
    COL_DONE(9);

    // Column 10
    MAC(3,7); MAC(4,6); MAC(5,5); MAC(6,4); MAC(7,3);
    COL_DONE(10);

    // Column 11
    MAC(4,7); MAC(5,6); MAC(6,5); MAC(7,4);
    COL_DONE(11);

    // Column 12
    MAC(5,7); MAC(6,6); MAC(7,5);
    COL_DONE(12);

    // Column 13
    MAC(6,7); MAC(7,6);
    COL_DONE(13);

    // Column 14
    MAC(7,7);
    COL_DONE(14);

    // Column 15: only the top carry remains
    out[15] = c0;

    #undef MAC
    #undef COL_DONE
}
