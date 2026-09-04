import sys
lines=[]
def emit(s): lines.append("\t"+s)

def col_cross(c):   # i<j, i+j==c
    return [(i, c-i) for i in range(max(0,c-7), (c+1)//2) if c-i<=7]

def side(parity, arr):
    cols=[c for c in range(0,15) if c%2==parity and col_cross(c)]
    maxn=max(len(col_cross(c)) for c in cols)
    blocks=[]
    for t in range(maxn):
        chain=[(c,col_cross(c)[t]) for c in cols if len(col_cross(c))>t]
        if not chain: continue
        rows=[]
        first=True
        for (c,(i,j)) in chain:
            base = c if parity==0 else c-1
            lo = "mad_lo_cc" if first else "madc_lo_cc"
            rows.append(f"{lo}({arr}[{base}], a[{i}], a[{j}], {arr}[{base}]);")
            rows.append(f"madc_hi_cc({arr}[{base+1}], a[{i}], a[{j}], {arr}[{base+1}]);")
            first=False
        lb = chain[-1][0] if parity==0 else chain[-1][0]-1
        if lb+2 <= 15: rows.append(f"addc_32({arr}[{lb+2}], {arr}[{lb+2}], 0);")
        blocks.append((lb, t, rows))
    # Rows are emitted NARROWEST-REACH FIRST (ascending last_base).
    #
    # A row-end fixup is addc_32 == addc.u32 (RCGpuUtils.h:13): it CONSUMES the chain's carry
    # flag and emits none. Emitting the widest row first -- the original order -- let a later,
    # narrower row's fixup land on a word ALREADY holding product data, so the add could wrap and
    # the carry was silently lost, leaving the 512-bit product short by 2^(32*k). Witness for the
    # old order: mul512_split(A,B) == A*B - 2^480 at
    #   A = 0x00000003cffa6cddf963a7efe00111e5d29dc5dfcf1da1100cc36d8c77863fe5
    #   B = 0x55555555fa83ada4a2121ac5f689a4a5ffda03368c6e90373020da5c6a46721a
    # It needs a word to be exactly 0xFFFFFFFF (~2^-32 per site), which is why 5,000,064 random
    # pairs never hit it.
    #
    # Ascending last_base makes every fixup target hold nothing but earlier rows' fixup carries
    # (at most one each), so it provably cannot wrap. Rows are independent accumulations into the
    # same words, so this is a pure permutation -- the emitted instruction multiset is unchanged.
    blocks.sort(key=lambda b: (b[0], b[1]))
    for _, _, rows in blocks:
        for r in rows: emit(r)
    return len(blocks)

out=["// generated: split-column fused-MAC 256-bit square -> 512",
     "__device__ __forceinline__ void sqr512_split(uint32_t* p, const uint32_t* a)",
     "{", "\tuint32_t e[16], o[16];", "\t#pragma unroll",
     "\tfor (int i = 0; i < 16; i++) { e[i] = 0; o[i] = 0; }"]
ne=side(0,"e"); out+=lines; lines=[]
no=side(1,"o"); out+=lines; lines=[]
out.append("\t// p = 2*(e + (o << 32))")
out.append("\tadd_cc_32(p[1], e[1], o[0]);")
for k in range(2,16): out.append(f"\taddc_cc_32(p[{k}], e[{k}], o[{k-1}]);")
out.append("\tp[0] = e[0];")
out.append("\tadd_cc_32(p[0], p[0], p[0]);")
for k in range(1,16): out.append(f"\taddc_cc_32(p[{k}], p[{k}], p[{k}]);")
out.append("\t// += diagonal squares a[i]^2 at column 2i")
out.append("\tmad_lo_cc(p[0], a[0], a[0], p[0]);")
out.append("\tmadc_hi_cc(p[1], a[0], a[0], p[1]);")
for i in range(1,8):
    out.append(f"\tmadc_lo_cc(p[{2*i}], a[{i}], a[{i}], p[{2*i}]);")
    out.append(f"\tmadc_hi_cc(p[{2*i+1}], a[{i}], a[{i}], p[{2*i+1}]);")
out.append("}")
sys.stderr.write(f"chains even={ne} odd={no}\n")
print("\n".join(out))
