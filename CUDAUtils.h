__host__ __forceinline__ void add256_u64(const uint64_t a[4], uint64_t b, uint64_t out[4]) {
    __uint128_t sum = (__uint128_t)a[0] + b;
    out[0] = (uint64_t)sum;
    uint64_t carry = (uint64_t)(sum >> 64);
    sum = (__uint128_t)a[1] + carry; out[1] = (uint64_t)sum; carry = (uint64_t)(sum >> 64);
    sum = (__uint128_t)a[2] + carry; out[2] = (uint64_t)sum; carry = (uint64_t)(sum >> 64);
    sum = (__uint128_t)a[3] + carry; out[3] = (uint64_t)sum; carry = (uint64_t)(sum >> 64);
}

__host__ __forceinline__ void add256(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    __uint128_t carry = 0;
    { __uint128_t s = (__uint128_t)a[0] + b[0] + carry; out[0] = (uint64_t)s; carry = s >> 64; }
    { __uint128_t s = (__uint128_t)a[1] + b[1] + carry; out[1] = (uint64_t)s; carry = s >> 64; }
    { __uint128_t s = (__uint128_t)a[2] + b[2] + carry; out[2] = (uint64_t)s; carry = s >> 64; }
    { __uint128_t s = (__uint128_t)a[3] + b[3] + carry; out[3] = (uint64_t)s; carry = s >> 64; }
}

__host__ __forceinline__ void sub256(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    uint64_t borrow = 0;
    { uint64_t bi = b[0] + borrow; if (a[0] < bi) { out[0] = (uint64_t)(((__uint128_t(1) << 64) + a[0]) - bi); borrow = 1; } else { out[0] = a[0] - bi; borrow = 0; } }
    { uint64_t bi = b[1] + borrow; if (a[1] < bi) { out[1] = (uint64_t)(((__uint128_t(1) << 64) + a[1]) - bi); borrow = 1; } else { out[1] = a[1] - bi; borrow = 0; } }
    { uint64_t bi = b[2] + borrow; if (a[2] < bi) { out[2] = (uint64_t)(((__uint128_t(1) << 64) + a[2]) - bi); borrow = 1; } else { out[2] = a[2] - bi; borrow = 0; } }
    { uint64_t bi = b[3] + borrow; if (a[3] < bi) { out[3] = (uint64_t)(((__uint128_t(1) << 64) + a[3]) - bi); borrow = 1; } else { out[3] = a[3] - bi; borrow = 0; } }
}

__host__ void divmod_256_by_u64(const uint64_t value[4], uint64_t divisor, uint64_t quotient[4], uint64_t &remainder) {
    remainder = 0;
    { __uint128_t cur = (__uint128_t(remainder) << 64) | value[3]; quotient[3] = (uint64_t)(cur / divisor); remainder = (uint64_t)(cur % divisor); }
    { __uint128_t cur = (__uint128_t(remainder) << 64) | value[2]; quotient[2] = (uint64_t)(cur / divisor); remainder = (uint64_t)(cur % divisor); }
    { __uint128_t cur = (__uint128_t(remainder) << 64) | value[1]; quotient[1] = (uint64_t)(cur / divisor); remainder = (uint64_t)(cur % divisor); }
    { __uint128_t cur = (__uint128_t(remainder) << 64) | value[0]; quotient[0] = (uint64_t)(cur / divisor); remainder = (uint64_t)(cur % divisor); }
}

bool hexToLE64(const std::string& h_in, uint64_t w[4]) {
    std::string h = h_in;
    if (h.size() >= 2 && (h[0] == '0') && (h[1] == 'x' || h[1] == 'X')) h = h.substr(2);
    if (h.size() > 64) return false;
    if (h.size() < 64) h = std::string(64 - h.size(), '0') + h;
    if (h.size() != 64) return false;
    w[3] = std::stoull(h.substr( 0, 16), nullptr, 16);
    w[2] = std::stoull(h.substr(16, 16), nullptr, 16);
    w[1] = std::stoull(h.substr(32, 16), nullptr, 16);
    w[0] = std::stoull(h.substr(48, 16), nullptr, 16);
    return true;
}
bool hexToHash160(const std::string& h, uint8_t hash160[20]) {
    if (h.size() != 40) return false;
    hash160[ 0] = (uint8_t)std::stoul(h.substr( 0, 2), nullptr, 16);
    hash160[ 1] = (uint8_t)std::stoul(h.substr( 2, 2), nullptr, 16);
    hash160[ 2] = (uint8_t)std::stoul(h.substr( 4, 2), nullptr, 16);
    hash160[ 3] = (uint8_t)std::stoul(h.substr( 6, 2), nullptr, 16);
    hash160[ 4] = (uint8_t)std::stoul(h.substr( 8, 2), nullptr, 16);
    hash160[ 5] = (uint8_t)std::stoul(h.substr(10, 2), nullptr, 16);
    hash160[ 6] = (uint8_t)std::stoul(h.substr(12, 2), nullptr, 16);
    hash160[ 7] = (uint8_t)std::stoul(h.substr(14, 2), nullptr, 16);
    hash160[ 8] = (uint8_t)std::stoul(h.substr(16, 2), nullptr, 16);
    hash160[ 9] = (uint8_t)std::stoul(h.substr(18, 2), nullptr, 16);
    hash160[10] = (uint8_t)std::stoul(h.substr(20, 2), nullptr, 16);
    hash160[11] = (uint8_t)std::stoul(h.substr(22, 2), nullptr, 16);
    hash160[12] = (uint8_t)std::stoul(h.substr(24, 2), nullptr, 16);
    hash160[13] = (uint8_t)std::stoul(h.substr(26, 2), nullptr, 16);
    hash160[14] = (uint8_t)std::stoul(h.substr(28, 2), nullptr, 16);
    hash160[15] = (uint8_t)std::stoul(h.substr(30, 2), nullptr, 16);
    hash160[16] = (uint8_t)std::stoul(h.substr(32, 2), nullptr, 16);
    hash160[17] = (uint8_t)std::stoul(h.substr(34, 2), nullptr, 16);
    hash160[18] = (uint8_t)std::stoul(h.substr(36, 2), nullptr, 16);
    hash160[19] = (uint8_t)std::stoul(h.substr(38, 2), nullptr, 16);
    return true;
}
std::string formatHex256(const uint64_t limbs[4]) {
    std::ostringstream oss;
    oss << std::hex << std::uppercase << std::setfill('0');
    oss << std::setw(16) << limbs[3];
    oss << std::setw(16) << limbs[2];
    oss << std::setw(16) << limbs[1];
    oss << std::setw(16) << limbs[0];
    return oss.str();
}

// h5/target_w hold a hash160 as 5 little-endian 32-bit words (word i = bytes [4i..4i+3]).
static __device__ __forceinline__ bool hash160_prefix_equals(
    const uint32_t h5[5], uint32_t target_prefix)
{
    return h5[0] == target_prefix;
}

static __device__ __forceinline__ bool hash160_matches_full(
    const uint32_t h5[5], const uint32_t target_w[5])
{
    if (h5[0] != target_w[0]) return false;
    if (h5[1] != target_w[1]) return false;
    if (h5[2] != target_w[2]) return false;
    if (h5[3] != target_w[3]) return false;
    if (h5[4] != target_w[4]) return false;
    return true;
}

// вспомогательная: a (256-бит) >= b (u64)?
__device__ __forceinline__ bool ge256_u64(const uint64_t a[4], uint64_t b) {
    if (a[3] | a[2] | a[1]) return true;  // >= 2^64
    return a[0] >= b;
}

__device__ __forceinline__ void sub256_u64_inplace(uint64_t a[4], uint64_t dec) {
    uint64_t borrow = (a[0] < dec) ? 1ull : 0ull;
    a[0] = a[0] - dec;
    { uint64_t ai = a[1]; a[1] = ai - borrow; borrow = (ai < borrow) ? 1ull : 0ull; }
    { uint64_t ai = a[2]; a[2] = ai - borrow; borrow = (ai < borrow) ? 1ull : 0ull; }
    { uint64_t ai = a[3]; a[3] = ai - borrow; borrow = (ai < borrow) ? 1ull : 0ull; }
}

__device__ __forceinline__ unsigned long long warp_reduce_add_ull(unsigned long long v) {
    unsigned mask = 0xFFFFFFFFu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

static inline std::string human_bytes(double bytes) {
    static const char* u[]={"B","KB","MB","GB","TB","PB"};
    int k=0;
    while(bytes>=1024.0 && k<5){ bytes/=1024.0; ++k; }
    std::ostringstream o; o.setf(std::ios::fixed); o<<std::setprecision(bytes<10?2:1)<<bytes<<" "<<u[k];
    return o.str();
}

static inline long double ld_from_u256(const uint64_t v[4]) {
    return std::ldexp((long double)v[3],192) + std::ldexp((long double)v[2],128) + std::ldexp((long double)v[1],64) + (long double)v[0];
}

static inline std::string formatCompressedPubHex(const uint64_t Rx[4], const uint64_t Ry[4]) {
    uint8_t out[33];
    out[0] = (Ry[0] & 1ULL) ? 0x03 : 0x02;
    { uint64_t v = Rx[3]; out[ 1]=(uint8_t)(v>>56); out[ 2]=(uint8_t)(v>>48); out[ 3]=(uint8_t)(v>>40); out[ 4]=(uint8_t)(v>>32); out[ 5]=(uint8_t)(v>>24); out[ 6]=(uint8_t)(v>>16); out[ 7]=(uint8_t)(v>>8); out[ 8]=(uint8_t)(v>>0); }
    { uint64_t v = Rx[2]; out[ 9]=(uint8_t)(v>>56); out[10]=(uint8_t)(v>>48); out[11]=(uint8_t)(v>>40); out[12]=(uint8_t)(v>>32); out[13]=(uint8_t)(v>>24); out[14]=(uint8_t)(v>>16); out[15]=(uint8_t)(v>>8); out[16]=(uint8_t)(v>>0); }
    { uint64_t v = Rx[1]; out[17]=(uint8_t)(v>>56); out[18]=(uint8_t)(v>>48); out[19]=(uint8_t)(v>>40); out[20]=(uint8_t)(v>>32); out[21]=(uint8_t)(v>>24); out[22]=(uint8_t)(v>>16); out[23]=(uint8_t)(v>>8); out[24]=(uint8_t)(v>>0); }
    { uint64_t v = Rx[0]; out[25]=(uint8_t)(v>>56); out[26]=(uint8_t)(v>>48); out[27]=(uint8_t)(v>>40); out[28]=(uint8_t)(v>>32); out[29]=(uint8_t)(v>>24); out[30]=(uint8_t)(v>>16); out[31]=(uint8_t)(v>>8); out[32]=(uint8_t)(v>>0); }
    static const char* hexd="0123456789ABCDEF";
    std::string s; s.resize(66);
    s[ 0]=hexd[(out[ 0]>>4)&0xF]; s[ 1]=hexd[out[ 0]&0xF];
    s[ 2]=hexd[(out[ 1]>>4)&0xF]; s[ 3]=hexd[out[ 1]&0xF];
    s[ 4]=hexd[(out[ 2]>>4)&0xF]; s[ 5]=hexd[out[ 2]&0xF];
    s[ 6]=hexd[(out[ 3]>>4)&0xF]; s[ 7]=hexd[out[ 3]&0xF];
    s[ 8]=hexd[(out[ 4]>>4)&0xF]; s[ 9]=hexd[out[ 4]&0xF];
    s[10]=hexd[(out[ 5]>>4)&0xF]; s[11]=hexd[out[ 5]&0xF];
    s[12]=hexd[(out[ 6]>>4)&0xF]; s[13]=hexd[out[ 6]&0xF];
    s[14]=hexd[(out[ 7]>>4)&0xF]; s[15]=hexd[out[ 7]&0xF];
    s[16]=hexd[(out[ 8]>>4)&0xF]; s[17]=hexd[out[ 8]&0xF];
    s[18]=hexd[(out[ 9]>>4)&0xF]; s[19]=hexd[out[ 9]&0xF];
    s[20]=hexd[(out[10]>>4)&0xF]; s[21]=hexd[out[10]&0xF];
    s[22]=hexd[(out[11]>>4)&0xF]; s[23]=hexd[out[11]&0xF];
    s[24]=hexd[(out[12]>>4)&0xF]; s[25]=hexd[out[12]&0xF];
    s[26]=hexd[(out[13]>>4)&0xF]; s[27]=hexd[out[13]&0xF];
    s[28]=hexd[(out[14]>>4)&0xF]; s[29]=hexd[out[14]&0xF];
    s[30]=hexd[(out[15]>>4)&0xF]; s[31]=hexd[out[15]&0xF];
    s[32]=hexd[(out[16]>>4)&0xF]; s[33]=hexd[out[16]&0xF];
    s[34]=hexd[(out[17]>>4)&0xF]; s[35]=hexd[out[17]&0xF];
    s[36]=hexd[(out[18]>>4)&0xF]; s[37]=hexd[out[18]&0xF];
    s[38]=hexd[(out[19]>>4)&0xF]; s[39]=hexd[out[19]&0xF];
    s[40]=hexd[(out[20]>>4)&0xF]; s[41]=hexd[out[20]&0xF];
    s[42]=hexd[(out[21]>>4)&0xF]; s[43]=hexd[out[21]&0xF];
    s[44]=hexd[(out[22]>>4)&0xF]; s[45]=hexd[out[22]&0xF];
    s[46]=hexd[(out[23]>>4)&0xF]; s[47]=hexd[out[23]&0xF];
    s[48]=hexd[(out[24]>>4)&0xF]; s[49]=hexd[out[24]&0xF];
    s[50]=hexd[(out[25]>>4)&0xF]; s[51]=hexd[out[25]&0xF];
    s[52]=hexd[(out[26]>>4)&0xF]; s[53]=hexd[out[26]&0xF];
    s[54]=hexd[(out[27]>>4)&0xF]; s[55]=hexd[out[27]&0xF];
    s[56]=hexd[(out[28]>>4)&0xF]; s[57]=hexd[out[28]&0xF];
    s[58]=hexd[(out[29]>>4)&0xF]; s[59]=hexd[out[29]&0xF];
    s[60]=hexd[(out[30]>>4)&0xF]; s[61]=hexd[out[30]&0xF];
    s[62]=hexd[(out[31]>>4)&0xF]; s[63]=hexd[out[31]&0xF];
    s[64]=hexd[(out[32]>>4)&0xF]; s[65]=hexd[out[32]&0xF];
    return s;
}


