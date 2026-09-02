// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC

//PTX asm
//"volatile" is important
#define add_cc_64(res, a, b)			asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b)  );
#define addc_64(res, a, b)				asm volatile ("addc.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b));
#define addc_cc_64(res, a, b)			asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b)  );

#define add_cc_32(res, a, b)			asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b)  );
#define addc_32(res, a, b)				asm volatile ("addc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
#define addc_cc_32(res, a, b)			asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b)  );

#define sub_cc_64(res, a, b)			asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b) );
#define subc_cc_64(res, a, b)			asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b)  );
#define subc_64(res, a, b)				asm volatile ("subc.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b));

#define sub_cc_32(res, a, b)			asm volatile ("sub.cc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b) );
#define subc_cc_32(res, a, b)			asm volatile ("subc.cc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b)  );
#define subc_32(res, a, b)				asm volatile ("subc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));

#define mad_lo_cc_32(res, a, b, c)		asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
#define madc_lo_32(res, a, b, c)		asm volatile ("madc.lo.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
#define madc_lo_cc_32(res, a, b, c)		asm volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
#define madc_hi_cc_32(res, a, b, c)		asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));

#define mul_wide_32(res, a, b)			asm volatile ("mul.wide.u32 %0, %1, %2;" : "=l"(res) : "r"(a), "r"(b));

//P-related constants
#define P_0			0xFFFFFFFEFFFFFC2Full
#define P_123		0xFFFFFFFFFFFFFFFFull
#define P_INV32		0x000003D1

#define Add192to192(res, val) { \
  add_cc_64((res)[0], (res)[0], (val)[0]); \
  addc_cc_64((res)[1], (res)[1], (val)[1]); \
  addc_64((res)[2], (res)[2], (val)[2]); }

#define Sub192from192(res, val) { \
  sub_cc_64((res)[0], (res)[0], (val)[0]); \
  subc_cc_64((res)[1], (res)[1], (val)[1]); \
  subc_64((res)[2], (res)[2], (val)[2]); }

#define Copy_int4_x2(dst, src) {\
  ((int4*)(dst))[0] = ((int4*)(src))[0]; \
  ((int4*)(dst))[1] = ((int4*)(src))[1]; }

//mul 256bit by 0x1000003D1
__device__ __forceinline__ void mul_256_by_P0inv(u32* res, u32* val)
{
	u64 tmp64[7];
	u32* tmp = (u32*)tmp64;
	mul_wide_32(*(u64*)res, val[0], P_INV32);
	mul_wide_32(tmp64[0], val[1], P_INV32);
	mul_wide_32(tmp64[1], val[2], P_INV32);
	mul_wide_32(tmp64[2], val[3], P_INV32);
	mul_wide_32(tmp64[3], val[4], P_INV32);
	mul_wide_32(tmp64[4], val[5], P_INV32);
	mul_wide_32(tmp64[5], val[6], P_INV32);
	mul_wide_32(tmp64[6], val[7], P_INV32);

	add_cc_32(res[1], res[1], tmp[0]);
	addc_cc_32(res[2], tmp[1], tmp[2]);
	addc_cc_32(res[3], tmp[3], tmp[4]);
	addc_cc_32(res[4], tmp[5], tmp[6]);
	addc_cc_32(res[5], tmp[7], tmp[8]);
	addc_cc_32(res[6], tmp[9], tmp[10]);
	addc_cc_32(res[7], tmp[11], tmp[12]);
	addc_32(res[8], tmp[13], 0); //t[13] cannot be MAX_UINT so we wont have carry here for r[9]

	add_cc_32(res[1], res[1], val[0]);
	addc_cc_32(res[2], res[2], val[1]);
	addc_cc_32(res[3], res[3], val[2]);
	addc_cc_32(res[4], res[4], val[3]);
	addc_cc_32(res[5], res[5], val[4]);
	addc_cc_32(res[6], res[6], val[5]);
	addc_cc_32(res[7], res[7], val[6]);
	addc_cc_32(res[8], res[8], val[7]);
	addc_32(res[9], 0, 0);
}

__device__ __forceinline__ void add_288(u32* res, u32* val1, u32* val2)
{
	add_cc_32(res[0], val1[0], val2[0]);
	addc_cc_32(res[1], val1[1], val2[1]);
	addc_cc_32(res[2], val1[2], val2[2]);
	addc_cc_32(res[3], val1[3], val2[3]);
	addc_cc_32(res[4], val1[4], val2[4]);
	addc_cc_32(res[5], val1[5], val2[5]);
	addc_cc_32(res[6], val1[6], val2[6]);
	addc_cc_32(res[7], val1[7], val2[7]);
	addc_32(res[8], val1[8], val2[8]);
}

__device__ __forceinline__ void neg_288(u32* res)
{
	sub_cc_32(res[0], 0, res[0]);
	subc_cc_32(res[1], 0, res[1]);
	subc_cc_32(res[2], 0, res[2]);
	subc_cc_32(res[3], 0, res[3]);
	subc_cc_32(res[4], 0, res[4]);
	subc_cc_32(res[5], 0, res[5]);
	subc_cc_32(res[6], 0, res[6]);
	subc_cc_32(res[7], 0, res[7]);
	subc_32(res[8], 0, res[8]);
}

__device__ __forceinline__ void mul_288_by_i32(u32* res, u32* val288, int ival32)
{
	u32 val32 = abs(ival32);
	u64 tmp64[4];
	u32* tmp = (u32*)tmp64;
	u64* r32 = (u64*)res; 
	mul_wide_32(r32[0], val288[0], val32);
	mul_wide_32(r32[1], val288[2], val32);
	mul_wide_32(r32[2], val288[4], val32);
	mul_wide_32(r32[3], val288[6], val32);
	mul_wide_32(tmp64[0], val288[1], val32);
	mul_wide_32(tmp64[1], val288[3], val32);
	mul_wide_32(tmp64[2], val288[5], val32);
	mul_wide_32(tmp64[3], val288[7], val32);

	add_cc_32(res[1], res[1], tmp[0]);
	addc_cc_32(res[2], res[2], tmp[1]);
	addc_cc_32(res[3], res[3], tmp[2]);
	addc_cc_32(res[4], res[4], tmp[3]);
	addc_cc_32(res[5], res[5], tmp[4]);
	addc_cc_32(res[6], res[6], tmp[5]);
	addc_cc_32(res[7], res[7], tmp[6]);
	madc_lo_32(res[8], val288[8], val32, tmp[7]);

	if (ival32 < 0)
		neg_288(res);
}

__device__ __forceinline__ void set_288_i32(u32* res, int val)
{
	res[0] = val;
	res[1] = (val < 0) ? 0xFFFFFFFF : 0;
	res[2] = res[1];
	res[3] = res[1];
	res[4] = res[1];
	res[5] = res[1];
	res[6] = res[1];
	res[7] = res[1];
	res[8] = res[1];
}

//mul P by 32bit, get 288bit result
__device__ __forceinline__ void mul_P_by_32(u32* res, u32 val)
{
	__align__(8) u32 tmp[3];
	mul_wide_32(*(u64*)tmp, val, P_INV32);
	add_cc_32(tmp[1], tmp[1], val);
	addc_32(tmp[2], 0, 0);

	sub_cc_32(res[0], 0, tmp[0]);
	subc_cc_32(res[1], 0, tmp[1]);
	subc_cc_32(res[2], 0, tmp[2]);
	subc_cc_32(res[3], 0, 0);
	subc_cc_32(res[4], 0, 0);
	subc_cc_32(res[5], 0, 0);
	subc_cc_32(res[6], 0, 0);
	subc_cc_32(res[7], 0, 0);
	subc_32(res[8], val, 0);
}

__device__ __forceinline__ void shiftR_288_by_30(u32* res)
{
	res[0] = __funnelshift_r(res[0], res[1], 30);
	res[1] = __funnelshift_r(res[1], res[2], 30);
	res[2] = __funnelshift_r(res[2], res[3], 30);
	res[3] = __funnelshift_r(res[3], res[4], 30);
	res[4] = __funnelshift_r(res[4], res[5], 30);
	res[5] = __funnelshift_r(res[5], res[6], 30);
	res[6] = __funnelshift_r(res[6], res[7], 30);
	res[7] = __funnelshift_r(res[7], res[8], 30);
	res[8] = ((int)res[8]) >> 30;
}

__device__ __forceinline__ void add_288_P(u32* res)
{
	add_cc_32(res[0], res[0], 0xFFFFFC2F);
	addc_cc_32(res[1], res[1], 0xFFFFFFFE);
	addc_cc_32(res[2], res[2], 0xFFFFFFFF);
	addc_cc_32(res[3], res[3], 0xFFFFFFFF);
	addc_cc_32(res[4], res[4], 0xFFFFFFFF);
	addc_cc_32(res[5], res[5], 0xFFFFFFFF);
	addc_cc_32(res[6], res[6], 0xFFFFFFFF);
	addc_cc_32(res[7], res[7], 0xFFFFFFFF);
	addc_32(res[8], res[8], 0);
}

__device__ __forceinline__ void sub_288_P(u32* res)
{
	sub_cc_32(res[0], res[0], 0xFFFFFC2F);
	subc_cc_32(res[1], res[1], 0xFFFFFFFE);
	subc_cc_32(res[2], res[2], 0xFFFFFFFF);
	subc_cc_32(res[3], res[3], 0xFFFFFFFF);
	subc_cc_32(res[4], res[4], 0xFFFFFFFF);
	subc_cc_32(res[5], res[5], 0xFFFFFFFF);
	subc_cc_32(res[6], res[6], 0xFFFFFFFF);
	subc_cc_32(res[7], res[7], 0xFFFFFFFF);
	subc_32(res[8], res[8], 0);
}

#define APPLY_DIV_SHIFT()	matrix[0] <<= index; matrix[1] <<= index; kbnt -= index; _val >>= index;  
#define DO_INV_STEP()		{kbnt = -kbnt; int tmp = -_modp; _modp = _val; _val = tmp; tmp = -matrix[0]; \
							matrix[0] = matrix[2]; matrix[2] = tmp; tmp = -matrix[1]; matrix[1] = matrix[3]; matrix[3] = tmp;}

// https://tches.iacr.org/index.php/TCHES/article/download/8298/7648/4494
//a bit tricky
//res must be at least 288bits
__device__ __forceinline__ void InvModP(u32* res)
{
	int matrix[4], _val, _modp, index, cnt, mx, kbnt;
	__align__(8) u32 modp[9];
	__align__(8) u32 val[9];
	__align__(8) u32 a[9];
	__align__(8) u32 tmp[4][9+1]; //+1 because we need alignment 64bit for tmp[>0]

	((u64*)modp)[0] = P_0;
	((u64*)modp)[1] = P_123;
	((u64*)modp)[2] = P_123;
	((u64*)modp)[3] = P_123;
	modp[8] = 0;
	res[8] = 0;
	val[0] = res[0]; val[1] = res[1]; val[2] = res[2]; val[3] = res[3];
	val[4] = res[4]; val[5] = res[5]; val[6] = res[6]; val[7] = res[7];
	val[8] = 0;
	matrix[0] = matrix[3] = 1;
	matrix[1] = matrix[2] = 0;
	kbnt = -1;
	_val = (int)res[0];
	_modp = (int)P_0;
	index = __ffs(_val | 0x40000000) - 1;
	APPLY_DIV_SHIFT();
	cnt = 30 - index;
	while (cnt > 0)
	{
		if (kbnt < 0)
			DO_INV_STEP();
		mx = (kbnt + 1 < cnt) ? 31 - kbnt : 32 - cnt;
		i32 mul = (-_modp * _val) & 7;
		mul &= 0xFFFFFFFF >> mx;
		_val += _modp * mul;
		matrix[2] += matrix[0] * mul;
		matrix[3] += matrix[1] * mul;
		index = __ffs(_val | (1 << cnt)) - 1;
		APPLY_DIV_SHIFT();
		cnt -= index;
	}
	mul_288_by_i32(tmp[0], modp, matrix[0]);
	mul_288_by_i32(tmp[1], val, matrix[1]);
	mul_288_by_i32(tmp[2], modp, matrix[2]);
	mul_288_by_i32(tmp[3], val, matrix[3]);
	add_288(modp, tmp[0], tmp[1]);
	shiftR_288_by_30(modp);
	add_288(val, tmp[2], tmp[3]);
	shiftR_288_by_30(val);
	set_288_i32(tmp[1], matrix[1]);
	set_288_i32(tmp[3], matrix[3]);
	mul_P_by_32(res, (tmp[1][0] * 0xD2253531) & 0x3FFFFFFF);
	add_288(res, res, tmp[1]);
	shiftR_288_by_30(res);
	mul_P_by_32(a, (tmp[3][0] * 0xD2253531) & 0x3FFFFFFF);
	add_288(a, a, tmp[3]);
	shiftR_288_by_30(a);
	while (1)
	{
		matrix[0] = matrix[3] = 1;
		matrix[1] = matrix[2] = 0;
		_val = val[0];
		_modp = modp[0];
		index = __ffs(_val | 0x40000000) - 1;
		APPLY_DIV_SHIFT();
		cnt = 30 - index;
		while (cnt > 0)
		{
			if (kbnt < 0)
				DO_INV_STEP();
			mx = (kbnt + 1 < cnt) ? 31 - kbnt : 32 - cnt;
			i32 mul = (-_modp * _val) & 7;
			mul &= 0xFFFFFFFF >> mx;
			_val += _modp * mul;
			matrix[2] += matrix[0] * mul;
			matrix[3] += matrix[1] * mul;
			index = __ffs(_val | (1 << cnt)) - 1;
			APPLY_DIV_SHIFT();
			cnt -= index;
		}
		mul_288_by_i32(tmp[0], modp, matrix[0]);
		mul_288_by_i32(tmp[1], val, matrix[1]);
		mul_288_by_i32(tmp[2], modp, matrix[2]);
		mul_288_by_i32(tmp[3], val, matrix[3]);
		add_288(modp, tmp[0], tmp[1]);
		shiftR_288_by_30(modp);
		add_288(val, tmp[2], tmp[3]);
		shiftR_288_by_30(val);
		mul_288_by_i32(tmp[0], res, matrix[0]);
		mul_288_by_i32(tmp[1], a, matrix[1]);

		if ((val[0] | val[1] | val[2] | val[3] | val[4] | val[5] | val[6] | val[7]) == 0)
			break;

		mul_288_by_i32(tmp[2], res, matrix[2]);
		mul_288_by_i32(tmp[3], a, matrix[3]);
		mul_P_by_32(res, ((tmp[0][0] + tmp[1][0]) * 0xD2253531) & 0x3FFFFFFF);
		add_288(res, res, tmp[0]);
		add_288(res, res, tmp[1]);
		shiftR_288_by_30(res);
		mul_P_by_32(a, ((tmp[2][0] + tmp[3][0]) * 0xD2253531) & 0x3FFFFFFF);
		add_288(a, a, tmp[2]);
		add_288(a, a, tmp[3]);	
		shiftR_288_by_30(a);
	}
	mul_P_by_32(res, ((tmp[0][0] + tmp[1][0]) * 0xD2253531) & 0x3FFFFFFF);
	add_288(res, res, tmp[0]);
	add_288(res, res, tmp[1]);
	shiftR_288_by_30(res);
	if ((int)modp[8] < 0)
		neg_288(res);	
	while ((int)res[8] < 0)
		add_288_P(res);
	while ((int)res[8] > 0)
		sub_288_P(res);
}

