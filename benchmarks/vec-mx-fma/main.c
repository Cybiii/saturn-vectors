#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "rvv_mx.h"

extern size_t N;
size_t avl;
size_t vl;

#define IF0(t, f) f
#define IF1(t, f) t

#define TEST_DATA_INPUTS(type, name, rm) \
	extern type name ## a_ ## rm[] __attribute__((aligned(64))); \
	extern type name ## b_ ## rm[] __attribute__((aligned(64))); \
	extern type name ## c_ ## rm[] __attribute__((aligned(64)));

#define TEST_DATA(type, name) \
	TEST_DATA_INPUTS(type, name, rne) \
	TEST_DATA_INPUTS(type, name, rtz) \
	TEST_DATA_INPUTS(type, name, rdn) \
	TEST_DATA_INPUTS(type, name, rup) \
	TEST_DATA_INPUTS(type, name, rmm) \
	type *name ## a_; \
	type *name ## b_; \
	type *name ## c_;

#define TEST_OP(type, name, op) \
	extern type name ## _ ## op ## _rne [] __attribute__((aligned(64))); \
	extern type name ## _ ## op ## _rtz [] __attribute__((aligned(64))); \
	extern type name ## _ ## op ## _rdn [] __attribute__((aligned(64))); \
	extern type name ## _ ## op ## _rup [] __attribute__((aligned(64))); \
	extern type name ## _ ## op ## _rmm [] __attribute__((aligned(64))); \
	type *name ## _ ## op ## _; \
	type name ## _ ## res ## _ ## op ## _; \
	int name ## _ ## neq ## _ ## op ## _;

#define TEST_DATA_OPS(type, wtype, name) \
	TEST_DATA(type, name) \
	TEST_OP(wtype, name, mul) \
	TEST_OP(wtype, name, add) \
	TEST_OP(wtype, name, sub) \
	TEST_OP(wtype, name, macc) \
	TEST_OP(wtype, name, ident)

/*
	name - float name
	op - operation name
	sew - input sew
	wsew - output sew
	alt - altfmt
	walt - output altfmt
	vle - RVV instruction for loading the input type
	wvle - RVV instruction for loading the output type
	vfop - RVV instruction for performing the operation
	use_vs2 - 1 if instruction uses vs2, otherwise 0
*/
#define TEST(name, op, sew, wsew, alt, walt, vle, wvle, vfop, use_vs2, rm) \
	printf("Testing " #name "_" #op "_" #rm "\n"); \
	avl = N; \
	vl = 0; \
	name ## a_ = name ## a_ ## rm; /* input A pointer */ \
	name ## b_ = name ## b_ ## rm; /* input B pointer */ \
	name ## c_ = name ## c_ ## rm; /* input C pointer */ \
	name ## _ ## op ## _ = name ## _ ## op ## _ ## rm; /* result pointer */ \
	while (avl > 0) { \
		VSETVLI_ALTFMT(vl, avl, sew, LMUL_M1, alt); /* vsetvli */ \
		asm volatile(vle " v0, (%0)" : : "r"(name ## a_)); /* load A */ \
		asm volatile(vle " v8, (%0)" : : "r"(name ## b_)); /* load B */\
		asm volatile(vle " v24, (%0)" : : "r"(name ## c_)); /* load C */\
		asm volatile(vfop " v24, v0" IF ## use_vs2(", v8", "")); /* operation */ \
		if (1/*sew != wsew*/) { \
			VSETVLI_ALTFMT_X0(vl, wsew, LMUL_M2, walt); /* vsetvli */ \
		} \
		asm volatile(wvle " v8, (%0)" : : "r"(name ## _ ## op ## _)); /* load result */ \
		asm volatile("vmsne.vv v16, v24, v8"); /* compare */ \
		asm volatile("vfirst.m %0, v16" : "=r"(name ## _ ## neq ## _ ## op ## _)); /* extract comparison */ \
		name ## a_ += vl; /* increment input A pointer */ \
		name ## b_ += vl; /* increment input B pointer */ \
		name ## c_ += vl; /* increment input C pointer */ \
		name ## _ ## op ## _ += vl; /* increment result pointer */ \
		if (name ## _ ## neq ## _ ## op ## _ != -1) { /* fail if not equal */ \
			printf("Test failed\n"); \
			printf("Index: %d\n", avl); \
			printf("%d\n", name ## _ ## neq ## _ ## op ## _); \
			for (size_t i = 0; i < vl; i ++) { \
				asm volatile("vmv.x.s %0, v24" : "=r"(name ## _ ## res ## _ ## op ## _)); \
				printf("%#010x\n", name ## _ ## res ## _ ## op ## _); \
				asm volatile("vmv.x.s %0, v8" : "=r"(name ## _ ## res ## _ ## op ## _)); \
				printf("%#010x\nNext\n", name ## _ ## res ## _ ## op ## _); \
				asm volatile("vslidedown.vi v24, v24, 1"); \
				asm volatile("vslidedown.vi v8, v8, 1"); \
			} \
			exit(1); \
		} \
		avl -= vl; \
	}

TEST_DATA_OPS(uint32_t, uint32_t, fp32)
TEST_DATA_OPS(uint16_t, uint16_t, fp16)
TEST_DATA_OPS(uint16_t, uint16_t, bf16)
TEST_DATA_OPS(uint8_t, uint8_t, ofp8e5m2)
TEST_DATA_OPS(uint8_t, uint8_t, ofp8e4m3)

TEST_DATA_OPS(uint16_t, uint32_t, fp16W)
TEST_DATA_OPS(uint16_t, uint32_t, bf16W)
TEST_DATA_OPS(uint8_t, uint16_t, ofp8e5m2W)
TEST_DATA_OPS(uint8_t, uint16_t, ofp8e4m3W)

#define TEST_RM(rm, frm_val) \
	asm volatile("csrwi frm, " #frm_val); \
	\
	TEST(fp32, mul, SEW_E32, SEW_E32, 0, 0, "vle32.v", "vle32.v", "vfmul.vv", 1, rm) \
	TEST(fp16, mul, SEW_E16, SEW_E16, 0, 0, "vle16.v", "vle16.v", "vfmul.vv", 1, rm) \
	TEST(bf16, mul, SEW_E16, SEW_E16, 1, 1, "vle16.v", "vle16.v", "vfmul.vv", 1, rm) \
	TEST(ofp8e5m2, mul, SEW_E8, SEW_E8, 1, 1, "vle8.v", "vle8.v", "vfmul.vv", 1, rm) \
	TEST(ofp8e4m3, mul, SEW_E8, SEW_E8, 0, 0, "vle8.v", "vle8.v", "vfmul.vv", 1, rm) \
	TEST(bf16W, mul, SEW_E16, SEW_E32, 1, 0, "vle16.v", "vle32.v", "vfwmul.vv", 1, rm) \
	TEST(fp16W, mul, SEW_E16, SEW_E32, 0, 0, "vle16.v", "vle32.v", "vfwmul.vv", 1, rm) \
	TEST(ofp8e5m2W, mul, SEW_E8, SEW_E16, 1, 1, "vle8.v", "vle16.v", "vfwmul.vv", 1, rm) \
	TEST(ofp8e4m3W, mul, SEW_E8, SEW_E16, 0, 1, "vle8.v", "vle16.v", "vfwmul.vv", 1, rm) \
	\
	TEST(fp32, add, SEW_E32, SEW_E32, 0, 0, "vle32.v", "vle32.v", "vfadd.vv", 1, rm) \
	TEST(fp16, add, SEW_E16, SEW_E16, 0, 0, "vle16.v", "vle16.v", "vfadd.vv", 1, rm) \
	TEST(bf16, add, SEW_E16, SEW_E16, 1, 1, "vle16.v", "vle16.v", "vfadd.vv", 1, rm) \
	TEST(ofp8e5m2, add, SEW_E8, SEW_E8, 1, 1, "vle8.v", "vle8.v", "vfadd.vv", 1, rm) \
	TEST(ofp8e4m3, add, SEW_E8, SEW_E8, 0, 0, "vle8.v", "vle8.v", "vfadd.vv", 1, rm) \
	TEST(bf16W, add, SEW_E16, SEW_E32, 1, 0, "vle16.v", "vle32.v", "vfwadd.vv", 1, rm) \
	TEST(fp16W, add, SEW_E16, SEW_E32, 0, 0, "vle16.v", "vle32.v", "vfwadd.vv", 1, rm) \
	TEST(ofp8e5m2W, add, SEW_E8, SEW_E16, 1, 1, "vle8.v", "vle16.v", "vfwadd.vv", 1, rm) \
	TEST(ofp8e4m3W, add, SEW_E8, SEW_E16, 0, 1, "vle8.v", "vle16.v", "vfwadd.vv", 1, rm) \
	\
	TEST(fp32, sub, SEW_E32, SEW_E32, 0, 0, "vle32.v", "vle32.v", "vfsub.vv", 1, rm) \
	TEST(fp16, sub, SEW_E16, SEW_E16, 0, 0, "vle16.v", "vle16.v", "vfsub.vv", 1, rm) \
	TEST(bf16, sub, SEW_E16, SEW_E16, 1, 1, "vle16.v", "vle16.v", "vfsub.vv", 1, rm) \
	TEST(ofp8e5m2, sub, SEW_E8, SEW_E8, 1, 1, "vle8.v", "vle8.v", "vfsub.vv", 1, rm) \
	TEST(ofp8e4m3, sub, SEW_E8, SEW_E8, 0, 0, "vle8.v", "vle8.v", "vfsub.vv", 1, rm) \
	TEST(bf16W, sub, SEW_E16, SEW_E32, 1, 0, "vle16.v", "vle32.v", "vfwsub.vv", 1, rm) \
	TEST(fp16W, sub, SEW_E16, SEW_E32, 0, 0, "vle16.v", "vle32.v", "vfwsub.vv", 1, rm) \
	TEST(ofp8e5m2W, sub, SEW_E8, SEW_E16, 1, 1, "vle8.v", "vle16.v", "vfwsub.vv", 1, rm) \
	TEST(ofp8e4m3W, sub, SEW_E8, SEW_E16, 0, 1, "vle8.v", "vle16.v", "vfwsub.vv", 1, rm) \

int main() {
	
	TEST_RM(rne, 0)

	// TEST(fp32, macc, SEW_E32, SEW_E32, 0, 0, "vle32.v", "vle32.v", "vfmacc.vv", 1, rne)
	// TEST(fp16, macc, SEW_E16, SEW_E16, 0, 0, "vle16.v", "vle16.v", "vfmacc.vv", 1, rne)
	// TEST(bf16, macc, SEW_E16, SEW_E16, 1, 1, "vle16.v", "vle16.v", "vfmacc.vv", 1, rne)
	// TEST(ofp8e5m2, macc, SEW_E8, SEW_E8, 1, 1, "vle8.v", "vle8.v", "vfmacc.vv", 1, rne)
	// TEST(ofp8e4m3, macc, SEW_E8, SEW_E8, 0, 0, "vle8.v", "vle8.v", "vfmacc.vv", 1, rne)

	// TEST(fp16W, ident, SEW_E16, SEW_E32, 0, 0, "vle16.v", "vle32.v", "vfwcvt.f.f.v", 0, rne)
	// TEST(bf16W, ident, SEW_E16, SEW_E32, 1, 0, "vle16.v", "vle32.v", "vfwcvt.f.f.v", 0, rne)
	// TEST(ofp8e5m2W, ident, SEW_E8, SEW_E16, 1, 1, "vle8.v", "vle16.v", "vfwcvt.f.f.v", 0, rne)
	// TEST(ofp8e4m3W, ident, SEW_E8, SEW_E16, 0, 1, "vle8.v", "vle16.v", "vfwcvt.f.f.v", 0, rne)

	TEST_RM(rtz, 1)
	TEST_RM(rdn, 2)
	TEST_RM(rup, 3)
	TEST_RM(rmm, 4)

	// asm volatile("csrwi frm, " "1");

	// TEST(ofp8e4m3, mul, SEW_E8, SEW_E8, 0, 0, "vle8.v", "vle8.v", "vfmul.vv", 1, rtz)

	printf("All tests passed\n");

	return 0;
}
