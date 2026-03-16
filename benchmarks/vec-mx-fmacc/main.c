#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "rvv_mx.h"

extern size_t N;
size_t avl;
size_t vl;

#define TEST_DATA(type, name, otype) \
	extern type name ## _a[] __attribute__((aligned(64))); \
	extern type name ## _b[] __attribute__((aligned(64))); \
	extern type name ## _c[] __attribute__((aligned(64))); \
	extern otype name ## _out[] __attribute__((aligned(64))); \
	type *name ## _a_; \
	type *name ## _b_; \
	type *name ## _c_; \
	otype *name ## _out_; \
	int name ## _neq; \
	otype name ## _res;

#define TEST(name, isew, osew, esew, ealt, ivle, ovle, ilmul, olmul, op) \
	printf("Testing " #name " (vfmacc.vv)\n"); \
	avl = N; \
	vl = 0; \
	name ## _a_ = name ## _a; \
	name ## _b_ = name ## _b; \
	name ## _c_ = name ## _c; \
	name ## _out_ = name ## _out; \
	while (avl > 0) { \
		VSETVLI_ALTFMT(vl, avl, isew, ilmul, 0); \
		asm volatile(ivle " v0, (%0)" : : "r"(name ## _a_)); \
		asm volatile(ivle " v4, (%0)" : : "r"(name ## _b_)); \
		asm volatile(ivle " v8, (%0)" : : "r"(name ## _c_)); \
		VSETVLI_ALTFMT_X0(vl, esew, LMUL_M1, ealt); \
		op; \
		VSETVLI_ALTFMT_X0(vl, osew, olmul, 0); \
		asm volatile(ovle " v16, (%0)" : : "r"(name ## _out_)); \
		asm volatile("vmsne.vv v20, v24, v16"); \
		asm volatile("vfirst.m %0, v20" : "=r"(name ## _neq)); \
		name ## _a_ += vl; \
		name ## _b_ += vl; \
		name ## _c_ += vl; \
		name ## _out_ += vl; \
		if (name ## _neq != -1) { \
			printf("Test failed\n"); \
			printf("Index: %zu\n", avl); \
			printf("%d\n", name ## _neq); \
			for (size_t i = 0; i < vl; i++) { \
				asm volatile("vmv.x.s %0, v24" : "=r"(name ## _res)); \
				printf("rtl=%#x ", name ## _res); \
				asm volatile("vmv.x.s %0, v16" : "=r"(name ## _res)); \
				printf("gold=%#x\n", name ## _res); \
				asm volatile("vslidedown.vi v24, v24, 1"); \
				asm volatile("vslidedown.vi v16, v16, 1"); \
			} \
			exit(1); \
		} \
		avl -= vl; \
	}

TEST_DATA(uint8_t, e4m3_fmacc, uint8_t)
TEST_DATA(uint8_t, e5m2_fmacc, uint8_t)

int main() {

	TEST(e4m3_fmacc, SEW_E8, SEW_E8, SEW_E8, 0, "vle8.v", "vle8.v", LMUL_M1, LMUL_M1,
	     do { asm volatile("vmv.v.v v24, v8"); asm volatile("vfmacc.vv v24, v0, v4"); } while(0))
	TEST(e5m2_fmacc, SEW_E8, SEW_E8, SEW_E8, 1, "vle8.v", "vle8.v", LMUL_M1, LMUL_M1,
	     do { asm volatile("vmv.v.v v24, v8"); asm volatile("vfmacc.vv v24, v0, v4"); } while(0))

	printf("All tests passed\n");

	return 0;
}
