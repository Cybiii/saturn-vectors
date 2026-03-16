/*
 * vec-mx-l2norm: sqrt(sum(x*x)) via vfwmul + vfredosum + scalar sqrt
 * Application: normalization, loss functions.
 * Compares RTL vs Spike MX golden.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "rvv_mx.h"

extern size_t N;
size_t avl, vl;

#define TEST_L2NORM(name, sew, ealt) \
	do { \
		extern uint16_t name ## _x[] __attribute__((aligned(64))); \
		extern uint32_t name ## _out[] __attribute__((aligned(64))); \
		uint16_t *x_ = name ## _x; \
		printf("Testing " #name " (L2 norm)\n"); \
		avl = N; \
		VSETVLI_ALTFMT_X0(1, SEW_E32, LMUL_M1, 0); \
		asm volatile("vmv.v.x v8, x0"); \
		while (avl > 0) { \
			VSETVLI_ALTFMT(vl, avl, sew, LMUL_M1, ealt); \
			asm volatile("vle16.v v0, (%0)" : : "r"(x_)); \
			VSETVLI_ALTFMT_X0(vl, sew, LMUL_M1, ealt); \
			asm volatile("vfwmul.vv v16, v0, v0"); \
			VSETVLI_ALTFMT_X0(vl, SEW_E32, LMUL_M1, 0); \
			asm volatile("vfredosum.vs v8, v16, v8"); \
			x_ += vl; avl -= vl; \
		} \
		VSETVLI_ALTFMT_X0(1, SEW_E32, LMUL_M1, 0); \
		uint32_t sum_sq; \
		asm volatile("vmv.x.s %0, v8" : "=r"(sum_sq)); \
		float f = *(float *)&sum_sq; \
		float rtl = sqrtf(f); \
		uint32_t gold = name ## _out[0]; \
		uint32_t rtl_bits = *(uint32_t *)&rtl; \
		if (rtl_bits != gold) { printf("Test failed: rtl=%#x gold=%#x\n", rtl_bits, gold); exit(1); } \
	} while (0)

int main() {
	TEST_L2NORM(fp16_l2norm, SEW_E16, 0);
	TEST_L2NORM(bf16_l2norm, SEW_E16, 1);
	printf("All tests passed\n");
	return 0;
}
