/*
 * vec-mx-bias-add: y = x + b (broadcast bias)
 * Application: bias addition after linear layer.
 * Compares RTL vs Spike MX golden.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "rvv_mx.h"

extern size_t N;
size_t avl, vl;

#define TEST_BIAS(name, type, sew, ealt, vle) \
	do { \
		extern type name ## _b[] __attribute__((aligned(64))); \
		extern type name ## _x[] __attribute__((aligned(64))); \
		extern type name ## _out[] __attribute__((aligned(64))); \
		printf("Testing " #name " (bias-add: y=x+b)\n"); \
		avl = N; vl = 0; \
		type *x_ = name ## _x, *out_ = name ## _out; \
		type b_val = name ## _b[0]; \
		int neq; type res; \
		while (avl > 0) { \
			VSETVLI_ALTFMT(vl, avl, sew, LMUL_M1, 0); \
			asm volatile(vle " v0, (%0)" : : "r"(x_)); \
			asm volatile("vmv.v.x v4, %0" : : "r"(b_val)); \
			VSETVLI_ALTFMT_X0(vl, sew, LMUL_M1, ealt); \
			asm volatile("vfadd.vv v24, v0, v4"); \
			asm volatile(vle " v16, (%0)" : : "r"(out_)); \
			asm volatile("vmsne.vv v20, v24, v16"); \
			asm volatile("vfirst.m %0, v20" : "=r"(neq)); \
			x_ += vl; out_ += vl; \
			if (neq != -1) { printf("Test failed\n"); exit(1); } \
			avl -= vl; \
		} \
	} while (0)

int main() {
	TEST_BIAS(e4m3_bias, uint8_t, SEW_E8, 0, "vle8.v");
	TEST_BIAS(e5m2_bias, uint8_t, SEW_E8, 1, "vle8.v");
	TEST_BIAS(fp16_bias, uint16_t, SEW_E16, 0, "vle16.v");
	TEST_BIAS(bf16_bias, uint16_t, SEW_E16, 1, "vle16.v");
	printf("All tests passed\n");
	return 0;
}
