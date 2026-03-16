/*
 * vec-mx-scale: y = alpha * x (broadcast scale)
 * Application: pre-softmax scaling, layer norm scaling.
 * Compares RTL vs Spike MX golden.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "rvv_mx.h"

extern size_t N;
size_t avl, vl;

#define TEST_SCALE(name, type, sew, ealt, vle) \
	do { \
		extern type name ## _alpha[] __attribute__((aligned(64))); \
		extern type name ## _x[] __attribute__((aligned(64))); \
		extern type name ## _out[] __attribute__((aligned(64))); \
		printf("Testing " #name " (scale: y=alpha*x)\n"); \
		avl = N; vl = 0; \
		type *x_ = name ## _x, *out_ = name ## _out; \
		type alpha_val = name ## _alpha[0]; \
		int neq; type res; \
		while (avl > 0) { \
			VSETVLI_ALTFMT(vl, avl, sew, LMUL_M1, 0); \
			asm volatile(vle " v0, (%0)" : : "r"(x_)); \
			asm volatile("vmv.v.x v4, %0" : : "r"(alpha_val)); \
			VSETVLI_ALTFMT_X0(vl, sew, LMUL_M1, ealt); \
			asm volatile("vfmul.vv v24, v4, v0"); \
			asm volatile(vle " v16, (%0)" : : "r"(out_)); \
			asm volatile("vmsne.vv v20, v24, v16"); \
			asm volatile("vfirst.m %0, v20" : "=r"(neq)); \
			x_ += vl; out_ += vl; \
			if (neq != -1) { printf("Test failed\n"); asm volatile("vmv.x.s %0, v24" : "=r"(res)); printf("rtl=%#x\n", (unsigned)res); exit(1); } \
			avl -= vl; \
		} \
	} while (0)

int main() {
	TEST_SCALE(e4m3_scale, uint8_t, SEW_E8, 0, "vle8.v");
	TEST_SCALE(e5m2_scale, uint8_t, SEW_E8, 1, "vle8.v");
	TEST_SCALE(fp16_scale, uint16_t, SEW_E16, 0, "vle16.v");
	TEST_SCALE(bf16_scale, uint16_t, SEW_E16, 1, "vle16.v");
	printf("All tests passed\n");
	return 0;
}
