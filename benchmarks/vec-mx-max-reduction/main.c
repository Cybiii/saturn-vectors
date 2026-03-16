/*
 * vec-mx-max-reduction: max(x) via vfredmax
 * Application: softmax (max before exp for numerical stability).
 * Compares RTL vs Spike MX golden.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "rvv_mx.h"

extern size_t N;
size_t avl, vl;

#define TEST_MAX(name, sew, ealt, ninf) \
	do { \
		extern uint16_t name ## _x[] __attribute__((aligned(64))); \
		extern uint16_t name ## _out[] __attribute__((aligned(64))); \
		uint16_t *x_ = name ## _x; \
		printf("Testing " #name " (reduction max)\n"); \
		avl = N; \
		VSETVLI_ALTFMT_X0(1, sew, LMUL_M1, ealt); \
		asm volatile("vmv.v.x v8, %0" :: "r"((uint32_t)(ninf))); \
		while (avl > 0) { \
			VSETVLI_ALTFMT(vl, avl, sew, LMUL_M1, ealt); \
			asm volatile("vle16.v v16, (%0)" : : "r"(x_)); \
			VSETVLI_ALTFMT_X0(vl, sew, LMUL_M1, ealt); \
			asm volatile("vfredmax.vs v8, v16, v8"); \
			x_ += vl; avl -= vl; \
		} \
		VSETVLI_ALTFMT_X0(1, sew, LMUL_M1, ealt); \
		uint16_t rtl, gold = name ## _out[0]; \
		asm volatile("vmv.x.s %0, v8" : "=r"(rtl)); \
		if (rtl != gold) { printf("Test failed: rtl=%#x gold=%#x\n", rtl, gold); exit(1); } \
	} while (0)

int main() {
	TEST_MAX(fp16_max, SEW_E16, 0, 0xfc00);  /* -inf */
	TEST_MAX(bf16_max, SEW_E16, 1, 0xff80);   /* -inf */
	printf("All tests passed\n");
	return 0;
}
