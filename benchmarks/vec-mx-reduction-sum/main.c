#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "rvv_mx.h"

extern size_t N;
size_t avl, vl;

#define TEST_SUM(name, sew, ealt) \
	do { \
		extern uint16_t name ## _x[] __attribute__((aligned(64))); \
		extern uint16_t name ## _out[] __attribute__((aligned(64))); \
		uint16_t *x_ = name ## _x; \
		printf("Testing " #name " (reduction sum)\n"); \
		avl = N; \
		VSETVLI_ALTFMT_X0(1, sew, LMUL_M1, ealt); \
		asm volatile("vmv.v.x v8, x0"); \
		while (avl > 0) { \
			VSETVLI_ALTFMT(vl, avl, sew, LMUL_M1, ealt); \
			asm volatile("vle16.v v16, (%0)" : : "r"(x_)); \
			VSETVLI_ALTFMT_X0(vl, sew, LMUL_M1, ealt); \
			asm volatile("vfredosum.vs v8, v16, v8"); \
			x_ += vl; avl -= vl; \
		} \
		VSETVLI_ALTFMT_X0(1, sew, LMUL_M1, ealt); \
		uint16_t rtl, gold = name ## _out[0]; \
		asm volatile("vmv.x.s %0, v8" : "=r"(rtl)); \
		if (rtl != gold) { printf("Test failed: rtl=%#x gold=%#x\n", rtl, gold); exit(1); } \
	} while (0)

int main() {
	TEST_SUM(fp16_sum, SEW_E16, 0);
	TEST_SUM(bf16_sum, SEW_E16, 1);
	printf("All tests passed\n");
	return 0;
}
