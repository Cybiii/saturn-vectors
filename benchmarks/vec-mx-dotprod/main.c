/*
 * vec-mx-dotprod: dot(a,b) = sum(a[i]*b[i])
 * Application: attention scores, embeddings, linear layer pre-activation.
 * Compares RTL vs Spike MX golden.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "rvv_mx.h"

extern size_t N;
size_t avl, vl;

#define TEST_DOT(name, sew, ealt) \
	do { \
		extern uint16_t name ## _a[] __attribute__((aligned(64))); \
		extern uint16_t name ## _b[] __attribute__((aligned(64))); \
		extern uint32_t name ## _out[] __attribute__((aligned(64))); \
		uint16_t *a_ = name ## _a; \
		uint16_t *b_ = name ## _b; \
		printf("Testing " #name " (dot product)\n"); \
		avl = N; \
		VSETVLI_ALTFMT_X0(1, SEW_E32, LMUL_M1, 0); \
		asm volatile("vmv.v.x v8, x0"); \
		while (avl > 0) { \
			VSETVLI_ALTFMT(vl, avl, sew, LMUL_M1, ealt); \
			asm volatile("vle16.v v0, (%0)" : : "r"(a_)); \
			asm volatile("vle16.v v4, (%0)" : : "r"(b_)); \
			VSETVLI_ALTFMT_X0(vl, sew, LMUL_M1, ealt); \
			asm volatile("vmv.v.x v16, x0"); \
			asm volatile("vfwmacc.vv v16, v0, v4"); \
			VSETVLI_ALTFMT_X0(vl, SEW_E32, LMUL_M1, 0); \
			asm volatile("vfredosum.vs v8, v16, v8"); \
			a_ += vl; b_ += vl; avl -= vl; \
		} \
		VSETVLI_ALTFMT_X0(1, SEW_E32, LMUL_M1, 0); \
		uint32_t rtl, gold = name ## _out[0]; \
		asm volatile("vmv.x.s %0, v8" : "=r"(rtl)); \
		if (rtl != gold) { \
			printf("Test failed: rtl=%#x gold=%#x\n", rtl, gold); \
			exit(1); \
		} \
	} while (0)

int main() {
	TEST_DOT(fp16_dot, SEW_E16, 0);
	TEST_DOT(bf16_dot, SEW_E16, 1);
	printf("All tests passed\n");
	return 0;
}
