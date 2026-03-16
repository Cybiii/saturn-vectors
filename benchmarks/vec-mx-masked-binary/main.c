/*
 * vec-mx-masked-binary: masked vfmul, vfadd
 * Application: sparse ops, irregular workloads.
 * Compares RTL vs Spike MX golden.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "rvv_mx.h"

extern size_t N;
size_t avl, vl;

#define TEST_MASKED(name, op, sew, ealt, vle) \
	do { \
		extern uint8_t name ## _mask[] __attribute__((aligned(64))); \
		extern uint16_t name ## _a[] __attribute__((aligned(64))); \
		extern uint16_t name ## _b[] __attribute__((aligned(64))); \
		extern uint16_t name ## _out[] __attribute__((aligned(64))); \
		printf("Testing " #name " (masked " #op ")\n"); \
		avl = N; vl = 0; \
		uint8_t *m_ = name ## _mask; \
		uint16_t *a_ = name ## _a, *b_ = name ## _b, *out_ = name ## _out; \
		int neq; \
		while (avl > 0) { \
			VSETVLI_ALTFMT(vl, avl, SEW_E8, LMUL_M1, 0); \
			asm volatile("vle8.v v20, (%0)" : : "r"(m_)); \
			asm volatile("vmsne.vi v0, v20, 0"); \
			VSETVLI_ALTFMT(vl, avl, sew, LMUL_M1, ealt); \
			asm volatile("vmv.v.x v24, x0"); \
			asm volatile(vle " v8, (%0)" : : "r"(a_)); \
			asm volatile(vle " v4, (%0)" : : "r"(b_)); \
			asm volatile(op " v24, v8, v4, v0.t"); \
			asm volatile(vle " v16, (%0)" : : "r"(out_)); \
			asm volatile("vmsne.vv v20, v24, v16"); \
			asm volatile("vfirst.m %0, v20" : "=r"(neq)); \
			m_ += vl; a_ += vl; b_ += vl; out_ += vl; \
			if (neq != -1) { printf("Test failed\n"); exit(1); } \
			avl -= vl; \
		} \
	} while (0)

int main() {
	TEST_MASKED(fp16_masked_mul, "vfmul.vv", SEW_E16, 0, "vle16.v");
	TEST_MASKED(bf16_masked_mul, "vfmul.vv", SEW_E16, 1, "vle16.v");
	TEST_MASKED(fp16_masked_add, "vfadd.vv", SEW_E16, 0, "vle16.v");
	TEST_MASKED(bf16_masked_add, "vfadd.vv", SEW_E16, 1, "vle16.v");
	printf("All tests passed\n");
	return 0;
}
