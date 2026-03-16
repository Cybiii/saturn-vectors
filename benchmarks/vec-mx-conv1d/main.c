/*
 * vec-mx-conv1d: 1D convolution
 * Application: audio, sequence models.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "rvv_mx.h"

extern size_t IN_LEN;
extern size_t K_LEN;
size_t avl, vl;

#define TEST_CONV1D(name, sew, ealt) \
	do { \
		extern uint16_t name ## _in[] __attribute__((aligned(64))); \
		extern uint16_t name ## _ker[] __attribute__((aligned(64))); \
		extern uint32_t name ## _out[] __attribute__((aligned(64))); \
		size_t out_len = IN_LEN - K_LEN + 1; \
		printf("Testing " #name " (conv1d)\n"); \
		for (size_t i = 0; i < out_len; i++) { \
			avl = K_LEN; \
			VSETVLI_ALTFMT_X0(1, SEW_E32, LMUL_M1, 0); \
			asm volatile("vmv.v.x v8, x0"); \
			uint16_t *in_ = name ## _in + i; \
			uint16_t *ker_ = name ## _ker; \
			while (avl > 0) { \
				VSETVLI_ALTFMT(vl, avl, SEW_E16, LMUL_M1, ealt); \
				asm volatile("vle16.v v0, (%0)" : : "r"(in_)); \
				asm volatile("vle16.v v4, (%0)" : : "r"(ker_)); \
				VSETVLI_ALTFMT_X0(vl, SEW_E16, LMUL_M1, ealt); \
				asm volatile("vmv.v.x v16, x0"); \
				asm volatile("vfwmacc.vv v16, v0, v4"); \
				VSETVLI_ALTFMT_X0(vl, SEW_E32, LMUL_M1, 0); \
				asm volatile("vfredosum.vs v8, v16, v8"); \
				in_ += vl; ker_ += vl; avl -= vl; \
			} \
			uint32_t rtl; \
			asm volatile("vmv.x.s %0, v8" : "=r"(rtl)); \
			uint32_t gold = name ## _out[i]; \
			if (rtl != gold) { printf("Test failed out[%zu]\n", i); exit(1); } \
		} \
	} while (0)

int main() {
	TEST_CONV1D(fp16_conv1d, SEW_E16, 0);
	TEST_CONV1D(bf16_conv1d, SEW_E16, 1);
	printf("All tests passed\n");
	return 0;
}
