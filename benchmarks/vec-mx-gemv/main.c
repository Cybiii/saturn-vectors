/*
 * vec-mx-gemv: y = A*x (matrix-vector multiply)
 * Application: linear layers, inference.
 * Compares RTL vs Spike MX golden.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "rvv_mx.h"

extern size_t M;
extern size_t K;
size_t avl, vl;

#define TEST_GEMV_FP8(name, ealt, ivle) \
	do { \
		extern uint8_t name ## _A[] __attribute__((aligned(64))); \
		extern uint8_t name ## _x[] __attribute__((aligned(64))); \
		extern uint16_t name ## _out[] __attribute__((aligned(64))); \
		printf("Testing " #name " (GEMV)\n"); \
		for (size_t i = 0; i < M; i++) { \
			avl = K; \
			VSETVLI_ALTFMT_X0(1, SEW_E16, LMUL_M1, 0); \
			asm volatile("vmv.v.x v8, x0"); \
			uint8_t *A_row = name ## _A + i * K; \
			uint8_t *x_ = name ## _x; \
			while (avl > 0) { \
				VSETVLI_ALTFMT(vl, avl, SEW_E8, LMUL_M1, ealt); \
				asm volatile(ivle " v0, (%0)" : : "r"(A_row)); \
				asm volatile(ivle " v4, (%0)" : : "r"(x_)); \
				VSETVLI_ALTFMT_X0(vl, SEW_E8, LMUL_M1, ealt); \
				asm volatile("vmv.v.x v16, x0"); \
				asm volatile("vfwmacc.vv v16, v0, v4"); \
				VSETVLI_ALTFMT_X0(vl, SEW_E16, LMUL_M1, 0); \
				asm volatile("vfredosum.vs v8, v16, v8"); \
				A_row += vl; x_ += vl; avl -= vl; \
			} \
			VSETVLI_ALTFMT_X0(1, SEW_E16, LMUL_M1, 0); \
			uint16_t rtl; \
			asm volatile("vmv.x.s %0, v8" : "=r"(rtl)); \
			uint16_t gold = name ## _out[i]; \
			if (rtl != gold) { printf("Test failed row %zu: rtl=%#x gold=%#x\n", i, (unsigned)rtl, (unsigned)gold); exit(1); } \
		} \
	} while (0)

#define TEST_GEMV_FP16(name, ealt, ivle) \
	do { \
		extern uint16_t name ## _A[] __attribute__((aligned(64))); \
		extern uint16_t name ## _x[] __attribute__((aligned(64))); \
		extern uint32_t name ## _out[] __attribute__((aligned(64))); \
		printf("Testing " #name " (GEMV)\n"); \
		for (size_t i = 0; i < M; i++) { \
			avl = K; \
			VSETVLI_ALTFMT_X0(1, SEW_E32, LMUL_M1, 0); \
			asm volatile("vmv.v.x v8, x0"); \
			uint16_t *A_row = name ## _A + i * K; \
			uint16_t *x_ = name ## _x; \
			while (avl > 0) { \
				VSETVLI_ALTFMT(vl, avl, SEW_E16, LMUL_M1, ealt); \
				asm volatile(ivle " v0, (%0)" : : "r"(A_row)); \
				asm volatile(ivle " v4, (%0)" : : "r"(x_)); \
				VSETVLI_ALTFMT_X0(vl, SEW_E16, LMUL_M1, ealt); \
				asm volatile("vmv.v.x v16, x0"); \
				asm volatile("vfwmacc.vv v16, v0, v4"); \
				VSETVLI_ALTFMT_X0(vl, SEW_E32, LMUL_M1, 0); \
				asm volatile("vfredosum.vs v8, v16, v8"); \
				A_row += vl; x_ += vl; avl -= vl; \
			} \
			VSETVLI_ALTFMT_X0(1, SEW_E32, LMUL_M1, 0); \
			uint32_t rtl; \
			asm volatile("vmv.x.s %0, v8" : "=r"(rtl)); \
			uint32_t gold = name ## _out[i]; \
			if (rtl != gold) { printf("Test failed row %zu: rtl=%#x gold=%#x\n", i, (unsigned)rtl, (unsigned)gold); exit(1); } \
		} \
	} while (0)

int main() {
	TEST_GEMV_FP8(e4m3_gemv, 0, "vle8.v");
	TEST_GEMV_FP8(e5m2_gemv, 1, "vle8.v");
	TEST_GEMV_FP16(fp16_gemv, 0, "vle16.v");
	TEST_GEMV_FP16(bf16_gemv, 1, "vle16.v");
	printf("All tests passed\n");
	return 0;
}
