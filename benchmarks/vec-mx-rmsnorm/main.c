/*
 * vec-mx-rmsnorm: y = x / sqrt(mean(x*x) + eps)
 * Application: LLM normalization.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "rvv_mx.h"

extern size_t N;
extern float eps;
size_t avl, vl;

#define TEST_RMSNORM(name, sew, ealt) \
	do { \
		extern uint16_t name ## _x[] __attribute__((aligned(64))); \
		extern uint16_t name ## _out[] __attribute__((aligned(64))); \
		printf("Testing " #name " (RMSNorm)\n"); \
		avl = N; vl = 0; \
		uint16_t *x_ = name ## _x; \
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
		uint32_t sum_sq; \
		asm volatile("vmv.x.s %0, v8" : "=r"(sum_sq)); \
		float mean_sq = *(float *)&sum_sq / (float)N; \
		float rms = sqrtf(mean_sq + eps); \
		float scale = 1.0f / rms; \
		uint16_t scale_h = (ealt) ? ((*(uint32_t *)&scale >> 16) & 0xFFFF) : 0; \
		if (!(ealt)) asm volatile("fmv.s fa0, %1; fcvt.h.s fa1, fa0; fmv.x.h %0, fa1" : "=r"(scale_h) : "f"(scale)); \
		x_ = name ## _x; \
		uint16_t *out_ = name ## _out; \
		avl = N; vl = 0; \
		int neq; \
		while (avl > 0) { \
			VSETVLI_ALTFMT(vl, avl, sew, LMUL_M1, ealt); \
			asm volatile("vle16.v v0, (%0)" : : "r"(x_)); \
			asm volatile("vmv.v.x v4, %0" :: "r"((uint32_t)scale_h)); \
			VSETVLI_ALTFMT_X0(vl, sew, LMUL_M1, ealt); \
			asm volatile("vfmul.vv v24, v0, v4"); \
			asm volatile("vle16.v v16, (%0)" : : "r"(out_)); \
			asm volatile("vmsne.vv v20, v24, v16"); \
			asm volatile("vfirst.m %0, v20" : "=r"(neq)); \
			x_ += vl; out_ += vl; \
			if (neq != -1) { printf("Test failed\n"); exit(1); } \
			avl -= vl; \
		} \
	} while (0)

int main() {
	TEST_RMSNORM(fp16_rmsnorm, SEW_E16, 0);
	TEST_RMSNORM(bf16_rmsnorm, SEW_E16, 1);
	printf("All tests passed\n");
	return 0;
}
