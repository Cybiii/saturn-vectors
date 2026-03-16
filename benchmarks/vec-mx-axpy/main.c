/*
 * vec-mx-axpy: y = alpha*x + y (BLAS SAXPY)
 * Application: residual connections, bias scaling, linear combinations in ML.
 * Compares RTL vs Spike MX golden.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "rvv_mx.h"

extern size_t N;
size_t avl, vl;

#define TEST_DATA(type, name) \
	extern type name ## _alpha[] __attribute__((aligned(64))); \
	extern type name ## _x[] __attribute__((aligned(64))); \
	extern type name ## _y[] __attribute__((aligned(64))); \
	extern type name ## _out[] __attribute__((aligned(64)));

#define TEST_AXPY(name, type, sew, ealt, vle, op) \
	do { \
		printf("Testing " #name " (axpy: y=alpha*x+y)\n"); \
		avl = N; \
		vl = 0; \
		type *x_ = name ## _x; \
		type *y_ = name ## _y; \
		type *out_ = name ## _out; \
		type alpha_val = name ## _alpha[0]; \
		int neq; \
		type res; \
		while (avl > 0) { \
			VSETVLI_ALTFMT(vl, avl, sew, LMUL_M1, 0); \
			asm volatile(vle " v0, (%0)" : : "r"(x_)); \
			asm volatile(vle " v8, (%0)" : : "r"(y_)); \
			asm volatile("vmv.v.x v4, %0" : : "r"(alpha_val)); \
			VSETVLI_ALTFMT_X0(vl, sew, LMUL_M1, ealt); \
			op; \
			asm volatile(vle " v16, (%0)" : : "r"(out_)); \
			asm volatile("vmsne.vv v20, v24, v16"); \
			asm volatile("vfirst.m %0, v20" : "=r"(neq)); \
			x_ += vl; y_ += vl; out_ += vl; \
			if (neq != -1) { \
				printf("Test failed at idx %d\n", neq); \
				asm volatile("vmv.x.s %0, v24" : "=r"(res)); \
				printf("rtl=%#x ", (unsigned)res); \
				asm volatile("vmv.x.s %0, v16" : "=r"(res)); \
				printf("gold=%#x\n", (unsigned)res); \
				exit(1); \
			} \
			avl -= vl; \
		} \
	} while (0)

TEST_DATA(uint8_t, e4m3_axpy)
TEST_DATA(uint8_t, e5m2_axpy)
TEST_DATA(uint16_t, fp16_axpy)
TEST_DATA(uint16_t, bf16_axpy)

int main() {
	TEST_AXPY(e4m3_axpy, uint8_t, SEW_E8, 0, "vle8.v",
		do { asm volatile("vmv.v.v v24, v8"); asm volatile("vfmacc.vv v24, v4, v0"); } while (0));
	TEST_AXPY(e5m2_axpy, uint8_t, SEW_E8, 1, "vle8.v",
		do { asm volatile("vmv.v.v v24, v8"); asm volatile("vfmacc.vv v24, v4, v0"); } while (0));
	TEST_AXPY(fp16_axpy, uint16_t, SEW_E16, 0, "vle16.v",
		do { asm volatile("vmv.v.v v24, v8"); asm volatile("vfmacc.vv v24, v4, v0"); } while (0));
	TEST_AXPY(bf16_axpy, uint16_t, SEW_E16, 1, "vle16.v",
		do { asm volatile("vmv.v.v v24, v8"); asm volatile("vfmacc.vv v24, v4, v0"); } while (0));

	printf("All tests passed\n");
	return 0;
}
