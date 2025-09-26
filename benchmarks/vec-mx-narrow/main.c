#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "../vec-mx-fma/rvv_mx.h"

extern size_t N;
size_t avl;
size_t vl;

#define TEST_DATA(type, name, ntype) \
	extern type name[] __attribute__((aligned(64))); \
	extern ntype name ## _out[] __attribute__((aligned(64))); \
	type *name ## _; \
	ntype *name ## _out_; \
	int name ## _neq; \
	ntype name ## _res;

/*
	name - float name
	wsew - input sew
	nsew - output sew
	walt - input altfmt
	nalt - output altfmt
	wvle - RVV instruction for loading the input type
	nvle - RVV instruction for loading the output type
*/
#define TEST(name, wsew, nsew, walt, nalt, wvle, nvle) \
	printf("Testing " #name "\n"); \
	avl = N; \
	vl = 0; \
	name ## _ = name; /* input pointer */ \
	name ## _out_ = name ## _out; /* result pointer */ \
	while (avl > 0) { \
		VSETVLI_ALTFMT(vl, avl, wsew, LMUL_M2, walt); /* vsetvli */ \
		asm volatile(wvle " v0, (%0)" : : "r"(name ## _)); /* load A */ \
        VSETVLI_ALTFMT_X0(vl, nsew, LMUL_M1, nalt); /* vsetvli */ \
		asm volatile("vfncvt.f.f.w v24, v0"); /* operation */ \
		asm volatile(nvle " v8, (%0)" : : "r"(name ## _out_)); /* load result */ \
		asm volatile("vmsne.vv v16, v24, v8"); /* compare */ \
		asm volatile("vfirst.m %0, v16" : "=r"(name ## _neq)); /* extract comparison */ \
		name ## _ += vl; /* increment input pointer */ \
		name ## _out_ += vl; /* increment result pointer */ \
		if (name ## _neq != -1) { /* fail if not equal */ \
			printf("Test failed\n"); \
			printf("Index: %d\n", avl); \
			printf("%d\n", name ## _neq); \
			for (size_t i = 0; i < vl; i ++) { \
				asm volatile("vmv.x.s %0, v24" : "=r"(name ## _res)); \
				printf("%#010x\n", name ## _res); \
				asm volatile("vmv.x.s %0, v8" : "=r"(name ## _res)); \
				printf("%#010x\nNext\n", name ## _res); \
				asm volatile("vslidedown.vi v24, v24, 1"); \
				asm volatile("vslidedown.vi v8, v8, 1"); \
			} \
			exit(1); \
		} \
		avl -= vl; \
	}

TEST_DATA(uint32_t, fp16, uint16_t)
TEST_DATA(uint32_t, bf16, uint16_t)
TEST_DATA(uint16_t, ofp8e5m2, uint8_t)
TEST_DATA(uint16_t, ofp8e4m3, uint8_t)

int main() {

    TEST(fp16, SEW_E32, SEW_E16, 0, 0, "vle32.v", "vle16.v")
    TEST(bf16, SEW_E32, SEW_E16, 0, 1, "vle32.v", "vle16.v")
    TEST(ofp8e5m2, SEW_E16, SEW_E8, 1, 1, "vle16.v", "vle8.v")
    TEST(ofp8e4m3, SEW_E16, SEW_E8, 1, 0, "vle16.v", "vle8.v")

    printf("All tests passed\n");

    return 0;
}