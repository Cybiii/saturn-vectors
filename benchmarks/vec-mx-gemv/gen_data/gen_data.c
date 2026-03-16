/*
 * vec-mx-gemv: y = A*x
 * Golden from Spike MX.
 */
#include <stdio.h>
#include <stdint.h>
#include "../../common-data-gen/mx_data_gen.h"

#define M 32
#define K 32

void test_gemv_fp8(char *name, double min, double max, mx_generator gen,
                   void (*gemv_fn)(uint16_t *, const fp_t *, const fp_t *, size_t, size_t)) {
	fp_t A[M][K];
	fp_t x[K];
	uint16_t y[M];

	for (size_t i = 0; i < M; i++)
		for (size_t j = 0; j < K; j++)
			A[i][j] = gen(GM_RAND, min, max);
	for (size_t j = 0; j < K; j++)
		x[j] = gen(GM_RAND, min, max);

	gemv_fn(y, (const fp_t *)A, (const fp_t *)x, M, K);

	print_array(name, "_A", (fp_t *)A, 1, M * K);
	print_array(name, "_x", x, 1, K);
	print_array(name, "_out", (fp_t *)y, 2, M);
}

void test_gemv_fp16(char *name, double min, double max, mx_generator gen,
                    void (*gemv_fn)(fp_t *, const fp_t *, const fp_t *, size_t, size_t)) {
	fp_t A[M][K];
	fp_t x[K];
	fp_t y[M];

	for (size_t i = 0; i < M; i++)
		for (size_t j = 0; j < K; j++)
			A[i][j] = gen(GM_RAND, min, max);
	for (size_t j = 0; j < K; j++)
		x[j] = gen(GM_RAND, min, max);

	gemv_fn(y, (const fp_t *)A, (const fp_t *)x, M, K);

	print_array(name, "_A", (fp_t *)A, 2, M * K);
	print_array(name, "_x", x, 2, K);
	print_array(name, "_out", y, 4, M);
}

int main() {
	print_header();
	print_uint32("M", M);
	print_uint32("K", K);

	test_gemv_fp8("e4m3_gemv", -3e1, 3e1, gen_e4m3, e4m3_gemv);
	test_gemv_fp8("e5m2_gemv", -1e2, 1e2, gen_e5m2, e5m2_gemv);
	test_gemv_fp16("fp16_gemv", -1e2, 1e2, gen_fp16, fp16_gemv);
	test_gemv_fp16("bf16_gemv", -1e2, 1e2, gen_bf16, bf16_gemv);

	return 0;
}
