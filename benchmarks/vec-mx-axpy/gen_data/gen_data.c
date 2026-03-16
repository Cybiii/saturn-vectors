/*
 * vec-mx-axpy: y = alpha*x + y (BLAS SAXPY)
 * Application: residual connections, bias scaling, linear combinations in ML.
 * Golden from Spike MX.
 */
#include <stdio.h>
#include <stdint.h>
#include "../../common-data-gen/mx_data_gen.h"

#define N 128

void test_axpy(char *name, size_t esize, double min, double max,
               mx_generator gen, mx_op_ternary fmacc) {
	fp_t alpha;
	fp_t alpha_arr[4] = {0};
	fp_t x[N];
	fp_t y[N];
	fp_t out[N];

	alpha = gen(GM_RAND, min, max);
	alpha_arr[0] = alpha;
	for (size_t i = 0; i < N; i++) {
		x[i] = gen(GM_RAND, min, max);
		y[i] = gen(GM_RAND, min, max);
	}
	/* Edge cases */
	x[0] = gen(GM_ZERO, min, max);
	y[1] = gen(GM_ZERO, min, max);
	x[2] = gen(GM_INF, min, max);
	alpha = gen(GM_RAND, 0.5, 2.0); /* avoid Inf*Inf */
	alpha_arr[0] = alpha;

	for (size_t i = 0; i < N; i++) {
		out[i] = fmacc(alpha, x[i], y[i]);
	}

	print_array(name, "_alpha", alpha_arr, esize, 4 / esize);
	print_array(name, "_x", x, esize, N);
	print_array(name, "_y", y, esize, N);
	print_array(name, "_out", out, esize, N);
}

int main() {
	print_header();
	print_uint32("N", N);

	test_axpy("e4m3_axpy", 1, -3e1, 3e1, gen_e4m3, e4m3_fmacc);
	test_axpy("e5m2_axpy", 1, -1e2, 1e2, gen_e5m2, e5m2_fmacc);
	test_axpy("fp16_axpy", 2, -1e2, 1e2, gen_fp16, fp16_fmacc);
	test_axpy("bf16_axpy", 2, -1e2, 1e2, gen_bf16, bf16_fmacc);

	return 0;
}
