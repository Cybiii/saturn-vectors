/*
 * vec-mx-scale: y = alpha * x (broadcast scale)
 * Application: pre-softmax scaling, layer norm scaling, temperature scaling.
 * Golden from Spike MX.
 */
#include <stdio.h>
#include <stdint.h>
#include "../../common-data-gen/mx_data_gen.h"

#define N 128

void test_scale(char *name, size_t esize, double min, double max,
                mx_generator gen, mx_op_binary mul) {
	fp_t alpha;
	fp_t alpha_arr[4] = {0};
	fp_t x[N];
	fp_t out[N];

	alpha = gen(GM_RAND, min, max);
	alpha_arr[0] = alpha;
	for (size_t i = 0; i < N; i++)
		x[i] = gen(GM_RAND, min, max);
	x[0] = gen(GM_ZERO, min, max);
	x[1] = gen(GM_INF, min, max);
	alpha = gen(GM_RAND, 0.5, 2.0);
	alpha_arr[0] = alpha;

	for (size_t i = 0; i < N; i++)
		out[i] = mul(alpha, x[i]);

	print_array(name, "_alpha", alpha_arr, esize, 4 / esize);
	print_array(name, "_x", x, esize, N);
	print_array(name, "_out", out, esize, N);
}

int main() {
	print_header();
	print_uint32("N", N);

	test_scale("e4m3_scale", 1, -3e1, 3e1, gen_e4m3, e4m3_mul);
	test_scale("e5m2_scale", 1, -1e2, 1e2, gen_e5m2, e5m2_mul);
	test_scale("fp16_scale", 2, -1e2, 1e2, gen_fp16, fp16_mul);
	test_scale("bf16_scale", 2, -1e2, 1e2, gen_bf16, bf16_mul);

	return 0;
}
