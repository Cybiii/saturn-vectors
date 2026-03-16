#include <stdio.h>
#include <stdint.h>
#include "../../common-data-gen/mx_data_gen.h"

#define N 128

void test_bias(char *name, size_t esize, double min, double max,
               mx_generator gen, mx_op_binary add) {
	fp_t b;
	fp_t b_arr[4] = {0};
	fp_t x[N];
	fp_t out[N];

	b = gen(GM_RAND, min, max);
	b_arr[0] = b;
	for (size_t i = 0; i < N; i++)
		x[i] = gen(GM_RAND, min, max);
	x[0] = gen(GM_ZERO, min, max);
	b = gen(GM_RAND, -0.5, 0.5);
	b_arr[0] = b;

	for (size_t i = 0; i < N; i++)
		out[i] = add(x[i], b);

	print_array(name, "_b", b_arr, esize, 4 / esize);
	print_array(name, "_x", x, esize, N);
	print_array(name, "_out", out, esize, N);
}

int main() {
	print_header();
	print_uint32("N", N);
	test_bias("e4m3_bias", 1, -3e1, 3e1, gen_e4m3, e4m3_add);
	test_bias("e5m2_bias", 1, -1e2, 1e2, gen_e5m2, e5m2_add);
	test_bias("fp16_bias", 2, -1e2, 1e2, gen_fp16, fp16_add);
	test_bias("bf16_bias", 2, -1e2, 1e2, gen_bf16, bf16_add);
	return 0;
}
