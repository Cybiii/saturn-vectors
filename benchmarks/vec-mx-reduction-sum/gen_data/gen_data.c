#include <stdio.h>
#include <stdint.h>
#include "../../common-data-gen/mx_data_gen.h"

#define N 64

void test_sum(char *name, size_t esize, size_t osize, double min, double max,
              mx_generator gen, fp_t (*sum_fn)(fp_t *, size_t)) {
	fp_t x[N];
	fp_t out_arr[4] = {0};
	size_t i;

	for (i = 0; i < N; i++)
		x[i] = gen(GM_RAND, min, max);
	x[0] = gen(GM_ZERO, min, max);

	out_arr[0] = sum_fn(x, N);

	print_array(name, "_x", x, esize, N);
	print_array(name, "_out", out_arr, osize, 4 / osize);
}

int main() {
	print_header();
	print_uint32("N", N);
	test_sum("fp16_sum", 2, 2, -1e2, 1e2, gen_fp16, fp16_sum);
	test_sum("bf16_sum", 2, 2, -1e2, 1e2, gen_bf16, bf16_sum);
	return 0;
}
