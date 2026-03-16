/*
 * vec-mx-max-reduction: max(x) via vfredmax
 * Golden from Spike MX.
 */
#include <stdio.h>
#include <stdint.h>
#include "../../common-data-gen/mx_data_gen.h"

#define N 64

void test_max(char *name, size_t esize, size_t osize, double min, double max,
              mx_generator gen, fp_t (*max_fn)(fp_t *, size_t)) {
	fp_t x[N];
	fp_t out_arr[4] = {0};
	size_t i;

	for (i = 0; i < N; i++)
		x[i] = gen(GM_RAND, min, max);
	x[0] = gen(GM_ZERO, min, max);
	x[1] = gen(GM_INF, min, max);

	out_arr[0] = max_fn(x, N);

	print_array(name, "_x", x, esize, N);
	print_array(name, "_out", out_arr, osize, 4 / osize);
}

int main() {
	print_header();
	print_uint32("N", N);
	test_max("fp16_max", 2, 2, -1e2, 1e2, gen_fp16, fp16_max);
	test_max("bf16_max", 2, 2, -1e2, 1e2, gen_bf16, bf16_max);
	return 0;
}
