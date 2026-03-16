#include <stdio.h>
#include <stdint.h>
#include "../../common-data-gen/mx_data_gen.h"

#define IN_LEN 64
#define K_LEN 8

void test_conv1d(char *name, double min, double max, mx_generator gen,
                 void (*fn)(fp_t *, const fp_t *, const fp_t *, size_t, size_t)) {
	fp_t in[IN_LEN], ker[K_LEN];
	fp_t out[IN_LEN - K_LEN + 1];

	for (size_t i = 0; i < IN_LEN; i++)
		in[i] = gen(GM_RAND, min, max);
	for (size_t i = 0; i < K_LEN; i++)
		ker[i] = gen(GM_RAND, min, max);

	fn(out, in, ker, IN_LEN, K_LEN);

	print_array(name, "_in", in, 2, IN_LEN);
	print_array(name, "_ker", ker, 2, K_LEN);
	print_array(name, "_out", out, 4, IN_LEN - K_LEN + 1);
}

int main() {
	print_header();
	print_uint32("IN_LEN", IN_LEN);
	print_uint32("K_LEN", K_LEN);
	test_conv1d("fp16_conv1d", -1e2, 1e2, gen_fp16, fp16_conv1d);
	test_conv1d("bf16_conv1d", -1e2, 1e2, gen_bf16, bf16_conv1d);
	return 0;
}
