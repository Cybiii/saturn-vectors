/*
 * vec-mx-masked-binary: masked vfmul, vfadd
 * Golden from Spike MX.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "../../common-data-gen/mx_data_gen.h"

#define N 128

void test_masked(char *name, double min, double max, mx_generator gen,
                 void (*fn)(uint16_t *, const uint8_t *, const fp_t *, const fp_t *, size_t)) {
	fp_t a[N], b[N];
	uint8_t mask[N];
	uint16_t out[N];
	fp_t out_fp[N];

	for (size_t i = 0; i < N; i++) {
		a[i] = gen(GM_RAND, min, max);
		b[i] = gen(GM_RAND, min, max);
		mask[i] = (rand() % 3 == 0) ? 0 : 1;
	}

	fn(out, mask, a, b, N);
	for (size_t i = 0; i < N; i++) out_fp[i] = out[i];

	print_mask_bytes(name, "_mask", mask, N);
	print_array(name, "_a", a, 2, N);
	print_array(name, "_b", b, 2, N);
	print_array(name, "_out", out_fp, 2, N);
}

int main() {
	srand(42);
	print_header();
	print_uint32("N", N);

	test_masked("fp16_masked_mul", -1e2, 1e2, gen_fp16, fp16_masked_mul);
	test_masked("bf16_masked_mul", -1e2, 1e2, gen_bf16, bf16_masked_mul);
	test_masked("fp16_masked_add", -1e2, 1e2, gen_fp16, fp16_masked_add);
	test_masked("bf16_masked_add", -1e2, 1e2, gen_bf16, bf16_masked_add);

	return 0;
}
