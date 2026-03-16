/*
 * vec-mx-dotprod: dot(a,b) = sum(a[i]*b[i])
 * Application: attention scores, embeddings, linear layer pre-activation.
 * Golden from Spike MX (uses vfwmacc + vfredosum).
 */
#include <stdio.h>
#include <stdint.h>
#include "../../common-data-gen/mx_data_gen.h"

#define N 64

void test_dot(char *name, size_t esize, size_t osize, double min, double max,
              mx_generator gen, fp_t (*dot_fn)(fp_t *, fp_t *, size_t)) {
	fp_t a[N];
	fp_t b[N];
	fp_t out;
	fp_t out_arr[4] = {0};

	for (size_t i = 0; i < N; i++) {
		a[i] = gen(GM_RAND, min, max);
		b[i] = gen(GM_RAND, min, max);
	}
	a[0] = gen(GM_ZERO, min, max);
	b[1] = gen(GM_ZERO, min, max);

	out = dot_fn(a, b, N);
	out_arr[0] = out;

	print_array(name, "_a", a, esize, N);
	print_array(name, "_b", b, esize, N);
	print_array(name, "_out", out_arr, osize, 4 / osize);
}

int main() {
	print_header();
	print_uint32("N", N);

	test_dot("fp16_dot", 2, 4, -1e2, 1e2, gen_fp16, fp16_dot);
	test_dot("bf16_dot", 2, 4, -1e2, 1e2, gen_bf16, bf16_dot);

	return 0;
}
