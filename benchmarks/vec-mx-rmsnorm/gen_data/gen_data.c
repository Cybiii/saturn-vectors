#include <stdio.h>
#include <stdint.h>
#include "../../common-data-gen/mx_data_gen.h"

#define N 64
#define EPS 1e-5f

void test_rmsnorm(char *name, double min, double max, mx_generator gen,
                  void (*fn)(fp_t *, const fp_t *, size_t, float)) {
	fp_t x[N], out[N];

	for (size_t i = 0; i < N; i++)
		x[i] = gen(GM_RAND, min, max);
	x[0] = gen(GM_ZERO, min, max);

	fn(out, x, N, EPS);

	print_array(name, "_x", x, 2, N);
	print_array(name, "_out", out, 2, N);
}

int main() {
	print_header();
	print_uint32("N", N);
	printf(".global eps\n.balign 4\neps:\n    .word 0x%08X\n", *(uint32_t *)&EPS);
	test_rmsnorm("fp16_rmsnorm", -1e2, 1e2, gen_fp16, fp16_rmsnorm);
	test_rmsnorm("bf16_rmsnorm", -1e2, 1e2, gen_bf16, bf16_rmsnorm);
	return 0;
}

