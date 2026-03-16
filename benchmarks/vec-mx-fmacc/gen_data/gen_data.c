#include <stdio.h>
#include <stdint.h>
#include "../../common-data-gen/mx_data_gen.h"

#define COUNT 128
#define SPECIAL_COUNT 20

void test(size_t isize, size_t osize, char *name, double min, double max, double sn_min, double sn_max,
          mx_generator gen, mx_op_ternary op) {

	fp_t vals_a[COUNT];
	fp_t vals_b[COUNT];
	fp_t vals_c[COUNT];
	fp_t vals_out[COUNT];

	for (size_t i = 0; i < COUNT; i++) {
		vals_a[i] = gen(GM_RAND, min, max);
		vals_b[i] = gen(GM_RAND, min, max);
		vals_c[i] = gen(GM_RAND, min, max);
	}

	for (size_t i = 9; i < SPECIAL_COUNT; i++) {
		vals_a[i] = gen(GM_RAND, sn_min, sn_max);
		vals_b[i] = gen(GM_RAND, sn_min, sn_max);
		vals_c[i] = gen(GM_RAND, sn_min, sn_max);
	}

	vals_a[0] = gen(GM_INF, min, max);
	vals_a[1] = gen(GM_NAN, min, max);
	vals_b[2] = gen(GM_INF, min, max);
	vals_b[3] = gen(GM_NAN, min, max);
	vals_a[4] = gen(GM_NINF, min, max);
	vals_a[5] = gen(GM_ZERO, min, max);
	vals_a[6] = gen(GM_NZERO, min, max);
	vals_a[7] = gen(GM_INF, min, max);
	vals_b[7] = gen(GM_NAN, min, max);
	vals_a[8] = gen(GM_INF, min, max);
	vals_b[8] = gen(GM_ZERO, min, max);

	for (size_t i = 0; i < COUNT; i++) {
		vals_out[i] = op(vals_a[i], vals_b[i], vals_c[i]);
	}

	print_array(name, "_a", vals_a, isize, COUNT);
	print_array(name, "_b", vals_b, isize, COUNT);
	print_array(name, "_c", vals_c, isize, COUNT);
	print_array(name, "_out", vals_out, osize, COUNT);
}

int main() {

	print_header();

	print_uint32("N", COUNT);

	test(1, 1, "e4m3_fmacc", -3e1, 3e1, -1e-1, 1e-1, gen_e4m3, e4m3_fmacc);
	test(1, 1, "e5m2_fmacc", -1e2, 1e2, -1e-4, 1e-4, gen_e5m2, e5m2_fmacc);

	return 0;
}
