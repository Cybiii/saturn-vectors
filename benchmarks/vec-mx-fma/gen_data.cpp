#include <cstdio>
#include <iostream>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <functional>
#include <bitset>
#include "fp_types.hpp"

#define COUNT 128
#define SPECIAL_COUNT 20

const char fp32name[] = "fp32";
const char fp16name[] = "fp16";
const char bf16name[] = "bf16";
const char ofp8e5m2name[] = "ofp8e5m2";
const char ofp8e4m3name[] = "ofp8e4m3";

const char fp16Wname[] = "fp16W";
const char bf16Wname[] = "bf16W";
const char ofp8e5m2Wname[] = "ofp8e5m2W";
const char ofp8e4m3Wname[] = "ofp8e4m3W";

const char rm_rne[] = "rne";
const char rm_rtz[] = "rtz";
const char rm_rdn[] = "rdn";
const char rm_rup[] = "rup";
const char rm_rmm[] = "rmm";

std::mt19937 e2;

template <class T, class W, auto NAME, size_t WSIZE, auto RM_NAME>
void operand_test(T vals_a[], T vals_b[], T vals_c[], const char op_name[], std::function<double(double, double, double)> op_func) {
	std::cout << ".global " << NAME << '_' << op_name << '_' << RM_NAME << std::endl << ".balign 64" << std::endl << NAME << '_' << op_name << '_' << RM_NAME << ':' << std::endl;
	for (size_t i = 0; i < COUNT / (4 / WSIZE); i ++) {
		std::cout << "    .word 0x";
		for (int j = 4 / WSIZE - 1; j >= 0; j --) {
			double out_a = static_cast<double>(vals_a[i * (4 / WSIZE) + j]);
			double out_b = static_cast<double>(vals_b[i * (4 / WSIZE) + j]);
			double out_c = static_cast<double>(vals_c[i * (4 / WSIZE) + j]);
			std::cout << std::hex << std::setfill('0') << std::setw(WSIZE * 2) << (int) static_cast<W>(op_func(out_a, out_b, out_c)).rep();
		}
		std::cout << std::endl;
	}
}

template <class T, size_t SIZE, auto NAME, double MIN, double MAX, double SUBNORM_MIN, double SUBNORM_MAX, bool WIDEN, class W, auto RM_NAME>
void test() {

	const size_t WSIZE = WIDEN ? SIZE * 2 : SIZE;

	T vals_a[COUNT];
	T vals_b[COUNT];
	T vals_c[COUNT];

	std::uniform_real_distribution<> dist(MIN, MAX);
	std::uniform_real_distribution<> subnorm_dist(SUBNORM_MIN, SUBNORM_MAX);

	for (size_t i = 0; i < SPECIAL_COUNT; i ++) {
		vals_a[i] = static_cast<T>(subnorm_dist(e2));
		vals_b[i] = static_cast<T>(subnorm_dist(e2));
		vals_c[i] = static_cast<T>(subnorm_dist(e2));
	}

	for (size_t i = SPECIAL_COUNT; i < COUNT; i ++) {
		vals_a[i] = static_cast<T>(dist(e2));
		vals_b[i] = static_cast<T>(dist(e2));
		vals_c[i] = static_cast<T>(dist(e2));
	}

	vals_a[0] = static_cast<T>(INFINITY);
	vals_a[1] = static_cast<T>(NAN);

	vals_b[2] = static_cast<T>(INFINITY);
	vals_b[3] = static_cast<T>(NAN);

	vals_a[4] = static_cast<T>(-INFINITY);
	vals_a[5] = static_cast<T>(-NAN);

	vals_a[6] = static_cast<T>(INFINITY);
	vals_b[6] = static_cast<T>(NAN);

	vals_a[7] = static_cast<T>(0);

	vals_a[8] = static_cast<T>(INFINITY);
	vals_b[8] = static_cast<T>(0);

	std::cout << ".global " << NAME << "a_" << RM_NAME << std::endl << ".balign 64" << std::endl << NAME << "a_" << RM_NAME << ':' << std::endl;
	for (size_t i = 0; i < COUNT / (4 / SIZE); i ++) {
		std::cout << "    .word 0x";
		for (int j = 4 / SIZE - 1; j >= 0; j --) {
			std::cout << std::hex << std::setfill('0') << std::setw(SIZE * 2) << (int) vals_a[i * (4 / SIZE) + j].rep();
		}
		std::cout << std::endl;
	}

	std::cout << ".global " << NAME << "b_" << RM_NAME << std::endl << ".balign 64" << std::endl << NAME << "b_" << RM_NAME << ':' << std::endl;
	for (size_t i = 0; i < COUNT / (4 / SIZE); i ++) {
		std::cout << "    .word 0x";
		for (int j = 4 / SIZE - 1; j >= 0; j --) {
			std::cout << std::hex << std::setfill('0') << std::setw(SIZE * 2) << (int) vals_b[i * (4 / SIZE) + j].rep();
		}
		std::cout << std::endl;
	}

	std::cout << ".global " << NAME << "c_" << RM_NAME << std::endl << ".balign 64" << std::endl << NAME << "c_" << RM_NAME << ':' << std::endl;
	for (size_t i = 0; i < COUNT / (4 / SIZE); i ++) {
		std::cout << "    .word 0x";
		for (int j = 4 / SIZE - 1; j >= 0; j --) {
			std::cout << std::hex << std::setfill('0') << std::setw(SIZE * 2) << (int) vals_c[i * (4 / SIZE) + j].rep();
		}
		std::cout << std::endl;
	}

	operand_test<T, W, NAME, WSIZE, RM_NAME>(vals_a, vals_b, vals_c, "add", [](double a, double b, double c) -> double {return a + b;});
	operand_test<T, W, NAME, WSIZE, RM_NAME>(vals_a, vals_b, vals_c, "mul", [](double a, double b, double c) -> double {return a * b;});
	operand_test<T, W, NAME, WSIZE, RM_NAME>(vals_a, vals_b, vals_c, "sub", [](double a, double b, double c) -> double {return a - b;});
	operand_test<T, W, NAME, WSIZE, RM_NAME>(vals_a, vals_b, vals_c, "macc", [](double a, double b, double c) -> double {return a * b + c;});
	operand_test<T, W, NAME, WSIZE, RM_NAME>(vals_a, vals_b, vals_c, "ident", [](double a, double b, double c) -> double {return a;});
}

#define GEN_ROUND_MODE(rm, rm_name) \
	test<fp32<rm>, 4, fp32name, -1e2, 1e2, -1e-6, 1e-6, false, fp32<rm>, rm_name>(); \
	test<fp16<rm>, 2, fp16name, -1e2, 1e2, -1e-6, 1e-6, false, fp16<rm>, rm_name>(); \
	test<bf16<rm>, 2, bf16name, -1e15, 1e15, -2e-38, 2e-38, false, bf16<rm>, rm_name>(); \
	test<ofp8e5m2<rm>, 1, ofp8e5m2name, -1e2, 1e2, -1e-4, 1e-4, false, ofp8e5m2<rm>, rm_name>(); \
	test<ofp8e4m3<rm>, 1, ofp8e4m3name, -3e1, 3e1, -1e-1, 1e-1, false, ofp8e4m3<rm>, rm_name>(); \
	test<fp16<rm>, 2, fp16Wname, -1e2, 1e2, -1e-6, 1e-6, true, fp32<rm>, rm_name>(); \
	test<bf16<rm>, 2, bf16Wname, -1e15, 1e15, -2e-38, 2e-38, true, fp32<rm>, rm_name>(); \
	test<ofp8e5m2<rm>, 1, ofp8e5m2Wname, -1e2, 1e2, -1e-4, 1e-4, true, bf16<rm>, rm_name>(); \
	test<ofp8e4m3<rm>, 1, ofp8e4m3Wname, -1e2, 1e2, -1e-3, 1e-3, true, bf16<rm>, rm_name>(); \

int main() {

	
	
	exit(0);

	e2.seed(0);

	std::cout << ".section .data,\"aw\",@progbits" << std::endl;

	std::cout << ".global N" << std::endl \
		<< ".balign 8" << std::endl \
		<< "N:" << std::endl \
		<< "    .word 0x" << std::hex << std::setfill('0') << std::setw(8) << COUNT << std::endl \
		<< "    .word 0x00000000" << std::endl;
	
	GEN_ROUND_MODE(lo_float::Rounding_Mode::RoundToNearestEven, rm_rne)
	GEN_ROUND_MODE(lo_float::Rounding_Mode::RoundTowardsZero, rm_rtz)
	GEN_ROUND_MODE(lo_float::Rounding_Mode::RoundDown, rm_rdn)
	GEN_ROUND_MODE(lo_float::Rounding_Mode::RoundUp, rm_rup)
	GEN_ROUND_MODE(lo_float::Rounding_Mode::RoundTiesToAway, rm_rmm)

	return 0;
}