#include <cstdio>
#include <iostream>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <functional>
#include "fp_types.hpp"

#define COUNT 128
#define SPECIAL_COUNT 20

const char fp16name[] = "fp16";
const char bf16name[] = "bf16";
const char ofp8e5m2name[] = "ofp8e5m2";
const char ofp8e4m3name[] = "ofp8e4m3";

std::mt19937 e2;

template <class T, size_t NSIZE, auto NAME, double MIN, double MAX, class N>
void test() {

	const size_t WSIZE = NSIZE * 2;

	T vals[COUNT];
	N vals_out[COUNT];

	std::uniform_real_distribution<> dist(MIN, MAX);

	for (size_t i = 0; i < COUNT; i ++) {
		vals[i] = static_cast<T>(dist(e2));
	}

	vals[0] = static_cast<T>(INFINITY);
	vals[1] = static_cast<T>(NAN);

	vals[2] = static_cast<T>(-INFINITY);
	vals[3] = static_cast<T>(-NAN);

	vals[4] = static_cast<T>(0.0);
	vals[5] = static_cast<T>(-0.0);

    for (size_t i = 0; i < COUNT; i ++) {
		vals_out[i] = static_cast<N>(static_cast<double>(vals[i]));
	}

	std::cout << ".global " << NAME << std::endl << ".balign 64" << std::endl << NAME << ':' << std::endl;
	for (size_t i = 0; i < COUNT / (4 / WSIZE); i ++) {
		std::cout << "    .word 0x";
		for (int j = 4 / WSIZE - 1; j >= 0; j --) {
			std::cout << std::hex << std::setfill('0') << std::setw(WSIZE * 2) << (int) vals[i * (4 / WSIZE) + j].rep();
		}
		std::cout << std::endl;
	}

    std::cout << ".global " << NAME << "_out" << std::endl << ".balign 64" << std::endl << NAME << "_out:" << std::endl;
	for (size_t i = 0; i < COUNT / (4 / NSIZE); i ++) {
		std::cout << "    .word 0x";
		for (int j = 4 / NSIZE - 1; j >= 0; j --) {
			std::cout << std::hex << std::setfill('0') << std::setw(NSIZE * 2) << (int) vals_out[i * (4 / NSIZE) + j].rep();
		}
		std::cout << std::endl;
	}
}

int main() {

	using fp32 = lo_float::Templated_Float<param_std<8, 23>>;
	using fp16 = lo_float::Templated_Float<param_std<5, 10>>;
	using bf16 = lo_float::Templated_Float<param_std<8, 7>>;
	using ofp8e5m2 = lo_float::Templated_Float<param_std<5, 2>>;
	using ofp8e4m3 = lo_float::Templated_Float<param_fn<4, 3>>;

	e2.seed(0);

	std::cout << ".section .data,\"aw\",@progbits" << std::endl;

	std::cout << ".global N" << std::endl \
		<< ".balign 8" << std::endl \
		<< "N:" << std::endl \
		<< "    .word 0x" << std::hex << std::setfill('0') << std::setw(8) << COUNT << std::endl \
		<< "    .word 0x00000000" << std::endl;
	
	test<fp32, 2, fp16name, -1e2, 1e2, fp16>();
	test<fp32, 2, bf16name, -1e15, 1e15, bf16>();
	test<bf16, 1, ofp8e5m2name, -1e2, 1e2, ofp8e5m2>();
	test<bf16, 1, ofp8e4m3name, -3e1, 3e1, ofp8e4m3>();

	return 0;
}