#include "lo_float.h"

template <int exp, int sig>
struct Std_InfChecker {
	bool operator()(uint32_t bits) const {
		return (bits & ((1 << (exp + sig)) - 1)) == ((1 << exp) - 1) << sig;
	}
	
	uint32_t minNegInf() const {
		return ((1 << (exp + 1)) - 1) << sig;
	}

	uint32_t minPosInf() const {
		return ((1 << exp) - 1) << sig;
	}
};

template <int exp, int sig>
struct Std_NaNChecker {
	bool operator()(uint32_t bits) const {
		return (bits & ((1 << exp) - 1) << sig) == ((1 << exp) - 1) << sig && (bits & ((1 << sig) - 1)) != 0;
	}
	
	uint32_t qNanBitPattern() const {
		return ((1 << (exp + 1)) - 1) << (sig - 1);
	}

	uint32_t sNanBitPattern() const {
		return ((1 << (exp + 1)) - 1) << (sig - 1);
	}
};

template <int exp, int sig>
struct FN_InfChecker {
	bool operator()(uint32_t bits) const {
		return false;
	}
	
	uint32_t minNegInf() const {
		return (1 << (exp + sig)) - 1;
	}

	uint32_t minPosInf() const {
		return (1 << (exp + sig)) - 1;
	}
};

template <int exp, int sig>
struct FN_NaNChecker {
	bool operator()(uint32_t bits) const {
		return (bits & ((1 << (exp + sig)) - 1)) == (1 << (exp + sig)) - 1;
	}
	
	uint32_t qNanBitPattern() const {
		return (1 << (exp + sig)) - 1;
	}

	uint32_t sNanBitPattern() const {
		return (1 << (exp + sig)) - 1;
	}
};

template <int exp, int sig, lo_float::Rounding_Mode rm=lo_float::Rounding_Mode::RoundToNearestEven>
constexpr lo_float::FloatingPointParams param_std(
	exp + sig + 1, // Total bitwidth
	sig, // Mantissa
	(1 << (exp - 1)) - 1, // Bias
	rm,
	lo_float::Inf_Behaviors::Extended,
	lo_float::NaN_Behaviors::QuietNaN,
	lo_float::Signedness::Signed,
	Std_InfChecker<exp, sig>(),
	Std_NaNChecker<exp, sig>(),
	1 // stoch_len
);

template <int exp, int sig, lo_float::Rounding_Mode rm=lo_float::Rounding_Mode::RoundToNearestEven>
constexpr lo_float::FloatingPointParams param_fn(
	exp + sig + 1, // Total bitwidth
	sig, // Mantissa
	(1 << (exp - 1)) - 1, // Bias
	rm,
	lo_float::Inf_Behaviors::Extended,
	lo_float::NaN_Behaviors::QuietNaN,
	lo_float::Signedness::Signed,
	FN_InfChecker<exp, sig>(),
	FN_NaNChecker<exp, sig>(),
	1 // stoch_len
);

template <lo_float::Rounding_Mode rm>
using fp32 = lo_float::Templated_Float<param_std<8, 23, rm>>;
template <lo_float::Rounding_Mode rm>
using fp16 = lo_float::Templated_Float<param_std<5, 10, rm>>;
template <lo_float::Rounding_Mode rm>
using bf16 = lo_float::Templated_Float<param_std<8, 7, rm>>;
template <lo_float::Rounding_Mode rm>
using ofp8e5m2 = lo_float::Templated_Float<param_std<5, 2, rm>>;
template <lo_float::Rounding_Mode rm>
using ofp8e4m3 = lo_float::Templated_Float<param_fn<4, 3, rm>>;