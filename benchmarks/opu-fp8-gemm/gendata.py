#!/usr/bin/env python3

import numpy as np
import math, struct, sys, json, argparse

# ---------- FP8 E4M3 encode/ decode (IEEE-like, bias=7) ----------
BIAS_E4M3 = 7
BIAS_E5M2 = 15

def _round_half_even(x: float) -> int:
    f = math.floor(x)
    r = x - f
    if r > 0.5: return f + 1
    if r < 0.5: return f
    return f if (f & 1) == 0 else f + 1

def float_to_fp8_e4m3(x: float) -> int:
    if math.isnan(x):
        return 0b01111101  # quiet NaN
    sign = 1 if math.copysign(1.0, x) < 0 else 0
    ax = abs(x)
    if math.isinf(ax):
        return (sign << 7) | (0xF << 3) | 0
    if ax == 0.0:
        return sign << 7

    m, e = math.frexp(ax)  # ax = m * 2^e, m in [0.5,1)
    E = e - 1
    y = m * 2.0           # in [1,2)

    exp_code = E + BIAS_E4M3
    if 1 <= exp_code <= 0xE:
        frac_unrounded = (y - 1.0) * 8.0
        frac = _round_half_even(frac_unrounded)
        if frac == 8:
            frac = 0
            exp_code += 1
            if exp_code >= 0xF:
                return (sign << 7) | (0xF << 3) | 0
        if exp_code > 0:
            return (sign << 7) | ((exp_code & 0xF) << 3) | (frac & 0x7)

    # subnormal
    frac_target = ax * 512.0  # 2^(BIAS_E4M3+2)
    frac = _round_half_even(frac_target)
    if frac == 0:
        return sign << 7
    if frac >= 8:
        return (sign << 7) | (0x1 << 3) | 0  # smallest normal
    return (sign << 7) | (0x0 << 3) | (frac & 0x7)

def float_to_fp8_e5m2(x: float) -> int:
    if math.isnan(x):
        return 0b01111101  # quiet NaN
    sign = 1 if math.copysign(1.0, x) < 0 else 0
    ax = abs(x)
    if math.isinf(ax):
        return (sign << 7) | (0xF << 3) | 0
    if ax == 0.0:
        return sign << 7

    m, e = math.frexp(ax)  # ax = m * 2^e, m in [0.5,1)
    E = e - 1
    y = m * 2.0           # in [1,2)

    exp_code = E + BIAS_E5M2
    if 1 <= exp_code <= 0xE:
        frac_unrounded = (y - 1.0) * 16.0
        frac = _round_half_even(frac_unrounded)
        if frac == 16:
            frac = 0
            exp_code += 1
            if exp_code >= 0x1F:
                return (sign << 7) | (0x1F << 2) | 0
        if exp_code > 0:
            return (sign << 7) | ((exp_code & 0x1F) << 2) | (frac & 0x3)

    # subnormal
    frac_target = ax * 2**(BIAS_E5M2+2)
    frac = _round_half_even(frac_target)
    if frac == 0:
        return sign << 7
    if frac >= 16:
        return (sign << 7) | (0x1 << 2) | 0  # smallest normal
    return (sign << 7) | (0x0 << 2) | (frac & 0x3)

def int8_signed(u8: int) -> int:
    u8 &= 0xFF
    return u8 - 256 if u8 >= 128 else u8

m_dim = 16
k_dim = 16
n_dim = 16

parser = argparse.ArgumentParser(description='A script to generate input data for an SGEMM kernel.')

parser.add_argument('--mdim', type=int, help='M dimension of inputs')
parser.add_argument('--kdim', type=int, help='K dimension of inputs')
parser.add_argument('--ndim', type=int, help='N dimension of inputs')
parser.add_argument('--size', type=int, help='Dimensions of NxN inputs')

args = parser.parse_args()

if args.size:
    m_dim = args.size
    k_dim = args.size
    n_dim = args.size
else:
    if args.mdim:
        m_dim = args.mdim
    if args.kdim:
        k_dim = args.kdim
    if args.ndim:
        n_dim = args.ndim

a_array_size = m_dim * k_dim
b_array_size = k_dim * n_dim
c_array_size = m_dim * n_dim

maxmant = 2**3 - 1
minexp = -6
maxexp = 6

# Generate floating-point values with exact mantissa and exponent
randf = lambda n: np.ldexp(
    np.random.randint(-1*maxmant, maxmant, size=n),
    np.random.randint(minexp, maxexp, size=n))

a_matrix = randf((m_dim, k_dim)).reshape(k_dim, m_dim)
b_matrix = randf((k_dim, n_dim)).reshape(k_dim, n_dim)
c_matrix = np.matmul(a_matrix.T, b_matrix)

a_e4m3 = np.array([int8_signed(float_to_fp8_e4m3(x)) for x in a_matrix.flatten()]).reshape(k_dim, m_dim)
b_e4m3 = np.array([int8_signed(float_to_fp8_e4m3(x)) for x in b_matrix.flatten()]).reshape(k_dim, n_dim)

print(f'''#define M_DIM {m_dim}
#define K_DIM {k_dim}
#define N_DIM {n_dim}

''')

def print_array(name, data, data_size, data_type='float', data_fmt='{}', fold=10):
    print(f"{name} [{data_size}] = {{")
    for i in range(0, len(data), fold):
        print('  ', ', '.join(data_fmt.format(x) for x in data[i:i+fold]), ',', sep='')
    print('};')

print_array('static int8_t a_matrix', a_e4m3.flatten(), 'M_DIM*K_DIM', data_type='int8_t')
print_array('static int8_t b_matrix', b_e4m3.flatten(), 'K_DIM*N_DIM', data_type='int8_t')
print_array('static float verify_data', c_matrix.flatten(), 'M_DIM*N_DIM', data_type='float')
