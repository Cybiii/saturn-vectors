#!/usr/bin/env python3
import math, struct, sys, json, argparse

# ---------- FP8 E4M3 encode/ decode (IEEE-like, bias=7) ----------
BIAS_E4M3 = 7

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

def fp8_e4m3_to_float(bits: int) -> float:
    bits &= 0xFF
    sign = -1.0 if (bits >> 7) & 1 else 1.0
    exp  = (bits >> 3) & 0xF
    frac = bits & 0x7
    if exp == 0:
        if frac == 0:
            return math.copysign(0.0, -1.0 if sign < 0 else 1.0) * 0.0
        return sign * math.ldexp(frac / 8.0, 1 - BIAS_E4M3)
    if exp == 0xF:
        if frac == 0:
            return sign * float('inf')
        return float('nan')
    return sign * math.ldexp(1.0 + (frac / 8.0), exp - BIAS_E4M3)

# ---------- float32 helpers ----------
def to_float32(x: float) -> float:
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]

def float32_to_int32_bits(x: float) -> int:
    u = int.from_bytes(struct.pack('<f', float(x)), 'little', signed=False)
    return u - (1 << 32) if u >= (1 << 31) else u

def int8_signed(u8: int) -> int:
    u8 &= 0xFF
    return u8 - 256 if u8 >= 128 else u8

# ---------- Outer product with accumulate ----------
# CHANGED: added repeats to accumulate identical outer-product results before adding bias.
def outer_product_fp8_e4m3(a_fp, b_fp, acc_scalar=1.5, repeats=2):
    """
    Computes: sum_{r=1..repeats} (a ⊗ b)  +  acc_scalar
    All arithmetic is simulated in float32 with round-to-nearest-even on each add.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    acc_s32 = to_float32(acc_scalar)
    a_bits_u8 = [float_to_fp8_e4m3(x) for x in a_fp]
    b_bits_u8 = [float_to_fp8_e4m3(x) for x in b_fp]
    a_quant = [fp8_e4m3_to_float(u) for u in a_bits_u8]
    b_quant = [fp8_e4m3_to_float(u) for u in b_bits_u8]
    a_q32 = [to_float32(x) for x in a_quant]
    b_q32 = [to_float32(x) for x in b_quant]

    C_float32, C_int32 = [], []
    for i in range(16):
        row_f, row_i = [], []
        for j in range(16):
            prod = to_float32(a_q32[i] * b_q32[j])
            acc = to_float32(0.0)
            for _ in range(repeats):
                acc = to_float32(acc + prod)
            out  = to_float32(acc + acc_s32)
            row_f.append(out)
            row_i.append(float32_to_int32_bits(out))
        C_float32.append(row_f)
        C_int32.append(row_i)

    return {
        "repeats": repeats,
        "a_input": a_fp,              # NEW: keep originals for printing
        "b_input": b_fp,              # NEW
        "a_bits": [int8_signed(u) for u in a_bits_u8],
        "b_bits": [int8_signed(u) for u in b_bits_u8],
        "a_bits_bin": [format(u, '08b') for u in a_bits_u8],
        "b_bits_bin": [format(u, '08b') for u in b_bits_u8],
        "a_quant": a_q32,
        "b_quant": b_q32,
        "C_float32": C_float32,
        "C_int32": C_int32,
    }

# CHANGED: pretty printer mentions repeats
def pretty_print_result_outer(res, comma_int32=False, acc_scalar=1.5):
    def fmtf(x: float) -> str: return f"{x:.6g}"
    repeats = res.get("repeats", 1)
    print(f"=== Outer Product: result = sum_{{r=1..{repeats}}}(fp8 * fp8) + {acc_scalar} (all in float32) ===")
    print("=== Inputs encoded to FP8 E4M3 (as signed int8 and 8-bit binary) ===")
    print("a_bits (int8):", res["a_bits"])
    print("a_bits (bin) :", res["a_bits_bin"])
    print("b_bits (int8):", res["b_bits"])
    print("b_bits (bin) :", res["b_bits_bin"])

    # NEW: copy-friendly A, B, and At (A is a 16x1 vector → At is 1x16)
    if "a_input" in res and "b_input" in res:
        print("\n=== A (comma-separated) ===")
        print("  ", ", ".join(fmtf(x) for x in res["a_input"]))
        print("=== At (comma-separated) ===")
        print("  ", ", ".join(fmtf(x) for x in res["a_input"]))  # At is row form
        print("=== B (comma-separated) ===")
        print("  ", ", ".join(fmtf(x) for x in res["b_input"]))

    print("\n=== Quantized inputs (decoded FP8 -> float32) ===")
    print("a_quant:", [fmtf(x) for x in res["a_quant"]])
    print("b_quant:", [fmtf(x) for x in res["b_quant"]])

    print(f"\n=== C = sum_{{r=1..{repeats}}}(a ⊗ b) + acc (float32) ===")
    for row in res["C_float32"]:
        print("  ", " ".join(f"{fmtf(x):>10}" for x in row))

    print("\n=== C as int32 raw bit patterns (two's complement) ===")
    for row in res["C_int32"]:
        print("  ", ",".join(f"{x:>11d}" for x in row), ",")

    if comma_int32:
        print("\n=== C as int32 (comma-separated for easy copy) ===")
        for row in res["C_int32"]:
            print("  ", ", ".join(str(x) for x in row))

# ---------- Matrix multiply with accumulate (unchanged) ----------
def quantize_matrix_fp8_e4m3(M):
    bits = [[float_to_fp8_e4m3(x) for x in row] for row in M]
    q = [[to_float32(fp8_e4m3_to_float(b)) for b in row] for row in bits]
    bits_i8 = [[int8_signed(b) for b in row] for row in bits]
    bits_bin = [[format(b & 0xFF, '08b') for b in row] for row in bits]
    return bits_i8, bits_bin, q

def matmul_fp8_e4m3(A_fp, B_fp, acc_scalar=1.5):
    # A: MxK, B: KxN
    M = len(A_fp)
    K = len(A_fp[0]) if M > 0 else 0
    K2 = len(B_fp)
    N = len(B_fp[0]) if K2 > 0 else 0
    if K2 != K:
        raise ValueError("Inner dims must match: A is MxK, B is KxN")

    acc_s32 = to_float32(acc_scalar)
    A_i8, A_bin, A_q32 = quantize_matrix_fp8_e4m3(A_fp)
    B_i8, B_bin, B_q32 = quantize_matrix_fp8_e4m3(B_fp)

    C_f32 = [[to_float32(0.0) for _ in range(N)] for _ in range(M)]
    C_i32 = [[0 for _ in range(N)] for _ in range(M)]
    for i in range(M):
        for j in range(N):
            acc = to_float32(0.0)
            for k in range(K):
                prod = to_float32(A_q32[i][k] * B_q32[k][j])
                acc  = to_float32(acc + prod)
            out = to_float32(acc + acc_s32)
            C_f32[i][j] = out
            C_i32[i][j] = float32_to_int32_bits(out)

    return {
        "A_input": A_fp,  # NEW: keep originals for printing
        "B_input": B_fp,  # NEW
        "A_bits": A_i8, "B_bits": B_i8,
        "A_bits_bin": A_bin, "B_bits_bin": B_bin,
        "A_quant": A_q32, "B_quant": B_q32,
        "C_float32": C_f32, "C_int32": C_i32
    }

def pretty_print_result_matmul(res, comma_int32=False, acc_scalar=1.5):
    def fmtf(x: float) -> str: return f"{x:.6g}"
    def csv(row): return ", ".join(str(v) for v in row)

    print(f"=== MatMul: result = (A@B) + {acc_scalar} (all in float32) ===")

    # ===== NEW: int8-encoded matrices in comma-separated form =====
    print("=== A encoded to FP8 E4M3 (int8) — comma-separated ===")
    for row in res["A_bits"]:
        print("  ", csv(row), ",")

    print("=== At (A transposed) encoded to FP8 E4M3 (int8) — comma-separated ===")
    for col in zip(*res["A_bits"]):  # transpose A_bits (int8)
        print("  ", csv(col), ",")

    print("=== B encoded to FP8 E4M3 (int8) — comma-separated ===")
    for row in res["B_bits"]:
        print("  ", csv(row), ",")
    # ===== end new section =====

    # (keep the rest as-is; these are float views & results)
    print("\n=== A_quant (decoded FP8 -> float32) ===")
    for row in res["A_quant"]: print("  ", [fmtf(x) for x in row])
    print("=== B_quant (decoded FP8 -> float32) ===")
    for row in res["B_quant"]: print("  ", [fmtf(x) for x in row])

    print("\n=== C = A @ B + acc (float32) ===")
    for row in res["C_float32"]:
        print("  ", " ".join(f"{fmtf(x):>10}" for x in row))

    print("\n=== C as int32 raw bit patterns (two's complement) ===")
    for row in res["C_int32"]:
        print("  ", ",".join(f"{x:>11d}" for x in row), ",")

    if comma_int32:
        print("\n=== C as int32 (comma-separated for easy copy) ===")
        for row in res["C_int32"]:
            print("  ", ", ".join(str(x) for x in row), ",")

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="FP8 E4M3 outer product and matmul utilities with accumulate.")
    # Outer product
    parser.add_argument("--a", type=str, default="", help="JSON list of 16 floats for vector a")
    parser.add_argument("--b", type=str, default="", help="JSON list of 16 floats for vector b")
    parser.add_argument("--outer", action="store_true", help="Run outer product with the given --a/--b or example")
    parser.add_argument("--repeats", type=int, default=2,  # CHANGED: new flag, default 2
                        help="Repeat the same outer product this many times before adding bias (default 2)")
    # Matrix multiply
    parser.add_argument("--A", type=str, default="", help="JSON 2D list for matrix A (MxK)")
    parser.add_argument("--B", type=str, default="", help="JSON 2D list for matrix B (KxN)")
    parser.add_argument("--matmul", action="store_true", help="Run matrix multiply with given --A/--B or example")
    # Accumulate
    parser.add_argument("--acc", type=float, default=1.5, help="Scalar to add after multiply (default 1.5)")
    # Output tweaks
    parser.add_argument("--comma-int32", action="store_true", help="Also print C_int32 rows as comma-separated lists for easy copy")
    parser.add_argument("--example-size", type=int, default=16, help="Size for example square matrices (default 16)")
    args = parser.parse_args()

    ran_any = False

    if args.outer:
        if args.a and args.b:
            a = json.loads(args.a)
            b = json.loads(args.b)
            if len(a) != 16 or len(b) != 16:
                print("Both --a and --b must be length-16 lists.", file=sys.stderr)
                sys.exit(1)
        else:
            a = [ -1.0 + (2.0 * i / 15.0) for i in range(16) ]
            b = [  1.0 - (2.0 * i / 15.0) for i in range(16) ]
        res = outer_product_fp8_e4m3(a, b, acc_scalar=args.acc, repeats=args.repeats)  # CHANGED
        pretty_print_result_outer(res, comma_int32=args.comma_int32, acc_scalar=args.acc)
        ran_any = True

    if args.matmul:
        if args.A and args.B:
            A = json.loads(args.A); B = json.loads(args.B)
        else:
            n = args.example_size
            A = [[-1.0 + 2.0*(i*n+j)/(n*n-1) for j in range(n)] for i in range(n)]
            B = [[ 1.0 - 2.0*(j*n+i)/(n*n-1) for j in range(n)] for i in range(n)]
        resM = matmul_fp8_e4m3(A, B, acc_scalar=args.acc)
        pretty_print_result_matmul(resM, comma_int32=args.comma_int32, acc_scalar=args.acc)
        ran_any = True

    if not ran_any:
        print("No mode selected; running both examples. Use --outer and/or --matmul to choose.\n")
        # Outer example (defaults to repeats=2 to reflect op(a,b)+op(a,b)+bias)
        a = [ -1.0 + (2.0 * i / 15.0) for i in range(16) ]
        b = [  1.0 - (2.0 * i / 15.0) for i in range(16) ]
        res = outer_product_fp8_e4m3(a, b, acc_scalar=1.5, repeats=2)  # CHANGED
        pretty_print_result_outer(res, comma_int32=True, acc_scalar=1.5)
        print("\n" + "="*80 + "\n")
        # Matmul example: 16x16 x 16x16
        n = 16
        A = [[-1.0 + 2.0*(i*n+j)/(n*n-1) for j in range(n)] for i in range(n)]
        B = [[ 1.0 - 2.0*(j*n+i)/(n*n-1) for j in range(n)] for i in range(n)]
        resM = matmul_fp8_e4m3(A, B, acc_scalar=1.5)
        pretty_print_result_matmul(resM, comma_int32=True, acc_scalar=1.5)

if __name__ == "__main__":
    main()
