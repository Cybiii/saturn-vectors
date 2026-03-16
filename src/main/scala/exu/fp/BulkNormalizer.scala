package saturn.exu

import chisel3._
import chisel3.util._
/*
object maxExp {
  def apply(in: Vec[SInt], n: Int): SInt = {
    if (n == 1) {
      in(0)
    }
    else {
      val half = n / 2
      val l = maxExp(in(n - 1, half), n - half)
      val r = maxExp(in(half - 1, 0), half)
      if (l >= r) l else r
    }
  }
}

class BulkNormalizerIntermediates(val input_type: FType, val n: Int) extends Bundle {
  val prod_signs = Output(Vec(n, Bool()))
  val prod_exps = Output(Vec(n, SInt((input_type.exp + 3).W)))
  val prod_sigs = Output(Vec(n, UInt(((input_type.sig + 1) * 2).W)))
  
  val any_nan = Output(Bool())
  val any_inf = Output(Bool())
  val any_pos_inf = Output(Bool())
  val any_neg_inf = Output(Bool())
}

class BulkNormalizerMultiplier(input_type: FType, n: Int)(implicit p: Parameters) extends Module {
  
  val io = IO(new Bundle {
    val in_a = Input(Vec(n, UInt(input_type.ieeeWidth.W)))
    val in_b = Input(Vec(n, UInt(input_type.ieeeWidth.W)))

    val out = Output(new BulkNormalizerIntermediates(input_type, n))
  })

  val rec_a = in_a.map { e => rawFloatFromFN(input_type.exp, input_type.sig, e) }
  val rec_b = in_b.map { e => rawFloatFromFN(input_type.exp, input_type.sig, e) }

  val prod_invalid_nan = Wire(Vec(n, Bool()))
  val prod_nan = Wire(Vec(n, Bool()))
  val prod_inf = Wire(Vec(n, Bool()))
  val prod_pos_inf = Wire(Vec(n, Bool()))
  val prod_neg_inf = Wire(Vec(n, Bool()))

  (0 until n).foreach { i =>
    val a = rec_a(i)
    val b = rec_b(i)

    val sign = a.sign ^ b.sign
    val invalid_nan = (a.isInf && b.isZero) || (b.isInf && a.isZero)
    prod_invalid_nan(i) := invalid_nan
    prod_nan(i) := invalid_nan || a.isNaN || b.isNaN
    val inf = a.isInf || b.isInf
    prod_inf(i) := inf
    prod_pos_inf(i) := inf && !sign
    prod_neg_inf(i) := inf && sign
    // TODO: sigNan might matter

    io.out.prod_signs(i) := sign
    io.out.prod_exps(i) := a.sExp +& b.sExp
    io.out.prod_sigs(i) := a.sig * b.sig
  }

  val any_pos_inf = prod_pos_inf.asBools.orR
  val any_neg_inf = prod_neg_inf.asBools.orR

  io.out.any_nan := prod_nan.asBools.orR
  io.out.any_inf := prod_inf.asBools.orR
  io.out.any_pos_inf := prod_pos_inf.asBools.orR
  io.out.any_neg_inf := prod_neg_inf.asBools.orR
}

class BulkNormalizerAccumulator(input_type: FType, acc_type: FType, n_elems: Int, accumulate: Bool)(implicit p: Parameters) extends Module {
  val n = n_elems + (if (accumulate) 1 else 0)
  val p = input_type.sig - 1
  val q = acc_type.sig - 1
  val g = log2Ceil(n)

  val io = IO(new Bundle {
    val inter = Input(new BulkNormalizerIntermediates(input_type, n_elems))
    val acc_in = Input(new RawFloat(acc_type.exp, acc_type.sig))

    val acc_out = Output(new RawFloat(acc_type.exp, acc_type.sig))
  })

  val invalid = io.inter.any_nan ||
    (
      (io.inter.any_pos_inf || (accumulate.B && io.acc_in.isInf && !io.acc_in.sign)) &&
      (io.inter.any_neg_inf || (accumulate.B && io.acc_in.isInf && io.acc_in.sign))
    )
  io.acc_out.isNaN := invalid || io.inter.any_nan || (accumulate.B && io.acc_in.isNaN)
  io.acc_out.isInf := io.inter.any_inf || (accumulate.B && io.acc_in.isInf)

  val signs = io.inter.prod_signs ++ (if (accumulate) acc_in.sign else Nil)
  val exps = io.inter.prod_exps ++ (if (accumulate) acc_in.sExp else Nil)
  val sigs = io.inter.prod_sigs ++ (if (accumulate) acc_in.sig else Nil)

  val max_exp = maxExp(exps, n)

  val pad_right = q + 1 + g - (2 * p)

  val aligned_sigs = Wire(Vec(n, SInt(((input_type.sig + 1) * 2 + 1).W)))

  (0 until n).map { i =>
    val exp = exps(i)
    val sig = sigs(i)
    val shift = (max_exp - exp).asUInt
    
    val aligned_sig_unrounded = (sig << pad_right) >> shift
    val discard_mask = ((1 << (2 * p)) - 1).U >> ((q + 1 + g).U - shift)
    val discard_bits = sig & discard_mask
    val jam = (if (shift >= (q + 1 + g)) sig else discard_bits) =/= 0.U
    val aligned_sig = (aligned_sig_unrounded | Mux(jam, 1.U, 0.U)).asSInt

    aligned_sigs(i) := Mux(signs(i), -aligned_sig, aligned_sig)
  }

  val acc = aligned_sigs.foldLeft(0.S) { (x, y) => x + y }
  
}
*/