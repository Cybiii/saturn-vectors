package saturn.backend

import chisel3._
import chisel3.util._
import chisel3.experimental.dataview._
import org.chipsalliance.cde.config._
import freechips.rocketchip.rocket._
import freechips.rocketchip.util._
import freechips.rocketchip.tile._
import saturn.common._
import saturn.insns._
import scala.math._
import saturn.exu._

class BDotSequencerIO(pipe_depth: Int, acc_delay: Int)(implicit p: Parameters) extends SequencerIO(new BDotSequencerControl) with HasVectorParams {
  val rvs1 = Decoupled(new VectorReadReq)
  val rvs2 = Decoupled(new VectorReadReq)
  val rvd  = Decoupled(new VectorReadReq)
  val rvm  = Decoupled(new VectorReadReq)
  val pipe_write_req = new VectorPipeWriteReqIO(pipe_depth + acc_delay * (vLen/dLen) * 8)
  val batch_read_vs2 = Output(Bool())
  val tail = Output(Bool())
}

class BDotSequencer(pipe_depth: Int, acc_delay: Int)(implicit p: Parameters) extends Sequencer[BDotSequencerControl]()(p) with HasVectorParams with HasCoreParameters {

  val bdot_insns = vParams.bdotInsns

  def accepts(inst: VectorIssueInst) = !inst.vmu && new VectorDecoder(inst, bdot_insns, Nil).matched

  val io = IO(new BDotSequencerIO(pipe_depth, acc_delay))

  // Current instruction
  val valid = RegInit(false.B)
  val busy = RegInit(false.B)
  val inst = Reg(new BackendIssueInst)
  
  val eg_idx = RegInit(0.U(log2Ceil(maxVLMax).W))
  val head = Reg(Bool())

  val set_acc = RegInit(false.B)
  val set_acc_zero = Reg(Bool())
  val set_acc_bc = Reg(Bool())
  val writeback = RegInit(false.B)
  val acc_sel = Reg(UInt(5.W))

  val wvd_mask = Reg(UInt(egsTotal.W))
  val rvs1_mask = Reg(UInt(egsTotal.W))
  val rvs2_mask = Reg(UInt(egsTotal.W))
  val rvd_mask = Reg(UInt(egsTotal.W))
  val rvm_mask = Reg(UInt(egsTotal.W))

  val fp = Reg(Bool())
  val altfmt = Reg(Bool())
  val signed = Reg(Bool())
  val batched = Reg(Bool())
  val widen = Reg(UInt(2.W))

  val emul = Reg(UInt(2.W))
  val in_eew = Reg(UInt(2.W))
  // val acc_eew = in_eew + widen
  val renv1 = Reg(Bool())
  val renv2 = Reg(Bool())
  val renvd = Reg(Bool())
  val renvm = Reg(Bool())
  val vl = Reg(UInt((1+log2Ceil(maxVLMax)).W))

  val eg_idx_max = (vLen / dLen).U << emul

  val next_eg_idx = eg_idx +& 1.U
  val elements_width = (dLen / 8).U >> in_eew
  val next_vl = Mux(vl >= elements_width, vl - elements_width, 0.U)

  val tail = (next_eg_idx === eg_idx_max) && io.iss.fire

  // TODO: Is acc_delay correct? Can this be optimized further?
  // val tail_pipe = Pipe(io.iss.fire && tail, 0.U, acc_delay)

  io.dis.ready := (!busy || tail) /*(!valid || (tail && io.iss.fire))*/ && !io.dis_stall && io.iss.ready // TODO: Optimize

  // New instruction
  when (io.dis.fire) {
    val dis_inst = io.dis.bits

    assert(dis_inst.vstart === 0.U)

    val dis_ctrl = new VectorDecoder(dis_inst, bdot_insns, Seq(BDotSet, BDotSetZero, BDotSetBC, BDotWB, BDotSigned, BDotFP, BDotBatched, BDotWiden))

    set_acc := dis_ctrl.bool(BDotSet)
    set_acc_zero := dis_ctrl.bool(BDotSetZero)
    set_acc_bc := dis_ctrl.bool(BDotSetBC)
    writeback := dis_ctrl.bool(BDotWB)
    fp := dis_ctrl.bool(BDotFP)
    altfmt := dis_inst.vconfig.vtype.altfmt
    signed := dis_ctrl.bool(BDotSigned)
    batched := dis_ctrl.bool(BDotBatched)
    widen := dis_ctrl.uint(BDotWiden)

    val dis_vd_arch_mask  = get_arch_mask(dis_inst.rd, 0.U) // TODO: Multi-vd outputs
    val dis_vs1_arch_mask = get_arch_mask(dis_inst.rs1, Mux(dis_ctrl.bool(BDotBatched), 0.U, dis_inst.emul))
    val dis_vs2_arch_mask = get_arch_mask(dis_inst.rs2, Mux(dis_ctrl.bool(BDotBatched), 3.U, dis_inst.emul))

    valid := true.B
    busy := true.B
    inst := io.dis.bits

    renv1 := !(dis_ctrl.bool(BDotSet) || dis_ctrl.bool(BDotWB))
    renv2 := !((dis_ctrl.bool(BDotSet) && !(dis_ctrl.bool(BDotSetZero))) || dis_ctrl.bool(BDotWB))
    renvd := dis_ctrl.bool(BDotSet) && !dis_ctrl.bool(BDotSetZero)
    renvm := false.B

    wvd_mask      := FillInterleaved(egsPerVReg, dis_vd_arch_mask)
    rvs1_mask     := FillInterleaved(egsPerVReg, dis_vs1_arch_mask)
    rvs2_mask     := FillInterleaved(egsPerVReg, dis_vs2_arch_mask)
    rvd_mask      := FillInterleaved(egsPerVReg, dis_vd_arch_mask)
    rvm_mask      := Mux(renvm, ~(0.U(egsPerVReg.W)), 0.U)

    emul := Mux(batched, 0.U, dis_inst.emul)
    in_eew := dis_inst.vconfig.vtype.vsew
    vl := dis_inst.vconfig.vl

    acc_sel := Mux(dis_ctrl.bool(BDotSet) || dis_ctrl.bool(BDotWB), dis_inst.rs2, dis_inst.rd)
    
    head := true.B
  } .elsewhen (io.iss.fire) {
    valid := !tail
    head := false.B
  }

  when (tail && !io.dis.fire) {
    busy := false.B
  }
  io.tail := tail

  io.vat := inst.vat
  io.seq_hazard.valid := busy // TODO: Make this chain well between instructions
  io.seq_hazard.bits.rintent := hazardMultiply(rvs1_mask | rvs2_mask | rvd_mask | rvm_mask)
  io.seq_hazard.bits.wintent := hazardMultiply(wvd_mask)
  io.seq_hazard.bits.vat := inst.vat

  val wvd_eg = (inst.rd << log2Ceil(egsPerVReg)) + eg_idx

  val vs1_read_oh = Mux(renv1, UIntToOH(io.rvs1.bits.eg), 0.U)
  val vs2_oh = UIntToOH(io.rvs2.bits.eg)
  val vs2_oh_batched = vs2_oh.asTypeOf(Vec(4, Vec(8, UInt(egsPerVReg.W)))).map { grp =>
    Fill(8, grp(0))
  }.asUInt
  val vs2_read_oh = Mux(renv2, Mux(batched, vs2_oh_batched, vs2_oh), 0.U)
  dontTouch(vs2_read_oh)
  val vd_read_oh  = Mux(renvd, UIntToOH(io.rvd.bits.eg), 0.U)
  val vm_read_oh  = Mux(renvm, UIntToOH(io.rvm.bits.eg), 0.U)
  val vd_write_oh = Mux(writeback, UIntToOH(wvd_eg), 0.U)

  val raw_hazard = ((vs1_read_oh | vs2_read_oh | vd_read_oh | vm_read_oh) & io.older_writes) =/= 0.U
  val waw_hazard = (vd_write_oh & io.older_writes) =/= 0.U
  val war_hazard = (vd_write_oh & io.older_reads) =/= 0.U
  val data_hazard = raw_hazard || waw_hazard || war_hazard

  val oldest = inst.vat === io.vat_head

  io.rvs1.valid := valid && renv1
  io.rvs1.bits.eg := (inst.rs1 << log2Ceil(egsPerVReg)) + eg_idx
  io.rvs2.valid := valid && renv2
  io.rvs2.bits.eg := (Mux(batched, inst.rs2 & "h1F".U(5.W), inst.rs2) << log2Ceil(egsPerVReg)) + eg_idx
  io.rvd.valid := valid && renvd
  io.rvd.bits.eg := (inst.rd << log2Ceil(egsPerVReg)) + eg_idx
  io.rvm.valid := valid && renvm
  io.rvm.bits.eg := eg_idx + (eg_idx >> in_eew)
  io.batch_read_vs2 := batched && valid && renv2

  io.rvs1.bits.oldest := oldest
  io.rvs2.bits.oldest := oldest
  io.rvd.bits.oldest := oldest
  io.rvm.bits.oldest := oldest

  io.pipe_write_req.request := valid && writeback
  io.pipe_write_req.bank_sel := (if (vrfBankBits == 0) 1.U else UIntToOH(wvd_eg(vrfBankBits,1)))
  io.pipe_write_req.pipe_depth := 0.U
  io.pipe_write_req.oldest := oldest
  io.pipe_write_req.fire := io.iss.fire

  when (io.iss.fire) {
    when (!tail) {
      eg_idx := next_eg_idx
      vl := next_vl
    } .otherwise {
      eg_idx := 0.U
    }
  }

  dontTouch(vs1_read_oh)
  dontTouch(data_hazard)
  dontTouch(io.rvs1.ready)
  dontTouch(io.rvs2.ready)
  dontTouch(io.rvd.ready)

  io.iss.valid := (valid &&
    !data_hazard &&
    io.rvs1.ready &&
    io.rvs2.ready &&
    io.rvd.ready &&
    !(renvm && !io.rvm.ready)
  )
  io.iss.bits.base_eg := wvd_eg
  io.iss.bits.fp := fp
  io.iss.bits.altfmt := altfmt
  io.iss.bits.signed := signed
  io.iss.bits.batched := batched
  io.iss.bits.emul := emul
  io.iss.bits.in_eew := in_eew
  // io.iss.bits.acc_eew := acc_eew
  io.iss.bits.set_acc := set_acc
  io.iss.bits.set_acc_zero := set_acc_zero
  io.iss.bits.set_acc_bc := set_acc_bc
  io.iss.bits.writeback := writeback
  io.iss.bits.acc_sel := acc_sel
  io.iss.bits.src_valid := !(set_acc || writeback)
  io.iss.bits.vl := vl

  io.busy := busy
  io.head := head
}