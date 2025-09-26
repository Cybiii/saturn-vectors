#define LMUL_M1 "000"
#define LMUL_M2 "001"
#define LMUL_M4 "010"
#define LMUL_M8 "011"
#define LMUL_MF2 "111"
#define LMUL_MF4 "110"
#define LMUL_MF8 "101"

#define SEW_E8 "00"
#define SEW_E16 "01"
#define SEW_E32 "10"
#define SEW_E64 "11"

#define VSETVLI_ALTFMT(vl, avl, sew, lmul, alt) \
	asm volatile(".insn i 0x57, 0x7, %0, %1, 0b000" #alt "000" sew lmul : "=r"(vl) : "r"(avl))

#define VSETVLI_ALTFMT_X0(avl, sew, lmul, alt) \
	asm volatile(".insn i 0x57, 0x7, zero, %0, 0b000" #alt "000" sew lmul :: "r"(avl))
