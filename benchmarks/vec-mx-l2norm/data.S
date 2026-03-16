.section .data,"aw",@progbits
.global N
.balign 64
N:
    .word 0x00000040
    .word 0x00000000
.global fp16_l2norm_x
.balign 64
fp16_l2norm_x:
    .space 128
.global fp16_l2norm_out
.balign 64
fp16_l2norm_out:
    .word 0x00000000
.global bf16_l2norm_x
.balign 64
bf16_l2norm_x:
    .space 128
.global bf16_l2norm_out
.balign 64
bf16_l2norm_out:
    .word 0x00000000
