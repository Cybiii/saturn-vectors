.section .data,"aw",@progbits
.global N
.balign 64
N:
    .word 0x00000040
    .word 0x00000000
.global fp16_max_x
.balign 64
fp16_max_x:
    .space 128
.global fp16_max_out
.balign 64
fp16_max_out:
    .word 0x00000000
.global bf16_max_x
.balign 64
bf16_max_x:
    .space 128
.global bf16_max_out
.balign 64
bf16_max_out:
    .word 0x00000000
