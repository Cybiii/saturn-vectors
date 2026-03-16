.section .data,"aw",@progbits
.global N
.balign 64
N:
    .word 0x00000040
    .word 0x00000000
.global fp16_dot_a
.balign 64
fp16_dot_a:
    .space 128
.global fp16_dot_b
.balign 64
fp16_dot_b:
    .space 128
.global fp16_dot_out
.balign 64
fp16_dot_out:
    .word 0x00000000
.global bf16_dot_a
.balign 64
bf16_dot_a:
    .space 128
.global bf16_dot_b
.balign 64
bf16_dot_b:
    .space 128
.global bf16_dot_out
.balign 64
bf16_dot_out:
    .word 0x00000000
