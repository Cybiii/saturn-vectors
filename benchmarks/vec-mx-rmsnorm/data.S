.section .data,"aw",@progbits
.global N
.balign 64
N:
    .word 0x00000040
    .word 0x00000000
.global eps
.balign 4
eps:
    .word 0x3727C5AC
.global fp16_rmsnorm_x
.balign 64
fp16_rmsnorm_x:
    .space 128
.global fp16_rmsnorm_out
.balign 64
fp16_rmsnorm_out:
    .space 128
.global bf16_rmsnorm_x
.balign 64
bf16_rmsnorm_x:
    .space 128
.global bf16_rmsnorm_out
.balign 64
bf16_rmsnorm_out:
    .space 128
