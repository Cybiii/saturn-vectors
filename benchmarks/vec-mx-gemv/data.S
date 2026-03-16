.section .data,"aw",@progbits
.global M
.balign 64
M:
    .word 0x00000020
    .word 0x00000000
.global K
.balign 64
K:
    .word 0x00000020
    .word 0x00000000
.global e4m3_gemv_A
.balign 64
e4m3_gemv_A:
    .space 1024
.global e4m3_gemv_x
.balign 64
e4m3_gemv_x:
    .space 32
.global e4m3_gemv_out
.balign 64
e4m3_gemv_out:
    .space 64
.global e5m2_gemv_A
.balign 64
e5m2_gemv_A:
    .space 1024
.global e5m2_gemv_x
.balign 64
e5m2_gemv_x:
    .space 32
.global e5m2_gemv_out
.balign 64
e5m2_gemv_out:
    .space 64
.global fp16_gemv_A
.balign 64
fp16_gemv_A:
    .space 2048
.global fp16_gemv_x
.balign 64
fp16_gemv_x:
    .space 64
.global fp16_gemv_out
.balign 64
fp16_gemv_out:
    .space 128
.global bf16_gemv_A
.balign 64
bf16_gemv_A:
    .space 2048
.global bf16_gemv_x
.balign 64
bf16_gemv_x:
    .space 64
.global bf16_gemv_out
.balign 64
bf16_gemv_out:
    .space 128
