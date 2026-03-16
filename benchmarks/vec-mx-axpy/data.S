.section .data,"aw",@progbits
.global N
.balign 64
N:
    .word 0x00000080
    .word 0x00000000
.global e4m3_axpy_alpha
.balign 64
e4m3_axpy_alpha:
    .word 0x00000000
.global e4m3_axpy_x
.balign 64
e4m3_axpy_x:
    .space 128
.global e4m3_axpy_y
.balign 64
e4m3_axpy_y:
    .space 128
.global e4m3_axpy_out
.balign 64
e4m3_axpy_out:
    .space 128
.global e5m2_axpy_alpha
.balign 64
e5m2_axpy_alpha:
    .word 0x00000000
.global e5m2_axpy_x
.balign 64
e5m2_axpy_x:
    .space 128
.global e5m2_axpy_y
.balign 64
e5m2_axpy_y:
    .space 128
.global e5m2_axpy_out
.balign 64
e5m2_axpy_out:
    .space 128
.global fp16_axpy_alpha
.balign 64
fp16_axpy_alpha:
    .word 0x00000000
.global fp16_axpy_x
.balign 64
fp16_axpy_x:
    .space 256
.global fp16_axpy_y
.balign 64
fp16_axpy_y:
    .space 256
.global fp16_axpy_out
.balign 64
fp16_axpy_out:
    .space 256
.global bf16_axpy_alpha
.balign 64
bf16_axpy_alpha:
    .word 0x00000000
.global bf16_axpy_x
.balign 64
bf16_axpy_x:
    .space 256
.global bf16_axpy_y
.balign 64
bf16_axpy_y:
    .space 256
.global bf16_axpy_out
.balign 64
bf16_axpy_out:
    .space 256
