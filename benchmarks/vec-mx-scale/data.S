.section .data,"aw",@progbits
.global N
.balign 64
N:
    .word 0x00000080
    .word 0x00000000
.global e4m3_scale_alpha
.balign 64
e4m3_scale_alpha:
    .word 0x00000000
.global e4m3_scale_x
.balign 64
e4m3_scale_x:
    .space 128
.global e4m3_scale_out
.balign 64
e4m3_scale_out:
    .space 128
.global e5m2_scale_alpha
.balign 64
e5m2_scale_alpha:
    .word 0x00000000
.global e5m2_scale_x
.balign 64
e5m2_scale_x:
    .space 128
.global e5m2_scale_out
.balign 64
e5m2_scale_out:
    .space 128
.global fp16_scale_alpha
.balign 64
fp16_scale_alpha:
    .word 0x00000000
.global fp16_scale_x
.balign 64
fp16_scale_x:
    .space 256
.global fp16_scale_out
.balign 64
fp16_scale_out:
    .space 256
.global bf16_scale_alpha
.balign 64
bf16_scale_alpha:
    .word 0x00000000
.global bf16_scale_x
.balign 64
bf16_scale_x:
    .space 256
.global bf16_scale_out
.balign 64
bf16_scale_out:
    .space 256
