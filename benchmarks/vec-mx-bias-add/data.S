.section .data,"aw",@progbits
.global N
.balign 64
N:
    .word 0x00000080
    .word 0x00000000
.global e4m3_bias_b
.balign 64
e4m3_bias_b:
    .word 0x00000000
.global e4m3_bias_x
.balign 64
e4m3_bias_x:
    .space 128
.global e4m3_bias_out
.balign 64
e4m3_bias_out:
    .space 128
.global e5m2_bias_b
.balign 64
e5m2_bias_b:
    .word 0x00000000
.global e5m2_bias_x
.balign 64
e5m2_bias_x:
    .space 128
.global e5m2_bias_out
.balign 64
e5m2_bias_out:
    .space 128
.global fp16_bias_b
.balign 64
fp16_bias_b:
    .word 0x00000000
.global fp16_bias_x
.balign 64
fp16_bias_x:
    .space 256
.global fp16_bias_out
.balign 64
fp16_bias_out:
    .space 256
.global bf16_bias_b
.balign 64
bf16_bias_b:
    .word 0x00000000
.global bf16_bias_x
.balign 64
bf16_bias_x:
    .space 256
.global bf16_bias_out
.balign 64
bf16_bias_out:
    .space 256
