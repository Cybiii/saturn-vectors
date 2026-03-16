.section .data,"aw",@progbits
.global IN_LEN
.balign 64
IN_LEN:
    .word 0x00000040
    .word 0x00000000
.global K_LEN
.balign 64
K_LEN:
    .word 0x00000008
    .word 0x00000000
.global fp16_conv1d_in
.balign 64
fp16_conv1d_in:
    .space 128
.global fp16_conv1d_ker
.balign 64
fp16_conv1d_ker:
    .space 16
.global fp16_conv1d_out
.balign 64
fp16_conv1d_out:
    .space 228
.global bf16_conv1d_in
.balign 64
bf16_conv1d_in:
    .space 128
.global bf16_conv1d_ker
.balign 64
bf16_conv1d_ker:
    .space 16
.global bf16_conv1d_out
.balign 64
bf16_conv1d_out:
    .space 228
