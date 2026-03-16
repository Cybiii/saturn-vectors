.section .data,"aw",@progbits
.global N
.balign 64
N:
    .word 0x00000080
    .word 0x00000000
.global fp16_masked_mul_mask
.balign 64
fp16_masked_mul_mask:
    .space 128
.global fp16_masked_mul_a
.balign 64
fp16_masked_mul_a:
    .space 256
.global fp16_masked_mul_b
.balign 64
fp16_masked_mul_b:
    .space 256
.global fp16_masked_mul_out
.balign 64
fp16_masked_mul_out:
    .space 256
.global bf16_masked_mul_mask
.balign 64
bf16_masked_mul_mask:
    .space 128
.global bf16_masked_mul_a
.balign 64
bf16_masked_mul_a:
    .space 256
.global bf16_masked_mul_b
.balign 64
bf16_masked_mul_b:
    .space 256
.global bf16_masked_mul_out
.balign 64
bf16_masked_mul_out:
    .space 256
.global fp16_masked_add_mask
.balign 64
fp16_masked_add_mask:
    .space 128
.global fp16_masked_add_a
.balign 64
fp16_masked_add_a:
    .space 256
.global fp16_masked_add_b
.balign 64
fp16_masked_add_b:
    .space 256
.global fp16_masked_add_out
.balign 64
fp16_masked_add_out:
    .space 256
.global bf16_masked_add_mask
.balign 64
bf16_masked_add_mask:
    .space 128
.global bf16_masked_add_a
.balign 64
bf16_masked_add_a:
    .space 256
.global bf16_masked_add_b
.balign 64
bf16_masked_add_b:
    .space 256
.global bf16_masked_add_out
.balign 64
bf16_masked_add_out:
    .space 256
