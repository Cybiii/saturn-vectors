#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "rvv_mx.h"
#include "bme.h"
#include <math.h>

extern size_t M;
extern size_t N;
extern size_t K;
size_t avl;
size_t vl;

#define TEST_DATA(type, name, otype) \
    extern type name ## _a[] __attribute__((aligned(64))); \
    extern type name ## _b[] __attribute__((aligned(64))); \
    extern otype name ## _c[] __attribute__((aligned(64))); \
    extern type name ## _at[] __attribute__((aligned(64))); \
    extern otype name ## _out[] __attribute__((aligned(64)));;

void verify_result(float* out, float* ref, size_t M, size_t N) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float res = out[i*N+j];
            float gold = ref[i*N+j];
            if (fabsf(res - gold) > 1e-6) {
                printf("err r - gold = %d at (%d, %d)\n", (int32_t)(fabsf(res - gold)), i, j);
                printf("res = %d, gold = %d\n", (int32_t)res, (int32_t)gold);
            }
        }
    }
    printf("All tests passed\n");
}

void mm_opu(int8_t* A, int8_t* B, float* C, size_t M, size_t N, size_t K) {
    size_t maxvl;
    size_t vl;
    asm volatile("vsetvli %[vl], zero, e32, m4, ta, ma" : [vl]"=r"(maxvl));
    
    size_t i = 0;
    while (i < M) {
        size_t rows;
        asm volatile("vsetvli %[vl], %[avl], e8, m1, ta, ma" : [vl]"=r"(rows) : [avl]"r"(M-i));
    
        size_t j = 0;
        while (j < N) {
        // Clear the m1 tile
        asm volatile("vsetvli %[vl], x0, e32, m4, ta, ma" : [vl]"=r"(vl));
        asm volatile("vmv.v.i v0, 0x0");
        OPMVINBCAST(m1, v0);
    
        // Set rows/cols to remaining rows/cols using vsetvli
        size_t cols;
        asm volatile("vsetvli %[vl], %[avl], e8, m1, ta, ma" : [vl]"=r"(cols) : [avl]"r"(N-j));
    
        // do the k-loop
        for (size_t k = 0; k < K; k++) {
            asm volatile("vsetvli x0, %[avl], e8, m1, ta, ma" : : [avl]"r"(N-j));
            asm volatile("vle8.v v1, (%0)" : : "r"(&B[N*k+j]));
            asm volatile("vsetvli x0, %[avl], e8, m1, ta, ma" : : [avl]"r"(M-i));
            asm volatile("vle8.v v0, (%0)" : : "r"(&A[M*k+i]));
            OPFMACC(m1, v1, v0);
        }
    
        // move row of c-tile to v-reg, accmulate wth c-row from memory, store back out
        asm volatile("vsetvli x0, %[avl], e32, m4, ta, ma" : : [avl]"r"(cols));
        for (size_t r = 0; r < rows; r++) {
            VMV_VR(v0, r, m1);
            // printf("i = %d, j = %d, r = %d, [(i+r)*N+j] = %d\n", i, j, r, (i+r)*N+j);
            // asm volatile("vle32.v v4, (%0)" : : "r"(&C[(i+r)*N+j]));
            // asm volatile("vadd.vv v0, v0, v4");
            asm volatile("vse32.v v0, (%0)" : : "r"(&C[(i+r)*N+j]));
        }
        j += cols;
        }
        i += rows;
    }
}

TEST_DATA(uint8_t, e4m3, float)
TEST_DATA(uint8_t, e5m2, float)

int main() {
    float out[M*N];

    printf("running e4m3\n");
    mm_opu(e4m3_at, e4m3_b, out, M, N, K);
    verify_result(out, e4m3_c, M, N);

    printf("running e5m2\n");
    mm_opu(e5m2_at, e5m2_b, out, M, N, K);
    verify_result(out, e5m2_c, M, N);

    return 0;
}