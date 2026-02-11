#include <stdio.h>
#include <riscv-pk/encoding.h>
#include <riscv_vector.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "bme.h"
#include "util.h"
#include "dataset.h"
#include "rvv_mx.h"

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
        // asm volatile("vsetvli x0, %[avl], e8, m1, ta, ma" : : [avl]"r"(M-i));
        VSETVLI_ALTFMT_X0(M-i, SEW_E8, LMUL_M1, 1);
        asm volatile("vle8.v v0, (%0)" : : "r"(&A[M*k+i]));
        OPFMACC(m1, v1, v0);
      }

      // move row of c-tile to v-reg, accmulate wth c-row from memory, store back out
      asm volatile("vsetvli x0, %[avl], e32, m4, ta, ma" : : [avl]"r"(cols));
      for (size_t r = 0; r < rows; r++) {
        VMV_VR(v0, r, m1);
        asm volatile("vse32.v v0, (%0)" : : "r"(&C[(i+r)*N+j]));
      }
      j += cols;
    }
    i += rows;
  }
}

int main(void) {
  float C[M_DIM*N_DIM];
  //Test outer product
  mm_opu(a_matrix, b_matrix, C, M_DIM, N_DIM, K_DIM);
  int err = verifyFloat(M_DIM*N_DIM, C, verify_data);
  if (err != 0) {
    printf("error: %d\n", err);
    return 1;
  }

  printf("done\n");
  return 0;
  }

