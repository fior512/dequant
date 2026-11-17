// verbatim from llama.cpp master (ggml/src/ggml-common.h, ggml-impl.h, ggml-quants.c)
#include <assert.h>
#include <stdint.h>
#include <string.h>

#define QK_MXFP4 32
typedef struct {
  uint8_t e;  // E8M0
  uint8_t qs[QK_MXFP4 / 2];
} block_mxfp4;

// e2m1 values (doubled), shared by MXFP4 and NVFP4
static const int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

// Equal to ggml_e8m0_to_fp32/2
static inline float ggml_e8m0_to_fp32_half(uint8_t x) {
  uint32_t bits;
  if (x < 2) {
    bits = 0x00200000 << x;  // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
  } else {
    bits = (uint32_t)(x - 1) << 23;  // 2^(x-128)
  }
  float result;
  memcpy(&result, &bits, sizeof(float));
  return result;
}

void dequantize_row_mxfp4(
    const block_mxfp4* restrict x, float* restrict y, int64_t k
) {
  static const int qk = QK_MXFP4;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    const float d = ggml_e8m0_to_fp32_half(x[i].e);

    for (int j = 0; j < qk / 2; ++j) {
      const int8_t x0 = kvalues_mxfp4[x[i].qs[j] & 0x0F];
      const int8_t x1 = kvalues_mxfp4[x[i].qs[j] >> 4];

      y[i * qk + j + 0] = x0 * d;
      y[i * qk + j + qk / 2] = x1 * d;
    }
  }
}
