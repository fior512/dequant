// verbatim from llama.cpp master (ggml/src/ggml-common.h, ggml-impl.h, ggml-quants.c)
#include <assert.h>
#include <math.h>
#include <stdint.h>

#define QK_NVFP4 64
#define QK_NVFP4_SUB 16
typedef struct {
  uint8_t
      d[QK_NVFP4 /
        QK_NVFP4_SUB];  // UE4M3 scales (4 bytes, one per 16-element sub-block)
  uint8_t qs[QK_NVFP4 / 2];  // packed 4-bit E2M1 values (32 bytes)
} block_nvfp4;

// e2m1 values (doubled), shared by MXFP4 and NVFP4
static const int8_t kvalues_mxfp4[16] = {
    0,
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    0,
    -1,
    -2,
    -3,
    -4,
    -6,
    -8,
    -12,
};

static inline float ggml_ue4m3_to_fp32(uint8_t x) {
  if (x == 0 || x == 0x7F) {
    return 0.0f;
  }
  int exp = (x >> 3) & 0xF;
  int man = x & 0x7;
  float raw;
  if (exp == 0) {
    raw = ldexpf((float)man, -9);
  } else {
    raw = ldexpf(1.0f + (float)man / 8.0f, exp - 7);
  }
  return raw * 0.5f;
}

void dequantize_row_nvfp4(
    const block_nvfp4* restrict x, float* restrict y, int64_t k
) {
  static const int qk = QK_NVFP4;
  static const int qk_sub = QK_NVFP4_SUB;
  static const int n_sub = QK_NVFP4 / QK_NVFP4_SUB;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    for (int s = 0; s < n_sub; s++) {
      const float d = ggml_ue4m3_to_fp32(x[i].d[s]);
      float* yb = y + i * qk + s * qk_sub;

      for (int j = 0; j < qk_sub / 2; ++j) {
        const int8_t v0 = kvalues_mxfp4[x[i].qs[s * (qk_sub / 2) + j] & 0x0F];
        const int8_t v1 = kvalues_mxfp4[x[i].qs[s * (qk_sub / 2) + j] >> 4];

        yb[j + 0] = v0 * d;
        yb[j + qk_sub / 2] = v1 * d;
      }
    }
  }
}
