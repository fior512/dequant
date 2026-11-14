NVFP4 dequantizer in zig

## Project

### Rules
- .safetensor only
- dequant on the fly:? chunk by chunk
- SIMD: chunk
- under the shape `std.Io.reader`: no mmap
- multithreaded: threadpool over chunks

### Pipeline:

We parse the json (`safetensors.parseJson()`) in single thread before the threadpool (high ratio binary_bytes/json_bytes) by using the **std** lib with std.json.parseFromSliceLeaky to parse straight inside struct `safetensors.Layer{}`, then spawn the threadpool around an atomic incremental counter that defines the layers of the json already dequantized. Finally comes the dequantization part in 5 steps: `unpack()` -> `decode()` -> `loadScale()` -> *translate* -> `encode()`:

1) `unpack()`: split high and low nibbles
2) `decode()`: extract sign, exp, and mantissa to translate whatever is accepted into f32
3) `loadScale()`: is a lookup table for fast translation of scales
4) *translate*: shift the f32 intermediate state via std-affine or Round-to-nearest(RTN)
5) `encode()`: convert the f32 state to desired output_t

For the sake of the benchmark, since there are no real endpoints, everything is discarded through `std.mem.doNotOptimizeAway()` (`main.worker()`)

#### Details:

- the entire dequant module is modular via `types.Dtype` and `types.Scheme`, which both contain *"recipes"*, then only the `switch() {}`s need to be aligned if new types comes in. 
- No c-interop for simdjson since `parseJson()` is nowhere near being a bottleneck (see below). 

AI has only been only used to search for syntax (primarily for json parsing), parsing as array the performance metrics below and generating test cases to verify accuracy.
Where usually i use LLMs to confront my design hypothesis, i chose not to for this repo.

## Bench

Setup: AMD Ryzen 5 7600X 6-Core Processor\
Machine states: epp: perf, boost and SMT on, unpinned freq\
Protocol: 3 runs per permutation, taskset, 1/6/12 threads, 5 models, input nvfp4 only, f32/f16/bf16 as outputs\
tools: perf stat/report, llvm-mca and MAQAO CQA

**Models**:

| Label | Hugging Face repo, file | Size | Quantized layers | Quantized bytes | binary:json |
|---|---|---|---|---|---|
| whisper | `Reza2kn/Mojtabazarrin_whisper-base-fa-NVFP4`, `model.safetensors` | 0.13 GiB | 96 | 16% | 2,621:1 |
| glm-ocr | `AlphaOxO/GLM-OCR-NVFP4`, `model.safetensors` | 1.27 GiB | 120 | 22% | 14,886:1 |
| wan22 | `changwangss/Wan2.2-T2V-A14B-Diffusers-NVFP4-RTN`, `transformer/diffusion_pytorch_model-00001-of-00002.safetensors` | 4.65 GiB | 228 | 80% | 36,361:1 |
| qwen-image | `darrellbest/Qwen-Image-2.1-PE-I2I-Heretic-NVFP4`, `model.safetensors` | 10.43 GiB | 128 | 24% | 78,456:1 |
| qwen36 | `RedHatAI/Qwen3.6-35B-A3B-NVFP4`, `model.safetensors` | 20.92 GiB | 30,880 | 73% | 1,284:1 |


### Global

**Wall clock**:

| Model | Threads | Wall | JSON | Dequant | File GiB/s | Weights/s | Idle |
|---|---|---|---|---|---|---|---|
| whisper | 1 | 16.7 ms | 0.28 ms | 16.0 ms | 8.1 | 2.8 G | 1% |
| | 6 | 12.6 ms | 0.29 ms | 11.4 ms | 11.3 | 3.9 G | 410% |
| | 12 | 13.1 ms | 0.29 ms | 12.1 ms | 10.7 | 3.6 G | 940% |
| glm-ocr | 1 | 177 ms | 0.39 ms | 176 ms | 7.2 | 3.4 G | 0% |
| | 6 | 42.0 ms | 0.47 ms | 40.8 ms | 31.2 | 14.5 G | 55% |
| | 12 | 45.2 ms | 0.57 ms | 43.7 ms | 29.1 | 13.5 G | 328% |
| wan22 | 1 | 1236 ms | 0.62 ms | 1235 ms | 3.8 | 6.4 G | 0% |
| | 6 | 233 ms | 0.78 ms | 232 ms | 20.1 | 34.2 G | 39% |
| | 12 | 209 ms | 0.76 ms | 208 ms | 22.4 | 38.2 G | 106% |
| qwen-image | 1 | 1201 ms | 0.54 ms | 1200 ms | 8.7 | 4.4 G | 0% |
| | 6 | 301 ms | 0.66 ms | 300 ms | 34.8 | 17.7 G | 77% |
| | 12 | 298 ms | 0.53 ms | 296 ms | 35.3 | 17.9 G | 304% |
| qwen36 | 1 | 5159 ms | 51.7 ms | 5105 ms | 4.1 | 6.4 G | 0% |
| | 6 | 1009 ms | 51.4 ms | 956 ms | 21.9 | 34.1 G | 41% |
| | 12 | 910 ms | 51.8 ms | 857 ms | 24.4 | 38.0 G | 127% |

> GiB/s: (bytes read) / process time (`time` adds 1ms)
> Idle: (Threads*100) - CPU usage (rusage)

**stats**:

| Model | Threads | Cycles | Instructions | CPI | User CPI | Retiring | Frontend | Bad spec | Backend |
|---|---|---|---|---|---|---|---|---|---|
| whisper | 1 | 8.51e7 | 8.11e7 | 1.05 | 0.44 | 15% | 25% | 0.6% | 51%|
| | 6 | 1.07e8 | 9.19e7 | 1.16 | 0.45 | 14% | 31% | 0.8% | 49% |
| | 12 | 1.54e8 | 1.03e8 | 1.50 | 0.59 | 8% | 35% | 0.7% | 40% |
| glm-ocr | 1 | 8.95e8 | 9.87e8 | 0.91 | 0.45 | 21% | 28% | 0.3% | 51%|
| | 6 | 1.23e9 | 9.95e8 | 1.24 | 0.45 | 13% | 29% | 0.2% | 52% |
| | 12 | 1.95e9 | 1.01e9 | 1.94 | 0.74 | 9% | 27% | 0.2% | 52% |
| wan22 | 1 | 6.62e9 | 1.23e10 | 0.54 | 0.44 | 33% | 12% | 0.2% | 55% |
| | 6 | 6.83e9 | 1.23e10 | 0.56 | 0.44 | 31% | 13% | 0.2% | 55% |
| | 12 | 1.19e10 | 1.23e10 | 0.97 | 0.79 | 18% | 10% | 0.1% | 50% |
| qwen-image | 1 | 6.45e9 | 8.71e9 | 0.74 | 0.45 | 26% | 26% | 0.3% | 48% |
| | 6 | 8.33e9 | 8.72e9 | 0.95 | 0.44 | 20% | 32% | 0.2% | 48% |
| | 12 | 1.38e10 | 8.74e9 | 1.58 | 0.78 | 12% | 25% | 0.1% | 49% |
| qwen36 | 1 | 2.76e10 | 5.14e10 | 0.54 | 0.44 | 33% | 12% | 0.3% | 54% |
| | 6 | 2.93e10 | 5.15e10 | 0.57 | 0.44 | 31% | 15% | 0.2% | 54% |
| | 12 | 5.18e10 | 5.15e10 | 1.01 | 0.79 | 17% | 11% | 0.1% | 51% |


### Dequantization

| out_t | unpack | scale load | decode | translate | encode |
|---|---|---|---|---|---|
| f32 | 16% | 13% | 52% | 5% | 9% |
| f16 | 12% | 14% | 43% | 3% | 26% |
| bf16 | 11% | 10% | 43% | 4% | 26% |

> translate is max 5%, fused kernels w/ emulation isn't worth

**Lines**:

| Code | Phase | f32 | f16 | bf16 |
|---|---|---|---|---|
| `packed_w: Half = bytes.*` | unpack | 3.5% | 2.1% | 1.6% |
| `lo = packed_w & 0x0F` | unpack | 2.8% | 1.8% | 1.8% |
| `hi = packed_w >> 4` | unpack | 2.9% | 1.9% | 2.1% |
| `@shuffle(u8, lo, hi, interleaved)` | unpack | 3.1% | 2.5% | 2.5% |
| `wide: Lanes = elements` | unpack | 3.4% | 3.3% | 3.1% |
| `sign = (wide >> 3) << 31` | decode | 3.9% | 3.2% | 3.1% |
| `exp = (wide >> mant_bits) & mask` | decode | 7.6% | 6.6% | 6.7% |
| `mant = wide & mask` | decode | 4.7% | 3.3% | 3.2% |
| `implicit = @select(exp != zero, ..)` | decode | 4.4% | 3.6% | 3.6% |
| `significand = @floatFromInt(implicit \| mant)` | decode | 9.1% | 7.6% | 7.8% |
| `@max(exp, 1)` | decode | 4.5% | 4.0% | 3.7% |
| `(.. + exp_off) << 23` | decode | 8.6% | 8.0% | 7.4% |
| `magnitude = significand * power` | decode | 4.5% | 3.8% | 4.0% |
| `sign \| magnitude` | decode | 4.9% | 3.0% | 3.6% |
| `lookup[scales[block_idx]]` | scale load | 3.3% | 1.9% | 1.7% |
| `.. * in.global_scale` | scale load | 4.8% | 5.0% | 3.7% |
| `@splat(scales)` | scale load | 5.1% | 6.8% | 4.1% |
| `values * scales` | translate | 4.5% | 3.1% | 4.0% |
| `little_end = (bits >> 16) & 1` | encode | | | 8.3% |
| `rounded = bits + 0x7FFF + little_end` | encode | | | 8.9% |
| `@truncate(rounded >> 16)` | encode | | | 3.4% |
| `out[..].* = @bitCast(encoded)` | encode | 9.1% | 25.6% | 5.8% |


**Cycles per block of 16 weights**:

| Source | f32 | f16 | bf16 | Stated bottleneck |
|---|---|---|---|---|
| `llvm-mca`, znver4 | 8.1 | 8.6 | 11.5 | vector ports, FP1 at 87 to 91% |
| MAQAO CQA, ZEN_V4 | 6.25 | 7.0 | 7.75 | FP multiply unit |
| Measured, user cycles | 10.7 | 12.3 | 15.7 | |

| out_t | Instructions per iteration | Blocks per iteration | llvm-mca IPC | vec length efficiency (MAQAO) |
|---|---|---|---|---|
| f32 | 48 | 2 | 2.96 | 72% |
| f16 | 48 | 2 | 2.78 | 72% |
| bf16 | 31 | 1 | 2.69 | 77% |

> all 3 out_t use 512-bits


**Kernel against llama.cpp**:

output: f32
threads: 1
code: kernels only (no file read nor orchestration).

| Values | Fits in | llama.cpp min cycles / 32 values | dequant | Speedup | llama.cpp f32 GB/s | dequant |
|---|---|---|---|---|---|---|
| 4,096 | L1 | 66.09 | 17.63 | 3.7x | 8.22 | 30.3 |
| 65,536 | L2 | 67.65 | 18.08 | 3.7x | 8.12 | 30.3 |
| 655,360 | L3 | 68.54 | 18.40 | 3.7x | 8.09 | 29.7 |
| 65,536,000 | DRAM | 69.07 | 21.49 | 3.2x | 8.09 | 25.0 |

> their kernels comes from llama.cpp `tests/test-quantize-perf.cpp` (HEAD master 2026-11-13 20:27 UTC *ded522f8304554317e7ff8f88698dbbbca5425aa*)

---

**TODO**: 
- NVFP4 alternative layouts
- `_copy_to_iter`(memcpy) inside `std.Io.Reader.interface.take`
- mostly backend-bound
- decode is 12 instr/block
- `unpack()` use mm128bold text
