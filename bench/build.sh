#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p out

zig cc -O3 -march=native -fno-lto -DNDEBUG -c llama_nvfp4.c -o out/llama_nvfp4.o
zig cc -O3 -march=native -fno-lto -DNDEBUG -c llama_mxfp4.c -o out/llama_mxfp4.o

zig build-exe -femit-bin=out/bench_llama -OReleaseFast -mcpu=native -fno-lto -lc out/llama_nvfp4.o out/llama_mxfp4.o -Mroot=bench_llama.zig

zig build-exe -femit-bin=out/bench_dequant -OReleaseFast -mcpu=native -fno-lto --dep dequant -Mroot=bench_dequant.zig -OReleaseFast -mcpu=native -Mdequant=../src/dequant.zig
