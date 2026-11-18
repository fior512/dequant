#!/bin/sh
set -eu
cd "$(dirname "$0")/.."

targets="x86_64 x86_64_v2 x86_64_v3 x86_64_v4 native"
types="f32 f16 bf16"
shards="models/gpt-oss-20b/model-*.safetensors"
out_dir="bench/out"
mkdir -p "$out_dir"
time_file="$out_dir/target_time_$$"

isa_name() {
  case "$1" in
    x86_64)     echo "SSE2" ;;
    x86_64_v2)  echo "SSE4.2" ;;
    x86_64_v3)  echo "AVX2+F16C" ;;
    x86_64_v4)  echo "AVX-512" ;;
    native)     echo "znver4+GFNI" ;;
  esac
}

target_name() {
  case "$1" in
    x86_64) echo "x86_64_v1" ;;
    *)      echo "$1" ;;
  esac
}

measure_gpt_ms() {
  target=$1
  type=$2
  min_s=999999
  i=0
  while [ "$i" -lt 3 ]; do
    /usr/bin/time -f '%e %M' -o "$time_file" taskset -c 2 "out/$target/bin/dequant" $shards "$type" 1 >/dev/null 2>&1
    t=$(awk '{print $1}' "$time_file")
    min_s=$(awk -v a="$min_s" -v b="$t" 'BEGIN{print (b<a)?b:a}')
    i=$((i+1))
  done
  awk -v s="$min_s" 'BEGIN{printf "%.0f", s*1000}'
}

echo "| target | ISA | gpt f32 | gpt f16 | gpt bf16 | kernel nvfp4 ns/w |"
echo "|---|---|---|---|---|---|"

for c in $targets; do
  echo "== $c: build ==" >&2
  zig build -Doptimize=ReleaseFast -Dcpu="$c" --prefix "out/$c"

  echo "== $c: gpt ==" >&2
  f32=$(measure_gpt_ms "$c" f32)
  f16=$(measure_gpt_ms "$c" f16)
  bf16=$(measure_gpt_ms "$c" bf16)

  echo "== $c: kernel ns/w ==" >&2
  zig build-exe -femit-bin="$out_dir/bench_dequant_$c" -OReleaseFast -mcpu="$c" -fno-lto \
    --dep dequant -Mroot=bench/bench_dequant.zig -Mdequant=src/dequant.zig
  "$out_dir/bench_dequant_$c" nvfp4 L1 > "$out_dir/bench_nvfp4_$c.txt" 2>&1
  ns_w=$(awk '/L1.*summary/{print $7}' "$out_dir/bench_nvfp4_$c.txt")

  printf '| %-11s | %-9s | %6s ms | %6s ms | %6s ms | %s |\n' \
    "$(target_name "$c")" "$(isa_name "$c")" "$f32" "$f16" "$bf16" "$ns_w"
done

rm -f "$time_file"
