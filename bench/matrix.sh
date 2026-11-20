#!/bin/sh
set -eu
cd "$(dirname "$0")/.."

size="${1:-L1}"
bench="${BENCH:-bench/out/bench_dequant}" # built by bench/build.sh
schemes="nvfp4 mxfp4 i4 f8e4m3 f16 bf16 f32"
types="f4e2m1 i4 f8e4m3 f8e8m0 f16 bf16 f32"
out_dir="bench/out"
mkdir -p "$out_dir"

echo "# $size median ns/w, in scheme (rows) -> out_t (cols)"
printf '| in \\ out |'
for t in $types; do printf ' %s |' "$t"; done
echo
printf '|---|'
for t in $types; do printf -- '---|'; done
echo

for s in $schemes; do
  printf '| %s |' "$s"
  for t in $types; do
    echo "== $s -> $t ==" >&2
    "$bench" "$s" "$t" "$size" > "$out_dir/matrix_${s}_${t}.txt" 2>&1
    ns_w=$(awk '/summary/{print $7}' "$out_dir/matrix_${s}_${t}.txt")
    printf ' %s |' "$ns_w"
  done
  echo
done
