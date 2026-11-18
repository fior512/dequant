#!/bin/sh
set -eu
cd "$(dirname "$0")/.."

shards="models/gpt-oss-20b/model-*.safetensors"
time_file="bench/out/gptoss_time_$$"

measure_total_ms() {
  type=$1
  threads=$2
  total_ms=0
  for f in $shards; do
    min_s=999999
    i=0
    while [ "$i" -lt 3 ]; do
      /usr/bin/time -f '%e' -o "$time_file" "zig-out/bin/dequant" "$f" "$type" "$threads" >/dev/null 2>&1
      t=$(cat "$time_file")
      min_s=$(awk -v a="$min_s" -v b="$t" 'BEGIN{print (b<a)?b:a}')
      i=$((i+1))
    done
    total_ms=$(awk -v a="$total_ms" -v b="$min_s" 'BEGIN{printf "%.0f", a+b*1000}')
  done
  echo "$total_ms"
}

# build native once
zig build -Doptimize=ReleaseFast -Dcpu=native

# model stats
input_bytes=0
blocks_bytes=0
for f in $shards; do
  s=$(stat -c %s "$f")
  input_bytes=$((input_bytes + s))
  hs=$(od -An -N8 -t u8 "$f" | tr -d ' ')
  b=$(tail -c +9 "$f" | head -c "$hs" | jq '[to_entries[] | select(.key | endswith("_blocks")) | (.value.data_offsets[1] - .value.data_offsets[0])] | add')
  blocks_bytes=$((blocks_bytes + b))
done
weights_count=$((blocks_bytes * 2))
f32_out_bytes=$((blocks_bytes * 2 * 4))

echo "# gpt-oss-20b, min of 3 per shard, summed over shards"
echo "| out_t | 1t | 6t | 12t | GB/s in (6t) | GB/s out (6t) | qw/s (6t) |"
echo "|---|---|---|---|---|---|---|"
for type in f32 f16 bf16; do
  case "$type" in
    f32) out_elem=4 ;;
    f16) out_elem=2 ;;
    bf16) out_elem=2 ;;
  esac
  t1=$(measure_total_ms "$type" 1)
  t6=$(measure_total_ms "$type" 6)
  t12=$(measure_total_ms "$type" 12)
  out_bytes=$((blocks_bytes * 2 * out_elem))
  gb_in=$(awk -v b="$input_bytes" -v t="$t6" 'BEGIN{printf "%.1f", b*1e-6/t}')
  gb_out=$(awk -v b="$out_bytes" -v t="$t6" 'BEGIN{printf "%.1f", b*1e-6/t}')
  qw=$(awk -v w="$weights_count" -v t="$t6" 'BEGIN{printf "%.1f", w*1e-6/t}')
  printf '| %s | %s ms | %s ms | %s ms | %s | %s | %s G |\n' "$type" "$t1" "$t6" "$t12" "$gb_in" "$gb_out" "$qw"
done

echo ""
echo "# xychart: gpt-oss-20b f32, throughput vs threads"
echo "threads: 1 2 3 4 6 8 12"
in_line=""
out_line=""
for t in 1 2 3 4 6 8 12; do
  ms=$(measure_total_ms f32 "$t")
  gb_in=$(awk -v b="$input_bytes" -v t="$ms" 'BEGIN{printf "%.1f", b*1e-6/t}')
  gb_out=$(awk -v b="$f32_out_bytes" -v t="$ms" 'BEGIN{printf "%.1f", b*1e-6/t}')
  in_line="$in_line$gb_in "
  out_line="$out_line$gb_out "
done
echo "in  [$in_line]"
echo "out [$out_line]"

rm -f "$time_file"
