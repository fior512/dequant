const std = @import("std");
const harness = @import("harness.zig");
const generic = @import("generic.zig");

const ALIGN: std.mem.Alignment = .fromByteUnits(generic.CACHE_LINE_SIZE);

// llama.cpp block_nvfp4: 64 weights, 36 B
//   d[s]          e4m3 of w[16s..16s+15]
//   qs[8s + idx]  [w(16s + 8 + idx) << 4 | w(16s + idx)]
const BlockNvfp4 = extern struct { d: [4]u8, qs: [32]u8 };
extern fn dequantize_row_nvfp4(blocks: [*]const BlockNvfp4, out: [*]f32, weight_count: i64) void;

const LlamaKernel = struct {
    blocks: []const BlockNvfp4,

    pub fn Prepare(allocator: std.mem.Allocator, logical: *const harness.Logical) !LlamaKernel {
        const blocks = try allocator.alignedAlloc(BlockNvfp4, ALIGN, logical.weight_count / 64);
        for (blocks, 0..) |*block, block_index| {
            block.d = logical.scales[block_index * 4 ..][0..4].*;
            for (0..4) |sub_index| for (0..8) |pair_index| {
                const sub_nibbles = logical.nibbles[block_index * 64 + sub_index * 16 ..][0..16];
                block.qs[sub_index * 8 + pair_index] = sub_nibbles[pair_index] | (sub_nibbles[8 + pair_index] << 4);
            };
        }
        return .{ .blocks = blocks };
    }

    pub noinline fn Run(kernel: *const LlamaKernel, out: []f32) void {
        dequantize_row_nvfp4(kernel.blocks.ptr, out.ptr, @intCast(out.len));
    }
};

pub fn main() !void {
    try harness.Main(LlamaKernel, "llama.cpp");
}
