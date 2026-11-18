const std = @import("std");
const harness = @import("harness.zig");
const generic = @import("generic.zig");

const ALIGN: std.mem.Alignment = .fromByteUnits(generic.CACHE_LINE_SIZE);

// llama.cpp block_nvfp4: 64 weights, 36 B
//   d[s]          e4m3 of w[16s..16s+15]
//   qs[8s + idx]  [w(16s + 8 + idx) << 4 | w(16s + idx)]
const BlockNvfp4 = extern struct { d: [4]u8, qs: [32]u8 };
extern fn dequantize_row_nvfp4(blocks: [*]const BlockNvfp4, out: [*]f32, weight_count: i64) void;

// llama.cpp block_mxfp4: 32 weights, 17 B, align 1
//   e        e8m0 of w[0..31]
//   qs[idx]  [w(16 + idx) << 4 | w(idx)]
const BlockMxfp4 = extern struct { e: u8, qs: [16]u8 };
extern fn dequantize_row_mxfp4(blocks: [*]const BlockMxfp4, out: [*]f32, weight_count: i64) void;

const LlamaNvfp4Kernel = struct {
    pub const SCHEME: generic.Scheme = .nvfp4;

    blocks: []const BlockNvfp4,

    pub fn Prepare(allocator: std.mem.Allocator, logical: *const harness.Logical) !LlamaNvfp4Kernel {
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

    pub noinline fn Run(kernel: *const LlamaNvfp4Kernel, out: []f32) void {
        dequantize_row_nvfp4(kernel.blocks.ptr, out.ptr, @intCast(out.len));
    }
};

const LlamaMxfp4Kernel = struct {
    pub const SCHEME: generic.Scheme = .mxfp4;

    blocks: []const BlockMxfp4,

    pub fn Prepare(allocator: std.mem.Allocator, logical: *const harness.Logical) !LlamaMxfp4Kernel {
        const blocks = try allocator.alignedAlloc(BlockMxfp4, ALIGN, logical.weight_count / 32);
        for (blocks, 0..) |*block, block_index| {
            block.e = logical.scales[block_index];
            const block_nibbles = logical.nibbles[block_index * 32 ..][0..32];
            for (0..16) |pair_index| {
                block.qs[pair_index] = block_nibbles[pair_index] | (block_nibbles[16 + pair_index] << 4);
            }
        }
        return .{ .blocks = blocks };
    }

    pub noinline fn Run(kernel: *const LlamaMxfp4Kernel, out: []f32) void {
        dequantize_row_mxfp4(kernel.blocks.ptr, out.ptr, @intCast(out.len));
    }
};

pub fn main(init: std.process.Init) !void {
    const args = harness.ParseArgs(init.minimal.args.iterate());
    switch (args.scheme) {
        .nvfp4 => try harness.Main(LlamaNvfp4Kernel, "llama.cpp", args.size_filter),
        .mxfp4 => try harness.Main(LlamaMxfp4Kernel, "llama.cpp", args.size_filter),
    }
}
