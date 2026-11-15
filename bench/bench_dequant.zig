const std = @import("std");
const dequant = @import("dequant");
const harness = @import("harness.zig");
const generic = @import("generic.zig");

const ALIGN: std.mem.Alignment = .fromByteUnits(generic.CACHE_LINE_SIZE);

// layout: weights[k] = [w(2k+1) << 4 | w(2k)], scales[s] = e4m3 of w[16s..16s+15]
const DequantKernel = struct {
    input: dequant.Input,

    pub fn Prepare(allocator: std.mem.Allocator, logical: *const harness.Logical) !DequantKernel {
        const weights = try allocator.alignedAlloc(u8, ALIGN, logical.weight_count / 2);
        for (weights, 0..) |*byte, byte_index| {
            byte.* = logical.nibbles[2 * byte_index] | (logical.nibbles[2 * byte_index + 1] << 4);
        }
        const scales = try allocator.alignedAlloc(u8, ALIGN, logical.scales.len);
        @memcpy(scales, logical.scales);
        return .{
            .input = .{
                .weights = weights,
                .scales = scales,
                .midpoint = &.{},
                .global_scale = 1.0,
            },
        };
    }

    pub noinline fn Run(kernel: *const DequantKernel, out: []f32) void {
        dequant.dequant(.nvfp4, .f32, kernel.input, std.mem.sliceAsBytes(out));
    }
};

pub fn main() !void {
    try harness.Main(DequantKernel, "dequant  ");
}
