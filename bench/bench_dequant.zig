const std = @import("std");
const dequant = @import("dequant");
const harness = @import("harness.zig");
const generic = @import("generic.zig");

const ALIGN: std.mem.Alignment = .fromByteUnits(generic.CACHE_LINE_SIZE);

// layout: weights[k] = [w(2k+1) << 4 | w(2k)], scales[s] = e4m3|e8m0 of scale block s
fn DequantKernel(comptime scheme: generic.Scheme) type {
    return struct {
        pub const SCHEME: generic.Scheme = scheme;

        input: dequant.Input,

        pub fn Prepare(allocator: std.mem.Allocator, logical: *const harness.Logical) !@This() {
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

        pub noinline fn Run(kernel: *const @This(), out: []f32) void {
            const dequant_scheme = switch (scheme) {
                .nvfp4 => .nvfp4,
                .mxfp4 => .mxfp4,
            };
            dequant.dequant(dequant_scheme, .f32, kernel.input, std.mem.sliceAsBytes(out));
        }
    };
}

pub fn main(init: std.process.Init) !void {
    const args = harness.ParseArgs(init.minimal.args.iterate());
    switch (args.scheme) {
        inline else => |scheme| try harness.Main(DequantKernel(scheme), "dequant  ", args.size_filter),
    }
}
