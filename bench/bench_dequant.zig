const std = @import("std");
const dequant = @import("dequant");
const harness = @import("harness.zig");
const generic = @import("generic.zig");

const ALIGN: std.mem.Alignment = .fromByteUnits(generic.CACHE_LINE_SIZE);

fn DequantKernel(comptime scheme: generic.Scheme, comptime output_t: generic.Dtype) type {
    return struct {
        pub const SCHEME: generic.Scheme = scheme;
        pub const OUTPUT: generic.Dtype = output_t;

        input: dequant.Input,

        pub fn Prepare(allocator: std.mem.Allocator, logical: *const harness.Logical) !@This() {
            const weights = try allocator.alignedAlloc(u8, ALIGN, logical.weights.len);
            @memcpy(weights, logical.weights);
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

        pub noinline fn Run(kernel: *const @This(), out: []u8) void {
            dequant.dequant(scheme, output_t, kernel.input, out);
        }
    };
}

pub fn main(init: std.process.Init) !void {
    const args = harness.parseArgs(init.minimal.args.iterate());
    switch (args.scheme) {
        inline else => |c_scheme| switch (args.output_t) {
            inline else => |c_output_t| try harness.run(DequantKernel(c_scheme, c_output_t), "dequant  ", args.size_filter),
        },
    }
}
