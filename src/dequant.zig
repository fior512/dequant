const std = @import("std");
const types = @import("types.zig");
const kernels = @import("kernels.zig");

pub const Input = struct {
    weights: []const u8,
    scales: []const u8,
    midpoint: []const u8, // .affine only
    global_scale: f32,
};

pub fn dequant(scheme: types.Scheme, output_t: types.Dtype, in: Input, out: []u8) void {
    const translation: types.Scheme.Translation = switch (in.midpoint.len) {
        0 => .RTN,
        else => .affine,
    };
    switch (scheme) {
        inline else => |c_scheme| switch (output_t) {
            inline else => |c_output_t| switch (translation) {
                inline else => |c_translation| block(
                    types.Scheme.RECIPE.get(c_scheme),
                    types.Dtype.RECIPE.get(c_output_t),
                    c_translation,
                    in,
                    out,
                ),
            },
        },
    }
}

fn block(
    comptime scheme: types.Scheme.Recipe,
    comptime out_t: types.Dtype.Recipe,
    comptime translation: types.Scheme.Translation,
    in: Input,
    out: []u8,
) void { // dequant module's main
    // Pipeline: input_t -> unpack() -> decode() -> (f32)loadScale()+translate -> encode() -> pack() -> out_t
    const weight_t = comptime types.Dtype.RECIPE.get(scheme.weight);
    const Values = @Vector(scheme.block, f32);

    // BLOCKs
    const block_size = scheme.block * weight_t.bits / 8; //bytes
    const out_block_size = scheme.block * out_t.bits / 8; //bytes
    const block_count = in.weights.len / block_size;

    for (0..block_count) |idx| {
        // DECODE (int_t -> f32)
        const unpacked = kernels.unpack(
            weight_t,
            scheme.block,
            in.weights[idx * block_size ..][0..block_size],
        );
        const values = kernels.decode(weight_t, scheme.block, unpacked);

        // TRANSLATE (f32)
        const scales = if (comptime scheme.scale) |c_scale_t| loadScale(
            c_scale_t,
            in.scales,
            idx,
        ) * in.global_scale else in.global_scale;
        const weights: Values = switch (translation) {
            .affine => blk: { // s(q-z) -> sq + -zs; FMA rewrite
                const midpoint: f32 = @floatFromInt(in.midpoint[idx]);
                break :blk @mulAdd(Values, values, @splat(scales), @splat(-midpoint * scales));
            },
            .RTN => values * @as(Values, @splat(scales)),
        };

        // ENCODE (f32 -> out_t)
        const encoded = kernels.encode(out_t, scheme.block, weights);
        out[idx * out_block_size ..][0..out_block_size].* = kernels.pack(
            out_t,
            scheme.block,
            encoded,
        );
    }
}

fn loadScale(comptime in_t: types.Dtype, scales: []const u8, block_idx: usize) f32 {
    const input_t = comptime types.Dtype.RECIPE.get(in_t);
    const size = input_t.bits / 8;
    const values = kernels.unpack(input_t, 1, scales[block_idx * size ..][0..size]);
    return kernels.decode(input_t, 1, values)[0];
}
