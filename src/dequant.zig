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
    if (!types.Dtype.RECIPE.get(output_t).encodable) @panic("Args: dtype not encodable");
    switch (scheme) { // compiler hint to avoid branchs cascade
        inline else => |c_scheme| switch (output_t) { // c: comptime
            inline .f16 => |c_output_t| block(types.Scheme.RECIPE.get(c_scheme), c_output_t, in, out),
            inline .bf16=> |c_output_t| block(types.Scheme.RECIPE.get(c_scheme), c_output_t, in, out),
            inline .f32 => |c_output_t| block(types.Scheme.RECIPE.get(c_scheme), c_output_t, in, out),
            else => unreachable,
        } //TODO: do smth
    }
}


fn block(comptime scheme: types.Scheme.Recipe, comptime output_t: types.Dtype, in: Input, out: []u8) void { // dequant module's main
    // Pipeline: in_t -> decode() -> loadScale() -> (f32)translate -> encode() -> out_t
    const weight_format = comptime types.Dtype.RECIPE.get(scheme.weight);
    const Values = @Vector(scheme.block, f32);

    // BLOCKs
    const block_size = scheme.block * weight_format.bits / 8; //bytes
    const out_block_size = scheme.block * @sizeOf(types.Dtype.Storage(output_t));
    const block_count = in.weights.len / block_size;

    for (0..block_count) |idx| {
        // DECODE (int_t -> f32)
        const elements= kernels.unpack(weight_format, scheme.block, in.weights[idx * block_size..][0..block_size]);
        const values = kernels.decode(weight_format, scheme.block, elements);

        // TRANSLATE (f32)
        const scales = loadScale(scheme.scale, in.scales, idx) * in.global_scale; //TODO: 
        const weights: Values = switch (scheme.translation) {
            .affine => blk: { // s(z-q) -> sq + -zs; FMA rewrite
                const midpoint: f32 = @floatFromInt(in.midpoint[idx]);
                break :blk @mulAdd(Values, values, @splat(scales), @splat(-midpoint * scales));
            }, 
            .RTN => values * @as(Values, @splat(scales)),
        };

        // ENCODE (f32 -> out_t)
        const encoded = kernels.encode(output_t, scheme.block, weights);
        out[idx * out_block_size..][0..out_block_size].* = std.mem.toBytes(encoded);
    }
}



fn loadScale(comptime input_t: types.Dtype, scales: []const u8, block_idx: usize) f32 {
    // this precompute the scales at comptime,
    // scales are given as E4M3 for NVFP4
    // so during runtime we can fetch scale in .safetensor and lookup its translation as f32
    //
    // lookup is more efficient than it looks, this is a "dequant" so types are already small -> not much permutations
    //
    const format = comptime types.Dtype.RECIPE.get(input_t);
    switch (format.kind) {
        .float, .exponent => return (comptime kernels.tableTranslate(format))[scales[block_idx]],
        .native => { //skip lookup, `decode()` unnecessary
            const bin = std.meta.Int(.unsigned, format.bits);
            const size = format.bits / 8; // bytes
            const raw = std.mem.readInt(bin, scales[block_idx * size..][0..size], .little);
            return switch (input_t) {
                .f32 => @bitCast(raw),
                .f16 => @floatCast(@as(f16, @bitCast(raw))),
                .bf16 => @bitCast(@as(u32, raw) << 16),
                else => @compileError("loadScale: native input_t"),
            };
        },
        else => @compileError("loadScale: input_t scale"),
    }
}
