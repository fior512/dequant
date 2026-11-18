const std = @import("std");
const types = @import("types.zig");

pub inline fn unpack(comptime format: types.Dtype.Recipe, comptime count: usize, bytes: *const [count * format.bits / 8]u8) @Vector(count, u8) {
    switch (format.bits) {
        4 => {
            const Half = @Vector(count/2, u8);
            const packed_w: Half = bytes.*;

            const lo = packed_w & @as(Half, @splat(0x0F)); // 0x0L, 0x0L, ..
            const hi = packed_w >> @as(Half, @splat(4));   // 0x0H, 0x0H, ..
            const interleaved = comptime blk: {
                var mask: [count]i32 = undefined;
                for(&mask, 0..) |*lane, idx| {
                    const half_idx: i32 = @intCast(idx/2);
                    lane.* = if (idx%2 == 0) half_idx else ~half_idx;   // `~` is used a sign swap (signed int)
                                                                        // lane.* > 0 -> a
                                                                        // lane.* < 0 -> b
                }
                break :blk mask;
            };
            return @shuffle(u8, lo, hi, interleaved); // 0x0L, 0x0H, 0x0L, 0x0H, ..
                                                                     // lo first bcs little endian
        },
        8 => return bytes.*,
        else => @compileError("unpack: weight bits"),
    }
}

// decode for E_M_ types
pub inline fn decode(comptime format: types.Dtype.Recipe, comptime count: usize, elements: @Vector(count, u8)) @Vector(count, f32) {
    const Lanes = @Vector(count, u32);
    const Floats = @Vector(count, f32);
    const wide: Lanes = elements; // 8bits -> 32
    

    const t_size = 1 << format.bits;
    const reg_units = comptime @min(t_size, std.simd.suggestVectorLength(u32) orelse 1);
    if (comptime t_size <= 2 * reg_units) if (!@inComptime()) { // Q:fit in 2 registers? (hi,lo)
        const TABLE: [t_size]f32 = comptime tableTranslate(format);
        
        const idx = wide & @as(Lanes, @splat(reg_units - 1));

        const LO: @Vector(reg_units, f32) = TABLE[0..reg_units].*;
        const HI: @Vector(reg_units, f32) = TABLE[t_size - reg_units..].*;

        var lo: Floats = undefined;
        var hi: Floats = undefined;
        inline for (0..count) |lane_idx| {
            lo[lane_idx] = LO[idx[lane_idx]];
            hi[lane_idx] = HI[idx[lane_idx]];
        }
        return @select(f32, wide >= @as(Lanes, @splat(reg_units)), hi, lo);
    };

    switch (format.kind) {
        .float => {
            // element: [sign | exp | mant]
            // value:   significand * 2^(exp - bias - mant)
            //          significand = 1.mant (1 is not in bits)
            //
            //  f32: S1E8M23

            const zero: Lanes = @splat(0);
            
            // Extract
            const sign = (wide >> @splat(format.bits-1)) << @splat(31); // pre-setup Big endian i32
            const exp = (wide >> @splat(format.mant_bits)) & @as(Lanes, @splat((1 << format.exp_bits) - 1));
            const mant = wide & @as(Lanes, @splat((1 << format.mant_bits) - 1));

            // Translate
            const implicit = @select(u32, exp != zero, @as(Lanes, @splat(1 << format.mant_bits)), zero); // implicit only when `exp` exist
            const significand: Floats = @floatFromInt(implicit | mant);
            const exp_off = 127 - @as(u32, format.bias) - format.mant_bits; // exp offset in f32 
            const power: Floats = @bitCast(
                (@max(
                    exp,
                    @as(Lanes, @splat(1))) + @as(Lanes, @splat(exp_off))
                ) << @splat(23),
            );

            const magnitude: Lanes = @bitCast(significand * power);
            return @bitCast(sign | magnitude);
        },
        .exponent => {
            const zero: Lanes = @splat(0);

            const lo = @select(u32, wide == zero, @as(Lanes, @splat(0x00400000)), wide << @splat(23));
            const bits = @select(u32, wide == @as(Lanes, @splat(0xFF)), @as(Lanes, @splat(0x7FC00000)), lo);
            return @bitCast(bits);
        },
        .native => @compileError("decode: weight dtype"),
        .integer => return @floatFromInt(wide),
    }
}


pub inline fn encode(comptime dtype: types.Dtype, comptime count: usize, values: @Vector(count, f32)) [count]types.Dtype.Storage(dtype) {
    switch (dtype) {
        .f16 => return @as(@Vector(count, f16), @floatCast(values)),
        .f32 => return values,
        .bf16 => {
            // round [u16 high][u16 low] to [u16 high] via "rotary counter" logic
            // f32  (in_t) is E8M23
            // bf16 (out_t) is E8M7 (mantissa is little-side, so trunc low-u16)
            // only rounding + trunc(low u16) is necessary
            const Lanes = @Vector(count, u32);
            const bits: Lanes = @bitCast(values);
            const midbit = (bits >> @splat(16)) & @as(Lanes, @splat(1)); // extract lsb high-u16 as lsb low-u16
                                                                         // used for even/odd
                                                                         // if value is halfway (2..2.5..3) -> round based on even/odd
            const rounded = bits +% @as(Lanes, @splat(0x7FFF)) +% midbit; // 0x7FFF is u16-1 (midbit extracted)
                                                                            // by adding midbit, it will "snowball" its way to the upper u16 if round+1 is necessary
            return @as(@Vector(count, u16), @truncate(rounded >> @splat(16)));
        },
        else => @compileError("encode: not encodable"),
    }
}


pub fn tableTranslate(comptime format: types.Dtype.Recipe) [1 << format.bits]f32 {
    //Produce a lookup table that will be used by both loadScale() and decode()
    @setEvalBranchQuota(1_000_000);
    const t_size = 1 << format.bits;
    var elements: [t_size]u8 = undefined;
    for (&elements, 0..) |*element , idx| element.* = @intCast(idx);
    return comptime decode(format, t_size, elements);
}
