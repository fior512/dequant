const std = @import("std");

// NVFP4 per rows
//  packed: cols/2 bytes, byte = [+nibble= 2k+1] [-nibble= 2k]
//  scales: cols/16 bytes, one E4M3 scale per block of 16 elements (= 8 packed bytes)
//  value= E2M1(nibble) * E4M3(block_scale) * global_scale
//
//
//  safetensors:
//  obj                 dtype       shape
//  "X.weights"         u8          [2,16]  // 32 FP4 values per row/2 per byte = 16
//  "X.weights_scale"   F8_E4M3     [2,2]   // 32 values per row/16 per block = 2
//  "X.weights_scale_2" F32         []      // global_scale
//
// packed bytes (flat, row-major):
//   byte:      0..7        8..15           16..23          24..31
//   row:       0           0               1               1
//   cols:      0..15       16..31          0..15           16..31
//   scale: scales[0]       scales[1]       scales[2]       scales[3]
//
//   high nibble is n+1



//IDEA:
//  bytes, scale, zero
//  
//i4 base -> 2/byte
//  hi = v >> @as(@Vector(size/2, u3), @splat(4)); // 0x0H, 0x0H, 0x0H
//  lo = v &  @as(@Vector(size/2, u3), @splat(0x0F)); // 0x0L, 0x0L, 0x0L
//  q = shuffle(u8, lo, hi, i%2); // 0x0L, 0x0H, 0x0L, 0x0H (L first because little endian)
//
//1) std affine s(q-z)
//  bias = -@as(f32, @floatfromint(z)) * s      // negative because we goes from s(q-z) to qs-zs  (fma setup)
//  return FMA(type_o, @FloatFromInt(q), @splat(s), @splat(b))
//
//2) RTN s*q
//  signed = q-8  // [0..15] -> [-8..7]
//  return @as(type_o, @FloatFromInt(sign) * @Splat(s))
//
//
//swap @FloatFromInt for decode E_M_

pub const Dtype = enum { 
    f4e2m1,
    f8e4m3,
    f16,
    bf16,
    f32,

    pub const Kind = enum(u8) {
        float,      // sign + exp + mant; decoded through f16
        exponent,   // e8m0; decoded via shift in f32
        native,     // f16|bf16|f32, widen|shift|bitcast
        integer,
    };

    pub const Recipe = struct { //TODO rename
        header: ?[]const u8, // "dtype" given as str in header
        bits: u8,
        exp_bits: u8,
        mant_bits: u8,
        bias: u8,
        kind: Kind, // type of logic
        encodable: bool,
        decodable: bool,
    };

    pub const RECIPE = std.EnumArray(Dtype, Recipe).init(.{ // scalable rulesets
        .f4e2m1 = .{ .header = null, .bits = 4, .exp_bits = 2, .mant_bits = 1, .bias = 1, .kind = .float, .decodable = true, .encodable = false }, //"U8"
        .f8e4m3 = .{ .header = "F8_E4M3", .bits = 8, .exp_bits = 4, .mant_bits = 3, .bias = 7, .kind = .float, .decodable = true, .encodable = false },
        .f16    = .{ .header = "F16", .bits = 16, .exp_bits = 5, .mant_bits = 10, .bias = 15, .kind = .native, .decodable = true, .encodable = true },
        .bf16   = .{ .header = "BF16", .bits = 16, .exp_bits = 8, .mant_bits = 7, .bias = 127, .kind = .native, .decodable = true, .encodable = true },
        .f32    = .{ .header = "F32", .bits = 32, .exp_bits = 8, .mant_bits = 23, .bias = 127, .kind = .native, .decodable = true, .encodable = true },
    });

    pub fn NameToDtype(name: []const u8) ?Dtype {
        for (std.enums.values(Dtype)) |dtype| {
            const header = RECIPE.get(dtype).header orelse continue;
            if (std.mem.eql(u8, header, name)) return dtype;
        }
        return null;
    }


    pub fn Storage(comptime output_t: Dtype) type { // encode complexity bottleneck
        return switch (output_t) {
            .f16 => f16,
            .f32 => f32,
            .bf16 => u16,
            else => @compileError("Storage: not encodable"),
        };
    }


};




pub const Scheme = enum {
    nvfp4,

    pub const Translation = enum {
        affine, // s(q-z)
        RTN,    // sq
    };

    pub const Recipe = struct {
        block: u32,
        weight: Dtype,
        scale: Dtype,
        translation: Translation,
    };
    
    pub const RECIPE = std.EnumArray(Scheme, Recipe).init(.{
        .nvfp4 = .{ .weight = .f4e2m1, .scale = .f8e4m3, .block = 16, .translation = .RTN },
        //.mxfp4 => .{ .weight = .f4e2m1, .scale = .f8e8m0, .block = 32, .translation = .affine }, // later

    });

    pub fn ScaleToScheme(scale: Dtype) ?Scheme {
        for (std.enums.values(Scheme)) |scheme| {
            if (RECIPE.get(scheme).scale == scale) return scheme;
        }
        return null;
    }

    pub fn BlockSize(scheme: Scheme) usize {
        const recipe = RECIPE.get(scheme); //Scheme
        return recipe.block * Dtype.RECIPE.get(recipe.weight).bits / 8;
    }

    pub fn ScaleSize(scheme: Scheme) usize {
        return Dtype.RECIPE.get(RECIPE.get(scheme).scale).bits / 8;
    }
};












