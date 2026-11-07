const std = @import("std");

// NVFP4 per rows
//  packed: cols/2 bytes, byte = [+nibble= 2k+1] [-nibble= 2k]
//  scales: cols/16 bytes, one E4M3 scale per block of 16 elements (= 8 packed bytes)
//  value= E2M1(nibble) * E4M3(block_scale) * global_scale
//
//
//  safetensor:
//  obj                 dtype       shape
//  "X.weights"         u8          [2,16]  // 32 FP4 values per row/2 per byte = 16
//  "X.weights_scale"   F8_E4M3     [2,2]   // 32 values per row/16 per block = 2
//  "X.weights_scale_2" F32         []      // global_scale
//
// packed bytes (flat, row-major):
//   byte:      0..7    | 8..15     | 16..23    | 24..31
//   row:       0       | 0         | 1         | 1
//   cols:      0..15   | 16..31    | 0..15     | 16..31
//   scale:  scales[0]  | scales[1] | scales[2] | scales[3]
//
//   high nibble is n+1



pub const Dtype = enum { f4e2m1, f8e4m3, f8e5m2, f8e8m0, f16, bf16, f32 };

pub const Scheme = struct {
    weight: Dtype,
    scale: Dtype,
    block: u32,
};
pub const SchemeTag = enum { nvfp4, mxfp4 }; //TODO worth specialized file ?
pub fn preset(tag: SchemeTag) Scheme { //TODO  ''
    return switch (tag) {
        .nvfp4 => .{ .weight = .f4e2m1, .scale = .f8e4m3, .block = 16 },
        .mxfp4 => .{ .weight = .f4e2m1, .scale = .f8e8m0, .block = 32 },
    };
}
pub fn fromScale(scale: Dtype) ?SchemeTag {
    for (std.enums.values(SchemeTag)) |tag| if (preset(tag).scale == scale) return tag;
    return null;
}

pub const Path = enum(u8) {
    float,      // sign + exp + mant; decoded through f16
    exponant,   // e8m0; decoded via shift in f32
    native,     // f16|bf16|f32, widen|shift|bitcast
};

pub const Format = struct { //TODO rename
    safetensors: ?[]const u8, // "dtype" given as str in header
    bits: u8,
    exp_bits: u8,
    mant_bits: u8,
    bias: u8,
    path: Path, // type of logic
    encodable: u1,
    decodable: u1,
};

const RECIPE = std.EnumArray(Dtype, Format).init(.{ //scalable recipes
    .f4e2m1 = .{ .safetensors = null, .bits = 4, .exp_bits = 2, .mant_bits = 1, .bias =   1, .path = .float, .encodable = false, .decodable = true }, //"U8"
    .f8e4m3 = .{ .safetensors = "F8_E4M3", .bits = 8, .exp_bits = 4, .mant_bits = 3, .bias =   7, .path = .float, .encodable = false, .decodable = true },
    .f8e5m2 = .{ .safetensors = "F8_E5M2", .bits = 8, .exp_bits = 5, .mant_bits = 2, .bias =  15, .path = .float, .encodable = false, .decodable = true },
    .f8e8m0 = .{ .safetensors = "F8_E8M0", .bits = 8, .exp_bits = 8, .mant_bits = 0, .bias = 127, .path = .exponant, .encodable = false, .decodable = true },
    .f16    = .{ .safetensors = "F16", .bits = 16, .exp_bits = 5, .mant_bits = 10, .bias = 15, .path = .native, .encodable = true, .decodable = true },
    .bf16   = .{ .safetensors = "BF16", .bits = 16, .exp_bits = 8, .mant_bits = 7, .bias = 127, .path = .native, .encodable = true, .decodable = true },
    .f32    = .{ .safetensors = "F32", .bits = 32, .exp_bits = 8, .mant_bits = 23, .bias = 127, .path = .native, .encodable = true, .decodable = true },
});




pub fn fromSafetensors(name: []const u8) ?Dtype {
    for (std.enums.values(Dtype)) |dtype| {
        const safetensors = RECIPE.get(dtype).safetensors orelse continue;
        if (std.mem.eql(u8, safetensors, name)) return dtype;
    }
}



