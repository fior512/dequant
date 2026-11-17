const std = @import("std");

pub const Size = struct {
    label: []const u8,
    weight_count: usize,
};

pub const SIZES = [_]Size{ // working set = in + 4 B/w out; in: nvfp4 0.5625 B/w, mxfp4 0.53125 B/w; zen4: L1d 32K, L2 1M, L3 32M
    .{ .label = "L1  ", .weight_count = 1 << 12 }, //  2.25 KiB in +  16 KiB out =  18.25 KiB (nvfp4)
    .{ .label = "L2  ", .weight_count = 1 << 17 }, //    72 KiB in + 512 KiB out = 584 KiB
    .{ .label = "L3  ", .weight_count = 1 << 22 }, //  2.25 MiB in +  16 MiB out =  18.25 MiB
    .{ .label = "DRAM", .weight_count = 1 << 26 }, //    36 MiB in + 256 MiB out = 292 MiB
};

pub const Scheme = enum {
    nvfp4,
    mxfp4,

    pub const Recipe = struct {
        scale_block_weight_count: usize, // weights per scale
        llama_block_weight_count: usize, // weights per llama.cpp block
        scale_min: u8,
        scale_max: u8,
    };

    pub const RECIPE = std.EnumArray(Scheme, Recipe).init(.{
        .nvfp4 = .{ .scale_block_weight_count = 16, .llama_block_weight_count = 64, .scale_min = 0x01, .scale_max = 0x7E }, // e4m3 codes, 0x00 0x7F sign excluded
        .mxfp4 = .{ .scale_block_weight_count = 32, .llama_block_weight_count = 32, .scale_min = 0x02, .scale_max = 0xFC }, // e8m0 codes, 0.5 * 2^(e-127) >= 2^-126, 6 * 2^(e-127) < 2^128 -> normal f32 only
    });
};

pub const TRIAL_COUNT: u32 = 3;
pub const TRIAL_MIN_NS: u64 = 20 * NS_PER_S;
pub const TRIAL_SLEEP_NS: u64 = 10 * NS_PER_S;
pub const TSC_CALIBRATION_NS: u64 = 500 * NS_PER_MS;
pub const BATCH_WEIGHT_COUNT: usize = 1 << 26; // weights per inner batch, between two `TscEnd`

pub const PIN_CPU: u32 = 2;
pub const SEED: u64 = 0xdec0de;
pub const CACHE_LINE_SIZE: usize = 64;

pub const NS_PER_S: u64 = 1_000_000_000;
pub const NS_PER_MS: u64 = 1_000_000;
