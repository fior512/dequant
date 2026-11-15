pub const Size = struct {
    label: []const u8,
    weight_count: usize,
};

pub const SIZES = [_]Size{ // working set = 0.5625 B/w in + 4 B/w out; zen4: L1d 32K, L2 1M, L3 32M
    .{ .label = "L1  ", .weight_count = 1 << 12 }, //  2.25 KiB in +  16 KiB out =  18.25 KiB
    .{ .label = "L2  ", .weight_count = 1 << 17 }, //    72 KiB in + 512 KiB out = 584 KiB
    .{ .label = "L3  ", .weight_count = 1 << 22 }, //  2.25 MiB in +  16 MiB out =  18.25 MiB
    .{ .label = "DRAM", .weight_count = 1 << 26 }, //    36 MiB in + 256 MiB out = 292 MiB
};

pub const TRIAL_COUNT: u32 = 3;
pub const TRIAL_MIN_NS: u64 = 20 * NS_PER_S;
pub const TRIAL_SLEEP_NS: u64 = 10 * NS_PER_S;
pub const TSC_CALIBRATION_NS: u64 = 500 * NS_PER_MS;
pub const BATCH_WEIGHT_COUNT: usize = 1 << 26; // weights per inner batch, between two `TscEnd`

pub const PIN_CPU: u32 = 2;
pub const SEED: u64 = 0xdec0de;
pub const SCALE_MIN: u8 = 0x01; // e4m3 codes, 0x00 0x7F sign excluded
pub const SCALE_MAX: u8 = 0x7E;
pub const SUB_BLOCK_WEIGHT_COUNT: usize = 16; // weights per e4m3 scale
pub const CACHE_LINE_SIZE: usize = 64;

pub const NS_PER_S: u64 = 1_000_000_000;
pub const NS_PER_MS: u64 = 1_000_000;
