const std = @import("std");

pub const CACHE_LINE = std.atomic.cache_line; // 128 x86
pub const CHUNK_SIZE = 0x40000; // 256KiB, packed weights per chunk
pub const SCALES_SIZE = CHUNK_SIZE / 8; // smallest scale block: nvfp4 16 elements -> 8 B -> 1 scale
                                                      
pub const Error = error{ //TODO split per module
    InvalidThreadCount,
    InvalidFile,
    InvalidTypeArgument,
    MissingTypeArgument,
    InvalidNumberArguent,
};


