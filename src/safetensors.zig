const std = @import("std");
const generic = @import("generic.zig");
const types = @import("types.zig");

// LAYOUT:
// [u64 .little header_size][json, header, header_size][binary data]
// "name": { "dtype": "U8", "shape": [..], "data_offset": [begin, end) } // absolute = 8+header_size+offset
const Error = error {
    InvalidHeader,
    MissingScale,
    MissingGlobalScale,
    UnknownScheme,
    ScaleLayout,
};


pub const Chunk = struct {
    offset: u64, // from start
    scale_offset: u64, // from start
    global_scale: f32,
    len: u32,
    scales_len: u32,
    tag: types.SchemeTag,
};


//TODO find new name for HEADER, ambiguous
const Header = struct { // json format
    weight: []const u8,
    scale: []const u8,
    global: ?[]const u8,
    is_inverse: bool, // global scale may be 1/x 
};

const HEADERS = [_]Header { // format to parse
    .{ .weight = ".weight", .scale = ".weight_scale", .global = ".weight_scale_2", .is_inverse = false },
    .{ .weight = ".weight_packed", .scale = ".weight_scale", .global = "weight_global_scale", .is_inverse = true},
    .{ .weight = "_blocks", .scale = "_scales", .global = null, .is_inverse = false },
};




// parse once before threadpool
pub fn parseHeader(in: std.Io.File, io: std.Io, gpa: std.mem.Allocator) ![]Chunk {
    var buf: [4096]u8 = undefined;
    var reader = in.reader(io, &buf);

    //HEADER fetch
    const header_size = try reader.interface.takeInt(u64, .little);
    const start = 8 + header_size; // u64 + header
    const bytes = try gpa.alloc(u8, header_size); // HEADER
    defer gpa.free(bytes);
    try reader.interface.readSliceAll(bytes);

    //JSON parsing
    const header = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer header.deinit();
    const entries = switch (header.value) {
        .object => |object| object,
        else => return Error.InvalidHeader,
    };

    var chunks: std.ArrayList(Chunk) = .empty; // Array list aligned
    defer chunks.deinit(gpa);

    for (entries.keys(), entries.values()) |name, entry| {
        //TODO check what is quantized for dequant then parse it as chunks
        // maybe struct rework, i would love to avoid a new one
        // or find a way to define them in a readable way
    }
}


