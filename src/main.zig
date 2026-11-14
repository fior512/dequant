const std = @import("std");
const types = @import("types.zig");
const safetensors = @import("safetensors.zig");
const dequant = @import("dequant.zig");

const CACHE_LINE = std.atomic.cache_line; // 128 x86
const CHUNK_SIZE = 0x40000; // 256KiB, packed weights per chunk
const SCALES_SIZE = CHUNK_SIZE / 8; // smallest scale block: nvfp4 16 elements -> 8 B -> 1 scale


const max_weights_per_byte = blk: {
    var max_count: usize = 0;
    for (std.enums.values(types.Scheme)) |scheme| {
        const weight_t = types.Scheme.RECIPE.get(scheme).weight;
        max_count = @max(max_count, 8 / types.Dtype.RECIPE.get(weight_t).bits);
    }
    break :blk max_count;
};

const max_out_value_size = blk: {
    var max_size: usize = 0;
    for (std.enums.values(types.Dtype)) |dtype| {
        const recipe = types.Dtype.RECIPE.get(dtype);
        if (recipe.encodable) max_size = @max(max_size, recipe.bits / 8);
    }
    break :blk max_size;
};
const OUT_SIZE = CHUNK_SIZE * max_weights_per_byte * max_out_value_size;


fn worker(shared: *Shared) !void {
    var weight_buf: [CHUNK_SIZE]u8 align(CACHE_LINE) = undefined;
    var scales_buf: [SCALES_SIZE]u8 align(CACHE_LINE) = undefined;
    var out_buf: [OUT_SIZE]u8 align(CACHE_LINE) = undefined;

    var weight_reader = shared.in.reader(shared.io, &weight_buf);
    var scales_reader = shared.in.reader(shared.io, &scales_buf);
    const out_bits: usize = types.Dtype.RECIPE.get(shared.dequant_t).bits; // TODO: better name, this is single weight bits size

    while (true) {
        const layer_idx = shared.next.fetchAdd(1, .monotonic); // no sync
        if (layer_idx >= shared.layers.len) return; // done
        
        const layer = shared.layers[layer_idx];
        try weight_reader.seekTo(layer.weight_offset);
        try scales_reader.seekTo(layer.scale_offset);

        var rem_size = layer.weight_size;
        while (rem_size>0) {
            const weight_size: usize = @intCast(@min(CHUNK_SIZE, rem_size));
            rem_size -= weight_size;
            const weights = try weight_reader.interface.take(weight_size);

            // FAST PATH (not quantized weights)
            const scheme = layer.scheme orelse {
                std.mem.doNotOptimizeAway(weights); // whatever endpoint we need to send weights to (cluster V|RAM, writer, ..)
                continue;
            };

            // DEQUANT
            const weight_bits: usize = types.Dtype.RECIPE.get(types.Scheme.RECIPE.get(scheme).weight).bits;
            const scales = try scales_reader.interface.take(weight_size / scheme.BlockSize() * scheme.ScaleSize());
            const out = out_buf[
                0..weight_size 
                * 8 // bytes -> bits
                / weight_bits // how many weights
                * out_bits // output bits
                / 8 // bits -> bytes
            ]; 

            dequant.dequant(scheme, shared.dequant_t, .{
                .weights = weights,
                .scales = scales,
                .midpoint = &.{},
                .global_scale = layer.global_scale,
            }, out);

            std.mem.doNotOptimizeAway(out); //endpoint
        }
    }
}

const Args = struct {
    path: []const u8,
    dequant_t: types.Dtype,
    threads: usize,
};

inline fn flags(args: std.process.Args.Iterator) !Args {
    var it = args;
    _ = it.skip();

    //.SAFETENSORS
    const path = if (it.next()) |path| path else @panic("Args: missing .safetensors path");

    //DEQUANT TYPE
    const stype = it.next() orelse @panic("Args: Missing dequantization type");
    const dequant_type = std.meta.stringToEnum(
        types.Dtype,
        stype,
    ) orelse @panic("Args: Unknown dequant type");

    //THREADPOOL
    const max_threads = std.Thread.getCpuCount() catch 1;
    const threads = if (it.next()) |str|
        std.fmt.parseInt(usize, str, 10) catch @panic("Args: Invalid thread count")
    else
        max_threads;

    if (threads == 0 or threads > max_threads) @panic("Args: Invalid thread count");
    
    return Args{
        .path = path,
        .dequant_t = dequant_type,
        .threads = threads,
    };
}

const Shared = struct {
    next: std.atomic.Value(usize) align(CACHE_LINE) = .init(0), // done/in-progress
    io: std.Io,
    layers: []const safetensors.Layer,
    in: std.Io.File,
    dequant_t: types.Dtype,
};

pub fn main(init: std.process.Init) !void {
    const args = try flags(init.minimal.args.iterate());

    //FILE
    const in = try std.Io.Dir.cwd().openFile(init.io, args.path, .{});
    defer in.close(init.io);

    //HEADER MAPING
    const layers = try safetensors.parseJson(in, init.io, init.arena.allocator()); // per threads (ratio json/binary bytes is really big)
                                                                                                        // or
                                                                                                        // single specialized thread looping (multi .safet K3)

    var shared: Shared = .{
        .io = init.io,
        .layers = layers,
        .in = in,
        .dequant_t = args.dequant_t,
    };

    //THREADPOOL
    const threads: []std.Thread = try init.gpa.alloc(std.Thread, args.threads);
    defer init.gpa.free(threads);
    for (threads) |*thread| {
        thread.* = try std.Thread.spawn(.{}, worker, .{&shared});
    }

    for (threads) |thread| thread.join();
}



test "nvfp4 to f32" { // Vibecoded
    const weights = [_]u8{0x7A} ** 8; // 16 weights
    const scales = [_]u8{0x38}; //e4m1 1.0
    var out: [16 * 4]u8 align(4) = undefined;

    dequant.dequant(.nvfp4, .f32, .{
        .weights = &weights,
        .scales = &scales,
        .midpoint = &.{},
        .global_scale = 2.0
    }, &out);

    const values: *const [16]f32 = @ptrCast(&out);
    try std.testing.expectEqual(@as(f32, -2.0), values[0]);
    try std.testing.expectEqual(@as(f32, 12.0), values[1]);
}
