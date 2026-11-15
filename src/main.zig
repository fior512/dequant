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
    const out_bits: usize = types.Dtype.RECIPE.get(
        shared.dequant_t,
    ).bits; // TODO: better name, this is single weight bits size

    while (true) {
        const layer_idx = shared.next.fetchAdd(1, .monotonic); // no sync
        if (layer_idx >= shared.layers.len) return; // done

        const layer = shared.layers[layer_idx];
        try weight_reader.seekTo(layer.weight_offset);
        try scales_reader.seekTo(layer.scale_offset);

        var rem_size = layer.weight_size;
        while (rem_size > 0) {
            const weight_size: usize = @intCast(@min(CHUNK_SIZE, rem_size));
            rem_size -= weight_size;
            const weights = try weight_reader.interface.take(weight_size);

            // FAST PATH (not quantized weights)
            const scheme = layer.scheme orelse {
                std.mem.doNotOptimizeAway(weights); // endpoint
                continue;
            };

            // DEQUANT
            const weight_bits: usize = types.Dtype.RECIPE.get(
                types.Scheme.RECIPE.get(scheme).weight,
            ).bits;
            const scales = try scales_reader.interface.take(
                weight_size / scheme.BlockSize() * scheme.ScaleSize(),
            );
            const out = out_buf[0 .. weight_size * 8 // bytes -> bits
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
    const smp = std.heap.smp_allocator;
    var arena = std.heap.ArenaAllocator.init(smp);

    const args = try flags(init.minimal.args.iterate());

    //FILE
    const in = try std.Io.Dir.cwd().openFile(init.io, args.path, .{});
    defer in.close(init.io);

    //HEADER MAPING
    const layers = try safetensors.parseJson(
        in,
        init.io,
        arena.allocator(),
    ); // per threads (ratio json/binary bytes is really big)
    // or
    // single specialized thread looping (multi .safet K3)
    //THREADPOOL
    var shared: Shared = .{
        .io = init.io,
        .layers = layers,
        .in = in,
        .dequant_t = args.dequant_t,
    };

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

    dequant.dequant(
        .nvfp4,
        .f32,
        .{ .weights = &weights, .scales = &scales, .midpoint = &.{}, .global_scale = 2.0 },
        &out,
    );

    const values: *const [16]f32 = @ptrCast(&out);
    try std.testing.expectEqual(@as(f32, -2.0), values[0]);
    try std.testing.expectEqual(@as(f32, 12.0), values[1]);
}

// DEQUANT
test "dequant: nvfp4 to f32, f16, bf16" {
    // 2 blocks, each holds every nibble once (low nibble first): 0x0, 0x1, .. 0xF
    const weights = [_]u8{ 0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE } ** 2;
    const scales = [_]u8{ 0x38, 0x30 }; // e4m3: 1.0, 0.5
    const in: dequant.Input = .{
        .weights = &weights,
        .scales = &scales,
        .midpoint = &.{},
        .global_scale = 2.0,
    };

    const want = [32]f32{
        0, 1, 2, 3, 4, 6, 8, 12, -0.0, -1, -2, -3, -4, -6, -8, -12, // e2m1 * 1.0 * 2.0
        0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6, // e2m1 * 0.5 * 2.0
    };

    var out_f32: [32]f32 = undefined;
    dequant.dequant(.nvfp4, .f32, in, @ptrCast(&out_f32));
    for (want, out_f32) |w, got| try std.testing.expectEqual(
        @as(u32, @bitCast(w)),
        @as(u32, @bitCast(got)),
    ); // bits: -0.0 != 0.0

    var out_f16: [32]f16 = undefined;
    dequant.dequant(.nvfp4, .f16, in, @ptrCast(&out_f16));
    for (want, out_f16) |w, got| try std.testing.expectEqual(@as(f16, @floatCast(w)), got);

    var out_bf16: [32]u16 = undefined;
    dequant.dequant(.nvfp4, .bf16, in, @ptrCast(&out_bf16));
    for (want, out_bf16) |w, got| try std.testing.expectEqual(
        @as(u16, @truncate(@as(u32, @bitCast(w)) >> 16)),
        got,
    ); // exact values: bf16 == high u16
}

// PARSE JSON
test "safetensors: parseJson" {
    const io = std.testing.io;
    const json =
        \\{"__metadata__":{"format":"pt"},
        \\"fc.weight":{"dtype":"U8","shape":[2,8],"data_offsets":[0,16]},
        \\"fc.weight_scale":{"dtype":"F8_E4M3","shape":[2,1],"data_offsets":[16,18]},
        \\"fc.weight_scale_2":{"dtype":"F32","shape":[],"data_offsets":[18,22]},
        \\"fc.bias":{"dtype":"F32","shape":[2],"data_offsets":[22,30]}}
    ;
    const data = [_]u8{0x7A} ** 16 // fc.weight
    ++ [_]u8{ 0x38, 0x30 } // fc.weight_scale
    ++ [_]u8{ 0x00, 0x00, 0x00, 0x40 } // fc.weight_scale_2 = 2.0
    ++ [_]u8{0} ** 8; // fc.bias
    const size: [8]u8 = @bitCast(std.mem.nativeToLittle(u64, json.len));

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try tmp.dir.createFile(io, "model.safetensors", .{ .read = true });
    defer file.close(io);
    try file.writePositionalAll(io, &size ++ json ++ &data, 0);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const layers = try safetensors.parseJson(file, io, arena.allocator());
    const start = 8 + json.len;

    try std.testing.expectEqual(
        2,
        layers.len,
    ); // scale + global scale are merged in, __metadata__ skipped

    const fc = layers[0];
    try std.testing.expectEqualStrings("fc.weight", fc.name);
    try std.testing.expectEqual(.nvfp4, fc.scheme.?);
    try std.testing.expectEqualSlices(u64, &.{ 2, 16 }, fc.shape); // 2 weights per packed byte
    try std.testing.expectEqual(start + 0, fc.weight_offset);
    try std.testing.expectEqual(16, fc.weight_size);
    try std.testing.expectEqual(start + 16, fc.scale_offset);
    try std.testing.expectEqual(2.0, fc.global_scale);

    const bias = layers[1];
    try std.testing.expectEqualStrings("fc.bias", bias.name);
    try std.testing.expectEqual(null, bias.scheme);
    try std.testing.expectEqualSlices(u64, &.{2}, bias.shape);
    try std.testing.expectEqual(start + 22, bias.weight_offset);
    try std.testing.expectEqual(8, bias.weight_size);
}
