const std = @import("std");
const types = @import("types.zig");
const safetensors = @import("safetensors.zig");
const dequant = @import("dequant.zig");

const CACHE_LINE = std.atomic.cache_line;
const CHUNK_SIZE = 0x40000; // 256KiB, max layer slice
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
        shared.out_t,
    ).bits; // TODO: better name, this is single weight bits size

    while (true) {
        const layer_idx = shared.next.fetchAdd(1, .monotonic);
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
            / 8 // bytes
            ];

            dequant.dequant(scheme, shared.out_t, .{
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
    path: []const []const u8,
    out_t: types.Dtype,
    threads: usize,
};

inline fn flags(gpa: std.mem.Allocator, args: std.process.Args.Iterator) !Args {
    var it = args;
    _ = it.skip();

    var rest: std.ArrayList([]const u8) = .empty;
    defer rest.deinit(gpa);

    while (it.next()) |arg| try rest.append(gpa, arg);
    if (rest.items.len < 3) @panic(
        "Args: missing argument (binary {.safetensors} {out_t} {threads})",
    ); // missing

    //.SAFETENSORS
    const path = try gpa.dupe([]const u8, rest.items[0 .. rest.items.len - 2]);

    //DEQUANT TYPE
    const out_t =
        std.meta.stringToEnum(types.Dtype, rest.items[rest.items.len - 2]) orelse
            @panic("Args: Unknown output type");

    //THREADPOOL
    const threads =
        std.fmt.parseInt(usize, rest.items[rest.items.len - 1], 10) catch
            @panic("Args: Invalid thread count");

    if (threads == 0 or threads > 12) @panic("Args: Invalid thread count");

    return Args{
        .path = path,
        .out_t = out_t,
        .threads = threads,
    };
}

const Shared = struct {
    next: std.atomic.Value(usize) align(CACHE_LINE) = .init(0), // done/in-progress
    io: std.Io,
    layers: []const safetensors.Layer,
    in: std.Io.File,
    out_t: types.Dtype,
};

pub fn main(init: std.process.Init) !void {
    const smp = std.heap.smp_allocator;
    var arena = std.heap.ArenaAllocator.init(smp);
    defer arena.deinit();

    // ARGS
    const args = try flags(init.gpa, init.minimal.args.iterate());
    const threads: []std.Thread = try init.gpa.alloc(std.Thread, args.threads);
    defer init.gpa.free(args.path);
    defer init.gpa.free(threads);

    for (args.path) |file| {
        //FILE
        const in = try std.Io.Dir.cwd().openFile(init.io, file, .{});
        defer in.close(init.io);

        //HEADER MAPING
        const layers = try safetensors.parseJson(
            in,
            init.io,
            arena.allocator(),
        );

        //THREADPOOL
        var shared: Shared = .{
            .io = init.io,
            .layers = layers,
            .in = in,
            .out_t = args.out_t,
        };
        for (threads) |*thread| thread.* = try std.Thread.spawn(.{}, worker, .{&shared});
        for (threads) |thread| thread.join();
    }
}

// -------------------------------------------------------------------------------------
// TESTS

// DEQUANT
test "dequant: nvfp4" {
    const weights: [16]u8 = @bitCast(
        @as([2][8]u8, @splat(.{ 0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE })),
    );
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
    for (want, out_f16) |w, got| try std.testing.expectEqual(
        @as(u16, @bitCast(@as(f16, @floatCast(w)))),
        @as(u16, @bitCast(got)),
    );

    var out_bf16: [32]u16 = undefined;
    dequant.dequant(.nvfp4, .bf16, in, @ptrCast(&out_bf16));
    for (want, out_bf16) |w, got| try std.testing.expectEqual(
        @as(u16, @truncate(@as(u32, @bitCast(w)) >> 16)),
        got,
    ); // exact values: bf16 == high u16
}

test "dequant: mxfp4" {
    const weights: [32]u8 = @bitCast(
        @as([4][8]u8, @splat(.{ 0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE })),
    );
    const scales = [_]u8{ 0x80, 0x7F }; // e8m0: 2^1, 2^0
    const in: dequant.Input = .{
        .weights = &weights,
        .scales = &scales,
        .midpoint = &.{},
        .global_scale = 1.0,
    };

    const x2 = [16]f32{ 0, 1, 2, 3, 4, 6, 8, 12, -0.0, -1, -2, -3, -4, -6, -8, -12 };
    const x1 = [16]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6 };
    const want = x2 ++ x2 ++ x1 ++ x1;

    var out_f32: [64]f32 = undefined;
    dequant.dequant(.mxfp4, .f32, in, @ptrCast(&out_f32));
    for (want, out_f32) |w, got| try std.testing.expectEqual(
        @as(u32, @bitCast(w)),
        @as(u32, @bitCast(got)),
    );

    var out_f16: [64]f16 = undefined;
    dequant.dequant(.mxfp4, .f16, in, @ptrCast(&out_f16));
    for (want, out_f16) |w, got| try std.testing.expectEqual(
        @as(u16, @bitCast(@as(f16, @floatCast(w)))),
        @as(u16, @bitCast(got)),
    );

    var out_bf16: [64]u16 = undefined;
    dequant.dequant(.mxfp4, .bf16, in, @ptrCast(&out_bf16));
    for (want, out_bf16) |w, got| try std.testing.expectEqual(
        @as(u16, @truncate(@as(u32, @bitCast(w)) >> 16)),
        got,
    );
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
    const data = @as([16]u8, @splat(0x7A)) // fc.weight
        ++ [_]u8{ 0x38, 0x30 } // fc.weight_scale
        ++ [_]u8{ 0x00, 0x00, 0x00, 0x40 } // fc.weight_scale_2 = 2.0
        ++ @as([8]u8, @splat(0)); // fc.bias
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
