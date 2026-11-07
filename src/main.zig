const std = @import("std");
const generic = @import("generic.zig");
const types = @import("types.zig");
const safetensors = @import("safetensors.zig");


fn worker(shared: *Shared) !void {
    var buf: [generic.CHUNK_SIZE]u8 align(generic.CACHE_LINE) = undefined;
    var scales_buf: [generic.SCALES_SIZE]u8 align(generic.CACHE_LINE) = undefined;

    var weights_reader = shared.in.reader(shared.io, &buf); //pread
    var scales_reader = shared.in.reader(shared.io, &scales_buf);

    while (true) { //TODO is prefetch right ?
        const chunk_idx = shared.next.fetchAdd(1, .monotonic); //no sync
        if (chunk_idx >= shared.chunks.len) return; // all done
        const chunk = shared.chunks[chunk_idx];

        //READ
        try weights_reader.seekTo(chunk.offset);
        const pweights = try weights_reader.interface.take(chunk.len);
        try scales_reader.seekTo(chunk.scales_offset);
        const scales = try scales_reader.interface.take(chunk.scales_len);

        //TODO WORK: switch chunk.tag inline else x dequant_dtype inline else -> dequantRows
        _ = pweights;
        _ = scales;
    }
}

const Args = struct {
    path: []const u8,
    dequant_dtype: types.Dtype,
    threads: usize,
};

inline fn flags(args: std.process.Args.Iterator) generic.Error!Args {
    var it = args;
    _ = it.skip();

    //.SAFETENSORS
    const path = if (it.next()) |path| path else return generic.Error.InvalidFile;

    //DEQUANT TYPE
    const stype = it.next() orelse return generic.Error.MissingTypeArgument;
    const dequant_type = std.meta.stringToEnum(
        types.Dtype,
        stype,
    ) orelse return generic.Error.InvalidTypeArgument;

    //THREADPOOL
    const max_threads = std.Thread.getCpuCount() catch 1;
    const threads = if (it.next()) |str|
        std.fmt.parseInt(usize, str, 10) catch return generic.Error.InvalidThreadCount
    else
        max_threads;

    if (threads == 0 or threads > max_threads)
        return generic.Error.InvalidThreadCount;

    
    return Args{
        .path = path,
        .dequant_dtype = dequant_type,
        .threads = threads,
    };
}

const Shared = struct {
    next: std.atomic.Value(usize) align(generic.CACHE_LINE) = .init(0), // done/in-progress
    io: std.Io,
    chunks: []const safetensors.Chunk,
    in: std.Io.File,
    dequant_dtype: types.Dtype,
};

pub fn main(init: std.process.Init) !void {
    const args = try flags(init.minimal.args.iterate());

    //FILE
    const in = try std.Io.Dir.cwd().openFile(init.io, args.path, .{});
    defer in.close(init.io);

    //HEADER MAPING
    const chunks = try safetensors.parseHeader(in, init.io, init.gpa); // parse once
                                                                                             // json isn't that big
                                                                                             // check simdjson interop
    defer init.gpa.free(chunks);

    var shared: Shared = .{
        .io = init.io,
        .chunks = chunks,
        .in = in,
        .dequant_dtype = args.dequant_dtype,
    };

    //THREADPOOL
    const threads: []std.Thread = try init.gpa.alloc(std.Thread, args.threads);
    defer init.gpa.free(threads);
    for (threads) |*thread| {
        thread.* = try std.Thread.spawn(.{}, worker, .{&shared});
    }

    for (threads) |thread| thread.join();
}
