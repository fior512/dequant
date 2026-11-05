const std = @import("std");

const CACHE_LINE = 64;
const CHUNK_SIZE = 0x40000; // 256KiB

const Error = error{
    InvalidThreadCount,
    InvalidFile,
};

const Shared = struct {
    io: std.Io,
    in: std.Io.File,
    out: std.Io.File,
    start: u64,
    size: u64,
    chunk: usize,
    next: std.atomic.Value(usize) = .init(0), // id
};

fn worker(shared: *Shared) !void {
    var buf: [CHUNK_SIZE]u8 align(CACHE_LINE) = undefined;

    while (true) { //TODO how/where to async/prefetch next chunk
        const chunk = shared.next.fetchAdd(1, .monotonic);
        if (chunk >= shared.chunk) return;

        const offset = chunk * CHUNK_SIZE;
        const len: usize = @intCast(@min(CHUNK_SIZE, shared.size - offset));

        const bytes = try blk: {
            var reader = shared.in.reader(shared.io, buf[0..len]); //posix
            try reader.seekTo(offset);
            break :blk reader.interface.take(len);
        };

        //TODO WORK

        try shared.in.writePositionalAll(shared.io, bytes, offset);
    }
}

const Args = struct {
    path: []const u8,
    threads: usize,
};

inline fn flags(args: std.process.Args.Iterator) Error!Args {
    var it = args;
    _ = it.next();
    const path = if (it.next()) |path| path else return Error.InvalidFile;

    const max_threads = std.Thread.getCpuCount() catch 1;
    const threads = if (it.next()) |str|
        std.fmt.parseInt(usize, str, 10) catch return Error.InvalidThreadCount
    else
        max_threads;

    if (threads == 0 or threads > max_threads)
        return Error.InvalidThreadCount;

    return Args{
        .path = path,
        .threads = threads,
    }; // TODO: dequant target
}




pub fn main(init: std.process.Init) !void {
    const args = try flags(init.minimal.args.iterate());

    //FILE
    const in = try std.Io.Dir.cwd().openFile(init.io, args.path, .{});
    defer in.close(init.io);

    const size = (try in.stat(init.io)).size;

    var shared: Shared = .{ // in ctx
        .io = init.io,
        .in = in,
        .out = undefined,
        .start = 0, //TODO check .safetensor format
        .size = size,
        .chunk = @intCast((size + CHUNK_SIZE - 1) / CHUNK_SIZE),
    };

    //THREADPOOL
    const threads: []std.Thread = try init.gpa.alloc(std.Thread, args.threads);
    for (threads) |*thread| {
        thread.* = try std.Thread.spawn(.{}, worker, .{&shared});
    }

    for (threads) |thread| thread.join();
}


// SAFETENSOR WIKI:
// first 8bytes: header size, as little u64
// header: model archi with obj.idx.param_type:{dtype, shape, data_offset}
// 
// about NVFP4:
// - no "NVFP4" definition in `dtype`. 
// check if in u8 + .weight|_packed + .weight_scale=F8_E4M3
//      
