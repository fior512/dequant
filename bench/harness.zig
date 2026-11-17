const std = @import("std");
const linux = std.os.linux;
const generic = @import("generic.zig");

const ALIGN: std.mem.Alignment = .fromByteUnits(generic.CACHE_LINE_SIZE);

const E2M1_VALUES = [16]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6 };

pub const Logical = struct {
    nibbles: []u8, // weight_count, one e2m1 code per weight
    scales: []u8, // weight_count / `scale_block_weight_count`, one e4m3|e8m0 code per scale block
    reference: []f32, // weight_count, e2m1 * scale in f32
    weight_count: usize,
};

const TrialResult = struct {
    rep_count: u64,
    tsc_ticks: u64,
    wall_ns: u64,
};

pub fn ParseScheme(args: std.process.Args.Iterator) generic.Scheme {
    var iterator = args;
    _ = iterator.skip();
    const scheme_name = iterator.next() orelse @panic("Args: missing scheme (nvfp4|mxfp4)");
    return std.meta.stringToEnum(generic.Scheme, scheme_name) orelse @panic("Args: unknown scheme");
}

pub fn Main(comptime Kernel: type, comptime name: []const u8) !void {
    const recipe = comptime generic.Scheme.RECIPE.get(Kernel.SCHEME);
    PinCpu(generic.PIN_CPU);
    const tsc_hz = CalibrateTscHz();
    std.debug.print("{s}: {t}, cpu {d}, tsc {d:.4} GHz, trials {d} x >= {d} s, sleep {d} s\n", .{
        name,
        Kernel.SCHEME,
        generic.PIN_CPU,
        tsc_hz / 1e9,
        generic.TRIAL_COUNT,
        generic.TRIAL_MIN_NS / generic.NS_PER_S,
        generic.TRIAL_SLEEP_NS / generic.NS_PER_S,
    });

    var trial_index_global: u32 = 0;
    for (generic.SIZES) |size| {
        var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
        defer arena.deinit();
        const allocator = arena.allocator();

        const logical = try GenerateLogical(Kernel.SCHEME, recipe, allocator, size.weight_count);
        const kernel = try Kernel.Prepare(allocator, &logical);
        const out = try allocator.alignedAlloc(f32, ALIGN, size.weight_count);

        // verify + pre fault `out`
        Kernel.Run(&kernel, out);
        for (out, logical.reference, 0..) |value, expected, weight_index| {
            if (value != expected) std.debug.panic(
                "{s} mismatch @{d}: {e} vs {e}",
                .{ name, weight_index, value, expected },
            );
        }
        const out_hash = HashOutput(out);

        const batch_rep_count: u64 = @max(generic.BATCH_WEIGHT_COUNT / size.weight_count, 1);
        std.debug.print(
            "{s} {s} weights {d} batch_reps {d} out_hash {x:0>16}\n",
            .{ name, size.label, size.weight_count, batch_rep_count, out_hash },
        );

        var ns_per_weight: [generic.TRIAL_COUNT]f64 = undefined;
        for (&ns_per_weight, 0..) |*trial_ns_per_weight, trial_index| {
            if (trial_index_global > 0) Sleep(generic.TRIAL_SLEEP_NS);
            trial_index_global += 1;

            const trial = Trial(Kernel, &kernel, out, batch_rep_count, tsc_hz);

            // report trial
            const weight_total: f64 = @floatFromInt(trial.rep_count * size.weight_count);
            const ticks: f64 = @floatFromInt(trial.tsc_ticks);
            const tsc_ns = ticks / tsc_hz * 1e9;
            trial_ns_per_weight.* = tsc_ns / weight_total;
            std.debug.print("{s} {s} trial {d}  reps {d:>10}  {d:7.4} ticks/w  {d:7.4} ns/w  {d:7.3} Gw/s  {d:7.2} GiB/s out  wall {d:.3} s\n", .{
                name,
                size.label,
                trial_index,
                trial.rep_count,
                ticks / weight_total,
                trial_ns_per_weight.*,
                weight_total / tsc_ns,
                weight_total * @sizeOf(f32) / tsc_ns * 1e9 / (1 << 30),
                @as(f64, @floatFromInt(trial.wall_ns)) / 1e9,
            });
        }

        // report summary
        std.mem.sort(f64, &ns_per_weight, {}, std.sort.asc(f64));
        std.debug.print("{s} {s} summary  min {d:7.4}  median {d:7.4}  max {d:7.4} ns/w\n", .{
            name,
            size.label,
            ns_per_weight[0],
            ns_per_weight[generic.TRIAL_COUNT / 2],
            ns_per_weight[generic.TRIAL_COUNT - 1],
        });
    }
}

fn Trial(
    comptime Kernel: type,
    kernel: *const Kernel,
    out: []f32,
    batch_rep_count: u64,
    tsc_hz: f64,
) TrialResult {
    const deadline_ticks: u64 = @intFromFloat(
        @as(f64, @floatFromInt(generic.TRIAL_MIN_NS)) / 1e9 * tsc_hz,
    );
    var rep_count: u64 = 0;

    const wall_start = MonotonicNs();
    const tsc_start = TscStart();
    var tsc_now = tsc_start;
    while (tsc_now - tsc_start < deadline_ticks) {
        for (0..batch_rep_count) |_| {
            Kernel.Run(kernel, out);
            ClobberMemory(out);
        }
        rep_count += batch_rep_count;
        tsc_now = TscEnd();
    }
    const wall_end = MonotonicNs();

    return .{
        .rep_count = rep_count,
        .tsc_ticks = tsc_now - tsc_start,
        .wall_ns = wall_end - wall_start,
    };
}

fn GenerateLogical(
    comptime scheme: generic.Scheme,
    comptime recipe: generic.Scheme.Recipe,
    allocator: std.mem.Allocator,
    weight_count: usize,
) !Logical {
    std.debug.assert(weight_count % recipe.llama_block_weight_count == 0);
    std.debug.assert(weight_count % recipe.scale_block_weight_count == 0);
    const nibbles = try allocator.alignedAlloc(u8, ALIGN, weight_count);
    const scales = try allocator.alignedAlloc(u8, ALIGN, weight_count / recipe.scale_block_weight_count);
    const reference = try allocator.alignedAlloc(f32, ALIGN, weight_count);

    var prng: std.Random.DefaultPrng = .init(generic.SEED);
    const random = prng.random();
    for (nibbles) |*nibble| nibble.* = random.int(u4);
    for (scales) |*scale| scale.* = random.intRangeAtMost(u8, recipe.scale_min, recipe.scale_max);

    // reference: e2m1 * scale
    for (reference, nibbles, 0..) |*value, nibble, weight_index| {
        const scale_code = scales[weight_index / recipe.scale_block_weight_count];
        const scale: f32 = switch (scheme) {
            .nvfp4 => blk: { // ue4m3: (1 + m/8) * 2^(e-7), subnormal m * 2^-9
                const exponent: i32 = (scale_code >> 3) & 0xF;
                const mantissa: f32 = @floatFromInt(scale_code & 0x7);
                break :blk if (exponent == 0) std.math.ldexp(mantissa, -9) else std.math.ldexp(8 + mantissa, exponent - 10);
            },
            .mxfp4 => std.math.ldexp(@as(f32, 1), @as(i32, scale_code) - 127), // e8m0: 2^(e-127)
        };
        value.* = E2M1_VALUES[nibble] * scale;
    }

    return .{
        .nibbles = nibbles,
        .scales = scales,
        .reference = reference,
        .weight_count = weight_count,
    };
}

fn HashOutput(out: []const f32) u64 {
    var hasher: std.hash.Wyhash = .init(generic.SEED);
    for (out) |value| {
        const bits: u32 = @bitCast(value);
        const canonical: u32 = if (bits << 1 == 0) 0 else bits; // -0 -> +0
        hasher.update(std.mem.asBytes(&canonical));
    }
    return hasher.final();
}

fn CalibrateTscHz() f64 {
    const wall_start = MonotonicNs();
    const tsc_start = TscStart();
    Sleep(generic.TSC_CALIBRATION_NS);
    const tsc_end = TscEnd();
    const wall_end = MonotonicNs();
    return @as(
        f64,
        @floatFromInt(tsc_end - tsc_start),
    ) * 1e9 / @as(f64, @floatFromInt(wall_end - wall_start));
}

fn PinCpu(cpu: u32) void {
    var cpu_set: linux.cpu_set_t = @splat(0);
    cpu_set[cpu / @bitSizeOf(usize)] |= @as(usize, 1) << @intCast(cpu % @bitSizeOf(usize));
    linux.sched_setaffinity(
        0,
        &cpu_set,
    ) catch |err| std.debug.panic("sched_setaffinity cpu {d}: {t}", .{ cpu, err });
}

fn Sleep(duration_ns: u64) void {
    var request: linux.timespec = .{
        .sec = @intCast(duration_ns / generic.NS_PER_S),
        .nsec = @intCast(duration_ns % generic.NS_PER_S),
    };
    var remaining: linux.timespec = undefined;
    while (linux.errno(linux.nanosleep(&request, &remaining)) == .INTR) request = remaining;
}

fn MonotonicNs() u64 {
    var timespec: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &timespec);
    return @as(u64, @intCast(timespec.sec)) * generic.NS_PER_S + @as(u64, @intCast(timespec.nsec));
}

// lfence; rdtsc; lfence -> prior insns retired before read, later insns wait for read
inline fn TscStart() u64 {
    var low: u32 = undefined;
    var high: u32 = undefined;
    asm volatile ("lfence\n\trdtsc\n\tlfence"
        : [low] "={eax}" (low),
          [high] "={edx}" (high),
        :
        : .{ .memory = true });
    return (@as(u64, high) << 32) | low;
}

// rdtscp; lfence -> rdtscp waits prior insns, lfence blocks later insns
inline fn TscEnd() u64 {
    var low: u32 = undefined;
    var high: u32 = undefined;
    asm volatile ("rdtscp\n\tlfence"
        : [low] "={eax}" (low),
          [high] "={edx}" (high),
        :
        : .{ .ecx = true, .memory = true });
    return (@as(u64, high) << 32) | low;
}

// compiler barrier: `out` escapes + memory clobber -> `Run` stores kept
inline fn ClobberMemory(out: []f32) void {
    asm volatile (""
        :
        : [out_ptr] "r" (out.ptr),
        : .{ .memory = true });
}
