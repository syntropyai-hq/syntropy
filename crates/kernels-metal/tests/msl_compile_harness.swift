// SPDX-License-Identifier: MIT OR Apache-2.0

// Compiles one MSL source with the Metal compiler of the default device, for the tests that
// check a generated layout's `static_assert`s (TIER-E6-T20).
//
// Usage: `msl_compile_harness <file.metal>`. Exit codes: 0 when it compiled; 77 when the machine
// has no Metal device; 1 when the compiler rejected it, with its log on stderr; 2 for any other
// failure.

import Foundation
import Metal

let args = CommandLine.arguments
guard args.count == 2 else {
    FileHandle.standardError.write(Data("usage: msl_compile_harness <file.metal>\n".utf8))
    exit(2)
}
guard let source = try? String(contentsOfFile: args[1], encoding: .utf8) else {
    FileHandle.standardError.write(Data("cannot read \(args[1])\n".utf8))
    exit(2)
}
guard let device = MTLCreateSystemDefaultDevice() else {
    exit(77)
}
do {
    _ = try device.makeLibrary(source: source, options: nil)
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
