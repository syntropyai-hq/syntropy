// SPDX-License-Identifier: MIT OR Apache-2.0

// The Metal side of `tests/moe_assign.rs` (COMP-E1-T11): compiles the generated
// `shaders/moe_assign.metal` with the Metal compiler of the default device and runs its
// `syn_moe_assign` kernel on the batches the test writes to stdin, one dispatch per case.
//
// Protocol, every integer little-endian:
// - On start it writes `u32 0` and exits 0 when there is no Metal device, or `u32 1`, then
//   `u32 len` and the device name's `len` UTF-8 bytes. Any other failure goes to stderr with
//   exit code 2.
// - It then reads batches: `u32 n_cases`, `u32 n_entries`, `u32 n_words` (`n_cases` 0 ends the
//   run); `n_cases` case records of 32 bytes (`u32 route_off, n_route, words_off, n_experts,
//   b_copy, copy_slots`, `u64 expert_bytes`); `n_entries` route slots of 8 bytes; `n_words`
//   bitmap words. It answers each batch with `n_entries` `u32` codes.
//
// The output buffer starts as 0xA5A5A5A5 in every word, which no case can produce (a copy-slot
// index is below 300 and the other codes are `SYN_MOE_HIT` and `SYN_MOE_CPU`), so an entry the
// kernel did not write shows as a difference. Each case is dispatched as whole threadgroups of
// 32 threads, last case first, into an output that the next case's entries follow, so a thread
// at or past `n_route` that wrote would overwrite codes already written and show as a difference.

import Foundation
import Metal

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(("moe_assign harness: " + message + "\n").utf8))
    exit(2)
}

func readExact(_ n: Int) -> [UInt8] {
    var buf = [UInt8](repeating: 0, count: n)
    var got = 0
    while got < n {
        let r = buf.withUnsafeMutableBytes { read(0, $0.baseAddress! + got, n - got) }
        if r <= 0 { fail("stdin ended after \(got) of \(n) bytes") }
        got += r
    }
    return buf
}

func writeAll(_ bytes: UnsafeRawBufferPointer) {
    var done = 0
    while done < bytes.count {
        let r = write(1, bytes.baseAddress! + done, bytes.count - done)
        if r <= 0 { fail("stdout closed") }
        done += r
    }
}

func writeWords(_ words: [UInt32]) {
    words.map { $0.littleEndian }.withUnsafeBytes { writeAll($0) }
}

func u32(_ b: [UInt8], _ at: Int) -> UInt32 {
    UInt32(b[at]) | UInt32(b[at + 1]) << 8 | UInt32(b[at + 2]) << 16 | UInt32(b[at + 3]) << 24
}

func u64(_ b: [UInt8], _ at: Int) -> UInt64 {
    UInt64(u32(b, at)) | UInt64(u32(b, at + 4)) << 32
}

let args = CommandLine.arguments
if args.count != 2 { fail("usage: harness <moe_assign.metal>") }
guard let device = MTLCreateSystemDefaultDevice() else {
    writeWords([0])
    exit(0)
}
let source: String
do {
    source = try String(contentsOfFile: args[1], encoding: .utf8)
} catch {
    fail("cannot read \(args[1]): \(error)")
}
let pipeline: MTLComputePipelineState
do {
    let library = try device.makeLibrary(source: source, options: nil)
    guard let function = library.makeFunction(name: "syn_moe_assign") else {
        fail("the library has no syn_moe_assign")
    }
    pipeline = try device.makeComputePipelineState(function: function)
} catch {
    fail("the Metal compiler rejected the source: \(error)")
}
guard let queue = device.makeCommandQueue() else { fail("no command queue") }
let name = Array(device.name.utf8)
writeWords([1, UInt32(name.count)])
name.withUnsafeBytes { writeAll($0) }

let group = 32
if pipeline.maxTotalThreadsPerThreadgroup < group { fail("threadgroups below \(group) threads") }

func buffer(_ bytes: [UInt8], fill: UInt8) -> MTLBuffer {
    // One spare word, so an offset at the end of the data is still inside the buffer.
    var data = bytes
    data.append(contentsOf: [UInt8](repeating: fill, count: 8))
    guard let b = device.makeBuffer(bytes: data, length: data.count, options: .storageModeShared)
    else { fail("cannot allocate \(data.count) bytes") }
    return b
}

while true {
    let head = readExact(12)
    let nCases = Int(u32(head, 0))
    let nEntries = Int(u32(head, 4))
    let nWords = Int(u32(head, 8))
    if nCases == 0 { break }
    let cases = readExact(nCases * 32)
    let routes = buffer(readExact(nEntries * 8), fill: 0)
    let words = buffer(readExact(nWords * 4), fill: 0)
    let out = buffer([UInt8](repeating: 0xA5, count: nEntries * 4), fill: 0xA5)
    guard let cmd = queue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else {
        fail("cannot encode")
    }
    enc.setComputePipelineState(pipeline)
    // Last case first: a stray write past a case lands on entries already written.
    for k in (0..<nCases).reversed() {
        let at = k * 32
        let routeOff = Int(u32(cases, at))
        var nRoute = u32(cases, at + 4)
        let wordsOff = Int(u32(cases, at + 8))
        var nExperts = u32(cases, at + 12)
        var bCopy = u32(cases, at + 16)
        var copySlots = u32(cases, at + 20)
        var expertBytes = u64(cases, at + 24)
        if nRoute == 0 { continue }
        enc.setBuffer(routes, offset: routeOff * 8, index: 0)
        enc.setBytes(&nRoute, length: 4, index: 1)
        enc.setBuffer(words, offset: wordsOff * 4, index: 2)
        enc.setBytes(&nExperts, length: 4, index: 3)
        enc.setBytes(&bCopy, length: 4, index: 4)
        enc.setBytes(&expertBytes, length: 8, index: 5)
        enc.setBytes(&copySlots, length: 4, index: 6)
        enc.setBuffer(out, offset: routeOff * 4, index: 7)
        let groups = (Int(nRoute) + group - 1) / group
        enc.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: group, height: 1, depth: 1))
    }
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    if let error = cmd.error { fail("the command buffer failed: \(error)") }
    writeAll(UnsafeRawBufferPointer(start: out.contents(), count: nEntries * 4))
}
exit(0)
