# bench/lab

`rigs.toml` is the lab inventory (RFC-0061 §6.3). `capture-inventory.sh` produces a rig's record
for it. Installing and cabling a rig is a Maintainer action (ADR-0610 decision 7); the script is
what the maintainer runs afterwards.

## Capturing R-ADA and its spare GPU (QUAL-E15-T02)

On the rig, as root, with the pinned OS image booted:

```sh
sudo bench/lab/capture-inventory.sh --id R-ADA \
    --os-image <image ID> --room-sensor <sensor ID> \
    --memory-profile <BIOS memory profile> --power-limits "PL1=<w>W,PL2=<w>W" \
    --resizable-bar <true|false> \
    --attachment nvme0=cpu \
    --expect-compute-cap 8.9 > r-ada.toml
```

- The script is read-only and prints nothing if any value is missing. It never fills a default.
- The memory profile, power limits, resizable-BAR state and each NVMe slot's attachment come
  from the firmware setup screen and the board manual; the operating system does not report them.
  Pass one `--attachment <name>=<cpu|chipset>` per NVMe controller.
- NVMe and NIC links are the negotiated ones (sysfs `current_link_speed` and `_width`): a Gen4
  drive in a Gen3 slot is recorded `gen3x4`. A GPU's current link falls at idle, so its link is
  the highest the GPU and its slot support together; check it against the board manual. Some
  devices also drop their link at idle, so before committing the records confirm the NVMe and NIC
  links under load (for example `lspci -vv` `LnkSta` during a transfer).

The spare GPU has no record format of its own: RFC-0061 §6.3 defines one record per rig. Swap the
spare into the host, boot the same image, and run the script again with `--id R-ADA-SPARE`. The ID is a judgement call, since no source names one, so a maintainer
confirms it.

Then:

1. Append both records to `rigs.toml`, and update its header comment's "No rig is installed yet".
2. Names must match the spec records. The script records the CPU as `lscpu` prints it and the
   GPU as `nvidia-smi` prints it (for example `Intel(R) Core(TM) ... ` against a spec record
   named `Intel Core ...`), and the spec database (`bench/specs/`) matches models exactly. If
   `rigs check` or the specs check reports a model-name mismatch, add the printed name to that spec
   record's `models`; do not edit the captured value, which the lab agent compares against the
   live machine.
3. Check that R-ADA lists 2 × 64 GiB at 6000 MT/s (2 channels), the 24 GB GPU, the Gen4 NVMe on the
   CPU, the 10 GbE and 2.5 GbE NICs and the RoCEv2-capable NIC (RFC-0061 §6.2). A part the script
   did not find is a part to fix on the rig, not to add by hand.
4. `cargo run -p syntropy-bench -- rigs check` and `cargo xtask ci lint`.
5. Open a PR; `bench/lab/` is code-owned.

`crates/bench/tests/capture.rs` tests the script against stubbed tools with made-up fixture
values. They are not measurements of any lab rig.
