#[allow(unsafe_code)]
pub fn read2(p: *const u8) -> u8 {
    unsafe { *p }
}
