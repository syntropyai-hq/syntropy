// SPDX-License-Identifier: MIT OR Apache-2.0

//! A bin target whose root is outside `src/`.

#![deny(unsafe_code)]

mod helper;

fn main() {
    let byte = 7u8;
    println!("{}", helper::read(&byte));
}
