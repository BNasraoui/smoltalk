//! Print whisper.cpp system info for the linked whisper-rs build.

fn main() {
    let info = whisper_rs::print_system_info();
    println!("{info}");
}
