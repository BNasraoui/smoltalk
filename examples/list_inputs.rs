//! List cpal input devices as the daemon sees them.

use cpal::traits::{DeviceTrait, HostTrait};

fn main() {
    let host = cpal::default_host();
    if let Some(d) = host.default_input_device() {
        println!("default input: {}", d.name().unwrap_or_default());
    }
    match host.input_devices() {
        Ok(devices) => {
            for d in devices {
                println!("input: {}", d.name().unwrap_or_default());
            }
        }
        Err(e) => println!("enumeration failed: {e}"),
    }
}
