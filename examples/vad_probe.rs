//! Throwaway probe: run the VAD processor on a WAV file and print the decision.
//! Usage: cargo run --example vad_probe -- <wav> [threshold]

use chezwizper::vad::{VadEngine, VadProcessor, VadSettings};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: vad_probe <wav> [threshold]");
    let threshold: f32 = args.next().map(|t| t.parse().unwrap()).unwrap_or(0.02);

    let mut reader = hound::WavReader::open(&path).unwrap();
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect(),
    };
    println!(
        "wav: {} samples @ {} Hz ({:.2}s)",
        samples.len(),
        spec.sample_rate,
        samples.len() as f32 / spec.sample_rate as f32
    );

    for engine in [VadEngine::Silero, VadEngine::Amplitude] {
        let settings = VadSettings {
            enabled: true,
            engine,
            threshold,
            ..Default::default()
        };
        let out = VadProcessor::new(settings, spec.sample_rate).process(samples.clone());
        println!(
            "{:?} (threshold {}): skipped={} kept {}/{} samples",
            engine,
            threshold,
            out.skipped,
            out.samples.len(),
            samples.len()
        );
    }
}
