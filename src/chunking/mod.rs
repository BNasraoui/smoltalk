use crate::audio::{FinalRecordingSnapshot, RecordingBuffer};
use crate::bench_trace;
use crate::config::ChunkingConfig;
use crate::transcription::TranscriptionService;
use crate::vad::VadSettings;
use crate::whisper::provider::AudioFileRetention;
use anyhow::{Context, Result};
use std::future::Future;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;
use tracing::{debug, info};

const POLL_INTERVAL: Duration = Duration::from_millis(100);
static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug)]
pub struct PauseSegmenterSettings {
    pub sample_rate: u32,
    pub threshold: f32,
    pub min_speech_ms: u32,
    pub pause_ms: u32,
    pub min_chunk_ms: u32,
    pub overlap_ms: u32,
}

pub struct PauseSegmenter {
    settings: PauseSegmenterSettings,
    window: usize,
    hop: usize,
    min_speech_samples: usize,
    pause_samples: usize,
    min_chunk_samples: usize,
    overlap_samples: usize,
    scan_offset: usize,
    run_start: Option<usize>,
    run_end: usize,
    first_speech_start: Option<usize>,
    last_speech_end: Option<usize>,
    next_segment_start: usize,
    last_emitted_end: usize,
}

impl PauseSegmenter {
    pub fn new(settings: PauseSegmenterSettings) -> Self {
        let sample_rate = settings.sample_rate.max(1);
        let samples_for_ms = |ms| sample_rate.saturating_mul(ms) as usize / 1_000;
        let window = samples_for_ms(30).max(1);
        let hop = samples_for_ms(10).max(1);

        Self {
            settings,
            window,
            hop,
            min_speech_samples: samples_for_ms(settings.min_speech_ms).max(hop),
            pause_samples: samples_for_ms(settings.pause_ms),
            min_chunk_samples: samples_for_ms(settings.min_chunk_ms),
            overlap_samples: samples_for_ms(settings.overlap_ms),
            scan_offset: 0,
            run_start: None,
            run_end: 0,
            first_speech_start: None,
            last_speech_end: None,
            next_segment_start: 0,
            last_emitted_end: 0,
        }
    }

    pub fn ready_segments(&mut self, samples: &[f32]) -> Vec<Range<usize>> {
        let mut segments = Vec::new();
        while self.scan_offset.saturating_add(self.window) <= samples.len() {
            let end = self.scan_offset + self.window;
            self.inspect_window(&samples[self.scan_offset..end], end);
            self.maybe_emit(end, &mut segments);
            self.scan_offset = self.scan_offset.saturating_add(self.hop);
        }
        segments
    }

    pub fn finish_with_speech_end(
        &mut self,
        samples: &[f32],
        authoritative_speech_end: usize,
    ) -> Option<Range<usize>> {
        while self.scan_offset < samples.len() {
            let end = self
                .scan_offset
                .saturating_add(self.window)
                .min(samples.len());
            self.inspect_window(&samples[self.scan_offset..end], end);
            self.scan_offset = self.scan_offset.saturating_add(self.hop);
        }
        self.close_speech_run();

        let has_uncovered_speech =
            self.first_speech_start.is_some() || authoritative_speech_end > self.last_emitted_end;
        has_uncovered_speech.then(|| {
            let range = self.next_segment_start.min(samples.len())..samples.len();
            self.reset_segment();
            range
        })
    }

    fn inspect_window(&mut self, samples: &[f32], end: usize) {
        if window_is_speech(samples, self.settings.threshold) {
            self.run_start.get_or_insert(self.scan_offset);
            self.run_end = end;
        } else {
            self.close_speech_run();
        }
    }

    fn close_speech_run(&mut self) {
        let Some(start) = self.run_start.take() else {
            return;
        };
        if self.run_end.saturating_sub(start) < self.min_speech_samples {
            return;
        }

        self.first_speech_start.get_or_insert(start);
        self.last_speech_end = Some(self.run_end);
    }

    fn maybe_emit(&mut self, analyzed_through: usize, segments: &mut Vec<Range<usize>>) {
        if self.run_start.is_some() {
            return;
        }
        let (Some(first), Some(last)) = (self.first_speech_start, self.last_speech_end) else {
            return;
        };
        if last.saturating_sub(first) < self.min_chunk_samples
            || analyzed_through.saturating_sub(last) < self.pause_samples
        {
            return;
        }

        segments.push(self.next_segment_start..analyzed_through);
        self.next_segment_start = last.saturating_sub(self.overlap_samples);
        self.last_emitted_end = analyzed_through;
        self.reset_segment();
    }

    fn reset_segment(&mut self) {
        self.run_start = None;
        self.run_end = 0;
        self.first_speech_start = None;
        self.last_speech_end = None;
    }
}

pub fn stitch_transcripts<I, S>(chunks: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    chunks.into_iter().fold(String::new(), |combined, chunk| {
        stitch_pair(&combined, chunk.as_ref())
    })
}

fn stitch_pair(left: &str, right: &str) -> String {
    let left = left.trim();
    let right = right.trim();
    if left.is_empty() {
        return right.to_string();
    }
    if right.is_empty() {
        return left.to_string();
    }

    let left_words: Vec<_> = left.split_whitespace().collect();
    let right_words: Vec<_> = right.split_whitespace().collect();
    let overlap = (2..=left_words.len().min(right_words.len()))
        .rev()
        .find(|&count| {
            left_words[left_words.len() - count..]
                .iter()
                .zip(&right_words[..count])
                .all(|(left, right)| {
                    let left = overlap_key(left);
                    !left.is_empty() && left == overlap_key(right)
                })
        })
        .unwrap_or(0);

    let suffix = right_words[overlap..].join(" ");
    if suffix.is_empty() {
        left.to_string()
    } else {
        format!("{left} {suffix}")
    }
}

fn overlap_key(word: &str) -> String {
    word.trim_matches(|character: char| {
        matches!(
            character,
            '.' | ',' | '!' | '?' | ';' | ':' | '"' | '\'' | '(' | ')' | '[' | ']'
        )
    })
    .to_lowercase()
}

fn window_is_speech(samples: &[f32], threshold: f32) -> bool {
    if samples.is_empty() {
        return false;
    }

    let peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0_f32, f32::max);
    let rms =
        (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt();
    peak >= threshold || rms >= threshold
}

#[derive(Debug)]
enum SessionCommand {
    Finish(FinalRecordingSnapshot),
    Cancel,
}

pub struct PauseChunkingSession {
    command_tx: Option<oneshot::Sender<SessionCommand>>,
    task: tokio::task::JoinHandle<Result<Option<String>>>,
}

impl PauseChunkingSession {
    pub fn start(
        config: &ChunkingConfig,
        vad: &VadSettings,
        sample_rate: u32,
        buffer: RecordingBuffer,
        transcription_service: Arc<TranscriptionService>,
    ) -> Self {
        let session_id = NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed);
        let segmenter = PauseSegmenter::new(PauseSegmenterSettings {
            sample_rate,
            threshold: vad.threshold,
            min_speech_ms: vad.min_speech_ms,
            pause_ms: config.pause_ms,
            min_chunk_ms: config.min_chunk_ms,
            overlap_ms: config.overlap_ms.min(config.pause_ms),
        });
        let (command_tx, command_rx) = oneshot::channel();
        let task = tokio::spawn(run_session(
            segmenter,
            buffer,
            command_rx,
            POLL_INTERVAL,
            move |samples, chunk_index| {
                transcribe_chunk(
                    transcription_service.clone(),
                    samples,
                    session_id,
                    chunk_index,
                )
            },
        ));

        info!(session_id, "Started pause-triggered chunking session");
        Self {
            command_tx: Some(command_tx),
            task,
        }
    }

    pub async fn finish(mut self, snapshot: FinalRecordingSnapshot) -> Result<String> {
        if let Some(command_tx) = self.command_tx.take() {
            let _ = command_tx.send(SessionCommand::Finish(snapshot));
        }
        let text = self
            .task
            .await
            .context("pause chunking worker failed")??
            .context("pause chunking worker was cancelled")?;
        if text.is_empty() {
            return Err(anyhow::anyhow!("pause chunking produced no text"));
        }
        Ok(text)
    }

    pub async fn cancel(mut self) {
        if let Some(command_tx) = self.command_tx.take() {
            let _ = command_tx.send(SessionCommand::Cancel);
        }
        self.task.abort();
        let _ = self.task.await;
    }
}

async fn transcribe_chunk(
    transcription_service: Arc<TranscriptionService>,
    samples: Vec<f32>,
    session_id: u64,
    chunk_index: usize,
) -> Result<String> {
    let runtime = tokio::runtime::Handle::current();
    bench_trace::event_with_extra("chunk_transcription_begin", || {
        serde_json::json!({
            "session_id": session_id,
            "chunk_index": chunk_index,
            "samples": samples.len(),
        })
    });

    let result = tokio::task::spawn_blocking(move || {
        runtime.block_on(
            transcription_service.transcribe_samples(&samples, AudioFileRetention::Delete),
        )
    })
    .await
    .context("pause chunk transcription worker panicked")?;

    bench_trace::event_with_extra("chunk_transcription_end", || {
        serde_json::json!({
            "session_id": session_id,
            "chunk_index": chunk_index,
            "success": result.is_ok(),
        })
    });
    result
}

async fn run_session<F, Fut>(
    mut segmenter: PauseSegmenter,
    buffer: RecordingBuffer,
    mut command_rx: oneshot::Receiver<SessionCommand>,
    poll_interval: Duration,
    mut transcribe: F,
) -> Result<Option<String>>
where
    F: FnMut(Vec<f32>, usize) -> Fut,
    Fut: Future<Output = Result<String>>,
{
    let mut samples = Vec::new();
    let mut transcripts = Vec::new();
    let mut chunk_index = 0;
    let mut interval = tokio::time::interval(poll_interval);

    loop {
        tokio::select! {
            command = &mut command_rx => {
                let Ok(command) = command else {
                    return Ok(None);
                };
                match command {
                    SessionCommand::Cancel => return Ok(None),
                    SessionCommand::Finish(final_snapshot) => {
                        let authoritative_speech_end = final_snapshot.speech_end;
                        samples = final_snapshot.samples;
                        let ready = segmenter.ready_segments(&samples);
                        transcribe_ranges(
                            ready,
                            &samples,
                            &mut transcripts,
                            &mut chunk_index,
                            &mut transcribe,
                        ).await?;
                        if let Some(tail) = segmenter
                            .finish_with_speech_end(&samples, authoritative_speech_end)
                        {
                            transcribe_ranges(
                                vec![tail],
                                &samples,
                                &mut transcripts,
                                &mut chunk_index,
                                &mut transcribe,
                            ).await?;
                        }
                        debug!(chunks = transcripts.len(), "Pause chunking session finalized");
                        return Ok(Some(stitch_transcripts(transcripts)));
                    }
                }
            }
            _ = interval.tick() => {
                let new_samples = buffer.read_from(samples.len());
                if new_samples.is_empty() {
                    continue;
                }
                samples.extend(new_samples);
                let ready = segmenter.ready_segments(&samples);
                transcribe_ranges(
                    ready,
                    &samples,
                    &mut transcripts,
                    &mut chunk_index,
                    &mut transcribe,
                ).await?;
            }
        }
    }
}

async fn transcribe_ranges<F, Fut>(
    ranges: Vec<Range<usize>>,
    samples: &[f32],
    transcripts: &mut Vec<String>,
    chunk_index: &mut usize,
    transcribe: &mut F,
) -> Result<()>
where
    F: FnMut(Vec<f32>, usize) -> Fut,
    Fut: Future<Output = Result<String>>,
{
    for range in ranges {
        let chunk = samples[range].to_vec();
        let text = transcribe(chunk, *chunk_index).await?;
        if text.is_empty() {
            return Err(anyhow::anyhow!(
                "pause chunk {chunk_index} produced no text"
            ));
        }
        transcripts.push(text);
        *chunk_index += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::RecordingBuffer;
    use crate::whisper::provider::{AudioFileRetention, TranscriptionProvider};
    use crate::whisper::WhisperTranscriber;
    use std::path::Path;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tokio::sync::oneshot;

    #[derive(Debug, PartialEq)]
    struct ObservedChunkSamples {
        samples: Vec<f32>,
        retention: AudioFileRetention,
    }

    type SharedObservedChunkSamples = Arc<Mutex<Option<ObservedChunkSamples>>>;

    struct ChunkSampleProvider {
        observed: SharedObservedChunkSamples,
    }

    impl TranscriptionProvider for ChunkSampleProvider {
        fn name(&self) -> &'static str {
            "chunk-sample-provider"
        }

        fn is_available(&self) -> bool {
            true
        }

        fn transcribe<'a>(
            &'a self,
            _audio_path: &'a Path,
            _language: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
            panic!("chunk transcription must not use the path route")
        }

        fn transcribe_samples<'a>(
            &'a self,
            samples: &'a [f32],
            _language: &'a str,
            retention: AudioFileRetention,
        ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
            Box::pin(async move {
                *self.observed.lock().unwrap() = Some(ObservedChunkSamples {
                    samples: samples.to_vec(),
                    retention,
                });
                Ok("chunk transcript".to_string())
            })
        }
    }

    fn settings() -> PauseSegmenterSettings {
        PauseSegmenterSettings {
            sample_rate: 1_000,
            threshold: 0.02,
            min_speech_ms: 100,
            pause_ms: 600,
            min_chunk_ms: 5_000,
            overlap_ms: 300,
        }
    }

    fn audio(parts: &[(usize, f32)]) -> Vec<f32> {
        parts
            .iter()
            .flat_map(|(len, value)| std::iter::repeat_n(*value, *len))
            .collect()
    }

    #[test]
    fn emits_a_long_speech_segment_after_a_pause() {
        let samples = audio(&[(1_000, 0.0), (6_000, 0.1), (700, 0.0)]);
        let mut segmenter = PauseSegmenter::new(settings());

        let segments = segmenter.ready_segments(&samples);

        assert_eq!(segments.len(), 1);
        assert!(segments[0].start <= 1_000);
        assert!(segments[0].end >= 7_000);
        assert!(segments[0].end <= samples.len());
    }

    #[test]
    fn does_not_emit_before_the_minimum_chunk_duration() {
        let samples = audio(&[(500, 0.0), (2_000, 0.1), (1_000, 0.0)]);
        let mut segmenter = PauseSegmenter::new(settings());

        assert!(segmenter.ready_segments(&samples).is_empty());
    }

    #[test]
    fn short_pause_does_not_split_a_segment() {
        let samples = audio(&[
            (500, 0.0),
            (3_000, 0.1),
            (300, 0.0),
            (3_000, 0.1),
            (700, 0.0),
        ]);
        let mut segmenter = PauseSegmenter::new(settings());

        let segments = segmenter.ready_segments(&samples);

        assert_eq!(segments.len(), 1);
        assert!(segments[0].start <= 500);
        assert!(segments[0].end >= 6_800);
    }

    #[test]
    fn resumed_speech_prevents_a_pause_split() {
        let samples = audio(&[(500, 0.0), (6_000, 0.1), (500, 0.0), (200, 0.1)]);
        let mut segmenter = PauseSegmenter::new(settings());

        assert!(segmenter.ready_segments(&samples).is_empty());
        let tail = segmenter
            .finish_with_speech_end(&samples, samples.len())
            .expect("complete speech segment");

        assert!(tail.start <= 500);
        assert_eq!(tail.end, samples.len());
    }

    #[test]
    fn finalizes_unflushed_speech_tail() {
        let samples = audio(&[(500, 0.0), (2_000, 0.1)]);
        let mut segmenter = PauseSegmenter::new(settings());

        assert!(segmenter.ready_segments(&samples).is_empty());
        let tail = segmenter
            .finish_with_speech_end(&samples, samples.len())
            .expect("speech tail");

        assert!(tail.start <= 500);
        assert_eq!(tail.end, samples.len());
    }

    #[test]
    fn consecutive_segments_retain_only_the_configured_overlap() {
        let samples = audio(&[
            (500, 0.0),
            (6_000, 0.1),
            (700, 0.0),
            (6_000, 0.1),
            (700, 0.0),
        ]);
        let mut segmenter = PauseSegmenter::new(settings());

        let segments = segmenter.ready_segments(&samples);

        assert_eq!(segments.len(), 2);
        assert!(segments[1].start < 7_200);
        assert!(segments[0].end.saturating_sub(segments[1].start) <= 1_000);
    }

    #[test]
    fn covered_speech_does_not_create_a_silence_only_tail() {
        let samples = audio(&[(500, 0.0), (6_000, 0.1), (1_500, 0.0)]);
        let mut segmenter = PauseSegmenter::new(settings());

        assert_eq!(segmenter.ready_segments(&samples).len(), 1);

        assert!(segmenter.finish_with_speech_end(&samples, 6_500).is_none());
    }

    #[test]
    fn authoritative_vad_keeps_a_quiet_tail_after_an_earlier_chunk() {
        let samples = audio(&[(500, 0.0), (6_000, 0.1), (700, 0.0), (1_000, 0.01)]);
        let mut segmenter = PauseSegmenter::new(settings());

        assert_eq!(segmenter.ready_segments(&samples).len(), 1);
        let tail = segmenter
            .finish_with_speech_end(&samples, samples.len())
            .expect("authoritative VAD found uncovered speech");

        assert!(tail.start < 7_200);
        assert_eq!(tail.end, samples.len());
    }

    #[test]
    fn stitches_the_longest_word_overlap_without_changing_output_casing() {
        let chunks = ["Hello from the model.", "THE MODEL is already warm"];

        assert_eq!(
            stitch_transcripts(chunks),
            "Hello from the model. is already warm"
        );
    }

    #[test]
    fn stitching_preserves_repetition_that_is_not_at_a_chunk_boundary() {
        let chunks = ["very very useful", "different result"];

        assert_eq!(
            stitch_transcripts(chunks),
            "very very useful different result"
        );
    }

    #[test]
    fn stitching_preserves_a_repeated_single_word_at_a_boundary() {
        let chunks = ["go", "go now"];

        assert_eq!(stitch_transcripts(chunks), "go go now");
    }

    #[tokio::test]
    async fn chunk_transcription_uses_the_sample_pipeline_with_delete_retention() {
        let observed = Arc::new(Mutex::new(None));
        let whisper = WhisperTranscriber::from_provider(
            Box::new(ChunkSampleProvider {
                observed: observed.clone(),
            }),
            "en",
        );
        let service = Arc::new(TranscriptionService::new(whisper).unwrap());
        let samples = vec![0.125, -0.25, 0.5];
        let session_id = u64::MAX;
        let chunk_index = usize::MAX;

        let result = transcribe_chunk(service, samples.clone(), session_id, chunk_index).await;

        assert_eq!(result.unwrap(), "chunk transcript");
        assert_eq!(
            *observed.lock().unwrap(),
            Some(ObservedChunkSamples {
                samples,
                retention: AudioFileRetention::Delete,
            })
        );
    }

    #[tokio::test]
    async fn worker_transcribes_a_pause_chunk_before_finalizing_the_tail() {
        let first = audio(&[(500, 0.0), (6_000, 0.1), (700, 0.0)]);
        let mut complete = first.clone();
        complete.extend(audio(&[(6_000, 0.1)]));
        let mut expected_segmenter = PauseSegmenter::new(settings());
        let first_range = expected_segmenter.ready_segments(&first).remove(0);
        let expected_first = first[first_range].to_vec();
        assert!(expected_segmenter.ready_segments(&complete).is_empty());
        let tail_range = expected_segmenter
            .finish_with_speech_end(&complete, complete.len())
            .unwrap();
        let expected_tail = complete[tail_range].to_vec();
        let buffer = RecordingBuffer::new(Arc::new(Mutex::new(first)));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let transcribe_calls = calls.clone();
        let (command_tx, command_rx) = oneshot::channel();

        let worker = tokio::spawn(run_session(
            PauseSegmenter::new(settings()),
            buffer,
            command_rx,
            Duration::from_millis(1),
            move |samples, index| {
                transcribe_calls.lock().unwrap().push(samples);
                async move {
                    Ok(match index {
                        0 => "the first segment ends here".to_string(),
                        _ => "ends here and continues".to_string(),
                    })
                }
            },
        ));

        tokio::time::timeout(Duration::from_millis(100), async {
            loop {
                if calls.lock().unwrap().len() == 1 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("pause chunk should transcribe while recording");

        let speech_end = complete.len();
        command_tx
            .send(SessionCommand::Finish(FinalRecordingSnapshot {
                samples: complete,
                speech_end,
            }))
            .expect("worker is running");
        let text = worker.await.unwrap().unwrap().expect("finished transcript");

        assert_eq!(*calls.lock().unwrap(), vec![expected_first, expected_tail]);
        assert_eq!(text, "the first segment ends here and continues");
    }

    #[tokio::test]
    async fn worker_rejects_a_partial_transcript_when_any_chunk_is_empty() {
        let first = audio(&[(500, 0.0), (6_000, 0.1), (700, 0.0)]);
        let mut complete = first.clone();
        complete.extend(audio(&[(6_000, 0.1)]));
        let buffer = RecordingBuffer::new(Arc::new(Mutex::new(first)));
        let (command_tx, command_rx) = oneshot::channel();

        let worker = tokio::spawn(run_session(
            PauseSegmenter::new(settings()),
            buffer,
            command_rx,
            Duration::from_millis(1),
            move |_samples, index| async move {
                Ok(if index == 0 {
                    "first segment".to_string()
                } else {
                    String::new()
                })
            },
        ));

        tokio::task::yield_now().await;
        let speech_end = complete.len();
        command_tx
            .send(SessionCommand::Finish(FinalRecordingSnapshot {
                samples: complete,
                speech_end,
            }))
            .expect("worker is running");

        assert!(worker.await.unwrap().is_err());
    }

    #[tokio::test]
    async fn chunk_callback_errors_fail_the_session() {
        let samples = vec![0.1, 0.2, 0.3];
        let mut transcripts = Vec::new();
        let mut chunk_index = 0;
        let mut transcribe = |_samples, _index| async {
            Err::<String, _>(anyhow::anyhow!("injected chunk callback failure"))
        };

        let result = transcribe_ranges(
            std::iter::once(0..samples.len()).collect(),
            &samples,
            &mut transcripts,
            &mut chunk_index,
            &mut transcribe,
        )
        .await;

        assert!(result.is_err());
        assert!(transcripts.is_empty());
        assert_eq!(chunk_index, 0);
    }

    #[tokio::test]
    async fn cancelling_does_not_wait_for_an_in_flight_chunk() {
        let samples = audio(&[(500, 0.0), (6_000, 0.1), (700, 0.0)]);
        let buffer = RecordingBuffer::new(Arc::new(Mutex::new(samples)));
        let (command_tx, command_rx) = oneshot::channel();
        let started = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let transcribe_started = started.clone();
        let task = tokio::spawn(run_session(
            PauseSegmenter::new(settings()),
            buffer,
            command_rx,
            Duration::from_millis(1),
            move |_samples, _index| {
                transcribe_started.store(true, std::sync::atomic::Ordering::Relaxed);
                std::future::pending::<Result<String>>()
            },
        ));
        let session = PauseChunkingSession {
            command_tx: Some(command_tx),
            task,
        };

        tokio::time::timeout(Duration::from_millis(100), async {
            while !started.load(std::sync::atomic::Ordering::Relaxed) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("chunk transcription should start");

        tokio::time::timeout(Duration::from_millis(100), session.cancel())
            .await
            .expect("cancellation should not wait for inference");
    }
}
