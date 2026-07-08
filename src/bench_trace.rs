use serde::Serialize;
use serde_json::Value;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const TRACE_ENV: &str = "CHEZWIZPER_BENCH_TRACE";
const SCHEMA: &str = "chezwizper-bench-event-v1";

static GLOBAL_SINK: OnceLock<TraceSink> = OnceLock::new();
static PROCESS_STARTED_AT: OnceLock<Instant> = OnceLock::new();

#[derive(Debug)]
pub struct TraceSink {
    inner: Option<Mutex<BufWriter<File>>>,
    run_id: Option<String>,
    trial_id: Option<String>,
    phrase_id: Option<String>,
}

#[derive(Serialize)]
struct TraceEvent<'a> {
    schema: &'static str,
    event: &'a str,
    monotonic_ns: u128,
    wall_time_unix_ms: u128,
    run_id: Option<&'a str>,
    trial_id: Option<&'a str>,
    phrase_id: Option<&'a str>,
    pid: u32,
    thread_id: String,
    extra: Value,
}

impl TraceSink {
    pub fn from_env() -> Self {
        Self::from_env_var(TRACE_ENV)
    }

    fn from_env_var(var_name: &str) -> Self {
        let Some(path) = std::env::var_os(var_name) else {
            return Self::disabled();
        };

        match Self::new(Path::new(&path)) {
            Ok(sink) => sink,
            Err(err) => {
                eprintln!(
                    "failed to initialize benchmark trace sink at {}: {err}",
                    path.to_string_lossy()
                );
                Self::disabled()
            }
        }
    }

    fn new(path: &Path) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new().create(true).append(true).open(path)?;

        Ok(Self {
            inner: Some(Mutex::new(BufWriter::new(file))),
            run_id: std::env::var("CHEZWIZPER_BENCH_RUN_ID").ok(),
            trial_id: std::env::var("CHEZWIZPER_BENCH_TRIAL_ID").ok(),
            phrase_id: std::env::var("CHEZWIZPER_BENCH_PHRASE_ID").ok(),
        })
    }

    fn disabled() -> Self {
        Self {
            inner: None,
            run_id: None,
            trial_id: None,
            phrase_id: None,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    pub fn event(&self, event: &'static str) {
        self.event_with_extra(event, Value::Object(Default::default()));
    }

    pub fn event_with_extra(&self, event: &'static str, extra: Value) {
        let Some(inner) = &self.inner else {
            return;
        };

        let started_at = *PROCESS_STARTED_AT.get_or_init(Instant::now);
        let monotonic_ns = started_at.elapsed().as_nanos();
        let wall_time_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis())
            .unwrap_or_default();

        let row = TraceEvent {
            schema: SCHEMA,
            event,
            monotonic_ns,
            wall_time_unix_ms,
            run_id: self.run_id.as_deref(),
            trial_id: self.trial_id.as_deref(),
            phrase_id: self.phrase_id.as_deref(),
            pid: std::process::id(),
            thread_id: format!("{:?}", std::thread::current().id()),
            extra,
        };

        let Ok(line) = serde_json::to_string(&row) else {
            return;
        };

        if let Ok(mut writer) = inner.lock() {
            let _ = writeln!(writer, "{line}");
            let _ = writer.flush();
        }
    }
}

pub fn event(event: &'static str) {
    GLOBAL_SINK.get_or_init(TraceSink::from_env).event(event);
}

pub fn event_with_extra<F>(event: &'static str, extra: F)
where
    F: FnOnce() -> Value,
{
    let sink = GLOBAL_SINK.get_or_init(TraceSink::from_env);
    if sink.is_enabled() {
        sink.event_with_extra(event, extra());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_path(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("chezwizper-{name}-{nanos}.jsonl"))
    }

    #[test]
    fn disabled_when_env_var_is_unset() {
        // This var name is never set anywhere, so no env mutation is needed.
        let sink = TraceSink::from_env_var("CHEZWIZPER_TEST_TRACE_NEVER_SET");

        assert!(!sink.is_enabled());
    }

    #[test]
    fn serializes_trace_event_as_jsonl() {
        let path = unique_path("trace-event");

        let sink = TraceSink::new(&path).unwrap();
        sink.event_with_extra("unit_event", json!({"answer": 42}));
        drop(sink);

        let contents = std::fs::read_to_string(&path).unwrap();
        let value: Value = serde_json::from_str(contents.trim()).unwrap();

        assert_eq!(value["schema"], SCHEMA);
        assert_eq!(value["event"], "unit_event");
        assert_eq!(value["extra"]["answer"], 42);
        assert!(value["monotonic_ns"].as_u64().is_some());
        assert!(value["pid"].as_u64().is_some());

        let _ = std::fs::remove_file(path);
    }
}
