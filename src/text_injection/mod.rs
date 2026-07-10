use anyhow::{Context, Result};
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};
use which::which;

use crate::bench_trace;
use crate::config::{InjectionConfig, InjectionForceMethod};

const COMMAND_TIMEOUT: Duration = Duration::from_secs(2);
const COMMAND_POLL_INTERVAL: Duration = Duration::from_millis(10);
const CLIPBOARD_SETTLE_DELAY: Duration = Duration::from_millis(500);

pub struct TextInjector {
    type_method: Option<TypeMethod>,
    settings: InjectionConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TypeMethod {
    Wtype,
    Ydotool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InjectionPlan {
    Type,
    Paste,
}

#[derive(Debug)]
struct ClipboardBackend {
    name: &'static str,
    copy_cmd: &'static str,
    copy_args: &'static [&'static str],
    read_cmd: &'static str,
    read_args: &'static [&'static str],
}

const CLIPBOARD_BACKENDS: &[ClipboardBackend] = &[
    ClipboardBackend {
        name: "wl-clipboard",
        copy_cmd: "wl-copy",
        copy_args: &[],
        read_cmd: "wl-paste",
        read_args: &["--no-newline"],
    },
    ClipboardBackend {
        name: "xclip",
        copy_cmd: "xclip",
        copy_args: &["-selection", "clipboard"],
        read_cmd: "xclip",
        read_args: &["-selection", "clipboard", "-out"],
    },
    ClipboardBackend {
        name: "xsel",
        copy_cmd: "xsel",
        copy_args: &["--clipboard", "--input"],
        read_cmd: "xsel",
        read_args: &["--clipboard", "--output"],
    },
];

fn command_output_with_timeout(command: &mut Command, timeout: Duration) -> Result<Output> {
    command.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = command.spawn().context("Failed to spawn command")?;
    wait_for_child_with_timeout(&mut child, timeout)?;

    child
        .wait_with_output()
        .context("Failed to collect command output")
}

fn wait_for_child_with_timeout(child: &mut Child, timeout: Duration) -> Result<ExitStatus> {
    let started = Instant::now();

    loop {
        if let Some(status) = child.try_wait().context("Failed to check command status")? {
            return Ok(status);
        }

        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(anyhow::anyhow!(
                "command timed out after {}ms",
                timeout.as_millis()
            ));
        }

        thread::sleep(COMMAND_POLL_INTERVAL);
    }
}

impl TextInjector {
    pub fn new(preferred: Option<&str>, settings: InjectionConfig) -> Result<Self> {
        let type_method = detect_type_method(preferred);

        match type_method {
            Some(TypeMethod::Wtype) => info!("Using wtype for direct text injection"),
            Some(TypeMethod::Ydotool) => info!("Using ydotool for direct text injection"),
            None => warn!("No direct typing tool found; injection will use clipboard paste"),
        }

        Ok(Self {
            type_method,
            settings,
        })
    }

    pub async fn inject_text(&self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        let plan = choose_injection_plan(text, &self.settings);
        bench_trace::event_with_extra("injection_method_chosen", || {
            serde_json::json!({
                "method": plan.name(),
                "text_chars": text.chars().count(),
                "has_newline": text_contains_newline(text),
                "threshold": self.settings.paste_threshold_chars,
                "forced": self.settings.force_method.map(force_method_name),
            })
        });

        bench_trace::event_with_extra("injection_begin", || {
            serde_json::json!({
                "method": plan.name(),
                "type_backend": self.type_method.map(TypeMethod::name),
                "text_chars": text.chars().count(),
            })
        });

        info!(
            "Injecting {} chars using {}",
            text.chars().count(),
            plan.name()
        );
        debug!("Text to inject: {}", text);

        let result = match plan {
            InjectionPlan::Type => self.inject_by_type_with_paste_fallback(text).await,
            InjectionPlan::Paste => self.inject_by_guarded_paste(text).await,
        };

        match &result {
            Ok(()) => bench_trace::event_with_extra("injection_end", || {
                serde_json::json!({
                    "method": plan.name(),
                    "success": true,
                    "text_chars": text.chars().count(),
                })
            }),
            Err(error) => bench_trace::event_with_extra("injection_end", || {
                serde_json::json!({
                    "method": plan.name(),
                    "success": false,
                    "text_chars": text.chars().count(),
                    "error": error.to_string(),
                })
            }),
        }

        result
    }

    async fn inject_by_type_with_paste_fallback(&self, text: &str) -> Result<()> {
        let Some(type_method) = self.type_method else {
            warn!(
                "Direct typing requested but no typing backend is available; falling back to paste"
            );
            return self.inject_by_guarded_paste(text).await;
        };

        if let Err(error) = self.inject_with_type_method(type_method, text) {
            warn!(
                "{} direct injection failed: {}; falling back to guarded paste",
                type_method.name(),
                error
            );
            return self.inject_by_guarded_paste(text).await;
        }

        Ok(())
    }

    fn inject_with_type_method(&self, method: TypeMethod, text: &str) -> Result<()> {
        match method {
            TypeMethod::Wtype => run_checked(Command::new("wtype").arg(text), "wtype"),
            TypeMethod::Ydotool => run_checked(
                Command::new("ydotool").arg("type").arg(text),
                "ydotool type",
            ),
        }
    }

    async fn inject_by_guarded_paste(&self, text: &str) -> Result<()> {
        info!("Using guarded clipboard paste for text injection");

        let previous = if self.settings.restore_clipboard {
            match self.read_clipboard().await {
                Ok(value) => Some(value),
                Err(error) => {
                    warn!(
                        "Could not snapshot clipboard before paste: {}. wl-clipboard and X11 fallbacks can only restore plain text when it is readable.",
                        error
                    );
                    None
                }
            }
        } else {
            None
        };

        self.copy_to_clipboard_with_verify(text).await?;

        if let Err(error) = self.simulate_paste().await {
            warn!(
                "Paste shortcut failed after copying transcript; leaving transcript on clipboard for manual paste: {}",
                error
            );
            return Err(error);
        }

        tokio::time::sleep(CLIPBOARD_SETTLE_DELAY).await;
        self.restore_clipboard_if_unchanged(previous, text).await;
        Ok(())
    }

    async fn restore_clipboard_if_unchanged(&self, previous: Option<String>, transcript: &str) {
        let Some(previous) = previous else {
            return;
        };

        match self.read_clipboard().await {
            Ok(current) if current == transcript => {
                if let Err(error) = self.copy_to_clipboard(&previous).await {
                    warn!("Failed to restore previous clipboard text: {}", error);
                } else {
                    debug!("Restored previous clipboard text after paste");
                }
            }
            Ok(_) => {
                warn!("Clipboard changed during injection; skipping restore to preserve user copy");
            }
            Err(error) => {
                warn!(
                    "Could not verify clipboard after paste; skipping restore: {}",
                    error
                );
            }
        }
    }

    async fn copy_to_clipboard_with_verify(&self, text: &str) -> Result<()> {
        let mut delay_ms = 50;
        let max_total_ms = 1000;
        let mut total_ms = 0;

        loop {
            self.copy_to_clipboard(text).await?;
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

            if let Ok(clipboard_content) = self.read_clipboard().await {
                if clipboard_content == text {
                    debug!("Clipboard verified after {}ms", total_ms);
                    return Ok(());
                }
            }

            if total_ms >= max_total_ms {
                warn!(
                    "Clipboard verification failed after {}ms; proceeding with paste attempt",
                    total_ms
                );
                return Ok(());
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
            total_ms += delay_ms;
            delay_ms = (delay_ms * 2).min(200);
        }
    }

    async fn read_clipboard(&self) -> Result<String> {
        for backend in CLIPBOARD_BACKENDS {
            if which(backend.read_cmd).is_err() {
                continue;
            }

            let mut command = Command::new(backend.read_cmd);
            command.args(backend.read_args);

            match command_output_with_timeout(&mut command, COMMAND_TIMEOUT) {
                Ok(output) if output.status.success() => {
                    return Ok(String::from_utf8_lossy(&output.stdout).to_string());
                }
                Ok(output) => {
                    debug!(
                        "{} read failed: {}",
                        backend.name,
                        String::from_utf8_lossy(&output.stderr)
                    );
                }
                Err(error) => debug!("{} read failed: {}", backend.name, error),
            }
        }

        Err(anyhow::anyhow!(
            "Failed to read plain-text clipboard with wl-paste, xclip, or xsel"
        ))
    }

    async fn copy_to_clipboard(&self, text: &str) -> Result<()> {
        use std::io::Write;

        bench_trace::event_with_extra("clipboard_copy_begin", || {
            serde_json::json!({
                "text_chars": text.chars().count(),
                "source": "text_injection",
            })
        });

        let result = (|| -> Result<()> {
            for backend in CLIPBOARD_BACKENDS {
                if which(backend.copy_cmd).is_err() {
                    continue;
                }

                let mut child = Command::new(backend.copy_cmd)
                    .args(backend.copy_args)
                    .stdin(Stdio::piped())
                    .spawn()
                    .with_context(|| format!("Failed to spawn {}", backend.copy_cmd))?;

                if let Some(mut stdin) = child.stdin.take() {
                    if let Err(error) = stdin.write_all(text.as_bytes()) {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(error.into());
                    }
                }

                let status = wait_for_child_with_timeout(&mut child, COMMAND_TIMEOUT)?;
                if status.success() {
                    debug!("Text copied to clipboard with {}", backend.name);
                    return Ok(());
                }
            }

            Err(anyhow::anyhow!("No clipboard copy tool available"))
        })();

        match &result {
            Ok(()) => bench_trace::event_with_extra("clipboard_copy_end", || {
                serde_json::json!({
                    "success": true,
                    "text_chars": text.chars().count(),
                    "source": "text_injection",
                })
            }),
            Err(error) => bench_trace::event_with_extra("clipboard_copy_end", || {
                serde_json::json!({
                    "success": false,
                    "text_chars": text.chars().count(),
                    "source": "text_injection",
                    "error": error.to_string(),
                })
            }),
        }

        result
    }

    async fn simulate_paste(&self) -> Result<()> {
        info!("Simulating Ctrl+V paste");

        if let Some(method) = self.type_method {
            if self.try_paste_with_type_method(method).is_ok() {
                return Ok(());
            }
        }

        for method in [TypeMethod::Ydotool, TypeMethod::Wtype] {
            if Some(method) == self.type_method || !method.command_available() {
                continue;
            }

            if self.try_paste_with_type_method(method).is_ok() {
                return Ok(());
            }
        }

        if which("xdotool").is_ok() {
            let mut command = Command::new("xdotool");
            command.args(["key", "ctrl+v"]);
            if run_checked(&mut command, "xdotool paste").is_ok() {
                return Ok(());
            }
        }

        Err(anyhow::anyhow!("No paste shortcut backend succeeded"))
    }

    fn try_paste_with_type_method(&self, method: TypeMethod) -> Result<()> {
        match method {
            TypeMethod::Wtype => {
                let mut command = Command::new("wtype");
                command.args(["-M", "ctrl", "-P", "v", "-m", "ctrl", "-p", "v"]);
                run_checked(&mut command, "wtype paste")
            }
            TypeMethod::Ydotool => {
                let mut command = Command::new("ydotool");
                command.args(["key", "29:1", "47:1", "47:0", "29:0"]);
                run_checked(&mut command, "ydotool paste")
            }
        }
    }
}

fn detect_type_method(preferred: Option<&str>) -> Option<TypeMethod> {
    match preferred {
        Some("wtype") => return available_or_warn(TypeMethod::Wtype, "per config"),
        Some("ydotool") => return available_or_warn(TypeMethod::Ydotool, "per config"),
        Some(other) => warn!(
            "Unknown input_method '{}' in config; falling back to auto-detect",
            other
        ),
        None => {}
    }

    if TypeMethod::Ydotool.command_available() {
        return Some(TypeMethod::Ydotool);
    }

    if TypeMethod::Wtype.command_available() {
        return Some(TypeMethod::Wtype);
    }

    None
}

fn available_or_warn(method: TypeMethod, source: &str) -> Option<TypeMethod> {
    if method.command_available() {
        Some(method)
    } else {
        warn!(
            "{} requested {} but command was not found; falling back to auto-detect",
            method.name(),
            source
        );
        None
    }
}

fn choose_injection_plan(text: &str, settings: &InjectionConfig) -> InjectionPlan {
    match settings.force_method {
        Some(InjectionForceMethod::Type) => InjectionPlan::Type,
        Some(InjectionForceMethod::Paste) => InjectionPlan::Paste,
        None if text_contains_newline(text) => InjectionPlan::Paste,
        None if text.chars().count() > settings.paste_threshold_chars => InjectionPlan::Paste,
        None => InjectionPlan::Type,
    }
}

fn text_contains_newline(text: &str) -> bool {
    text.contains('\n') || text.contains('\r')
}

fn force_method_name(method: InjectionForceMethod) -> &'static str {
    match method {
        InjectionForceMethod::Type => "type",
        InjectionForceMethod::Paste => "paste",
    }
}

fn run_checked(command: &mut Command, label: &str) -> Result<()> {
    let output = command_output_with_timeout(command, COMMAND_TIMEOUT)
        .with_context(|| format!("Failed to execute {label}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!("{label} failed: {stderr}"));
    }

    Ok(())
}

impl TypeMethod {
    fn name(self) -> &'static str {
        match self {
            TypeMethod::Wtype => "wtype",
            TypeMethod::Ydotool => "ydotool",
        }
    }

    fn command(self) -> &'static str {
        match self {
            TypeMethod::Wtype => "wtype",
            TypeMethod::Ydotool => "ydotool",
        }
    }

    fn command_available(self) -> bool {
        which(self.command()).is_ok()
    }
}

impl InjectionPlan {
    fn name(self) -> &'static str {
        match self {
            InjectionPlan::Type => "type",
            InjectionPlan::Paste => "paste",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tokio::sync::Mutex;

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    async fn test_env_lock() -> tokio::sync::MutexGuard<'static, ()> {
        ENV_LOCK.get_or_init(|| Mutex::new(())).lock().await
    }

    fn settings() -> InjectionConfig {
        InjectionConfig {
            paste_threshold_chars: 10,
            force_method: None,
            restore_clipboard: true,
        }
    }

    fn unique_test_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("chezwizper-injection-{name}-{nanos}"))
    }

    fn write_executable_script(path: &Path, body: &str) {
        fs::write(path, body).unwrap();
        let mut permissions = fs::metadata(path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions).unwrap();
    }

    fn prepend_path(dir: &Path) -> String {
        let current = std::env::var("PATH").unwrap_or_default();
        format!("{}:{current}", dir.display())
    }

    fn install_clipboard_stubs(dir: &Path, log: &Path, clipboard: &Path) {
        write_executable_script(
            &dir.join("wl-copy"),
            &format!(
                "#!/bin/sh\ninput=$(cat)\nprintf 'copy:%s\\n' \"$input\" >> '{}'\nprintf '%s' \"$input\" > '{}'\n",
                log.display(),
                clipboard.display()
            ),
        );
        write_executable_script(
            &dir.join("wl-paste"),
            &format!("#!/bin/sh\ncat '{}'\n", clipboard.display()),
        );
    }

    #[test]
    fn command_output_with_timeout_kills_slow_command() {
        let mut command = Command::new("sh");
        command.args(["-c", "sleep 2"]);

        let started = Instant::now();
        let err = command_output_with_timeout(&mut command, Duration::from_millis(100))
            .expect_err("slow command should time out");

        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(err.to_string().contains("timed out"));
    }

    #[test]
    fn decision_uses_type_for_short_single_line_text() {
        assert_eq!(
            choose_injection_plan("short", &settings()),
            InjectionPlan::Type
        );
    }

    #[test]
    fn decision_uses_paste_above_threshold_or_for_multiline() {
        assert_eq!(
            choose_injection_plan("this is too long", &settings()),
            InjectionPlan::Paste
        );
        assert_eq!(
            choose_injection_plan("line one\nline two", &settings()),
            InjectionPlan::Paste
        );
    }

    #[test]
    fn decision_honors_forced_method() {
        let mut config = settings();
        config.force_method = Some(InjectionForceMethod::Paste);
        assert_eq!(
            choose_injection_plan("short", &config),
            InjectionPlan::Paste
        );

        config.force_method = Some(InjectionForceMethod::Type);
        assert_eq!(
            choose_injection_plan("line one\nline two", &config),
            InjectionPlan::Type
        );
    }

    #[tokio::test]
    async fn short_type_path_never_invokes_clipboard_commands() {
        let _guard = test_env_lock().await;
        let dir = unique_test_dir("short-type");
        fs::create_dir_all(&dir).unwrap();
        let log = dir.join("log");

        write_executable_script(
            &dir.join("wtype"),
            &format!(
                "#!/bin/sh\nprintf 'type:%s\\n' \"$*\" >> '{}'\n",
                log.display()
            ),
        );
        write_executable_script(&dir.join("wl-copy"), "#!/bin/sh\nexit 9\n");
        write_executable_script(&dir.join("wl-paste"), "#!/bin/sh\nexit 9\n");

        let old_path = std::env::var("PATH").ok();
        std::env::set_var("PATH", prepend_path(&dir));

        let injector = TextInjector::new(Some("wtype"), settings()).unwrap();
        injector.inject_text("hello").await.unwrap();
        injector.inject_text("again").await.unwrap();

        if let Some(path) = old_path {
            std::env::set_var("PATH", path);
        }

        assert_eq!(fs::read_to_string(log).unwrap(), "type:hello\ntype:again\n");
    }

    #[tokio::test]
    async fn long_paste_path_copies_pastes_then_restores_clipboard() {
        let _guard = test_env_lock().await;
        let dir = unique_test_dir("long-paste");
        fs::create_dir_all(&dir).unwrap();
        let log = dir.join("log");
        let clipboard = dir.join("clipboard");
        fs::write(&clipboard, "original").unwrap();

        install_clipboard_stubs(&dir, &log, &clipboard);
        write_executable_script(
            &dir.join("wtype"),
            &format!(
                "#!/bin/sh\nif [ \"$1\" = \"-M\" ]; then printf 'paste:wtype\\n' >> '{}'; exit 0; fi\nprintf 'type:%s\\n' \"$*\" >> '{}'\n",
                log.display(),
                log.display()
            ),
        );

        let old_path = std::env::var("PATH").ok();
        std::env::set_var("PATH", prepend_path(&dir));

        let injector = TextInjector::new(Some("wtype"), settings()).unwrap();
        injector.inject_text("this is too long").await.unwrap();

        if let Some(path) = old_path {
            std::env::set_var("PATH", path);
        }

        assert_eq!(fs::read_to_string(&clipboard).unwrap(), "original");
        assert_eq!(
            fs::read_to_string(log).unwrap(),
            "copy:this is too long\npaste:wtype\ncopy:original\n"
        );
    }

    #[tokio::test]
    async fn long_paste_remains_available_to_a_delayed_clipboard_consumer() {
        let _guard = test_env_lock().await;
        let dir = unique_test_dir("delayed-paste-consumer");
        fs::create_dir_all(&dir).unwrap();
        let log = dir.join("log");
        let clipboard = dir.join("clipboard");
        let pasted = dir.join("pasted");
        fs::write(&clipboard, "original").unwrap();

        install_clipboard_stubs(&dir, &log, &clipboard);
        write_executable_script(
            &dir.join("wtype"),
            &format!(
                "#!/bin/sh\nif [ \"$1\" = \"-M\" ]; then (sleep 0.25; cat '{}' > '{}') >/dev/null 2>&1 & exit 0; fi\nexit 1\n",
                clipboard.display(),
                pasted.display()
            ),
        );

        let old_path = std::env::var("PATH").ok();
        std::env::set_var("PATH", prepend_path(&dir));

        let injector = TextInjector::new(Some("wtype"), settings()).unwrap();
        injector.inject_text("this is too long").await.unwrap();
        tokio::time::sleep(Duration::from_millis(150)).await;

        if let Some(path) = old_path {
            std::env::set_var("PATH", path);
        }

        assert_eq!(fs::read_to_string(pasted).unwrap(), "this is too long");
    }

    #[tokio::test]
    async fn long_paste_skips_restore_when_clipboard_changes_during_injection() {
        let _guard = test_env_lock().await;
        let dir = unique_test_dir("user-copy");
        fs::create_dir_all(&dir).unwrap();
        let log = dir.join("log");
        let clipboard = dir.join("clipboard");
        fs::write(&clipboard, "original").unwrap();

        install_clipboard_stubs(&dir, &log, &clipboard);
        write_executable_script(
            &dir.join("wtype"),
            &format!(
                "#!/bin/sh\nif [ \"$1\" = \"-M\" ]; then printf 'paste:wtype\\n' >> '{}'; printf 'user-copy' > '{}'; exit 0; fi\nexit 1\n",
                log.display(),
                clipboard.display()
            ),
        );

        let old_path = std::env::var("PATH").ok();
        std::env::set_var("PATH", prepend_path(&dir));

        let injector = TextInjector::new(Some("wtype"), settings()).unwrap();
        injector.inject_text("this is too long").await.unwrap();

        if let Some(path) = old_path {
            std::env::set_var("PATH", path);
        }

        assert_eq!(fs::read_to_string(&clipboard).unwrap(), "user-copy");
        assert_eq!(
            fs::read_to_string(log).unwrap(),
            "copy:this is too long\npaste:wtype\n"
        );
    }

    #[tokio::test]
    async fn typing_failure_falls_back_to_guarded_paste() {
        let _guard = test_env_lock().await;
        let dir = unique_test_dir("type-fallback");
        fs::create_dir_all(&dir).unwrap();
        let log = dir.join("log");
        let clipboard = dir.join("clipboard");
        fs::write(&clipboard, "original").unwrap();

        install_clipboard_stubs(&dir, &log, &clipboard);
        write_executable_script(
            &dir.join("wtype"),
            &format!(
                "#!/bin/sh\nif [ \"$1\" = \"-M\" ]; then printf 'paste:wtype\\n' >> '{}'; exit 0; fi\nprintf 'type-failed:%s\\n' \"$*\" >> '{}'\nexit 1\n",
                log.display(),
                log.display()
            ),
        );

        let old_path = std::env::var("PATH").ok();
        std::env::set_var("PATH", prepend_path(&dir));

        let injector = TextInjector::new(Some("wtype"), settings()).unwrap();
        injector.inject_text("short").await.unwrap();

        if let Some(path) = old_path {
            std::env::set_var("PATH", path);
        }

        assert_eq!(fs::read_to_string(&clipboard).unwrap(), "original");
        assert_eq!(
            fs::read_to_string(log).unwrap(),
            "type-failed:short\ncopy:short\npaste:wtype\ncopy:original\n"
        );
    }
}
