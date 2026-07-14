use crate::bench_trace;
use crate::cancellation::CancellationToken;
use crate::config::{Config, WaybarConfig};
use crate::transcription::TranscriptionService;
use anyhow::Result;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::sync::mpsc;
use tower::ServiceBuilder;
use tracing::{error, info};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RecordingCommand {
    Start(ApiCommandSource, CancellationToken),
    Stop(ApiCommandSource, CancellationToken),
    Cancel(CancellationToken),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApiCommandSource {
    Start,
    Stop,
    Toggle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AppStatus {
    Idle,
    Recording,
    Processing,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum AppLifecycle {
    Idle,
    Recording(CancellationToken),
    Processing(CancellationToken),
}

impl AppLifecycle {
    pub(crate) fn status(&self) -> AppStatus {
        match self {
            Self::Idle => AppStatus::Idle,
            Self::Recording(_) => AppStatus::Recording,
            Self::Processing(_) => AppStatus::Processing,
        }
    }
}

pub(crate) type SharedAppLifecycle = Arc<Mutex<AppLifecycle>>;

pub(crate) fn lock_lifecycle(lifecycle: &SharedAppLifecycle) -> MutexGuard<'_, AppLifecycle> {
    lifecycle
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub(crate) fn release_recording_reservation(
    lifecycle: &SharedAppLifecycle,
    cancellation: &CancellationToken,
) {
    let mut lifecycle = lock_lifecycle(lifecycle);
    let owns_current = matches!(
        &*lifecycle,
        AppLifecycle::Recording(current) if current.same_session(cancellation)
    );

    if owns_current {
        *lifecycle = AppLifecycle::Idle;
        bench_trace::event("state_idle_set");
    }
}

pub(crate) struct ProcessingResetGuard {
    lifecycle: SharedAppLifecycle,
    cancellation: CancellationToken,
}

impl ProcessingResetGuard {
    pub(crate) fn new(lifecycle: SharedAppLifecycle, cancellation: CancellationToken) -> Self {
        Self {
            lifecycle,
            cancellation,
        }
    }
}

impl Drop for ProcessingResetGuard {
    fn drop(&mut self) {
        let reset = {
            let mut lifecycle = lock_lifecycle(&self.lifecycle);
            let owns_current = matches!(
                &*lifecycle,
                AppLifecycle::Processing(current)
                    if current.same_session(&self.cancellation)
            );

            if owns_current {
                *lifecycle = AppLifecycle::Idle;
                true
            } else {
                false
            }
        };
        if reset {
            bench_trace::event("state_idle_set");
        }
    }
}

impl AppStatus {
    fn as_str(self) -> &'static str {
        match self {
            AppStatus::Idle => "idle",
            AppStatus::Recording => "recording",
            AppStatus::Processing => "processing",
        }
    }
}

#[derive(Clone)]
pub struct AppState {
    tx: mpsc::Sender<RecordingCommand>,
    lifecycle: SharedAppLifecycle,
    waybar_config: WaybarConfig,
    transcription: Arc<TranscriptionService>,
}

pub(crate) struct ApiServer {
    port: u16,
    state: AppState,
}

impl ApiServer {
    pub(crate) fn new(
        tx: mpsc::Sender<RecordingCommand>,
        lifecycle: SharedAppLifecycle,
        config: &Config,
        transcription: Arc<TranscriptionService>,
    ) -> Self {
        Self {
            port: config.api.port,
            state: AppState {
                tx,
                lifecycle,
                waybar_config: config.ui.waybar.clone(),
                transcription,
            },
        }
    }

    pub(crate) async fn start(self) -> Result<()> {
        let app = Router::new()
            .route("/", get(status))
            .route("/start", post(start_recording))
            .route("/stop", post(stop_recording))
            .route("/cancel", post(cancel_recording))
            .route("/toggle", post(toggle_recording))
            .route("/status", get(recording_status))
            .route("/model/status", get(model_status))
            .route("/model/unload", post(unload_model))
            .route("/model/reload", post(reload_model))
            .layer(ServiceBuilder::new())
            .with_state(self.state);

        let listener = tokio::net::TcpListener::bind(&format!("127.0.0.1:{}", self.port)).await?;

        info!("API server listening on http://127.0.0.1:{}", self.port);
        info!("Endpoints:");
        info!("  POST /start  - Start recording");
        info!("  POST /stop   - Stop recording");
        info!("  POST /cancel - Cancel the current utterance");
        info!("  POST /toggle - Toggle recording");
        info!("  GET /status  - Get recording and model status");

        axum::serve(listener, app).await?;

        Ok(())
    }
}

async fn status() -> Json<Value> {
    Json(json!({
        "service": "chezwizper",
        "version": "0.1.0",
        "status": "running"
    }))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RecordingRequest {
    Start,
    Stop,
    Toggle,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct ReservationOutcome {
    success: bool,
    message: &'static str,
    status: AppStatus,
}

pub(crate) async fn reserve_recording_command(
    tx: &mpsc::Sender<RecordingCommand>,
    lifecycle: &SharedAppLifecycle,
    request: RecordingRequest,
) -> Result<ReservationOutcome, StatusCode> {
    let mut lifecycle = lock_lifecycle(lifecycle);
    let command = match (request, lifecycle.clone()) {
        (RecordingRequest::Start | RecordingRequest::Toggle, AppLifecycle::Idle) => {
            let cancellation = CancellationToken::new();
            *lifecycle = AppLifecycle::Recording(cancellation.clone());
            bench_trace::event("state_recording_set");
            Some((
                RecordingCommand::Start(request.command_source(), cancellation),
                AppLifecycle::Idle,
            ))
        }
        (
            RecordingRequest::Stop | RecordingRequest::Toggle,
            AppLifecycle::Recording(cancellation),
        ) => {
            *lifecycle = AppLifecycle::Processing(cancellation.clone());
            bench_trace::event("state_processing_set");
            Some((
                RecordingCommand::Stop(request.command_source(), cancellation.clone()),
                AppLifecycle::Recording(cancellation),
            ))
        }
        _ => None,
    };

    if let Some((command, rollback_state)) = command {
        if let Err(e) = tx.try_send(command) {
            error!("Failed to send recording command: {}", e);
            *lifecycle = rollback_state;
            return Err(StatusCode::SERVICE_UNAVAILABLE);
        }
    }

    let status = lifecycle.status();
    let success = !(request == RecordingRequest::Toggle && status == AppStatus::Processing);
    let message = match request {
        RecordingRequest::Start => "Recording started",
        RecordingRequest::Stop => "Recording stopped",
        RecordingRequest::Toggle if success => "Recording toggled",
        RecordingRequest::Toggle => "Previous recording is still processing",
    };

    Ok(ReservationOutcome {
        success,
        message,
        status,
    })
}

pub(crate) async fn reserve_cancel_command(
    tx: &mpsc::Sender<RecordingCommand>,
    lifecycle: &SharedAppLifecycle,
) -> Result<ReservationOutcome, StatusCode> {
    let mut lifecycle = lock_lifecycle(lifecycle);
    let mut success = true;
    let mut message = "Recording cancelled";

    match lifecycle.clone() {
        AppLifecycle::Recording(cancellation) => {
            if let Err(e) = tx.try_send(RecordingCommand::Cancel(cancellation.clone())) {
                error!("Failed to send cancel command: {}", e);
                return Err(StatusCode::SERVICE_UNAVAILABLE);
            }
            cancellation.cancel();
            *lifecycle = AppLifecycle::Idle;
        }
        AppLifecycle::Processing(cancellation) => {
            if cancellation.cancel() {
                *lifecycle = AppLifecycle::Idle;
            } else {
                success = false;
                message = "Text insertion has already started";
            }
        }
        AppLifecycle::Idle => {}
    }

    let status = lifecycle.status();
    Ok(ReservationOutcome {
        success,
        message,
        status,
    })
}

impl RecordingRequest {
    fn command_source(self) -> ApiCommandSource {
        match self {
            RecordingRequest::Start => ApiCommandSource::Start,
            RecordingRequest::Stop => ApiCommandSource::Stop,
            RecordingRequest::Toggle => ApiCommandSource::Toggle,
        }
    }
}

async fn start_recording(State(state): State<AppState>) -> Result<Json<Value>, StatusCode> {
    let status = lock_lifecycle(&state.lifecycle).status();
    bench_trace::event_with_extra("api_start_received", || {
        json!({
            "status": status.as_str(),
        })
    });

    let outcome =
        reserve_recording_command(&state.tx, &state.lifecycle, RecordingRequest::Start).await?;

    info!("Start recording command received via API");
    Ok(Json(json!({
        "success": outcome.success,
        "message": outcome.message,
        "status": outcome.status.as_str(),
    })))
}

async fn stop_recording(State(state): State<AppState>) -> Result<Json<Value>, StatusCode> {
    let status = lock_lifecycle(&state.lifecycle).status();
    bench_trace::event_with_extra("api_stop_received", || {
        json!({
            "status": status.as_str(),
        })
    });

    let outcome =
        reserve_recording_command(&state.tx, &state.lifecycle, RecordingRequest::Stop).await?;

    info!("Stop recording command received via API");
    Ok(Json(json!({
        "success": outcome.success,
        "message": outcome.message,
        "status": outcome.status.as_str(),
    })))
}

async fn cancel_recording(State(state): State<AppState>) -> Result<Json<Value>, StatusCode> {
    let status = lock_lifecycle(&state.lifecycle).status();
    bench_trace::event_with_extra("api_cancel_received", || {
        json!({
            "status": status.as_str(),
        })
    });

    let outcome = reserve_cancel_command(&state.tx, &state.lifecycle).await?;

    info!("Cancel recording command received via API");
    Ok(Json(json!({
        "success": outcome.success,
        "message": outcome.message,
        "status": outcome.status.as_str(),
    })))
}

async fn toggle_recording(State(state): State<AppState>) -> Result<Json<Value>, StatusCode> {
    let status = lock_lifecycle(&state.lifecycle).status();
    bench_trace::event_with_extra("api_toggle_received", || {
        json!({
            "status": status.as_str(),
        })
    });

    let outcome =
        reserve_recording_command(&state.tx, &state.lifecycle, RecordingRequest::Toggle).await?;

    info!("Toggle recording command received via API");
    if !outcome.success {
        return Ok(Json(json!({
            "success": false,
            "message": outcome.message,
            "status": outcome.status.as_str(),
        })));
    }

    Ok(Json(json!({
        "success": outcome.success,
        "message": outcome.message
    })))
}

async fn recording_status(
    Query(params): Query<HashMap<String, String>>,
    State(state): State<AppState>,
) -> Json<Value> {
    let status = lock_lifecycle(&state.lifecycle).status();

    // Check if waybar style is requested
    if params.get("style") == Some(&"waybar".to_string()) {
        return Json(generate_waybar_response(status, &state.waybar_config));
    }

    // Default JSON response
    Json(json!({
        "recording": status == AppStatus::Recording,
        "status": status.as_str(),
        "model": state.transcription.model_status()
    }))
}

async fn model_status(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "model": state.transcription.model_status()
    }))
}

async fn unload_model(State(state): State<AppState>) -> Result<Json<Value>, StatusCode> {
    state
        .transcription
        .unload_model()
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    Ok(Json(json!({
        "success": true,
        "model": state.transcription.model_status()
    })))
}

async fn reload_model(State(state): State<AppState>) -> Result<Json<Value>, StatusCode> {
    state
        .transcription
        .reload_model()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(json!({
        "success": true,
        "model": state.transcription.model_status()
    })))
}

fn generate_waybar_response(status: AppStatus, config: &WaybarConfig) -> Value {
    json!({
        "text": match status {
            AppStatus::Idle => &config.idle_text,
            AppStatus::Recording => &config.recording_text,
            AppStatus::Processing => &config.processing_text,
        },
        "class": match status {
            AppStatus::Idle => "chezwizper-idle",
            AppStatus::Recording => "chezwizper-recording",
            AppStatus::Processing => "chezwizper-processing",
        },
        "tooltip": match status {
            AppStatus::Idle => &config.idle_tooltip,
            AppStatus::Recording => &config.recording_tooltip,
            AppStatus::Processing => &config.processing_tooltip,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cancellation::CancellationToken;
    use tokio::sync::mpsc::error::TryRecvError;

    fn lifecycle_token(lifecycle: &AppLifecycle) -> Option<&CancellationToken> {
        match lifecycle {
            AppLifecycle::Idle => None,
            AppLifecycle::Recording(cancellation) | AppLifecycle::Processing(cancellation) => {
                Some(cancellation)
            }
        }
    }

    #[test]
    fn recording_lifecycle_owns_its_utterance_token() {
        let cancellation = CancellationToken::new();
        let lifecycle = AppLifecycle::Recording(cancellation.clone());

        assert_eq!(lifecycle.status(), AppStatus::Recording);
        assert!(lifecycle_token(&lifecycle).is_some_and(|token| token.same_session(&cancellation)));
    }

    #[test]
    fn waybar_response_reports_processing_state() {
        let config = WaybarConfig::default();

        let response = generate_waybar_response(AppStatus::Processing, &config);

        assert_eq!(response["text"], config.processing_text);
        assert_eq!(response["class"], "chezwizper-processing");
        assert_eq!(response["tooltip"], config.processing_tooltip);
    }

    #[test]
    fn processing_reset_guard_restores_idle() {
        let cancellation = CancellationToken::new();
        let lifecycle = Arc::new(Mutex::new(AppLifecycle::Processing(cancellation.clone())));

        {
            let _guard = ProcessingResetGuard::new(lifecycle.clone(), cancellation);
        }

        assert_eq!(*lock_lifecycle(&lifecycle), AppLifecycle::Idle);
    }

    #[test]
    fn processing_reset_guard_does_not_overwrite_a_new_recording() {
        let old = CancellationToken::new();
        let lifecycle = Arc::new(Mutex::new(AppLifecycle::Processing(old.clone())));
        let replacement = CancellationToken::new();

        {
            let _guard = ProcessingResetGuard::new(lifecycle.clone(), old);
            *lock_lifecycle(&lifecycle) = AppLifecycle::Recording(replacement.clone());
        }

        assert!(matches!(
            &*lock_lifecycle(&lifecycle),
            AppLifecycle::Recording(current) if current.same_session(&replacement)
        ));
    }

    #[test]
    fn processing_reset_guard_does_not_overwrite_new_processing_session() {
        let old = CancellationToken::new();
        let lifecycle = Arc::new(Mutex::new(AppLifecycle::Processing(old.clone())));
        let replacement = CancellationToken::new();

        {
            let _guard = ProcessingResetGuard::new(lifecycle.clone(), old);
            *lock_lifecycle(&lifecycle) = AppLifecycle::Processing(replacement.clone());
        }

        assert!(matches!(
            &*lock_lifecycle(&lifecycle),
            AppLifecycle::Processing(current) if current.same_session(&replacement)
        ));
    }

    #[test]
    fn app_lifecycle_lock_recovers_after_poisoning() {
        let lifecycle = Arc::new(Mutex::new(AppLifecycle::Idle));
        let panic_lifecycle = lifecycle.clone();

        let _ = std::panic::catch_unwind(move || {
            let _state = panic_lifecycle
                .lock()
                .expect("lifecycle should start unpoisoned");
            panic!("poison lifecycle mutex");
        });

        *lock_lifecycle(&lifecycle) = AppLifecycle::Idle;

        assert_eq!(*lock_lifecycle(&lifecycle), AppLifecycle::Idle);
    }

    #[tokio::test]
    async fn start_from_idle_reserves_recording_and_enqueues_start() {
        let (tx, mut rx) = mpsc::channel(1);
        let lifecycle = Arc::new(Mutex::new(AppLifecycle::Idle));

        let outcome = reserve_recording_command(&tx, &lifecycle, RecordingRequest::Start)
            .await
            .expect("start should enqueue");

        assert_eq!(outcome.status, AppStatus::Recording);
        let state = lifecycle.lock().unwrap().clone();
        let cancellation = lifecycle_token(&state).unwrap().clone();
        assert_eq!(state.status(), AppStatus::Recording);
        match rx.recv().await {
            Some(RecordingCommand::Start(ApiCommandSource::Start, queued)) => {
                assert!(queued.same_session(&cancellation));
            }
            other => panic!("expected start command, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn start_from_recording_is_idempotent_without_enqueue() {
        let (tx, mut rx) = mpsc::channel(1);
        let lifecycle = Arc::new(Mutex::new(
            AppLifecycle::Recording(CancellationToken::new()),
        ));

        let outcome = reserve_recording_command(&tx, &lifecycle, RecordingRequest::Start)
            .await
            .expect("duplicate start should be accepted");

        assert_eq!(outcome.status, AppStatus::Recording);
        assert_eq!(lock_lifecycle(&lifecycle).status(), AppStatus::Recording);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    }

    #[tokio::test]
    async fn start_from_processing_is_idempotent_without_enqueue() {
        let (tx, mut rx) = mpsc::channel(1);
        let lifecycle = Arc::new(Mutex::new(AppLifecycle::Processing(
            CancellationToken::new(),
        )));

        let outcome = reserve_recording_command(&tx, &lifecycle, RecordingRequest::Start)
            .await
            .expect("start during processing should be accepted as a no-op");

        assert_eq!(outcome.status, AppStatus::Processing);
        assert_eq!(lock_lifecycle(&lifecycle).status(), AppStatus::Processing);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    }

    #[tokio::test]
    async fn stop_from_recording_reserves_processing_and_enqueues_stop() {
        let (tx, mut rx) = mpsc::channel(1);
        let cancellation = CancellationToken::new();
        let lifecycle = Arc::new(Mutex::new(AppLifecycle::Recording(cancellation.clone())));

        let outcome = reserve_recording_command(&tx, &lifecycle, RecordingRequest::Stop)
            .await
            .expect("stop should enqueue");

        assert_eq!(outcome.status, AppStatus::Processing);
        assert_eq!(lock_lifecycle(&lifecycle).status(), AppStatus::Processing);
        assert_eq!(
            rx.recv().await,
            Some(RecordingCommand::Stop(ApiCommandSource::Stop, cancellation))
        );
    }

    #[tokio::test]
    async fn stop_from_idle_is_idempotent_without_enqueue() {
        let (tx, mut rx) = mpsc::channel(1);
        let lifecycle = Arc::new(Mutex::new(AppLifecycle::Idle));

        let outcome = reserve_recording_command(&tx, &lifecycle, RecordingRequest::Stop)
            .await
            .expect("stop from idle should be accepted as a no-op");

        assert_eq!(outcome.status, AppStatus::Idle);
        assert_eq!(*lock_lifecycle(&lifecycle), AppLifecycle::Idle);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    }

    #[tokio::test]
    async fn stop_from_processing_is_idempotent_without_enqueue() {
        let (tx, mut rx) = mpsc::channel(1);
        let lifecycle = Arc::new(Mutex::new(AppLifecycle::Processing(
            CancellationToken::new(),
        )));

        let outcome = reserve_recording_command(&tx, &lifecycle, RecordingRequest::Stop)
            .await
            .expect("stop during processing should be accepted as a no-op");

        assert_eq!(outcome.status, AppStatus::Processing);
        assert_eq!(lock_lifecycle(&lifecycle).status(), AppStatus::Processing);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    }

    #[tokio::test]
    async fn toggle_from_idle_uses_start_reservation() {
        let (tx, mut rx) = mpsc::channel(1);
        let lifecycle = Arc::new(Mutex::new(AppLifecycle::Idle));

        let outcome = reserve_recording_command(&tx, &lifecycle, RecordingRequest::Toggle)
            .await
            .expect("toggle from idle should enqueue start");

        assert_eq!(outcome.status, AppStatus::Recording);
        let cancellation = lifecycle_token(&lock_lifecycle(&lifecycle))
            .unwrap()
            .clone();
        assert_eq!(
            rx.recv().await,
            Some(RecordingCommand::Start(
                ApiCommandSource::Toggle,
                cancellation
            ))
        );
    }

    #[tokio::test]
    async fn toggle_from_recording_uses_stop_reservation() {
        let (tx, mut rx) = mpsc::channel(1);
        let cancellation = CancellationToken::new();
        let lifecycle = Arc::new(Mutex::new(AppLifecycle::Recording(cancellation.clone())));

        let outcome = reserve_recording_command(&tx, &lifecycle, RecordingRequest::Toggle)
            .await
            .expect("toggle from recording should enqueue stop");

        assert_eq!(outcome.status, AppStatus::Processing);
        assert_eq!(lock_lifecycle(&lifecycle).status(), AppStatus::Processing);
        assert_eq!(
            rx.recv().await,
            Some(RecordingCommand::Stop(
                ApiCommandSource::Toggle,
                cancellation
            ))
        );
    }

    #[tokio::test]
    async fn toggle_from_processing_preserves_existing_no_enqueue_response() {
        let (tx, mut rx) = mpsc::channel(1);
        let lifecycle = Arc::new(Mutex::new(AppLifecycle::Processing(
            CancellationToken::new(),
        )));

        let outcome = reserve_recording_command(&tx, &lifecycle, RecordingRequest::Toggle)
            .await
            .expect("toggle during processing should return the existing API response");

        assert!(!outcome.success);
        assert_eq!(outcome.status, AppStatus::Processing);
        assert_eq!(lock_lifecycle(&lifecycle).status(), AppStatus::Processing);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    }

    #[tokio::test]
    async fn fast_start_then_stop_preserves_queue_order() {
        let (tx, mut rx) = mpsc::channel(2);
        let lifecycle = Arc::new(Mutex::new(AppLifecycle::Idle));

        reserve_recording_command(&tx, &lifecycle, RecordingRequest::Start)
            .await
            .expect("start should enqueue");
        reserve_recording_command(&tx, &lifecycle, RecordingRequest::Stop)
            .await
            .expect("stop should enqueue before start is consumed");

        assert_eq!(lock_lifecycle(&lifecycle).status(), AppStatus::Processing);
        let cancellation = lifecycle_token(&lock_lifecycle(&lifecycle))
            .unwrap()
            .clone();
        assert_eq!(
            rx.recv().await,
            Some(RecordingCommand::Start(
                ApiCommandSource::Start,
                cancellation.clone()
            ))
        );
        assert_eq!(
            rx.recv().await,
            Some(RecordingCommand::Stop(ApiCommandSource::Stop, cancellation))
        );
    }

    #[tokio::test]
    async fn channel_full_on_start_rolls_status_back_to_idle() {
        let (tx, mut rx) = mpsc::channel(1);
        let queued = CancellationToken::new();
        tx.try_send(RecordingCommand::Stop(
            ApiCommandSource::Stop,
            queued.clone(),
        ))
        .expect("pre-fill command queue");
        let lifecycle = Arc::new(Mutex::new(AppLifecycle::Idle));

        let result = reserve_recording_command(&tx, &lifecycle, RecordingRequest::Start).await;

        assert_eq!(result, Err(StatusCode::SERVICE_UNAVAILABLE));
        assert_eq!(*lock_lifecycle(&lifecycle), AppLifecycle::Idle);
        assert_eq!(
            rx.recv().await,
            Some(RecordingCommand::Stop(ApiCommandSource::Stop, queued))
        );
    }

    #[tokio::test]
    async fn channel_full_on_stop_rolls_status_back_to_recording() {
        let (tx, mut rx) = mpsc::channel(1);
        let queued = CancellationToken::new();
        tx.try_send(RecordingCommand::Start(
            ApiCommandSource::Start,
            queued.clone(),
        ))
        .expect("pre-fill command queue");
        let cancellation = CancellationToken::new();
        let lifecycle = Arc::new(Mutex::new(AppLifecycle::Recording(cancellation)));

        let result = reserve_recording_command(&tx, &lifecycle, RecordingRequest::Stop).await;

        assert_eq!(result, Err(StatusCode::SERVICE_UNAVAILABLE));
        assert_eq!(lock_lifecycle(&lifecycle).status(), AppStatus::Recording);
        assert_eq!(
            rx.recv().await,
            Some(RecordingCommand::Start(ApiCommandSource::Start, queued))
        );
    }

    #[tokio::test]
    async fn cancel_from_recording_invalidates_utterance_and_enqueues_teardown() {
        let (tx, mut rx) = mpsc::channel(1);
        let cancellation = CancellationToken::new();
        let lifecycle = Arc::new(Mutex::new(AppLifecycle::Recording(cancellation.clone())));

        let outcome = reserve_cancel_command(&tx, &lifecycle)
            .await
            .expect("cancel should enqueue");

        assert!(outcome.success);
        assert_eq!(outcome.status, AppStatus::Idle);
        assert_eq!(*lock_lifecycle(&lifecycle), AppLifecycle::Idle);
        assert!(cancellation.is_cancelled());
        match rx.recv().await {
            Some(RecordingCommand::Cancel(queued)) => {
                assert!(queued.same_session(&cancellation));
            }
            other => panic!("expected cancel command, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn cancel_from_processing_invalidates_utterance_without_queueing_work() {
        let (tx, mut rx) = mpsc::channel(1);
        let cancellation = CancellationToken::new();
        let lifecycle = Arc::new(Mutex::new(AppLifecycle::Processing(cancellation.clone())));

        let outcome = reserve_cancel_command(&tx, &lifecycle)
            .await
            .expect("cancel should be accepted");

        assert_eq!(outcome.status, AppStatus::Idle);
        assert_eq!(*lock_lifecycle(&lifecycle), AppLifecycle::Idle);
        assert!(cancellation.is_cancelled());
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    }

    #[tokio::test]
    async fn cancel_after_commit_reports_too_late() {
        let (tx, mut rx) = mpsc::channel(1);
        let cancellation = CancellationToken::new();
        assert!(cancellation.try_commit());
        let lifecycle = Arc::new(Mutex::new(AppLifecycle::Processing(cancellation)));

        let outcome = reserve_cancel_command(&tx, &lifecycle)
            .await
            .expect("late cancellation should return a response");

        assert!(!outcome.success);
        assert_eq!(outcome.message, "Text insertion has already started");
        assert_eq!(outcome.status, AppStatus::Processing);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    }

    #[test]
    fn failed_old_start_does_not_release_new_recording_reservation() {
        let old = CancellationToken::new();
        let replacement = CancellationToken::new();
        let lifecycle = Arc::new(Mutex::new(AppLifecycle::Recording(replacement.clone())));

        release_recording_reservation(&lifecycle, &old);

        assert!(matches!(
            &*lock_lifecycle(&lifecycle),
            AppLifecycle::Recording(current) if current.same_session(&replacement)
        ));
    }

    #[tokio::test]
    async fn processing_cancellation_allows_immediate_replacement_start() {
        let (tx, mut rx) = mpsc::channel(3);
        let lifecycle = Arc::new(Mutex::new(AppLifecycle::Idle));

        reserve_recording_command(&tx, &lifecycle, RecordingRequest::Start)
            .await
            .unwrap();
        let cancelled = lifecycle_token(&lock_lifecycle(&lifecycle))
            .unwrap()
            .clone();
        reserve_recording_command(&tx, &lifecycle, RecordingRequest::Stop)
            .await
            .unwrap();
        reserve_cancel_command(&tx, &lifecycle).await.unwrap();
        reserve_recording_command(&tx, &lifecycle, RecordingRequest::Start)
            .await
            .unwrap();

        let replacement = lifecycle_token(&lock_lifecycle(&lifecycle))
            .unwrap()
            .clone();
        assert_eq!(lock_lifecycle(&lifecycle).status(), AppStatus::Recording);
        assert!(cancelled.is_cancelled());
        assert!(!replacement.is_cancelled());
        assert!(!replacement.same_session(&cancelled));
        assert!(
            matches!(rx.recv().await, Some(RecordingCommand::Start(_, token)) if token.same_session(&cancelled))
        );
        assert!(
            matches!(rx.recv().await, Some(RecordingCommand::Stop(_, token)) if token.same_session(&cancelled))
        );
        assert!(
            matches!(rx.recv().await, Some(RecordingCommand::Start(_, token)) if token.same_session(&replacement))
        );
    }

    #[tokio::test]
    async fn cancel_from_idle_is_an_idempotent_no_op() {
        let (tx, mut rx) = mpsc::channel(1);
        let lifecycle = Arc::new(Mutex::new(AppLifecycle::Idle));

        let outcome = reserve_cancel_command(&tx, &lifecycle).await.unwrap();

        assert!(outcome.success);
        assert_eq!(outcome.status, AppStatus::Idle);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    }
}
