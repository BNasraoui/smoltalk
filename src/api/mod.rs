use crate::bench_trace;
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
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tower::ServiceBuilder;
use tracing::{error, info};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ApiCommand {
    StartRecording(ApiCommandSource),
    StopRecording(ApiCommandSource),
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
    tx: mpsc::Sender<ApiCommand>,
    status: Arc<Mutex<AppStatus>>,
    waybar_config: WaybarConfig,
    transcription: Arc<TranscriptionService>,
}

pub struct ApiServer {
    port: u16,
    state: AppState,
}

impl ApiServer {
    pub fn new(
        tx: mpsc::Sender<ApiCommand>,
        status: Arc<Mutex<AppStatus>>,
        config: &Config,
        transcription: Arc<TranscriptionService>,
    ) -> Self {
        Self {
            port: config.api.port,
            state: AppState {
                tx,
                status,
                waybar_config: config.ui.waybar.clone(),
                transcription,
            },
        }
    }

    pub async fn start(self) -> Result<()> {
        let app = Router::new()
            .route("/", get(status))
            .route("/start", post(start_recording))
            .route("/stop", post(stop_recording))
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
enum RecordingRequest {
    Start,
    Stop,
    Toggle,
}

#[derive(Debug, Eq, PartialEq)]
struct ReservationOutcome {
    success: bool,
    message: &'static str,
    status: AppStatus,
}

async fn reserve_recording_command(
    tx: &mpsc::Sender<ApiCommand>,
    status: &Arc<Mutex<AppStatus>>,
    request: RecordingRequest,
) -> Result<ReservationOutcome, StatusCode> {
    let mut current_status = status.lock().await;
    let command = match (request, *current_status) {
        (RecordingRequest::Start | RecordingRequest::Toggle, AppStatus::Idle) => {
            *current_status = AppStatus::Recording;
            bench_trace::event("state_recording_set");
            Some((
                ApiCommand::StartRecording(request.command_source()),
                AppStatus::Idle,
            ))
        }
        (RecordingRequest::Stop | RecordingRequest::Toggle, AppStatus::Recording) => {
            *current_status = AppStatus::Processing;
            bench_trace::event("state_processing_set");
            Some((
                ApiCommand::StopRecording(request.command_source()),
                AppStatus::Recording,
            ))
        }
        _ => None,
    };

    if let Some((command, rollback_status)) = command {
        if let Err(e) = tx.try_send(command) {
            error!("Failed to send recording command: {}", e);
            *current_status = rollback_status;
            return Err(StatusCode::SERVICE_UNAVAILABLE);
        }
    }

    let success =
        !(request == RecordingRequest::Toggle && *current_status == AppStatus::Processing);
    let message = match request {
        RecordingRequest::Start => "Recording started",
        RecordingRequest::Stop => "Recording stopped",
        RecordingRequest::Toggle if success => "Recording toggled",
        RecordingRequest::Toggle => "Previous recording is still processing",
    };

    Ok(ReservationOutcome {
        success,
        message,
        status: *current_status,
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
    let status = *state.status.lock().await;
    bench_trace::event_with_extra("api_start_received", || {
        json!({
            "status": status.as_str(),
        })
    });

    let outcome =
        reserve_recording_command(&state.tx, &state.status, RecordingRequest::Start).await?;

    info!("Start recording command received via API");
    Ok(Json(json!({
        "success": outcome.success,
        "message": outcome.message,
        "status": outcome.status.as_str(),
    })))
}

async fn stop_recording(State(state): State<AppState>) -> Result<Json<Value>, StatusCode> {
    let status = *state.status.lock().await;
    bench_trace::event_with_extra("api_stop_received", || {
        json!({
            "status": status.as_str(),
        })
    });

    let outcome =
        reserve_recording_command(&state.tx, &state.status, RecordingRequest::Stop).await?;

    info!("Stop recording command received via API");
    Ok(Json(json!({
        "success": outcome.success,
        "message": outcome.message,
        "status": outcome.status.as_str(),
    })))
}

async fn toggle_recording(State(state): State<AppState>) -> Result<Json<Value>, StatusCode> {
    let status = *state.status.lock().await;
    bench_trace::event_with_extra("api_toggle_received", || {
        json!({
            "status": status.as_str(),
        })
    });

    let outcome =
        reserve_recording_command(&state.tx, &state.status, RecordingRequest::Toggle).await?;

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
    let status = *state.status.lock().await;

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
    use tokio::sync::mpsc::error::TryRecvError;

    #[test]
    fn waybar_response_reports_processing_state() {
        let config = WaybarConfig::default();

        let response = generate_waybar_response(AppStatus::Processing, &config);

        assert_eq!(response["text"], config.processing_text);
        assert_eq!(response["class"], "chezwizper-processing");
        assert_eq!(response["tooltip"], config.processing_tooltip);
    }

    #[tokio::test]
    async fn start_from_idle_reserves_recording_and_enqueues_start() {
        let (tx, mut rx) = mpsc::channel(1);
        let status = Arc::new(Mutex::new(AppStatus::Idle));

        let outcome = reserve_recording_command(&tx, &status, RecordingRequest::Start)
            .await
            .expect("start should enqueue");

        assert_eq!(outcome.status, AppStatus::Recording);
        assert_eq!(*status.lock().await, AppStatus::Recording);
        assert_eq!(
            rx.recv().await,
            Some(ApiCommand::StartRecording(ApiCommandSource::Start))
        );
    }

    #[tokio::test]
    async fn start_from_recording_is_idempotent_without_enqueue() {
        let (tx, mut rx) = mpsc::channel(1);
        let status = Arc::new(Mutex::new(AppStatus::Recording));

        let outcome = reserve_recording_command(&tx, &status, RecordingRequest::Start)
            .await
            .expect("duplicate start should be accepted");

        assert_eq!(outcome.status, AppStatus::Recording);
        assert_eq!(*status.lock().await, AppStatus::Recording);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    }

    #[tokio::test]
    async fn start_from_processing_is_idempotent_without_enqueue() {
        let (tx, mut rx) = mpsc::channel(1);
        let status = Arc::new(Mutex::new(AppStatus::Processing));

        let outcome = reserve_recording_command(&tx, &status, RecordingRequest::Start)
            .await
            .expect("start during processing should be accepted as a no-op");

        assert_eq!(outcome.status, AppStatus::Processing);
        assert_eq!(*status.lock().await, AppStatus::Processing);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    }

    #[tokio::test]
    async fn stop_from_recording_reserves_processing_and_enqueues_stop() {
        let (tx, mut rx) = mpsc::channel(1);
        let status = Arc::new(Mutex::new(AppStatus::Recording));

        let outcome = reserve_recording_command(&tx, &status, RecordingRequest::Stop)
            .await
            .expect("stop should enqueue");

        assert_eq!(outcome.status, AppStatus::Processing);
        assert_eq!(*status.lock().await, AppStatus::Processing);
        assert_eq!(
            rx.recv().await,
            Some(ApiCommand::StopRecording(ApiCommandSource::Stop))
        );
    }

    #[tokio::test]
    async fn stop_from_idle_is_idempotent_without_enqueue() {
        let (tx, mut rx) = mpsc::channel(1);
        let status = Arc::new(Mutex::new(AppStatus::Idle));

        let outcome = reserve_recording_command(&tx, &status, RecordingRequest::Stop)
            .await
            .expect("stop from idle should be accepted as a no-op");

        assert_eq!(outcome.status, AppStatus::Idle);
        assert_eq!(*status.lock().await, AppStatus::Idle);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    }

    #[tokio::test]
    async fn stop_from_processing_is_idempotent_without_enqueue() {
        let (tx, mut rx) = mpsc::channel(1);
        let status = Arc::new(Mutex::new(AppStatus::Processing));

        let outcome = reserve_recording_command(&tx, &status, RecordingRequest::Stop)
            .await
            .expect("stop during processing should be accepted as a no-op");

        assert_eq!(outcome.status, AppStatus::Processing);
        assert_eq!(*status.lock().await, AppStatus::Processing);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    }

    #[tokio::test]
    async fn toggle_from_idle_uses_start_reservation() {
        let (tx, mut rx) = mpsc::channel(1);
        let status = Arc::new(Mutex::new(AppStatus::Idle));

        let outcome = reserve_recording_command(&tx, &status, RecordingRequest::Toggle)
            .await
            .expect("toggle from idle should enqueue start");

        assert_eq!(outcome.status, AppStatus::Recording);
        assert_eq!(*status.lock().await, AppStatus::Recording);
        assert_eq!(
            rx.recv().await,
            Some(ApiCommand::StartRecording(ApiCommandSource::Toggle))
        );
    }

    #[tokio::test]
    async fn toggle_from_recording_uses_stop_reservation() {
        let (tx, mut rx) = mpsc::channel(1);
        let status = Arc::new(Mutex::new(AppStatus::Recording));

        let outcome = reserve_recording_command(&tx, &status, RecordingRequest::Toggle)
            .await
            .expect("toggle from recording should enqueue stop");

        assert_eq!(outcome.status, AppStatus::Processing);
        assert_eq!(*status.lock().await, AppStatus::Processing);
        assert_eq!(
            rx.recv().await,
            Some(ApiCommand::StopRecording(ApiCommandSource::Toggle))
        );
    }

    #[tokio::test]
    async fn toggle_from_processing_preserves_existing_no_enqueue_response() {
        let (tx, mut rx) = mpsc::channel(1);
        let status = Arc::new(Mutex::new(AppStatus::Processing));

        let outcome = reserve_recording_command(&tx, &status, RecordingRequest::Toggle)
            .await
            .expect("toggle during processing should return the existing API response");

        assert!(!outcome.success);
        assert_eq!(outcome.status, AppStatus::Processing);
        assert_eq!(*status.lock().await, AppStatus::Processing);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    }

    #[tokio::test]
    async fn fast_start_then_stop_preserves_queue_order() {
        let (tx, mut rx) = mpsc::channel(2);
        let status = Arc::new(Mutex::new(AppStatus::Idle));

        reserve_recording_command(&tx, &status, RecordingRequest::Start)
            .await
            .expect("start should enqueue");
        reserve_recording_command(&tx, &status, RecordingRequest::Stop)
            .await
            .expect("stop should enqueue before start is consumed");

        assert_eq!(*status.lock().await, AppStatus::Processing);
        assert_eq!(
            rx.recv().await,
            Some(ApiCommand::StartRecording(ApiCommandSource::Start))
        );
        assert_eq!(
            rx.recv().await,
            Some(ApiCommand::StopRecording(ApiCommandSource::Stop))
        );
    }

    #[tokio::test]
    async fn channel_full_on_start_rolls_status_back_to_idle() {
        let (tx, mut rx) = mpsc::channel(1);
        tx.try_send(ApiCommand::StopRecording(ApiCommandSource::Stop))
            .expect("pre-fill command queue");
        let status = Arc::new(Mutex::new(AppStatus::Idle));

        let result = reserve_recording_command(&tx, &status, RecordingRequest::Start).await;

        assert_eq!(result, Err(StatusCode::SERVICE_UNAVAILABLE));
        assert_eq!(*status.lock().await, AppStatus::Idle);
        assert_eq!(
            rx.recv().await,
            Some(ApiCommand::StopRecording(ApiCommandSource::Stop))
        );
    }

    #[tokio::test]
    async fn channel_full_on_stop_rolls_status_back_to_recording() {
        let (tx, mut rx) = mpsc::channel(1);
        tx.try_send(ApiCommand::StartRecording(ApiCommandSource::Start))
            .expect("pre-fill command queue");
        let status = Arc::new(Mutex::new(AppStatus::Recording));

        let result = reserve_recording_command(&tx, &status, RecordingRequest::Stop).await;

        assert_eq!(result, Err(StatusCode::SERVICE_UNAVAILABLE));
        assert_eq!(*status.lock().await, AppStatus::Recording);
        assert_eq!(
            rx.recv().await,
            Some(ApiCommand::StartRecording(ApiCommandSource::Start))
        );
    }
}
