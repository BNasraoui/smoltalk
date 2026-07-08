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

#[derive(Clone)]
pub enum ApiCommand {
    ToggleRecording,
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

async fn toggle_recording(State(state): State<AppState>) -> Result<Json<Value>, StatusCode> {
    let status = *state.status.lock().await;
    bench_trace::event_with_extra("api_toggle_received", || {
        json!({
            "status": status.as_str(),
        })
    });

    if status == AppStatus::Processing {
        return Ok(Json(json!({
            "success": false,
            "message": "Previous recording is still processing",
            "status": status.as_str()
        })));
    }

    match state.tx.try_send(ApiCommand::ToggleRecording) {
        Ok(_) => {
            info!("Toggle recording command received via API");
            Ok(Json(json!({
                "success": true,
                "message": "Recording toggled"
            })))
        }
        Err(e) => {
            error!("Failed to send toggle command: {}", e);
            Err(StatusCode::SERVICE_UNAVAILABLE)
        }
    }
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

    #[test]
    fn waybar_response_reports_processing_state() {
        let config = WaybarConfig::default();

        let response = generate_waybar_response(AppStatus::Processing, &config);

        assert_eq!(response["text"], config.processing_text);
        assert_eq!(response["class"], "chezwizper-processing");
        assert_eq!(response["tooltip"], config.processing_tooltip);
    }
}
