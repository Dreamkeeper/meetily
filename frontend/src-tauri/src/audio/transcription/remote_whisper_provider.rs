// audio/transcription/remote_whisper_provider.rs
//
// Remote Whisper provider: sends each speech segment to a Whisper server on
// another machine (e.g. a desktop GPU reached through an SSH tunnel) and falls
// back to the local Parakeet engine when the server is unreachable.
//
// Server contract: POST {base_url}/v1/live/transcribe?language=<auto|auto-translate|code>
// with the body as raw little-endian f32 samples (16 kHz mono); the response is
// JSON with a "text" field. GET {base_url}/v1/live/status returns {"ready": bool}; while the
// server is still loading its model, segments wait (queue up) instead of falling back, so the
// live transcript just starts later and then catches up.

use super::provider::{TranscriptionError, TranscriptionProvider, TranscriptResult};
use async_trait::async_trait;
use log::{info, warn};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const DEFAULT_REMOTE_WHISPER_URL: &str = "http://127.0.0.1:18765";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);
/// After this many consecutive failures, use the fallback and only retry the
/// server every RETRY_AFTER, so an offline server doesn't add a timeout to
/// every segment.
const FAILURES_BEFORE_FALLBACK: u32 = 2;
const RETRY_AFTER: Duration = Duration::from_secs(30);
/// How long a segment may wait for the server to finish loading its model (a cold load takes ~60 s).
const READY_WAIT: Duration = Duration::from_secs(180);
const READY_POLL: Duration = Duration::from_secs(2);

pub struct RemoteWhisperProvider {
    client: reqwest::Client,
    base_url: String,
    fallback: Option<Arc<crate::parakeet_engine::ParakeetEngine>>,
    consecutive_failures: AtomicU32,
    last_attempt: Mutex<Option<Instant>>,
    server_ready: AtomicBool,
}

impl RemoteWhisperProvider {
    pub fn new(base_url: String, fallback: Option<Arc<crate::parakeet_engine::ParakeetEngine>>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .no_proxy() // the server is local or tunneled; never route it through a system proxy
            .build()
            .unwrap_or_default();
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            fallback,
            consecutive_failures: AtomicU32::new(0),
            last_attempt: Mutex::new(None),
            server_ready: AtomicBool::new(false),
        }
    }

    fn should_try_remote(&self) -> bool {
        if self.fallback.is_none() || self.consecutive_failures.load(Ordering::SeqCst) < FAILURES_BEFORE_FALLBACK {
            return true;
        }
        let last = *self.last_attempt.lock().unwrap();
        last.map_or(true, |t| t.elapsed() >= RETRY_AFTER)
    }

    /// Wait until the server reports its model loaded. Errors only when the server is unreachable
    /// or not ready within READY_WAIT; "still loading" is waited out.
    async fn wait_until_ready(&self) -> Result<(), String> {
        if self.server_ready.load(Ordering::SeqCst) {
            return Ok(());
        }
        let started = Instant::now();
        let mut warmup_sent = false;
        loop {
            let status: serde_json::Value = self
                .client
                .get(format!("{}/v1/live/status", self.base_url))
                .timeout(Duration::from_secs(5))
                .send()
                .await
                .map_err(|e| e.to_string())?
                .json()
                .await
                .map_err(|e| e.to_string())?;
            if status.get("ready").and_then(|v| v.as_bool()).unwrap_or(false) {
                if started.elapsed() > READY_POLL {
                    info!("[REMOTE_WHISPER] Server model ready after {:.0}s — sending queued segments", started.elapsed().as_secs_f32());
                }
                self.server_ready.store(true, Ordering::SeqCst);
                return Ok(());
            }
            let loading = status.get("loading").and_then(|v| v.as_bool()).unwrap_or(false);
            if !loading && !warmup_sent {
                warmup_sent = true;
                let url = self.base_url.clone();
                tokio::spawn(async move {
                    let _ = RemoteWhisperProvider::warmup(&url).await;
                });
            }
            if started.elapsed() >= READY_WAIT {
                return Err(format!("server model not ready after {}s", READY_WAIT.as_secs()));
            }
            tokio::time::sleep(READY_POLL).await;
        }
    }

    async fn remote_transcribe(&self, audio: &[f32], language: &str) -> Result<String, String> {
        let mut body = Vec::with_capacity(audio.len() * 4);
        for sample in audio {
            body.extend_from_slice(&sample.to_le_bytes());
        }
        let response = self
            .client
            .post(format!("{}/v1/live/transcribe", self.base_url))
            .query(&[("language", language)])
            .body(body)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !response.status().is_success() {
            return Err(format!("HTTP {}", response.status()));
        }
        let json: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
        Ok(json.get("text").and_then(|t| t.as_str()).unwrap_or("").trim().to_string())
    }

    /// Quick reachability check (GET /health), used when a recording starts.
    pub async fn health(base_url: &str) -> Result<(), String> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(3))
            .no_proxy()
            .build()
            .map_err(|e| e.to_string())?;
        let response = client
            .get(format!("{}/health", base_url.trim_end_matches('/')))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if response.status().is_success() {
            Ok(())
        } else {
            Err(format!("HTTP {}", response.status()))
        }
    }

    /// Ask the server to load its model before the first segment arrives.
    pub async fn warmup(base_url: &str) -> Result<(), String> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .no_proxy()
            .build()
            .map_err(|e| e.to_string())?;
        let response = client
            .post(format!("{}/v1/live/warmup", base_url.trim_end_matches('/')))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if response.status().is_success() {
            Ok(())
        } else {
            Err(format!("HTTP {}", response.status()))
        }
    }
}

#[async_trait]
impl TranscriptionProvider for RemoteWhisperProvider {
    async fn transcribe(
        &self,
        audio: Vec<f32>,
        language: Option<String>,
    ) -> std::result::Result<TranscriptResult, TranscriptionError> {
        if self.should_try_remote() {
            *self.last_attempt.lock().unwrap() = Some(Instant::now());
            let lang = language.as_deref().unwrap_or("auto");
            let result = match self.wait_until_ready().await {
                Ok(()) => self.remote_transcribe(&audio, lang).await,
                Err(e) => Err(e),
            };
            match result {
                Ok(text) => {
                    if self.consecutive_failures.swap(0, Ordering::SeqCst) >= FAILURES_BEFORE_FALLBACK {
                        info!("[REMOTE_WHISPER] Server reachable again — back to remote transcription");
                    }
                    return Ok(TranscriptResult { text, confidence: None, is_partial: false });
                }
                Err(e) => {
                    // the server may have restarted or unloaded its model; re-check readiness next time
                    self.server_ready.store(false, Ordering::SeqCst);
                    let failures = self.consecutive_failures.fetch_add(1, Ordering::SeqCst) + 1;
                    warn!("[REMOTE_WHISPER] Segment failed on server ({} in a row): {}", failures, e);
                    if self.fallback.is_none() {
                        return Err(TranscriptionError::EngineFailed(format!("Remote Whisper: {}", e)));
                    }
                    if failures == FAILURES_BEFORE_FALLBACK {
                        warn!("[REMOTE_WHISPER] Switching to local Parakeet until the server responds again");
                    }
                }
            }
        }
        // Local fallback (Parakeet ignores the language preference)
        let engine = self.fallback.as_ref().ok_or(TranscriptionError::ModelNotLoaded)?;
        match engine.transcribe_audio(audio).await {
            Ok(text) => Ok(TranscriptResult { text: text.trim().to_string(), confidence: None, is_partial: false }),
            Err(e) => Err(TranscriptionError::EngineFailed(e.to_string())),
        }
    }

    async fn is_model_loaded(&self) -> bool {
        true // the model lives on the server; the fallback is optional
    }

    async fn get_current_model(&self) -> Option<String> {
        Some(format!("remote:{}", self.base_url))
    }

    fn provider_name(&self) -> &'static str {
        "Remote Whisper"
    }
}
