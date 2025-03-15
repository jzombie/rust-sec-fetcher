use async_trait::async_trait;
use bytes::Bytes;
use http::{HeaderMap, HeaderValue, StatusCode, Extensions};
use reqwest::{Request, Response};
use reqwest_middleware::{Middleware, Next, Result};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use std::convert::TryFrom;
use bincode; // Binary serialization

const CACHE_TTL: Duration = Duration::from_secs(60); // Cache expires in 60 seconds

#[derive(Serialize, Deserialize)]
struct CachedResponse {
    status: u16,
    headers: Vec<(String, Vec<u8>)>, // Serialize headers as raw bytes
    body: Vec<u8>,
    timestamp: u64, // Store time in epoch millis for cross-platform consistency
}

#[derive(Clone, Default)]
pub struct HashMapCache {
    store: Arc<RwLock<HashMap<String, Vec<u8>>>>, // Store raw binary
}

impl HashMapCache {
    /// **Checks if a request URL is cached and still valid**
    pub async fn is_cached(&self, cache_key: &str) -> bool {
        let store = self.store.read().await;
        if let Some(raw_data) = store.get(cache_key) {
            if let Ok(cached) = bincode::deserialize::<CachedResponse>(raw_data) {
                if let Ok(elapsed) = SystemTime::now().duration_since(UNIX_EPOCH) {
                    let cache_age = elapsed.as_millis() as u64 - cached.timestamp;
                    return Duration::from_millis(cache_age) < CACHE_TTL;
                }
            }
        }
        false
    }

    /// **Generates a unique cache key based on method, URL, and important headers**
    fn generate_cache_key(req: &Request) -> String {
        let method = req.method().as_str();
        let url = req.url().to_string();

        let relevant_headers = vec!["accept", "authorization"];
        let header_string = relevant_headers.iter()
            .filter_map(|h| req.headers().get(*h))
            .map(|v| v.to_str().unwrap_or_default())
            .collect::<Vec<_>>()
            .join(",");

        format!("{} {} {}", method, url, header_string)
    }
}

#[async_trait]
impl Middleware for HashMapCache {
    async fn handle(
        &self,
        req: Request,
        extensions: &mut Extensions,
        next: Next<'_>,
    ) -> Result<Response> {
        let cache_key = Self::generate_cache_key(&req);

        if req.method() == "GET" || req.method() == "HEAD" {
            {
                let store = self.store.read().await;
                if let Some(raw_data) = store.get(&cache_key) {
                    if let Ok(cached) = bincode::deserialize::<CachedResponse>(raw_data) {
                        if let Ok(elapsed) = SystemTime::now().duration_since(UNIX_EPOCH) {
                            let cache_age = elapsed.as_millis() as u64 - cached.timestamp;
                            if Duration::from_millis(cache_age) < CACHE_TTL {
                                let mut headers = HeaderMap::new();
                                for (k, v) in cached.headers {
                                    if let Ok(header_name) = k.parse::<http::HeaderName>() {
                                        if let Ok(header_value) = HeaderValue::from_bytes(&v) {
                                            headers.insert(header_name, header_value);
                                        }
                                    }
                                }
                                let status = StatusCode::from_u16(cached.status).unwrap_or(StatusCode::OK);
                                return Ok(build_response(status, headers, Bytes::from(cached.body)));
                            }
                        }
                    }
                }
            }

            let response = next.run(req, extensions).await?;
            let status = response.status();
            let headers = response.headers().clone();
            let body = response.bytes().await?.to_vec();
            
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards")
                .as_millis() as u64;

            let body_clone = body.clone(); // Fix: Clone before moving

            let serialized = bincode::serialize(&CachedResponse {
                status: status.as_u16(),
                headers: headers.iter().map(|(k, v)| (k.to_string(), v.as_bytes().to_vec())).collect(),
                body, // Move the original body here
                timestamp,
            }).expect("Serialization failed");

            {
                let mut store = self.store.write().await;
                store.insert(cache_key, serialized);
            }

            // Use the cloned body for returning the response
            return Ok(build_response(status, headers, Bytes::from(body_clone)));
        }

        next.run(req, extensions).await
    }
}

/// **Helper function to rebuild a `reqwest::Response`**
fn build_response(status: StatusCode, headers: HeaderMap, body: Bytes) -> Response {
    let mut response_builder = http::Response::builder().status(status);
    
    for (key, value) in headers.iter() {
        response_builder = response_builder.header(key, value);
    }

    let http_response = response_builder
        .body(body)
        .expect("Failed to create HTTP response");

    Response::try_from(http_response).expect("Failed to convert to reqwest::Response")
}
