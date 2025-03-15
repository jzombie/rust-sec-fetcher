use async_trait::async_trait;
use http::{HeaderMap, StatusCode, Extensions};
use reqwest::{Request, Response, Body};
use reqwest_middleware::{Middleware, Next, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use bytes::Bytes;
use std::convert::TryFrom;

const CACHE_TTL: Duration = Duration::from_secs(60); // Cache expires after 60 seconds

#[derive(Clone, Default)]
pub struct HashMapCache {
    store: Arc<RwLock<HashMap<String, (StatusCode, HeaderMap, Bytes, Instant)>>>,
}

impl HashMapCache {
    /// **Checks if a request URL is cached and still valid**
    pub async fn is_cached(&self, cache_key: &str) -> bool {
        let store = self.store.read().await;
        if let Some((_, _, _, timestamp)) = store.get(cache_key) {
            return timestamp.elapsed() < CACHE_TTL;
        }
        false
    }
    
    /// **Generates a unique cache key based on method, URL, and important headers**
    fn generate_cache_key(req: &Request) -> String {
        let method = req.method().as_str();
        let url = req.url().to_string();

        // Consider headers that affect responses
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

        // **Only cache GET and HEAD requests**
        if req.method() == "GET" || req.method() == "HEAD" {
            // **Check cache before making a request**
            {
                let store = self.store.read().await;
                if let Some((status, headers, body, timestamp)) = store.get(&cache_key) {
                    if timestamp.elapsed() < CACHE_TTL {
                        return Ok(build_response(*status, headers.clone(), body.clone()));
                    }
                }
            }

            // If not cached (or expired), proceed with the request
            let response = next.run(req, extensions).await?;
            let status = response.status();
            let headers = response.headers().clone();
            let body = response.bytes().await?;

            // **Respect `Vary` headers**
            let mut final_cache_key = cache_key.clone();
            if let Some(vary) = headers.get("vary") {
                if let Ok(vary_value) = vary.to_str() {
                    let vary_headers = vary_value.split(',').map(|s| s.trim().to_lowercase());
                    let vary_cache_string = vary_headers
                        .filter_map(|h| headers.get(&h).map(|v| v.to_str().unwrap_or_default()))
                        .collect::<Vec<_>>()
                        .join(",");
                    
                    if !vary_cache_string.is_empty() {
                        final_cache_key.push_str(&format!(" Vary:{}", vary_cache_string));
                    }
                }
            }

            // **Store response in cache with timestamp**
            {
                let mut store = self.store.write().await;
                store.insert(final_cache_key, (status, headers.clone(), body.clone(), Instant::now()));
            }

            return Ok(build_response(status, headers, body));
        }

        // If not GET/HEAD, pass through without caching
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
