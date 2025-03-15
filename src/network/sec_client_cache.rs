use async_trait::async_trait;
use bytes::Bytes;
use http::{HeaderMap, HeaderValue, StatusCode, Extensions};
use reqwest::{Request, Response};
use reqwest_middleware::{Middleware, Next, Result};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::path::PathBuf;
use std::convert::TryFrom;
use simd_r_drive::DataStore;
use bincode; // Binary serialization
use chrono::{DateTime, Utc}; // For parsing `Expires` headers

/// **Cache policy struct: Controls TTL behavior**
#[derive(Clone, Debug)]
pub struct CachePolicy {
    pub default_ttl: Duration, // Fallback TTL when headers are missing
    pub respect_headers: bool, // Whether to extract TTL from response headers
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self {
            default_ttl: Duration::from_secs(60 * 60 * 24), // Default 1 day TTL
            respect_headers: true, // Use headers if available
        }
    }
}

/// **Struct to store cached responses**
#[derive(Serialize, Deserialize)]
struct CachedResponse {
    status: u16,
    headers: Vec<(String, Vec<u8>)>, // Store headers as raw bytes
    body: Vec<u8>,
    expiration_timestamp: u64, // Timestamp when cache expires
}

#[derive(Clone)]
pub struct HashMapCache {
    // store: Arc<RwLock<HashMap<String, Vec<u8>>>>, // Store raw binary
    store: Arc<DataStore>,
    policy: CachePolicy, // Configurable policy
}

impl HashMapCache {
    pub fn new(policy: CachePolicy) -> Self {
        Self {
            // store: Arc::new(RwLock::new(HashMap::new())),
            // TODO: Don't hardcode props
            store: Arc::new(DataStore::open(&PathBuf::from("data/temp-cache.bin")).unwrap()),
            policy,
        }
    }

    /// **Determines if a request URL is cached and still valid based on CachePolicy**
    // pub async fn is_cached(&self, method: &str, url: &str, headers: &HeaderMap) -> bool {
    pub async fn is_cached(&self, req: &Request) -> bool {
        let store = self.store.as_ref();

        let cache_key = self.generate_cache_key(req);
        let cache_key_bytes = cache_key.as_bytes();

        // let store = self.store.read().await;
        if let Some(entry_handle) = store.read(cache_key_bytes) {
            eprintln!("Entry handle: {:?}", entry_handle);

            if let Ok(cached) = bincode::deserialize::<CachedResponse>(entry_handle.as_slice()) {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("Time went backwards")
                    .as_millis() as u64;

                // Extract TTL based on the policy (either from headers or default)
                let ttl = if self.policy.respect_headers {
                    // Convert headers back to HeaderMap to extract TTL
                    let mut headers = HeaderMap::new();
                    for (k, v) in cached.headers.iter() {
                        if let Ok(header_name) = k.parse::<http::HeaderName>() {
                            if let Ok(header_value) = HeaderValue::from_bytes(v) {
                                headers.insert(header_name, header_value);
                            }
                        }
                    }
                    Self::extract_ttl(&headers, &self.policy)
                } else {
                    self.policy.default_ttl
                };

                let expected_expiration = cached.expiration_timestamp + ttl.as_millis() as u64;

                // If expired, remove from cache
                if now >= expected_expiration {
                    // eprintln!("Determined cache is expired. now - expected_expiration: {:?}", now - expected_expiration);
                    eprintln!("Cache expires at: {}", chrono::DateTime::from_timestamp_millis(expected_expiration as i64).unwrap());
                    eprintln!("Expiration timestamp: {}", chrono::DateTime::from_timestamp_millis(cached.expiration_timestamp as i64).unwrap());
                    eprintln!("Now: {}", chrono::DateTime::from_timestamp_millis(now as i64).unwrap());

                    // TODO: Uncomment

                    // TODO: Rename API method to `delete`
                    // store.delete_entry(cache_key_bytes).ok();
                    // return false;
                }

                return true;
            }
        }
        false
    }


    /// **Generates a unique cache key based on method, URL, and important headers**
    // fn generate_cache_key(method: &str, url: &str, headers: &HeaderMap) -> String {
    fn generate_cache_key(&self, req: &Request) -> String {
        let method = req.method();
        let url = req.url().as_str();
        let headers = req.headers();

        // let method = req.method().as_str();
        // let url = req.url().to_string();

        let relevant_headers = vec!["accept", "authorization"];
        let header_string = relevant_headers.iter()
            .filter_map(|h| headers.get(*h))
            .map(|v| v.to_str().unwrap_or_default())
            .collect::<Vec<_>>()
            .join(",");

        format!("{} {} {}", method, url, header_string)
    }

    /// **Extracts TTL from headers if `respect_headers` is enabled**
    fn extract_ttl(headers: &HeaderMap, policy: &CachePolicy) -> Duration {
        if !policy.respect_headers {
            return policy.default_ttl;
        }

        // Check `Cache-Control: max-age=N`
        if let Some(cache_control) = headers.get("cache-control") {
            if let Ok(cache_control) = cache_control.to_str() {
                for directive in cache_control.split(',') {
                    if let Some(max_age) = directive.trim().strip_prefix("max-age=") {
                        if let Ok(seconds) = max_age.parse::<u64>() {
                            return Duration::from_secs(seconds);
                        }
                    }
                }
            }
        }

        // Check `Expires`
        if let Some(expires) = headers.get("expires") {
            if let Ok(expires) = expires.to_str() {
                if let Ok(expiry_time) = DateTime::parse_from_rfc2822(expires) {
                    if let Some(duration) = expiry_time.timestamp().checked_sub(Utc::now().timestamp()) {
                        if duration > 0 {
                            return Duration::from_secs(duration as u64);
                        }
                    }
                }
            }
        }

        // Fallback to default TTL
        policy.default_ttl
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
        let cache_key = self.generate_cache_key(&req);

        eprintln!("Handle cache key: {}", cache_key);

        let store = self.store.as_ref();
        let cache_key_bytes = cache_key.as_bytes();

        if req.method() == "GET" || req.method() == "HEAD" {
            // **Use is_cached() to determine if the cache should be used**
            if self.is_cached(&req).await {
                // let store = self.store.read().await;
                if let Some(entry_handle) = store.read(&cache_key_bytes) {
                    if let Ok(cached) = bincode::deserialize::<CachedResponse>(entry_handle.as_slice()) {
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

            let response = next.run(req, extensions).await?;
            let status = response.status();
            let headers = response.headers().clone();
            let body = response.bytes().await?.to_vec();
            
            let ttl = Self::extract_ttl(&headers, &self.policy);
            let expiration_timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards")
                .as_millis() as u64 + ttl.as_millis() as u64;

            let body_clone = body.clone(); // Fix: Clone before moving

            let serialized = bincode::serialize(&CachedResponse {
                status: status.as_u16(),
                headers: headers.iter().map(|(k, v)| (k.to_string(), v.as_bytes().to_vec())).collect(),
                body, // Move the original body here
                expiration_timestamp,
            }).expect("Serialization failed");

            {
                // let mut store = self.store.write().await;
                // store.insert(cache_key, serialized);

                let store = self.store.as_ref();

                eprintln!("Writing cache with key: {}", cache_key);
                store.write(cache_key_bytes, serialized.as_slice()).ok();
            }

            return Ok(build_response(status, headers, Bytes::from(body_clone)));
        }

        next.run(req, extensions).await
    }
}

/// **Allow `HashMapCache::default()` to work**
impl Default for HashMapCache {
    fn default() -> Self {
        Self::new(CachePolicy::default()) // Default instance
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
