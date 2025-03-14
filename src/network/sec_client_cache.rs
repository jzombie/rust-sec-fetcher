use async_trait::async_trait;
use http::{HeaderMap, StatusCode, Extensions};
use reqwest::{Request, Response, Version, Body};
use reqwest_middleware::{Middleware, Next, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use bytes::Bytes;
use std::convert::TryFrom;

#[derive(Clone, Default)]
pub struct HashMapCache {
    store: Arc<RwLock<HashMap<String, (StatusCode, HeaderMap, Bytes)>>>, // Store status, headers, and body
}

#[async_trait]
impl Middleware for HashMapCache {
    async fn handle(
        &self,
        req: Request,
        extensions: &mut Extensions,
        next: Next<'_>,
    ) -> Result<Response> {
        let url = req.url().to_string();
        
        // Check if response is cached
        {
            let store = self.store.read().await;
            if let Some((status, headers, body)) = store.get(&url) {
                return Ok(build_response(*status, headers.clone(), body.clone()));
            }
        }

        // Call the next middleware/request
        let response = next.run(req, extensions).await?;
        let status = response.status();
        let headers = response.headers().clone();
        let body = response.bytes().await?; // Read the response body

        // Store the response in cache
        {
            let mut store = self.store.write().await;
            store.insert(url, (status, headers.clone(), body.clone()));
        }

        // Return a new response since the original response body is consumed
        Ok(build_response(status, headers, body))
    }
}

/// **Helper function to rebuild a `reqwest::Response`**
fn build_response(status: StatusCode, headers: HeaderMap, body: Bytes) -> Response {
    let mut response_builder = http::Response::builder()
        .status(status);
    
    for (key, value) in headers.iter() {
        response_builder = response_builder.header(key, value);
    }

    let http_response = response_builder
        .body(body)
        .expect("Failed to create HTTP response");

    Response::try_from(http_response).expect("Failed to convert to reqwest::Response")
}