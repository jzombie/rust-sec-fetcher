use crate::config::ConfigManager;
use email_address::EmailAddress;
use reqwest::Client;
use reqwest_drive::{init_cache_with_throttle, CachePolicy, ThrottlePolicy};
use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use serde_json::Value;
use std::error::Error;
use tokio::time::Duration;

pub struct SecClient {
    email: String,
    client: ClientWithMiddleware,
}

impl SecClient {
    /// Creates a new async SEC HTTP client with optional rate limiting
    pub fn from_config_manager(
        config_manager: &ConfigManager,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let config = &config_manager.get_config();

        let email = config
            .email
            .as_ref()
            .ok_or_else(|| "Missing required field: email".to_string())?; // Error if missing

        let max_concurrent = config
            .max_concurrent
            .ok_or_else(|| "Missing required field: max_concurrent".to_string())?; // Error if missing

        let min_delay = config
            .min_delay_ms
            .ok_or_else(|| "Missing required field: min_delay_ms".to_string())?; // Error if missing

        let cache_policy = CachePolicy {
            default_ttl: Duration::from_secs(60 * 60 * 24 * 7), // 1 week,
            respect_headers: false,
            cache_status_override: None,
        };

        // Example: Custom Fixed Throttle Policy
        let throttle_policy = ThrottlePolicy {
            base_delay_ms: min_delay,
            max_concurrent,
            // TODO: Make configurable
            max_retries: 5,
            adaptive_jitter_ms: 500,
        };

        let (cache, throttle) = init_cache_with_throttle(
            &config_manager.get_config().get_http_cache_storage_bin(),
            cache_policy,
            throttle_policy,
        );

        // Build client with both cache and throttle middleware
        let cache_client = ClientBuilder::new(Client::new())
            .with_arc(cache) // Cache middleware
            .with_arc(throttle) // Throttle middleware (cache linked)
            .build();

        Ok(Self {
            email: email.to_string(),
            client: cache_client,
        })
    }

    pub fn get_user_agent(&self) -> String {
        // Note: The intention is to check it here vs. during instantiation as
        // every network path relies on this method, whereas the instance can
        // be instantiated different ways.
        if !EmailAddress::is_valid(&self.email) {
            // This is a non-recoverable error
            panic!("Invalid email format");
        }

        // TODO: Include repository URL

        format!(
            "{}/{} (+{})",
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
            self.email
        )
    }

    pub async fn raw_request(
        &self,
        method: reqwest::Method,
        url: &str,
        headers: Option<Vec<(&str, &str)>>,
    ) -> Result<reqwest::Response, Box<dyn Error>> {
        let mut request_builder = self
            .client
            .request(method, url)
            .header("User-Agent", self.get_user_agent());

        if let Some(hdrs) = headers {
            for (key, value) in hdrs {
                request_builder = request_builder.header(key, value);
            }
        }

        let response = request_builder.send().await?;

        Ok(response)
    }

    // TODO: Add optional headers
    /// Asynchronously fetches JSON data from a given SEC URL with rate limiting
    pub async fn fetch_json(&self, url: &str) -> Result<Value, Box<dyn Error>> {
        let response = self.raw_request(reqwest::Method::GET, url, None).await?;
        let json = response.json().await?;
        Ok(json)
    }
}
