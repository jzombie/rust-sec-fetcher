use crate::caches::Caches;
use crate::config::ConfigManager;
use email_address::EmailAddress;
use reqwest;
use reqwest_drive::{
    init_cache_with_drive_and_throttle, init_client_with_cache_and_throttle, CachePolicy,
    ClientWithMiddleware, ThrottlePolicy,
};
use serde_json::Value;
use std::error::Error;
use std::sync::Arc;
use tokio::time::Duration;

pub struct SecClient {
    email: String,
    http_client: ClientWithMiddleware,
    cache_policy: Arc<CachePolicy>,
    throttle_policy: Arc<ThrottlePolicy>,
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

        let max_retries = config
            .max_retries
            .ok_or_else(|| "Missing required field: max_retries".to_string())?; // Error if missing

        let cache_policy = Arc::new(CachePolicy {
            default_ttl: Duration::from_secs(60 * 60 * 24 * 7), // 1 week,
            respect_headers: false,
            cache_status_override: None,
        });

        // Example: Custom Fixed Throttle Policy
        let throttle_policy = Arc::new(ThrottlePolicy {
            base_delay_ms: min_delay,
            max_concurrent,
            max_retries,
            // TODO: Make configurable
            adaptive_jitter_ms: 500,
        });

        // let (cache, throttle) = init_cache_with_throttle(
        //     &config_manager.get_config().get_http_cache_storage_bin(),
        //     cache_policy,
        //     throttle_policy,
        // );

        // // Build client with both cache and throttle middleware
        // let cache_client = ClientBuilder::new(Client::new())
        //     .with_arc(cache) // Cache middleware
        //     .with_arc(throttle) // Throttle middleware (cache linked)
        //     .build();

        let http_cache = Caches::get_http_cache_store();

        let (drive_cache, throttle_cache) = init_cache_with_drive_and_throttle(
            http_cache,
            cache_policy.as_ref().clone(),
            throttle_policy.as_ref().clone(),
        );

        let cache_client = init_client_with_cache_and_throttle(drive_cache, throttle_cache);

        Ok(Self {
            email: email.to_string(),
            http_client: cache_client,
            cache_policy,
            throttle_policy,
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

    // pub async fn raw_request(
    //     &self,
    //     method: reqwest::Method,
    //     url: &str,
    //     headers: Option<Vec<(&str, &str)>>,
    // ) -> Result<reqwest::Response, Box<dyn Error>> {
    //     let mut request_builder = self
    //         .http_client
    //         .request(method, url)
    //         .header("User-Agent", self.get_user_agent());

    //     if let Some(hdrs) = headers {
    //         for (key, value) in hdrs {
    //             request_builder = request_builder.header(key, value);
    //         }
    //     }

    //     let response = request_builder.send().await?;

    //     Ok(response)
    // }

    pub fn get_cache_policy(&self) -> CachePolicy {
        self.cache_policy.as_ref().clone()
    }

    pub fn get_throttle_policy(&self) -> ThrottlePolicy {
        self.throttle_policy.as_ref().clone()
    }

    pub async fn raw_request(
        &self,
        method: reqwest::Method,
        url: &str,
        headers: Option<Vec<(&str, &str)>>,
        custom_throttle_policy: Option<ThrottlePolicy>, // Allow overriding throttle settings
    ) -> Result<reqwest::Response, Box<dyn Error>> {
        let mut request_builder = self
            .http_client
            .request(method, url)
            .header("User-Agent", self.get_user_agent());

        if let Some(hdrs) = headers {
            for (key, value) in hdrs {
                request_builder = request_builder.header(key, value);
            }
        }

        // Inject a custom throttle policy if provided
        if let Some(policy) = custom_throttle_policy {
            request_builder.extensions().insert(policy);
        }

        let response = request_builder.send().await?;
        Ok(response)
    }

    // TODO: Add optional headers
    /// Asynchronously fetches JSON data from a given SEC URL with rate limiting
    pub async fn fetch_json(
        &self,
        url: &str,
        custom_throttle_policy: Option<ThrottlePolicy>,
    ) -> Result<Value, Box<dyn Error>> {
        let response: reqwest::Response = self
            .raw_request(reqwest::Method::GET, url, None, custom_throttle_policy)
            .await?;
        let json = response.json().await?;
        Ok(json)
    }
}
