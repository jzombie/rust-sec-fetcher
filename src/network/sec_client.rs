use crate::config::ConfigManager;
use email_address::EmailAddress;
use rand::Rng;
use reqwest::Client;
use serde_json::Value;
use std::error::Error;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::time::{sleep, Duration};
use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use http_cache_reqwest::{Cache, CACacheManager, HttpCache, HttpCacheOptions};

pub struct SecClient {
    email: String,
    client: ClientWithMiddleware,
    semaphore: Arc<Semaphore>, // Limit concurrent requests
    min_delay: Duration,       // Enforce delay
    max_retries: Option<usize>,
}

pub trait SecClientDataExt {
    fn get_user_agent(&self) -> String;

    #[allow(async_fn_in_trait)]
    async fn raw_request_without_retry(
        &self,
        method: reqwest::Method,
        url: &str,
        headers: Option<Vec<(&str, &str)>>,
    ) -> Result<reqwest::Response, Box<dyn Error>>;

    #[allow(async_fn_in_trait)]
    async fn raw_request_with_retry(
        &self,
        method: reqwest::Method,
        url: &str,
        headers: Option<Vec<(&str, &str)>>,
    ) -> Result<reqwest::Response, Box<dyn Error>>;

    #[allow(async_fn_in_trait)]
    async fn fetch_json_without_retry(&self, url: &str) -> Result<Value, Box<dyn Error>>;

    #[allow(async_fn_in_trait)]
    async fn fetch_json_with_retry(&self, url: &str) -> Result<Value, Box<dyn Error>>;
}

impl SecClient {
    /// Creates a new async SEC HTTP client with optional rate limiting
    pub fn from_config_manager(
        config_manager: &ConfigManager,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let config = &config_manager.get_config();

        let email = match &config.email {
            Some(email) => email,
            None => return Err("No email specified.".into())
        };

        // TODO: Clean up
        let cache_client = ClientBuilder::new(Client::new())
        .with(Cache(HttpCache {
            // https://docs.rs/http-cache-reqwest/latest/http_cache_reqwest/enum.CacheMode.html
            // mode: CacheMode::Default,

            // This will cache https://data.sec.gov/api/xbrl/companyfacts/CIKXXXXXXXXXX.json
            // mode: CacheMode::IgnoreRules,
            mode: config.get_cache_mode(),
            //
            manager: CACacheManager {
            // https://docs.rs/http-cache-reqwest/latest/http_cache_reqwest/struct.CACacheManager.html
            // path: PathBuf::from("data/cache")
            path: config.get_cache_dir(),

            // or temp directory of OS
            // path: env::temp_dir();
            },
            options: HttpCacheOptions::default(),
        }))
        .build();

        Ok(
            Self {
                email: email.to_string(),
                client: cache_client,
                semaphore: Arc::new(Semaphore::new(config.max_concurrent.unwrap())),
                min_delay: Duration::from_millis(config.min_delay_ms.unwrap()),
                max_retries: config.max_retries,
            }
        )
    }

    pub async fn raw_request(
        &self,
        method: reqwest::Method,
        url: &str,
        headers: Option<Vec<(&str, &str)>>,
    ) -> Result<reqwest::Response, Box<dyn Error>> {
        match self.max_retries {
            Some(_) => self.raw_request_with_retry(method, url, headers).await,
            None => self.raw_request_without_retry(method, url, headers).await,
        }
    }

    pub async fn fetch_json(&self, url: &str) -> Result<Value, Box<dyn Error>> {
        match self.max_retries {
            Some(_) => self.fetch_json_with_retry(url).await,
            None => self.fetch_json_without_retry(url).await,
        }
    }
}

impl SecClientDataExt for SecClient {
    fn get_user_agent(&self) -> String {
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

    async fn raw_request_without_retry(
        &self,
        method: reqwest::Method,
        url: &str,
        headers: Option<Vec<(&str, &str)>>,
    ) -> Result<reqwest::Response, Box<dyn Error>> {
        // FIXME: Determine if is cached before sleeping; subsequent requests could be made much faster
        let _permit = self.semaphore.acquire().await?;
        sleep(self.min_delay).await;

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

    async fn raw_request_with_retry(
        &self,
        method: reqwest::Method,
        url: &str,
        headers: Option<Vec<(&str, &str)>>,
    ) -> Result<reqwest::Response, Box<dyn Error>> {
        let mut attempt = 0;
        let max_retries = self.max_retries.unwrap_or(0);
        let mut rng = rand::rng();

        loop {
            match self
                .raw_request_without_retry(method.clone(), url, headers.clone())
                .await
            {
                Ok(resp) if resp.status().is_success() => return Ok(resp),
                Ok(resp) => {
                    if attempt >= max_retries {
                        return Err(format!("Request failed with status: {}", resp.status()).into());
                    }
                }
                Err(e) => {
                    if attempt >= max_retries {
                        return Err(e);
                    }
                }
            }

            attempt += 1;
            let backoff_ms = (2_u64.pow(attempt as u32) * 500) + rng.random_range(0..200);
            eprintln!(
                "Retrying ({}/{}) after {} ms",
                attempt, max_retries, backoff_ms
            );
            sleep(Duration::from_millis(backoff_ms)).await;
        }
    }

    // TODO: Add optional headers
    /// Asynchronously fetches JSON data from a given SEC URL with rate limiting
    async fn fetch_json_without_retry(&self, url: &str) -> Result<Value, Box<dyn Error>> {
        let response = self
            .raw_request_without_retry(reqwest::Method::GET, url, None)
            .await?;
        let json = response.json().await?;
        Ok(json)
    }

    // TODO: Add optional headers
    async fn fetch_json_with_retry(&self, url: &str) -> Result<Value, Box<dyn Error>> {
        let response = self
            .raw_request_with_retry(reqwest::Method::GET, url, None)
            .await?;
        let json = response.json().await?;
        Ok(json)
    }
}
