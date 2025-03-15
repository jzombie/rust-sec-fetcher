use serde::Deserialize;
use rand::Rng;
use reqwest::{Request, Response};
use reqwest_middleware::{Middleware, Next, Error};
use http::Extensions;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::time::{sleep, Duration};
use async_trait::async_trait;
use crate::network::HashMapCache;

#[derive(Debug, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case")]
pub enum ThrottleConfig {
    /// **Fixed Delay Throttling:**
    /// - Applies a constant delay between requests.
    /// - Ensures requests do not exceed a predefined rate.
    /// - Useful when dealing with strict API rate limits.
    Fixed {
        fixed_delay_ms: u64,
        max_concurrent: Option<usize>,
        max_retries: Option<usize>,
    },

    /// **Adaptive Delay Throttling:**
    /// - Uses exponential backoff with added jitter to dynamically adjust delays.
    /// - Helps avoid overloading the server while maximizing throughput.
    /// - Suitable for APIs with rate limits that respond with 429 errors.
    Adaptive {
        adaptive_base_delay_ms: u64,
        adaptive_jitter_ms: u64,
        max_concurrent: Option<usize>,
        max_retries: Option<usize>,
    },

    /// **No Throttling:**
    /// - Does not enforce any delay between requests.
    /// - Still limits max concurrent requests if specified.
    /// - Can be useful when operating in a low-traffic environment.
    None {
        max_concurrent: Option<usize>,
        max_retries: Option<usize>,
    },
}

pub enum ThrottlePolicy {
    FixedDelay(Duration),
    AdaptiveDelay { base_delay: Duration, jitter: Duration },
    NoThrottle,
}

impl From<&ThrottleConfig> for ThrottlePolicy {
    fn from(config: &ThrottleConfig) -> Self {
        match config {
            ThrottleConfig::Fixed { fixed_delay_ms, .. } => {
                Self::FixedDelay(Duration::from_millis(*fixed_delay_ms))
            }
            ThrottleConfig::Adaptive { adaptive_base_delay_ms, adaptive_jitter_ms, .. } => {
                Self::AdaptiveDelay {
                    base_delay: Duration::from_millis(*adaptive_base_delay_ms),
                    jitter: Duration::from_millis(*adaptive_jitter_ms),
                }
            }
            ThrottleConfig::None { .. } => Self::NoThrottle,
        }
    }
}

pub struct ThrottleBackoffMiddleware {
    pub semaphore: Arc<Semaphore>,
    pub policy: ThrottlePolicy,
    pub max_retries: usize,
    pub cache: Arc<HashMapCache>,
}

impl ThrottleBackoffMiddleware {
    pub fn from_config(config: &ThrottleConfig, cache: Arc<HashMapCache>) -> Self {
        let max_concurrent = match config {
            ThrottleConfig::Fixed { max_concurrent, .. }
            | ThrottleConfig::Adaptive { max_concurrent, .. }
            | ThrottleConfig::None { max_concurrent, .. } => max_concurrent.unwrap_or(1),
        };

        let max_retries = match config {
            ThrottleConfig::Fixed { max_retries, .. }
            | ThrottleConfig::Adaptive { max_retries, .. }
            | ThrottleConfig::None { max_retries, .. } => max_retries.unwrap_or(0),
        };

        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            policy: ThrottlePolicy::from(config),
            max_retries,
            cache,
        }
    }
}

#[async_trait]
impl Middleware for ThrottleBackoffMiddleware {
    async fn handle(
        &self,
        req: Request,
        extensions: &mut Extensions,
        next: Next<'_>,
    ) -> Result<Response, Error> {
        let url = req.url().to_string();

        let cache_key = format!("{} {}", req.method(), &url);

        if self.cache.is_cached(&req).await {
            eprintln!("Using cache for: {}", &cache_key);

            return next.run(req, extensions).await;
        } else {
            eprintln!("No cache found for: {}", &cache_key);
        }

        let _permit = self.semaphore.acquire().await.map_err(|e| Error::Middleware(e.into()))?;

        match &self.policy {
            ThrottlePolicy::FixedDelay(delay) => sleep(*delay).await,
            ThrottlePolicy::AdaptiveDelay { base_delay, .. } => sleep(*base_delay).await,
            ThrottlePolicy::NoThrottle => {}
        }

        let mut attempt = 0;

        loop {
            let req_clone = req.try_clone().expect("Request cloning failed");
            let result = next.clone().run(req_clone, extensions).await;

            match result {
                Ok(resp) if resp.status().is_success() => return Ok(resp),
                result if attempt >= self.max_retries => return result,
                _ => {
                    attempt += 1;

                    let backoff_duration = match &self.policy {
                        ThrottlePolicy::AdaptiveDelay { base_delay, jitter } => {
                            let mut rng = rand::rng();
                            Duration::from_millis(
                                base_delay.as_millis() as u64 * 2u64.pow(attempt as u32)
                                    + rng.random_range(0..=jitter.as_millis() as u64),
                            )
                        }
                        ThrottlePolicy::FixedDelay(delay) => *delay,
                        ThrottlePolicy::NoThrottle => Duration::ZERO,
                    };

                    eprintln!(
                        "Retry {}/{} for URL {} after {} ms",
                        attempt + 1,
                        self.max_retries,
                        url,
                        backoff_duration.as_millis()
                    );

                    sleep(backoff_duration).await;

                    if attempt >= self.max_retries {
                        break;
                    }
                }
            }
        }

        next.run(req, extensions).await
    }
}
