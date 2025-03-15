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
pub struct ThrottleConfig {
    pub policy: String,
    pub fixed_delay_ms: Option<u64>,
    pub adaptive_base_delay_ms: Option<u64>,
    pub adaptive_jitter_ms: Option<u64>,
    pub max_concurrent: Option<usize>,
    pub max_retries: Option<usize>,
}

pub enum ThrottlePolicy {
    FixedDelay(Duration),
    AdaptiveDelay { base_delay: Duration, jitter: Duration },
    NoThrottle,
}

impl TryFrom<ThrottleConfig> for ThrottlePolicy {
    type Error = String;

    fn try_from(config: ThrottleConfig) -> Result<Self, Self::Error> {
        match config.policy.as_str() {
            "fixed" => Ok(Self::FixedDelay(Duration::from_millis(
                config.fixed_delay_ms.ok_or("Missing fixed_delay_ms")?,
            ))),
            "adaptive" => {
                let base_delay = Duration::from_millis(
                    config
                        .adaptive_base_delay_ms
                        .ok_or("Missing adaptive_base_delay_ms")?,
                );
                let jitter = Duration::from_millis(
                    config
                        .adaptive_jitter_ms
                        .ok_or("Missing adaptive_jitter_ms")?,
                );
                Ok(Self::AdaptiveDelay { base_delay, jitter })
            }
            "none" => Ok(Self::NoThrottle),
            other => Err(format!("Invalid throttle policy: {}", other)),
        }
    }
}

pub struct ThrottleBackoffMiddleware {
    pub semaphore: Arc<Semaphore>,
    pub policy: ThrottlePolicy,
    pub max_retries: usize,
    pub cache: Arc<HashMapCache>,
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

        if self.cache.is_cached(&url).await {
            return next.run(req, extensions).await;
        }

        let _permit = self.semaphore.acquire().await.map_err(|e| Error::Middleware(e.into()))?;

        match &self.policy {
            ThrottlePolicy::FixedDelay(delay) => {
                sleep(*delay).await;
            }
            ThrottlePolicy::AdaptiveDelay { base_delay, .. } => {
                sleep(*base_delay).await;
            }
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
