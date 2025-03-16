use rand::Rng;
use reqwest::{Request, Response};
use reqwest_middleware::{Middleware, Next, Error};
use http::Extensions;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::time::{sleep, Duration};
use async_trait::async_trait;
use crate::network::HashMapCache;

/// Cache policy struct: Controls TTL behavior
#[derive(Clone, Debug)]
pub struct ThrottlePolicy {
    pub base_delay_ms: u64,
    pub adaptive_jitter_ms: u64,
    pub max_concurrent: usize,
    pub max_retries: usize,
}

pub struct ThrottleBackoffMiddleware {
    semaphore: Arc<Semaphore>,
    policy: ThrottlePolicy,
    cache: Arc<HashMapCache>,
}

impl ThrottleBackoffMiddleware {
    pub fn new(policy: ThrottlePolicy, cache: Arc<HashMapCache>) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(policy.max_concurrent)),
            policy,
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

        sleep(Duration::from_millis(self.policy.base_delay_ms)).await;

        let mut attempt = 0;

        loop {
            let req_clone = req.try_clone().expect("Request cloning failed");
            let result = next.clone().run(req_clone, extensions).await;

            match result {
                Ok(resp) if resp.status().is_success() => return Ok(resp),
                result if attempt >= self.policy.max_retries => return result,
                _ => {
                    attempt += 1;

                    let backoff_duration = {
                        let mut rng = rand::rng();
                        Duration::from_millis(
                            self.policy.base_delay_ms * 2u64.pow(attempt as u32)
                                + rng.random_range(0..=self.policy.adaptive_jitter_ms),
                        )
                    };

                    eprintln!(
                        "Retry {}/{} for URL {} after {} ms",
                        attempt,
                        self.policy.max_retries,
                        url,
                        backoff_duration.as_millis()
                    );

                    sleep(backoff_duration).await;

                    if attempt >= self.policy.max_retries {
                        break;
                    }
                }
            }
        }

        next.run(req, extensions).await
    }
}
