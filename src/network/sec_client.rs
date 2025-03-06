use crate::network::CredentialManager;
use email_address::EmailAddress;
use rand::Rng;
use reqwest::Client;
use serde_json::Value;
use std::error::Error;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::time::{sleep, Duration};

pub struct SecClient {
    email: String,
    client: Client,
    semaphore: Arc<Semaphore>, // Limit concurrent requests
    min_delay: Duration,       // Enforce delay
    max_retries: Option<usize>,
}

trait SecClientDataExt {
    async fn raw_request_without_retry(
        &self,
        method: reqwest::Method,
        url: &str,
        headers: Option<Vec<(&str, &str)>>,
    ) -> Result<reqwest::Response, Box<dyn Error>>;
    async fn raw_request_with_retry(
        &self,
        method: reqwest::Method,
        url: &str,
        headers: Option<Vec<(&str, &str)>>,
    ) -> Result<reqwest::Response, Box<dyn Error>>;

    async fn fetch_json_without_retry(&self, url: &str) -> Result<Value, Box<dyn Error>>;
    async fn fetch_json_with_retry(&self, url: &str) -> Result<Value, Box<dyn Error>>;
}

impl SecClient {
    /// Creates a new async SEC HTTP client with optional rate limiting
    pub fn new(
        email: &str,
        max_concurrent: usize,
        min_delay_ms: u64,
        max_retries: Option<usize>,
    ) -> Self {
        SecClient {
            email: email.to_string(),
            client: Client::new(),
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            min_delay: Duration::from_millis(min_delay_ms),
            max_retries,
        }
    }

    pub fn from_credential_manager(
        credential_manager: &CredentialManager,
        max_concurrent: usize,
        max_delay_ms: u64,
        max_retries: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let email = credential_manager.get_credential()?;

        let instance = Self {
            email,
            client: Client::new(),
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            min_delay: Duration::from_millis(max_delay_ms),
            max_retries,
        };

        Ok(instance)
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
    async fn raw_request_without_retry(
        &self,
        method: reqwest::Method,
        url: &str,
        headers: Option<Vec<(&str, &str)>>,
    ) -> Result<reqwest::Response, Box<dyn Error>> {
        if !EmailAddress::is_valid(&self.email) {
            return Err("No valid email defined".into());
        }

        let _permit = self.semaphore.acquire().await?;
        sleep(self.min_delay).await;

        let mut request_builder = self.client.request(method, url).header(
            "User-Agent",
            format!("SECDataFetcher/1.0 (+{})", self.email),
        );

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

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;

    #[tokio::test]
    async fn test_fetch_json_without_retry_success() -> Result<(), Box<dyn Error>> {
        let mut server = Server::new_async().await;

        let _mock = server
            .mock("GET", "/files/company_tickers.json")
            .with_status(200)
            .with_header("Content-Type", "application/json")
            .with_body(r#"{"AAPL": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc"}}"#)
            .create_async()
            .await;

        let client = SecClient::new("test@example.com", 1, 1000, None);

        let result = client
            .fetch_json(&format!("{}/files/company_tickers.json", server.url()))
            .await?;

        assert_eq!(result["AAPL"]["ticker"].as_str(), Some("AAPL"));
        Ok(())
    }

    #[tokio::test]
    async fn test_fetch_json_with_retry_success() -> Result<(), Box<dyn Error>> {
        let mut server = Server::new_async().await;

        let _mock = server
            .mock("GET", "/files/company_tickers.json")
            .with_status(200)
            .with_header("Content-Type", "application/json")
            .with_body(r#"{"AAPL": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc"}}"#)
            .create_async()
            .await;

        let client = SecClient::new("test@example.com", 1, 1000, Some(3));

        let result = client
            .fetch_json(&format!("{}/files/company_tickers.json", server.url()))
            .await?;

        assert_eq!(result["AAPL"]["ticker"].as_str(), Some("AAPL"));
        Ok(())
    }

    #[tokio::test]
    async fn test_fetch_json_with_retry_failure() -> Result<(), Box<dyn Error>> {
        let mut server = Server::new_async().await;

        let _mock = server
            .mock("GET", "/files/company_tickers.json")
            .with_status(500)
            .expect(3)
            .create_async()
            .await;

        let client = SecClient::new("test@example.com", 1, 500, Some(2));

        let result = client
            .fetch_json(&format!("{}/files/company_tickers.json", server.url()))
            .await;

        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_fetch_json_with_retry_backoff() -> Result<(), Box<dyn Error>> {
        let mut server = Server::new_async().await;

        let _mock_fail = server
            .mock("GET", "/files/company_tickers.json")
            .with_status(500)
            .expect(1)
            .create_async()
            .await;

        let _mock_success = server
            .mock("GET", "/files/company_tickers.json")
            .with_status(200)
            .with_header("Content-Type", "application/json")
            .with_body(r#"{"AAPL": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc"}}"#)
            .expect(1)
            .create_async()
            .await;

        let client = SecClient::new("test@example.com", 1, 500, Some(2));

        let result = client
            .fetch_json(&format!("{}/files/company_tickers.json", server.url()))
            .await?;

        assert_eq!(result["AAPL"]["ticker"].as_str(), Some("AAPL"));
        Ok(())
    }
}
