/// Network integration tests for [`sec_fetcher::network::SecClient`].
///
/// These tests use [`mockito`] to simulate EDGAR HTTP responses and verify
/// that the client correctly handles various response conditions.
use mockito::Server;
use sec_fetcher::config::{AppConfig, ConfigManager};
use sec_fetcher::network::SecClient;
use std::error::Error;

fn make_client() -> SecClient {
    let app_config = AppConfig {
        email: Some("test@example.com".into()),
        max_retries: Some(0),
        ..Default::default()
    };
    let config_manager = ConfigManager::from_app_config(&app_config);
    SecClient::from_config_manager(&config_manager).unwrap()
}

#[tokio::test]
async fn test_raw_request_with_custom_throttle_policy() -> Result<(), Box<dyn Error>> {
    let mut server = Server::new_async().await;

    let _mock = server
        .mock("GET", "/test-endpoint")
        .with_status(200)
        .with_body("response body")
        .create_async()
        .await;

    let client = make_client();
    let throttle_policy = client.get_throttle_policy();

    let response = client
        .raw_request(
            reqwest::Method::GET,
            &format!("{}/test-endpoint", server.url()),
            None,
            Some(throttle_policy),
        )
        .await?;

    assert!(response.status().is_success());
    let body = response.text().await?;
    assert_eq!(body, "response body");
    Ok(())
}

#[tokio::test]
async fn test_raw_request_with_custom_headers() -> Result<(), Box<dyn Error>> {
    let mut server = Server::new_async().await;

    let _mock = server
        .mock("GET", "/with-headers")
        .match_header("X-Custom", "test-value")
        .with_status(200)
        .with_body("custom header ok")
        .create_async()
        .await;

    let client = make_client();
    let response = client
        .raw_request(
            reqwest::Method::GET,
            &format!("{}/with-headers", server.url()),
            Some(vec![("X-Custom", "test-value")]),
            None,
        )
        .await?;

    assert!(response.status().is_success());
    let body = response.text().await?;
    assert_eq!(body, "custom header ok");
    Ok(())
}

#[tokio::test]
async fn test_raw_request_with_empty_headers() -> Result<(), Box<dyn Error>> {
    let mut server = Server::new_async().await;

    let _mock = server
        .mock("GET", "/no-headers")
        .with_status(200)
        .with_body("no headers")
        .create_async()
        .await;

    let client = make_client();
    let response = client
        .raw_request(
            reqwest::Method::GET,
            &format!("{}/no-headers", server.url()),
            None, // No custom headers
            None,
        )
        .await?;

    assert!(response.status().is_success());
    let body = response.text().await?;
    assert_eq!(body, "no headers");
    Ok(())
}

#[tokio::test]
async fn test_raw_request_nocache() -> Result<(), Box<dyn Error>> {
    let mut server = Server::new_async().await;

    let _mock = server
        .mock("GET", "/nocache-test")
        .with_status(200)
        .with_body("nocache response")
        .create_async()
        .await;

    let client = make_client();
    let response = client
        .raw_request_nocache(
            reqwest::Method::GET,
            &format!("{}/nocache-test", server.url()),
            None,
        )
        .await?;

    assert!(response.status().is_success());
    let body = response.text().await?;
    assert_eq!(body, "nocache response");
    Ok(())
}

#[tokio::test]
async fn test_raw_request_nocache_with_headers() -> Result<(), Box<dyn Error>> {
    let mut server = Server::new_async().await;

    let _mock = server
        .mock("GET", "/nocache-headers")
        .match_header("Accept", "application/json")
        .with_status(200)
        .with_body("json data")
        .create_async()
        .await;

    let client = make_client();
    let response = client
        .raw_request_nocache(
            reqwest::Method::GET,
            &format!("{}/nocache-headers", server.url()),
            Some(vec![("Accept", "application/json")]),
        )
        .await?;

    assert!(response.status().is_success());
    let body = response.text().await?;
    assert_eq!(body, "json data");
    Ok(())
}

#[tokio::test]
async fn test_fetch_json_with_404_response() -> Result<(), Box<dyn Error>> {
    let mut server = Server::new_async().await;

    let _mock = server
        .mock("GET", "/not-found")
        .with_status(404)
        .with_body("Not Found")
        .create_async()
        .await;

    let client = make_client();
    let result = client
        .fetch_json(&format!("{}/not-found", server.url()), None)
        .await;

    assert!(result.is_err(), "Expected error for 404 response");
    Ok(())
}

#[tokio::test]
async fn test_fetch_json_with_non_json_response() -> Result<(), Box<dyn Error>> {
    let mut server = Server::new_async().await;

    let _mock = server
        .mock("GET", "/plain-text")
        .with_status(200)
        .with_header("Content-Type", "text/plain")
        .with_body("not json at all")
        .create_async()
        .await;

    let client = make_client();
    let result = client
        .fetch_json(&format!("{}/plain-text", server.url()), None)
        .await;

    // Should return an error since the body is not valid JSON
    assert!(result.is_err(), "Expected error for non-JSON response");
    Ok(())
}

#[tokio::test]
async fn test_raw_request_server_error() -> Result<(), Box<dyn Error>> {
    let mut server = Server::new_async().await;

    let _mock = server
        .mock("GET", "/server-error")
        .with_status(500)
        .with_body("Internal Server Error")
        .create_async()
        .await;

    let client = make_client();
    let response = client
        .raw_request(
            reqwest::Method::GET,
            &format!("{}/server-error", server.url()),
            None,
            None,
        )
        .await?;

    assert!(response.status().is_server_error());
    Ok(())
}

#[tokio::test]
async fn test_get_cache_and_throttle_policy() {
    let client = make_client();
    let cache_policy = client.get_cache_policy();
    let throttle_policy = client.get_throttle_policy();

    // Sanity-check default values
    assert_eq!(cache_policy.default_ttl.as_secs(), 60 * 60 * 24 * 7); // 1 week
    assert_eq!(throttle_policy.max_concurrent, 1);
    assert_eq!(throttle_policy.max_retries, 0); // We set this to 0 in make_client
}

#[tokio::test]
async fn test_get_preprocessor_cache() {
    let client = make_client();
    let cache = client.get_preprocessor_cache();
    // The cache should be a valid DataStore instance (non-null Arc)
    assert!(std::sync::Arc::strong_count(&cache) >= 1);
}
