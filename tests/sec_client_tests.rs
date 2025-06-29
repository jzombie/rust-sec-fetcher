use mockito::Server;
use sec_fetcher::config::{AppConfig, ConfigManager};
use sec_fetcher::network::SecClient;
use std::error::Error;

#[test]
fn test_user_agent() {
    let mut app_config = AppConfig::default();
    app_config.email = Some("test@example.com".into());
    let config_manager = ConfigManager::from_app_config(&app_config);

    let client = SecClient::from_config_manager(&config_manager).unwrap();

    assert_eq!(
        client.get_user_agent(),
        format!(
            "sec-fetcher/{} (+test@example.com)",
            env!("CARGO_PKG_VERSION")
        )
    );
}

#[test]
#[should_panic(expected = "Invalid email format")]
fn test_invalid_email_panic() {
    let mut app_config = AppConfig::default();
    app_config.email = Some("invalid-email".into());
    let config_manager = ConfigManager::from_app_config(&app_config);

    let client = SecClient::from_config_manager(&config_manager).unwrap();

    client.get_user_agent();
}

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

    let mut app_config = AppConfig::default();
    app_config.email = Some("test@example.com".into());
    let config_manager = ConfigManager::from_app_config(&app_config);

    let client = SecClient::from_config_manager(&config_manager).unwrap();

    let result = client
        .fetch_json(
            &format!("{}/files/company_tickers.json", server.url()),
            None,
        )
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

    let mut app_config = AppConfig::default();
    app_config.email = Some("test@example.com".into());
    let config_manager = ConfigManager::from_app_config(&app_config);

    let client = SecClient::from_config_manager(&config_manager).unwrap();

    let result = client
        .fetch_json(
            &format!("{}/files/company_tickers.json", server.url()),
            None,
        )
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

    let mut app_config = AppConfig::default();
    app_config.email = Some("test@example.com".into());
    app_config.max_retries = Some(2);
    let config_manager = ConfigManager::from_app_config(&app_config);

    let client = SecClient::from_config_manager(&config_manager).unwrap();

    let result = client
        .fetch_json(
            &format!("{}/files/company_tickers.json", server.url()),
            None,
        )
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

    let mut app_config = AppConfig::default();
    app_config.email = Some("test@example.com".into());
    app_config.max_retries = Some(2);
    let config_manager = ConfigManager::from_app_config(&app_config);

    let client = SecClient::from_config_manager(&config_manager).unwrap();

    let result = client
        .fetch_json(
            &format!("{}/files/company_tickers.json", server.url()),
            None,
        )
        .await?;

    assert_eq!(result["AAPL"]["ticker"].as_str(), Some("AAPL"));
    Ok(())
}
