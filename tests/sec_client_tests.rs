use mockito::Server;
use sec_fetcher::config::{AppConfig, ConfigManager};
use sec_fetcher::network::SecClient;
use std::error::Error;

#[test]
fn test_user_agent() {
    let app_config = AppConfig {
        email: Some("test@example.com".into()),
        ..Default::default()
    };
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
fn test_user_agent_with_custom_app_name() {
    // No app_version set — should still use sec-fetcher's version
    let app_config = AppConfig {
        email: Some("test@example.com".into()),
        app_name: Some("my-custom-app".into()),
        ..Default::default()
    };
    let config_manager = ConfigManager::from_app_config(&app_config);

    let client = SecClient::from_config_manager(&config_manager).unwrap();

    assert_eq!(
        client.get_user_agent(),
        format!(
            "my-custom-app/{} (+test@example.com)",
            env!("CARGO_PKG_VERSION")
        )
    );
}

#[test]
fn test_user_agent_with_custom_app_name_and_version() {
    let app_config = AppConfig {
        email: Some("test@example.com".into()),
        app_name: Some("my-custom-app".into()),
        app_version: Some("2.0.0".into()),
        ..Default::default()
    };
    let config_manager = ConfigManager::from_app_config(&app_config);

    let client = SecClient::from_config_manager(&config_manager).unwrap();

    assert_eq!(
        client.get_user_agent(),
        "my-custom-app/2.0.0 (+test@example.com)"
    );
}

#[test]
fn test_user_agent_default_app_version_when_none() {
    // app_version not set — falls back to sec-fetcher's crate version
    let app_config = AppConfig {
        email: Some("test@example.com".into()),
        ..Default::default()
    };
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
fn test_user_agent_default_app_name_when_none() {
    // When app_name is None, the crate name is used — verify the exact string sent to the SEC
    // app_name not set — stays None
    let app_config = AppConfig {
        email: Some("test@example.com".into()),
        ..Default::default()
    };
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
    let app_config = AppConfig {
        email: Some("invalid-email".into()),
        ..Default::default()
    };
    let config_manager = ConfigManager::from_app_config(&app_config);

    let client = SecClient::from_config_manager(&config_manager).unwrap();

    client.get_user_agent();
}

#[test]
fn test_missing_email_returns_error() {
    let app_config = AppConfig::default(); // email is None by default
    let config_manager = ConfigManager::from_app_config(&app_config);

    let result = SecClient::from_config_manager(&config_manager);

    assert!(result.is_err());
    let err = result.err().expect("Expected error when email missing");
    assert_eq!(err.to_string(), "Missing required field: email");
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

    let app_config = AppConfig {
        email: Some("test@example.com".into()),
        ..Default::default()
    };
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

    let app_config = AppConfig {
        email: Some("test@example.com".into()),
        ..Default::default()
    };
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

    let app_config = AppConfig {
        email: Some("test@example.com".into()),
        max_retries: Some(2),
        ..Default::default()
    };
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

    let app_config = AppConfig {
        email: Some("test@example.com".into()),
        max_retries: Some(2),
        ..Default::default()
    };
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
