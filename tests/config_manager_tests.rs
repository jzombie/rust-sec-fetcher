use sec_fetcher::config::{
    ConfigManager, APP_NAME_ENV_VAR, APP_VERSION_ENV_VAR, DEFAULT_APP_NAME, DEFAULT_APP_VERSION,
    EMAIL_ENV_VAR,
};
use sec_fetcher::network::SecClient;
use sec_fetcher::utils::set_interactive_mode_override;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use tempfile::tempdir;

// Tests that mutate process-level env vars or the interactive-mode override
// must hold this lock for their entire duration, otherwise concurrent tests
// will race on shared global state.
static ENV_MUTEX: Mutex<()> = Mutex::new(());

/// Helper function to create a temporary config file
fn create_temp_config(contents: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempdir().expect("Failed to create temp dir"); // TempDir is now returned
    let path = dir.path().join("config.toml");

    let mut file = fs::File::create(&path).expect("Failed to create config file");
    writeln!(file, "{}", contents).expect("Failed to write to config file");

    (dir, path) // Return both the directory and path
}

// TODO: Fix (this fails if there is a configured .toml)
// #[test]
// fn test_fails_if_no_email_available() {
//     set_interactive_mode_override(Some(false));

//     let result = ConfigManager::load(); // Expect this to fail

//     assert!(result.is_err()); // Ensure it fails
//     assert_eq!(
//         result.unwrap_err().to_string(),
//         "Could not obtain email credential"
//     ); // Ensure correct error

//     set_interactive_mode_override(None);
// }

#[test]
fn test_load_custom_config() {
    let config_contents = r#"
        email = "test@example.com"
        max_concurrent = 10
        min_delay_ms = 500
        max_retries = 3
    "#;

    let (temp_dir, config_path) = create_temp_config(config_contents); // Store TempDir

    let config_manager = ConfigManager::from_config(Some(config_path.clone()))
        .expect("Failed to load custom config");

    let config = config_manager.get_config();

    assert_eq!(config.email, Some("test@example.com".to_string()));
    assert_eq!(config.max_concurrent, Some(10));
    assert_eq!(config.min_delay_ms, Some(500));
    assert_eq!(config.max_retries, Some(3));

    drop(temp_dir); // Explicitly drop temp_dir (not necessary but ensures cleanup after test)
}

#[test]
fn test_load_non_existent_config() {
    let config_path = PathBuf::from("non_existent_config.toml");
    let result = ConfigManager::from_config(Some(config_path));

    assert!(result.is_err());
}

#[test]
fn test_fails_if_no_email_available_non_interactive() {
    let _lock = ENV_MUTEX.lock().unwrap();
    // Force non-interactive mode and provide an existing but empty config file
    set_interactive_mode_override(Some(false));
    std::env::remove_var(EMAIL_ENV_VAR); // ensure env var is not set

    let (_temp_dir, config_path) = create_temp_config("");

    let result = ConfigManager::from_config(Some(config_path));

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    // Error message should mention both the config file and the env var
    assert!(
        err.contains(EMAIL_ENV_VAR),
        "Error message should mention the env var: {}",
        err
    );

    set_interactive_mode_override(None);
}

#[test]
fn test_email_from_env_var_non_interactive() {
    let _lock = ENV_MUTEX.lock().unwrap();
    set_interactive_mode_override(Some(false));
    std::env::set_var(EMAIL_ENV_VAR, "env@example.com");

    let (_temp_dir, config_path) = create_temp_config(""); // no email in file

    let result = ConfigManager::from_config(Some(config_path));
    assert!(
        result.is_ok(),
        "Expected Ok when email is set via env var: {:?}",
        result.err()
    );
    assert_eq!(
        result.unwrap().get_config().email,
        Some("env@example.com".to_string())
    );

    std::env::remove_var(EMAIL_ENV_VAR);
    set_interactive_mode_override(None);
}

#[test]
fn test_config_file_email_takes_precedence_over_env_var() {
    let _lock = ENV_MUTEX.lock().unwrap();
    set_interactive_mode_override(Some(false));
    std::env::set_var(EMAIL_ENV_VAR, "env@example.com");

    let (_temp_dir, config_path) = create_temp_config(r#"email = "file@example.com""#);

    let result = ConfigManager::from_config(Some(config_path));
    assert!(result.is_ok());
    // Config file email should win
    assert_eq!(
        result.unwrap().get_config().email,
        Some("file@example.com".to_string())
    );

    std::env::remove_var(EMAIL_ENV_VAR);
    set_interactive_mode_override(None);
}

#[test]
fn test_fails_on_invalid_key() {
    let config_contents = r#"
        email = "test@example.com"
        max_concurrent = 10
        min_delay_ms = 500
        max_retries = 3
        invalid_key = "this_should_fail"
    "#;

    let (_temp_dir, config_path) = create_temp_config(config_contents);

    let result = ConfigManager::from_config(Some(config_path));

    assert!(
        result.is_err(),
        "Expected an error due to an invalid key, but got Ok()"
    );

    let error_message = result.unwrap_err().to_string();

    assert!(error_message.contains("Valid configuration keys are:"));
    assert!(error_message.contains("email (String | Null)"));
    assert!(error_message.contains("app_name (String | Null)"));
    // assert!(error_message.contains("http_cache_storage_bin (String | Null)")); // TODO: Change?
    assert!(error_message.contains("max_concurrent (Integer | Null)"));
    assert!(error_message.contains("max_retries (Integer | Null)"));
    assert!(error_message.contains("min_delay_ms (Integer | Null)"));
}

#[test]
fn test_app_name_from_config_file() {
    let _lock = ENV_MUTEX.lock().unwrap();
    std::env::remove_var(APP_NAME_ENV_VAR);
    std::env::remove_var(APP_VERSION_ENV_VAR);

    let config_contents = r#"
        email = "test@example.com"
        app_name = "my-custom-app"
    "#;

    let (_temp_dir, config_path) = create_temp_config(config_contents);

    let config_manager =
        ConfigManager::from_config(Some(config_path)).expect("Failed to load config with app_name");

    assert_eq!(
        config_manager.get_config().app_name,
        Some("my-custom-app".to_string())
    );
}

#[test]
fn test_app_name_absent_is_none() {
    let _lock = ENV_MUTEX.lock().unwrap();
    std::env::remove_var(APP_NAME_ENV_VAR); // ensure env var doesn't interfere

    let config_contents = r#"
        email = "test@example.com"
    "#;

    let (_temp_dir, config_path) = create_temp_config(config_contents);

    let config_manager = ConfigManager::from_config(Some(config_path))
        .expect("Failed to load config without app_name");

    assert_eq!(config_manager.get_config().app_name, None);

    // Verify the fallback: the exact User-Agent string sent to the SEC uses the crate name
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
fn test_app_name_from_env_var() {
    let _lock = ENV_MUTEX.lock().unwrap();
    std::env::set_var(APP_NAME_ENV_VAR, "my-env-app");

    let (_temp_dir, config_path) = create_temp_config(r#"email = "test@example.com""#);

    let config_manager = ConfigManager::from_config(Some(config_path))
        .expect("Failed to load config with app_name from env var");

    assert_eq!(
        config_manager.get_config().app_name,
        Some("my-env-app".to_string())
    );

    let client = SecClient::from_config_manager(&config_manager).unwrap();
    assert_eq!(
        client.get_user_agent(),
        format!(
            "my-env-app/{} (+test@example.com)",
            env!("CARGO_PKG_VERSION")
        )
    );

    std::env::remove_var(APP_NAME_ENV_VAR);
}

#[test]
fn test_app_name_config_file_takes_precedence_over_env_var() {
    let _lock = ENV_MUTEX.lock().unwrap();
    std::env::set_var(APP_NAME_ENV_VAR, "env-app");

    let (_temp_dir, config_path) = create_temp_config(
        r#"email = "test@example.com"
app_name = "file-app""#,
    );

    let config_manager =
        ConfigManager::from_config(Some(config_path)).expect("Failed to load config");

    assert_eq!(
        config_manager.get_config().app_name,
        Some("file-app".to_string()),
        "Config file app_name should take precedence over env var"
    );

    std::env::remove_var(APP_NAME_ENV_VAR);
}

#[test]
fn test_app_name_default_constant() {
    // Sanity-check that the exported DEFAULT_APP_NAME is the expected value
    assert_eq!(DEFAULT_APP_NAME, "sec-fetcher");
}

#[test]
fn test_app_version_from_config_file() {
    let _lock = ENV_MUTEX.lock().unwrap();
    std::env::remove_var(APP_NAME_ENV_VAR);
    std::env::remove_var(APP_VERSION_ENV_VAR);

    let config_contents = r#"
        email = "test@example.com"
        app_version = "3.1.4"
    "#;

    let (_temp_dir, config_path) = create_temp_config(config_contents);
    let config_manager = ConfigManager::from_config(Some(config_path))
        .expect("Failed to load config with app_version");

    assert_eq!(
        config_manager.get_config().app_version,
        Some("3.1.4".to_string())
    );

    let client = SecClient::from_config_manager(&config_manager).unwrap();
    assert_eq!(
        client.get_user_agent(),
        "sec-fetcher/3.1.4 (+test@example.com)"
    );
}

#[test]
fn test_app_version_from_env_var() {
    let _lock = ENV_MUTEX.lock().unwrap();
    std::env::set_var(APP_VERSION_ENV_VAR, "9.9.9");

    let (_temp_dir, config_path) = create_temp_config(r#"email = "test@example.com""#);
    let config_manager = ConfigManager::from_config(Some(config_path))
        .expect("Failed to load config with app_version from env var");

    assert_eq!(
        config_manager.get_config().app_version,
        Some("9.9.9".to_string())
    );

    let client = SecClient::from_config_manager(&config_manager).unwrap();
    assert_eq!(
        client.get_user_agent(),
        format!("sec-fetcher/9.9.9 (+test@example.com)")
    );

    std::env::remove_var(APP_VERSION_ENV_VAR);
}

#[test]
fn test_app_version_config_file_takes_precedence_over_env_var() {
    let _lock = ENV_MUTEX.lock().unwrap();
    std::env::set_var(APP_VERSION_ENV_VAR, "1.0.0-env");

    let (_temp_dir, config_path) = create_temp_config(
        r#"email = "test@example.com"
app_version = "2.0.0-file""#,
    );
    let config_manager =
        ConfigManager::from_config(Some(config_path)).expect("Failed to load config");

    assert_eq!(
        config_manager.get_config().app_version,
        Some("2.0.0-file".to_string()),
        "Config file app_version should take precedence over env var"
    );

    std::env::remove_var(APP_VERSION_ENV_VAR);
}

#[test]
fn test_app_version_default_constant() {
    // DEFAULT_APP_VERSION must equal the crate version baked in at compile time
    assert_eq!(DEFAULT_APP_VERSION, env!("CARGO_PKG_VERSION"));
}
