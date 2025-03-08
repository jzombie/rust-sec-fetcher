use std::fs;
use std::io::Write;
use tempfile::tempdir;
use std::path::PathBuf;
use sec_fetcher::config::ConfigManager;
use sec_fetcher::utils::set_interactive_mode_override;

/// Helper function to create a temporary config file
fn create_temp_config(contents: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempdir().expect("Failed to create temp dir"); // TempDir is now returned
    let path = dir.path().join("config.toml");

    let mut file = fs::File::create(&path).expect("Failed to create config file");
    writeln!(file, "{}", contents).expect("Failed to write to config file");

    (dir, path) // Return both the directory and path
}


#[test]
fn test_fails_if_no_email_available() {
    set_interactive_mode_override(Some(false));

    let result = ConfigManager::load(); // Expect this to fail

    assert!(result.is_err()); // Ensure it fails
    assert_eq!(
        result.unwrap_err().to_string(),
        "Could not obtain email credential"
    ); // Ensure correct error

    set_interactive_mode_override(None);
}


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
