use config::{Config, File};
use dirs::config_dir;
use serde::Deserialize;
use std::error::Error;
use std::path::{PathBuf};

#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    pub email: String,
    pub max_concurrent: usize,
    pub min_delay_ms: u64,
    pub max_retries: Option<usize>,
    pub cache_dir: String,
    pub cache_mode: String, // Will be converted later
}

pub struct ConfigManager {
    config: AppConfig,
}

impl ConfigManager {
    /// Determines the standard config file location.
    fn get_config_path() -> PathBuf {
        let default_path = PathBuf::from("config.toml"); // Fallback if no global config

        if let Some(config_dir) = config_dir() {
            let path = config_dir.join(env!("CARGO_PKG_NAME")).join("config.toml");
            if path.exists() {
                return path;
            }
        }
        default_path
    }

    /// Loads configuration from file and environment variables.
    pub fn load() -> Result<Self, Box<dyn Error>> {
        let config_path = Self::get_config_path();

        let config = Config::builder()
            .add_source(File::with_name(config_path.to_str().unwrap()).required(false)) // Load if exists
            // TODO: Handle or remove
            // .add_source(config::Environment::with_prefix("SEC")) // Environment variable overrides (e.g., `SEC_EMAIL`)
            .build()?;

        let settings: AppConfig = config.try_deserialize()?;
        Ok(Self { config: settings })
    }

    /// Retrieves a reference to the configuration.
    pub fn get_config(&self) -> &AppConfig {
        &self.config
    }
}
