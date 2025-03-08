use config::{Config, File};
use dirs::config_dir;
use serde::Deserialize;
use std::error::Error;
use std::path::{PathBuf};
use http_cache_reqwest::{Cache, CacheMode, CACacheManager, HttpCache, HttpCacheOptions};

#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    pub email: Option<String>,
    pub max_concurrent: Option<usize>,
    pub min_delay_ms: Option<u64>,
    pub max_retries: Option<usize>,
    pub cache_dir: Option<String>,
    pub cache_mode: Option<String>,
}

impl AppConfig {
    pub fn get_cache_dir(&self) -> Option<PathBuf> {
        self.cache_dir.clone().map(|dir_string| PathBuf::from(dir_string))
    }

    pub fn get_cache_mode(&self) -> CacheMode {
        match &self.cache_mode {
            Some(cache_mode) => {
                match cache_mode.as_str() {
                    "Default" => CacheMode::Default,
                    "ForceCache" => CacheMode::ForceCache,
                    "NoCache" => CacheMode::NoCache,
                    "IgnoreRules" => CacheMode::IgnoreRules,
                    _ => CacheMode::Default, // Fallback
                }
            }
            _ => CacheMode::Default
        }
    }
}

pub struct ConfigManager {
    config: AppConfig,
}

impl ConfigManager {
    pub fn get_suggested_system_path() -> Option<PathBuf> {
        config_dir().map(|dir| dir.join(env!("CARGO_PKG_NAME")).join("config.toml"))
    }
    
    /// Determines the standard config file location.
    pub fn get_config_path() -> PathBuf {
        if let Some(path) = Self::get_suggested_system_path() {
            if path.exists() {
                return path;
            }
        }
        PathBuf::from("config.toml") // Fallback if no global config
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
